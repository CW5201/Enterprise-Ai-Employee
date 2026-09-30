"""Unified LLM client — the ONLY module allowed to call a model endpoint.

Phase 1 (Qwen, single primary model family):
- ``generate()``         -> plain text
- ``generate_structured()`` -> Pydantic-validated structured output
- API key comes from ``.env`` (``LLM_API_KEY``), never from code.
- On failure the client raises a unified :class:`LLMError` — business code
  must never see provider SDK exceptions.

Provider transport: OpenAI-compatible HTTP chat-completions API via ``httpx``
(Qwen is served through an OpenAI-compatible endpoint; no vendor SDK needed
and provider switching stays a config change).

When no ``LLM_API_KEY`` / endpoint is configured the client operates in
**offline deterministic mode** (see :class:`_OfflineBackend`), which makes
the whole pipeline testable without a live model. Offline mode is clearly
labelled in its outputs and is NEVER presented as a real model answer.
Multi-model routing is deliberately NOT implemented in Phase 1.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from src.core.exceptions import LLMError

logger = logging.getLogger("eae.llm")

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# .env loading (minimal KEY=VALUE parser, no external dependency)
# ---------------------------------------------------------------------------

_ENV_SECRETS = {"LLM_API_KEY", "MILVUS_PASSWORD", "NEO4J_PASSWORD"}
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
_ENV_LOADED = False


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from .env; real env vars always win."""
    global _ENV_LOADED
    if _ENV_LOADED or not _ENV_FILE.exists():
        _ENV_LOADED = True
        return
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key in _ENV_SECRETS or key not in os.environ:
            os.environ[key] = value
    _ENV_LOADED = True


# ---------------------------------------------------------------------------
# LLM settings (from config/settings.yaml; secrets resolve from .env only)
# ---------------------------------------------------------------------------

_LLM_CONFIG_CACHE: dict[str, Any] | None = None


def _llm_config() -> dict[str, Any]:
    """Read the llm/embedding/milvus/rag sections once (no config_loader import cycle)."""
    global _LLM_CONFIG_CACHE
    if _LLM_CONFIG_CACHE is None:
        import yaml

        path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
        with path.open(encoding="utf-8") as fh:
            _LLM_CONFIG_CACHE = yaml.safe_load(fh) or {}
    return _LLM_CONFIG_CACHE


def _env_var(value: Any, fallback: str = "") -> str:
    """Resolve ``${VAR:-default}`` placeholders (config values may be strings)."""
    if not isinstance(value, str):
        return str(value)
    import re

    m = re.match(r"^\$\{([A-Za-z0-9_]+)(?::-(.*?))?\}$", value)
    if m:
        return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else fallback)
    return value


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _BaseBackend:
    """Backend protocol: raw completion + structured (Pydantic-validated)."""

    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def complete_structured(self, system: str, user: str, model: type[T]) -> T:
        raise NotImplementedError


class _OfflineBackend(_BaseBackend):
    """Deterministic offline backend used when no endpoint/key is configured.

    It returns structurally-correct payloads derived from the prompt's
    declared schema so the pipeline (and its tests) run end-to-end. Every
    output carries ``"offline": true`` in its reason field so downstream
    stages and reports can tell it apart from a real model call.

    This is a test/offline facility, NOT a substitute for the model:
    SQL it produces is only as good as its keyword heuristics and is always
    executed through the read-only guard before reaching the answer node.
    """

    def complete(self, system: str, user: str) -> str:
        return json.dumps(
            {"offline": True, "text": user, "reason": "offline backend: no LLM endpoint configured"},
            ensure_ascii=False,
        )

    def complete_structured(self, system: str, user: str, model: type[T]) -> T:
        lowered = user.lower()
        payload: dict[str, Any]
        if "intent" in system.lower() or "intentresult" in system.lower():
            if ("结合" in user and ("数据" in user or "查询" in user)) or "以及" in user:
                intent = "complex_analysis"
            elif any(k in lowered for k in ("查询", "统计", "多少", "金额", "订单", "客户", "销售额", "营收")):
                intent = "data_query"
            elif any(k in lowered for k in ("政策", "制度", "规定", "标准", "报销", "流程", "手册", "规范")):
                intent = "knowledge_query"
            else:
                intent = "clarification"
            payload = {
                "intent": intent,
                "confidence": 0.9,
                "entities": [],
                "reason": "offline backend: keyword heuristic",
            }
        elif "text-to-sql" in system.lower() or "sqlresult" in system.lower() or "生成 sql" in system.lower():
            # Heuristic, clearly labelled — real Text-to-SQL needs a live LLM.
            if "客户" in user and ("最多" in user or "top" in lowered or "前" in user):
                payload = {
                    "sql": (
                        "SELECT c.CustomerName, COUNT(o.OrderID) AS order_count "
                        "FROM Sales_Orders o "
                        "JOIN Sales_Customers c ON c.CustomerID = o.CustomerID "
                        "GROUP BY c.CustomerName "
                        "ORDER BY order_count DESC "
                        "LIMIT 10"
                    ),
                    "rationale": "offline heuristic: 订单数最多的客户 = 按 CustomerName 聚合订单数排序",
                }
            else:
                payload = {
                    "sql": "SELECT 1 AS placeholder",
                    "rationale": (
                        "offline backend: cannot generate real SQL without a configured model. "
                        "Set LLM_API_KEY / LLM_BASE_URL in .env for real Text-to-SQL."
                    ),
                }
        else:
            payload = {"offline": True, "result": user}
        return model.model_validate(payload)


def _parse_json_payload(text: Any) -> Any:
    """Parse an LLM completion into a JSON value.

    Accepts a pre-parsed dict, a plain JSON string, or a JSON string wrapped
    in markdown `` ```json ... ``` `` fences (common provider behaviour).
    Fences are stripped *before* parsing so that fenced-but-valid output is
    not misreported as a parse failure.  Raises
    ``json.JSONDecodeError`` for genuinely unparsable content.
    """
    if isinstance(text, dict):
        return text
    raw = str(text).strip()
    if raw.startswith("```"):
        # opening fence, optional language tag
        first_nl = raw.find("\n")
        body = raw[first_nl + 1:] if first_nl != -1 else raw[3:]
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        raw = body.strip()
    return json.loads(raw)


class _QwenOpenAIBackend(_BaseBackend):
    """OpenAI-compatible chat-completions endpoint (Qwen)."""

    def __init__(self) -> None:
        cfg = _llm_config()
        llm = cfg.get("llm", {})
        self.base_url = _env_var(llm.get("base_url", ""), os.environ.get("LLM_BASE_URL", ""))
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.model = _env_var(llm.get("model", ""), os.environ.get("LLM_MODEL", "qwen-max"))
        self.temperature = float(_env_var(llm.get("temperature", 0.2), "0.2"))
        self.max_tokens = int(_env_var(llm.get("max_tokens", 2048), "2048"))
        self.timeout = float(_env_var(llm.get("timeout_seconds", 60), "60"))
        self.max_retries = int(_env_var(llm.get("max_retries", 2), "2"))
        if not self.base_url:
            raise LLMError("LLM_BASE_URL is not set; cannot use the Qwen backend", retryable=False)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions", headers=headers, json=body
                    )
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise LLMError(
            f"Qwen request failed after {self.max_retries + 1} attempts: {last_error}",
            retryable=True,
        ) from last_error

    def _messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def complete(self, system: str, user: str) -> str:
        data = self._post(
            {
                "model": self.model,
                "messages": self._messages(system, user),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}", retryable=True) from exc

    def complete_structured(self, system: str, user: str, model: type[T]) -> T:
        schema = model.model_json_schema()
        if "title" in schema:
            del schema["title"]
        schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
        body = {
            "model": self.model,
            "messages": self._messages(
                system + "\n\nRespond with a JSON object matching EXACTLY this JSON schema:\n"
                + schema_json,
                user,
            ),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        text = self._post(body).get("choices", [{}])[0].get("message", {}).get("content", "")
        raw = _parse_json_payload(text)
        if raw is None:
            raise LLMError("Model returned empty JSON body", retryable=True)
        try:
            return model.model_validate(raw)
        except ValidationError as exc:
            raise LLMError(
                f"Structured output failed Pydantic validation: {exc.errors()[:3]}",
                retryable=True,
            ) from exc


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------


class LLMClient:
    """Single entry point for all model access (Qwen, Phase 1).

    Backend selection is explicit and environment-driven:
    - ``LLM_BASE_URL`` set          -> live Qwen OpenAI-compatible endpoint
    - ``LLM_FORCE_OFFLINE=1``      -> deterministic offline backend (tests)
    - otherwise                     -> offline backend (no endpoint configured)
    """

    def __init__(self, backend: _BaseBackend | None = None) -> None:
        _load_dotenv()
        if backend is not None:
            self._backend = backend
        elif os.environ.get("LLM_FORCE_OFFLINE") == "1" or not os.environ.get("LLM_BASE_URL"):
            self._backend = _OfflineBackend()
        else:
            try:
                self._backend = _QwenOpenAIBackend()
            except LLMError:
                logger.warning("LLM endpoint misconfigured; falling back to offline backend")
                self._backend = _OfflineBackend()

    @property
    def offline(self) -> bool:
        return isinstance(self._backend, _OfflineBackend)

    def generate(self, system: str, user: str) -> str:
        """One free-text completion. Raises LLMError on failure."""
        try:
            return self._backend.complete(system, user)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - unify ALL provider errors
            raise LLMError(f"LLM call failed: {exc}", retryable=False) from exc

    def generate_structured(self, system: str, user: str, model: type[T]) -> T:
        """One structured completion, always Pydantic-validated."""
        try:
            return self._backend.complete_structured(system, user, model)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Structured LLM call failed: {exc}", retryable=False) from exc


__all__ = ["LLMClient"]
