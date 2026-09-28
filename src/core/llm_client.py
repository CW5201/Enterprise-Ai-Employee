"""Unified LLM client - the only entry point for calling the LLM.

Business code must NEVER import a model SDK directly; it goes through
:func:`chat_json` (structured output) or :func:`chat_text` (plain text).

Implementation notes (Phase 1):
- Qwen is accessed via an **OpenAI-compatible chat API** using ``httpx``
  (no vendor SDK in the dependency list - keeps the stack minimal and
  makes provider switching a config change).
- A **deterministic mock backend** is available for offline testing
  (``LLM_BACKEND=mock``), so the pipeline and tests never depend on a
  live model endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import httpx

from src.core.config_loader import LLMSettings, get_settings
from src.core.exceptions import LLMError
from src.core.observability import observe

logger = logging.getLogger("eae.llm")

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON object from an LLM response."""
    text = text.strip()
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fall back: first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(
                "LLM response did not contain JSON",
                code="llm_bad_format",
                details={"raw": text[:500]},
            ) from None
        return json.loads(text[start : end + 1])


class LLMClient:
    """Thin wrapper around the configured Qwen endpoint (or a mock)."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or get_settings().llm
        self._backend: str = os.environ.get("LLM_BACKEND", "auto")

    # -- public API --------------------------------------------------------

    @observe("llm_chat")
    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        """Run one chat turn and parse the reply as a JSON object."""
        text = self._complete(system, user)
        result = _extract_json(text)
        if not isinstance(result, dict):
            raise LLMError(
                "LLM JSON response is not an object",
                code="llm_bad_format",
                details={"value": result},
            )
        return result

    @observe("llm_chat")
    def chat_text(self, system: str, user: str) -> str:
        return self._complete(system, user).strip()

    # -- backends ----------------------------------------------------------

    def _complete(self, system: str, user: str) -> str:
        if self._backend in ("mock", "auto") and not self.settings.model:
            return _mock_complete(system, user, self.settings.provider)
        if not self.settings.base_url:
            raise LLMError(
                "LLM_BASE_URL is not configured; set LLM_BACKEND=mock for offline tests",
                retryable=False,
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self._call_endpoint(messages)
                if response:
                    return response
                last_error = LLMError("Empty LLM response")
            except LLMError as exc:
                last_error = exc
                if not exc.retryable:
                    raise
            time.sleep(0.5 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _call_endpoint(self, messages: list[dict[str, str]]) -> str:
        """POST to the OpenAI-compatible /chat/completions endpoint."""
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        try:
            with httpx.Client(timeout=self.settings.timeout_seconds) as client:
                resp = client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}", retryable=True) from exc
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {data}", retryable=True) from exc


# ---------------------------------------------------------------------------
# Mock backend (offline / deterministic)
# ---------------------------------------------------------------------------


def _mock_complete(system: str, user: str, provider: str) -> str:
    """Deterministic stand-in used when no model is configured.

    It is enough for the Phase 1 pipeline: the prompt templates state that
    the model must return JSON, so the mock answers the two node prompts
    (intent, router, answer) with structurally-correct JSON derived from
    the user message.
    """
    task = user
    if "intent" in system.lower() or "intent" in task:
        # crude but deterministic intent guess
        lowered = task.lower()
        if any(k in lowered for k in ("是多少", "总额", "多少", "营收", "金额", "统计", "数量")):
            intent = "text_to_sql"
        elif any(k in lowered for k in ("制度", "政策", "规范", "流程", "规定", "手册", "要求")):
            intent = "knowledge_qa"
        else:
            intent = "information_query"
        return json.dumps(
            {
                "intent": intent,
                "slots": {},
                "constraints": {},
                "confidence": 0.9,
                "reason": "mock backend: keyword heuristic",
            },
            ensure_ascii=False,
        )
    if "router" in system.lower() or "route" in system.lower():
        return json.dumps(
            {"selected_tools": ["sql"], "routing_plan": "mock: default sql path", "confidence": 0.9},
            ensure_ascii=False,
        )
    if "text-to-sql" in system.lower() or "text to sql" in system.lower():
        # Deterministic mock SQL good enough to exercise the guard + executor:
        return json.dumps(
            {
                "sql": "SELECT department_id, COUNT(*) AS n, SUM(amount) AS total FROM finance_expenses WHERE approved = true GROUP BY department_id ORDER BY total DESC",
                "rationale": "mock backend: aggregate approved reimbursement by department",
            },
            ensure_ascii=False,
        )
    # answer node: echo the question and declared evidence, as plain text
    return f"[mock/{provider}] Answer to: {task}\n(证据以 AgentState.evidence 中的条目为准，mock 后端不产生真实结论)"


__all__ = ["LLMClient"]
