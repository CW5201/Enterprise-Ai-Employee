"""Node: Claim extraction — turn the generated answer into structured claims.

Phase 4 Commit 2.  The final answer is the *output* of the LLM; it must not
be trusted on its own.  This node decomposes the answer into checkable
:class:`src.core.verification_types.Claim` records so the verification
engine can check each one against *real* evidence.

Design rules (RQ2 discipline):

- LLM structured output is the primary path (``generate_structured``),
  always Pydantic-validated.  Free-text answers stay intact inside
  ``Claim.text`` — we never regex-chop the answer into "sentences" as
  the final implementation.
- Failure is **never silent**: an LLM error or a schema-invalid payload
  raises :class:`ClaimExtractionError` (recorded as an ``errors`` item,
  and the downstream verification stage treats the claim set as
  *extraction_failed* — it does not pretend claims were extracted).
- Regex is only used for two *deterministic post-processing* steps,
  explicitly labelled as such: (a) parse a scalar ``value`` out of a
  numeric claim text, and (b) detect "增长 / 下降 / 环比" derived
  markers.  Both are pure text analysis of already-extracted claims,
  not the extraction itself.

Output is written to ``state["claims"]`` (append channel) as plain
dicts via ``Claim.model_dump()``.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.core.exceptions import LLMError
from src.core.llm_client import LLMClient
from src.core.observability import observe
from src.core.state import AgentState, ErrorRecord
from src.core.verification_types import Claim

__all__ = ["ClaimExtractionError", "ClaimExtractionNode", "ClaimsOut"]


class ClaimExtractionError(Exception):
    """Raised when claim extraction fails (LLM error or schema failure).

    The caller records an ``ErrorRecord``; no silent fallback to
    regex-chopped claims is allowed.
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


# ---------------------------------------------------------------------------
# Structured LLM schema
# ---------------------------------------------------------------------------


class _ClaimOut(BaseModel):
    """One claim as the LLM proposes it (pre-validation)."""

    text: str = Field(min_length=2)
    claim_type: Literal[
        "factual", "numerical", "relational", "rule_based", "derived",
        "opinion_or_summary",
    ] = "factual"
    importance: Literal["critical", "normal", "minor"] = "normal"
    source_refs: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    formula: str = ""

    @field_validator("claim_type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        from src.core.verification_types import CLAIM_TYPES

        if v not in CLAIM_TYPES:
            raise ValueError(f"unknown claim_type {v!r}")
        return v


class _ExtractionPayload(BaseModel):
    claims: list[_ClaimOut]


class ClaimsOut(BaseModel):
    """Wrapper so the structured-output call has a stable top-level key."""

    claims: list[_ClaimOut] = Field(default_factory=list)


_CLAIM_SYSTEM = """你是企业 AI 数字员工的结论抽取模块。
把一段"最终答案"拆分成可逐条核验的 claim（结论/断言）。
规则：
1. 只抽取答案里实际出现的断言；不要把问题或推理过程算作 claim；
2. 每条 claim 必须具体、可被数据/文档证据支持或反驳，不能是空泛套话；
3. 数值结论（销售额、数量、日期、百分比）用 claim_type=numerical；
   规则/政策/制度类用 rule_based；实体关系（A 购买了 B、A 属于 B）用
   relational；由其他数值计算得到的（增长率、占比、均值）用 derived，
   并在 formula 字段写出计算式、在 entities 里列出用到的源数值；
   一般事实陈述用 factual；总结性/评价性表述用 opinion_or_summary；
4. importance：critical = 核心数据结论或会影响决策的断言；normal = 一般
   事实；minor = 附带说明。
返回 JSON: {{"claims": [{{"text": str, "claim_type": str, "importance": str,
"source_refs": [str], "entities": [str], "formula": str}}]}}"""


# ---------------------------------------------------------------------------
# Deterministic post-processing (NOT the extraction itself)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|元|件|个|条|天|人|笔|家|名))?", re.UNICODE)
_GROWTH_MARKERS = ("增长", "上涨", "下降", "下跌", "环比", "同比", "提升", "减少")


def _parse_scalar(text: str) -> float | int | None:
    """Best-effort scalar extraction from a numeric claim text."""
    m = _NUM_RE.search(text)
    if not m:
        return None
    raw = m.group(0).strip()
    is_pct = raw.endswith("%")
    # keep only the leading numeric token (drop unit suffix like 元/件/人)
    num_match = re.match(r"-?\d[\d,]*(?:\.\d+)?", raw)
    if not num_match:
        return None
    cleaned = num_match.group(0).replace(",", "")
    try:
        val: float = float(cleaned)
    except ValueError:
        return None
    if is_pct:
        return round(val / 100.0, 6)
    return int(val) if val == int(val) and abs(val) < 1e15 else val


def _normalise_claim(out: _ClaimOut, claim_id: str) -> dict[str, Any]:
    """Turn a validated LLM claim into a Claim dict with deterministic fields."""
    value = _parse_scalar(out.text) if out.claim_type in ("numerical", "derived") else None
    formula = out.formula.strip()
    if not formula and out.claim_type == "derived" and any(mk in out.text for mk in _GROWTH_MARKERS):
        # mark the claim as derived-without-explicit-formula so the verifier
        # can still recompute from evidence when possible; never pretend a
        # formula was given.
        formula = "(derived, formula not explicit)"
    claim = Claim(
        claim_id=claim_id,
        text=out.text,
        claim_type=out.claim_type,  # type: ignore[arg-type]
        importance=out.importance,  # type: ignore[arg-type]
        source_refs=list(out.source_refs),
        entities=[e for e in out.entities if e],
        formula=formula,
        value=value,
    )
    return claim.model_dump()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class ClaimExtractionNode:
    """Extract structured claims from the generated answer.

    Input:  ``state["answer"]``, ``state["user_query"]``.
    Output: ``state["claims"]`` (append), ``state["errors"]`` (append).

    Construction:
    - ``llm=None`` builds a default :class:`LLMClient` (offline backend in
      tests when no endpoint is configured — the payload is structurally
      correct but its ``reason`` carries the ``offline: true`` marker).
    - ``extract_fn`` injects a deterministic extractor for unit tests /
      offline evaluation (no LLM call at all).
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        extract_fn: Any | None = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._extract_fn = extract_fn

    @observe("claim_extraction")
    def run(self, state: AgentState) -> AgentState:
        answer = str(state.get("answer") or "")
        question = str(state.get("user_query") or "")
        errors: list[ErrorRecord] = []

        if not answer.strip():
            # No answer -> nothing to verify.  Honest empty result, not an
            # error: the guard stage will mark the task as no-evidence.
            return {"claims": [], "errors": errors}

        try:
            raw = self._extract(answer, question)
        except ClaimExtractionError as exc:
            errors.append(ErrorRecord(
                stage="claim_extraction",
                message=exc.message,
                details=exc.details,
            ))
            # Explicit marker so the verification stage knows extraction
            # failed; downstream claims are NOT silently replaced.
            return {
                "claims": [],
                "errors": errors,
                "claim_extraction_failed": True,
                "status": "claims_extraction_failed",
            }

        claims: list[dict[str, Any]] = [
            _normalise_claim(c, f"clm-{i:03d}") for i, c in enumerate(raw, start=1)
        ]
        return {
            "claims": claims,
            "errors": errors,
            "status": "claims_extracted" if claims else "claims_empty",
        }

    # -- extraction lanes ------------------------------------------------------

    def _extract(self, answer: str, question: str) -> list[_ClaimOut]:
        if self._extract_fn is not None:
            payload = self._extract_fn(answer, question)
            model: type[BaseModel] = _ExtractionPayload
            data: dict[str, Any]
            if isinstance(payload, list):
                # injected extractor may return the bare claim list
                data = {"claims": payload}
            elif isinstance(payload, dict) and "claims" in payload:
                data = payload
            else:
                data = {"claims": []}
            return model.model_validate(data).claims

        user = f"用户问题: {question}\n\n最终答案:\n{answer}"
        try:
            result: ClaimsOut = self._llm.generate_structured(_CLAIM_SYSTEM, user, ClaimsOut)
        except LLMError as exc:
            raise ClaimExtractionError(
                f"claim extraction LLM failed: {exc.message}",
                details={"retryable": exc.retryable},
            ) from exc
        except Exception as exc:  # noqa: BLE001 — malformed structured output
            raise ClaimExtractionError(
                f"claim extraction payload failed validation: {exc}",
            ) from exc
        return result.claims


__all__ = ["ClaimExtractionError", "ClaimExtractionNode", "ClaimsOut"]
