"""Node: Answer Guard — post-verification answer policy (Phase 4).

Sits *after* verification and *before* the final answer.  It never
regenerates data and never re-invokes tools (no self-healing loop in
this version — spec).  It only decides how the verified claims are
presented to the user, and can *block* a critical unsupported answer in
favour of an honest clarification.

Policy (deterministic, auditable via ``state["guard_decision"]``):

- **supported** claims      -> included in the answer, with evidence refs.
- **unsupported** claims    -> not presented as fact; flagged in a
  "待核实" (to-be-verified) section.
- **conflict** claims       -> presented with an explicit
  "数据来源存在冲突" (data-source conflict) notice.
- **critical unsupported**  -> the guard *blocks* the definitive answer:
  ``block=True`` and the final answer is replaced with an honest
  clarification asking for more information (spec: "可以阻止最终回答
  并要求更多信息").

The guard also assembles the user-facing ``answer_sources`` list
(doc titles / chunk refs / tables / templates) so the API can expose
``sources`` without leaking internal credentials or stack traces.
"""

from __future__ import annotations

from typing import Any

from src.core.observability import observe
from src.core.state import AgentState, EvidenceItem

__all__ = ["AnswerGuardNode"]

_CRITICAL_BLOCK_MESSAGE = (
    "当前回答中存在**未经证据支持的关键结论**（critical unsupported）。"
    "为避免输出无法追溯的信息，请提供更多信息或放宽问题范围，"
    "以便重新取证。已完成的中间证据保留在 sources 中供核对。"
)


def _is_critical(claim: dict[str, Any]) -> bool:
    return str(claim.get("importance", "normal")) == "critical"


def _evidence_index(state: AgentState) -> dict[str, EvidenceItem]:
    idx: dict[str, EvidenceItem] = {}
    for ev in (state.get("evidence") or []):
        idx[ev.evidence_id] = ev
    return idx


def _source_label(ev: EvidenceItem) -> str:
    """A user-safe source label (doc title / table / KG template)."""
    meta = ev.metadata or {}
    title = meta.get("title")
    if title:
        return str(title)
    stype = ev.source_type
    if stype in ("milvus", "fake"):
        return f"知识库文档 {ev.source_ref}"
    if stype == "duckdb":
        return f"数据查询 {ev.source_ref[:80]}"
    if stype == "neo4j":
        return f"知识图谱 {ev.source_ref[:80]}"
    return ev.source_ref


class AnswerGuardNode:
    """Apply the post-verification policy to the answer.

    Reads ``state["verification_results"]`` + ``state["claims"]`` +
    ``state["evidence"]``.  Writes ``answer`` (possibly annotated or
    replaced), ``answer_sources`` (user-safe citations), and
    ``guard_decision`` (the audit record).
    """

    @observe("answer_guard")
    def run(self, state: AgentState) -> AgentState:
        results: list[dict[str, Any]] = list(state.get("verification_results") or [])
        claims: list[dict[str, Any]] = list(state.get("claims") or [])
        answer = str(state.get("answer") or "")
        summary: dict[str, Any] = dict(state.get("verification_summary") or {})

        # No verification ran (verification disabled / no claims) -> pass
        # the answer through untouched, no guard annotations.
        if not results:
            return {
                "guard_decision": {"block": False, "reason": "no_verification_ran"},
                "answer_sources": _collect_sources(state, []),
            }

        claim_by_id = {str(c.get("claim_id")): c for c in claims}
        critical_unsupported: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        unsupported: list[dict[str, Any]] = []

        for r in results:
            claim = claim_by_id.get(str(r.get("claim_id")), {})
            if r.get("conflict"):
                conflicts.append(r)
            elif not r.get("supported"):
                unsupported.append(r)
                if _is_critical(claim):
                    critical_unsupported.append(r)

        block = bool(critical_unsupported)
        decision = {
            "block": block,
            "critical_unsupported": [r["claim_id"] for r in critical_unsupported],
            "conflicts": [r["claim_id"] for r in conflicts],
            "unsupported": [r["claim_id"] for r in unsupported],
            "supported_count": int(summary.get("supported_claims", 0)),
        }

        if block:
            new_answer = _CRITICAL_BLOCK_MESSAGE + annotate_answer(answer, conflicts, unsupported, critical_unsupported)
            status = "guard_blocked"
        else:
            new_answer = annotate_answer(answer, conflicts, unsupported, [])
            status = "guarded" if (conflicts or unsupported) else "verified"

        return {
            "answer": new_answer,
            "guard_decision": decision,
            "answer_sources": _collect_sources(state, results),
            "status": status,
        }


def _annotate(
    answer: str,
    conflicts: list[dict[str, Any]],
    unsupported: list[dict[str, Any]],
    critical_unsupported: list[dict[str, Any]],
) -> str:
    blocks = [answer]
    if conflicts:
        ids = ", ".join(str(r["claim_id"]) for r in conflicts)
        blocks.append(
            "\n\n> ⚠️ **数据来源存在冲突**：结论 " + ids +
            " 在不同证据源中不一致，请人工核对后再采信。"
        )
    if unsupported:
        ids = ", ".join(str(r["claim_id"]) for r in unsupported)
        extra = "（含关键结论，已阻止最终确认）" if critical_unsupported else ""
        blocks.append(
            f"\n\n> ℹ️ **待核实**：以下结论未找到支持证据{extra}：{ids}。"
        )
    return "".join(blocks)


def _collect_sources(state: AgentState, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """User-safe citation list: only doc titles / chunk / table / template."""
    idx = _evidence_index(state)
    cited_ids: list[str] = []
    seen: set[str] = set()
    for r in results:
        for eid in r.get("evidence_ids", []):
            if eid in idx and eid not in seen:
                seen.add(eid)
                cited_ids.append(eid)
    if not cited_ids:
        # no verified claims: still surface the raw evidence refs so the
        # user knows what was consulted (no credentials involved).
        cited_ids = [ev.evidence_id for ev in (state.get("evidence") or [])]
    out: list[dict[str, Any]] = []
    for eid in cited_ids:
        ev = idx.get(eid)
        if ev is None:
            continue
        out.append({
            "evidence_id": eid,
            "source_type": ev.source_type,
            "source": _source_label(ev),
            "score": ev.score,
        })
    return out


# expose the annotator for direct testing without a node instance
def annotate_answer(
    answer: str,
    conflicts: list[dict[str, Any]],
    unsupported: list[dict[str, Any]],
    critical_unsupported: list[dict[str, Any]] | None = None,
) -> str:
    return _annotate(answer, conflicts, unsupported, critical_unsupported or [])


__all__ = ["AnswerGuardNode", "annotate_answer"]
