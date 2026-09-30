"""Node: Evidence collection — gather + dedupe Phase 4 evidence.

Phase 4 Commit 2.  Builds the unified ``evidence_v4`` channel from every
real tool result that reached the state, via :mod:`src.core.evidence_adapter`
(the *only* module that knows each tool's native shape).

Sources (in priority order; later ones only add source types not yet
covered):

1. ``state["tool_results"]`` — Phase 3 multi-tool payloads (sql / rag /
   kg / analysis), adapted per tool.
2. ``state["evidence"]`` — legacy Phase 1/2 :class:`EvidenceItem` channel
   (sql / rag evidence produced by the earlier nodes), adapted to Phase 4.
3. ``state["sql_result"]`` + ``state["sql"]`` — direct SQL node result.
4. ``state["retrieved_context"]`` — RAG chunks (covers Phase 2 wiring where
   no Phase 3 tool_results exist).

Guarantees:

- Deduplication by ``source_ref`` (identical source never appears twice).
- Provenance preserved per source type (query for SQL, template for KG,
  doc/chunk for RAG, operation+inputs for analysis).
- No evidence is ever fabricated from LLM text: every record traces to a
  tool result that actually ran.
"""

from __future__ import annotations

from src.core.evidence_adapter import (
    adapt_rag,
    adapt_sql,
    adapt_state_evidence,
    adapt_tool_result,
)
from src.core.observability import observe
from src.core.state import AgentState
from src.core.verification_types import Evidence

__all__ = ["EvidenceCollectionNode", "collect_evidence"]


def collect_evidence(state: AgentState) -> list[Evidence]:
    """Collect + dedupe Phase 4 evidence from every source channel."""
    evidence: list[Evidence] = []
    seen: set[str] = set()

    def add(items: list[Evidence]) -> None:
        for ev in items:
            key = ev.source_ref or ev.evidence_id
            if key in seen:
                continue
            seen.add(key)
            evidence.append(ev)

    # 1. Phase 3 multi-tool results (richest provenance)
    step_index = 0
    for tr in (state.get("tool_results") or []):
        tool = str(tr.get("tool", ""))
        if not tr.get("success"):
            continue
        step_index += 1
        step_id = tr.get("step", step_index)
        data = tr.get("data") or {}
        if tool == "sql":
            sql_text = str(tr.get("sql") or "").strip()
            rows = data.get("rows") or []
            add([adapt_sql(
                {"sql": sql_text, "columns": data.get("columns") or [],
                 "rows": rows, "row_count": len(rows)},
                evidence_id=f"ev-tool-sql-{step_id}",
            )])
        elif tool == "rag":
            hits = tr.get("results") or data.get("hits") or []
            add([adapt_rag(h) for h in hits])
        elif tool == "kg":
            add(adapt_tool_result("kg", {
                "success": tr.get("success"),
                "data": data.get("rows") or [],
                "template_id": data.get("template_id") or data.get("query_type") or "",
                "query_type": data.get("query_type") or data.get("template_id") or "",
            }))
        elif tool == "analysis":
            # link back to the sql evidence that fed this analysis
            sql_refs = [e.source_ref for e in evidence if e.source_type == "sql"]
            add(adapt_tool_result("analysis", {
                "ok": tr.get("success"),
                "data": data,
                "metadata": {"operation": (data.get("result") or {}).get("operation", "")},
            }, input_evidence_ids=sql_refs))

    # 2. Legacy EvidenceItem channel (Phase 1/2 sql/rag nodes)
    add(adapt_state_evidence(state.get("evidence") or []))

    # 3. Direct SQL node result (Phase 1/2 wiring)
    sql_result = state.get("sql_result")
    if sql_result is not None:
        add([adapt_sql({
            "sql": str(state.get("sql") or ""),
            "columns": list(getattr(sql_result, "columns", [])),
            "rows": list(getattr(sql_result, "rows", [])),
            "row_count": int(getattr(sql_result, "row_count", 0)),
        })])

    # 4. RAG chunks (Phase 2 wiring without tool_results)
    for chunk in (state.get("retrieved_context") or []):
        add([adapt_rag(chunk)])

    return evidence


class EvidenceCollectionNode:
    """Collect + dedupe Phase 4 evidence into the ``evidence_v4`` channel."""

    @observe("evidence_collection")
    def run(self, state: AgentState) -> AgentState:
        evidence = collect_evidence(state)
        return {
            "evidence_v4": [e.model_dump() for e in evidence],
        }


__all__ = ["EvidenceCollectionNode", "collect_evidence"]
