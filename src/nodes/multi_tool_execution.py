"""Node: Multi-Tool Execution + Result Fusion (Phase 3 Commit 3).

Executes a validated :class:`RoutingDecision`'s ``execution_plan`` in
dependency order and fuses the results into the unified
``tool_results`` / ``evidence`` channels, so the answer node sees *all*
successful tools.

Principles (ADR-009 discipline carried into Phase 3):

- READS the decision — it never re-routes.  If the decision names a step
  that cannot run (unsupported tool, missing KG template), that step is
  recorded as an honest failure, **not** silently replaced by another tool.
- Order = execution plan order.  A step runs only after its
  ``depends_on`` steps have succeeded; a failure marks dependents as
  ``skipped_upstream_failed`` and the pipeline continues with what worked.
- No overwrite: every tool's result is appended to ``tool_results``;
  a failing tool never clobbers earlier successful results.
- Evidence metadata is preserved (unified ``EvidenceItem``) so Phase 4
  claim-evidence verification can consume it.

This node is the *orchestrator*: it dispatches to the existing tool
implementations and does not reimplement any of them.
"""

from __future__ import annotations

import time
from typing import Any, cast

from src.core.observability import observe
from src.core.state import (
    AgentState,
    ErrorRecord,
    EvidenceItem,
    RetrievedChunk,
    ToolCallRecord,
)
from src.tools.analysis_tool import AnalysisTool
from src.tools.kg_tool import KGTool
from src.tools.rag_tool import RAGTool

# Tools the orchestrator can dispatch.  Any step tool NOT in this set is
# recorded as skipped_unsupported (no silent fallback).
_SUPPORTED = {"sql", "rag", "kg", "analysis"}


def _to_chunk(item: dict[str, Any]) -> RetrievedChunk:
    """Normalise one RAG hit into a ``RetrievedChunk`` (same shape as the
    standalone RAG node produces, so the answer prompt sees identical data)."""
    return RetrievedChunk(
        chunk_id=str(item.get("chunk_id", "")),
        text=str(item.get("text", "")),
        source=str(item.get("source", "")),
        metadata={
            **(item.get("metadata") or {}),
            "document_id": item.get("document_id", ""),
            "title": item.get("title", ""),
            "category": item.get("category", ""),
            "dense_score": item.get("dense_score", 0.0),
            "bm25_score": item.get("bm25_score", 0.0),
            "fusion_score": item.get("fusion_score", 0.0),
        },
        score=float(item.get("fusion_score", item.get("score", 0.0)) or 0.0),
    )


def _evidence_ref(payload: dict[str, Any]) -> str | None:
    """Inspectable reference for one dispatched step (see state.py contract)."""
    ids: list[str] = []
    for item in payload.get("evidence") or []:
        if isinstance(item, EvidenceItem):
            ids.append(item.evidence_id)
        elif isinstance(item, dict) and item.get("evidence_id"):
            ids.append(str(item["evidence_id"]))
    if ids:
        return ",".join(ids[:3])
    tool = str(payload.get("tool") or "tool")
    return f"{tool}:step{payload.get('step')}" if payload.get("step") is not None else tool


def _call_input(tool: str, step_id: int, depends: list[int], state: AgentState) -> dict[str, Any]:
    """The parameters that identify one step invocation."""
    payload: dict[str, Any] = {"step": step_id, "depends_on": list(depends)}
    if tool == "sql" and state.get("sql"):
        payload["sql"] = str(state.get("sql"))
    if tool == "rag":
        payload["query"] = str(state.get("user_query") or "")
    return payload


def _sql_failure_reason(out: dict[str, Any]) -> str:
    """A diagnosable failure string instead of a generic placeholder.

    The SQL node already records the LLM/executor failure in ``errors``
    together with the statement it tried to run.  Dropping both made an
    intermittent bad generation indistinguishable from an empty result set,
    which is impossible to debug from the API payload alone.
    """
    records = out.get("errors") or []
    if not records:
        return "sql execution failed or produced no result"
    parts = []
    for rec in records:
        stage = getattr(rec, "stage", None) or "?"
        msg = getattr(rec, "message", None) or str(rec)
        parts.append(f"{stage}: {msg}")
    return "; ".join(parts)


class MultiToolExecutionNode:
    """Run a RoutingDecision execution_plan and fuse the results."""

    def __init__(
        self,
        sql_node: Any | None = None,
        rag_tool: RAGTool | None = None,
        kg_tool: KGTool | None = None,
        analysis_tool: AnalysisTool | None = None,
        llm: Any | None = None,
    ) -> None:
        # lazy import to avoid cycles with the graph builder
        from src.nodes.sql_execution import SQLExecutionNode

        self._sql_node = sql_node or SQLExecutionNode(llm=llm)
        self._rag_tool = rag_tool
        self._kg_tool = kg_tool
        self._analysis_tool = analysis_tool or AnalysisTool()

    @observe("multi_tool_execution")
    def run(self, state: AgentState) -> AgentState:
        decision = state.get("routing_decision") or {}
        plan = list(decision.get("execution_plan") or [])
        errors: list[ErrorRecord] = []
        tool_results: list[dict[str, Any]] = []
        tool_calls: list[ToolCallRecord] = []
        evidence: list[EvidenceItem] = []

        if not plan:
            errors.append(ErrorRecord(stage="execution", message="empty execution plan"))
            return {"tool_results": [], "errors": errors, "status": "execution_no_plan"}

        succeeded: set[int] = set()
        # The orchestrator feeds earlier steps' results into later ones
        # (e.g. analysis reads the sql step's table) via a live working list
        # that it also exposes as ``state["tool_results"]`` for downstream
        # nodes; we mirror it into a private channel so dispatch can read it.
        working_results: list[dict[str, Any]] = []

        # The answer node reads the sql / sql_result / retrieved_context
        # channels directly and never looks inside tool_results, so the
        # orchestrator lifts the executed tool output back into those
        # channels.  Without this every phase-3 run reaches the answer node
        # with no evidence and answers "the tool produced no data".
        last_sql = ""
        last_sql_result: Any | None = None
        rag_chunks: list[RetrievedChunk] = []

        for step in sorted(plan, key=lambda s: int(s.get("step", 0))):
            tool = str(step.get("tool", ""))
            step_id = int(step.get("step", 0))
            audit_id = f"{state.get('request_id', 'req')}-step{step_id}"

            # dependency gate: all depends_on must have succeeded
            depends = [int(d) for d in (step.get("depends_on") or [])]
            missing = [d for d in depends if d not in succeeded]
            if missing:
                tool_results.append({
                    "tool": tool, "success": False, "error": "skipped_upstream_failed",
                    "step": step_id, "depends_on": depends,
                })
                # every invocation is recorded in the tool ledger, including
                # the ones that never ran (state.py contract / ARCHITECTURE §4)
                tool_calls.append(ToolCallRecord(
                    tool=tool, ok=False, input=_call_input(tool, step_id, depends, state),
                    error="skipped_upstream_failed", evidence_ref=None,
                    duration_ms=0.0, audit_id=audit_id,
                ))
                working_results.append(tool_results[-1])
                errors.append(ErrorRecord(
                    stage="execution",
                    message=f"step {step_id} ({tool}) skipped: upstream {missing} not satisfied",
                    details={"step": step_id, "tool": tool, "depends_on": missing},
                ))
                continue

            # dispatch (pass the working list so analysis sees upstream data)
            state["_working_tool_results"] = working_results
            t0 = time.perf_counter()
            payload = dict(self._dispatch(tool, state, step))
            payload["step"] = step_id
            payload.setdefault("latency_ms", round((time.perf_counter() - t0) * 1000.0, 2))

            ok = bool(payload.get("success"))
            tool_calls.append(ToolCallRecord(
                tool=tool,
                ok=ok,
                input=_call_input(tool, step_id, depends, state),
                error=None if ok else str(payload.get("error") or "failed"),
                evidence_ref=_evidence_ref(payload),
                duration_ms=float(payload.get("latency_ms") or 0.0),
                audit_id=audit_id,
            ))

            if ok:
                succeeded.add(step_id)
                evidence.extend(payload.get("evidence") or [])
                if tool == "sql":
                    last_sql = str(payload.get("sql") or "")
                    last_sql_result = payload.get("sql_result")
                elif tool == "rag":
                    rag_chunks.extend(payload.get("retrieved_context") or [])
            else:
                err_details = payload.get("error_details")
                details: dict[str, Any] = err_details if isinstance(err_details, dict) else (
                    {"error": err_details} if err_details else {}
                )
                errors.append(ErrorRecord(
                    stage="execution",
                    message=f"step {step_id} ({tool}) failed: {payload.get('error')}",
                    details=details,
                ))

            tool_results.append(payload)
            working_results.append(payload)

        return {
            "tool_results": tool_results,
            "tool_calls": tool_calls,
            "errors": errors,
            "evidence": evidence,
            "status": "executed" if succeeded else "execution_all_failed",
            # Lifted tool output for AnswerGenerationNode (see the comment at
            # the tracking variables above).  An empty retrieved_context is a
            # no-op for the _append reducer; sql / sql_result are
            # last-write-wins and simply keep their make_state() defaults
            # when no sql step succeeded.
            "sql": last_sql,
            "sql_result": last_sql_result,
            "retrieved_context": rag_chunks,
        }

    # -- dispatch -------------------------------------------------------------

    def _dispatch(self, tool: str, state: AgentState, step: dict[str, Any]) -> dict[str, Any]:
        if tool not in _SUPPORTED:
            return {"tool": tool, "success": False, "error": "skipped_unsupported",
                    "error_details": {}, "data": {}, "evidence": []}
        if tool == "sql":
            return self._run_sql(state)
        if tool == "rag":
            return self._run_rag(state)
        if tool == "kg":
            return self._run_kg(state, step)
        if tool == "analysis":
            return self._run_analysis(state, step)
        return {"tool": tool, "success": False, "error": f"tool {tool!r} not dispatchable",
                "error_details": {}, "data": {}, "evidence": []}

    def _run_sql(self, state: AgentState) -> dict[str, Any]:
        """Run the SQL node; lift its result into the uniform payload."""
        out = cast(AgentState, dict(self._sql_node.run(state)))
        sql_result = out.get("sql_result")
        ok = sql_result is not None
        columns = list(sql_result.columns) if ok else []
        rows = list(sql_result.rows) if ok else []
        return {
            "tool": "sql",
            "success": ok,
            # AnswerGenerationNode reads the ``sql`` / ``sql_result`` channels
            # directly and never inspects ``tool_results``, so the executor
            # has to surface them here for run() to lift back into state.
            "sql": str(out.get("sql") or state.get("sql") or ""),
            "sql_result": sql_result,
            "data": {"columns": columns, "rows": rows, "row_count": len(rows)},
            "error": None if ok else _sql_failure_reason(out),
            "error_details": ([e.details for e in out.get("errors", [])] if not ok else {}),
            "evidence": out.get("evidence") or [],
        }

    def _run_rag(self, state: AgentState) -> dict[str, Any]:
        tool = self._rag_tool or RAGTool()
        res = tool.run(query=str(state.get("user_query", "")))
        ok = bool(res.get("ok"))
        hits = res.get("results", [])
        chunks = [_to_chunk(h) for h in hits]
        return {
            "tool": "rag",
            "success": ok,
            # Same reason as _run_sql: the answer node reads retrieved_context,
            # not tool_results, so the chunks have to be lifted into state.
            "retrieved_context": chunks,
            "data": {"hits": len(hits), "top": [
                {"chunk_id": h.get("chunk_id"), "source": h.get("source"), "score": h.get("score")}
                for h in hits[:3]
            ]},
            "results": hits,
            "error": None if ok else str(res.get("error")),
            "error_details": {},
            "evidence": [
                EvidenceItem(
                    evidence_id=f"ev-{state.get('request_id', 'req')}-rag-{h.get('chunk_id', i)}",
                    source_type="milvus",
                    source_ref=f"milvus:{h.get('chunk_id', '')}",
                    content=str(h.get("text", ""))[:300],
                    payload={"chunk_id": h.get("chunk_id"), "source": h.get("source"),
                             "score": h.get("score", 0.0)},
                    metadata=h.get("metadata") or {},
                )
                for i, h in enumerate(hits)
            ],
        }

    def _run_kg(self, state: AgentState, step: dict[str, Any]) -> dict[str, Any]:
        # The KG tool needs a concrete template_id + parameters.  When the
        # decision does not carry an explicit template, we cannot guess one
        # — record an honest unsupported result instead of fabricating a query.
        template = str(step.get("template_id") or "")
        if not template:
            return {
                "tool": "kg", "success": False,
                "error": "skipped_no_template: no kg template_id in the execution step",
                "error_details": {}, "data": {}, "evidence": [],
            }
        kg = self._kg_tool or KGTool()
        try:
            res = kg.run(template_id=template, parameters=step.get("parameters") or {})
        except Exception as exc:  # noqa: BLE001
            return {"tool": "kg", "success": False, "error": f"kg failure: {exc}",
                    "error_details": {}, "data": {}, "evidence": []}
        finally:
            kg.close()
        ok = bool(res.get("success"))
        return {
            "tool": "kg",
            "success": ok,
            "data": {"rows": res.get("rows"), "query_type": res.get("query_type")},
            "error": None if ok else (res.get("error") or {}).get("code"),
            "error_details": (res.get("error") or {}),
            "evidence": [],
        }

    def _run_analysis(self, state: AgentState, step: dict[str, Any]) -> dict[str, Any]:
        # The analysis tool consumes the *immediately preceding* data tool's
        # result (here: the last successful sql step in tool_results).
        params = dict(step.get("parameters") or step.get("params") or {})
        op = str(params.get("operation", params.get("op", "mean")))
        value = params.get("value")
        tool_results = list(state.get("_working_tool_results") or state.get("tool_results") or [])
        sql_data = next(
            (tr for tr in reversed(tool_results) if tr.get("tool") == "sql" and tr.get("success")),
            None,
        )
        rows: list[Any] = []
        columns: list[str] = []
        if sql_data and sql_data.get("data"):
            rows = sql_data["data"].get("rows", [])
            columns = sql_data["data"].get("columns", [])
        res = self._analysis_tool.run(
            operation=op,
            rows=rows or None,
            columns=columns or None,
            params={"value": value} if value else {},
        )
        ok = bool(res.ok)
        return {
            "tool": "analysis",
            "success": ok,
            "data": (res.data or {}) if ok else {},
            "error": None if ok else str(res.error),
            "error_details": {},
            "evidence": [],
        }


__all__ = ["MultiToolExecutionNode", "_SUPPORTED"]
