"""API route: POST /api/chat/stream - SSE streaming of the task pipeline.

Phase 6.3: Server-Sent Events endpoint that streams the pipeline's
per-node progress to the frontend in real time.  The endpoint wraps the
existing DigitalEmployee (no pipeline re-implementation); it simply
yields an SSE event after each LangGraph node completes.

Event schema (JSON in the ``data`` field):
    {
        "type": "task_started | routing | tool | verification | answer |
                 guard | completed | error",
        "timestamp": "ISO-8601",
        "step": "node name",
        "status": "ok | error",
        "payload": { ... node-specific data ... }
    }

Security: payload is filtered.  Every value leaves through ``_to_plain``
(models -> JSON objects) and, for anything that can carry raw business data,
``_strip_raw_content`` / ``_sanitize_evidence``: raw SQL rows, full RAG hit
lists, chunk text and evidence payloads are never streamed, and no passwords,
secrets or stack traces are included (see ADR-014).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.api.routes.chat import ChatRequest
from src.api.routes.reports import record_run_report, should_record_report
from src.api.routes.tasks import record_task, task_status_for_run
from src.core.state import APPEND_CHANNELS

router = APIRouter(tags=["chat-stream"])
logger = logging.getLogger("eae.api.chat_stream")

# Map LangGraph node names to frontend event types.  Tool nodes emit a
# tool_started event (plan) followed by tool_completed (results), so the
# trace can show a pending -> running -> success transition per tool.
_NODE_EVENT_MAP: dict[str, str] = {
    "intent": "task_started",
    "route": "routing",
    "sql": "tool",
    "rag": "tool",
    "execute": "tool",
    "clarify": "answer",
    "answer": "answer",
    "claim_extraction": "verification",
    "evidence_collection": "verification",
    "verification": "verification",
    "answer_guard": "guard",
}


def _ts() -> str:
    """UTC ISO-8601 timestamp (millisecond precision) for event framing."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time() % 1) * 1000):03d}Z"


def _to_plain(value: Any) -> Any:
    """Normalise a state value for the wire: models -> dicts, lists -> lists.

    Nodes write Pydantic models into the state channels (``EvidenceItem``,
    ``ToolCallRecord``, ``SQLResultPayload`` …).  ``stream_mode="updates"``
    hands them back untouched, so converting here is what keeps the browser
    payload structured instead of a stringified repr.
    """
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    return value


def _sanitize_evidence(ev: Any) -> dict[str, Any]:
    """Strip raw row data / chunk text from an evidence record.

    The guard node only sanitises ``answer_sources``; the raw ``evidence``
    records in state still carry full row data and chunk text (PII).  The
    SSE stream exposes only user-safe fields.
    """
    data = _to_plain(ev)
    if not isinstance(data, dict):
        return {"source_type": "unknown", "source_ref": str(data)}
    out = {k: v for k, v in data.items() if k not in ("payload", "content")}
    out["row_count"] = data.get("row_count")
    return out


# Keys that carry raw business rows or document text.  They are dropped from
# everything the stream sends: identifiers, counts and scores are enough for
# the trace / evidence panel, and the disclosure rule for this endpoint is
# "user-safe projection only" (see the module docstring and ADR-013).
_RAW_CONTENT_KEYS: frozenset[str] = frozenset({
    "content", "payload", "rows", "results", "evidence", "error_details",
})


def _strip_raw_content(value: Any) -> Any:
    """Recursively drop raw rows / chunk text from a value headed for the wire.

    Applied to ``sql_result`` (drops ``rows``), ``tool_results`` (drops the
    per-tool ``data.rows`` / full RAG ``results``) and the Phase 4
    ``evidence_v4`` records (drops ``content``).
    """
    if isinstance(value, list):
        return [_strip_raw_content(v) for v in value]
    if isinstance(value, dict):
        return {
            k: _strip_raw_content(v)
            for k, v in value.items()
            if k not in _RAW_CONTENT_KEYS
        }
    return value


def _fold_delta(state: dict[str, Any], delta: dict[str, Any]) -> None:
    """Fold one node's ``updates`` delta into the running final state.

    Append channels must be *merged*: a node returns only the items it adds
    (``src/core/state.py::_append``), so assigning would drop every earlier
    node's items — the stored report used to keep only the answer node's
    evidence (FH-SSE-003).
    """
    for k, v in delta.items():
        if k in APPEND_CHANNELS and isinstance(v, list):
            state[k] = list(state.get(k) or []) + list(v)
        else:
            state[k] = v


def _safe_payload(node_name: str, delta: dict[str, Any]) -> dict[str, Any]:
    """Pick user-safe fields from the node state delta.

    Key names follow what each node ACTUALLY writes into AgentState
    (see src/nodes/*), not the node name itself — e.g. the
    claim_extraction node writes "claims", evidence_collection writes
    "evidence_v4".
    """
    p: dict[str, Any] = {}
    if node_name == "intent":
        p["intent"] = delta.get("intent", "")
        p["confidence"] = delta.get("confidence", 0.0)
        p["plan"] = delta.get("plan", "")
    elif node_name == "route":
        p["route"] = delta.get("route", "")
        p["route_confidence"] = delta.get("route_confidence", 0.0)
        p["tools_selected"] = delta.get("tools_selected", [])
        p["routing_decision"] = delta.get("routing_decision", {})
        p["routing_trace"] = delta.get("routing_trace", {})
    elif node_name in ("sql", "rag", "execute"):
        tool = node_name if node_name != "execute" else "multi_tool"
        p["tool"] = tool
        p["tool_calls"] = _to_plain(delta.get("tool_calls", []))
        p["sql"] = delta.get("sql", "")
        p["sql_result"] = _strip_raw_content(_to_plain(delta.get("sql_result")))
        p["evidence"] = [_sanitize_evidence(e) for e in delta.get("evidence", [])]
        p["tool_results"] = [_strip_raw_content(_to_plain(tr)) for tr in delta.get("tool_results", [])]
        p["errors"] = _to_plain(delta.get("errors", []))
    elif node_name == "answer":
        p["answer"] = delta.get("answer", "")
        p["intent"] = delta.get("intent", "")
        p["sql"] = delta.get("sql", "")
        p["evidence"] = [_sanitize_evidence(e) for e in delta.get("evidence", [])]
        p["tool_calls"] = _to_plain(delta.get("tool_calls", []))
        p["status"] = delta.get("status", "")
        p["latency"] = delta.get("latency", {})
    elif node_name == "claim_extraction":
        p["claims"] = _to_plain(delta.get("claims", []))
        p["status"] = delta.get("status", "")
    elif node_name == "evidence_collection":
        p["evidence_v4"] = [_strip_raw_content(_to_plain(e)) for e in delta.get("evidence_v4", [])]
    elif node_name == "verification":
        p["verification_results"] = _to_plain(delta.get("verification_results", []))
        p["verification_summary"] = _to_plain(delta.get("verification_summary", {}))
    elif node_name == "answer_guard":
        p["guard_decision"] = _to_plain(delta.get("guard_decision", {}))
        p["answer"] = delta.get("answer", "")
        p["answer_sources"] = _to_plain(delta.get("answer_sources", []))
    return p


@router.post("/chat/stream")
def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    """Stream pipeline events as SSE."""
    text = body.message.strip()
    if not text:
        raise HTTPException_400()

    employee = request.app.state.employee

    def _sse_generator() -> Generator[str, None, None]:
        # --- initial event ---
        yield _sse_event({"type": "task_started", "timestamp": _ts(), "step": "intent", "status": "running", "payload": {"message": text}})

        try:
            from src.core.state import make_state

            # LangGraph's .stream() accumulates state internally and does NOT
            # write back to the dict passed in, so read the final values by
            # folding the per-node deltas ourselves.
            final_state: dict[str, Any] = dict(make_state(text, request_id=body.request_id))

            for update in employee._graph.stream(final_state, stream_mode="updates"):
                for node_name, delta in update.items():
                    # fold this node's delta into the running final state
                    # (append channels merged, not overwritten)
                    _fold_delta(final_state, delta)
                    event_type = _NODE_EVENT_MAP.get(node_name, "step")
                    # Tool nodes: emit a started event before the results
                    # event so the frontend trace shows pending->running.
                    if event_type == "tool":
                        yield _sse_event({
                            "type": "tool_started",
                            "timestamp": _ts(),
                            "step": node_name,
                            "status": "running",
                            "payload": {"tool": _safe_payload(node_name, delta).get("tool", node_name)},
                        })
                    yield _sse_event({
                        "type": "tool_completed" if event_type == "tool" else event_type,
                        "timestamp": _ts(),
                        "step": node_name,
                        "status": "ok",
                        "payload": _safe_payload(node_name, delta),
                    })

            # --- final completed event (reads the folded final state) ---
            completed_payload: dict[str, Any] = {
                "answer": final_state.get("answer", ""),
                "status": final_state.get("status", ""),
                "route": final_state.get("route", ""),
                "intent": final_state.get("intent", ""),
                "sql": final_state.get("sql", ""),
                "latency": final_state.get("latency", {}),
            }
            # Evidence / tool_calls are sent sanitised: the same projection the
            # chat response exposes (source_type / source_ref / score, no raw
            # rows or chunk text), so the workbench's evidence panel and the
            # export match the one-shot endpoint instead of staying empty.
            # The report reuses this exact list - never a wider one.
            evidence_projection = [
                _sanitize_evidence(e) for e in final_state.get("evidence", [])
            ]
            completed_payload["evidence"] = evidence_projection
            completed_payload["tool_calls"] = _to_plain(final_state.get("tool_calls", []))
            completed_payload["verification"] = _to_plain(final_state.get("verification_results", []))
            completed_payload["sources"] = _to_plain(final_state.get("answer_sources", []))
            completed_payload["verification_summary"] = _to_plain(final_state.get("verification_summary", {}))
            completed_payload["guard"] = _to_plain(final_state.get("guard_decision", {}))

            terminal_status = str(final_state.get("status") or "")
            terminal_answer = str(final_state.get("answer") or "")

            # Phase 6.7: assemble the same run into a downloadable report.
            # The evidence was already sanitised for the browser, so the
            # report inherits exactly that projection (never more).  A report
            # failure is logged, never allowed to fail an otherwise
            # successful run.
            if should_record_report(status=terminal_status, answer=terminal_answer):
                try:
                    record_run_report(
                        request_id=final_state.get("request_id") or body.request_id or "",
                        question=text,
                        answer=terminal_answer,
                        evidence=evidence_projection,
                        verification_summary=final_state.get("verification_summary") or {},
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("report assembly failed for streamed run")

            # Streaming is the workbench's default mode, so a streamed run has
            # to land in 任务历史 too - it used to be missing there while its
            # report existed (FH-TASK-001).  History recording is best-effort:
            # it must never break an already successful run.
            try:
                record_task(
                    question=text,
                    status=task_status_for_run(status=terminal_status, answer=terminal_answer),
                    route=str(final_state.get("route") or ""),
                    answer=terminal_answer,
                    latency_ms=float((final_state.get("latency") or {}).get("total_ms") or 0.0),
                    task_id=str(final_state.get("request_id") or body.request_id or "") or None,
                )
            except Exception:  # noqa: BLE001
                logger.exception("failed to record streamed run in task history")

            yield _sse_event({
                "type": "completed",
                "timestamp": _ts(),
                "step": "end",
                "status": "ok",
                "payload": completed_payload,
            })
        except Exception:  # noqa: BLE001
            logger.exception("stream pipeline failed")
            # Surface a safe, generic message to the client; the internal
            # exception detail stays in the server log (never a stack trace
            # or secret in the SSE stream).
            yield _sse_event({
                "type": "error",
                "timestamp": _ts(),
                "step": "error",
                "status": "error",
                "payload": {"message": "任务执行失败，请稍后重试或联系管理员"},
            })

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(data: dict[str, Any]) -> str:
    # _to_plain first: Pydantic models must reach the browser as JSON objects,
    # not as the stringified repr that json.dumps(default=str) would produce.
    return f"data: {json.dumps(_to_plain(data), ensure_ascii=False, default=str)}\n\n"


def HTTPException_400() -> Any:
    from fastapi import HTTPException
    raise HTTPException(status_code=400, detail="`message` must be a non-empty string")


__all__ = ["router"]
