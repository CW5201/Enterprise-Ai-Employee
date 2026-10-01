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

Security: payload is filtered - no passwords, secrets, or stack traces
are included (the guard node already sanitises internal fields).
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


def _sanitize_evidence(ev: dict[str, Any]) -> dict[str, Any]:
    """Strip raw row data / chunk text from an evidence record.

    The guard node only sanitises ``answer_sources``; the raw ``evidence``
    records in state still carry full row data and chunk text (PII).  The
    SSE stream exposes only user-safe fields.
    """
    out = {k: v for k, v in ev.items() if k not in ("payload", "content")}
    out["row_count"] = ev.get("row_count")
    return out


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
        p["tool_calls"] = delta.get("tool_calls", [])
        p["sql"] = delta.get("sql", "")
        p["sql_result"] = delta.get("sql_result")
        p["evidence"] = [_sanitize_evidence(e) for e in delta.get("evidence", [])]
        p["tool_results"] = delta.get("tool_results", [])
        p["errors"] = delta.get("errors", [])
    elif node_name == "answer":
        p["answer"] = delta.get("answer", "")
        p["intent"] = delta.get("intent", "")
        p["sql"] = delta.get("sql", "")
        p["evidence"] = [_sanitize_evidence(e) for e in delta.get("evidence", [])]
        p["tool_calls"] = delta.get("tool_calls", [])
        p["status"] = delta.get("status", "")
        p["latency"] = delta.get("latency", {})
    elif node_name == "claim_extraction":
        p["claims"] = delta.get("claims", [])
        p["status"] = delta.get("status", "")
    elif node_name == "evidence_collection":
        p["evidence_v4"] = delta.get("evidence_v4", [])
    elif node_name == "verification":
        p["verification_results"] = delta.get("verification_results", [])
        p["verification_summary"] = delta.get("verification_summary", {})
    elif node_name == "answer_guard":
        p["guard_decision"] = delta.get("guard_decision", {})
        p["answer"] = delta.get("answer", "")
        p["answer_sources"] = delta.get("answer_sources", [])
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
            final_state: dict[str, Any] = make_state(text, request_id=body.request_id)

            for update in employee._graph.stream(final_state, stream_mode="updates"):
                for node_name, delta in update.items():
                    # fold this node's delta into the running final state
                    for k, v in delta.items():
                        final_state[k] = v
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
            # Evidence / tool_calls / verification rows are already in the
            # per-node step events; only expose them in `completed` when the
            # guard stripped internal fields.  This avoids streaming raw PII
            # rows a second time.
            completed_payload["verification"] = final_state.get("verification_results", [])
            completed_payload["sources"] = final_state.get("answer_sources", [])
            completed_payload["verification_summary"] = final_state.get("verification_summary", {})
            completed_payload["guard"] = final_state.get("guard_decision", {})

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
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def HTTPException_400() -> Any:
    from fastapi import HTTPException
    raise HTTPException(status_code=400, detail="`message` must be a non-empty string")


__all__ = ["router"]
