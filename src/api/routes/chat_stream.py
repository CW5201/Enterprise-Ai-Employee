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

# Map LangGraph node names to frontend event types.
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
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_payload(node_name: str, delta: dict[str, Any]) -> dict[str, Any]:
    """Pick user-safe fields from the node state delta."""
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
        p["tool"] = node_name if node_name != "execute" else "multi_tool"
        p["tool_calls"] = delta.get("tool_calls", [])
        p["sql"] = delta.get("sql", "")
        p["sql_result"] = delta.get("sql_result")
        p["evidence"] = delta.get("evidence", [])
        p["errors"] = delta.get("errors", [])
    elif node_name == "answer":
        p["answer"] = delta.get("answer", "")
        p["status"] = delta.get("status", "")
        p["latency"] = delta.get("latency", {})
    elif node_name in ("claim_extraction", "evidence_collection", "verification", "answer_guard"):
        p[node_name] = delta.get(node_name, [])
        if node_name == "verification":
            p["verification_results"] = delta.get("verification_results", [])
            p["verification_summary"] = delta.get("verification_summary", {})
        if node_name == "answer_guard":
            p["guard_decision"] = delta.get("guard_decision", {})
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

            state: dict[str, Any] = make_state(text, request_id=body.request_id)

            for update in employee._graph.stream(state, stream_mode="updates"):
                for node_name, delta in update.items():
                    event_type = _NODE_EVENT_MAP.get(node_name, "step")
                    payload = _safe_payload(node_name, delta)
                    yield _sse_event({
                        "type": event_type,
                        "timestamp": _ts(),
                        "step": node_name,
                        "status": "ok",
                        "payload": payload,
                    })

            # --- final completed event ---
            yield _sse_event({
                "type": "completed",
                "timestamp": _ts(),
                "step": "end",
                "status": "ok",
                "payload": {
                    "answer": state.get("answer", ""),
                    "status": state.get("status", ""),
                    "route": state.get("route", ""),
                    "latency": state.get("latency", {}),
                    "verification_summary": state.get("verification_summary", {}),
                },
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream pipeline failed")
            yield _sse_event({
                "type": "error",
                "timestamp": _ts(),
                "step": "error",
                "status": "error",
                "payload": {"message": str(exc)},
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
