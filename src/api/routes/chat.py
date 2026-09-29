"""Route: POST /api/chat — run one enterprise task end to end.

The handler delegates to the LangGraph pipeline (:class:`DigitalEmployee`),
serialises the final AgentState into the Phase 1 response contract and
records stage latencies for observability.  SSE streaming joins in Phase 6
together with the workbench.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.api.main import ChatRequest, ChatResponse
from src.core.state import AgentState
from src.graph.builder import DigitalEmployee

logger = logging.getLogger("eae.api.chat")

router = APIRouter(tags=["chat"])


def _employee(request: Request) -> DigitalEmployee:
    return request.app.state.employee


@router.post("/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest) -> dict[str, Any]:
    """Run one natural-language enterprise task through the full pipeline."""
    text = body.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`message` must be a non-empty string")

    try:
        final: AgentState = _employee(request).run(text, request_id=body.request_id)
    except Exception as exc:  # noqa: BLE001 - surface a structured error, never a 500 stack
        logger.exception("pipeline failed for request %s", body.request_id)
        raise HTTPException(status_code=502, detail=f"pipeline error: {exc}") from exc

    return {
        "request_id": final.get("request_id", ""),
        "answer": final.get("answer", ""),
        "intent": final.get("intent", ""),
        "route": final.get("route", ""),
        "sql": final.get("sql", ""),
        "evidence": [ev.model_dump() for ev in (final.get("evidence") or [])],
        "tool_calls": [tc.model_dump() for tc in (final.get("tool_calls") or [])],
        "status": final.get("status", ""),
        "latency": final.get("latency", {}),
    }


__all__ = ["router"]
