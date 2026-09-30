"""Route: POST /api/chat — run one enterprise task end to end.

The handler delegates to the LangGraph pipeline (:class:`DigitalEmployee`),
serialises the final AgentState into the response contract and records
stage latencies for observability.  SSE streaming joins in Phase 6
together with the workbench.

The Pydantic request/response models live here (not in ``main.py``) to
avoid the circular import between the app factory and its routes.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.core.state import AgentState
from src.graph.builder import DigitalEmployee

logger = logging.getLogger("eae.api.chat")


class ChatRequest(BaseModel):
    """POST /api/chat request body."""

    message: str
    request_id: str | None = None


class ChatResponse(BaseModel):
    """POST /api/chat response body."""

    request_id: str
    answer: str
    intent: str
    route: str
    sql: str
    evidence: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    status: str
    latency: dict[str, float]
    # Phase 4 Claim-Evidence Verification
    sources: list[dict[str, Any]] = []
    verification: list[dict[str, Any]] = []
    verification_summary: dict[str, Any] = {}
    guard: dict[str, Any] = {}


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
        # Phase 4: user-safe citations + verification verdicts.  The guard
        # has already stripped internal credentials / stack traces; only
        # doc titles, chunk refs, table names and KG templates are exposed.
        "sources": final.get("answer_sources") or [],
        "verification": final.get("verification_results") or [],
        "verification_summary": final.get("verification_summary") or {},
        "guard": final.get("guard_decision") or {},
    }


__all__ = ["router", "ChatRequest", "ChatResponse"]
