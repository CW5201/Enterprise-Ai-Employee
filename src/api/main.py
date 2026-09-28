"""FastAPI entry point - app factory and router mounting (Phase 1 minimum).

Exposes a single POST /chat endpoint that runs the LangGraph pipeline and
returns the final AgentState as JSON.  SSE streaming, auth and the other
routes are added in Phase 6/7 together with the Vue3 workbench.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from src.graph.builder import DigitalEmployee

app = FastAPI(title="Enterprise AI Employee", version="0.1.0")
_employee = DigitalEmployee()


class ChatRequest(BaseModel):
    task: str
    task_id: str | None = None


class ChatResponse(BaseModel):
    task_id: str
    status: str
    intent: str
    selected_tools: list[str]
    answer: str
    citations: list[str]
    evidence: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    latency_ms: dict[str, float]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    state = _employee.run(request.task, task_id=request.task_id)
    return ChatResponse(
        task_id=str(state.get("task_id", "")),
        status=str(state.get("status", "")),
        intent=str(state.get("intent", "")),
        selected_tools=list(state.get("selected_tools") or []),
        answer=str(state.get("answer", "")),
        citations=list(state.get("citations") or []),
        evidence=[ev.model_dump() for ev in (state.get("evidence") or [])],
        errors=list(state.get("errors") or []),
        latency_ms=dict(state.get("latency_ms") or {}),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


__all__ = ["app"]
