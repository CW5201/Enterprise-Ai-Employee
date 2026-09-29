"""FastAPI application factory for the Enterprise AI Employee service.

Phase 1 exposes a single JSON endpoint, POST /api/chat:
    {"message": "..."}  ->  {request_id, answer, intent, route, sql,
                             evidence, tool_calls}
SSE streaming, auth and the report/task routes are added together with
the Vue3 workbench (Phase 6).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.routes.chat import router as chat_router
from src.graph.builder import DigitalEmployee

logger = logging.getLogger("eae.api")


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


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise AI Employee", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Phase 1: single local dev server; tightened in Phase 7
        allow_methods=["POST"],
        allow_headers=["*"],
    )
    app.state.employee = DigitalEmployee()
    app.include_router(chat_router, prefix="/api")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

__all__ = ["app", "ChatRequest", "ChatResponse", "create_app"]
