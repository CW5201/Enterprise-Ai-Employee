"""FastAPI application factory for the Enterprise AI Employee service.

Phase 2.1:
- POST /api/chat — one enterprise task end to end (unchanged from Phase 1).
- GET  /health   — component health for LLM / embedding / milvus / database.

SSE streaming, auth and the report/task routes are added together with
the Vue3 workbench (Phase 6).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.chat import ChatRequest, ChatResponse
from src.api.routes.chat import router as chat_router
from src.graph.builder import DigitalEmployee

logger = logging.getLogger("eae.api")


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise AI Employee", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Phase 1: single local dev server; tightened in Phase 7
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )
    app.state.employee = DigitalEmployee()
    app.include_router(chat_router, prefix="/api")

    @app.get("/health")
    def health() -> dict[str, Any]:
        from src.core.config_loader import get_settings

        settings = get_settings()

        # LLM
        try:
            from src.core.llm_client import LLMClient

            llm = LLMClient()
            llm_status = "ok" if not llm.offline else "offline (no endpoint configured)"
        except Exception as exc:  # noqa: BLE001
            llm_status = f"error: {exc}"

        # Embedding
        try:
            from src.core.embedder import create_embedder

            embedder = create_embedder()
            embedding_status = f"ok ({embedder.backend})"
        except Exception as exc:  # noqa: BLE001
            embedding_status = f"error: {exc}"

        # Milvus
        milvus_status = "not checked"
        milvus_info: dict[str, Any] = {}
        try:
            from src.core.vector_store import create_vector_store

            store = create_vector_store(backend="milvus")
            milvus_info = store.health_check()
            milvus_status = "ok"
        except Exception as exc:  # noqa: BLE001
            milvus_status = f"error: {exc}"

        # Database (DuckDB)
        db_status = "ok"
        try:
            import duckdb

            db_path = settings.db_path
            if db_path.exists():
                conn = duckdb.connect(str(db_path), read_only=True)
                conn.close()
            else:
                db_status = "not found"
        except Exception as exc:  # noqa: BLE001
            db_status = f"error: {exc}"

        return {
            "status": "ok" if "error" not in milvus_status else "degraded",
            "llm": llm_status,
            "embedding": embedding_status,
            "milvus": milvus_status,
            "milvus_details": milvus_info,
            "database": db_status,
        }

    return app


app = create_app()

__all__ = ["app", "ChatRequest", "ChatResponse", "create_app"]
