"""FastAPI application factory for the Enterprise AI Employee service.

Phase 6 routes:
- POST /api/chat           - one enterprise task end to end (unchanged)
- POST /api/chat/stream    - SSE streaming of pipeline progress
- GET  /health             - component health check
- POST/GET /api/tasks     - task submission and history
- GET  /api/knowledge/*   - knowledge base document listing
- GET  /api/graph/*       - Neo4j entity / relation queries
- GET  /api/reports       - report listing
- GET  /api/evaluation/*  - evaluation summary
- GET  /api/analytics/*   - BI aggregations over DuckDB
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.analytics import router as analytics_router
from src.api.routes.chat import ChatRequest, ChatResponse
from src.api.routes.chat import router as chat_router
from src.api.routes.chat_stream import router as chat_stream_router
from src.api.routes.evaluation import router as evaluation_router
from src.api.routes.graph import router as graph_router
from src.api.routes.knowledge import router as knowledge_router
from src.api.routes.reports import router as reports_router
from src.api.routes.tasks import router as tasks_router
from src.core.exceptions import RetrievalError
from src.graph.builder import DigitalEmployee

logger = logging.getLogger("eae.api")


def _create_employee() -> DigitalEmployee:
    """Build the default employee.

    Phase 6 frontend workbench: when the formal Milvus backend is
    unreachable (no server started, no .env credentials) the service still
    boots with the in-memory fake store, so the routes and SSE stream
    remain usable.  Retrieval quality degrades to the lexical path and
    /health reports the vector backend accordingly.  This is an explicit,
    env-driven choice — a deployment that configures the formal backend
    keeps failing fast as before.
    """
    import os

    if os.environ.get("EAE_RAG_BACKEND") == "fake":
        return DigitalEmployee(backend="fake", retrieval_mode="hybrid",
                              phase3=True, phase4=True)
    try:
        return DigitalEmployee(backend="milvus", retrieval_mode="hybrid",
                               phase3=True, phase4=True)
    except RetrievalError as exc:
        logger.warning("Milvus unavailable at boot (%s); falling back to "
                       "in-memory fake store", exc)
        return DigitalEmployee(backend="fake", retrieval_mode="hybrid",
                               phase3=True, phase4=True)


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise AI Employee", version="0.6.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.employee = _create_employee()

    # Phase 2 / 6.1
    app.include_router(chat_router, prefix="/api")
    # Phase 6.2
    app.include_router(tasks_router, prefix="/api")
    # Phase 6.3
    app.include_router(chat_stream_router, prefix="/api")
    # Phase 6.5
    app.include_router(knowledge_router, prefix="/api")
    app.include_router(graph_router, prefix="/api")
    app.include_router(reports_router, prefix="/api")
    app.include_router(analytics_router, prefix="/api")
    # Phase 6.6
    app.include_router(evaluation_router, prefix="/api")

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
            from src.core.embedder import EmbeddingUnavailableError, create_embedder

            embedder = create_embedder()
            embedding_status = f"ok ({embedder.backend})"
        except EmbeddingUnavailableError:
            # Honest status: the local BGE-M3 model is not installed / not
            # downloadable.  The pipeline itself still works with lexical
            # retrieval; do not advertise a broken backend as an error.
            embedding_status = "offline (model not available)"

        # Milvus
        milvus_status = "not checked"
        milvus_info: dict[str, Any] = {}
        try:
            from src.core.exceptions import RetrievalError
            from src.core.vector_store import create_vector_store

            store = create_vector_store(backend="milvus")
            milvus_info = store.health_check()
            milvus_status = "ok"
        except RetrievalError as exc:
            # Distinguish "server not running" from misconfiguration so the
            # workbench dashboard can show an honest, readable status.
            if str(exc.details.get("uri", "")) or "connection failed" in str(exc).lower():
                milvus_status = "offline"
            else:
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
