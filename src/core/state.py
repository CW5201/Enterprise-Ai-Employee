"""AgentState — the single shared state of the LangGraph task pipeline.

Design rules (docs/ARCHITECTURE.md):
- Every node reads/writes ONLY this state; no global variables carry task
  state between nodes.
- All fields have explicit types; optional fields default to empty and are
  always nullable-safe.
- List channels use an append reducer: a node returns the items to ADD
  (or an empty list to leave the channel untouched).

Node I/O contract (field -> producer/consumer):

    intent_understanding      : intent, confidence, entities, plan, errors
    supervisor_router         : route, plan, errors
    sql_execution             : sql, sql_result, tool_calls, evidence, errors
    rag_retrieval            : retrieved_context, tool_calls, evidence, errors
    clarification             : answer, errors
    answer_generation        : answer, evidence, sql, tool_calls, latency
"""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Structured value types
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    """One tool invocation, as recorded in ``tool_calls``."""

    tool: str
    ok: bool
    input: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    evidence_ref: str | None = None
    duration_ms: float = 0.0
    audit_id: str = ""


class SQLResultPayload(BaseModel):
    """Structured SQL execution result (see src/core/sql_executor.py)."""

    sql: str
    columns: list[str] = Field(default_factory=list)
    rows: list[Any] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0
    truncated: bool = False


class RetrievedChunk(BaseModel):
    """One RAG hit, source preserved (Phase 4 claim-evidence dependency)."""

    chunk_id: str
    text: str
    source: str  # doc_id / title / license — never dropped
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class ErrorRecord(BaseModel):
    """One recorded error (never crashes the pipeline; answer stays honest)."""

    stage: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    """Unified evidence item shared by SQL and RAG results.

    ``evidence_id`` is stable across reruns so that Phase 4
    claim-evidence verification and Phase 5 evaluation can reference it.
    """

    evidence_id: str
    source_type: str = Field(description="duckdb | milvus (Phase 1)")
    source_ref: str = Field(description="executed SQL, or milvus:chunk_id")
    content: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Short alias used in tests and docs.
Evidence = EvidenceItem

# ---------------------------------------------------------------------------
# LangGraph channel reducers
# ---------------------------------------------------------------------------


def _append(current: list[Any], update: list[Any]) -> list[Any]:
    """Append-style reducer: nodes return items to ADD, not the full list."""
    if not update:
        return current
    return list(current or []) + list(update)


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """LangGraph state shared by every node of the task pipeline.

    ``total=False`` because nodes add keys incrementally; :func:`make_state`
    initialises every field up front so nothing is ever missing at runtime.
    """

    # --- task -------------------------------------------------------------
    request_id: str
    user_query: str
    conversation: Annotated[list[dict[str, str]], _append]

    # --- intent understanding ----------------------------------------------
    intent: str  # data_query | knowledge_query | complex_analysis | clarification
    confidence: float
    entities: Annotated[list[dict[str, Any]], _append]
    plan: str  # short human-readable execution plan (router-owned)

    # --- routing -------------------------------------------------------------
    route: str  # sql | rag | sql_rag | clarification
    route_confidence: float

    # --- execution -------------------------------------------------------------
    tool_calls: Annotated[list[ToolCallRecord], _append]
    errors: Annotated[list[ErrorRecord], _append]
    sql: str
    sql_result: SQLResultPayload | None

    # --- retrieval -------------------------------------------------------------
    retrieved_context: Annotated[list[RetrievedChunk], _append]

    # --- evidence -------------------------------------------------------------
    evidence: Annotated[list[EvidenceItem], _append]

    # --- output -------------------------------------------------------------
    answer: str
    latency: dict[str, float]  # stage name -> ms (observability-owned)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(user_query: str, *, request_id: str | None = None) -> AgentState:
    """Create a fresh, fully-initialised state (graph entry and tests)."""
    return {
        "request_id": request_id or f"req-{int(time.time() * 1000)}",
        "user_query": user_query,
        "conversation": [],
        "intent": "",
        "confidence": 0.0,
        "entities": [],
        "plan": "",
        "route": "",
        "route_confidence": 0.0,
        "tool_calls": [],
        "errors": [],
        "sql": "",
        "sql_result": None,
        "retrieved_context": [],
        "evidence": [],
        "answer": "",
        "latency": {},
    }


__all__ = [
    "AgentState",
    "ErrorRecord",
    "EvidenceItem",
    "RetrievedChunk",
    "SQLResultPayload",
    "ToolCallRecord",
    "make_state",
]
