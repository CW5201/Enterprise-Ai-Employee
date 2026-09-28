"""AgentState - unified task state shared by every node in the graph.

Phase 1 implements the minimum state needed for the
``Intent -> Router -> RAG/SQL -> Answer`` loop.  Later phases extend it
(verification, feedback, ...) without breaking the existing fields.

The state is deliberately split into two parts:

- :class:`AgentState` (a plain dict) - the LangGraph state schema.  Every
  key is documented; nodes read and write only these keys.
- :class:`Evidence` - the unified evidence item.  Phase 1 only fills
  ``source_type in {"duckdb", "milvus"}``; Phase 2 adds
  ``"neo4j"`` / ``"analysis"``.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


# LangGraph channel reducers ------------------------------------------------
def _reducer_list(current: list[Any], update: list[Any]) -> list[Any]:
    """Append-style reducer: node returns the items to ADD, not the full list."""
    if not update:
        return current
    return list(current or []) + list(update)


# ---------------------------------------------------------------------------
# Unified evidence model
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """One retrievable / citable evidence item.

    ``evidence_id`` must be stable across reruns so that Phase 4
    claim-evidence verification and Phase 5 evaluation can reference it.
    """

    evidence_id: str = Field(description="Stable identifier, e.g. 'ev-0001-1'")
    source_type: str = Field(description="duckdb | milvus (phase 1); neo4j | analysis later")
    source_ref: str = Field(description="SQL result set, Milvus chunk id, template id, ...")
    content: str = Field(default="", description="Human/LLM-readable text of the evidence")
    payload: dict[str, Any] = Field(default_factory=dict, description="Structured raw data")
    score: float | None = Field(default=None, description="Retrieval / relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="doc / table / time range")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """LangGraph state.  ``total=False`` because nodes add keys incrementally.

    List fields use the :func:`_reducer_list` append reducer: a node returns
    the items to *add* (or an empty list to leave the channel untouched).
    Scalar fields are plain last-write-wins.
    """

    # --- task -------------------------------------------------------------
    task_id: str
    user_task: str
    conversation: list[dict[str, str]]

    # --- intent understanding ---------------------------------------------
    intent: str
    slots: dict[str, Any]
    constraints: dict[str, Any]
    intent_confidence: float
    intent_reason: str

    # --- routing ----------------------------------------------------------
    selected_tools: list[str]
    routing_plan: str
    routing_confidence: float
    routing_history: Annotated[list[dict[str, Any]], _reducer_list]

    # --- execution ----------------------------------------------------------
    tool_calls: Annotated[list[dict[str, Any]], _reducer_list]
    tool_results: Annotated[list[dict[str, Any]], _reducer_list]
    errors: Annotated[list[dict[str, Any]], _reducer_list]

    # --- evidence -----------------------------------------------------------
    evidence: Annotated[list[Evidence], _reducer_list]

    # --- answer -------------------------------------------------------------
    answer: str
    citations: Annotated[list[str], _reducer_list]

    # --- control --------------------------------------------------------------
    status: str
    next_action: str
    iteration: int
    latency_ms: dict[str, float]

    # --- messages ------------------------------------------------------------
    messages: Annotated[list[dict[str, str]], _reducer_list]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(user_task: str, *, task_id: str | None = None) -> AgentState:
    """Create a fresh state for a new task (used by the graph and by tests)."""
    return {
        "task_id": task_id or f"task-{int(time.time() * 1000)}",
        "user_task": user_task,
        "conversation": [],
        "selected_tools": [],
        "tool_calls": [],
        "tool_results": [],
        "errors": [],
        "evidence": [],
        "routing_history": [],
        "citations": [],
        "slots": {},
        "constraints": {},
        "status": "pending",
        "iteration": 0,
        "latency_ms": {},
        "messages": [],
    }


__all__ = ["AgentState", "Evidence", "make_state"]
