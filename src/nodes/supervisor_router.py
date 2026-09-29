"""Node: Supervisor Router — stable conditional routing (Phase 1).

Phase 1 intentionally has NO complex planner: one deterministic table maps
each intent to its execution path, and a confidence gate diverts weak
decisions to clarification.  The dynamic routing research contribution
(Innovation 1, ablation A) is built on top of this node in Phase 3.

    data_query        -> sql
    knowledge_query   -> rag
    complex_analysis  -> sql_rag   (SQL then RAG, results aggregated)
    clarification / * -> clarification
"""

from __future__ import annotations

from src.core.config_loader import get_settings
from src.core.observability import observe
from src.core.state import AgentState, ErrorRecord

_ROUTE_BY_INTENT: dict[str, str] = {
    "data_query": "sql",
    "knowledge_query": "rag",
    "complex_analysis": "sql_rag",
    "clarification": "clarification",
}


class SupervisorRouterNode:
    def __init__(self, min_confidence: float | None = None) -> None:
        self._min_confidence = (
            min_confidence if min_confidence is not None else float(get_settings().routing.get("route", {}).get("min_confidence", 0.6))
        )

    @observe("routing")
    def run(self, state: AgentState) -> AgentState:
        intent = str(state.get("intent") or "")
        confidence = float(state.get("confidence") or 0.0)

        if intent in _ROUTE_BY_INTENT and confidence >= self._min_confidence:
            route = _ROUTE_BY_INTENT[intent]
            plan = f"{intent} -> {route}"
            errors: list[ErrorRecord] = []
        else:
            route = "clarification"
            plan = f"{intent or 'unknown'} (confidence {confidence:.2f}) -> clarification"
            errors = [
                ErrorRecord(
                    stage="routing",
                    message="route to clarification",
                    details={"intent": intent, "confidence": confidence},
                )
            ]

        return {
            "route": route,
            "route_confidence": confidence,
            "plan": plan,
            "errors": errors,
        }


__all__ = ["SupervisorRouterNode", "_ROUTE_BY_INTENT"]
