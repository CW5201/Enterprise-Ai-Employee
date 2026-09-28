"""Node: Supervisor Router - task-adaptive routing over RAG / SQL (Phase 1).

Phase 1 implements the *minimum* routing needed for the
``Intent -> Router -> RAG/SQL -> Answer`` loop:

- consults ``config/routing_rules.yaml`` (intent -> tool mapping);
- applies the confidence gate (``route.min_confidence``): below the gate the
  router sends the task to the clarification node;
- supports multi-tool lists but executes them in the declared priority order.

The full dynamic-routing research contribution (Innovation 1, with fallback,
escalation and ablation A) is implemented in Phase 3 on top of this node.
"""

from __future__ import annotations

from src.core.config_loader import RoutingRules
from src.core.observability import observe
from src.core.state import AgentState

# Phase 1 only activates the two tools that exist; others are ignored.
_ACTIVE_TOOLS = ("sql", "rag")


def _select_tools(rules: RoutingRules, intent: str, intent_confidence: float) -> tuple[list[str], float, str]:
    eligible = [t for t in rules.intent_tools.get(intent, []) if t in _ACTIVE_TOOLS]
    if intent_confidence < rules.min_confidence:
        return [], intent_confidence, "low_confidence -> clarify"
    if not eligible:
        # out-of-scope intent: nothing to do; answer node will explain scope
        return [], 1.0, "no active tool for intent"
    return eligible, min(intent_confidence + 0.1, 1.0), f"intent {intent} -> {eligible}"


class SupervisorRouterNode:
    def __init__(self, rules: RoutingRules) -> None:
        self.rules = rules

    @observe("routing")
    def run(self, state: AgentState) -> AgentState:
        intent = state.get("intent", "")
        confidence = float(state.get("intent_confidence", 0.0))
        tools, routing_conf, plan = _select_tools(self.rules, intent, confidence)

        entry = {
            "intent": intent,
            "intent_confidence": confidence,
            "selected_tools": tools,
            "routing_confidence": routing_conf,
            "plan": plan,
        }

        return {
            "selected_tools": tools,
            "routing_plan": plan,
            "routing_confidence": routing_conf,
            "routing_history": [entry],  # append reducer merges this into the list
            "next_action": "clarify" if not tools and confidence < self.rules.min_confidence
            else ("answer" if not tools else "execute"),
        }


__all__ = ["SupervisorRouterNode"]
