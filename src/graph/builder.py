"""Graph builder - assemble the LangGraph StateGraph (Phase 1 minimum loop).

Wires:

    START
      -> intent_understanding
      -> supervisor_router
      -> (route by next_action)
           execute  -> sql_execution (if sql selected)
                      -> rag_retrieval (if rag selected)
                      -> answer_generation
                      -> END
           clarify  -> clarification -> END
           answer   -> answer_generation -> END

Notes:
- Phase 1 executes at most ONE tool per run (the first active tool in the
  router's selection order), which keeps the loop minimal.  Phase 3 adds the
  multi-tool orchestration node.
- The graph is constructed once and reused; each run gets a fresh
  AgentState from ``make_state``.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from src.core.config_loader import get_routing_rules, get_settings
from src.core.llm_client import LLMClient
from src.core.observability import set_task_latencies
from src.core.state import AgentState, make_state
from src.nodes.answer_generation import AnswerGenerationNode
from src.nodes.clarification import ClarificationNode
from src.nodes.intent_understanding import IntentUnderstandingNode
from src.nodes.rag_retrieval import RAGRetrievalNode
from src.nodes.sql_execution import SQLExecutionNode
from src.nodes.supervisor_router import SupervisorRouterNode
from src.tools.base import default_registry
from src.tools.rag_tool import RAGTool


def build_graph() -> Any:
    """Build the Phase 1 LangGraph StateGraph."""
    settings = get_settings()
    rules = get_routing_rules(settings)
    llm = LLMClient()
    registry = default_registry()

    intent_node = IntentUnderstandingNode(rules, llm)
    router_node = SupervisorRouterNode(rules)
    sql_node = SQLExecutionNode(registry, llm)
    rag_node = RAGRetrievalNode(registry, RAGTool())
    answer_node = AnswerGenerationNode(llm)
    clarify_node = ClarificationNode()

    graph = StateGraph(AgentState)

    def _intent(state: AgentState) -> AgentState:
        return cast(AgentState, dict(intent_node.run(state)))

    def _route(state: AgentState) -> AgentState:
        return cast(AgentState, dict(router_node.run(state)))

    def _sql(state: AgentState) -> AgentState:
        if "sql" not in (state.get("selected_tools") or []):
            return {}
        return cast(AgentState, dict(sql_node.run(state)))

    def _rag(state: AgentState) -> AgentState:
        if "rag" not in (state.get("selected_tools") or []):
            return {}
        return cast(AgentState, dict(rag_node.run(state)))

    def _answer(state: AgentState) -> AgentState:
        return cast(AgentState, dict(answer_node.run(state)))

    def _clarify(state: AgentState) -> AgentState:
        return cast(AgentState, dict(clarify_node.run(state)))

    graph.add_node("intent", _intent)
    graph.add_node("route", _route)
    graph.add_node("sql", _sql)
    graph.add_node("rag", _rag)
    graph.add_node("answer", _answer)
    graph.add_node("clarify", _clarify)

    graph.add_edge(START, "intent")
    graph.add_edge("intent", "route")

    def _route_next(state: AgentState) -> str:
        next_action = state.get("next_action", "answer")
        if next_action == "clarify":
            return "clarify"
        if "sql" in (state.get("selected_tools") or []):
            return "sql"
        if "rag" in (state.get("selected_tools") or []):
            return "rag"
        return "answer"

    graph.add_conditional_edges("route", _route_next, ["sql", "rag", "answer", "clarify"])
    graph.add_edge("sql", "rag")  # execute both selected tools in sequence (Phase 3 parallelises)
    graph.add_edge("rag", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("clarify", END)

    return graph.compile()


class DigitalEmployee:
    """High-level entry point: run one natural-language task end to end."""

    def __init__(self) -> None:
        self._graph = build_graph()

    def run(self, user_task: str, *, task_id: str | None = None) -> AgentState:
        state: AgentState = make_state(user_task, task_id=task_id)
        set_task_latencies(state.get("latency_ms") or {})
        final = self._graph.invoke(state)
        return cast(AgentState, final)


__all__ = ["DigitalEmployee", "build_graph"]
