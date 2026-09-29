"""Graph builder — assemble the Phase 1 minimum StateGraph.

Wiring (docs/ROADMAP.md Phase 1)::

    START
      -> intent_understanding
      -> supervisor_router
      -> (conditional on route)
           sql    -> sql_execution -> answer_generation -> END
           rag    -> rag_retrieval -> answer_generation -> END
           mixed  -> sql_execution -> rag_retrieval -> answer -> END
           clarify-> clarification -> END

Design notes:
- One graph, one shared AgentState, no multi-agent.
- The router emits ``route in {sql, rag, mixed, clarify}``.  To keep the
  StateGraph conditional edges simple, the **sql node** executes the SQL and
  the **rag node** executes retrieval; each node internally no-ops when the
  route does not call for it.  The conditional edge after the router picks
  the entry point (sql, rag, or sql which forwards into rag for ``mixed``),
  and a second conditional edge after the sql node decides whether to run
  rag or go straight to the answer.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from src.core.llm_client import LLMClient
from src.core.state import AgentState, make_state
from src.nodes.answer_generation import AnswerGenerationNode
from src.nodes.clarification import ClarificationNode
from src.nodes.intent_understanding import IntentUnderstandingNode
from src.nodes.rag_retrieval import RAGRetrievalNode
from src.nodes.sql_execution import SQLExecutionNode
from src.nodes.supervisor_router import SupervisorRouterNode
from src.tools.rag_tool import RAGTool


def build_graph() -> Any:
    llm = LLMClient()
    rag_tool = RAGTool()

    intent_node = IntentUnderstandingNode(llm)
    router_node = SupervisorRouterNode()
    sql_node = SQLExecutionNode(llm=llm)
    rag_node = RAGRetrievalNode(rag_tool)
    answer_node = AnswerGenerationNode(llm)
    clarify_node = ClarificationNode()

    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("intent", lambda s: cast(AgentState, dict(intent_node.run(s))))
    graph.add_node("route", lambda s: cast(AgentState, dict(router_node.run(s))))
    graph.add_node("sql", lambda s: cast(AgentState, dict(sql_node.run(s))))
    graph.add_node("rag", lambda s: cast(AgentState, dict(rag_node.run(s))))
    graph.add_node("answer", lambda s: cast(AgentState, dict(answer_node.run(s))))
    graph.add_node("clarify", lambda s: cast(AgentState, dict(clarify_node.run(s))))

    graph.add_edge(START, "intent")
    graph.add_edge("intent", "route")

    def _after_route(state: AgentState) -> str:
        route = str(state.get("route", "clarification"))
        if route == "rag":
            return "rag"
        if route in ("sql", "sql_rag"):
            return "sql"
        return "clarify"

    def _after_sql(state: AgentState) -> str:
        # sql_rag (complex_analysis) runs SQL THEN RAG; pure SQL stops.
        return "rag" if str(state.get("route", "")) == "sql_rag" else "answer"

    graph.add_conditional_edges("route", _after_route, ["sql", "rag", "clarify"])
    graph.add_conditional_edges("sql", _after_sql, ["rag", "answer"])
    graph.add_edge("rag", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("clarify", END)

    return graph.compile()


class DigitalEmployee:
    """High-level entry point: run one natural-language task end to end."""

    def __init__(self) -> None:
        self._graph = build_graph()

    def run(self, user_query: str, *, request_id: str | None = None) -> AgentState:
        state: AgentState = make_state(user_query, request_id=request_id)
        return cast(AgentState, self._graph.invoke(state))


__all__ = ["DigitalEmployee", "build_graph"]
