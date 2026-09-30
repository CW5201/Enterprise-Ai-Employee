"""Graph builder — assemble the task pipeline StateGraph.

Wiring (docs/ROADMAP.md)::

    Phase 2.4 (phase3=False)::
        START -> intent -> route -> sql/rag -> answer -> END

    Phase 4 (phase4=True)::
        START -> intent -> route
               -> (phase3: execute / else sql+rag)
               -> answer_generation
               -> claim_extraction
               -> evidence_collection
               -> verification
               -> answer_guard
               -> END

``phase4`` is independent of ``phase3``: verification is a *post-hoc
result-checking layer* and only needs ``answer`` + the evidence that
produced it.  It never re-routes and never re-invokes tools.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from src.core.embedder import create_embedder
from src.core.llm_client import LLMClient
from src.core.observability import set_task_latencies
from src.core.state import AgentState, make_state
from src.nodes.answer_generation import AnswerGenerationNode
from src.nodes.answer_guard import AnswerGuardNode
from src.nodes.claim_extraction import ClaimExtractionNode
from src.nodes.clarification import ClarificationNode
from src.nodes.evidence_collection import EvidenceCollectionNode
from src.nodes.intent_understanding import IntentUnderstandingNode
from src.nodes.multi_tool_execution import MultiToolExecutionNode
from src.nodes.rag_retrieval import RAGRetrievalNode
from src.nodes.sql_execution import SQLExecutionNode
from src.nodes.supervisor_router import SupervisorRouterNode
from src.nodes.task_router import TaskRouterNode
from src.nodes.verification import VerificationNode
from src.tools.rag_tool import RAGTool


def build_graph(
    backend: str | None = None,
    rag_top_k: int = 5,
    store_override: Any | None = None,
    retrieval_mode: str = "hybrid",
    phase3: bool = False,
    phase4: bool = False,
    router_node: Any | None = None,
    executor_node: Any | None = None,
) -> Any:
    """Build the task pipeline.

    ``phase3=False`` (default) preserves the Phase 2.4 wiring exactly.
    ``phase3=True`` swaps in the Task-Adaptive Dynamic Router
    (:class:`TaskRouterNode`) and the multi-tool orchestrator
    (:class:`MultiToolExecutionNode`).
    ``phase4=True`` appends the Claim-Evidence Verification tail
    (claim_extraction -> evidence_collection -> verification ->
    answer_guard).  ``phase4`` may be combined with either ``phase3``
    value.
    """
    llm = LLMClient()
    if store_override is not None:
        rag_tool = RAGTool(store=store_override, retrieval_mode=retrieval_mode)
    else:
        rag_tool = RAGTool(backend=backend, retrieval_mode=retrieval_mode)

    intent_node = IntentUnderstandingNode(llm)
    sql_node = SQLExecutionNode(llm=llm)
    rag_node = RAGRetrievalNode(rag_tool, top_k=rag_top_k, backend=backend, retrieval_mode=retrieval_mode)
    answer_node = AnswerGenerationNode(llm)
    clarify_node = ClarificationNode()

    graph: StateGraph = StateGraph(AgentState)

    # -- nodes ----------------------------------------------------------------
    graph.add_node("intent", lambda s: cast(AgentState, dict(intent_node.run(s))))
    graph.add_node("answer", lambda s: cast(AgentState, dict(answer_node.run(s))))
    graph.add_node("clarify", lambda s: cast(AgentState, dict(clarify_node.run(s))))
    graph.add_edge(START, "intent")
    graph.add_edge("clarify", END)

    # Phase 4 tail (shared by both base wirings)
    if phase4:
        embedder = None
        try:
            embedder = create_embedder()
        except Exception:  # noqa: BLE001 - embedding unavailable: lexical fallback
            embedder = None
        claim_node = ClaimExtractionNode(llm=llm)
        evidence_node = EvidenceCollectionNode()
        verification_node = VerificationNode(llm=llm, embedder=embedder)
        guard_node = AnswerGuardNode()

        graph.add_node("claim_extraction", lambda s: cast(AgentState, dict(claim_node.run(s))))
        graph.add_node("evidence_collection", lambda s: cast(AgentState, dict(evidence_node.run(s))))
        graph.add_node("verification", lambda s: cast(AgentState, dict(verification_node.run(s))))
        graph.add_node("answer_guard", lambda s: cast(AgentState, dict(guard_node.run(s))))
        graph.add_edge("answer", "claim_extraction")
        graph.add_edge("claim_extraction", "evidence_collection")
        graph.add_edge("evidence_collection", "verification")
        graph.add_edge("verification", "answer_guard")
        graph.add_edge("answer_guard", END)

    if not phase3:
        router_node = SupervisorRouterNode()
        graph.add_node("route", lambda s: cast(AgentState, dict(router_node.run(s))))
        graph.add_node("sql", lambda s: cast(AgentState, dict(sql_node.run(s))))
        graph.add_node("rag", lambda s: cast(AgentState, dict(rag_node.run(s))))
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
        return graph.compile()

    # -- Phase 3: dynamic router + multi-tool orchestrator -------------------
    router_node = router_node or TaskRouterNode(llm=llm)
    executor_node = executor_node or MultiToolExecutionNode(
        sql_node=sql_node, rag_tool=rag_tool, llm=llm,
    )
    graph.add_node("route", lambda s: cast(AgentState, dict(router_node.run(s))))
    graph.add_node("execute", lambda s: cast(AgentState, dict(executor_node.run(s))))
    graph.add_edge("intent", "route")

    def _after_route(state: AgentState) -> str:
        route = str(state.get("route", "clarification"))
        if route == "clarification":
            return "clarify"
        return "execute"

    graph.add_conditional_edges("route", _after_route, ["execute", "clarify"])
    graph.add_edge("execute", "answer")
    return graph.compile()


class DigitalEmployee:
    """High-level entry point: run one natural-language task end to end.

    ``backend=None`` (default) uses the formal Milvus backend.  Pass
    ``backend="fake"`` explicitly for unit tests / test mode.
    ``phase3`` selects the Task-Adaptive Routing pipeline; ``phase4``
    appends the Claim-Evidence Verification tail.
    """

    def __init__(
        self,
        backend: str | None = None,
        rag_top_k: int = 5,
        retrieval_mode: str = "hybrid",
        phase3: bool = False,
        phase4: bool = False,
    ) -> None:
        self._graph = build_graph(
            backend=backend, rag_top_k=rag_top_k,
            retrieval_mode=retrieval_mode, phase3=phase3, phase4=phase4,
        )
        self.phase4 = phase4

    def run(self, user_query: str, *, request_id: str | None = None) -> AgentState:
        state: AgentState = make_state(user_query, request_id=request_id)
        set_task_latencies(state["latency"])
        return cast(AgentState, self._graph.invoke(state))


__all__ = ["DigitalEmployee", "build_graph"]
