"""Node: RAG retrieval - hybrid retrieval over the local knowledge base (Phase 1).

Uses the RAG tool (``src/tools/rag_tool.py``): dense (in-process cosine index)
+ BM25 + RRF fusion.  Results are shaped into ``Evidence`` items so the
answer node treats SQL and RAG evidence uniformly (unified evidence model,
docs/ARCHITECTURE.md #7).
"""

from __future__ import annotations

from src.core.observability import observe
from src.core.state import AgentState, Evidence
from src.tools.base import ToolRegistry, ToolResult
from src.tools.rag_tool import RAGTool


class RAGRetrievalNode:
    def __init__(self, registry: ToolRegistry, rag_tool: RAGTool | None = None,
                 top_k: int = 5) -> None:
        self._registry = registry
        self._rag_tool = rag_tool or RAGTool()
        self._top_k = top_k

    @observe("rag_node")
    def run(self, state: AgentState) -> AgentState:
        question = state.get("user_task", "")
        result: ToolResult = self._registry.call(
            "rag", query=question, top_k=self._top_k
        )

        tool_calls = list(state.get("tool_calls") or [])
        tool_calls.append(result.to_call_record())

        evidence: list[Evidence] = []
        errors = list(state.get("errors") or [])
        if result.ok:
            for item in result.data or []:
                evidence.append(
                    Evidence(
                        evidence_id=f"ev-{state.get('task_id', 'task')}-rag-{item['chunk_id']}",
                        source_type="milvus",
                        source_ref=f"milvus:{item['chunk_id']}",
                        content=item["text"],
                        payload=item,
                        score=item.get("score"),
                        metadata=item.get("metadata", {}),
                    )
                )
        else:
            errors.append({"stage": "rag_retrieval", "message": result.error or "rag tool failed"})

        return {"evidence": evidence, "tool_calls": tool_calls, "errors": errors}


__all__ = ["RAGRetrievalNode"]
