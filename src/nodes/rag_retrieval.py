"""Node: RAG retrieval — knowledge retrieval into AgentState.

Consumes the RAG tool's top-k hits and writes them to
``state.retrieved_context`` (source preserved) and folds them into the
unified ``state.evidence`` list so the answer node treats RAG and SQL
evidence uniformly (unified evidence model, docs/ARCHITECTURE.md #7).

Phase 1 = dense retrieval only (Milvus / local vector store).  BM25,
reranker and metadata filtering are deliberately deferred to Phase 2 while
the chunk interface already carries ``metadata`` so they can slot in later.
"""

from __future__ import annotations

from typing import Any

from src.core.observability import observe
from src.core.state import AgentState, EvidenceItem, RetrievedChunk, ToolCallRecord
from src.tools.rag_tool import RAGTool


class RAGRetrievalNode:
    def __init__(self, tool: RAGTool | None = None, top_k: int = 5) -> None:
        self._tool = tool or RAGTool()
        self._top_k = top_k

    @observe("rag_retrieval")
    def run(self, state: AgentState) -> AgentState:
        question = state.get("user_query", "")
        result: dict[str, Any] = self._tool.run(query=question, top_k=self._top_k)
        hits: list[dict[str, Any]] = result.get("results", [])

        chunks: list[RetrievedChunk] = [
            RetrievedChunk(
                chunk_id=str(item["chunk_id"]),
                text=str(item["text"]),
                source=str(item.get("source", "")),
                metadata=item.get("metadata", {}),
                score=float(item.get("score", 0.0)),
            )
            for item in hits
        ]

        evidence: list[EvidenceItem] = [
            EvidenceItem(
                evidence_id=f"ev-{state.get('request_id', 'req')}-rag-{c.chunk_id}",
                source_type="milvus" if self._tool.store.backend == "milvus" else "local_index",
                source_ref=f"milvus:{c.chunk_id}" if self._tool.store.backend == "milvus" else f"local:{c.chunk_id}",
                content=c.text,
                payload=c.model_dump(),
                score=c.score,
                metadata={**c.metadata, "source": c.source},
            )
            for c in chunks
        ]

        tool_calls = [
            ToolCallRecord(
                tool=self._tool.name,
                ok=bool(result.get("ok")),
                input={"query": question, "top_k": self._top_k},
                evidence_ref=f"{self._tool.store.backend}:{','.join(c.chunk_id for c in chunks)}" if chunks else "",
                duration_ms=float(result.get("execution_time_ms", 0.0)),
            )
        ]
        return {
            "retrieved_context": chunks,
            "evidence": evidence,
            "tool_calls": tool_calls,
        }


__all__ = ["RAGRetrievalNode"]
