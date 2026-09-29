"""Node: RAG retrieval — knowledge retrieval into AgentState (Phase 2.1).

Reads ``state.user_query``, calls the RAG tool (BGE-M3 + Milvus dense
retrieval), and writes the top-k chunks to:

- ``state["retrieved_context"]`` — :class:`RetrievedChunk` list, source
  always preserved;
- ``state["evidence"]`` — unified :class:`EvidenceItem` list so the answer
  node treats RAG and SQL evidence uniformly.

Phase 2.1 is dense-only.  BM25, reranker, knowledge graph and
claim-evidence verification are deliberately NOT implemented here
(Phase 2.2+).
"""

from __future__ import annotations

from typing import Any

from src.core.observability import observe
from src.core.state import AgentState, EvidenceItem, RetrievedChunk, ToolCallRecord
from src.tools.rag_tool import RAGTool


class RAGRetrievalNode:
    def __init__(self, tool: RAGTool | None = None, top_k: int = 5, backend: str | None = None) -> None:
        self._tool = tool or RAGTool(backend=backend)
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
                metadata={
                    **item.get("metadata", {}),
                    "document_id": item.get("document_id", ""),
                    "title": item.get("title", ""),
                    "category": item.get("category", ""),
                },
                score=float(item.get("score", 0.0)),
            )
            for item in hits
        ]

        backend = self._tool.store.backend
        evidence: list[EvidenceItem] = [
            EvidenceItem(
                evidence_id=f"ev-{state.get('request_id', 'req')}-rag-{c.chunk_id}",
                source_type=backend,
                source_ref=f"{backend}:{c.chunk_id}",
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
                ok=bool(result.get("ok", True)),
                input={"query": question, "top_k": self._top_k},
                evidence_ref=f"{backend}:{','.join(c.chunk_id for c in chunks)}" if chunks else "",
            )
        ]
        return {
            "retrieved_context": chunks,
            "evidence": evidence,
            "tool_calls": tool_calls,
        }


__all__ = ["RAGRetrievalNode"]
