"""Integration test: the RAG end-to-end path (Phase 1).

Exercises: knowledge-base document loading, chunking, dense retrieval,
evidence shaping and the answer node.

Runs fully offline: the deterministic offline LLM backend plus the
hash-fallback embedder (EMBEDDING_FORCE_OFFLINE) keep the suite fast and
independent of a live model endpoint or GPU.
"""

from __future__ import annotations

import os

import pytest

os.environ["LLM_FORCE_OFFLINE"] = "1"
os.environ.pop("LLM_BASE_URL", None)
os.environ["EMBEDDING_FORCE_OFFLINE"] = "1"

from src.graph.builder import build_graph  # noqa: E402
from src.tools.rag_tool import RAGTool  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture()
def rag_tool() -> RAGTool:
    tool = RAGTool()
    count = tool.build_index()
    assert count > 0, (
        "knowledge base corpus is empty - check data/knowledge_base/*.md "
        "and the README in each domain folder for the expected format"
    )
    return tool


def test_hybrid_retrieval_returns_policy_chunks(rag_tool: RAGTool) -> None:
    result = rag_tool.run(query="差旅费报销标准是多少", top_k=3)
    assert result["ok"], result.get("error")
    assert result["results"], "expected retrieved chunks"
    for item in result["results"]:
        assert item["chunk_id"]
        assert item["text"]
        assert item["score"] >= 0
        assert item.get("metadata")


def test_rag_end_to_end_offline_llm() -> None:
    """Full pipeline against the knowledge base (offline LLM backend)."""
    graph = build_graph()
    state = graph.invoke(
        {
            "request_id": "req-rag-e2e",
            "user_query": "出差住宿标准和报销流程有哪些规定？",
            "conversation": [],
        }
    )
    assert state["intent"] == "knowledge_query", state["intent"]
    assert state["route"] == "rag", state["route"]
    rag_evidence = [e for e in state["evidence"] if e.source_type in ("milvus", "local_index")]
    assert rag_evidence, "expected knowledge evidence from the finance policy doc"
    assert state.get("answer"), "answer should not be empty"


def test_local_vector_store_metadata_filtering(rag_tool: RAGTool) -> None:
    """The local store's filter= argument filters on top-level record fields."""
    store = rag_tool.store
    hits = store.search("报销", top_k=5, filter={"department": "finance"})
    for hit in hits:
        assert hit.metadata.get("department") == "finance"
