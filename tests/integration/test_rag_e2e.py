"""Integration test: RAG end-to-end path on the FAKE backend (Phase 1 offline).

Exercises: chunking, dense retrieval with the fake (hash) embedder,
evidence shaping and the answer node — fully offline, no model, no Milvus.

Phase 2.1 contract: this test explicitly uses ``backend="fake"`` (the
test-mode in-memory store + hash embedder).  Real-model tests live in
``test_real_milvus_rag.py``.
"""

from __future__ import annotations

import os

import pytest

os.environ["LLM_FORCE_OFFLINE"] = "1"
os.environ.pop("LLM_BASE_URL", None)
os.environ["EMBEDDING_FORCE_OFFLINE"] = "1"

from src.core.embedder import HashEmbedder  # noqa: E402
from src.core.vector_store import FakeVectorStore  # noqa: E402
from src.graph.builder import build_graph  # noqa: E402
from src.tools.rag_tool import (  # noqa: E402
    RAGTool,
    iter_kb_chunks_for_store,
)

pytestmark = pytest.mark.integration


def _fake_store() -> FakeVectorStore:
    embed = HashEmbedder()
    store = FakeVectorStore("kb_e2e_fake", dim=embed.dim, embed_fn=embed)
    records = iter_kb_chunks_for_store()
    store.insert(records)
    return store


@pytest.fixture()
def rag_tool() -> RAGTool:
    store = _fake_store()
    assert store.size() > 0, (
        "knowledge base corpus is empty - check data/knowledge_base/*.md "
        "and the README in each domain folder for the expected format"
    )
    return RAGTool(store=store)


def test_fake_retrieval_returns_policy_chunks(rag_tool: RAGTool) -> None:
    result = rag_tool.run(query="差旅费报销标准是多少", top_k=3)
    assert result["ok"], result.get("error")
    assert result["results"], "expected retrieved chunks"
    assert result["backend"] == "fake"
    for item in result["results"]:
        assert item["chunk_id"]
        assert item["text"]
        assert item.get("source")


def test_rag_end_to_end_offline_llm() -> None:
    """Full pipeline against the knowledge base (offline LLM + fake embedder).

    A fresh graph creates an empty fake store, so we populate it first via
    the test helper and hand it to the graph so retrieval actually works.
    """
    store = _fake_store()
    assert store.size() > 0
    graph = build_graph(backend="fake", store_override=store)
    state = graph.invoke(
        {
            "request_id": "req-rag-e2e",
            "user_query": "出差住宿标准和报销流程有哪些规定？",
            "conversation": [],
        }
    )
    assert state["intent"] == "knowledge_query", state["intent"]
    assert state["route"] == "rag", state["route"]
    rag_evidence = [e for e in state["evidence"] if e.source_type in ("milvus", "fake")]
    assert rag_evidence, "expected knowledge evidence from the finance policy doc"
    assert state.get("answer"), "answer should not be empty"


def test_fake_vector_store_metadata_filtering(rag_tool: RAGTool) -> None:
    """The fake store's filter= argument filters on record fields."""
    store = rag_tool.store
    assert isinstance(store, FakeVectorStore)
    hits = store.search("报销", top_k=5, filter={"category": "finance"})
    for hit in hits:
        assert hit.category == "finance"
