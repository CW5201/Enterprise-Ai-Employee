"""Unit tests for the RAG tool (Phase 2.2, fake backend only).

Exercises RAGTool against the in-memory fake vector store: result format,
parameter validation, empty results, source preservation, backend labeling
and the three retrieval modes (dense / bm25 / hybrid).  No live Milvus and
no real model are involved.

Phase 2.2 contract: an empty **dense** store no longer means "no results" —
the hybrid retriever also consults its BM25 index.  A truly empty retrieval
corpus requires BOTH the dense store and the BM25 index to be empty, which
is what ``test_truly_empty_corpus`` verifies.
"""

from __future__ import annotations

import os

import pytest

os.environ["EMBEDDING_FORCE_OFFLINE"] = "1"

from src.core.bm25_store import BM25Index  # noqa: E402
from src.core.embedder import HashEmbedder  # noqa: E402
from src.core.hybrid_retriever import HybridRetriever  # noqa: E402
from src.core.vector_store import FakeVectorStore  # noqa: E402
from src.tools.rag_tool import RAGTool, RetrievalResult  # noqa: E402


def _make_retriever() -> HybridRetriever:
    embed = HashEmbedder()
    store = FakeVectorStore("kb_rag_test", dim=1024, embed_fn=embed)
    records = [
        {
            "chunk_id": "kb-rag-1",
            "document_id": "finance-0001",
            "title": "员工差旅与费用报销管理制度",
            "source": "finance/finance-0001-travel-reimbursement-policy",
            "category": "finance",
            "text": "报销流程：员工提交报销单 -> 直属主管审批 -> 财务审核 -> 付款。",
            "metadata": {"doc_id": "finance-0001"},
            "vector": embed("报销流程：员工提交报销单 -> 直属主管审批 -> 财务审核 -> 付款。"),
        },
        {
            "chunk_id": "kb-rag-2",
            "document_id": "sec-0001",
            "title": "信息安全与数据保护制度",
            "source": "security/sec-0001-information-security-policy",
            "category": "security",
            "text": "一人一号，禁止共用账号；权限按最小必要原则授予。",
            "metadata": {"doc_id": "sec-0001"},
            "vector": embed("一人一号，禁止共用账号；权限按最小必要原则授予。"),
        },
    ]
    store.insert(records)
    bm25 = BM25Index()
    bm25.build(
        [
            {"chunk_id": r["chunk_id"], "text": r["text"], "source": r["source"],
             "title": r["title"], "category": r["category"], "metadata": r["metadata"]}
            for r in records
        ]
    )
    return HybridRetriever(vector_store=store, bm25=bm25, backend="fake")


@pytest.mark.unit
def test_run_returns_expected_result_shape() -> None:
    retriever = _make_retriever()
    tool = RAGTool(retriever=retriever)
    result = tool.run(query="差旅报销流程", top_k=2)
    assert result["ok"] is True
    assert result["tool"] == "rag"
    assert result["backend"] == "fake"
    assert isinstance(result["results"], list)


@pytest.mark.unit
def test_result_items_carry_all_required_fields() -> None:
    tool = RAGTool(retriever=_make_retriever())
    result = tool.run(query="差旅报销", top_k=2, retrieval_mode="hybrid")
    assert result["results"], "expected hybrid hits over the 2-chunk corpus"
    for item in result["results"]:
        for key in (
            "chunk_id", "document_id", "text", "title", "source", "category",
            "metadata", "dense_score", "bm25_score", "fusion_score", "score",
        ):
            assert key in item, f"missing {key}"
        assert item["source"], "source must never be dropped"


@pytest.mark.unit
def test_retrieval_mode_dense_bm25_hybrid() -> None:
    tool = RAGTool(retriever=_make_retriever())
    for mode in ("dense", "bm25", "hybrid"):
        result = tool.run(query="报销", top_k=2, retrieval_mode=mode)
        assert result["ok"] is True, mode
        assert result["retrieval_mode"] == mode, mode
        # dense is backed by the fake store (hash vectors) so it must return hits
        if mode in ("dense", "hybrid"):
            assert result["results"], f"{mode} should return hits over a populated store"
        for item in result["results"]:
            assert item["chunk_id"].startswith("kb-")


@pytest.mark.unit
def test_retrieval_mode_defaults_to_hybrid() -> None:
    tool = RAGTool(retriever=_make_retriever())
    result = tool.run(query="报销", top_k=2)
    assert result["retrieval_mode"] == "hybrid"


@pytest.mark.unit
def test_top_k_limits_results() -> None:
    tool = RAGTool(retriever=_make_retriever())
    result = tool.run(query="制度", top_k=1, retrieval_mode="bm25")
    assert len(result["results"]) <= 1


@pytest.mark.unit
def test_empty_query_is_rejected_cleanly() -> None:
    tool = RAGTool(retriever=_make_retriever())
    result = tool.run(query="", top_k=3)
    assert result["ok"] is False
    assert result["results"] == []
    result2 = tool.run(query="   ", top_k=3)
    assert result2["ok"] is False
    assert result2["results"] == []


@pytest.mark.unit
def test_top_k_minimum_is_one() -> None:
    """top_k < 1 is clamped up to 1 rather than erroring."""
    tool = RAGTool(retriever=_make_retriever())
    result = tool.run(query="制度", top_k=0, retrieval_mode="bm25")
    assert len(result["results"]) <= 1


@pytest.mark.unit
def test_empty_dense_store_with_bm25_corpus() -> None:
    """Empty dense store but a populated BM25 index: BM25/hybrid still find rows.

    This is the Phase 2.2 contract that made the old empty-store test stale —
    an empty *dense* store no longer implies an empty *retrieval corpus*.
    """
    embed = HashEmbedder()
    empty_store = FakeVectorStore("kb_empty_dense", dim=1024, embed_fn=embed)
    bm25 = BM25Index()
    bm25.build(
        [
            {
                "chunk_id": "kb-x",
                "text": "差旅报销须先审批",
                "source": "finance/x",
                "title": "t",
                "category": "finance",
                "metadata": {"doc_id": "x"},
            }
        ]
    )
    tool = RAGTool(retriever=HybridRetriever(vector_store=empty_store, bm25=bm25, backend="fake"))
    dense = tool.run(query="差旅", top_k=3, retrieval_mode="dense")
    assert dense["results"] == [], "empty dense store must yield no dense hits"
    bm25_result = tool.run(query="差旅", top_k=3, retrieval_mode="bm25")
    assert bm25_result["results"], "BM25 over a populated corpus must find the chunk"
    hybrid = tool.run(query="差旅", top_k=3, retrieval_mode="hybrid")
    assert hybrid["results"], "hybrid must surface the BM25-only chunk"


@pytest.mark.unit
def test_truly_empty_corpus() -> None:
    """dense empty AND bm25 empty -> all three modes return no results."""
    embed = HashEmbedder()
    empty_store = FakeVectorStore("kb_empty_all", dim=1024, embed_fn=embed)
    tool = RAGTool(
        retriever=HybridRetriever(vector_store=empty_store, bm25=BM25Index(), backend="fake")
    )
    for mode in ("dense", "bm25", "hybrid"):
        result = tool.run(query="任意查询", top_k=5, retrieval_mode=mode)
        assert result["ok"] is True
        assert result["results"] == [], f"{mode} over an empty corpus must be empty"


@pytest.mark.unit
def test_retrieval_result_dataclass_shape() -> None:
    r = RetrievalResult(
        query="q",
        results=[{"chunk_id": "x", "source": "s"}],
        backend="fake",
        index_size=1,
    )
    assert r.query == "q"
    assert r.backend == "fake"
    assert r.index_size == 1


@pytest.mark.unit
def test_invalid_retrieval_mode_rejected() -> None:
    with pytest.raises(ValueError):
        RAGTool(retriever=_make_retriever(), retrieval_mode="rerank")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
