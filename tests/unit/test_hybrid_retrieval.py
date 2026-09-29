"""Unit tests for the HybridRetriever (Dense + BM25 + RRF, Phase 2.2).

Everything runs against the in-memory fake vector store + an in-process
BM25 index — no live Milvus, no real model.  This verifies the unified
interface: mode dispatch, chunk_id alignment, context preservation,
Top-K behaviour and duplicate-chunk fusion.
"""

from __future__ import annotations

import os

import pytest

os.environ["EMBEDDING_FORCE_OFFLINE"] = "1"

from src.core.bm25_store import BM25Index  # noqa: E402
from src.core.embedder import HashEmbedder  # noqa: E402
from src.core.exceptions import RetrievalError  # noqa: E402
from src.core.hybrid_retriever import HybridRetriever  # noqa: E402
from src.core.vector_store import FakeVectorStore  # noqa: E402

pytestmark = pytest.mark.unit

_RECORDS = [
    {
        "chunk_id": "kb-fin",
        "document_id": "finance-0001",
        "title": "员工差旅与费用报销管理制度",
        "source": "finance/finance-0001",
        "category": "finance",
        "text": "报销流程：员工提交报销单，直属主管审批，财务审核付款。",
        "metadata": {"doc_id": "finance-0001"},
    },
    {
        "chunk_id": "kb-sec",
        "document_id": "sec-0001",
        "title": "信息安全与数据保护制度",
        "source": "security/sec-0001",
        "category": "security",
        "text": "一人一号，禁止共用账号；权限按最小必要原则授予。",
        "metadata": {"doc_id": "sec-0001"},
    },
    {
        "chunk_id": "kb-ops",
        "document_id": "ops-0001",
        "title": "库存与供应链管理制度",
        "source": "operations/ops-0001",
        "category": "operations",
        "text": "采购申请须注明用途、数量与期望到货期。",
        "metadata": {"doc_id": "ops-0001"},
    },
]


def _retriever() -> HybridRetriever:
    embed = HashEmbedder()
    store = FakeVectorStore("kb_hybrid_test", dim=embed.dim, embed_fn=embed)
    store.insert(
        [
            {**r, "vector": embed(r["text"])} for r in _RECORDS
        ]
    )
    bm25 = BM25Index()
    bm25.build(
        [
            {k: r[k] for k in ("chunk_id", "text", "source", "title", "category", "metadata")}
            for r in _RECORDS
        ]
    )
    return HybridRetriever(vector_store=store, bm25=bm25, backend="fake")


# ---------------------------------------------------------------------------
# mode dispatch
# ---------------------------------------------------------------------------


def test_search_dense_returns_dense_only_hits() -> None:
    r = _retriever()
    hits = r.search_dense("报销", top_k=3)
    for h in hits:
        assert h.dense_rank >= 1
        assert h.bm25_rank == 0, "dense mode must not carry a BM25 rank"
        assert h.fusion_score == h.dense_score


def test_search_bm25_returns_bm25_only_hits() -> None:
    r = _retriever()
    hits = r.search_bm25("采购申请", top_k=3)
    assert hits, "expected BM25 hits"
    for h in hits:
        assert h.bm25_rank >= 1
        assert h.dense_rank == 0, "bm25 mode must not carry a dense rank"
        assert h.fusion_score == h.bm25_score
    assert hits[0].chunk_id == "kb-ops"


def test_search_hybrid_fuses_both_channels() -> None:
    r = _retriever()
    hits = r.hybrid_search("报销", top_k=3)
    assert hits
    # at least one hit participates from both channels
    assert any(h.dense_rank > 0 and h.bm25_rank > 0 for h in hits), (
        "hybrid results should include chunks found by both channels"
    )
    scores = [h.fusion_score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_dispatches_by_mode() -> None:
    r = _retriever()
    for mode in ("dense", "bm25", "hybrid"):
        hits = r.search("报销", top_k=2, mode=mode)
        if mode == "dense":
            assert all(h.bm25_rank == 0 for h in hits)
        elif mode == "bm25":
            assert all(h.dense_rank == 0 for h in hits)
        else:
            assert all(h.fusion_score > 0 for h in hits)


def test_invalid_mode_raises() -> None:
    r = _retriever()
    with pytest.raises(RetrievalError):
        r.search("报销", top_k=3, mode="rerank")
    with pytest.raises(RetrievalError):
        r.search("报销", top_k=3, mode="")


# ---------------------------------------------------------------------------
# context preservation
# ---------------------------------------------------------------------------


def test_chunk_id_aligned_across_channels() -> None:
    r = _retriever()
    dense_ids = {h.chunk_id for h in r.search_dense("报销", top_k=3)}
    bm25_ids = {h.chunk_id for h in r.search_bm25("采购", top_k=3)}
    hybrid_ids = {h.chunk_id for h in r.hybrid_search("报销", top_k=3)}
    valid = {"kb-fin", "kb-sec", "kb-ops"}
    assert dense_ids <= valid
    assert bm25_ids <= valid
    assert hybrid_ids <= valid
    # both channels draw from the same chunk namespace
    assert dense_ids | bm25_ids <= valid


def test_metadata_source_title_text_preserved() -> None:
    r = _retriever()
    hits = r.search_bm25("采购", top_k=3)
    hit = next(h for h in hits if h.chunk_id == "kb-ops")
    assert hit.source == "operations/ops-0001"
    assert hit.title == "库存与供应链管理制度"
    assert hit.category == "operations"
    assert hit.text.startswith("采购申请")
    assert hit.metadata.get("doc_id") == "ops-0001"


def test_hybrid_hit_payload_shape() -> None:
    r = _retriever()
    hit = r.hybrid_search("报销", top_k=1)[0]
    payload = hit.to_payload()
    for key in (
        "chunk_id", "dense_score", "dense_rank", "bm25_score", "bm25_rank",
        "fusion_score", "text", "source", "title", "category", "document_id", "metadata",
    ):
        assert key in payload, f"missing {key}"


# ---------------------------------------------------------------------------
# Top-K + empty + duplicates
# ---------------------------------------------------------------------------


def test_top_k_respected_in_all_modes() -> None:
    r = _retriever()
    for mode in ("dense", "bm25", "hybrid"):
        assert len(r.search("制度", top_k=1, mode=mode)) <= 1
        assert len(r.search("制度", top_k=2, mode=mode)) <= 2


def test_empty_result_is_honest() -> None:
    """BM25 (the keyword channel) returns nothing for a term in no chunk.

    The *fake* dense store always returns the requested number of rows
    (hash vectors have no notion of "no match" — every cosine score is
    still a number), so the honest empty-result guarantee lives on the
    BM25 side and on a truly empty corpus.
    """
    r = _retriever()
    assert r.search_bm25("黄河", top_k=5) == [], "BM25 must not match a foreign term"
    assert r.search("黄河", top_k=5, mode="bm25") == []


def test_empty_corpus_returns_empty() -> None:
    embed = HashEmbedder()
    store = FakeVectorStore("kb_empty", dim=embed.dim, embed_fn=embed)
    r = HybridRetriever(vector_store=store, bm25=BM25Index(), backend="fake")
    for mode in ("dense", "bm25", "hybrid"):
        assert r.search("报销", top_k=5, mode=mode) == []


def test_duplicate_chunk_fused_once() -> None:
    """The same chunk from both channels appears exactly once in hybrid
    results, with both its dense and bm25 contributions recorded."""
    r = _retriever()
    hits = r.hybrid_search("报销", top_k=5)
    ids = [h.chunk_id for h in hits]
    assert len(ids) == len(set(ids)), "duplicate chunk_id must not repeat in hybrid results"
    hit = next(h for h in hits if h.chunk_id == "kb-fin")
    assert hit.dense_rank > 0
    assert hit.bm25_rank > 0
    # RRF of two ranks in [1,5] with k=60 lands in a bounded window
    assert 1 / (60 + 5) < hit.fusion_score <= 2 / (60 + 1)


def test_backend_label_is_fake() -> None:
    r = _retriever()
    assert r.backend == "fake"
    assert r.health_check()["backend"] == "fake"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
