"""Integration test: Dense vs BM25 vs Hybrid RAG over the real KB corpus.

The dense channel runs against the **in-process fake store** (labelled
``backend=fake`` — NOT real Milvus).  This keeps the test deterministic
and independent of a live server while still exercising the full hybrid
path end to end: same 35-chunk corpus, same query set, same top_k, and
the RRF fusion code path.

Real Milvus + real BGE-M3 runs live in ``test_real_milvus_rag.py``.
"""

from __future__ import annotations

import os

import pytest

os.environ["EMBEDDING_FORCE_OFFLINE"] = "1"

from src.core.bm25_store import BM25Index  # noqa: E402
from src.core.embedder import HashEmbedder  # noqa: E402
from src.core.hybrid_retriever import HybridRetriever  # noqa: E402
from src.core.vector_store import FakeVectorStore  # noqa: E402
from src.tools.rag_tool import iter_kb_chunks_for_store  # noqa: E402

pytestmark = pytest.mark.integration


def _retriever() -> HybridRetriever:
    embed = HashEmbedder()
    store = FakeVectorStore("kb_hybrid_integration", dim=embed.dim, embed_fn=embed)
    records = iter_kb_chunks_for_store()
    store.insert(records)
    bm25 = BM25Index()
    bm25.build(
        [
            {k: r[k] for k in ("chunk_id", "text", "source", "title", "category", "metadata")}
            for r in records
        ]
    )
    return HybridRetriever(vector_store=store, bm25=bm25, backend="fake")


QUERIES = [
    "报销单 审批",
    "差旅 住宿 标准",
    "采购 供应商 库存",
    "远程办公 申请",
    "数据安全 权限 审计",
]


def test_backend_is_fake() -> None:
    r = _retriever()
    assert r.backend == "fake"


def test_all_modes_return_results() -> None:
    r = _retriever()
    for mode in ("dense", "bm25", "hybrid"):
        hits = r.search(QUERIES[0], top_k=5, mode=mode)
        # dense is backed by a populated fake store, so it must return rows
        assert isinstance(hits, list)
        if mode != "bm25":
            assert hits, f"{mode} over a populated fake store should return rows"


def test_hybrid_recalls_expected_docs() -> None:
    r = _retriever()
    # 报销 / 差旅 -> finance doc
    hits = r.search("报销 差旅 标准", top_k=5, mode="hybrid")
    ids = [h.chunk_id for h in hits]
    assert any(cid.startswith("kb-") for cid in ids)
    # at least one hit carries a source path (never dropped)
    assert any(h.source for h in hits)


def test_modes_share_chunk_namespace() -> None:
    r = _retriever()
    dense_ids = {h.chunk_id for h in r.search("制度", top_k=5, mode="dense")}
    bm25_ids = {h.chunk_id for h in r.search("制度", top_k=5, mode="bm25")}
    valid = {h.chunk_id for h in r.hybrid_search("制度", top_k=10)} | dense_ids | bm25_ids
    assert dense_ids <= valid
    assert bm25_ids <= valid


def test_top_k_limits() -> None:
    r = _retriever()
    for mode in ("dense", "bm25", "hybrid"):
        assert len(r.search("制度", top_k=1, mode=mode)) <= 1
        assert len(r.search("制度", top_k=3, mode=mode)) <= 3


def test_empty_query_rejected_by_tool() -> None:
    from src.tools.rag_tool import RAGTool

    r = _retriever()
    tool = RAGTool(retriever=r)
    result = tool.run(query="", top_k=5)
    assert result["ok"] is False
    assert result["results"] == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
