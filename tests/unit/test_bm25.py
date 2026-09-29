"""Unit tests for the BM25 keyword index (Phase 2.2).

Covers: build / search / add / delete, empty corpus, exact keyword,
Chinese bigram tokenisation, ASCII tokens, rank numbering and chunk_id
alignment.  The index is pure in-process Python — no model, no Milvus.
"""

from __future__ import annotations

import pytest

from src.core.bm25_store import BM25Hit, BM25Index, tokenize

pytestmark = pytest.mark.unit


def _records() -> list[dict]:
    return [
        {
            "chunk_id": "kb-fin",
            "document_id": "finance-0001",
            "title": "员工差旅与费用报销管理制度",
            "source": "finance/finance-0001",
            "category": "finance",
            "text": "报销流程：员工提交报销单，直属主管审批，财务审核付款。制度编号 FIN-001。",
            "metadata": {"doc_id": "finance-0001"},
        },
        {
            "chunk_id": "kb-sec",
            "document_id": "sec-0001",
            "title": "信息安全与数据保护制度",
            "source": "security/sec-0001",
            "category": "security",
            "text": "一人一号，禁止共用账号；权限按最小必要原则授予。制度编号 SEC-001。",
            "metadata": {"doc_id": "sec-0001"},
        },
        {
            "chunk_id": "kb-ops",
            "document_id": "ops-0001",
            "title": "库存与供应链管理制度",
            "source": "operations/ops-0001",
            "category": "operations",
            "text": "采购申请须注明用途、数量与期望到货期。制度编号 OPS-001。",
            "metadata": {"doc_id": "ops-0001"},
        },
    ]


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


def test_tokenize_chinese_bigram() -> None:
    # 报销 -> single chars 报/销 plus bigram 报销
    toks = tokenize("报销")
    assert "报" in toks
    assert "销" in toks
    assert "报销" in toks


def test_tokenize_ascii_word() -> None:
    toks = tokenize("ERP approval 2026")
    assert "erp" in toks
    assert "approval" in toks
    assert "2026" in toks


def test_tokenize_mixed_keeps_both() -> None:
    toks = tokenize("订单 order")
    assert "订" in toks
    assert "订单" in toks
    assert "order" in toks


def test_tokenize_empty() -> None:
    assert tokenize("") == []


# ---------------------------------------------------------------------------
# build / size / empty
# ---------------------------------------------------------------------------


def test_build_and_size() -> None:
    idx = BM25Index()
    n = idx.build(_records())
    assert n == 3
    assert idx.size() == 3


def test_build_replaces_previous_state() -> None:
    idx = BM25Index()
    idx.build(_records())
    idx.build([_records()[0]])
    assert idx.size() == 1
    assert idx.search("报销", top_k=5)[0].chunk_id == "kb-fin"


def test_empty_corpus_search_returns_empty() -> None:
    idx = BM25Index()
    assert idx.search("报销", top_k=5) == []


def test_empty_query_returns_empty() -> None:
    idx = BM25Index()
    idx.build(_records())
    assert idx.search("", top_k=5) == []


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_exact_keyword_hits_right_chunk() -> None:
    idx = BM25Index()
    idx.build(_records())
    hits = idx.search("采购申请", top_k=3)
    assert hits, "expected a keyword hit"
    assert hits[0].chunk_id == "kb-ops"
    assert hits[0].rank == 1
    assert hits[0].score > 0.0


def test_search_isolated_keyword_discriminates_chunks() -> None:
    idx = BM25Index()
    idx.build(_records())
    fin = idx.search("报销单", top_k=1)
    sec = idx.search("账号", top_k=1)
    assert fin and fin[0].chunk_id == "kb-fin"
    assert sec and sec[0].chunk_id == "kb-sec"


def test_search_result_fields_and_rank() -> None:
    idx = BM25Index()
    idx.build(_records())
    hits = idx.search("制度", top_k=3)  # shared term -> multiple hits
    assert len(hits) >= 2
    for i, hit in enumerate(hits, start=1):
        assert isinstance(hit, BM25Hit)
        assert hit.rank == i, "rank must be the 1-based position in the list"
        assert hit.chunk_id.startswith("kb-")
        assert hit.source, "source must be preserved"
        assert hit.text, "text must be preserved"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), "BM25 results must be descending"


def test_search_top_k_limits() -> None:
    idx = BM25Index()
    idx.build(_records())
    assert len(idx.search("制度", top_k=1)) == 1
    assert len(idx.search("制度", top_k=5)) <= 3


def test_search_out_of_corpus_term_returns_empty() -> None:
    idx = BM25Index()
    idx.build(_records())
    # No chunk text contains either of these characters, so no BM25 hit is
    # possible (single CJK char query -> exactly one token, no bigram side).
    assert idx.search("河", top_k=5) == []
    assert idx.search("雪", top_k=5) == []


# ---------------------------------------------------------------------------
# add / delete
# ---------------------------------------------------------------------------


def test_add_upserts_and_grows_corpus() -> None:
    idx = BM25Index()
    idx.build(_records()[:2])
    idx.add([_records()[2]])
    assert idx.size() == 3
    assert idx.search("采购申请", top_k=1)[0].chunk_id == "kb-ops"


def test_add_replaces_duplicate_chunk() -> None:
    idx = BM25Index()
    idx.build(_records()[:1])
    idx.add([{"chunk_id": "kb-fin", "text": "差旅住宿标准", "source": "finance/x", "metadata": {}}])
    assert idx.size() == 1
    assert idx.search("住宿标准", top_k=1)[0].chunk_id == "kb-fin"


def test_delete_removes_chunks() -> None:
    idx = BM25Index()
    idx.build(_records())
    removed = idx.delete(["kb-fin", "kb-sec"])
    assert removed == 2
    assert idx.size() == 1
    assert idx.search("报销", top_k=3) == []
    assert idx.search("采购", top_k=3)[0].chunk_id == "kb-ops"


def test_delete_missing_is_noop() -> None:
    idx = BM25Index()
    idx.build(_records())
    assert idx.delete(["no-such-chunk"]) == 0
    assert idx.size() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
