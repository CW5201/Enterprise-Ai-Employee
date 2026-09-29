"""Unit tests for the vector store backends (Phase 2.1).

- ``FakeVectorStore`` (test-mode): schema, parameter validation, empty
  results, dimension handling, result format — all offline, no Milvus.
- ``create_vector_store`` factory: explicit backend selection (formal
  failures must not silently downgrade to fake).
- ``MilvusVectorStore`` unit-level behaviour that does not need a live
  server: constructor validation and filter-expression building.

Real-server tests live in tests/integration/test_real_milvus_rag.py.
"""

from __future__ import annotations

import os

import pytest

os.environ["EMBEDDING_FORCE_OFFLINE"] = "1"

from src.core.embedder import HashEmbedder  # noqa: E402
from src.core.exceptions import RetrievalError  # noqa: E402
from src.core.vector_store import (  # noqa: E402
    FakeVectorStore,
    SearchHit,
    _build_filter_expr,
    create_vector_store,
)


def _records() -> list[dict]:
    embed = HashEmbedder()
    return [
        {
            "chunk_id": "kb-a",
            "document_id": "finance-0001",
            "title": "员工差旅与费用报销管理制度",
            "source": "finance/finance-0001",
            "category": "finance",
            "text": "报销流程须先经直属主管审批，再交财务审核。",
            "metadata": {"doc_id": "finance-0001"},
            "vector": embed("报销流程须先经直属主管审批，再交财务审核。"),
        },
        {
            "chunk_id": "kb-b",
            "document_id": "ops-0001",
            "title": "库存与仓储管理制度",
            "source": "operations/ops-0001",
            "category": "operations",
            "text": "采购申请须注明用途、数量与期望到货期。",
            "metadata": {"doc_id": "ops-0001"},
            "vector": embed("采购申请须注明用途、数量与期望到货期。"),
        },
    ]


@pytest.mark.unit
def test_fake_store_insert_and_size() -> None:
    store = FakeVectorStore("kb_test", dim=1024, embed_fn=HashEmbedder())
    n = store.insert(_records())
    assert n == 2
    assert store.size() == 2


@pytest.mark.unit
def test_fake_store_search_result_format() -> None:
    store = FakeVectorStore("kb_test", dim=1024, embed_fn=HashEmbedder())
    store.insert(_records())
    hits = store.search("报销流程", top_k=2)
    assert hits, "expected at least one hit"
    hit = hits[0]
    assert isinstance(hit, SearchHit)
    payload = hit.to_payload()
    for key in ("chunk_id", "document_id", "text", "title", "source", "category", "score", "metadata"):
        assert key in payload, f"missing key {key}"
    # source must never be dropped
    assert payload["source"]


@pytest.mark.unit
def test_fake_store_empty_search_returns_empty() -> None:
    store = FakeVectorStore("kb_empty", dim=1024, embed_fn=HashEmbedder())
    assert store.search("任意查询") == []


@pytest.mark.unit
def test_fake_store_search_requires_embed_fn() -> None:
    store = FakeVectorStore("kb_noembed", dim=1024)
    store.insert(_records())
    with pytest.raises(RetrievalError):
        store.search("报销")


@pytest.mark.unit
def test_fake_store_filtering() -> None:
    store = FakeVectorStore("kb_test", dim=1024, embed_fn=HashEmbedder())
    store.insert(_records())
    hits = store.search("报销", top_k=5, filter={"category": "finance"})
    assert hits, "expected finance hits"
    assert all(h.category == "finance" for h in hits)


@pytest.mark.unit
def test_fake_store_delete() -> None:
    store = FakeVectorStore("kb_test", dim=1024, embed_fn=HashEmbedder())
    store.insert(_records())
    removed = store.delete(["kb-a"])
    assert removed == 1
    assert store.size() == 1
    assert store.delete(["kb-missing"]) == 0


@pytest.mark.unit
def test_fake_store_drop_clears_state() -> None:
    store = FakeVectorStore("kb_test", dim=1024, embed_fn=HashEmbedder())
    store.insert(_records())
    store.drop()
    assert store.size() == 0
    assert store.search("报销") == []


@pytest.mark.unit
def test_fake_store_dimension_mismatch_reports_zero() -> None:
    """A vector of the wrong size never silently corrupts the fake store:
    it still inserts but scores 0.0 in cosine (guarded by _cosine)."""
    store = FakeVectorStore("kb_test", dim=1024, embed_fn=HashEmbedder())
    store.insert(
        [
            {
                "chunk_id": "kb-x",
                "document_id": "d",
                "title": "t",
                "source": "s",
                "category": "c",
                "text": "x" * 10,
                "metadata": {},
                "vector": [0.1] * 4,  # deliberately wrong dim
            }
        ]
    )
    hits = store.search("任意", top_k=1)
    assert hits[0].score == 0.0


@pytest.mark.unit
def test_factory_fake_backend_explicit() -> None:
    store = create_vector_store(backend="fake", dim=1024, collection="kb_factory")
    assert store.backend == "fake"
    assert store.dim == 1024


@pytest.mark.unit
def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(RetrievalError):
        create_vector_store(backend="bogus")


@pytest.mark.unit
def test_build_filter_expr_equivalence() -> None:
    assert _build_filter_expr(None) is None
    assert _build_filter_expr({}) is None
    expr = _build_filter_expr({"category": "finance", "document_id": "finance-0001"})
    assert expr == 'category == "finance" and document_id == "finance-0001"'
    # a filter dict containing ONLY the reserved vector field yields no expr
    assert _build_filter_expr({"vector": [0.1] * 4}) is None


@pytest.mark.unit
def test_milvus_store_constructor_defaults() -> None:
    """Constructing the Milvus store object is offline-safe (no connect)."""
    from src.core.vector_store import MilvusVectorStore

    store = MilvusVectorStore("kb", 1024, uri="http://localhost:19530")
    assert store.backend == "milvus"
    assert store.dim == 1024
    assert store._metric_type == "IP"
    assert store._index_type == "HNSW"
    assert store._index_params == {"M": 16, "efConstruction": 200}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
