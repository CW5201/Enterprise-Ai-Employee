"""Unit tests for the RAG tool (Phase 2.1, fake backend only).

Exercises RAGTool against the in-memory fake vector store: result format,
parameter validation, empty results, source preservation and backend
labeling.  No live Milvus and no real model are involved.
"""

from __future__ import annotations

import os

import pytest

os.environ["EMBEDDING_FORCE_OFFLINE"] = "1"

from src.core.embedder import HashEmbedder  # noqa: E402
from src.core.vector_store import FakeVectorStore  # noqa: E402
from src.tools.rag_tool import RAGTool, RetrievalResult  # noqa: E402


def _make_tool() -> tuple[RAGTool, FakeVectorStore]:
    embed = HashEmbedder()
    store = FakeVectorStore("kb_rag_test", dim=1024, embed_fn=embed)
    tool = RAGTool(store=store)
    store.insert(
        [
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
    )
    return tool, store


@pytest.mark.unit
def test_run_returns_expected_result_shape() -> None:
    tool, _ = _make_tool()
    result = tool.run(query="差旅报销流程", top_k=2)
    assert result["ok"] is True
    assert result["tool"] == "rag"
    assert result["backend"] == "fake"
    assert isinstance(result["results"], list)
    assert result["index_size"] == 2


@pytest.mark.unit
def test_result_items_carry_all_required_fields() -> None:
    tool, _ = _make_tool()
    result = tool.run(query="差旅报销", top_k=2)
    for item in result["results"]:
        for key in ("chunk_id", "document_id", "text", "title", "source", "category", "score", "metadata"):
            assert key in item, f"missing {key}"
        # source must never be empty when a document has one
        assert item["source"]


@pytest.mark.unit
def test_top_k_limits_results() -> None:
    tool, _ = _make_tool()
    result = tool.run(query="制度", top_k=1)
    assert len(result["results"]) <= 1


@pytest.mark.unit
def test_empty_query_is_rejected_cleanly() -> None:
    tool, _ = _make_tool()
    result = tool.run(query="", top_k=3)
    assert result["ok"] is False
    assert result["results"] == []
    result2 = tool.run(query="   ", top_k=3)
    assert result2["ok"] is False
    assert result2["results"] == []


@pytest.mark.unit
def test_top_k_minimum_is_one() -> None:
    """top_k < 1 is clamped up to 1 rather than erroring."""
    tool, _ = _make_tool()
    result = tool.run(query="制度", top_k=0)
    assert len(result["results"]) <= 1


@pytest.mark.unit
def test_empty_store_returns_no_results() -> None:
    empty_store = FakeVectorStore("kb_empty", dim=1024, embed_fn=HashEmbedder())
    tool = RAGTool(store=empty_store)
    result = tool.run(query="任意查询", top_k=5)
    assert result["ok"] is True
    assert result["results"] == []


@pytest.mark.unit
def test_retrieval_result_dataclass_shape() -> None:
    r = RetrievalResult(query="q", results=[{"chunk_id": "x", "source": "s"}], backend="fake", index_size=1)
    assert r.query == "q"
    assert r.backend == "fake"
    assert r.index_size == 1


@pytest.mark.unit
def test_tool_defaults_to_backend_argument() -> None:
    tool = RAGTool(backend="fake")
    assert tool.store.backend == "fake"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
