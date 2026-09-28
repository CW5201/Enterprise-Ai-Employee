"""Integration test: the RAG end-to-end path (Phase 1).

Exercises: knowledge-base document loading, chunking, dense + BM25 hybrid
retrieval with RRF fusion, evidence shaping and the answer node — all
offline (mock LLM, in-process vector index, no Milvus server).
"""

from __future__ import annotations

import os

import pytest

from src.graph.builder import build_graph
from src.tools.rag_tool import RAGTool

pytestmark = pytest.mark.integration


@pytest.fixture()
def rag_tool() -> RAGTool:
    tool = RAGTool()
    count = tool.build_index()
    assert count > 0, (
        "knowledge base corpus is empty - run "
        "data/knowledge_base/finance/finance-0001-*.md exists? "
        "see data/knowledge_base/*/README.md for the format"
    )
    return tool


def test_hybrid_retrieval_returns_policy_chunks(rag_tool: RAGTool) -> None:
    result = rag_tool.run(query="差旅费报销标准是多少", top_k=3)
    assert result.ok, result.error
    assert result.data, "expected retrieved chunks"
    for item in result.data:
        assert item["doc_id"]
        assert item["text"]
        assert item["score"] >= 0


def test_metadata_filtering(rag_tool: RAGTool) -> None:
    result = rag_tool.run(query="报销", top_k=5, filters={"department": "finance"})
    assert result.ok
    for item in result.data:
        assert item["metadata"].get("department") == "finance"


def test_rag_end_to_end_mock_llm() -> None:
    os.environ["LLM_BACKEND"] = "mock"
    graph = build_graph()
    state = graph.invoke(
        {
            "task_id": "t-rag-e2e",
            "user_task": "出差住宿标准和报销流程有哪些规定？",
            "conversation": [],
            "selected_tools": [],
            "tool_calls": [],
            "errors": [],
            "evidence": [],
            "citations": [],
            "slots": {},
            "constraints": {},
            "routing_history": [],
            "iteration": 0,
            "latency_ms": {},
            "messages": [],
        }
    )
    assert state["intent"] == "knowledge_qa", state["intent"]
    assert "rag" in state["selected_tools"], state["selected_tools"]
    rag_evidence = [e for e in state["evidence"] if e.source_type == "milvus"]
    assert rag_evidence, "expected milvus evidence from the finance policy doc"
    assert state.get("answer"), "answer should not be empty"
    # citations reference the produced evidence ids
    for ev in rag_evidence:
        assert ev.evidence_id in state["citations"]
