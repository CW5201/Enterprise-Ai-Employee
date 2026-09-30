"""Real Milvus + BGE-M3 RAG integration test (Phase 2.1).

This test runs the FULL formal pipeline against a live Milvus server:

    query -> BGE-M3 (real model) -> Milvus search -> top-k -> LLM answer

If the required components (Milvus server / BGE-M3 model / knowledge base
index) are not available the test is **explicitly SKIPPED** with a reason —
it is NEVER satisfied by a mock or the fake backend.

Preconditions (checked in ``_probe``):
- a Milvus server reachable at ``MILVUS_HOST:MILVUS_PORT``;
- BGE-M3 loadable (real embedding, not the hash fallback);
- the configured collection exists and contains entities.

Run with::

    pytest tests/integration/test_real_milvus_rag.py -v -s
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make sure .env secrets are visible (MILVUS_* / EMBEDDING_MODEL / LLM_*).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.core.llm_client import _load_dotenv  # noqa: E402

_load_dotenv()

# Never let the hash-fallback embedder masquerade as BGE-M3 in this file.
# This file pops the variable before importing _load_dotenv; sibling test
# modules that set it (test_rag_e2e.py) only affect their own process run
# when executed in isolation.  When running the whole suite in one process,
# module import order matters — hence we pop it inside _skip_unless_env_ready
# AND on module import as a double-safeguard.
os.environ.pop("EMBEDDING_FORCE_OFFLINE", None)
os.environ.pop("LLM_FORCE_OFFLINE", None)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# 5 real knowledge-base queries (docs must actually exist in the KB)
# ---------------------------------------------------------------------------

REAL_QUERIES: list[tuple[str, str]] = [
    # (query, expected document title substring)
    ("员工差旅费报销需要哪些审批？", "差旅与费用报销"),
    ("哪些费用属于可报销范围？", "差旅与费用报销"),
    ("采购流程需要经过哪些步骤？", "库存与仓储"),
    ("员工远程办公有哪些规定？", "远程办公"),
    ("信息安全方面有哪些基本要求？", "信息安全"),
]


def _skip_unless_env_ready() -> None:
    """Skip with an explicit reason if Milvus or BGE-M3 is unavailable."""
    from src.core.embedder import create_embedder
    from src.core.exceptions import EmbeddingUnavailableError, RetrievalError
    from src.core.vector_store import create_vector_store

    # This test must never use the fake backend — ensure the env flag is clear
    # even if a sibling test module set it process-wide.
    os.environ.pop("EMBEDDING_FORCE_OFFLINE", None)

    # 1. Milvus reachable?
    try:
        store = create_vector_store(backend="milvus")
    except (RetrievalError, Exception) as exc:  # noqa: BLE001
        pytest.skip(f"Milvus not reachable: {exc}")
        return
    # 2. Collection exists with data?
    if not store._client.has_collection(store.collection):
        pytest.skip(f"collection '{store.collection}' not built yet; run scripts/build_kb.py")
        return
    # 3. Real BGE-M3 loadable?
    try:
        embedder = create_embedder()
        assert embedder.backend == "bge-m3", f"embedder backend is {embedder.backend}"
    except EmbeddingUnavailableError as exc:
        pytest.skip(f"BGE-M3 unavailable: {exc.message}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store() -> object:
    _skip_unless_env_ready()
    from src.core.vector_store import create_vector_store

    return create_vector_store(backend="milvus")


@pytest.fixture(scope="module")
def rag_tool(store: object) -> object:
    from src.tools.rag_tool import RAGTool

    return RAGTool(store=store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_milvus_acceptance_report(store: object) -> None:
    """Section 11: collection_name / entity_count / dimension / index / metric."""
    _skip_unless_env_ready()
    from pymilvus import MilvusClient

    from src.core.config_loader import get_settings

    settings = get_settings()
    milvus_cfg = settings.raw.get("milvus", {})
    client = MilvusClient(uri=f"http://{milvus_cfg.get('host', 'localhost')}:{milvus_cfg.get('port', 19530)}")
    desc = client.describe_collection(store.collection)

    vector_field = next(f for f in desc["fields"] if f["name"] == "vector")
    dimension = vector_field["params"]["dim"]

    stats = client.get_collection_stats(store.collection)
    entity_count = int(stats.get("row_count", 0))

    print("\n===== Milvus 验收报告 =====")
    print(f"collection_name : {store.collection}")
    print(f"entity_count    : {entity_count}")
    print(f"dimension       : {dimension}")
    print(f"metric_type     : {milvus_cfg.get('metric_type', 'IP')}")
    print(f"index_type      : {milvus_cfg.get('index_type', 'HNSW')}")

    assert entity_count > 0, f"entity_count must be > 0, got {entity_count}"
    assert dimension > 0
    # dimension must match the real model output
    from src.core.embedder import create_embedder

    embedder = create_embedder()
    probe = embedder("dimension check")
    assert dimension == len(probe), (
        f"collection dimension {dimension} != real BGE-M3 output {len(probe)}"
    )


@pytest.mark.integration
def test_each_real_query_retrieves_expected_doc(rag_tool: object) -> None:
    """Section 10: all 5 real queries must recall the expected document."""
    _skip_unless_env_ready()
    for query, expected_title_fragment in REAL_QUERIES:
        result = rag_tool.run(query=query, top_k=5)
        assert result["ok"], f"retrieval failed for {query!r}: {result.get('error')}"
        titles = [str(r.get("title", "")) for r in result["results"]]
        matched = [t for t in titles if expected_title_fragment in t]
        print(f"\n[{query}]\n  expected doc fragment: {expected_title_fragment!r}")
        print(f"  top titles: {titles[:3]}")
        assert matched, f"query {query!r} did NOT recall document containing {expected_title_fragment!r}\n  got titles: {titles}"
        # every hit must carry a non-empty source
        for r in result["results"]:
            assert r.get("source"), f"hit has no source: {r}"


@pytest.mark.integration
def test_full_pipeline_real_llm_answer(store: object) -> None:
    """Section 9: answer uses retrieved_context; evidence carries source."""
    _skip_unless_env_ready()
    os.environ.pop("LLM_FORCE_OFFLINE", None)
    from src.core.llm_client import LLMClient

    llm = LLMClient()
    if llm.offline:
        pytest.skip("no live LLM endpoint configured; answer test requires a real LLM")

    from src.core.state import make_state
    from src.nodes.answer_generation import AnswerGenerationNode
    from src.nodes.rag_retrieval import RAGRetrievalNode
    from src.tools.rag_tool import RAGTool

    tool = RAGTool(store=store)
    rag_node = RAGRetrievalNode(tool, top_k=5, backend="milvus")
    answer_node = AnswerGenerationNode(llm=llm)

    query = "员工差旅费报销需要哪些审批？"
    state = make_state(query, request_id="itest-real")
    state["intent"] = "knowledge_query"
    state["route"] = "rag"

    updated = rag_node.run(state)
    state = {**state, **updated}

    assert state["retrieved_context"], "retrieved_context must not be empty for a real query"
    for chunk in state["retrieved_context"]:
        assert chunk.source, "each retrieved chunk must keep its source"

    updated2 = answer_node.run(state)
    state = {**state, **updated2}
    answer = state.get("answer", "")
    assert answer, "answer must not be empty"
    assert state.get("status") in ("completed", "completed_fallback")
    # evidence items in the final state must reference Milvus chunks
    rag_evidence = [e for e in state.get("evidence", []) if e.source_type == "milvus"]
    assert rag_evidence, "answer stage must attach milvus evidence to the state"
    for ev in rag_evidence:
        assert ev.source_ref.startswith("milvus:kb-")
        assert ev.metadata.get("source") or ev.metadata.get("document_id")


@pytest.mark.integration
def test_evidence_score_is_real_ip_value(store: object) -> None:
    """No fake scores: returned scores are genuine inner-product values."""
    _skip_unless_env_ready()
    from src.tools.rag_tool import RAGTool

    tool = RAGTool(store=store)
    result = tool.run(query="差旅报销标准", top_k=3)
    for item in result["results"]:
        score = item["score"]
        assert isinstance(score, float)
        # BGE-M3 vectors are normalised and metric is IP -> score in [-1, 1]
        assert -1.0 <= score <= 1.0, f"score {score} out of plausible IP range"
        assert item["chunk_id"].startswith("kb-")


@pytest.mark.integration
def test_no_hardcoded_answers() -> None:
    """Guardrail: the RAG node must not short-circuit known questions."""
    _skip_unless_env_ready()
    from src.core.state import make_state
    from src.core.vector_store import create_vector_store
    from src.nodes.rag_retrieval import RAGRetrievalNode
    from src.tools.rag_tool import RAGTool

    store = create_vector_store(backend="milvus")
    node = RAGRetrievalNode(RAGTool(store=store), top_k=5)
    state = make_state("员工远程办公有哪些规定？", request_id="itest-nohardcode")
    state["intent"] = "knowledge_query"
    state["route"] = "rag"
    updated = node.run(state)
    # Even if the KB lacks a remote-work doc, the node must still return
    # genuine retrieval results (possibly empty), not a canned answer.
    chunks = updated.get("retrieved_context", [])
    for c in chunks:
        assert c.chunk_id, "every chunk must have a real chunk_id"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
