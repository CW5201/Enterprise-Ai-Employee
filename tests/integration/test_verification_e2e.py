"""Integration tests: Phase 4 verification pipeline wiring.

Runs the full Phase 4 graph (or a hand-wired node chain that mirrors it)
end to end across the required scenarios and asserts the guard /
verification verdicts.  Uses fake stores + injected LLMs so the tests are
deterministic and offline; no real Milvus / Neo4j / LLM is required.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.core.state import make_state
from src.nodes.answer_guard import AnswerGuardNode
from src.nodes.claim_extraction import ClaimExtractionNode
from src.nodes.evidence_collection import EvidenceCollectionNode
from src.nodes.verification import VerificationNode


def _run_phase4_nodes(state: dict[str, Any], *, claims: list[dict[str, Any]], llm: Any = None) -> dict[str, Any]:
    """Run the 4 Phase 4 nodes in order against a prepared state."""
    # The claim node extracts from state["answer"]; set it so the
    # deterministic extractor has non-empty input.
    state = {**state, "answer": state.get("answer") or "A"}
    # 1. claim extraction (inject the claim set deterministically)
    ce = ClaimExtractionNode(llm=llm or object(), extract_fn=lambda a, q: claims)
    out = ce.run(state)
    merged = {**state, **out}

    # 2. evidence collection -> evidence_v4
    merged = {**merged, **EvidenceCollectionNode().run(merged)}

    # 3. verification -> verification_results
    merged = {**merged, **VerificationNode(llm=llm, scorer=None).run(merged)}

    # 4. answer guard -> guard_decision / answer_sources / final answer
    merged = {**merged, **AnswerGuardNode().run(merged)}
    return merged


def _sql_state(**over: Any) -> dict[str, Any]:
    s = make_state("q")
    s["sql"] = "SELECT SUM(amount) FROM finance_expenses"
    s["sql_result"] = _FakeSQLResult()
    s.update(over)
    return s


class _FakeSQLResult:
    columns = ["sum_amount"]
    rows = [[4600.5]]
    row_count = 1
    execution_time_ms = 1.0
    truncated = False


# ---------------------------------------------------------------------------
# 1. RAG answer -> verify
# ---------------------------------------------------------------------------


def test_rag_answer_verify() -> None:
    state = _sql_state(
        retrieved_context=[
            {"chunk_id": "kb-1", "text": "差旅报销上限为 800 元", "source": "finance/fin-001",
             "score": 0.9, "metadata": {"title": "差旅报销制度", "document_id": "fin-001"}},
        ],
    )
    claims = [{"text": "差旅报销上限为 800 元", "claim_type": "rule_based",
               "importance": "normal", "source_refs": [], "entities": ["800"]}]
    out = _run_phase4_nodes(state, claims=claims)
    assert out["verification_results"], "verification must run"
    assert out["guard_decision"]["block"] is False
    assert out["status"] in ("verified", "guarded")
    assert any(r["status"] == "supported" for r in out["verification_results"])


# ---------------------------------------------------------------------------
# 2. SQL answer -> verify (numerical, exact layer)
# ---------------------------------------------------------------------------


def test_sql_answer_verify() -> None:
    state = _sql_state()
    claims = [{"text": "本月费用合计为 4600.5 元", "claim_type": "numerical",
               "importance": "critical", "source_refs": [], "entities": ["4600.5"], "value": 4600.5}]
    out = _run_phase4_nodes(state, claims=claims)
    assert out["verification_results"]
    supported = [r for r in out["verification_results"] if r["status"] == "supported"]
    assert supported, "a matching SQL value should be supported by the exact layer"


# ---------------------------------------------------------------------------
# 3. KG answer -> verify (relational, rule layer)
# ---------------------------------------------------------------------------


def test_kg_answer_verify() -> None:
    state = _sql_state(
        tool_results=[
            {"tool": "kg", "success": True,
             "data": {"rows": [{"customer_id": 1, "order_id": 100}],
                       "template_id": "customer_orders", "query_type": "customer_orders"}},
        ],
    )
    claims = [{"text": "客户 1 下过订单 100", "claim_type": "relational",
               "importance": "normal", "source_refs": [], "entities": ["1", "100"]}]
    out = _run_phase4_nodes(state, claims=claims)
    assert out["verification_results"]


# ---------------------------------------------------------------------------
# 4. SQL + Analysis -> verify (derived)
# ---------------------------------------------------------------------------


def test_sql_analysis_verify() -> None:
    state = _sql_state(
        tool_results=[
            {"tool": "sql", "success": True, "sql": "SELECT 100", "data": {"columns": ["v"], "rows": [[100]]}},
            {"tool": "sql", "success": True, "sql": "SELECT 120", "data": {"columns": ["v"], "rows": [[120]]}},
            {"tool": "analysis", "success": True,
             "data": {"operation": "growth_rate", "result": {"growth_rate": 0.2}, "input_rows": 2}},
        ],
    )
    claims = [{"text": "环比增长 20%", "claim_type": "derived",
               "importance": "critical", "source_refs": [], "entities": [], "value": 0.2,
               "formula": "(120-100)/100"}]
    out = _run_phase4_nodes(state, claims=claims)
    assert out["verification_results"]


# ---------------------------------------------------------------------------
# 5. Multi-source answer -> verify
# ---------------------------------------------------------------------------


def test_multi_source_verify() -> None:
    state = _sql_state(
        tool_results=[
            {"tool": "sql", "success": True, "sql": "SELECT 4600.5", "data": {"columns": ["v"], "rows": [[4600.5]]}},
            {"tool": "rag", "success": True,
             "results": [{"chunk_id": "kb-9", "text": "差旅报销上限 800 元", "source": "s", "score": 0.7,
                           "metadata": {"title": "差旅制度"}}]},
        ],
    )
    claims = [
        {"text": "费用合计 4600.5 元", "claim_type": "numerical", "importance": "normal",
         "source_refs": [], "entities": ["4600.5"], "value": 4600.5},
        {"text": "差旅报销上限 800 元", "claim_type": "rule_based", "importance": "normal",
         "source_refs": [], "entities": ["800"], "value": 800},
    ]
    out = _run_phase4_nodes(state, claims=claims)
    assert len(out["verification_results"]) == 2


# ---------------------------------------------------------------------------
# 6. Unsupported claim -> guard (no block, flagged)
# ---------------------------------------------------------------------------


def test_unsupported_claim_guard() -> None:
    state = _sql_state(
        sql_result=None,
        retrieved_context=[],
        tool_results=[],
    )
    # a normal-importance unsupported claim: guard flags it, does NOT block
    claims = [{"text": "某完全无证据的普通结论", "claim_type": "factual",
               "importance": "normal", "source_refs": [], "entities": []}]
    out = _run_phase4_nodes(state, claims=claims)
    assert out["guard_decision"]["block"] is False
    assert out["status"] == "guarded"
    assert out["guard_decision"]["unsupported"]


# ---------------------------------------------------------------------------
# 7. Conflict -> guard
# ---------------------------------------------------------------------------


def test_conflict_guard() -> None:
    # Two sql sources disagree with each other: one matches the claim
    # (100), the other contradicts it (120) -> the exact layer reports
    # conflict and never auto-picks a value.
    state = _sql_state(
        tool_results=[
            {"tool": "sql", "success": True, "sql": "A", "data": {"columns": ["v"], "rows": [[100]]}},
            {"tool": "sql", "success": True, "sql": "B", "data": {"columns": ["v"], "rows": [[120]]}},
        ],
    )
    claims = [{"text": "数值为 100", "claim_type": "numerical",
               "importance": "critical", "source_refs": [], "entities": ["100"], "value": 100}]
    out = _run_phase4_nodes(state, claims=claims)
    conflicts = [r for r in out["verification_results"] if r["conflict"]]
    assert conflicts, "disagreeing sql sources must surface as conflict"
    assert out["guard_decision"]["conflicts"]


# ---------------------------------------------------------------------------
# Graph wiring smoke test (no LLM needed: graph must compile + run)
# ---------------------------------------------------------------------------


def test_phase4_graph_compiles() -> None:
    from src.graph.builder import build_graph

    graph = build_graph(backend="fake", phase4=True, phase3=False)
    assert graph is not None


@pytest.mark.integration
def test_phase4_full_graph_fake_backend() -> None:
    from src.graph.builder import DigitalEmployee

    employee = DigitalEmployee(backend="fake", phase4=True)
    state = employee.run("差旅报销上限是多少？")
    assert "guard_decision" in state
    assert "verification_summary" in state
