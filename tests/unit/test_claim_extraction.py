"""Unit tests: claim extraction + evidence adapter + evidence collection + linking.

Covers: answer -> multiple claims, SQL/RAG/KG/Analysis evidence,
duplicate dedupe, multi-source claim, failed extraction (no silent
fallback), entity / similarity linking.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.core.claim_evidence_linking import candidate_evidence, link_claims_to_evidence
from src.core.evidence_adapter import (
    adapt_analysis,
    adapt_kg,
    adapt_rag,
    adapt_sql,
)
from src.core.state import EvidenceItem, make_state
from src.core.verification_types import Evidence
from src.nodes.claim_extraction import ClaimExtractionError, ClaimExtractionNode
from src.nodes.evidence_collection import EvidenceCollectionNode, collect_evidence

# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


def _llm_claims(answer: str, question: str) -> list[dict[str, Any]]:
    """Deterministic stand-in for the LLM structured output (bare claim list)."""
    return [
        {"text": "本月差旅报销金额为 4600.5 元", "claim_type": "numerical",
         "importance": "critical", "source_refs": ["ev-sql"], "entities": ["4600.5"]},
        {"text": "差旅报销上限为 800 元", "claim_type": "rule_based",
         "importance": "normal", "source_refs": [], "entities": ["800"]},
        {"text": "环比增长 20%", "claim_type": "derived", "importance": "critical",
         "source_refs": [], "entities": [], "formula": "(2600-1200.5)/1200.5"},
    ]


class TestClaimExtraction:
    def test_answer_to_multiple_claims(self) -> None:
        node = ClaimExtractionNode(llm=object(), extract_fn=lambda a, q: _llm_claims(a, q))
        out = node.run(make_state("q") | {"answer": "A"})
        claims = out["claims"]
        assert len(claims) == 3
        types = {c["claim_type"] for c in claims}
        assert "numerical" in types and "derived" in types
        # deterministic post-processing parsed the scalar
        num = next(c for c in claims if c["claim_type"] == "numerical")
        assert num["value"] == 4600.5

    def test_numeric_value_parsed_as_float(self) -> None:
        node = ClaimExtractionNode(llm=object(), extract_fn=lambda a, q: _llm_claims(a, q))
        out = node.run(make_state("q") | {"answer": "A"})
        num = next(c for c in out["claims"] if c["claim_type"] == "numerical")
        assert isinstance(num["value"], float) and num["value"] == 4600.5

    def test_derived_formula_preserved(self) -> None:
        node = ClaimExtractionNode(llm=object(), extract_fn=lambda a, q: _llm_claims(a, q))
        out = node.run(make_state("q") | {"answer": "A"})
        derived = next(c for c in out["claims"] if c["claim_type"] == "derived")
        assert derived["formula"] == "(2600-1200.5)/1200.5"

    def test_failed_extraction_not_silent(self) -> None:
        def broken(a: str, q: str) -> Any:
            raise ClaimExtractionError("schema failure")

        node = ClaimExtractionNode(llm=object(), extract_fn=broken)
        out = node.run(make_state("q") | {"answer": "A"})
        assert out["claim_extraction_failed"] is True
        assert out["claims"] == []
        assert any(e.stage == "claim_extraction" for e in out["errors"])

    def test_empty_answer_honest(self) -> None:
        node = ClaimExtractionNode(llm=object(), extract_fn=lambda a, q: {"claims": []})
        out = node.run(make_state("q") | {"answer": ""})
        assert out["claims"] == []
        assert out.get("claim_extraction_failed") is None

    def test_malformed_payload_rejected(self) -> None:
        # a payload with unknown claim_type must raise (schema validation)
        def bad(a: str, q: str) -> dict[str, Any]:
            return {"claims": [{"text": "x", "claim_type": "nope"}]}

        node = ClaimExtractionNode(llm=object(), extract_fn=bad)
        with pytest.raises(Exception, match="claim_type|invalid"):
            node.run(make_state("q") | {"answer": "A"})


# ---------------------------------------------------------------------------
# Evidence adapter — one per tool
# ---------------------------------------------------------------------------


class TestEvidenceAdapters:
    def test_sql_evidence(self) -> None:
        ev = adapt_sql({"sql": "SELECT SUM(amount) FROM finance_expenses",
                        "columns": ["sum_amount"], "rows": [[4600.5]], "row_count": 1})
        assert ev.source_type == "sql"
        assert ev.provenance["query"] == "SELECT SUM(amount) FROM finance_expenses"
        assert ev.structured_value == 4600.5
        assert ev.source_ref.startswith("sql:")

    def test_rag_evidence(self) -> None:
        ev = adapt_rag({"chunk_id": "kb-1", "text": "差旅报销上限 800 元",
                        "source": "finance/fin-001", "document_id": "fin-001",
                        "title": "差旅报销制度", "score": 0.7, "metadata": {}})
        assert ev.source_type == "rag"
        assert ev.provenance["chunk_id"] == "kb-1"
        assert ev.retrieved_score == 0.7

    def test_kg_evidence(self) -> None:
        ev = adapt_kg({"success": True, "data": [{"order_id": 1}],
                       "template_id": "customer_orders", "query_type": "customer_orders"})
        assert ev.source_type == "kg"
        assert ev.provenance["template_id"] == "customer_orders"
        assert ev.structured_value == [{"order_id": 1}]

    def test_analysis_evidence(self) -> None:
        ev = adapt_analysis(
            {"ok": True, "data": {"operation": "growth_rate",
                                   "result": {"growth_rate": 0.2}, "input_rows": 2}},
            input_evidence_ids=["ev-sql"],
        )
        assert ev.source_type == "analysis"
        assert ev.provenance["input_evidence_ids"] == ["ev-sql"]
        assert ev.structured_value == {"growth_rate": 0.2}


# ---------------------------------------------------------------------------
# Evidence collection — dedupe + multi-source
# ---------------------------------------------------------------------------


def _state_with_tool_results() -> Any:
    s = make_state("q")
    s["tool_results"] = [
        {"tool": "sql", "success": True, "sql": "SELECT 1",
         "data": {"columns": ["x"], "rows": [[1]]}},
        {"tool": "rag", "success": True,
         "results": [{"chunk_id": "kb-1", "text": "t", "source": "s", "score": 0.5}]},
        {"tool": "kg", "success": True,
         "data": {"rows": [{"order_id": 1}], "query_type": "customer_orders",
                   "template_id": "customer_orders"}},
        {"tool": "analysis", "success": True,
         "data": {"operation": "growth_rate",
                   "result": {"growth_rate": 0.2}, "input_rows": 2}},
    ]
    return s


class TestEvidenceCollection:
    def test_multi_source_evidence(self) -> None:
        evs = collect_evidence(_state_with_tool_results())
        types = {e.source_type for e in evs}
        assert "sql" in types and "rag" in types and "kg" in types and "analysis" in types

    def test_duplicate_source_ref_deduped(self) -> None:
        s = _state_with_tool_results()
        # also supply the same rag chunk via retrieved_context -> dedupe
        s["retrieved_context"] = [
            {"chunk_id": "kb-1", "text": "t", "source": "s", "score": 0.5, "metadata": {}},
        ]
        evs = collect_evidence(s)
        refs = [e.source_ref for e in evs]
        assert refs.count("rag:kb-1") == 1

    def test_legacy_evidence_channel_adapted(self) -> None:
        s = make_state("q")
        s["evidence"] = [
            EvidenceItem(evidence_id="ev-legacy", source_type="duckdb",
                         source_ref="duckdb:SELECT 1", content="x",
                         payload={"sql": "SELECT 1", "columns": ["x"], "rows": [[1]]}),
        ]
        evs = collect_evidence(s)
        assert any(e.source_type == "sql" for e in evs)

    def test_node_writes_evidence_v4_channel(self) -> None:
        node = EvidenceCollectionNode()
        out = node.run(_state_with_tool_results())
        assert out["evidence_v4"]
        assert all(isinstance(x, dict) for x in out["evidence_v4"])


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------


def _pool() -> list[Evidence]:
    return [
        adapt_sql({"sql": "SELECT SUM(amount) FROM finance_expenses",
                   "columns": ["s"], "rows": [[4600.5]]}),
        adapt_rag({"chunk_id": "kb-2", "text": "差旅报销上限为 800 元",
                   "source": "f", "document_id": "fin-001", "title": "差旅制度",
                   "score": 0.9, "metadata": {}}),
        adapt_kg({"success": True, "data": [{"customer_id": 1, "order_id": 10}],
                 "template_id": "customer_orders"}),
    ]


class TestLinking:
    def test_explicit_source_ref_wins(self) -> None:
        # when an explicit ref actually resolves against the pool, the
        # linker returns ONLY that record (explicit refs are authoritative
        # and must not be diluted with other candidates)
        pool = _pool()
        sql_id = next(e.evidence_id for e in pool if e.source_type == "sql")
        claim = {"claim_id": "c1", "text": "金额为 4600.5", "claim_type": "numerical",
                 "source_refs": [sql_id], "value": 4600.5, "entities": []}
        ids, reason = candidate_evidence(claim, pool)
        assert reason == "explicit_source_refs"
        assert ids == [sql_id]

    def test_entity_overlap_multi_source(self) -> None:
        claim = {"claim_id": "c2", "text": "差旅报销上限为 800 元", "claim_type": "rule_based",
                 "source_refs": [], "value": 800, "entities": ["800"]}
        ids, reason = candidate_evidence(claim, _pool())
        assert reason == "entity_overlap"
        # 800 only appears in the rag evidence
        pool = _pool()
        rag_id = next(e.evidence_id for e in pool if e.source_type == "rag")
        assert rag_id in ids

    def test_similarity_fallback(self) -> None:
        claim = {"claim_id": "c3", "text": "订单 10 属于客户 1", "claim_type": "relational",
                 "source_refs": [], "value": None, "entities": ["订单", "客户"]}
        ids, reason = candidate_evidence(claim, _pool())
        assert ids or reason in ("below_similarity_threshold", "no_evidence")

    def test_link_all_claims(self) -> None:
        claims = [
            {"claim_id": "a", "text": "金额为 4600.5", "claim_type": "numerical",
             "source_refs": [], "value": 4600.5, "entities": []},
            {"claim_id": "b", "text": "订单 10 属于客户", "claim_type": "relational",
             "source_refs": [], "value": None, "entities": ["订单", "客户"]},
        ]
        out = link_claims_to_evidence(claims, _pool())
        assert "a" in out and "b" in out

    def test_no_evidence_honest(self) -> None:
        ids, reason = candidate_evidence(
            {"claim_id": "c", "text": "x", "claim_type": "factual",
             "source_refs": [], "value": None, "entities": []},
            [],
        )
        assert ids == [] and reason == "no_evidence"
