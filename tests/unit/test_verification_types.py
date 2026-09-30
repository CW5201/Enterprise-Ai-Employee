"""Unit tests for Phase 4 claim / evidence / verification data models.

Covers: valid claim, valid evidence (all 4 source types), invalid
evidence, multiple evidence, unsupported claim, conflicting evidence,
derived claim, secret guard, AgentState channel compatibility.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from src.core.state import AgentState, make_state
from src.core.verification_types import (
    Claim,
    Evidence,
    VerificationResult,
    assert_no_secrets,
    normalize_source_type,
)

# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------


class TestClaim:
    def test_valid_claims(self) -> None:
        c = Claim(claim_id="c1", text="本月销售额为 120000", claim_type="numerical",
                  value=120000.0, entities=["销售额"])
        assert c.is_checkable()

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Claim(claim_id="c1", text="x", claim_type="bogus")

    def test_unknown_importance_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Claim(claim_id="c1", text="x", importance="huge")  # type: ignore[arg-type]

    def test_derived_claim_carries_formula_and_inputs(self) -> None:
        c = Claim(
            claim_id="c2",
            text="February 比 January 增长 20%",
            claim_type="derived",
            formula="(120-100)/100",
            derived_from=["ev-jan", "ev-feb"],
            value=0.20,
        )
        assert c.is_checkable()
        assert c.formula == "(120-100)/100"
        assert c.derived_from == ["ev-jan", "ev-feb"]

    def test_opinion_not_checkable(self) -> None:
        c = Claim(claim_id="c3", text="整体表现不错", claim_type="opinion_or_summary")
        assert not c.is_checkable()

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Claim(claim_id="c4", text="")


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def _mk_ev(**over: Any) -> Evidence:
    base: dict[str, Any] = {"evidence_id": "ev-1", "source_type": "sql",
                            "source_ref": "duckdb:finance_expenses"}
    base.update(over)
    return Evidence(**base)


class TestEvidence:
    def test_sql_evidence_keeps_query_provenance(self) -> None:
        ev = _mk_ev(provenance={
            "database": "duckdb", "table": "finance_expenses",
            "query": "SELECT SUM(amount) FROM finance_expenses",
        }, structured_value=4600.5)
        # a SQL evidence must never record only "the answer is 4600.5"
        assert ev.provenance["query"]
        assert ev.structured_value == 4600.5

    def test_rag_evidence_keeps_doc_and_chunk(self) -> None:
        ev = _mk_ev(source_type="rag", source_ref="milvus:kb-abc123",
                    provenance={"doc_id": "fin-001", "chunk_id": "kb-abc123",
                                "title": "差旅报销制度"},
                    retrieved_score=0.81)
        assert ev.provenance["chunk_id"] == "kb-abc123"
        assert ev.retrieved_score == 0.81

    def test_kg_evidence_keeps_entities_and_template(self) -> None:
        ev = _mk_ev(source_type="kg", source_ref="neo4j:customer_orders:1",
                    provenance={"entities": ["Customer:1", "Order"], "relation": "PLACED",
                                "template_id": "customer_orders"},
                    structured_value=[{"order_id": 100}])
        assert ev.provenance["template_id"] == "customer_orders"

    def test_analysis_evidence_carries_operation_and_inputs(self) -> None:
        ev = _mk_ev(source_type="analysis", source_ref="analysis:growth_rate",
                    provenance={"operation": "growth_rate",
                                "input_evidence_ids": ["ev-sql-1"]},
                    structured_value={"growth_rate": 0.2})
        assert ev.provenance["input_evidence_ids"] == ["ev-sql-1"]

    def test_invalid_source_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _mk_ev(source_type="llm_output")  # LLM text is NOT evidence

    def test_invalid_evidence_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Evidence(source_type="sql")  # evidence_id required

    def test_multiple_evidence_records(self) -> None:
        evs = [
            _mk_ev(evidence_id="ev-a", source_type="sql", source_ref="duckdb:a"),
            _mk_ev(evidence_id="ev-b", source_type="rag", source_ref="milvus:b"),
        ]
        assert {e.source_type for e in evs} == {"sql", "rag"}

    def test_legacy_source_type_normalization(self) -> None:
        assert normalize_source_type("duckdb") == "sql"
        assert normalize_source_type("milvus") == "rag"
        assert normalize_source_type("fake") == "rag"
        assert normalize_source_type("neo4j") == "kg"
        assert normalize_source_type("sql") == "sql"  # idempotent


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def test_supported_result(self) -> None:
        vr = VerificationResult(claim_id="c1", supported=True, support_score=1.0,
                                status="supported", verifier_type="exact",
                                evidence_ids=["ev-1"], reason="120000 == 120000")
        assert vr.status == "supported"

    def test_unsupported_claim(self) -> None:
        vr = VerificationResult(claim_id="c1", supported=False, support_score=0.0,
                                status="unsupported", verifier_type="semantic",
                                evidence_ids=[], reason="no candidate evidence")
        assert not vr.supported
        assert not vr.conflict

    def test_conflict_evidence(self) -> None:
        vr = VerificationResult(claim_id="c1", supported=False, support_score=0.0,
                                status="conflict", conflict=True,
                                verifier_type="exact",
                                evidence_ids=["ev-a", "ev-b"],
                                reason="A=100 vs B=120; no auto-pick")
        assert vr.conflict
        assert not vr.supported

    def test_consistency_guard(self) -> None:
        # conflict=True must force status conflict + supported False
        with pytest.raises(ValidationError):
            VerificationResult(claim_id="c1", supported=True, status="supported",
                               conflict=True)
        with pytest.raises(ValidationError):
            VerificationResult(claim_id="c1", supported=False, status="supported")

    def test_support_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(claim_id="c1", supported=True, status="supported",
                               support_score=1.5)


# ---------------------------------------------------------------------------
# Secret guard
# ---------------------------------------------------------------------------


class TestSecretGuard:
    def test_clean_payload_passes(self) -> None:
        ev = _mk_ev(provenance={"query": "SELECT 1"})
        assert_no_secrets(ev.model_dump())

    def test_password_captured(self) -> None:
        with pytest.raises(ValueError, match="secret material"):
            assert_no_secrets({"url": "mysql://u:SuperSecretPass123@host/db"})

    def test_bearer_key_captured(self) -> None:
        with pytest.raises(ValueError, match="secret material"):
            assert_no_secrets("api_key: " + "sk-" + "A" * 24)


# ---------------------------------------------------------------------------
# AgentState channels
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_phase4_channels_initialised(self) -> None:
        s: AgentState = make_state("q")
        assert s["claims"] == []
        assert s["evidence_v4"] == []
        assert s["verification_results"] == []
        assert s["guard_decision"] == {}

    def test_phase3_fields_untouched(self) -> None:
        s = make_state("q")
        assert "routing_decision" in s and "routing_trace" in s
        assert "tool_results" in s and "route_confidence" in s

    def test_channels_are_appendable_dicts(self) -> None:
        s = make_state("q")
        s["claims"].append({"claim_id": "c1"})
        s["evidence_v4"].append({"evidence_id": "ev-1"})
        s["verification_results"].append({"claim_id": "c1", "status": "supported"})
        assert len(s["claims"]) == 1
