"""Unit tests for the layered claim-evidence verification engine.

Covers: exact supported / conflict, unsupported, semantic supported /
unsupported, KG relation supported, derived calculation, conflicting
evidence (never auto-pick), empty evidence, layer ablation (rule-only),
and config-driven thresholds.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.core.claim_verifier import (
    LexicalScorer,
    VerificationConfig,
    summarize,
    verify,
    verify_semantic,
)
from src.core.evidence_adapter import adapt_analysis, adapt_kg, adapt_rag, adapt_sql
from src.core.verification_types import Evidence

CFG = VerificationConfig(support_threshold=0.6, semantic_threshold=0.5)


def test_committed_thresholds_match_the_published_experiments() -> None:
    """ADR-015: config/settings.yaml must equal the口径 the papers report.

    The Phase 4 / Phase 5 numbers in docs/ and artifacts/ were produced with
    these exact thresholds.  Raising them silently (as an earlier repair
    session did: 0.6 -> 0.75) invalidates every verification metric quoted in
    the thesis until both evaluations are re-run — so pin them here.
    """
    from src.core.config_loader import get_settings

    verification = get_settings().verification
    assert verification.min_support_score == 0.6
    assert verification.support_threshold == 0.6
    assert verification.semantic_threshold == 0.5
    assert verification.numeric_tolerance == 0.01
    # ... and the code defaults must not drift from the config either.
    assert VerificationConfig.support_threshold == 0.6
    assert VerificationConfig.semantic_threshold == 0.5
    assert VerificationConfig.numeric_tolerance == 0.01


def _sql_ev(value: Any, eid: str = "ev-sql") -> Evidence:
    return adapt_sql({"sql": "SELECT 1", "columns": ["v"], "rows": [[value]]}, evidence_id=eid)


def _rag_ev(text: str, eid: str = "ev-rag") -> Evidence:
    return adapt_rag({"chunk_id": "kb-x", "text": text, "source": "s", "score": 0.9, "metadata": {}},
                     evidence_id=eid)


# ---------------------------------------------------------------------------
# Layer 1 — exact
# ---------------------------------------------------------------------------


class TestExact:
    def test_exact_supported(self) -> None:
        claim = {"claim_id": "c1", "text": "本月销售额为 120000", "claim_type": "numerical",
                 "value": 120000, "source_refs": [], "entities": []}
        r = verify(claim, [_sql_ev(120000)], CFG, layers=("exact",))
        assert r.status == "supported" and r.supported and r.verifier_type == "exact"

    def test_exact_conflict(self) -> None:
        claim = {"claim_id": "c1", "text": "销售额为 120000", "claim_type": "numerical",
                 "value": 120000, "source_refs": [], "entities": []}
        r = verify(claim, [_sql_ev(130000)], CFG, layers=("exact",))
        assert r.status == "conflict" and r.conflict and not r.supported

    def test_unsupported_no_value_in_evidence(self) -> None:
        claim = {"claim_id": "c1", "text": "X", "claim_type": "factual",
                 "value": None, "source_refs": [], "entities": []}
        # factual claim: exact layer returns None, rule layer inconclusive,
        # semantic layer -> no compatible rag evidence -> unsupported
        r = verify(claim, [], CFG)
        assert r.status == "unsupported"

    def test_conflicting_evidence_never_autopick(self) -> None:
        claim = {"claim_id": "c1", "text": "A=120", "claim_type": "numerical",
                 "value": 120, "source_refs": [], "entities": []}
        pool = [
            adapt_sql({"sql": "q", "columns": ["v"], "rows": [[100]]}, evidence_id="ev-a"),
            adapt_sql({"sql": "q", "columns": ["v"], "rows": [[120]]}, evidence_id="ev-b"),
        ]
        r = verify(claim, pool, CFG, layers=("exact",))
        assert r.status == "conflict" and r.conflict
        assert set(r.evidence_ids) == {"ev-a", "ev-b"}


# ---------------------------------------------------------------------------
# Layer 2 — rule
# ---------------------------------------------------------------------------


class TestRule:
    def test_kg_relation_supported(self) -> None:
        claim = {"claim_id": "c1", "text": "客户 1 购买过订单 10", "claim_type": "relational",
                 "value": None, "source_refs": [], "entities": ["1", "10"]}
        kg = adapt_kg({"success": True, "data": [{"customer_id": 1, "order_id": 10}],
                       "template_id": "customer_orders"})
        kg.content = "Customer customer_id=1 placed order order_id=10"
        kg.structured_value = {"customer_id": 1, "order_id": 10}
        r = verify(claim, [kg], CFG, layers=("rule",))
        assert r is not None and r.status == "supported" and r.verifier_type == "rule"

    def test_rule_only_layer(self) -> None:
        claim = {"claim_id": "c1", "text": "差旅报销上限为 800 元", "claim_type": "rule_based",
                 "value": 800, "source_refs": [], "entities": []}
        r = verify(claim, [_rag_ev("差旅报销上限为 800 元")], CFG, layers=("rule",))
        assert r.status == "supported" and r.verifier_type == "rule"


# ---------------------------------------------------------------------------
# Layer 3 — semantic
# ---------------------------------------------------------------------------


class TestSemantic:
    def test_semantic_supported_with_llm(self) -> None:
        class FakeLLM:
            def generate_structured(self, system: str, user: str, model: Any) -> Any:
                return model(supported=True, support_score=0.9,
                             reason="the policy text backs the claim", conflicting_evidence_ids=[])

        claim = {"claim_id": "c1", "text": "差旅报销需要附发票", "claim_type": "rule_based",
                 "value": None, "source_refs": [], "entities": []}
        ev = _rag_ev("差旅报销需要附发票及审批单")
        r = verify_semantic(claim, [ev], CFG, scorer=LexicalScorer(), llm=FakeLLM())
        assert r.status == "supported" and r.verifier_type == "semantic"

    def test_semantic_unsupported_low_similarity(self) -> None:
        claim = {"claim_id": "c1", "text": "公司允许无限期借款给高管亲属使用", "claim_type": "factual",
                 "value": None, "source_refs": [], "entities": []}
        ev = _rag_ev("差旅报销上限为 800 元")
        tight = VerificationConfig(support_threshold=0.6, semantic_threshold=0.6)
        r = verify_semantic(claim, [ev], tight, scorer=LexicalScorer(), llm=None)
        assert r.status == "unsupported"

    def test_semantic_conflict_detected_by_llm(self) -> None:
        class FakeLLM:
            def generate_structured(self, system: str, user: str, model: Any) -> Any:
                return model(supported=False, support_score=0.0,
                             reason="values differ", conflicting_evidence_ids=["ev-rag"])

        claim = {"claim_id": "c1", "text": "报销上限 800 元", "claim_type": "rule_based",
                 "value": 800, "source_refs": [], "entities": []}
        ev = _rag_ev("制度规定报销上限 800 元，但另一制度写的是 1200 元")
        r = verify_semantic(claim, [ev], CFG, scorer=LexicalScorer(), llm=FakeLLM())
        assert r.conflict and r.status == "conflict"

    def test_empty_evidence_honest_unsupported(self) -> None:
        claim = {"claim_id": "c1", "text": "X", "claim_type": "factual",
                 "value": None, "source_refs": [], "entities": []}
        r = verify_semantic(claim, [], CFG, scorer=LexicalScorer())
        assert r.status == "unsupported"


# ---------------------------------------------------------------------------
# Derived claims
# ---------------------------------------------------------------------------


class TestDerived:
    def test_derived_calculation_recomputed(self) -> None:
        # February grew 20% over January: (120-100)/100 = 0.20
        claim = {"claim_id": "c1", "text": "February 比 January 增长 20%",
                 "claim_type": "derived", "value": 0.20, "source_refs": [],
                 "entities": ["100", "120"]}
        jan = _sql_ev(100, "ev-jan")
        feb = _sql_ev(120, "ev-feb")
        r = verify(claim, [jan, feb], CFG, layers=("exact",))
        # the growth rate itself is computed by the analysis tool; here we
        # check the exact layer against the recomputed value
        assert r is not None
        # 0.20 should not match 100/120 directly -> not supported by exact
        assert r.status in ("conflict", "unsupported")

    def test_derived_via_analysis_evidence(self) -> None:
        claim = {"claim_id": "c1", "text": "增长率 20%", "claim_type": "derived",
                 "value": 0.2, "source_refs": [], "entities": []}
        ev = adapt_analysis({"ok": True,
                             "data": {"operation": "growth_rate",
                                       "result": {"growth_rate": 0.2}}},
                            input_evidence_ids=["ev-sql"])
        r = verify(claim, [ev], CFG, layers=("exact",))
        # analysis structured_value is a dict {growth_rate: 0.2} -> exact
        # layer cannot parse a scalar from it -> inconclusive -> conflict/unsup
        assert r is not None


# ---------------------------------------------------------------------------
# Layer ablation (rule-only vs full)
# ---------------------------------------------------------------------------


class TestLayerAblation:
    def test_rule_only_does_not_call_semantic(self) -> None:
        claim = {"claim_id": "c1", "text": "一个完全无关的事实陈述",
                 "claim_type": "factual", "value": None,
                 "source_refs": [], "entities": []}
        ev = _rag_ev("差旅报销上限为 800 元")
        r = verify(claim, [ev], CFG, layers=("rule",))
        # rule layer cannot confirm an unrelated factual claim -> None ->
        # no semantic layer -> honest unsupported
        assert r.status == "unsupported" and r.verifier_type == "none"

    def test_full_stack_falls_through(self) -> None:
        claim = {"claim_id": "c1", "text": "差旅报销上限为 800 元",
                 "claim_type": "rule_based", "value": 800,
                 "source_refs": [], "entities": []}
        ev = _rag_ev("差旅报销上限为 800 元")
        r = verify(claim, [ev], CFG, layers=("exact", "rule", "semantic"))
        # exact (value non-None but claim_type rule_based -> exact skips),
        # rule confirms value in rag -> supported
        assert r.status == "supported"

    def test_threshold_is_config_driven(self) -> None:
        tight = VerificationConfig(support_threshold=0.99, semantic_threshold=0.9)
        claim = {"claim_id": "c1", "text": "某事实陈述", "claim_type": "factual",
                 "value": None, "source_refs": [], "entities": []}
        ev = _rag_ev("某事实陈述的相关内容")
        r = verify(claim, [ev], tight, layers=("semantic",))
        assert not r.supported or r.support_score < 0.99


@pytest.mark.unit
class TestSummarize:
    """The three summary buckets must be mutually exclusive.

    ``AnswerGuardNode`` builds the "待核实" list as ``not conflict and not
    supported``, and the eval script counts the same way.  ``summarize``
    used to add a conflicting result to *both* ``conflicts`` and
    ``unsupported_claims``, so the three numbers summed to more than
    ``total`` and the UI showed a claim as both "证据不足" and "冲突".
    """

    @staticmethod
    def _res(cid: str, *, supported: bool, conflict: bool) -> Any:
        from src.core.verification_types import VerificationResult
        return VerificationResult(
            claim_id=cid,
            supported=supported,
            conflict=conflict,
            status="conflict" if conflict else ("supported" if supported else "unsupported"),
        )

    def test_conflict_is_counted_once(self) -> None:
        results = [
            self._res("clm-001", supported=False, conflict=True),
            self._res("clm-002", supported=True, conflict=False),
            self._res("clm-003", supported=False, conflict=False),
        ]
        out = summarize(results)
        assert out["total"] == 3
        assert out["conflicts"] == 1
        assert out["unsupported_claims"] == 1, "conflict must not also be unsupported"
        assert out["supported_claims"] == 1

    def test_buckets_sum_to_total(self) -> None:
        results = [
            self._res("a", supported=True, conflict=False),
            self._res("b", supported=False, conflict=False),
            self._res("c", supported=False, conflict=True),
            self._res("d", supported=False, conflict=False),
        ]
        out = summarize(results)
        assert (out["supported_claims"] + out["unsupported_claims"]
                + out["conflicts"]) == out["total"]

    def test_unsupported_matches_the_guard_definition(self) -> None:
        """The count must equal the length of the guard's 待核实 list."""
        results = [
            self._res("a", supported=False, conflict=False),
            self._res("b", supported=False, conflict=True),
            self._res("c", supported=True, conflict=False),
        ]
        guard_unsupported = [r.claim_id for r in results
                             if not r.conflict and not r.supported]
        assert summarize(results)["unsupported_claims"] == len(guard_unsupported)

    def test_all_supported(self) -> None:
        results = [self._res(f"c{i}", supported=True, conflict=False) for i in range(4)]
        out = summarize(results)
        assert out == {"supported_claims": 4, "unsupported_claims": 0,
                       "conflicts": 0, "total": 4}


# ---------------------------------------------------------------------------
# Multi-row SQL evidence — the numbers live inside ``rows``
# ---------------------------------------------------------------------------


class TestMultiRowEvidence:
    """Regression: ``adapt_sql`` wraps >1 row as a dict.

    ``_to_float`` returns ``None`` for a dict, so the exact layer used to find
    no value at all, the rule layer skipped ``numerical`` claims, and every
    correct multi-row answer fell through to the semantic layer and came back
    unsupported — which the guard then surfaced as ``guard_blocked``.
    """

    @staticmethod
    def _tabular_evidence() -> Any:
        return adapt_sql({
            "sql": "SELECT strftime(OrderDate, '%Y-%m') AS month, "
                   "SUM(Quantity*UnitPrice) AS revenue FROM t GROUP BY 1",
            "columns": ["month", "revenue"],
            "rows": [["2025-04", 5235672.00], ["2025-05", 1908059.49],
                     ["2025-06", 47491.55]],
            "row_count": 3,
        })

    @staticmethod
    def _numerical_claim(value: Any) -> dict[str, Any]:
        return {"claim_id": "c1", "text": "该月销售额为 5235672.00",
                "claim_type": "numerical", "importance": "critical",
                "source_refs": [], "entities": [], "value": value}

    def test_structured_value_is_a_dict_for_multi_row(self) -> None:
        ev = self._tabular_evidence()
        assert isinstance(ev.structured_value, dict)
        assert len(ev.structured_value["rows"]) == 3

    def test_claim_value_present_in_a_row_is_supported(self) -> None:
        ev = self._tabular_evidence()
        r = verify(self._numerical_claim(5235672.0), [ev], CFG)
        assert r.supported is True
        assert r.verifier_type == "exact"
        assert r.conflict is False

    @pytest.mark.parametrize("value", [1908059.49, 47491.55])
    def test_every_row_value_is_reachable(self, value: float) -> None:
        r = verify(self._numerical_claim(value), [self._tabular_evidence()], CFG)
        assert r.supported is True, r.reason

    def test_absent_number_is_inconclusive_not_conflict(self) -> None:
        """A multi-row result legitimately holds many unrelated numbers.

        The claimed value simply not being present must not be reported as
        sources disagreeing with each other.
        """
        r = verify(self._numerical_claim(999999.0), [self._tabular_evidence()], CFG)
        assert r.conflict is False

    def test_year_must_not_match_a_date_cell(self) -> None:
        """``2025-04`` must not be readable as the measurement ``2025``."""
        r = verify(self._numerical_claim(2025), [self._tabular_evidence()], CFG)
        assert r.supported is False

    def test_scalar_evidence_still_reports_conflict(self) -> None:
        """The existing single-value conflict contract must not regress."""
        a = adapt_sql({"sql": "SELECT 1", "columns": ["x"], "rows": [[10]],
                       "row_count": 1})
        b = adapt_sql({"sql": "SELECT 2", "columns": ["x"], "rows": [[20]],
                       "row_count": 1})
        r = verify(self._numerical_claim(10), [a, b], CFG)
        assert r.conflict is True
        assert r.supported is False


@pytest.mark.unit
class TestEvidenceIdsAreUnique:
    """A value repeated across rows used to append the same evidence id once
    per occurrence (``evidence_ids: ["ev-tool-sql-1", "ev-tool-sql-1"]``) —
    visible in the API payload and misleading in the audit trail."""

    @pytest.fixture
    def ev(self):
        from src.core.verification_types import Evidence
        return Evidence(
            evidence_id="ev-tool-sql-1",
            source_type="sql",
            source_ref="sql:SELECT name, n",
            content=("SQL: SELECT name, n Columns: name, n\n"
                     "Dalton Perez\t28\nMason D Roberts\t28\nNancy\t27"),
            structured_value={"columns": ["name", "n"],
                              "rows": [["Dalton Perez", 28], ["Mason D Roberts", 28],
                                       ["Nancy", 27]],
                              "row_count": 3},
        )

    @staticmethod
    def _claim(value):
        return {"claim_id": "c1", "claim_type": "numerical",
                "text": f"订单数量 {value}", "value": value,
                "source_refs": [], "entities": []}

    def test_repeated_value_yields_one_id(self, ev) -> None:
        r = verify(self._claim(28), [ev], CFG)
        assert r.supported is True, r.reason
        assert r.evidence_ids == ["ev-tool-sql-1"], r.evidence_ids

    def test_conflict_path_is_also_unique(self, ev) -> None:
        r = verify(self._claim(99), [ev], CFG)
        assert len(r.evidence_ids) == len(set(r.evidence_ids)), r.evidence_ids
        assert r.evidence_ids, "conflict must still cite its evidence"
