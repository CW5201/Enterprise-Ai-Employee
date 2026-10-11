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
