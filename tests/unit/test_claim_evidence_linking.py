"""Unit tests for src/core/claim_evidence_linking.py.

The regression that motivated these: a correct answer ("订单数量最多的前 3 个
客户" — 28/28/27 orders) came back ``guard_blocked``.  ``verification`` showed
``supported_claims: 3, unsupported_claims: 1`` with the failing claim reading
``evidence_ids: []`` and ``reason: "no candidate evidence available"``.

Cause: ``candidate_evidence`` returned ``[]`` for *any* claim whose explicit
``source_refs`` failed to resolve against the pool.  An empty candidate set
short-circuits the verifier before it reads a single character of evidence, so
the claim is reported unsupported regardless of whether the answer is right,
and ``answer_guard`` blocks a perfectly good response.  Only numeric/derived
claims had a fallback; ``relational`` / ``factual`` / ``opinion_or_summary``
did not — the three types the model emits for ranking and summary sentences.
"""

from __future__ import annotations

import pytest

from src.core.claim_evidence_linking import candidate_evidence, link_claims_to_evidence
from src.core.claim_verifier import LexicalScorer, VerificationConfig, verify_all
from src.core.verification_types import Evidence

_SQL_EV = Evidence(
    evidence_id="ev-tool-sql-1",
    source_type="sql",
    source_ref="sql:SELECT c.CustomerName, COUNT(o.OrderID) AS order_count FROM Sales_Orders o GROUP BY 1 ORDER BY 2 DESC LIMIT 3",
    content=(
        "SQL: SELECT c.CustomerName, COUNT(o.OrderID) AS order_count FROM Sales_Orders o "
        "JOIN Sales_Customers c ON c.CustomerID = o.CustomerID GROUP BY c.CustomerName "
        "ORDER BY order_count DESC LIMIT 3 Columns: CustomerName, order_count\n"
        "Dalton Perez\t28\nMason D Roberts\t28\nFernando Barnes\t27"
    ),
    structured_value={
        "columns": ["CustomerName", "order_count"],
        "rows": [["Dalton Perez", 28], ["Mason D Roberts", 28], ["Fernando Barnes", 27]],
        "row_count": 3,
    },
)

_RAG_EV = Evidence(
    evidence_id="ev-rag-7",
    source_type="rag",
    source_ref="milvus:7",
    content="Sales order aggregation is computed as COUNT of order headers per customer.",
)


def _claim(cid: str, ctype: str, text: str, refs: list[str],
           entities: list[str] | None = None, value=None) -> dict:
    return {"claim_id": cid, "claim_type": ctype, "text": text, "source_refs": refs,
            "value": value, "entities": list(entities or [])}


_TOP3 = ["Dalton Perez", "Mason D Roberts", "Fernando Barnes"]


@pytest.mark.unit
class TestCandidateEvidence:
    def test_resolvable_refs_win(self) -> None:
        ids, reason = candidate_evidence(
            _claim("c1", "numerical", "28 orders", ["ev-tool-sql-1"]), [_SQL_EV]
        )
        assert ids == ["ev-tool-sql-1"]
        assert reason == "explicit_source_refs"

    def test_empty_pool(self) -> None:
        ids, reason = candidate_evidence(_claim("c1", "factual", "x", ["nope"]), [])
        assert ids == []
        assert reason == "no_evidence"

    @pytest.mark.parametrize("ctype", ["relational", "factual", "opinion_or_summary", "rule_based"])
    def test_unresolved_refs_never_yield_an_empty_set(self, ctype: str) -> None:
        """The regression.  Refs the pool never had must fall through to the
        fallback rules instead of starving the verifier."""
        ids, reason = candidate_evidence(
            _claim("c1", ctype, "订单数量最多的前 3 个客户", ["sources", "引用[1]"]),
            [_SQL_EV, _RAG_EV],
        )
        assert ids, f"{ctype} got no candidates (reason={reason})"
        assert reason != "no_explicit_refs_matched"

    def test_numeric_with_unresolved_refs_falls_back_to_structured_pool(self) -> None:
        ids, reason = candidate_evidence(
            _claim("c1", "numerical", "28", ["引用[1]"]), [_SQL_EV, _RAG_EV]
        )
        assert ids == ["ev-tool-sql-1"]
        assert reason == "type_compat_all"

    def test_without_refs_the_rules_still_run(self) -> None:
        ids, _ = candidate_evidence(
            _claim("c1", "numerical", "订单数量 28", []), [_SQL_EV, _RAG_EV]
        )
        assert "ev-tool-sql-1" in ids

    def test_rule4_terminal_fallback_is_non_empty(self) -> None:
        """A rule_based claim whose only pool entry is source-incompatible
        still gets the full pool rather than nothing."""
        ids, reason = candidate_evidence(
            _claim("c1", "rule_based", "完全无关的一句话 甲乙丙丁戊", []), [_SQL_EV]
        )
        assert ids == ["ev-tool-sql-1"]
        assert reason in ("type_compat_all", "no_strict_match", "lexical_similarity")


@pytest.mark.unit
class TestLinkAndVerifyEndToEnd:
    def test_ranking_claim_is_supported_not_starved(self) -> None:
        """Full path: link -> verify_all.  Previously the relational claim
        returned unsupported with ``evidence_ids: []``."""
        claims = [
            _claim("clm-001", "numerical", "Dalton Perez 订单数量为 28", ["无此引用"],
                   value=28, entities=["Dalton Perez"]),
            _claim("clm-002", "relational",
                   "订单数量最多的前 3 个客户是 Dalton Perez、Mason D Roberts 和 Fernando Barnes",
                   ["sources"], entities=_TOP3),
        ]
        pool = {e.evidence_id: e for e in (_SQL_EV, _RAG_EV)}
        link = link_claims_to_evidence(claims, list(pool.values()), max_candidates=10)
        assert link["clm-002"]["evidence_ids"], "relational claim linked to nothing"
        assert link["clm-002"]["match_reason"] != "no_explicit_refs_matched"

        results = verify_all(
            claims, link, pool, VerificationConfig(),
            scorer=LexicalScorer(), llm=None,
        )
        by_id = {r.claim_id: r for r in results}
        assert by_id["clm-002"].supported is True, (
            f"ranking claim unsupported: {by_id['clm-002'].reason}"
        )
        assert by_id["clm-002"].evidence_ids

    def test_no_claim_starves_the_verifier(self) -> None:
        """The observable failure in production: a result carrying
        ``reason="no candidate evidence available"`` and ``evidence_ids=[]``
        for a pool that demonstrably has evidence."""
        claims = [
            _claim("clm-001", "numerical", "Dalton Perez 订单数量为 28", ["无此引用"], value=28),
            _claim("clm-002", "relational", "前三名是这三位客户", ["引用"], entities=_TOP3),
            _claim("clm-003", "factual", "该月销售额为 5235672.00", ["引用"], value=5235672.00),
            _claim("clm-004", "opinion_or_summary", "该月数据可能未完整更新", ["引用"]),
        ]
        pool = {e.evidence_id: e for e in (_SQL_EV, _RAG_EV)}
        link = link_claims_to_evidence(claims, list(pool.values()), max_candidates=10)
        results = verify_all(
            claims, link, pool, VerificationConfig(),
            scorer=LexicalScorer(), llm=None,
        )
        starved = [r for r in results if r.reason == "no candidate evidence available"]
        assert starved == [], f"starved claims: {[r.claim_id for r in starved]}"

    def test_every_claim_type_gets_a_candidate(self) -> None:
        claims = [
            _claim(f"c{i}", t, "订单数量最多的前 3 个客户", ["不存在的引用"])
            for i, t in enumerate(
                ["numerical", "relational", "factual", "opinion_or_summary", "derived", "rule_based"]
            )
        ]
        pool = {e.evidence_id: e for e in (_SQL_EV, _RAG_EV)}
        link = link_claims_to_evidence(claims, list(pool.values()), max_candidates=10)
        for cid, entry in link.items():
            assert entry["evidence_ids"], f"{cid} linked to nothing"

    def test_empty_pool_still_reports_no_evidence(self) -> None:
        link = link_claims_to_evidence([_claim("c1", "factual", "x", [])], [])
        assert link["c1"]["evidence_ids"] == []
        assert link["c1"]["match_reason"] == "no_evidence"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
