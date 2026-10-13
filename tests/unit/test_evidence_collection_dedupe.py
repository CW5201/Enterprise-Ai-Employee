"""Tests for src/nodes/evidence_collection.py deduplication.

The same SQL result reaches the collector through two channels that spell its
``source_ref`` differently:

* ``sql_execution`` appends an ``EvidenceItem`` with ``source_ref =
  "duckdb:{sql}"``;
* ``adapt_sql`` (the ``tool_results`` / ``sql_result`` channels) builds
  ``source_ref = "sql:{sql}"``.

Matching on the raw ref meant both survived, so a live run linked one claim to
``ev-tool-sql-1`` *and* ``ev-req-…-sql`` for a single query — the same rows
counted twice as candidates, and the claim total inflated for no reason.
"""

from __future__ import annotations

import pytest

from src.nodes.evidence_collection import _dedupe_key, collect_evidence


def _item(source_ref: str, evidence_id: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "source_type": "duckdb",
        "source_ref": source_ref,
        "content": "SQL: SELECT CustomerName FROM Sales_Customers LIMIT 5",
        "payload": {"sql": "SELECT CustomerName FROM Sales_Customers LIMIT 5",
                    "columns": ["CustomerName"], "rows": [["Robert Brown"]],
                    "row_count": 1},
        "metadata": {"intent": "data_query", "rationale": ""},
    }


_SQL = "SELECT CustomerName FROM Sales_Customers LIMIT 5"


@pytest.mark.unit
class TestDedupeKey:
    def test_duckdb_and_sql_aliases_collapse(self) -> None:
        class _E:
            source_ref = f"duckdb:{_SQL}"
            evidence_id = "ev-req-1-sql"

        class _E2:
            source_ref = f"sql:{_SQL}"
            evidence_id = "ev-tool-sql-1"

        assert _dedupe_key(_E()) == _dedupe_key(_E2())  # type: ignore[arg-type]

    def test_other_prefixes_are_left_alone(self) -> None:
        class _E:
            source_ref = "milvus:chunk-7"
            evidence_id = "ev-rag-7"

        assert _dedupe_key(_E()) == "milvus:chunk-7"  # type: ignore[arg-type]

    def test_falls_back_to_the_id(self) -> None:
        class _E:
            source_ref = ""
            evidence_id = "ev-analysis-sum"

        assert _dedupe_key(_E()) == "ev-analysis-sum"  # type: ignore[arg-type]


@pytest.mark.unit
class TestCollectEvidenceDedup:
    def test_one_query_yields_one_evidence_record(self) -> None:
        state = {
            "tool_results": [
                {"tool": "sql", "success": True, "step": 1, "sql": _SQL,
                 "data": {"columns": ["CustomerName"], "rows": [["Robert Brown"]]}},
            ],
            "evidence": [_item(f"duckdb:{_SQL}", "ev-req-1-sql")],
            "sql_result": None,
        }
        out = collect_evidence(state)  # type: ignore[arg-type]
        assert len(out) == 1, f"got {[e.evidence_id for e in out]}"
        assert out[0].source_type == "sql"

    def test_distinct_queries_stay_distinct(self) -> None:
        other = "SELECT COUNT(*) FROM Sales_Orders"
        state = {
            "tool_results": [
                {"tool": "sql", "success": True, "step": 1, "sql": _SQL,
                 "data": {"columns": ["a"], "rows": [["x"]]}},
                {"tool": "sql", "success": True, "step": 2, "sql": other,
                 "data": {"columns": ["b"], "rows": [[1]]}},
            ],
            "evidence": [_item(f"duckdb:{_SQL}", "ev-req-1-sql")],
        }
        out = collect_evidence(state)  # type: ignore[arg-type]
        assert len(out) == 2

    def test_failed_tool_results_are_skipped(self) -> None:
        state = {
            "tool_results": [
                {"tool": "sql", "success": False, "step": 1, "sql": _SQL, "data": {}},
            ],
            "evidence": [],
        }
        assert collect_evidence(state) == []  # type: ignore[arg-type]

    def test_legacy_evidence_channel_contributes_when_no_tool_result(self) -> None:
        state = {"tool_results": [], "evidence": [_item(f"duckdb:{_SQL}", "ev-req-1-sql")]}
        out = collect_evidence(state)  # type: ignore[arg-type]
        assert len(out) == 1
        assert out[0].source_type == "sql"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
