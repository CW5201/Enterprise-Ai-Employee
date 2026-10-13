"""Unit tests for src/nodes/sql_execution.py context rendering.

The regression that motivated these: a long ascending time-series result
used to be previewed with a bare ``rows[:5]``, so the answer generator only
ever saw the *oldest* months and answered "最近三个月" with 2022/2023 values.
"""

from __future__ import annotations

import pytest

from src.nodes.sql_execution import _PREVIEW_ALL, _summarise, rewrite_tsql


@pytest.mark.unit
class TestSummarise:
    def test_short_result_shows_every_row(self) -> None:
        payload = {"columns": ["month", "revenue"],
                   "rows": [["2025-04", 5235672.00], ["2025-05", 1908059.49]]}
        out = _summarise("SELECT 1", payload)
        assert "Rows (2 shown)" in out
        assert "2025-04" in out and "2025-05" in out
        assert "elided" not in out

    def test_realistic_monthly_series_is_never_truncated(self) -> None:
        """30 monthly rows is a normal analytics answer and must arrive intact."""
        rows = [[f"2023-{((i - 1) % 12) + 1:02d}-{2022 + i // 12}", i]
                for i in range(1, 31)]
        payload = {"columns": ["month", "revenue"], "rows": rows}
        out = _summarise("SELECT 1", payload)
        assert f"Rows ({len(rows)} shown)" in out
        assert "elided" not in out
        assert rows[0][0] in out and rows[-1][0] in out

    def test_oversized_result_shows_head_and_tail(self) -> None:
        rows = [[f"m{i:02d}", i] for i in range(_PREVIEW_ALL + 10)]
        payload = {"columns": ["month", "revenue"], "rows": rows}
        out = _summarise("SELECT 1", payload)

        assert "first 4 and last 4 shown" in out
        assert "34 middle rows elided" in out
        assert "m00" in out                      # oldest block
        assert "m41" in out                      # most recent rows survive
        assert "m20" not in out                  # elided middle

    def test_tail_carries_the_newest_time_series_values(self) -> None:
        """The exact failure mode: ascending months, question asks for the
        most recent three.  All three newest must survive the preview."""
        rows = [[f"m{i:02d}", i] for i in range(_PREVIEW_ALL + 20)]
        out = _summarise("SELECT 1", {"columns": ["month", "revenue"], "rows": rows})
        for newest in ("m51", "m50", "m49"):
            assert newest in out

    def test_preview_flags_non_contiguity_when_truncated(self) -> None:
        rows = [[i] for i in range(_PREVIEW_ALL + 5)]
        out = _summarise("SELECT 1", {"columns": ["x"], "rows": rows})
        assert "NOT contiguous" in out

    def test_includes_sql_and_columns(self) -> None:
        out = _summarise("SELECT a FROM t", {"columns": ["a"], "rows": [[1]]})
        assert out.startswith("SQL: SELECT a FROM t")
        assert "Columns: a" in out



@pytest.mark.unit
class TestRewriteTsql:
    """``SELECT TOP n`` -> ``LIMIT n``.

    The schema is AdventureWorks (a SQL Server sample) while the engine is
    DuckDB, so the model emits T-SQL's ``TOP``.  DuckDB answers ``syntax error
    at or near "TOP"`` and, since SQLExecutionNode generates once and never
    retries, that single slip cost the task its only data source — the task
    finished with ``completed_no_evidence``.
    """

    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            pytest.param(
                "SELECT TOP 5 CustomerName FROM Sales_Customers WHERE CustomerName LIKE '%Robert%'",
                "SELECT CustomerName FROM Sales_Customers WHERE CustomerName LIKE '%Robert%' LIMIT 5",
                id="basic_top",
            ),
            pytest.param(
                "SELECT TOP 10 * FROM Sales_Orders ORDER BY OrderDate DESC",
                "SELECT * FROM Sales_Orders ORDER BY OrderDate DESC LIMIT 10",
                id="top_with_order_by",
            ),
            pytest.param(
                "SELECT TOP 5 CustomerName FROM Sales_Customers;",
                "SELECT CustomerName FROM Sales_Customers LIMIT 5;",
                id="trailing_semicolon",
            ),
            pytest.param(
                "SELECT DISTINCT TOP 3 a FROM t",
                "SELECT DISTINCT a FROM t LIMIT 3",
                id="distinct",
            ),
        ],
    )
    def test_rewrites(self, sql: str, expected: str) -> None:
        out, rewritten = rewrite_tsql(sql)
        assert rewritten is True
        assert out == expected
        assert "TOP" not in out.split("(")[0]

    @pytest.mark.parametrize(
        "sql",
        [
            pytest.param("SELECT a FROM t", id="no_top"),
            pytest.param("SELECT a FROM t LIMIT 5", id="already_limited"),
            pytest.param("SELECT count(TOP) FROM t", id="top_is_an_identifier"),
            pytest.param(
                "WITH x AS (SELECT TOP 3 a FROM t) SELECT * FROM x",
                id="cta_would_move_the_limit_to_the_wrong_level",
            ),
            pytest.param("SELECT a FROM t WHERE b > 3", id="ordinary_select"),
        ],
    )
    def test_leaves_safe_sql_alone(self, sql: str) -> None:
        out, rewritten = rewrite_tsql(sql)
        assert rewritten is False
        assert out == sql

    def test_no_double_space_left_behind(self) -> None:
        out, _ = rewrite_tsql("SELECT TOP 5 a FROM t")
        assert "  " not in out

    def test_leading_whitespace_is_preserved(self) -> None:
        out, _ = rewrite_tsql("  SELECT TOP 2 a FROM t  ")
        assert out.startswith("  SELECT")
        assert out.endswith("LIMIT 2")

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
