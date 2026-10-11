"""Tests for the SQL evidence block handed to the answer generator.

``_render_sql_block`` used to slice ``rows[:10]``.  For an ascending monthly
series that showed only the oldest months, so the model either reported those
as "最近三个月" or invented the newest ones (observed: evidence held the
correct 2025-04/05/06 figures and the answer claimed "2024-03/04/05：未知").
"""

from __future__ import annotations

import pytest

from src.nodes.answer_generation import _render_sql_block


class _FakeResult:
    def __init__(self, columns: list[str], rows: list[tuple]) -> None:
        self.columns = columns
        self.rows = rows
        self.row_count = len(rows)


@pytest.mark.unit
class TestRenderSqlBlock:
    def test_no_result(self) -> None:
        assert _render_sql_block(None) == "(无 SQL 结果)"

    def test_empty_rows(self) -> None:
        out = _render_sql_block(_FakeResult(["a"], []))
        assert "0 行" in out

    def test_realistic_monthly_series_is_shown_in_full(self) -> None:
        """30 monthly rows is a normal answer and must reach the model intact."""
        rows = [(f"2023-{((i - 1) % 12) + 1:02d}-{2022 + i // 12}", i * 1000)
                for i in range(1, 31)]
        out = _render_sql_block(_FakeResult(["month", "revenue"], rows))
        assert rows[0][0] in out          # oldest month visible
        assert rows[-1][0] in out         # newest month visible
        assert "省略" not in out

    def test_oversized_result_shows_head_and_tail(self) -> None:
        rows = [(f"m{i:02d}", i) for i in range(40)]
        out = _render_sql_block(_FakeResult(["month", "revenue"], rows))
        assert "m00" in out and "m03" in out      # head
        assert "m36" in out and "m39" in out      # tail (newest)
        assert "m20" not in out                   # elided middle
        assert "省略中间" in out

    def test_newest_values_survive_for_recent_questions(self) -> None:
        """The failure mode: '最近三个月' must still be answerable."""
        rows = [(f"m{i:02d}", i) for i in range(50)]
        out = _render_sql_block(_FakeResult(["month", "revenue"], rows))
        for newest in ("m47", "m48", "m49"):
            assert newest in out

    def test_includes_column_headers(self) -> None:
        out = _render_sql_block(_FakeResult(["month", "revenue"], [("2025-06", 47491)]))
        assert out.startswith("month | revenue")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
