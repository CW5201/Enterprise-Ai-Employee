"""Security and robustness tests for src/core/sql_executor.py.

Covers the guard bypass and the executor defects found in audit:
- multi-statement payloads (``SELECT 1; SET threads=1``) mutate DuckDB
  process configuration even though only the leading keyword was checked
- file/table functions turn a read-only analytics query into arbitrary
  local file disclosure
- ``read_only`` used to silently degrade to read-write when the database
  file was missing, creating an empty file and poisoning every other
  read-only connection in the process
- a raw ``duckdb`` exception from ``connect()`` escaped every caller
- the ThreadPoolExecutor timeout was not real (``shutdown(wait=True)``
  blocked until the runaway query finished)
"""

from __future__ import annotations

import time

import pytest

from src.core.exceptions import SQLExecutionError, SQLGuardError
from src.core.sql_executor import SQLExecutor, guard_sql


@pytest.mark.unit
class TestGuardRejects:
    @pytest.mark.parametrize("sql", [
        "SELECT 1 AS a; SET threads=1",
        "SELECT 1; SELECT 2",
        "SELECT 1 AS a; DROP TABLE t",
    ])
    def test_multi_statement_is_rejected(self, sql: str) -> None:
        """DuckDB executes every statement and returns the last result, so
        a keyword check on the leading token alone is not a defence."""
        with pytest.raises(SQLGuardError) as e:
            guard_sql(sql)
        assert "one statement" in str(e.value.message)

    @pytest.mark.parametrize("sql", [
        "SELECT * FROM read_csv_auto('C:/Windows/win.ini')",
        "SELECT read_text('/etc/passwd')",
        "SELECT * FROM read_parquet('/tmp/x.parquet')",
        "SELECT * FROM glob('C:/Windows/*')",
        "SELECT * FROM sqlite_scan('x.db', 't')",
    ])
    def test_file_table_functions_are_rejected(self, sql: str) -> None:
        """These read arbitrary local files from inside a SELECT."""
        with pytest.raises(SQLGuardError) as e:
            guard_sql(sql)
        assert "table function" in str(e.value.message)

    @pytest.mark.parametrize("sql", [
        "SET threads=1",
        "RESET memory_limit",
        "INSERT INTO t VALUES (1)",
        "DROP TABLE t",
        "CHECKPOINT",
    ])
    def test_non_select_is_rejected(self, sql: str) -> None:
        with pytest.raises(SQLGuardError):
            guard_sql(sql)

    def test_empty_is_rejected(self) -> None:
        with pytest.raises(SQLGuardError):
            guard_sql("   ")

    def test_garbage_is_rejected(self) -> None:
        with pytest.raises(SQLGuardError):
            guard_sql("DELETE FROM t")


@pytest.mark.unit
class TestGuardAccepts:
    @pytest.mark.parametrize("sql", [
        "SELECT 1 AS a",
        "SELECT 1 AS a;",  # trailing punctuation is harmless
        "WITH mx AS (SELECT MAX(OrderDate) AS d FROM Sales_Orders) "
        "SELECT SUM(ol.Quantity*ol.UnitPrice) AS revenue "
        "FROM Sales_Orders o JOIN Sales_OrderLines ol ON ol.OrderID=o.OrderID, mx "
        "WHERE o.OrderDate >= mx.d - INTERVAL 3 MONTH",
        "SELECT strftime(o.OrderDate, '%Y-%m') AS month, "
        "SUM(ol.Quantity*ol.UnitPrice) AS revenue "
        "FROM Sales_Orders o JOIN Sales_OrderLines ol ON ol.OrderID=o.OrderID "
        "WHERE o.OrderDate >= (SELECT MAX(OrderDate) - INTERVAL 3 MONTH FROM Sales_Orders) "
        "GROUP BY 1 ORDER BY 1",
    ])
    def test_legitimate_analytics_sql_passes(self, sql: str) -> None:
        assert isinstance(guard_sql(sql), str)

    def test_string_literal_is_returned_unchanged(self) -> None:
        out = guard_sql("SELECT 1 AS a  -- trailing comment-ish")
        assert out.startswith("SELECT")


@pytest.mark.unit
class TestExecutorRobustness:
    def test_missing_database_raises_app_error_not_raw_duckdb(self, tmp_path) -> None:
        ex = SQLExecutor(db_path=tmp_path / "nope.duckdb", read_only=True)
        with pytest.raises(SQLExecutionError) as e:
            ex.connect()
        assert "not found" in str(e.value.message)

    def test_read_only_never_degrades_to_read_write(self, tmp_path) -> None:
        """Silent degradation created an empty database file and made every
        later read_only open in the process fail with a configuration clash."""
        path = tmp_path / "missing.duckdb"
        ex = SQLExecutor(db_path=path, read_only=True)
        with pytest.raises(SQLExecutionError):
            ex.connect()
        assert not path.exists(), "read_only connect must not create the file"

    def test_connect_error_is_wrapped(self, tmp_path) -> None:
        # A directory that cannot become a database file.
        ex = SQLExecutor(db_path=tmp_path, read_only=False)
        with pytest.raises(SQLExecutionError):
            ex.connect()

    def test_execute_reports_the_statement_it_failed_on(self) -> None:
        ex = SQLExecutor()
        with pytest.raises(SQLExecutionError) as e:
            ex.execute("SELECT * FROM definitely_not_a_table")
        assert "sql" in e.value.details
        assert "definitely_not_a_table" in e.value.details["sql"]

    def test_timeout_is_real_not_a_blocking_shutdown(self) -> None:
        """The ThreadPoolExecutor variant blocked ~11s past a 1s timeout
        because ``with`` calls ``shutdown(wait=True)``."""
        ex = SQLExecutor(timeout_seconds=2)
        ex.connect()
        try:
            start = time.perf_counter()
            with pytest.raises(SQLExecutionError) as e:
                ex.execute(
                    "SELECT COUNT(*) FROM (SELECT a.OrderID FROM "
                    "Sales_Orders a, Sales_Orders b, Sales_Orders c)"
                )
            elapsed = time.perf_counter() - start
            assert "timed out" in str(e.value.message)
            assert elapsed < 8, f"timeout took {elapsed:.1f}s; interrupt() not effective"
        finally:
            ex.close()

    def test_concurrent_execute_serializes_on_one_connection(self) -> None:
        """One DuckDB connection is not safe from several worker threads."""
        import threading

        ex = SQLExecutor()
        errors: list[BaseException] = []

        def _worker() -> None:
            try:
                for _ in range(5):
                    ex.execute("SELECT 1 AS a")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
            assert not errors, f"concurrent execute raised: {errors[:3]}"
        finally:
            ex.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
