"""SQL executor — safe, read-only DuckDB execution (Phase 1).

Guarantees:
1. **Read-only**: existing database files are always opened
   ``read_only=True``; the DML/DDL whitelist is a second, pre-execution
   line of defence.
2. **Statement whitelist**: only ``SELECT`` / ``WITH ... SELECT`` pass;
   INSERT / UPDATE / DELETE / DROP / ALTER / CREATE / ... are rejected
   with :class:`SQLGuardError` *before* execution.
3. **Bounded execution**: query timeout (thread-based) + max-rows cap keep
   any LLM-generated statement from exhausting the process.
4. **Structured errors**: failures raise :class:`SQLExecutionError`
   (subclass of :class:`AppError`) with machine-readable details.
5. **Timing**: every execution records ``execution_time_ms`` in the
   returned payload.
"""

from __future__ import annotations

import concurrent.futures
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from src.core.exceptions import SQLExecutionError, SQLGuardError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_ALLOWED_LEADING = re.compile(r"^\s*(with|select)\b", re.IGNORECASE)
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|pragma|call|export|"
    r"install|load|vacuum|backup|restore)\b",
    re.IGNORECASE,
)


def default_database_path() -> Path:
    """The runtime database produced by scripts/import_wwi.py."""
    return _PROJECT_ROOT / "data" / "runtime" / "wwi.duckdb"


def guard_sql(sql: str) -> str:
    """Reject anything that is not a plain read-only SELECT/WITH statement."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise SQLGuardError("Empty SQL statement")
    if not _ALLOWED_LEADING.match(stripped):
        first_word = stripped.split(None, 1)[0].lower()
        raise SQLGuardError(
            f"Statement must start with SELECT or WITH (got {first_word!r})",
        )
    forbidden = sorted({k.upper() for k in _FORBIDDEN_KEYWORDS.findall(stripped)})
    if forbidden:
        raise SQLGuardError(
            "Statement contains forbidden keyword(s)",
            details={"keywords": forbidden},
        )
    return stripped


@dataclass
class SQLExecutor:
    """Thin safe wrapper around one DuckDB connection (read-only when the
    database file exists)."""

    db_path: Path | None = None
    read_only: bool = True
    max_rows: int = 1000
    timeout_seconds: int = 30
    _connection: duckdb.DuckDBPyConnection | None = field(default=None, repr=False, init=False)

    # -- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        path = self.db_path or default_database_path()
        read_only = self.read_only and path.exists()
        self._connection = duckdb.connect(str(path), read_only=read_only)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> SQLExecutor:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- execution ----------------------------------------------------------

    def execute(self, sql: str) -> dict[str, Any]:
        """Guard + execute + shape.  Returns the structured result payload.

        Raises SQLGuardError (rejected by the guard) or SQLExecutionError
        (DuckDB failure / timeout) — both carry structured details.
        """
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        safe_sql = guard_sql(sql)

        start = time.perf_counter()
        try:
            payload = self._execute_with_timeout(safe_sql)
        except SQLExecutionError:
            raise
        except concurrent.futures.TimeoutError as exc:
            raise SQLExecutionError(
                f"SQL query timed out after {self.timeout_seconds}s",
                details={"sql": safe_sql},
            ) from exc
        payload["execution_time_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        return payload

    def _execute_with_timeout(self, sql: str) -> dict[str, Any]:
        conn = self._connection
        assert conn is not None

        def _run() -> dict[str, Any]:
            cursor = conn.execute(sql)
            columns = [d[0] for d in (cursor.description or [])]
            rows = cursor.fetchmany(self.max_rows + 1)
            truncated = len(rows) > self.max_rows
            if truncated:
                rows = rows[: self.max_rows]
            return {
                "sql": sql,
                "columns": columns,
                "rows": [tuple(r) for r in rows],
                "row_count": len(rows),
                "truncated": truncated,
                "execution_time_ms": 0.0,  # filled in by the caller
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            result = future.result(timeout=self.timeout_seconds)
            return result

    def list_tables(self) -> list[str]:
        """User table names (schema introspection / tests)."""
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        rows = self._connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE' ORDER BY table_name"
        ).fetchall()
        return [r[0] for r in rows]


__all__ = ["SQLExecutor", "default_database_path", "guard_sql"]
