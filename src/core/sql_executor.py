"""SQL executor - read-only DuckDB execution with a safety guard.

Guarantees (ADR-005):

1. **Read-only connection**: the DuckDB database is opened with
   ``read_only=True`` when the file already exists, so even a leaked DML
   cannot mutate business data.
2. **Statement whitelist**: only ``SELECT`` / ``WITH`` statements pass
   the guard; everything else (DDL, DML, ``ATTACH``, ``INSTALL``, ...)
   is rejected with :class:`SQLGuardError` BEFORE execution.
3. **Row cap**: results are truncated to ``max_rows`` (default 1000) to
   keep the evidence payload bounded.

The executor is stateless and thread-confined to one process; the graph
creates one executor per run via :func:`open_database`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from src.core.config_loader import DuckDBSettings, get_settings
from src.core.exceptions import SQLExecutionError, SQLGuardError
from src.core.observability import observe

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# statements allowed to enter execution (everything else is rejected)
_ALLOWED_LEADING = re.compile(r"^\s*(with|select)\b", re.IGNORECASE)
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|call|export|install|load|vacuum|backup|restore)\b",
    re.IGNORECASE,
)


@dataclass
class SQLResult:
    """Executed result set, already shaped for the evidence model."""

    columns: list[str]
    rows: list[Any]
    row_count: int
    truncated: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
        }

    @property
    def empty(self) -> bool:
        return not self.rows


def guard_sql(sql: str, *, allow_dml: bool = False, allow_ddl: bool = False) -> str:
    """Validate a statement against the read-only policy.  Returns stripped SQL."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise SQLGuardError("Empty SQL statement")
    if not _ALLOWED_LEADING.match(stripped):
        first_word = stripped.split(None, 1)[0].lower()
        raise SQLGuardError(
            f"Statement must start with SELECT or WITH (got: {first_word!r})",
        )
    forbidden = _FORBIDDEN_KEYWORDS.findall(stripped)
    if forbidden:
        raise SQLGuardError(
            "Statement contains forbidden keyword(s)",
            details={"keywords": sorted({k.upper() for k in forbidden})},
        )
    if not allow_dml or not allow_ddl:
        # covered by the keyword check above; explicit for clarity
        pass
    return stripped


class SQLExecutor:
    """Thin, safe wrapper around a DuckDB connection."""

    def __init__(self, settings: DuckDBSettings | None = None) -> None:
        self.settings = settings or get_settings().duckdb
        self._connection: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> None:
        from pathlib import Path

        path = self.settings.path
        db = Path(path)
        if not db.is_absolute():
            db = _PROJECT_ROOT / db
        db.parent.mkdir(parents=True, exist_ok=True)
        read_only = self.settings.read_only and db.exists()
        self._connection = duckdb.connect(str(db), read_only=read_only)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> SQLExecutor:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @observe("sql_exec")
    def execute(self, sql: str) -> SQLResult:
        """Guard + execute + shape.  Raises SQLGuardError / SQLExecutionError."""
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        safe_sql = guard_sql(sql, allow_dml=self.settings.allow_dml, allow_ddl=self.settings.allow_ddl)
        try:
            cursor = self._connection.execute(safe_sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            max_rows = self.settings.max_rows
            rows = cursor.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            if truncated:
                rows = rows[:max_rows]
            return SQLResult(
                columns=columns,
                rows=[tuple(r) for r in rows],
                row_count=len(rows),
                truncated=truncated,
            )
        except SQLGuardError:
            raise
        except duckdb.Error as exc:
            raise SQLExecutionError(
                f"SQL execution failed: {exc}",
                details={"sql": safe_sql},
            ) from exc

    def list_tables(self) -> list[str]:
        """Return user table names (for schema introspection / tests)."""
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        result = self._connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        return [r[0] for r in result]


def open_database(settings: DuckDBSettings | None = None) -> SQLExecutor:
    """Create and connect an executor (use as context manager)."""
    executor = SQLExecutor(settings)
    executor.connect()
    return executor


__all__ = ["SQLExecutor", "SQLResult", "guard_sql", "open_database"]
