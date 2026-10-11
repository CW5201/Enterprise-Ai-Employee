"""Tool: SQL - read-only DuckDB query over business data.

The SQL tool is a thin, safe wrapper around :class:`src.core.sql_executor`:

- it takes a **generated SQL statement** (produced by the LLM node) or a
  raw question (Phase 1 keeps the tool simple: callers pass ``sql``);
- it always executes through the guard (SELECT/WITH only, row cap);
- it returns a :class:`ToolResult` whose ``data`` is a dict with
  ``columns / rows / row_count / truncated`` and whose ``evidence_ref``
  carries the executed SQL so the answer node can cite it.

The **Text-to-SQL generation** itself (natural language -> SQL) lives in
``src/nodes/sql_execution.py``.  Phase 1 keeps that node minimal: it uses
the configured LLM client, falls back to an empty SQL on LLM failure and
records the failure in ``state.errors``.
"""

from __future__ import annotations

from src.core.exceptions import SQLExecutionError, SQLGuardError
from src.core.observability import observe
from src.core.sql_executor import SQLExecutor, default_database_path
from src.tools.base import ToolResult


class SQLTool:
    """Read-only SQL execution tool (one DuckDB connection per process)."""

    name: str = "sql"

    def __init__(self, executor: SQLExecutor | None = None) -> None:
        self._executor = executor

    @observe("sql_exec")
    def run(self, *, sql: str, question: str | None = None) -> ToolResult:
        if not sql or not sql.strip():
            return ToolResult(tool_name=self.name, ok=False, error="empty sql")
        if self._executor is None:
            # Same database the SQL execution node uses (data/runtime/wwi.duckdb).
            self._executor = SQLExecutor(db_path=default_database_path())
        executor = self._executor
        try:
            payload = executor.execute(sql)
        except SQLGuardError as exc:
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=f"guard rejected: {exc.message}",
                metadata={"sql": sql, "details": exc.details},
            )
        except SQLExecutionError as exc:
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=exc.message,
                metadata={"sql": sql, "details": exc.details},
            )
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=payload,
            evidence_ref=f"sql:{sql.strip()}",
            metadata={
                "question": question,
                "row_count": payload.get("row_count", 0),
                "truncated": payload.get("truncated", False),
            },
        )

    @property
    def columns_available(self) -> list[str] | None:
        """Best-effort schema hint for prompt building (None if no db yet)."""
        if self._executor is None:
            return None
        try:
            return self._executor.list_tables()
        except Exception:  # noqa: BLE001
            return None


__all__ = ["SQLTool"]
