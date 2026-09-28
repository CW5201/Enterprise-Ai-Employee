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

from src.core.config_loader import get_settings
from src.core.exceptions import SQLGuardError
from src.core.observability import observe
from src.core.sql_executor import SQLExecutor, SQLResult
from src.tools.base import ToolResult


class SQLTool:
    """Read-only SQL execution tool (one DuckDB connection per process)."""

    name: str = "sql"

    def __init__(self, executor: SQLExecutor | None = None) -> None:
        self._executor = executor
        self._settings = get_settings().duckdb

    @observe("sql_exec")
    def run(self, *, sql: str, question: str | None = None) -> ToolResult:
        if not sql or not sql.strip():
            return ToolResult(tool_name=self.name, ok=False, error="empty sql")
        if self._executor is None:
            self._executor = SQLExecutor(self._settings)
            self._executor.connect()
        assert self._executor is not None
        try:
            result: SQLResult = self._executor.execute(sql)
        except SQLGuardError as exc:
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error=f"guard rejected: {exc.message}",
                metadata={"sql": sql, "details": exc.details},
            )
        return self._to_result(result, sql, question)

    @staticmethod
    def _to_result(result: SQLResult, sql: str, question: str | None) -> ToolResult:
        return ToolResult(
            tool_name="sql",
            ok=True,
            data=result.to_payload(),
            evidence_ref=f"sql:{sql.strip()}",
            metadata={"question": question, "row_count": result.row_count,
                      "truncated": result.truncated},
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
