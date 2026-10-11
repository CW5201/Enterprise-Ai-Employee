"""Unit tests for src/tools/sql_tool.py.

The module used to import a nonexistent ``SQLResult`` from
``src.core.sql_executor`` and then call ``SQLExecutor(settings)`` /
``result.to_payload()`` / ``result.row_count`` — none of which exist, so
``import src.tools.sql_tool`` raised ImportError and the whole tool-registry
path (``default_registry`` -> ``build_registry`` -> ``ToolInvocationNode``)
was dead on arrival.  ``SQLExecutor.execute()`` returns a plain dict payload.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.sql_executor import SQLExecutor
from src.tools.sql_tool import SQLTool


@pytest.fixture()
def tool() -> SQLTool:
    """An SQLTool bound to an in-memory DuckDB (never touches data/)."""
    executor = SQLExecutor(db_path=Path(":memory:"), read_only=False)
    executor.connect()
    return SQLTool(executor=executor)


@pytest.mark.unit
def test_module_imports(tool: SQLTool) -> None:
    """The ImportError regression guard: the module must import at all."""
    from src.tools import sql_tool

    assert "SQLTool" in sql_tool.__all__


@pytest.mark.unit
def test_run_executes_and_returns_a_dict_payload(tool: SQLTool) -> None:
    """execute() returns a dict, not a SQLResult object with .to_payload()."""
    result = tool.run(sql="SELECT 1 AS a, 'x' AS b", question="q")

    assert result.ok, result.error
    assert result.data["columns"] == ["a", "b"]
    assert result.data["rows"] == [(1, "x")]
    assert result.data["row_count"] == 1
    assert result.data["truncated"] is False
    assert result.evidence_ref == "sql:SELECT 1 AS a, 'x' AS b"
    assert result.metadata["row_count"] == 1


@pytest.mark.unit
def test_run_rejects_empty_sql(tool: SQLTool) -> None:
    result = tool.run(sql="   ")
    assert not result.ok
    assert result.error == "empty sql"


@pytest.mark.unit
def test_run_surfaces_guard_rejection(tool: SQLTool) -> None:
    result = tool.run(sql="DROP TABLE customers")
    assert not result.ok
    assert result.error is not None and "guard rejected" in result.error


@pytest.mark.unit
def test_run_surfaces_execution_error(tool: SQLTool) -> None:
    result = tool.run(sql="SELECT * FROM table_that_does_not_exist")
    assert not result.ok
    assert result.error is not None
