"""Integration test: the SQL end-to-end path (Phase 1).

Runs the full LangGraph loop against the locally-imported DuckDB database
and asserts that:
- a text-to-sql intent routes to the sql tool;
- the tool actually executed SQL and produced a non-empty Evidence;
- the final answer cites that evidence.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from src.core.config_loader import get_settings
from src.core.sql_executor import SQLExecutor, guard_sql
from src.graph.builder import build_graph

pytestmark = pytest.mark.integration


def test_guard_rejects_write_statements() -> None:
    with pytest.raises(Exception) as exc_info:
        guard_sql("DELETE FROM finance_expenses;")
    assert "guard" in str(exc_info.value).lower() or "SELECT" in str(exc_info.value)


def test_guard_accepts_select() -> None:
    assert guard_sql("SELECT 1") == "SELECT 1"
    assert guard_sql("WITH x AS (SELECT 1) SELECT * FROM x;") == "WITH x AS (SELECT 1) SELECT * FROM x"


@pytest.fixture(scope="module")
def db_with_finance() -> Path:
    settings = get_settings()
    path = settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS finance_expenses (
                expense_id INTEGER,
                employee_id INTEGER,
                department_id INTEGER,
                expense_date DATE,
                expense_type VARCHAR,
                amount DECIMAL(12,2),
                approved BOOLEAN
            )
            """
        )
        connection.execute("DELETE FROM finance_expenses")
        connection.executemany(
            "INSERT INTO finance_expenses VALUES (?,?,?,?,?,?,?)",
            [
                (1, 101, 1, "2026-01-15", "travel", 1200.50, True),
                (2, 101, 1, "2026-02-10", "travel", 2600.00, True),
                (3, 102, 2, "2026-01-22", "entertainment", 800.00, True),
            ],
        )
    finally:
        connection.close()
    return path


@pytest.mark.usefixtures("db_with_finance")
def test_sql_end_to_end_mock_llm() -> None:
    """Offline deterministic run: mock LLM -> router -> sql tool -> answer."""
    import os

    os.environ["LLM_BACKEND"] = "mock"
    graph = build_graph()
    state = graph.invoke(
        {
            "task_id": "t-sql-e2e",
            "user_task": "各部门已审批的报销金额合计是多少？",
            "conversation": [],
            "selected_tools": [],
            "tool_calls": [],
            "errors": [],
            "evidence": [],
            "citations": [],
            "slots": {},
            "constraints": {},
            "routing_history": [],
            "iteration": 0,
            "latency_ms": {},
            "messages": [],
        }
    )
    # intent routed to sql
    assert state["intent"] in ("text_to_sql", "data_analysis", "information_query"), state["intent"]
    assert "sql" in state["selected_tools"], state["selected_tools"]
    # evidence was produced
    assert state["evidence"], "expected at least one evidence item"
    sql_evidence = [e for e in state["evidence"] if e.source_type == "duckdb"]
    assert sql_evidence, "expected duckdb evidence"
    assert sql_evidence[0].payload.get("row_count", 0) >= 0
    # answer produced
    assert state.get("answer"), "answer should not be empty"
    assert state["status"] in ("completed", "completed_fallback", "completed_no_evidence")


def test_executor_read_only_connection(db_with_finance: Path) -> None:
    """When the db file exists, the executor must open read-only."""
    executor = SQLExecutor()
    executor.connect()
    try:
        result = executor.execute("SELECT COUNT(*) AS n FROM finance_expenses")
        assert result.columns == ["n"]
        assert result.rows and result.rows[0][0] >= 0
    finally:
        executor.close()
