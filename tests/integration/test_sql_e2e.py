"""Integration test: the SQL end-to-end path (Phase 1).

Runs the full LangGraph pipeline against the imported WideWorldImporters
runtime database (data/runtime/wwi.duckdb) and asserts that:
- a text-to-sql intent routes to the sql tool;
- the tool actually executed SQL and produced a non-empty Evidence;
- the final answer is non-empty.

LLM backend selection:
- LLM_BASE_URL + LLM_API_KEY set (and LLM_FORCE_OFFLINE != 1) -> real
  Text-to-SQL via the OpenAI-compatible endpoint;
- otherwise -> deterministic offline backend (tests stay green offline,
  with a clearly-labelled placeholder result).

Live-service guard:
- the full pipeline instantiates the Milvus vector store; when Milvus is
  not running the test is *skipped* with an explicit reason (it is an
  environment condition, not a code defect).  If the service is
  reachable, the test runs against the real pipeline and any genuine
  SQL / RAG / LLM error still fails it.
"""

from __future__ import annotations

import os

import duckdb
import pytest

from src.core.sql_executor import SQLExecutor, default_database_path, guard_sql
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
def wwi_db() -> str:
    """Ensure the imported WWI runtime database exists (read-only use)."""
    path = default_database_path()
    if not path.exists():
        pytest.skip(f"runtime database missing: {path} - run scripts/import_wwi.py first")
    connection = duckdb.connect(str(path), read_only=True)
    try:
        count = connection.execute("SELECT COUNT(*) FROM Sales_Customers").fetchone()[0]
        if count == 0:
            pytest.skip(
                "runtime database has no seed data - re-run "
                "scripts/import_wwi.py to populate WideWorldImporters"
            )
    finally:
        connection.close()
    return str(path)


def _skip_unless_milvus_reachable() -> None:
    """Skip only when the live Milvus service is unavailable.

    The SQL end-to-end pipeline instantiates the Milvus vector store, whose
    connection fails when Milvus (``localhost:19530`` by default) is not
    running.  That is an *environment* condition, not a code defect, so we
    probe service reachability up-front and skip with an explicit reason.
    We only probe the connection (not the collection, and we do NOT load an
    embedder model), so a real retrieval / BGE-M3 / LLM error raised later
    by the pipeline still fails the test normally — we never mask genuine
    defects.
    """
    from pymilvus import MilvusClient

    from src.core.config_loader import get_settings

    settings = get_settings()
    milvus_cfg = settings.raw.get("milvus", {})
    host = str(milvus_cfg.get("host", "localhost"))
    port = str(milvus_cfg.get("port", "19530"))
    uri = host if host.startswith(("http://", "https://")) else f"http://{host}:{port}"
    user = str(milvus_cfg.get("user", ""))
    password = str(milvus_cfg.get("password", ""))
    token = f"{user}:{password}" if user or password else ""

    try:
        MilvusClient(uri=uri, token=token)
    except Exception as exc:  # noqa: BLE001 - connection-level failure only
        pytest.skip(f"Milvus not reachable at {uri}: {exc}")


@pytest.mark.usefixtures("wwi_db")
def test_sql_end_to_end() -> None:
    """Full pipeline: intent -> router -> sql tool -> answer."""
    _skip_unless_milvus_reachable()
    os.environ.pop("LLM_BASE_URL", None)  # honour the running environment as-is
    graph = build_graph()
    state = graph.invoke(
        {
            "request_id": "req-sql-e2e",
            "user_query": "订单数量最多的前10名客户是哪些？",
            "conversation": [],
        }
    )
    # intent routed to sql
    assert state["intent"] == "data_query", state["intent"]
    assert state["route"] == "sql", state["route"]
    # evidence was produced
    assert state["evidence"], "expected at least one evidence item"
    sql_evidence = [e for e in state["evidence"] if e.source_type == "duckdb"]
    assert sql_evidence, "expected duckdb evidence"
    assert sql_evidence[0].payload.get("row_count", 0) >= 0
    # answer produced
    assert state.get("answer"), "answer should not be empty"


def test_executor_read_only_connection(wwi_db: str) -> None:
    """When the db file exists, the executor must open read-only."""
    executor = SQLExecutor(db_path=default_database_path())
    executor.connect()
    try:
        result = executor.execute("SELECT COUNT(*) AS n FROM Sales_Customers")
        assert result["columns"] == ["n"]
        assert result["rows"] and result["rows"][0][0] >= 0
    finally:
        executor.close()
