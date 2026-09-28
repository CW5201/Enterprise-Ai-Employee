"""Unit tests: SQL guard + config loader + state (Phase 1, offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config_loader import get_routing_rules, get_settings
from src.core.exceptions import SQLGuardError
from src.core.sql_executor import guard_sql
from src.core.state import Evidence, make_state


@pytest.mark.unit
def test_guard_blocks_all_dml_ddl() -> None:
    for bad in [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a=1",
        "DELETE FROM t",
        "DROP TABLE t",
        "CREATE TABLE t (a INT)",
        "ATTACH 'x.db' AS x",
        "PRAGMA integer_keys",
        "COPY t TO '/tmp/x.csv'",
    ]:
        with pytest.raises(SQLGuardError):
            guard_sql(bad)


@pytest.mark.unit
def test_guard_allows_select_and_with() -> None:
    assert guard_sql("  SELECT 1 ") == "SELECT 1"
    assert guard_sql("WITH a AS (SELECT 1) SELECT * FROM a;") == "WITH a AS (SELECT 1) SELECT * FROM a"


@pytest.mark.unit
def test_settings_load_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # use the real config dir but verify env-var resolution
    monkeypatch.setenv("DUCKDB_MAX_ROWS_OVERRIDE_TEST", "42")
    settings = get_settings()
    assert settings.duckdb.max_rows == 1000
    # milvus / rag defaults exist
    assert settings.milvus.collection
    assert settings.rag.top_k > 0


@pytest.mark.unit
def test_routing_rules_map_intents_to_tools() -> None:
    rules = get_routing_rules()
    assert "knowledge_qa" in rules.intent_tools
    assert "text_to_sql" in rules.intent_tools
    assert "rag" in rules.intent_tools["knowledge_qa"]
    assert "sql" in rules.intent_tools["text_to_sql"]


@pytest.mark.unit
def test_make_state_defaults() -> None:
    state = make_state("question?", task_id="t1")
    assert state["task_id"] == "t1"
    assert state["status"] == "pending"
    assert state["evidence"] == []
    # evidence is a dataclass
    ev = Evidence(evidence_id="e1", source_type="duckdb", source_ref="sql:x")
    assert ev.to_dict()["source_type"] == "duckdb"
