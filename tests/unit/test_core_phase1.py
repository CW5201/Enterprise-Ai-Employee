"""Unit tests: SQL guard + config loader + state (Phase 1, offline)."""

from __future__ import annotations

import pytest

from src.core.config_loader import get_routing_rules, get_settings
from src.core.exceptions import SQLGuardError
from src.core.sql_executor import guard_sql
from src.core.state import Evidence, make_state


@pytest.mark.unit
def test_guard_blocks_all_dml_ddl() -> None:
    for bad in (
        "DELETE FROM x",
        "UPDATE x SET a=1",
        "DROP TABLE x",
        "CREATE TABLE x (a INT)",
        "ATTACH 'c.sqlite' AS c",
        "COPY x TO '/tmp/a'",
    ):
        with pytest.raises(SQLGuardError):
            guard_sql(bad)


@pytest.mark.unit
def test_guard_accepts_select_and_with() -> None:
    assert guard_sql("SELECT 1") == "SELECT 1"
    assert guard_sql("WITH x AS (SELECT 1) SELECT * FROM x;") == "WITH x AS (SELECT 1) SELECT * FROM x"


@pytest.mark.unit
def test_settings_load() -> None:
    settings = get_settings()
    assert settings.app["name"] == "enterprise-ai-employee"
    # DuckDB path resolves to an absolute project-root-anchored file.
    assert settings.db_path.is_absolute()
    assert "duckdb" in settings.db_path.name


@pytest.mark.unit
def test_routing_rules_loaded() -> None:
    rules = get_routing_rules()
    assert "knowledge_qa" in rules.intent_tools
    assert "text_to_sql" in rules.intent_tools
    assert "rag" in rules.intent_tools["knowledge_qa"]
    assert "sql" in rules.intent_tools["text_to_sql"]


@pytest.mark.unit
def test_make_state_defaults() -> None:
    state = make_state("question?", request_id="req-1")
    assert state["request_id"] == "req-1"
    assert state["evidence"] == []
    assert state["conversation"] == []
    # Evidence is a pydantic model
    ev = Evidence(evidence_id="e1", source_type="duckdb", source_ref="sql:x")
    assert ev.source_type == "duckdb"
