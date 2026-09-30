"""Unit tests for src/core/neo4j_client.py (no live Neo4j required).

Covers:
- environment-variable credential resolution
- read-only statement classification (what passes / what is rejected)
- exception mapping (credential-safe, no URI leakage)
- result shaping

Integration tests that require a live server live in
tests/integration/test_neo4j_client.py and are skipped when
``NEO4J_INTEGRATION`` is not set.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.core.neo4j_client import (
    Neo4jClient,
    Neo4jQueryError,
    Neo4jReadOnlyViolation,
    Neo4jResult,
    Neo4jUnavailableError,
    _classify_for_readonly,
    _diagnose,
)

# ---------------------------------------------------------------------------
# Credential / env resolution
# ---------------------------------------------------------------------------


class TestClientConstruction:
    def test_defaults(self) -> None:
        client = Neo4jClient()
        assert client.uri == "bolt://localhost:7687"
        assert client.user == "neo4j"
        assert client.database == "neo4j"

    @patch.dict(os.environ, {"NEO4J_URI": "bolt://db:7687", "NEO4J_USERNAME": "svc"})
    def test_env_override(self) -> None:
        client = Neo4jClient()
        assert client.uri == "bolt://db:7687"
        assert client.user == "svc"

    def test_explicit_values_win(self) -> None:
        client = Neo4jClient(uri="bolt://x:7687", user="a", password="p", database="d")
        assert client.uri == "bolt://x:7687"
        assert client.user == "a"
        assert client.password == "p"
        assert client.database == "d"


# ---------------------------------------------------------------------------
# Read-only classification
# ---------------------------------------------------------------------------


class TestReadOnlyClassification:
    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (c:Customer) RETURN c",
            "MATCH (o:Order)-[:INVOICED]->(i:Invoice) WHERE i.order_id = $id RETURN i",
            "WITH 1 AS x RETURN x",
            "RETURN 1",
        ],
    )
    def test_allowed_statements_pass(self, cypher: str) -> None:
        _classify_for_readonly(cypher)  # must not raise

    @pytest.mark.parametrize(
        "cypher",
        [
            "CREATE (n:Customer {name: 'x'})",
            "MATCH (n) DETACH DELETE n",
            "MERGE (n:Order {order_id: 1})",
            "MATCH (n) SET n.x = 1",
            "DROP INDEX idx",
            "LOAD CSV FROM 'file' AS line RETURN line",
        ],
    )
    def test_write_statements_rejected(self, cypher: str) -> None:
        with pytest.raises(Neo4jReadOnlyViolation):
            _classify_for_readonly(cypher)

    def test_empty_rejected(self) -> None:
        with pytest.raises(Neo4jReadOnlyViolation):
            _classify_for_readonly("   ")

    def test_return_only_rejected_without_match_prefix(self) -> None:
        # "RETURN 1" on its own is allowed (leading RETURN is in the allow-list);
        # but statements that neither start with MATCH nor RETURN are not.
        with pytest.raises(Neo4jReadOnlyViolation):
            _classify_for_readonly("CALL some.procedure()")


# ---------------------------------------------------------------------------
# Exception diagnosis
# ---------------------------------------------------------------------------


class TestDiagnose:
    def test_auth_error_kind(self) -> None:
        d = _diagnose(Exception("Authentication failed for user"))
        assert d["kind"] == "auth"
        assert d["driver_exception"] == "Exception"

    def test_uri_redacted(self) -> None:
        d = _diagnose(Exception("cannot connect to bolt://user:secret@db:7687"))
        assert "secret" not in d["message"]
        assert "<uri>" in d["message"]

    def test_message_truncated(self) -> None:
        d = _diagnose(Exception("x" * 5000))
        assert len(d["message"]) <= 500


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


class TestNeo4jResult:
    def test_to_dicts_roundtrip(self) -> None:
        res = Neo4jResult(cypher="RETURN 1", records=[{"x": 1}], latency_ms=2.0)
        assert res.to_dicts() == [{"x": 1}]


# ---------------------------------------------------------------------------
# _execute error mapping (mocked driver)
# ---------------------------------------------------------------------------


class TestExecuteErrorMapping:
    def _mock_driver_session(self) -> MagicMock:
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return driver, session

    def test_unavailable_error(self) -> None:
        client = Neo4jClient(password="x")
        driver, session = self._mock_driver_session()
        session.run.side_effect = Exception("connection refused")
        client._driver = driver
        with pytest.raises(Neo4jUnavailableError):
            client.execute_readonly("RETURN 1")

    def test_missing_password_raises_before_driver(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = Neo4jClient()
            assert client.password == ""
            with pytest.raises(Neo4jUnavailableError):
                client.get_driver()

    def test_query_error(self) -> None:
        class Neo4jQueryException(Exception):
            pass

        client = Neo4jClient(password="x")
        driver, session = self._mock_driver_session()
        session.run.side_effect = Neo4jQueryException(
            "Neo.ClientError.Query.InvalidStatement: property `x` does not exist"
        )
        client._driver = driver
        # _diagnose maps unknown driver exception names to kind "query"
        with pytest.raises(Neo4jQueryError):
            client.execute_readonly("MATCH (n) WHERE n.x = 1 RETURN n")
