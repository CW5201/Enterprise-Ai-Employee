"""Integration tests for src/core/neo4j_client.py against a live Neo4j.

Skipped unless ``NEO4J_INTEGRATION=1`` is set (and the credentials come from
the environment).  This keeps the default unit-test run green on machines
without a graph database while still giving Phase 2.4 a real smoke path.
"""

from __future__ import annotations

import os

import pytest

from src.core.neo4j_client import Neo4jClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("NEO4J_INTEGRATION") != "1",
        reason="set NEO4J_INTEGRATION=1 (with a running Neo4j + NEO4J_PASSWORD) to enable",
    ),
]


def _client() -> Neo4jClient:
    return Neo4jClient()


class TestSmoke:
    def test_health_check(self) -> None:
        client = _client()
        try:
            health = client.health_check()
        finally:
            client.close()
        assert health["ok"] is True
        # version may be None on Neo4j 5 where no Cypher version() function
        # exists; assert the probe itself succeeded.
        assert health["latency_ms"] >= 0

    def test_node_create_query_cleanup(self) -> None:
        client = _client()
        try:
            # Write path: create one throwaway node in a dedicated test label.
            client.run_write(
                "CREATE (t:__eae_smoke_test {marker: $m})",
                {"m": "smoke"},
            )
            # Read-only path: query it back through the guarded wrapper.
            result = client.execute_readonly(
                "MATCH (t:__eae_smoke_test {marker: $m}) RETURN t.marker AS marker",
                {"m": "smoke"},
            )
            assert result.records == [{"marker": "smoke"}]

            # Clean up so repeated runs do not accumulate nodes.
            client.run_write("MATCH (t:__eae_smoke_test) DETACH DELETE t")
        finally:
            client.close()

    def test_readonly_rejects_write(self) -> None:
        from src.core.neo4j_client import Neo4jReadOnlyViolation

        client = _client()
        try:
            with pytest.raises(Neo4jReadOnlyViolation):
                client.execute_readonly("MATCH (n) DETACH DELETE n")
        finally:
            client.close()
