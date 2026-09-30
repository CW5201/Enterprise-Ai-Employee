"""Integration test for scripts/build_kg.py (requires live Neo4j + DuckDB).

Skipped unless ``NEO4J_INTEGRATION=1`` (with ``NEO4J_PASSWORD`` set).  Runs a
``--limit 5 --reset`` build against the real runtime graph database and
checks the summary artifact, so it stays cheap and does not disturb a full
graph already present on the machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.neo4j_client import Neo4jClient

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ARTIFACT = _PROJECT_ROOT / "artifacts" / "phase2.4" / "kg_build_summary.json"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("NEO4J_INTEGRATION") != "1",
        reason="set NEO4J_INTEGRATION=1 (with a running Neo4j + NEO4J_PASSWORD) to enable",
    ),
]


def test_kg_build_limited_reset() -> None:
    env = dict(os.environ)
    env.setdefault("NEO4J_PASSWORD", "")
    if not env["NEO4J_PASSWORD"]:
        pytest.skip("NEO4J_PASSWORD not set")

    proc = subprocess.run(
        [sys.executable, str(_PROJECT_ROOT / "scripts" / "build_kg.py"), "--reset", "--limit", "5"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_PROJECT_ROOT),
        timeout=600,
    )
    assert proc.returncode == 0, f"build_kg failed:\n{proc.stdout}\n{proc.stderr}"

    summary = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    # limit=5 means at most 5 nodes per label (7 labels = 35).
    assert summary["total_nodes"] <= 35, summary["total_nodes"]
    assert summary["failed_records"] == 0, summary["errors"]
    assert summary["total_relationships"] >= 0

    # Restore a full graph so later phases are not left with a stub graph.
    full = subprocess.run(
        [sys.executable, str(_PROJECT_ROOT / "scripts" / "build_kg.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_PROJECT_ROOT),
        timeout=1200,
    )
    assert full.returncode == 0, f"full rebuild failed:\n{full.stdout}\n{full.stderr}"
    summary = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert summary["total_nodes"] > 4000
    assert summary["failed_records"] == 0

    # Spot check: core chain has edges.
    client = Neo4jClient()
    try:
        res = client.execute_readonly(
            "MATCH (:Customer)-[:PLACED]->(:Order)-[:INVOICED]->(:Invoice) RETURN count(*) AS c"
        )
        assert res.records[0]["c"] > 0
    finally:
        client.close()
