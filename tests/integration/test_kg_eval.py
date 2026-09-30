"""Integration test for scripts/run_kg_eval.py against a live Neo4j.

Skipped unless ``NEO4J_INTEGRATION=1``.  Runs a small subset of the eval
(dataset unchanged, ``--limit 10``) and checks the artifact files are
written with the expected schema.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS = _PROJECT_ROOT / "artifacts" / "phase2.4" / "kg_eval_results.json"
_SUMMARY = _PROJECT_ROOT / "artifacts" / "phase2.4" / "kg_eval_summary.json"
_MD = _PROJECT_ROOT / "artifacts" / "phase2.4" / "kg_eval_summary.md"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("NEO4J_INTEGRATION") != "1",
        reason="set NEO4J_INTEGRATION=1 (with a running Neo4j + NEO4J_PASSWORD) to enable",
    ),
]


def test_kg_eval_run_writes_artifacts() -> None:
    env = dict(os.environ)
    env.setdefault("NEO4J_PASSWORD", "")
    if not env["NEO4J_PASSWORD"]:
        pytest.skip("NEO4J_PASSWORD not set")

    proc = subprocess.run(
        [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "run_kg_eval.py"),
            "--limit", "10",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_PROJECT_ROOT),
        timeout=600,
    )
    assert proc.returncode == 0, f"run_kg_eval failed:\n{proc.stdout}\n{proc.stderr}"

    results = json.loads(_RESULTS.read_text(encoding="utf-8"))
    assert len(results) == 10

    for entry in results:
        # every task record carries the required fields
        for key in ("id", "task_type", "difficulty", "expected_answer",
                    "predicted", "error", "latency_ms", "precision", "recall", "f1"):
            assert key in entry, f"missing {key!r} in {entry['id']}"
        # failed tasks stay in the denominator with error set
        if entry["error"]:
            assert entry["exact_match"] is False

    summary = json.loads(_SUMMARY.read_text(encoding="utf-8"))
    assert summary["total_tasks"] == 10
    assert "overall" in summary
    assert "exact_match" in summary["overall"]
    assert "avg_f1" in summary["overall"]
    assert _MD.exists()
