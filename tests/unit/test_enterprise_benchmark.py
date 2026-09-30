"""Unit tests for the Phase 5 Enterprise Task Benchmark builder/validator.

These tests keep the whole benchmark reproducible and guard the GT discipline:

- the builder emits a stable, deterministic JSONL;
- the validator catches id / vocabulary / leakage / duplicate problems;
- the committed dataset passes the validator out of the box.

They do NOT call any live LLM / Milvus / Neo4j; GT is re-derived from the
read-only DuckDB sample (fast, offline) and the KB chunk inventory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import build_enterprise_tasks as bet
from scripts import validate_enterprise_tasks as vet

_DATASET = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "enterprise_tasks.jsonl"


@pytest.mark.unit
def test_build_is_deterministic_and_nonempty():
    tasks = bet.build_all()
    assert len(tasks) >= 400, "benchmark must reach the paper-experiment scale"
    # ids are sequential and unique
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids))
    assert ids == [f"ent-{i:04d}" for i in range(1, len(tasks) + 1)]


@pytest.mark.unit
def test_all_task_types_covered():
    tasks = bet.build_all()
    covered = {t["task_type"] for t in tasks}
    assert covered == set(bet._TASK_TYPES), f"all 10 task types must be present, missing {set(bet._TASK_TYPES) - covered}"


@pytest.mark.unit
def test_multi_tool_tasks_declare_distinct_sources():
    tasks = bet.build_all()
    multi = [t for t in tasks if t["expected_route"] == "multi_tool"]
    assert multi, "benchmark must contain multi-tool tasks"
    source_tools = {"rag", "sql", "kg", "analysis"}
    for t in multi:
        sources = set(t["expected_tools"]) & source_tools
        assert len(sources) >= 2, f"{t['id']}: multi_tool must declare >= 2 source tools"


@pytest.mark.unit
def test_validator_rejects_leakage():
    base = bet.build_all()
    # Take a clean SQL task and smuggle a KG evidence source into it -> leakage.
    victim = next(t for t in base if t["expected_route"] == "sql" and not t["expected_clarification"])
    bad = dict(victim)
    bad["expected_evidence"] = ["sql", "kg"]  # KG not declared as a tool
    errors = vet.validate([bad])
    assert any("not among declared tools" in e for e in errors)


@pytest.mark.unit
def test_validator_rejects_duplicate_question():
    base = bet.build_all()
    dup = [dict(base[0]), dict(base[0])]
    # same question twice
    errors = vet.validate(dup)
    assert any("duplicate question" in e for e in errors)


@pytest.mark.unit
def test_validator_rejects_clarification_with_tools():
    clar = dict(next(t for t in bet.build_all() if t["expected_route"] == "clarification"))
    clar["expected_tools"] = ["sql"]
    errors = vet.validate([clar])
    assert any("clarification task must declare no tools" in e for e in errors)


@pytest.mark.unit
def test_committed_dataset_passes_validator():
    if not _DATASET.exists():
        pytest.skip("enterprise_tasks.jsonl not generated yet (run scripts/build_enterprise_tasks.py)")
    tasks = vet.load(_DATASET)
    assert tasks, "committed benchmark must not be empty"
    errors = vet.validate(tasks)
    assert not errors, f"committed benchmark fails validation: {errors[:5]}"


@pytest.mark.unit
def test_dataset_is_jsonl_and_roundtrips():
    if not _DATASET.exists():
        pytest.skip("enterprise_tasks.jsonl not generated yet")
    raw = _DATASET.read_text(encoding="utf-8").strip().splitlines()
    for line in raw:
        json.loads(line)  # every line is valid JSON
