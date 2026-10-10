"""Ambiguity register + intent rule lane tests.

These pin the contract that broke once already (FH-INTENT-001): the register
in ``config/ambiguity_register.yaml`` was filled with ordinary metric words
("销售额" / "数量" / "平均" / "增长率" …), and because the rule fast lane
checks the register *before* the data / knowledge keywords, every such
question was hard-routed to clarification — 30 of the 487 benchmark questions
included.  The guards below keep that from coming back:

- the register may only hold phrasings with no defined 口径;
- metric questions must still reach the data lane;
- no benchmark task that expects a real route may be rejected by the register;
- a registered phrasing is decisive and never deferred to the LLM lane.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.state import make_state
from src.nodes.intent_understanding import (
    IntentUnderstandingNode,
    classify_by_rules,
)

_ROOT = Path(__file__).resolve().parent.parent.parent
_REGISTER = _ROOT / "config" / "ambiguity_register.yaml"
_BENCHMARK = _ROOT / "data" / "eval" / "enterprise_tasks.jsonl"

# Metric / business terms that DO have a defined 口径 (data dictionary, schema
# or knowledge base).  Registering one of these makes an answerable question
# unanswerable, so the register must never contain them.
_DEFINED_TERMS = (
    "销售额", "金额", "数量", "平均", "占比", "比例", "总额",
    "增长率", "环比", "同比", "订单量", "利润", "营收",
)


def _register_patterns() -> list[str]:
    import yaml

    data = yaml.safe_load(_REGISTER.read_text(encoding="utf-8")) or {}
    patterns = data["patterns"] if isinstance(data, dict) else data
    return [str(p) for p in patterns]


@pytest.mark.unit
def test_register_only_holds_undefined_phrasings() -> None:
    patterns = _register_patterns()
    assert patterns, "the register must not silently become empty"
    offenders = [p for p in patterns if any(t in p for t in _DEFINED_TERMS)]
    assert not offenders, (
        "ambiguity register holds terms that already have a defined 口径: "
        f"{offenders} — these belong in the data lane, not in the register"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        "按订单数量排名，第 1 位的客户是谁？",
        "2025 年相对 2024 年，订单数量是增长还是下降？增长率是多少？",
        "库存商品平均的补货提前期（Lead Time，天）是多少？",
        "统计最近三个月的销售额",
        "本月费用总额是多少",
    ],
)
def test_metric_question_keeps_the_data_lane(question: str) -> None:
    """A question with a defined metric must NOT be turned into a clarification."""
    rule = classify_by_rules(question)
    assert rule is not None, f"{question!r} fell through to the LLM lane"
    intent, confidence, _entities, reason = rule
    assert intent in ("data_query", "complex_analysis"), (
        f"{question!r} -> {intent!r} ({reason})"
    )
    assert confidence >= 0.8


@pytest.mark.unit
def test_registered_phrasing_is_clarification_and_skips_the_llm() -> None:
    llm = MagicMock()  # must never be consulted
    node = IntentUnderstandingNode(llm=llm)

    out = node.run(make_state("我们的大客户有哪些？"))

    assert out["intent"] == "clarification"
    assert out["confidence"] >= 0.8
    assert "ambiguity" in out["plan"]
    llm.generate_structured.assert_not_called()


@pytest.mark.unit
def test_register_outranks_data_keywords() -> None:
    """Undefined 口径 wins even when the phrasing also looks like a data query."""
    rule = classify_by_rules("大客户的订单数量是多少？")
    assert rule is not None
    assert rule[0] == "clarification"


@pytest.mark.unit
def test_no_benchmark_task_is_rejected_by_the_register() -> None:
    """Regression guard for FH-INTENT-001 against the real 487-task benchmark."""
    rejected: list[str] = []
    for line in _BENCHMARK.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        if task.get("expected_route") == "clarification":
            continue  # the benchmark itself declares this one ambiguous
        rule = classify_by_rules(str(task["question"]))
        if rule is not None and rule[0] == "clarification":
            rejected.append(f"{task['id']}: {task['question']}")
    assert not rejected, (
        "the ambiguity register rejects benchmark tasks that expect a real "
        f"route (n={len(rejected)}): {rejected[:5]}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    [
        "统计最近三个月的销售额",              # the workbench's own example task
        "2025 年相对 2024 年，订单数量是增长还是下降？增长率是多少？",
    ],
)
def test_metric_question_routes_end_to_end(question: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """FH-INTENT-001, end to end: the pipeline routes, it does not refuse.

    The classifier-level guard above would still pass if the router dropped the
    intent again, so run the real graph (fake RAG backend, offline LLM) and
    assert the run reaches a data route instead of ``clarified_out_of_scope``.
    """
    monkeypatch.setenv("LLM_FORCE_OFFLINE", "1")
    from src.graph.builder import DigitalEmployee

    state = DigitalEmployee(backend="fake", phase3=True, phase4=True).run(question)

    assert state["status"] != "clarified_out_of_scope", (
        f"{question!r} was refused as out of scope (intent={state['intent']!r})"
    )
    assert state["route"] in ("sql", "multi_tool", "rag"), state["route"]
    assert state["tools_selected"], "a routed task must select at least one tool"
