"""Unit tests for src/evaluation/routing_metrics.py (Phase 3 Commit 4)."""

from __future__ import annotations

import pytest

from src.core.routing_types import ExecutionStep, RoutingDecision
from src.evaluation.routing_metrics import (
    clarification_accuracy,
    plan_exact_match,
    route_accuracy,
    score,
    task_success,
    tool_selection_metrics,
)


def _dec(route: str, tools: list[str], plan: list[tuple[int, str]] | None = None) -> RoutingDecision:
    steps = [ExecutionStep(step=i, tool=t) for i, t in (plan or [(i + 1, t) for i, t in enumerate(tools)])]
    return RoutingDecision(
        route_type=route,  # type: ignore[arg-type]
        tools=tools,
        execution_plan=steps,
    )


@pytest.mark.unit
class TestRouteAccuracy:
    def test_match(self) -> None:
        assert route_accuracy(_dec("rag", ["rag"]), "rag")
        assert not route_accuracy(_dec("sql", ["sql"]), "rag")


@pytest.mark.unit
class TestToolSelection:
    def test_exact(self) -> None:
        m = tool_selection_metrics(_dec("multi_tool", ["sql", "analysis"]), ["sql", "analysis"])
        assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0

    def test_over_select(self) -> None:
        m = tool_selection_metrics(_dec("multi_tool", ["sql", "rag", "kg"]), ["sql", "rag"])
        assert m["precision"] < 1.0 and m["recall"] == 1.0

    def test_under_select(self) -> None:
        m = tool_selection_metrics(_dec("sql", ["sql"]), ["sql", "analysis"])
        assert m["recall"] < 1.0 and m["precision"] == 1.0

    def test_empty_both(self) -> None:
        m = tool_selection_metrics(_dec("clarification", []), [])
        assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0

    def test_predicted_empty_expected_not(self) -> None:
        m = tool_selection_metrics(_dec("clarification", []), ["sql"])
        assert m["f1"] == 0.0


@pytest.mark.unit
class TestPlanExact:
    def test_order_sensitive(self) -> None:
        d = _dec("multi_tool", ["sql", "analysis"], plan=[(1, "sql"), (2, "analysis")])
        assert plan_exact_match(d, ["sql", "analysis"])
        d2 = _dec("multi_tool", ["sql", "analysis"], plan=[(1, "analysis"), (2, "sql")])
        assert not plan_exact_match(d2, ["sql", "analysis"])

    def test_set_when_no_order(self) -> None:
        d = _dec("multi_tool", ["sql", "analysis"])
        # None => order not asserted, always True
        assert plan_exact_match(d, None) is True
        # when an expected_order list is supplied, order matters
        assert plan_exact_match(d, ["sql", "analysis"]) is True
        assert plan_exact_match(d, ["analysis", "sql"]) is False


@pytest.mark.unit
class TestClarification:
    def test_agree(self) -> None:
        assert clarification_accuracy(_dec("clarification", []), True)
        assert not clarification_accuracy(_dec("clarification", []), False)
        assert not clarification_accuracy(_dec("rag", ["rag"]), True)

    def test_task_success_clarification(self) -> None:
        assert task_success(_dec("clarification", []), "clarification", [])
        assert not task_success(_dec("sql", ["sql"]), "clarification", [])


@pytest.mark.unit
class TestScore:
    def test_full_success(self) -> None:
        s = score(_dec("multi_tool", ["sql", "analysis"]),
                  expected_route="multi_tool", expected_tools=["sql", "analysis"],
                  expected_order=["sql", "analysis"])
        assert s["route_correct"] and s["task_success"] and s["tool_f1"] == 1.0

    def test_predicted_error_scores_zero(self) -> None:
        s = score(_dec("rag", ["rag"]), expected_route="sql", expected_tools=["sql"],
                  predicted_error="router crashed")
        assert s["route_correct"] is False and s["task_success"] is False
        assert s["tool_f1"] == 0.0 and s["error"] == "router crashed"
