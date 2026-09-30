"""Unit tests for src/core/routing_types.py (Phase 3 Commit 1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.routing_types import (
    ROUTE_TYPES,
    ExecutionStep,
    RoutingDecision,
    TaskProfile,
)


class TestTaskProfile:
    def test_default_flags_all_false(self) -> None:
        p = TaskProfile(task_id="t1")
        assert not p.requires_structured_data
        assert p.ambiguity == 0.0
        assert p.difficulty == "medium"

    def test_ambiguity_clamped(self) -> None:
        assert TaskProfile(task_id="t", ambiguity=5.0).ambiguity == 1.0
        assert TaskProfile(task_id="t", ambiguity=-3.0).ambiguity == 0.0

    def test_invalid_task_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TaskProfile(task_id="t", task_type="not_a_type")  # type: ignore[arg-type]

    def test_valid_task_type(self) -> None:
        TaskProfile(task_id="t", task_type="multi_source_analysis")

    def test_capability_vector_keys(self) -> None:
        p = TaskProfile(task_id="t", requires_structured_data=True)
        vec = p.capability_vector()
        assert set(vec) == {
            "requires_structured_data",
            "requires_unstructured_knowledge",
            "requires_relationship_reasoning",
            "requires_statistical_analysis",
            "requires_visualization",
            "requires_multi_source",
        }
        assert vec["requires_structured_data"] is True


class TestExecutionStep:
    def test_valid_step(self) -> None:
        step = ExecutionStep(step=1, tool="sql", purpose="get sales")
        assert step.tool == "sql"

    def test_unknown_tool_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionStep(step=1, tool="evil_tool")  # type: ignore[arg-type]

    def test_step_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ExecutionStep(step=0, tool="sql")


class TestRoutingDecision:
    def test_single_tool_consistent(self) -> None:
        d = RoutingDecision(
            route_type="sql",
            tools=["sql"],
            execution_plan=[ExecutionStep(step=1, tool="sql", purpose="fetch data")],
        )
        assert d.route_type == "sql"

    def test_multi_tool_allowed(self) -> None:
        d = RoutingDecision(
            route_type="multi_tool",
            tools=["sql", "analysis"],
            execution_plan=[
                ExecutionStep(step=1, tool="sql", purpose="fetch"),
                ExecutionStep(step=2, tool="analysis", purpose="compute", depends_on=[1]),
            ],
        )
        assert d.tool_set() == {"sql", "analysis"}

    def test_multi_tool_rejected_for_single_route(self) -> None:
        with pytest.raises(ValidationError):
            RoutingDecision(route_type="sql", tools=["sql", "analysis"])

    def test_plan_tool_must_be_declared(self) -> None:
        with pytest.raises(ValidationError):
            RoutingDecision(
                route_type="rag",
                tools=["rag"],
                execution_plan=[ExecutionStep(step=1, tool="sql", purpose="undeclared")],
            )

    def test_unknown_route_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RoutingDecision(route_type="teleport", tools=["rag"])  # type: ignore[arg-type]

    def test_clarification_must_be_empty(self) -> None:
        with pytest.raises(ValidationError):
            RoutingDecision(
                route_type="clarification",
                tools=["sql"],
            )

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            RoutingDecision(route_type="sql", tools=["sql"], confidence=1.5)
        with pytest.raises(ValidationError):
            RoutingDecision(route_type="sql", tools=["sql"], confidence=-0.1)
        RoutingDecision(route_type="sql", tools=["sql"], confidence=0.0)

    def test_to_state_keys(self) -> None:
        d = RoutingDecision(route_type="kg", tools=["kg"], reason="rel query")
        s = d.to_state()
        assert s["route"] == "kg"
        assert s["routing_decision"]["route_type"] == "kg"
        assert s["tools_selected"] == ["kg"]

    def test_route_types_frozen_vocabulary(self) -> None:
        assert ROUTE_TYPES == ("rag", "sql", "kg", "analysis", "multi_tool", "clarification")
