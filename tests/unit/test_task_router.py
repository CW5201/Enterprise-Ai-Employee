"""Unit tests for src/nodes/task_router.py (Phase 3 Commit 2).

The LLM is mocked; the rule engine + validation path is exercised for real.
"""

from __future__ import annotations

from typing import Any, TypeVar
from unittest.mock import MagicMock

import pytest

from src.core.exceptions import LLMError
from src.core.state import make_state
from src.core.routing_types import RoutingDecision, TaskProfile
from src.nodes.task_router import (
    RoutingDecisionError,
    TaskProfileProposal,
    TaskRouterNode,
)

T = TypeVar("T")


def _mock_llm(
    profile: TaskProfileProposal | None = None,
    raise_error: Exception | None = None,
) -> MagicMock:
    llm = MagicMock()
    if raise_error is not None:
        llm.generate_structured.side_effect = raise_error
    else:
        llm.generate_structured.return_value = profile or TaskProfileProposal(
            task_type="knowledge_lookup",
            requires_unstructured_knowledge=True,
        )
    return llm


def _router(llm: MagicMock, min_confidence: float = 0.3) -> TaskRouterNode:
    return TaskRouterNode(llm=llm, min_confidence=min_confidence)


@pytest.mark.unit
class TestRoutingLanes:
    def test_rule_lane_knowledge_query(self) -> None:
        llm = _mock_llm()  # must NOT be called
        node = _router(llm)
        state = make_state("报销政策是什么？")
        state.update({"intent": "knowledge_query", "confidence": 0.95})
        out = node.run(state)
        assert out["route"] == "rag"
        assert out["tools_selected"] == ["rag"]
        llm.generate_structured.assert_not_called()

    def test_rule_lane_data_query(self) -> None:
        node = _router(_mock_llm())
        state = make_state("统计客户订单数")
        state.update({"intent": "data_query", "confidence": 0.95})
        out = node.run(state)
        assert out["route"] == "sql"

    def test_llm_lane_relationship_to_kg(self) -> None:
        profile = TaskProfileProposal(
            task_type="relationship_query",
            requires_relationship_reasoning=True,
        )
        node = _router(_mock_llm(profile))
        state = make_state("某客户下过哪些订单，涉及哪些供应商？")
        state.update({"intent": "", "confidence": 0.0})
        out = node.run(state)
        assert out["route"] == "kg"
        assert out["routing_decision"]["route_type"] == "kg"

    def test_llm_lane_statistical_multi_tool(self) -> None:
        profile = TaskProfileProposal(
            task_type="statistical_analysis",
            requires_structured_data=True,
            requires_statistical_analysis=True,
            requires_multi_source=True,
        )
        node = _router(_mock_llm(profile))
        state = make_state("统计销售额并计算增长率")
        state.update({"intent": "", "confidence": 0.0})
        out = node.run(state)
        assert out["route"] == "multi_tool"
        assert set(out["tools_selected"]) >= {"sql", "analysis"}
        plan = out["routing_decision"]["execution_plan"]
        analysis_step = next(s for s in plan if s["tool"] == "analysis")
        assert any(s["tool"] == "sql" for s in plan if s["step"] in analysis_step["depends_on"])

    def test_ambiguous_routes_to_clarification(self) -> None:
        profile = TaskProfileProposal(task_type="ambiguous_task", ambiguity=0.95)
        node = _router(_mock_llm(profile))
        state = make_state("帮我查一下最近情况")
        state.update({"intent": "clarification", "confidence": 0.0})
        out = node.run(state)
        assert out["route"] == "clarification"
        assert out["routing_decision"]["requires_clarification"] is True
        assert out["tools_selected"] == []

    def test_no_flags_routes_to_clarification(self) -> None:
        profile = TaskProfileProposal(task_type="ambiguous_task")
        node = _router(_mock_llm(profile))
        state = make_state("？")
        state.update({"intent": "clarification", "confidence": 0.0})
        out = node.run(state)
        assert out["route"] == "clarification"

    def test_llm_error_degrades_to_clarification(self) -> None:
        node = _router(_mock_llm(raise_error=LLMError("backend down")))
        state = make_state("查一下数据")
        state.update({"intent": "", "confidence": 0.0})
        out = node.run(state)
        assert out["route"] == "clarification"
        assert any(e.stage == "routing" for e in out["errors"])

    def test_malformed_llm_output_rejected(self) -> None:
        llm = MagicMock()
        llm.generate_structured.side_effect = ValueError("bad json")
        node = _router(llm)
        state = make_state("查一下")
        state.update({"intent": "", "confidence": 0.0})
        out = node.run(state)
        assert out["route"] == "clarification"


@pytest.mark.unit
class TestRoutingTrace:
    def test_trace_fields_present(self) -> None:
        profile = TaskProfileProposal(
            task_type="structured_lookup", requires_structured_data=True,
        )
        node = _router(_mock_llm(profile))
        state = make_state("查询客户总数")
        state.update({"intent": "", "confidence": 0.0})
        out = node.run(state)
        trace = out["routing_trace"]
        for key in ("task_profile", "candidates", "selected_tools", "reason",
                    "confidence", "timestamp", "latency_ms"):
            assert key in trace, key
        assert trace["selected_tools"] == ["sql"]
        # no secrets in trace
        assert "password" not in str(trace).lower()

    def test_decision_is_structured(self) -> None:
        profile = TaskProfileProposal(task_type="knowledge_lookup",
                                      requires_unstructured_knowledge=True)
        out = _router(_mock_llm(profile)).run(
            {**make_state("制度"), "intent": "", "confidence": 0.0}
        )
        decision = RoutingDecision.model_validate(out["routing_decision"])
        assert decision.route_type == "rag"


@pytest.mark.unit
class TestConfidenceGate:
    def test_low_confidence_demoted_to_clarification(self) -> None:
        # ambiguity 0.99 -> decision confidence ~0.21, below the 0.6 gate
        profile = TaskProfileProposal(
            task_type="structured_lookup",
            requires_structured_data=True,
            ambiguity=0.99,
        )
        node = _router(_mock_llm(profile), min_confidence=0.6)
        state = make_state("查个数据")
        state.update({"intent": "", "confidence": 0.0})
        out = node.run(state)
        assert out["route"] == "clarification"
        assert out["route_confidence"] < 0.6


@pytest.mark.unit
class TestDecisionRejection:
    def test_rejected_decision_raises_explicitly(self) -> None:
        import src.nodes.task_router as tr

        node = _router(_mock_llm())
        orig = tr._decide_rule

        def _bad(profile, policy):
            return RoutingDecision(route_type="sql", tools=["sql", "analysis"])

        tr._decide_rule = _bad
        try:
            with pytest.raises(RoutingDecisionError):
                node._decide(TaskProfile(task_id="t", requires_structured_data=True), [])
        finally:
            tr._decide_rule = orig
