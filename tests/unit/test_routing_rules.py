"""Unit tests for src/core/routing_rules.py (Phase 3 Commit 1)."""

from __future__ import annotations

import pytest

from src.core.routing_rules import RoutingPolicy, decide, generate_candidates
from src.core.routing_types import TaskProfile


@pytest.fixture()
def policy() -> RoutingPolicy:
    return RoutingPolicy.load()


# ---------------------------------------------------------------------------
# Capability-driven routing (the pure rule path)
# ---------------------------------------------------------------------------


class TestDecide:
    def test_pure_rag(self, policy: RoutingPolicy) -> None:
        profile = TaskProfile(task_id="t", task_type="knowledge_lookup",
                              requires_unstructured_knowledge=True)
        d = decide(profile, policy)
        assert d.route_type == "rag"
        assert d.tools == ["rag"]
        assert not d.requires_clarification

    def test_pure_sql(self, policy: RoutingPolicy) -> None:
        profile = TaskProfile(task_id="t", task_type="structured_lookup",
                              requires_structured_data=True)
        d = decide(profile, policy)
        assert d.route_type == "sql"
        assert d.tools == ["sql"]

    def test_pure_kg(self, policy: RoutingPolicy) -> None:
        profile = TaskProfile(task_id="t", task_type="relationship_query",
                              requires_relationship_reasoning=True)
        d = decide(profile, policy)
        assert d.route_type == "kg"
        assert d.tools == ["kg"]

    def test_sql_plus_analysis_multi_tool(self, policy: RoutingPolicy) -> None:
        profile = TaskProfile(
            task_id="t", task_type="statistical_analysis",
            requires_structured_data=True, requires_statistical_analysis=True,
            requires_multi_source=True,
        )
        d = decide(profile, policy)
        assert d.route_type == "multi_tool"
        assert set(d.tools) >= {"sql", "analysis"}
        # execution plan: analysis must depend on sql
        sql_step = next(s for s in d.execution_plan if s.tool == "sql")
        analysis_step = next(s for s in d.execution_plan if s.tool == "analysis")
        assert sql_step.step in analysis_step.depends_on

    def test_rag_plus_sql_multi_tool(self, policy: RoutingPolicy) -> None:
        profile = TaskProfile(
            task_id="t", task_type="multi_source_analysis",
            requires_structured_data=True, requires_unstructured_knowledge=True,
            requires_multi_source=True,
        )
        d = decide(profile, policy)
        assert d.route_type == "multi_tool"
        assert set(d.tools) == {"rag", "sql"}

    def test_sql_plus_kg_multi_tool(self, policy: RoutingPolicy) -> None:
        profile = TaskProfile(
            task_id="t", task_type="multi_source_analysis",
            requires_structured_data=True, requires_relationship_reasoning=True,
            requires_multi_source=True,
        )
        d = decide(profile, policy)
        assert d.route_type == "multi_tool"
        assert set(d.tools) == {"sql", "kg"}

    def test_ambiguous_routes_to_clarification(self, policy: RoutingPolicy) -> None:
        profile = TaskProfile(task_id="t", task_type="ambiguous_task", ambiguity=0.9)
        d = decide(profile, policy)
        assert d.route_type == "clarification"
        assert d.requires_clarification
        assert d.tools == []

    def test_no_flags_routes_to_clarification(self, policy: RoutingPolicy) -> None:
        profile = TaskProfile(task_id="t")
        d = decide(profile, policy)
        assert d.route_type == "clarification"

    def test_high_ambiguity_overrides_flags(self, policy: RoutingPolicy) -> None:
        profile = TaskProfile(
            task_id="t", task_type="structured_lookup",
            requires_structured_data=True, ambiguity=0.9,
        )
        d = decide(profile, policy)
        assert d.route_type == "clarification"

    def test_single_tool_plan_is_one_step(self, policy: RoutingPolicy) -> None:
        profile = TaskProfile(task_id="t", task_type="knowledge_lookup",
                              requires_unstructured_knowledge=True)
        d = decide(profile, policy)
        assert len(d.execution_plan) == 1
        assert d.execution_plan[0].tool == "rag"


class TestGenerateCandidates:
    def test_candidates_by_weight(self, policy: RoutingPolicy) -> None:
        profile = TaskProfile(
            task_id="t", task_type="multi_source_analysis",
            requires_structured_data=True,
            requires_unstructured_knowledge=True,
            requires_visualization=True,
        )
        cands = generate_candidates(profile, policy)
        tools = [c.tool for c in cands]
        assert "sql" in tools and "rag" in tools and "chart" in tools
        # chart (weight 0.5) must sort after primary sources (1.0)
        assert cands[0].tool in ("sql", "rag")

    def test_no_flags_no_candidates(self, policy: RoutingPolicy) -> None:
        assert generate_candidates(TaskProfile(task_id="t"), policy) == []


class TestPolicyLoading:
    def test_known_tools_frozen(self, policy: RoutingPolicy) -> None:
        assert set(policy.known_tools) == {"rag", "sql", "kg", "analysis", "chart", "report"}

    def test_policy_file_parses(self, policy: RoutingPolicy) -> None:
        assert policy.capability_rules
        assert "clarification" in (policy.clarification_auto_types or []) or \
            policy.clarification_auto_types == ["ambiguous_task"]
