"""Unit tests for src/nodes/multi_tool_execution.py (Phase 3 Commit 3).

Tools are mocked so the tests verify orchestration (ordering, dependency
gates, failure propagation, result collection, evidence fusion) without
needing live Milvus / DuckDB / Neo4j.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.state import make_state
from src.nodes.multi_tool_execution import MultiToolExecutionNode


def _decision(tools: list[str], *, with_kg_template: bool = False,
              analysis_op: str = "mean", analysis_value: str | None = None) -> dict[str, Any]:
    """Build a routing_decision dict (as RoutingDecision.model_dump())."""
    plan: list[dict[str, Any]] = []
    step = 1
    for t in tools:
        step_spec: dict[str, Any] = {"step": step, "tool": t, "purpose": t}
        if t == "analysis" and step > 1:
            step_spec["depends_on"] = [1]
        if t == "kg":
            step_spec["template_id"] = "customer_orders" if with_kg_template else ""
            step_spec["parameters"] = {"customer_id": 2}
        if t == "analysis":
            p: dict[str, Any] = {"operation": analysis_op}
            if analysis_value:
                p["value"] = analysis_value
            step_spec["parameters"] = p
        plan.append(step_spec)
        step += 1
    return {
        "route_type": "multi_tool" if len(tools) > 1 else (tools[0] if tools else "clarification"),
        "tools": tools,
        "execution_plan": plan,
        "reason": "test",
        "confidence": 0.9,
        "requires_clarification": False,
    }


def _node(
    sql_ok: bool = True,
    rag_hits: int = 2,
    kg_ok: bool = True,
) -> tuple[MultiToolExecutionNode, dict[str, Any]]:
    sql_out: dict[str, Any] = {}
    if sql_ok:
        from src.core.state import SQLResultPayload
        sql_result = SQLResultPayload(
            sql="SELECT 1", columns=["n"], rows=[[1]], row_count=1,
        )
        sql_out = {"sql_result": sql_result, "errors": [], "evidence": []}
    else:
        sql_out = {"sql_result": None, "errors": [MagicMock(details={"reason": "boom"})], "evidence": []}

    sql_node = MagicMock()
    sql_node.run.return_value = sql_out

    rag = MagicMock()
    rag.run.return_value = {
        "ok": True,
        "results": [
            {"chunk_id": f"c{i}", "text": "doc text", "source": "s", "score": 0.9 - i * 0.1,
             "metadata": {"title": f"Doc {i}"}}
            for i in range(rag_hits)
        ],
    }

    kg = MagicMock()
    if kg_ok:
        kg.run.return_value = {"success": True, "rows": 2, "query_type": "customer_orders", "latency_ms": 5.0}
    else:
        kg.run.return_value = {"success": False, "error": {"code": "neo4j_error", "message": "down"}, "rows": 0}

    node = MultiToolExecutionNode(
        sql_node=sql_node, rag_tool=rag, kg_tool=kg,
    )
    node._sql_node = sql_node  # ensure dispatch uses the mock
    return node, {"sql": sql_node, "rag": rag, "kg": kg}


@pytest.mark.unit
class TestSingleTool:
    def test_sql_only(self) -> None:
        node, _ = _node(sql_ok=True)
        state = {**make_state("查询数据"), "routing_decision": _decision(["sql"])}
        out = node.run(state)
        assert out["status"] == "executed"
        assert len(out["tool_results"]) == 1
        assert out["tool_results"][0]["success"] is True
        assert out["tool_results"][0]["data"]["row_count"] == 1

    def test_sql_failure_recorded_not_fallback(self) -> None:
        node, _ = _node(sql_ok=False)
        state = {**make_state("x"), "routing_decision": _decision(["sql"])}
        out = node.run(state)
        assert out["status"] == "execution_all_failed"
        assert out["tool_results"][0]["success"] is False
        assert any(e.stage == "execution" for e in out["errors"])

    def test_kg_without_template_is_honest_failure(self) -> None:
        node, _ = _node()
        state = {**make_state("x"), "routing_decision": _decision(["kg"], with_kg_template=False)}
        out = node.run(state)
        assert out["tool_results"][0]["success"] is False
        assert "skipped_no_template" in out["tool_results"][0]["error"]

    def test_kg_with_template_success(self) -> None:
        node, _ = _node(kg_ok=True)
        state = {**make_state("x"), "routing_decision": _decision(["kg"], with_kg_template=True)}
        out = node.run(state)
        assert out["tool_results"][0]["success"] is True
        assert out["tool_results"][0]["data"]["rows"] == 2

    def test_rag_fuses_evidence(self) -> None:
        node, _ = _node(rag_hits=2)
        state = {**make_state("x"), "routing_decision": _decision(["rag"])}
        out = node.run(state)
        assert out["tool_results"][0]["success"] is True
        assert len(out["evidence"]) == 2
        assert all(e.source_type == "milvus" for e in out["evidence"])


@pytest.mark.unit
class TestMultiTool:
    def test_sql_then_analysis(self) -> None:
        node, _ = _node(sql_ok=True)
        state = {**make_state("x"), "routing_decision": _decision(
            ["sql", "analysis"], analysis_op="mean", analysis_value="n")}
        out = node.run(state)
        order = [tr["tool"] for tr in out["tool_results"]]
        assert order == ["sql", "analysis"]
        assert out["tool_results"][0]["success"] is True
        assert out["tool_results"][1]["success"] is True
        assert out["tool_results"][1]["data"]["operation"] == "mean"

    def test_rag_then_sql_both_results_kept(self) -> None:
        node, _ = _node(sql_ok=True, rag_hits=1)
        state = {**make_state("x"), "routing_decision": _decision(["rag", "sql"])}
        out = node.run(state)
        assert [tr["tool"] for tr in out["tool_results"]] == ["rag", "sql"]
        assert all(tr["success"] for tr in out["tool_results"])
        # rag result must NOT be overwritten by sql
        assert out["tool_results"][0]["data"]["hits"] == 1

    def test_sql_failure_cascades_to_analysis(self) -> None:
        node, _ = _node(sql_ok=False)
        state = {**make_state("x"), "routing_decision": _decision(["sql", "analysis"])}
        out = node.run(state)
        by_tool = {tr["tool"]: tr for tr in out["tool_results"]}
        assert by_tool["sql"]["success"] is False
        # analysis depends on sql; it must be skipped, not run on empty data
        assert by_tool["analysis"]["success"] is False
        assert "skipped" in by_tool["analysis"]["error"]

    def test_plan_ordering_respected(self) -> None:
        node, _ = _node()
        # deliberately out-of-order step numbers
        decision = _decision(["sql", "kg", "analysis"], with_kg_template=True)
        decision["execution_plan"][1]["step"] = 2  # kg
        decision["execution_plan"][2]["step"] = 3  # analysis depends on 1
        decision["execution_plan"][2]["depends_on"] = [1]
        state = {**make_state("x"), "routing_decision": decision}
        out = node.run(state)
        assert [tr["tool"] for tr in out["tool_results"]] == ["sql", "kg", "analysis"]


@pytest.mark.unit
class TestUnsupported:
    def test_unknown_tool_recorded_not_silently_replaced(self) -> None:
        node, _ = _node()
        state = {**make_state("x"), "routing_decision": _decision(["chart"])}
        out = node.run(state)
        tr = out["tool_results"][0]
        assert tr["success"] is False
        assert tr["error"] in ("skipped_unsupported", "tool 'chart' not dispatchable")
        # no other tool ran in its place
        assert len(out["tool_results"]) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
