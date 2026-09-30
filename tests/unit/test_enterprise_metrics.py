"""Unit tests for the Phase 5 enterprise end-to-end metrics.

Pure logic — no live LLM / Milvus / Neo4j.  Covers the per-layer metrics,
the failure-stays-in-denominator rule, and the leak/verification guards.
"""

from __future__ import annotations

from src.evaluation import enterprise_metrics as em


def _task(**overrides) -> dict:
    base = {
        "id": "ent-0001", "task_type": "aggregation", "difficulty": "easy",
        "expected_route": "sql", "expected_tools": ["sql"],
        "expected_tool_order": ["sql"], "expected_clarification": False,
        "expected_answer": "440", "expected_evidence": ["sql"],
        "expected_claims": [{"text": "订单量 440", "claim_type": "numerical",
                             "importance": "critical", "value": 440}],
    }
    base.update(overrides)
    return base


def _pred(route="sql", tools=("sql",), order=("sql",), answer="共 440 笔",
          evidence=("sql",), verification=None, guard=None, failures=(),
          any_error=False, llm=2, lat=None):
    return {
        "route": route, "tools": list(tools), "order": list(order),
        "answer": answer, "evidence": list(evidence),
        "verification": verification, "guard": guard,
        "any_error": any_error, "failure_categories": list(failures),
        "llm_calls": llm,
        "latency_ms": lat or {"routing": 5.0, "tool_execution": 10.0,
                              "verification": 2.0, "total": 17.0},
    }


def test_task_success_positive():
    r = em.score_task(_task(), _pred())
    assert r["task_success"] == 1
    assert r["route_correct"] == 1
    assert r["tool_f1"] == 1.0
    assert r["plan_exact_match"] == 1
    assert r["answer_exact"] == 1  # "440" appears in "共 440 笔"


def test_task_success_wrong_tool_is_zero_but_kept():
    r = em.score_task(_task(), _pred(route="rag", tools=["rag"], order=["rag"], evidence=["rag"]))
    assert r["task_success"] == 0
    assert r["route_correct"] == 0
    assert r["tool_f1"] == 0.0
    # still present in the record (denominator preserved by summarise)


def test_clarification_success():
    task = _task(expected_route="clarification", expected_tools=[], expected_tool_order=[],
                 expected_clarification=True, expected_answer="需要澄清",
                 expected_evidence=[], expected_claims=[])
    r = em.score_task(task, _pred(route="clarification", tools=[], order=[], answer="需要澄清",
                                  evidence=[]))
    assert r["task_success"] == 1
    assert r["clarification_correct"] == 1


def test_multi_tool_requires_exact_tools_and_order():
    task = _task(expected_route="multi_tool", expected_tools=["sql", "analysis"],
                 expected_tool_order=["sql", "analysis"],
                 expected_answer="增长 20%", expected_evidence=["sql", "analysis"])
    ok = em.score_task(task, _pred(route="multi_tool", tools=["sql", "analysis"],
                                   order=["sql", "analysis"], answer="增长 20%",
                                   evidence=["sql", "analysis"]))
    bad_order = em.score_task(task, _pred(route="multi_tool", tools=["sql", "analysis"],
                                          order=["analysis", "sql"], answer="增长 20%",
                                          evidence=["sql", "analysis"]))
    assert ok["task_success"] == 1
    assert bad_order["task_success"] == 0  # asserted order not met


def test_failure_categories_recorded_not_dropped():
    r = em.score_task(_task(), _pred(route="", tools=[], order=[], answer="", evidence=[],
                                     failures=["llm_429", "timeout"], any_error=True))
    assert r["any_error"]
    assert "llm_429" in r["failure_categories"]
    assert r["task_success"] == 0


def test_verification_leakage_metrics():
    # a critical claim left unsupported AND not blocked -> leakage
    ver = {"claims": [{"claim_id": "c1", "supported": False, "status": "unsupported",
                       "verifier_type": "exact"}],
           "n_claims": 1, "n_critical_unsupported": 1}
    r = em.score_task(_task(), _pred(verification=ver, guard={"block": False}))
    assert r["critical_unsupported_leakage"] == 1
    assert r["unsupported_claims"]["n_unsupported"] == 1

    # blocked -> no leak
    r2 = em.score_task(_task(), _pred(verification=ver, guard={"block": True}))
    assert r2["critical_unsupported_leakage"] == 0

    # no verification layer -> leakage is 0 and unsupported_claims is None
    r3 = em.score_task(_task(), _pred())
    assert r3["critical_unsupported_leakage"] == 0
    assert r3["unsupported_claims"] is None


def test_summarise_keeps_failures_in_denominator():
    good = em.score_task(_task(), _pred())
    bad = em.score_task(_task(), _pred(route="rag", tools=["rag"], order=["rag"],
                                       evidence=["rag"], failures=["sql_error"],
                                       any_error=True))
    summary = em.summarise([good, bad])
    assert summary["overall"]["tasks"] == 2
    assert summary["overall"]["task_success_rate"] == 0.5
    assert summary["overall"]["error_rate"] == 0.5
    assert summary["overall"]["failure_categories"].get("sql_error") == 1


def test_evidence_attribution_accuracy():
    assert em.evidence_attribution_accuracy(["sql"], ["sql"]) == 1.0
    assert em.evidence_attribution_accuracy(["sql", "kg"], ["sql"]) < 1.0
    assert em.evidence_attribution_accuracy([], []) == 1.0


def test_answer_exact_edge_cases():
    assert em.answer_exact("共 440 笔", "440") is True
    assert em.answer_exact("Soylent Ltd", "Soylent Ltd") is True
    assert em.answer_exact("", "440") is False
    assert em.answer_exact("441", "440") is False


def test_grouping_by_task_type_and_difficulty():
    t1 = _task()
    t2 = _task(task_type="knowledge_lookup", difficulty="hard")
    s = em.summarise([em.score_task(t1, _pred()), em.score_task(t2, _pred())])
    assert "knowledge_lookup" in s["by_task_type"]
    assert "hard" in s["by_difficulty"]
