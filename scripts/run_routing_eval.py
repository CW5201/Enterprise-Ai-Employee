"""Run the Phase 3 routing evaluation across baselines.

Baselines (docs/RESEARCH.md RQ1, ablation A-D):

- **A — Static RAG**: every task routed to the RAG tool.
- **B — Static SQL**: every task routed to SQL (a fixed single-source policy).
- **C — Rule-based**: capability flags derived by *deterministic keyword
  rules* (no LLM), then the same rule engine as the dynamic router.
- **D — LLM Task-Adaptive**: the real :class:`TaskRouterNode` (LLM proposes
  the structured TaskProfile, rule engine + validation decide).

Ground truth is read from ``data/eval/routing_eval.jsonl`` and is never
produced by any baseline.  Every task is scored; failures stay in the
denominator.

Outputs (gitignored): ``artifacts/phase3/routing_eval_results.json``,
``routing_eval_summary.json``, ``routing_eval_summary.md``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.llm_client import LLMClient  # noqa: E402
from src.core.routing_rules import RoutingPolicy, decide  # noqa: E402
from src.core.routing_types import (  # noqa: E402
    ExecutionStep,
    RoutingDecision,
    TaskProfile,
)
from src.evaluation.routing_metrics import score  # noqa: E402
from src.nodes.task_router import TaskRouterNode  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = _PROJECT_ROOT / "data" / "eval" / "routing_eval.jsonl"
ARTIFACT_DIR = _PROJECT_ROOT / "artifacts" / "phase3"

# ---------------------------------------------------------------------------
# Baseline predictors — each returns (RoutingDecision, error|None)
# ---------------------------------------------------------------------------


def _static(tool: str, reason: str) -> RoutingDecision:
    return RoutingDecision(
        route_type=tool,
        tools=[tool],
        execution_plan=[ExecutionStep(step=1, tool=tool, purpose=reason)],
        reason=reason,
        confidence=0.5,
    )


def predict_static_rag(question: str) -> tuple[RoutingDecision, str | None]:
    return _static("rag", "baseline A: static RAG"), None


def predict_static_sql(question: str) -> tuple[RoutingDecision, str | None]:
    return _static("sql", "baseline B: static SQL"), None


# --- baseline C: deterministic keyword capability rules -------------------

_STRUCTURED_KW = ["统计", "多少", "数量", "金额", "客户", "订单", "销售额", "排名",
                  "top", "前", "平均", "合计", "占比", "城市", "商品", "供应商", "发票", "信用"]
_UNSTRUCTURED_KW = ["政策", "制度", "规定", "标准", "流程", "手册", "规范", "要求", "依据", "对照"]
_RELATION_KW = ["关系", "路径", "涉及", "关联", "哪些供应商", "哪些商品", "下过哪些", "购买过"]
_STAT_KW = ["增长率", "趋势", "环比", "同比", "对比", "分布", "统计量", "变异", "占比", "均值", "中位"]
_AMBIGUOUS_KW = ["最近", "情况", "看看", "分析一下", "处理下", "重要的事", "那个"]


def predict_rule_based(question: str) -> tuple[RoutingDecision, str | None]:
    """Deterministic keyword rules -> capability flags -> rule engine."""
    q = question.lower()
    has_struct = any(k in q for k in _STRUCTURED_KW)
    has_unstruct = any(k in q for k in _UNSTRUCTURED_KW)
    has_rel = any(k in q for k in _RELATION_KW)
    has_stat = any(k in q for k in _STAT_KW)
    # vague queries with no concrete subject -> ambiguity
    has_ambig = any(k in q for k in _AMBIGUOUS_KW) and not (has_struct or has_unstruct or has_rel)
    ambiguity = 0.9 if has_ambig else 0.0
    sources = sum([has_struct, has_unstruct, has_rel])
    profile = TaskProfile(
        task_id="rule",
        task_type=("ambiguous_task" if has_ambig else
                   "relationship_query" if has_rel else
                   "multi_source_analysis" if sources >= 2 else
                   "knowledge_lookup" if has_unstruct else
                   "aggregation" if has_struct else "ambiguous_task"),
        requires_structured_data=has_struct,
        requires_unstructured_knowledge=has_unstruct,
        requires_relationship_reasoning=has_rel,
        requires_statistical_analysis=has_stat,
        requires_multi_source=(sources >= 2 or has_stat),
        ambiguity=ambiguity,
    )
    try:
        return decide(profile, RoutingPolicy.load()), None
    except Exception as exc:  # noqa: BLE001
        return _static("rag", "rule engine failure"), str(exc)


# --- baseline D: real LLM Task-Adaptive router ----------------------------


def make_llm_predictor() -> Any:
    node = TaskRouterNode(llm=LLMClient(), min_confidence=0.3)

    def predict(question: str) -> tuple[RoutingDecision, str | None]:
        from src.core.state import make_state

        state = make_state(question)
        state.update({"intent": "", "confidence": 0.0})
        try:
            out = node.run(state)
            decision = RoutingDecision.model_validate(out["routing_decision"])
            return decision, None
        except Exception as exc:  # noqa: BLE001
            return _static("rag", "router failure"), str(exc)

    return predict


BASELINES: dict[str, Any] = {
    "A_static_rag": predict_static_rag,
    "B_static_sql": predict_static_sql,
    "C_rule_based": predict_rule_based,
    "D_llm_adaptive": None,  # resolved lazily in main() (needs a live LLM)
}


def _get_predictor(name: str) -> Any:
    if name == "D_llm_adaptive":
        return make_llm_predictor()
    predictor = BASELINES.get(name)
    if predictor is None:
        raise KeyError(f"unknown baseline {name!r}; known: {[k for k in BASELINES if k]}")
    return predictor


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_baseline(name: str, predict: Any, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for task in tasks:
        t0 = time.perf_counter()
        try:
            decision, error = predict(task["question"])
        except Exception as exc:  # noqa: BLE001
            decision = RoutingDecision(route_type="clarification", tools=[],
                                       reason="predictor crashed", confidence=0.0,
                                       requires_clarification=True)
            error = str(exc)
        latency = round((time.perf_counter() - t0) * 1000.0, 2)
        metrics = score(
            decision,
            expected_route=task["expected_route"],
            expected_tools=task["expected_tools"],
            expected_order=task.get("expected_order"),
            predicted_error=error,
        )
        results.append({
            "id": task["id"],
            "question": task["question"],
            "task_type": task["task_type"],
            "difficulty": task["difficulty"],
            "expected_route": task["expected_route"],
            "expected_tools": task["expected_tools"],
            "predicted_route": decision.route_type,
            "predicted_tools": list(decision.tools),
            "predicted_order": [s.tool for s in sorted(decision.execution_plan, key=lambda s: s.step)],
            "error": error,
            "latency_ms": latency,
            **metrics,
        })
    return {"baseline": name, "results": results, "summary": _summarize(results)}


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"tasks": 0}

    def agg(subset: list[dict[str, Any]]) -> dict[str, Any]:
        m = len(subset)
        if m == 0:
            return {"tasks": 0}
        lats = [r["latency_ms"] for r in subset]
        return {
            "tasks": m,
            "route_accuracy": round(sum(r["route_correct"] for r in subset) / m, 4),
            "tool_f1": round(statistics.mean(r["tool_f1"] or 0.0 for r in subset), 4),
            "tool_precision": round(statistics.mean(r["tool_precision"] or 0.0 for r in subset), 4),
            "tool_recall": round(statistics.mean(r["tool_recall"] or 0.0 for r in subset), 4),
            "plan_exact_match": round(sum(r["plan_exact"] for r in subset) / m, 4),
            "clarification_accuracy": round(sum(r["clarification_correct"] for r in subset) / m, 4),
            "task_success": round(sum(r["task_success"] for r in subset) / m, 4),
            "failure_count": sum(1 for r in subset if r.get("error")),
            "avg_latency_ms": round(statistics.mean(lats), 2),
            "p50_latency_ms": round(statistics.median(lats), 2),
            "p95_latency_ms": round(_pct(lats, 95), 2),
        }

    out: dict[str, Any] = {"tasks": n, "overall": agg(results),
                           "by_task_type": {}, "by_difficulty": {}}
    for ttype in sorted({r["task_type"] for r in results}):
        out["by_task_type"][ttype] = agg([r for r in results if r["task_type"] == ttype])
    for diff in ("easy", "medium", "hard"):
        sub = [r for r in results if r["difficulty"] == diff]
        if sub:
            out["by_difficulty"][diff] = agg(sub)
    return out


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def load_tasks(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks[:limit] if limit else tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 3 routing evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--baseline", action="append", default=None,
                        help="run only these baselines (repeatable)")
    args = parser.parse_args(argv)

    tasks = load_tasks(DATASET, args.limit)
    selected = args.baseline or list(BASELINES)
    report: dict[str, Any] = {"dataset": str(DATASET), "tasks": len(tasks), "baselines": {}}

    for name in selected:
        predict = _get_predictor(name)
        print(f"running baseline {name} over {len(tasks)} tasks …")
        report["baselines"][name] = run_baseline(name, predict, tasks)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "routing_eval_summary.json").write_text(
        json.dumps({k: v["summary"] for k, v in report["baselines"].items()},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (ARTIFACT_DIR / "routing_eval_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(report, ARTIFACT_DIR / "routing_eval_summary.md")

    for name, payload in report["baselines"].items():
        o = payload["summary"]["overall"]
        print(f"{name:16s} route_acc={o['route_accuracy']:.3f} tool_f1={o['tool_f1']:.3f} "
              f"plan_em={o['plan_exact_match']:.3f} task_success={o['task_success']:.3f} "
              f"lat={o['avg_latency_ms']:.1f}ms")
    print(f"artifacts written to {ARTIFACT_DIR}")
    return 0


def _write_md(report: dict[str, Any], path: Path) -> None:
    lines = ["# Phase 3 Routing Evaluation Summary", "",
             f"- dataset: {report['dataset']}", f"- tasks: {report['tasks']}", "",
             "## Overall", "",
             "| baseline | route acc | tool F1 | tool P | tool R | plan EM | clarif acc | task success | avg lat (ms) | failures |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, payload in report["baselines"].items():
        o = payload["summary"]["overall"]
        lines.append(
            f"| {name} | {o['route_accuracy']:.3f} | {o['tool_f1']:.3f} | {o['tool_precision']:.3f} | "
            f"{o['tool_recall']:.3f} | {o['plan_exact_match']:.3f} | {o['clarification_accuracy']:.3f} | "
            f"{o['task_success']:.3f} | {o['avg_latency_ms']:.1f} | {o['failure_count']} |"
        )
    lines += ["", "## By task type (route accuracy / tool F1)", "",
              "| baseline | " + " | ".join(sorted({
                  t for p in report["baselines"].values() for t in p["summary"]["by_task_type"]
              })) + " |", "|---" + "|---" * len({
                  t for p in report["baselines"].values() for t in p["summary"]["by_task_type"]
              }) + "|"]
    for name, payload in report["baselines"].items():
        cells = []
        for ttype in sorted(payload["summary"]["by_task_type"]):
            d = payload["summary"]["by_task_type"][ttype]
            cells.append(f"{d['route_accuracy']:.2f}/{d['tool_f1']:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## By difficulty (route accuracy / task success)", "",
              "| baseline | easy | medium | hard |", "|---|---|---|---|"]
    for name, payload in report["baselines"].items():
        cells = []
        for diff in ("easy", "medium", "hard"):
            d = payload["summary"]["by_difficulty"].get(diff)
            cells.append(f"{d['route_accuracy']:.2f}/{d['task_success']:.2f}" if d else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
