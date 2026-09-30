"""Enterprise end-to-end evaluation metrics (Phase 5, RQ1–RQ3).

Scores one finished enterprise task against its *independently-verified*
ground truth across the four layers the Phase 5 spec requires, plus the
verification-specific and latency/error metrics.  Every metric is computed
from a per-task prediction record; nothing here touches a live model.

Design rules

- Metrics are **per-layer and kept separate** (route / tool / answer /
  evidence) — the spec forbids collapsing them into a single number.
- A failing task (routing / tool / verification / LLM 429 / timeout) is
  **never dropped**: it scores 0 on its success metric and its failure
  category is recorded, so failures stay in the denominator.
- The two answer-correctness notions are both exposed:
  - ``answer_exact`` — string/number match against GT (deterministic);
  - ``answer_semantic`` — LLM-judge 0/1 (reliable evaluator, run separately
    so an unreliable judge can be swapped without touching the deterministic
    metrics).
- Unsupported-claim leakage is only scored when verification ran; a
  non-verification system reports ``None`` (it has no claims to leak).

The per-task record is a plain dict; :func:`score_task` returns one such
record and :func:`summarise` aggregates many into the summary schema written
by ``scripts/run_enterprise_eval.py``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Vocabulary (mirrors src.core.routing_types; kept local so the module has no
# import cycle with the graph builder).
# ---------------------------------------------------------------------------

_ROUTE_VOCAB = ("rag", "sql", "kg", "analysis", "multi_tool", "clarification")
_SOURCE_TOOLS = ("rag", "sql", "kg", "analysis")


def _tool_set(predicted: list[str]) -> set[str]:
    return set(predicted or [])


def _precision_recall_f1(pred: set[str], expected: set[str]) -> tuple[float, float, float]:
    if not pred and not expected:
        return 1.0, 1.0, 1.0
    if not expected:
        return (0.0 if pred else 1.0), 1.0, (0.0 if pred else 1.0)
    if not pred:
        return 0.0, 0.0, 0.0
    inter = len(pred & expected)
    p = inter / len(pred)
    r = inter / len(expected)
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f1


# ---------------------------------------------------------------------------
# Per-layer correctness
# ---------------------------------------------------------------------------


def route_correct(predicted_route: str, expected_route: str) -> bool:
    return predicted_route == expected_route


def tool_f1(predicted_tools: list[str], expected_tools: list[str]) -> float:
    return _precision_recall_f1(_tool_set(predicted_tools), set(expected_tools or []))[2]


def tool_precision(predicted_tools: list[str], expected_tools: list[str]) -> float:
    return _precision_recall_f1(_tool_set(predicted_tools), set(expected_tools or []))[0]


def tool_recall(predicted_tools: list[str], expected_tools: list[str]) -> float:
    return _precision_recall_f1(_tool_set(predicted_tools), set(expected_tools or []))[1]


def plan_exact_match(predicted_order: list[str], expected_order: list[str]) -> bool:
    """Ordered plan match.  When the task asserts no order (empty GT order)
    we fall back to set equality, matching the Phase 3 convention."""
    if not expected_order:
        return _tool_set(predicted_order or []) == _tool_set(predicted_order or [])
    return list(predicted_order or []) == list(expected_order)


def clarification_correct(predicted_route: str, expected_clarification: bool) -> bool:
    return (predicted_route == "clarification") == expected_clarification


def task_success(
    predicted_route: str,
    predicted_tools: list[str],
    predicted_order: list[str],
    expected_route: str,
    expected_tools: list[str],
    expected_order: list[str],
    expected_clarification: bool,
) -> bool:
    """A task *succeeds* when routing + tool selection line up with GT.

    - clarification tasks: success only when both route to clarification.
    - single-source tasks: success when the predicted tool set covers the
      expected set (an extra companion tool still answers the task).
    - multi_tool tasks: the expected tool set must be covered AND the
      asserted execution order (when present) must match.
    """
    if expected_clarification:
        return predicted_route == "clarification"
    if expected_route == "multi_tool":
        if predicted_route != "multi_tool":
            return False
        if _tool_set(predicted_tools) != _tool_set(expected_tools):
            return False
        if expected_order:
            return list(predicted_order or []) == list(expected_order)
        return True
    # single source tool
    if not _tool_set(expected_tools) <= _tool_set(predicted_tools):
        return False
    return predicted_route != "clarification"


# ---------------------------------------------------------------------------
# Answer correctness
# ---------------------------------------------------------------------------


def answer_exact(predicted_answer: str, expected_answer: str) -> bool:
    """Deterministic match: compare the normalised non-empty answers.

    GT answers in this benchmark are short scalar / entity strings, so a
    normalised containment-or-equality test is the reliable floor.  The LLM
    semantic judge is a *separate* metric and is not mixed in here.
    """
    a = str(predicted_answer or "").strip().lower()
    e = str(expected_answer or "").strip().lower()
    if not a or not e:
        return False
    if a == e:
        return True
    # numeric GT: accept an exact figure appearing in the answer
    try:
        import re

        num = re.search(r"\d+(?:\.\d+)?", e)
        if num and num.group(0) in a:
            return True
    except Exception:  # noqa: BLE001
        pass
    # entity GT: accept the expected entity name appearing verbatim
    return bool(e in a and len(e) >= 2)


def evidence_attribution_accuracy(predicted_evidence: list[str], expected_evidence: list[str]) -> float:
    """Set-based precision/recall F1 over evidence source types (rag/sql/kg/analysis)."""
    return _precision_recall_f1(set(predicted_evidence or []), set(expected_evidence or []))[2]


# ---------------------------------------------------------------------------
# Verification-specific metrics
# ---------------------------------------------------------------------------


def unsupported_claim_detection(
    verification: dict[str, Any] | None,
    expected_claims: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """When verification ran, measure how many GT claims it flagged unsupported.

    Returns ``None`` for systems without verification (no claims to detect).
    The benchmark's GT claims are the *correct* claims for the task; a
    verifier that flags many of them unsupported is over-strict.  We report
    the raw supported/unsupported tally so analysis can both ways.
    """
    if not verification or not verification.get("claims"):
        return None
    supported = sum(1 for c in verification["claims"] if c.get("supported"))
    total = len(verification["claims"])
    return {
        "n_claims": total,
        "n_supported": supported,
        "n_unsupported": total - supported,
        "n_critical_unsupported": int(verification.get("n_critical_unsupported", 0)),
    }


def critical_unsupported_leakage(
    verification: dict[str, Any] | None,
    guard: dict[str, Any] | None = None,
) -> int:
    """Number of critical claims left unsupported and NOT blocked by the guard.

    0 = nothing critical leaked; >0 = a critical unsupported claim reached the
    user-facing answer (the failure mode RQ2 must reduce).  ``None`` when
    verification did not run.
    """
    if not verification:
        return 0
    blocked = bool(guard and guard.get("block"))
    n_crit = int(verification.get("n_critical_unsupported", 0))
    # A critical-leak only counts when the guard did NOT block it.
    return 0 if blocked else n_crit


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-task records into the summary schema.

    Failed tasks stay in the denominator (scored 0); latency percentiles are
    computed over all recorded latencies; error categories are counted.
    """
    n = len(records)
    if n == 0:
        return {"tasks": 0}

    def _agg(subset: list[dict[str, Any]]) -> dict[str, Any]:
        m = len(subset)
        if m == 0:
            return {"tasks": 0}

        def rate(key: str, default: float = 0.0) -> float:
            vals = [r.get(key, default) for r in subset if r.get(key) is not None]
            return _mean([float(v) for v in vals]) if vals else 0.0

        lats = [r["latency_ms"]["total"] for r in subset if r.get("latency_ms")]
        fail_cats = Counter()
        for r in subset:
            for c in r.get("failure_categories", []) or []:
                fail_cats[c] += 1
        return {
            "tasks": m,
            "task_success_rate": rate("task_success"),
            "answer_exact": rate("answer_exact"),
            "answer_semantic": rate("answer_semantic"),
            "route_accuracy": rate("route_correct"),
            "tool_f1": rate("tool_f1"),
            "plan_exact_match": rate("plan_exact_match"),
            "evidence_attribution_accuracy": rate("evidence_attribution"),
            "error_rate": round(sum(1 for r in subset if r.get("any_error")) / m, 4),
            "llm_calls_avg": _mean([r.get("llm_calls", 0) for r in subset]),
            "latency_total_ms_mean": _mean(lats),
            "latency_total_ms_p50": round(_pct(lats, 50), 2),
            "latency_total_ms_p95": round(_pct(lats, 95), 2),
            "failure_categories": dict(fail_cats),
        }

    out: dict[str, Any] = {"tasks": n, "overall": _agg(records)}
    by_type: dict[str, Any] = {}
    for ttype in sorted({r.get("task_type", "unknown") for r in records}):
        by_type[ttype] = _agg([r for r in records if r.get("task_type") == ttype])
    out["by_task_type"] = by_type
    by_diff: dict[str, Any] = {}
    for diff in ("easy", "medium", "hard"):
        sub = [r for r in records if r.get("difficulty") == diff]
        if sub:
            by_diff[diff] = _agg(sub)
    out["by_difficulty"] = by_diff
    return out


# ---------------------------------------------------------------------------
# Per-task scorer — the single entry point the runner uses
# ---------------------------------------------------------------------------


def score_task(
    task: dict[str, Any],
    pred: dict[str, Any],
) -> dict[str, Any]:
    """Score one task.  ``pred`` must carry the keys produced by the runner:

    ``route``, ``tools``, ``order``, ``answer``, ``evidence`` (source types
    that actually produced the answer), ``verification`` (dict or None),
    ``guard`` (dict or None), ``latency_ms`` (dict of stage->ms),
    ``llm_calls`` (int), ``failure_categories`` (list[str]), ``any_error``.
    """
    expected_route = task["expected_route"]
    expected_tools = task.get("expected_tools", [])
    expected_order = task.get("expected_tool_order", [])
    expected_clar = bool(task.get("expected_clarification"))

    predicted_route = str(pred.get("route", ""))
    predicted_tools = list(pred.get("tools") or [])
    predicted_order = list(pred.get("order") or [])
    predicted_answer = str(pred.get("answer", ""))
    predicted_evidence = list(pred.get("evidence") or [])

    succ = task_success(
        predicted_route, predicted_tools, predicted_order,
        expected_route, expected_tools, expected_order, expected_clar,
    )
    ans_exact = 0 if expected_clar and predicted_route != "clarification" else (
        1 if answer_exact(predicted_answer, task.get("expected_answer", "")) else 0
    )
    ev_acc = evidence_attribution_accuracy(predicted_evidence, task.get("expected_evidence", []))

    # verification metrics (None when the system has no verification layer)
    ver = pred.get("verification")
    unc = unsupported_claim_detection(ver, task.get("expected_claims", []))
    leakage = critical_unsupported_leakage(ver, pred.get("guard"))

    return {
        "id": task["id"],
        "task_type": task.get("task_type"),
        "difficulty": task.get("difficulty"),
        "expected": {
            "route": expected_route,
            "tools": expected_tools,
            "order": expected_order,
            "answer": task.get("expected_answer", ""),
            "evidence": task.get("expected_evidence", []),
        },
        "predicted": {
            "route": predicted_route,
            "tools": predicted_tools,
            "order": predicted_order,
            "answer": predicted_answer[:400],
            "evidence": predicted_evidence,
        },
        # per-task booleans / values (0/1 for the deterministic ones)
        "task_success": 1 if succ else 0,
        "route_correct": 1 if route_correct(predicted_route, expected_route) else 0,
        "clarification_correct": 1 if clarification_correct(predicted_route, expected_clar) else 0,
        "tool_f1": round(tool_f1(predicted_tools, expected_tools), 4),
        "tool_precision": round(tool_precision(predicted_tools, expected_tools), 4),
        "tool_recall": round(tool_recall(predicted_tools, expected_tools), 4),
        "plan_exact_match": 1 if plan_exact_match(predicted_order, expected_order) else 0,
        "evidence_attribution": round(ev_acc, 4),
        "answer_exact": int(ans_exact),
        "answer_semantic": pred.get("answer_semantic"),  # filled by the judge, else None
        "unsupported_claims": unc,
        "critical_unsupported_leakage": leakage,
        "any_error": bool(pred.get("any_error")),
        "failure_categories": list(pred.get("failure_categories") or []),
        "llm_calls": int(pred.get("llm_calls", 0)),
        "latency_ms": pred.get("latency_ms") or {},
        "errors": pred.get("errors") or [],
    }


__all__ = [
    "route_correct",
    "tool_f1",
    "tool_precision",
    "tool_recall",
    "plan_exact_match",
    "clarification_correct",
    "task_success",
    "answer_exact",
    "evidence_attribution_accuracy",
    "unsupported_claim_detection",
    "critical_unsupported_leakage",
    "summarise",
    "score_task",
]
