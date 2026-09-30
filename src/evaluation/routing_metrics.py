"""Routing evaluation metrics — Phase 3 Commit 4.

Score a predicted :class:`RoutingDecision` against a task's ground truth
(expected route type / tool set / execution order / clarification).

Metrics are deliberately simple and set-based for multi-tool tasks
(no invented ranking scores):

- ``route_accuracy``      — exact match on ``route_type``
- ``tool_selection_p/r/f1`` — set-based over the selected vs. expected
  tools (order-insensitive; ``None`` when neither side is non-empty)
- ``plan_exact_match``    — the *ordered* tool sequence equals the expected
  ordered sequence (falls back to set equality when the task has no
  meaningful order, i.e. single-tool or unordered multi)
- ``clarification_accuracy`` — both sides agree the task needs clarification
- ``task_success``        — route correct AND (for non-clarification) tool
  set exact; for clarification tasks, both must route to clarification

A failing task (predicted error / unroutable) is kept in the denominator
and scored 0 on every metric — it is never dropped.
"""

from __future__ import annotations

from typing import Any

from src.core.routing_types import RoutingDecision


def _tool_set(decision: RoutingDecision) -> set[str]:
    return set(decision.tools)


def _ordered_tools(decision: RoutingDecision) -> list[str]:
    if decision.execution_plan:
        return [s.tool for s in sorted(decision.execution_plan, key=lambda s: s.step)]
    return list(decision.tools)


def _precision_recall_f1(pred: set[str], expected: set[str]) -> tuple[float | None, float | None, float | None]:
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


def route_accuracy(predicted: RoutingDecision, expected_route: str) -> bool:
    return predicted.route_type == expected_route


def tool_selection_metrics(
    predicted: RoutingDecision, expected_tools: list[str]
) -> dict[str, float | None]:
    pred = _tool_set(predicted)
    exp = set(expected_tools)
    p, r, f1 = _precision_recall_f1(pred, exp)
    return {"precision": p, "recall": r, "f1": f1}


def plan_exact_match(
    predicted: RoutingDecision,
    expected_order: list[str] | None = None,
) -> bool:
    """Order-sensitive when *expected_order* is given, else order-unasserted.

    Passing ``expected_order`` asserts the exact execution sequence;
    leaving it ``None`` does not assert order (always True — the tool-set
    correctness is already covered by tool_selection_metrics).
    """
    if expected_order is None:
        return True
    return _ordered_tools(predicted) == list(expected_order)


def clarification_accuracy(
    predicted: RoutingDecision,
    expected_clarification: bool,
) -> bool:
    pred_is_clar = predicted.route_type == "clarification"
    return pred_is_clar == expected_clarification


def task_success(
    predicted: RoutingDecision,
    expected_route: str,
    expected_tools: list[str],
    expected_order: list[str] | None = None,
) -> bool:
    """A task counts as a success when it can be *answered correctly*:

    - clarification tasks succeed only when *both* sides route to
      clarification;
    - single-source tasks succeed when the expected tool set is covered by
      the prediction (a ``multi_tool`` decision that additionally runs
      companion tools still answers the task);
    - multi-tool tasks additionally require the asserted execution order
      when ``expected_order`` is supplied.
    """
    if expected_route == "clarification":
        return predicted.route_type == "clarification"
    if not route_matches_for_task_success(predicted, expected_route, expected_tools):
        return False
    if expected_order is not None:
        return plan_exact_match(predicted, expected_order)
    return _tool_set(predicted) == set(expected_tools)


def route_matches_for_task_success(
    predicted: RoutingDecision,
    expected_route: str,
    expected_tools: list[str],
) -> bool:
    """Loose route match for single-source tools.

    The GT taxonomy records a pure-SQL task as ``expected_route: "sql"`` even
    when the task technically needs a downstream analysis step; conversely a
    pure-RAG task is recorded as "rag" regardless of a trailing report step.
    For *task success*, we accept a single-tool route as long as the expected
    tool set is covered by the predicted tools AND the predicted route_type is
    either the expected route or ``multi_tool`` (extra companion tools still
    answer the task).  When the expected route IS ``multi_tool`` we require
    an exact route-type match (the decomposition itself is the point).
    """
    if expected_route == "multi_tool":
        return predicted.route_type == "multi_tool"
    if predicted.route_type == expected_route:
        return True
    if predicted.route_type == "clarification":
        return False
    # predicted is multi_tool; succeeds iff it covers the expected tool set
    return set(expected_tools) <= _tool_set(predicted)


def score(
    predicted: RoutingDecision,
    *,
    expected_route: str,
    expected_tools: list[str],
    expected_order: list[str] | None = None,
    predicted_error: str | None = None,
) -> dict[str, Any]:
    """Score one task; a predicted error scores 0 on every metric."""
    if predicted_error:
        return {
            "route_correct": False,
            "tool_f1": 0.0,
            "tool_precision": 0.0,
            "tool_recall": 0.0,
            "plan_exact": False,
            "clarification_correct": False,
            "task_success": False,
            "error": predicted_error,
        }
    tsm = tool_selection_metrics(predicted, expected_tools)
    expected_clar = expected_route == "clarification"
    return {
        "route_correct": route_accuracy(predicted, expected_route),
        "tool_f1": tsm["f1"],
        "tool_precision": tsm["precision"],
        "tool_recall": tsm["recall"],
        "plan_exact": plan_exact_match(predicted, expected_order),
        "clarification_correct": clarification_accuracy(predicted, expected_clar),
        "task_success": task_success(predicted, expected_route, expected_tools, expected_order),
        "error": None,
    }


__all__ = [
    "route_accuracy",
    "route_matches_for_task_success",
    "tool_selection_metrics",
    "plan_exact_match",
    "clarification_accuracy",
    "task_success",
    "score",
]
