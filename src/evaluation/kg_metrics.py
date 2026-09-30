"""KG evaluation metrics — Phase 2.4 Commit 4.

Metrics are computed over one task's *predicted* result and its
*expected* ground truth.  They are deliberately simple (no invented
complex metrics):

- ``exact_match``: predicted answer equals expected answer (string compare
  after normalisation, or set compare when the expected value is a list).
- ``set_precision`` / ``set_recall`` / ``set_f1``: computed on the
  predicted vs. expected entity-id sets.
- ``path_accuracy``: fraction of *required* relationship hops in the
  expected path that the predicted path actually traverses.  Only defined
  when the task's ground truth declares ``expected_relations``; otherwise
  the metric is ``None`` and excluded from aggregation.

All metrics return values in [0, 1] (or ``None`` when not applicable).
"""

from __future__ import annotations

from typing import Any


def _as_set(value: Any) -> set:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return set(value)
    return {value}


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match(predicted: Any, expected: Any) -> bool:
    """Return True when *predicted* equals *expected*.

    - scalar expected: case/space-normalised string equality
    - list expected:    set equality of ids
    - dict expected:    every key in ``expected`` must be present in
      ``predicted`` with a matching value (superset is allowed)
    """
    if isinstance(expected, dict):
        if not isinstance(predicted, dict):
            return False
        return all(predicted.get(k) == v for k, v in expected.items())
    if isinstance(expected, (list, tuple, set, frozenset)):
        return _as_set(predicted) == _as_set(expected)
    if predicted is None or expected is None:
        return predicted is expected
    return _normalize(str(predicted)) == _normalize(str(expected))


def set_precision(predicted: Any, expected: Any) -> float:
    """|predicted ∩ expected| / |predicted| (1.0 when expected is empty and predicted is empty)."""
    p = _as_set(predicted)
    e = _as_set(expected)
    if not e:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    return len(p & e) / len(p)


def set_recall(predicted: Any, expected: Any) -> float:
    """|predicted ∩ expected| / |expected| (1.0 when expected is empty)."""
    p = _as_set(predicted)
    e = _as_set(expected)
    if not e:
        return 1.0
    if not p:
        return 0.0
    return len(p & e) / len(e)


def set_f1(predicted: Any, expected: Any) -> float:
    """Harmonic mean of set precision and recall (0.0 when both are undefined)."""
    p = set_precision(predicted, expected)
    r = set_recall(predicted, expected)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def path_accuracy(
    predicted_relations: list[tuple[str, str, str]],
    expected_relations: list[tuple[str, str, str]],
) -> float | None:
    """Fraction of expected (from_label, rel_type, to_label) hops present in the predicted path.

    Returns ``None`` when *expected_relations* is empty — the metric is
    undefined for tasks whose ground truth does not declare a required
    relationship path.
    """
    if not expected_relations:
        return None
    expected_set = {
        (str(a).upper(), str(b).upper(), str(c).upper())
        for a, b, c in expected_relations
    }
    predicted_set = {
        (str(a).upper(), str(b).upper(), str(c).upper())
        for a, b, c in predicted_relations
    }
    return len(expected_set & predicted_set) / len(expected_set)


__all__ = [
    "exact_match",
    "path_accuracy",
    "set_f1",
    "set_precision",
    "set_recall",
]
