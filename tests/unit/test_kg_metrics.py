"""Unit tests for src/evaluation/kg_metrics.py (no Neo4j required)."""

from __future__ import annotations

import pytest

from src.evaluation.kg_metrics import (
    exact_match,
    path_accuracy,
    set_f1,
    set_precision,
    set_recall,
)


class TestSetMetrics:
    def test_perfect(self) -> None:
        assert set_precision({1, 2}, {1, 2}) == 1.0
        assert set_recall({1, 2}, {1, 2}) == 1.0
        assert set_f1({1, 2}, {1, 2}) == 1.0

    def test_empty_expected(self) -> None:
        assert set_recall({1}, set()) == 1.0
        assert set_precision(set(), set()) == 1.0
        assert set_precision({1}, set()) == 0.0

    def test_partial(self) -> None:
        assert set_precision({1, 2}, {1, 3}) == 0.5
        assert set_recall({1, 2}, {1, 3}) == 0.5
        assert set_f1({1, 2}, {1, 3}) == pytest.approx(0.5)

    def test_f1_zero_when_both_zero(self) -> None:
        # predicted non-empty, expected empty -> recall 1.0, precision 0.0
        assert set_f1({1}, set()) == pytest.approx(0.0)

    def test_scalars_and_lists_normalise(self) -> None:
        assert set_recall([1, 2], {1, 2}) == 1.0
        assert set_precision(1, [1]) == 1.0


class TestExactMatch:
    def test_string_normalisation(self) -> None:
        assert exact_match("Yes", " yes ")
        assert not exact_match("Yes", "no")

    def test_set_equality(self) -> None:
        assert exact_match([1, 501], [501, 1])
        assert not exact_match([1], [1, 2])

    def test_dict_superset_ok(self) -> None:
        assert exact_match({"a": 1, "b": 2}, {"a": 1})
        assert not exact_match({"a": 1}, {"a": 1, "b": 2})

    def test_none(self) -> None:
        assert exact_match(None, None)
        assert not exact_match(None, 1)


class TestPathAccuracy:
    def test_full_path(self) -> None:
        exp = [("Customer", "PLACED", "Order"), ("Order", "HAS_LINE", "StockItem")]
        assert path_accuracy(list(exp), exp) == 1.0

    def test_partial_path(self) -> None:
        exp = [("Customer", "PLACED", "Order"), ("Order", "HAS_LINE", "StockItem")]
        pred = [("Customer", "PLACED", "Order")]
        assert path_accuracy(pred, exp) == 0.5

    def test_case_insensitive(self) -> None:
        exp = [("Customer", "PLACED", "Order")]
        pred = [("customer", "placed", "order")]
        assert path_accuracy(pred, exp) == 1.0

    def test_undefined_when_no_expected_hops(self) -> None:
        assert path_accuracy([("a", "b", "c")], []) is None

    def test_unrelated_hops(self) -> None:
        assert path_accuracy([("X", "Y", "Z")], [("Customer", "PLACED", "Order")]) == 0.0
