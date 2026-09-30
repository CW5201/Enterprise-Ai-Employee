"""Unit tests for retrieval experiment metrics (Phase 2.3).

Covers the pure functions in ``src/evaluation/retrieval_metrics.py``:
Recall@K, MRR, NDCG@5 (binary + single-relevant), no-relevant ground truth,
first-relevant-rank edge cases, and the latency helpers.
"""

from __future__ import annotations

import os

import pytest

os.environ["EMBEDDING_FORCE_OFFLINE"] = "1"

from src.evaluation.retrieval_metrics import (  # noqa: E402
    mean,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

pytestmark = pytest.mark.unit

REL = {"a", "b"}
REL_ONE = {"a"}


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


def test_recall_at_k_multi_relevant() -> None:
    assert recall_at_k(["a", "b", "c"], REL, 2) == 1.0
    assert recall_at_k(["c", "a"], REL, 1) == 0.0  # top-1 is 'c' — not relevant
    assert recall_at_k(["c", "a", "b"], REL, 3) == 1.0
    assert recall_at_k(["x", "y"], REL, 5) == 0.0


def test_recall_at_k_single_relevant() -> None:
    assert recall_at_k(["x", "a"], REL_ONE, 2) == 1.0
    assert recall_at_k(["x", "y"], REL_ONE, 5) == 0.0


def test_recall_at_k_no_relevant_ground_truth_is_zero_not_nan() -> None:
    assert recall_at_k(["a", "b"], set(), 3) == 0.0


# ---------------------------------------------------------------------------
# MRR / reciprocal rank
# ---------------------------------------------------------------------------


def test_reciprocal_rank_positions() -> None:
    assert reciprocal_rank(["a", "x"], REL) == 1.0
    assert reciprocal_rank(["x", "a"], REL) == 0.5
    assert reciprocal_rank(["x", "y", "a"], REL) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_relevant_is_zero() -> None:
    assert reciprocal_rank(["x", "y"], REL) == 0.0
    assert reciprocal_rank([], REL) == 0.0


def test_mrr_mean_aggregates_cleanly() -> None:
    # One query at rank1, one at rank3 -> MRR = (1 + 1/3)/2
    vals = [reciprocal_rank(["a", "x"], REL), reciprocal_rank(["x", "y", "a"], REL)]
    assert mean(vals) == pytest.approx((1.0 + 1 / 3) / 2)


# ---------------------------------------------------------------------------
# NDCG@K
# ---------------------------------------------------------------------------


def test_ndcg_perfect_ranking() -> None:
    # Both relevants on top -> NDCG = 1
    assert ndcg_at_k(["a", "b", "c"], REL, 5) == pytest.approx(1.0)


def test_ndcg_single_relevant_at_rank2() -> None:
    # One relevant at position 2: DCG = 0/log2(1)?? no — i=2 -> 1/log2(3).
    # IDCG (relevant at position 1) = 1/log2(2) = 1.0
    got = ndcg_at_k(["x", "a"], REL_ONE, 5)
    import math

    assert got == pytest.approx((1.0 / math.log2(3)) / 1.0)


def test_ndcg_single_relevant_at_top_is_one() -> None:
    assert ndcg_at_k(["a", "x"], REL_ONE, 5) == pytest.approx(1.0)


def test_ndcg_worst_ranking_below_best() -> None:
    best = ndcg_at_k(["a", "b", "c"], REL, 5)
    worst = ndcg_at_k(["c", "x", "a"], REL, 5)
    assert best > worst
    assert 0.0 < worst < 1.0


def test_ndcg_no_relevant_ground_truth_is_zero() -> None:
    assert ndcg_at_k(["a", "b"], set(), 5) == 0.0


def test_ndcg_respects_k_cutoff() -> None:
    # A relevant at position 6 is outside top-5, so NDCG@5 must not credit it.
    assert ndcg_at_k(["c", "d", "e", "f", "g", "a"], REL_ONE, 5) == 0.0
    assert ndcg_at_k(["c", "d", "e", "f", "g", "a"], REL_ONE, 6) > 0.0


def test_ndcg_uses_ideal_denominator_capped_at_relevant_count() -> None:
    # With only 1 relevant, IDCG@5 uses 1 ideal item, not 5.
    import math

    got = ndcg_at_k(["a"], REL_ONE, 5)
    assert got == pytest.approx(1.0 / math.log2(2))


# ---------------------------------------------------------------------------
# latency helpers
# ---------------------------------------------------------------------------


def test_mean_empty_is_zero() -> None:
    assert mean([]) == 0.0


def test_percentile_edge() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([4.0], 95) == 4.0
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 50) == 3.0
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 5.0
