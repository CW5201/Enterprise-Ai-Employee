"""Retrieval experiment metrics — Recall@K / MRR / NDCG@5 (Phase 2.3).

Pure functions over a retrieved ordering and the per-query ground truth.
Used by ``scripts/run_retrieval_eval.py`` so every reported number is
computed by the program (never hand-filled).

Definitions
-----------
- Relevant(query)  = expected_chunk_ids for that query.
- Recall@K         = |Relevant ∩ Retrieved@K| / |Relevant|   (0.0 when no
  ground-truth relevant chunk exists — recorded, not silently skipped).
- MRR              = mean over queries of 1/rank(first relevant in the
  retrieved list); 0.0 when no retrieved chunk is relevant.
- NDCG@K           = standard NDCG with binary relevance
  (rel = 1 if chunk in Relevant else 0):
      DCG@K  = sum_{i=1..K} rel_i / log2(i + 1)
      IDCG@K = DCG@K of the ideal ordering (all relevants first).
      NDCG@K = DCG@K / IDCG@K  (0.0 when IDCG@K == 0, i.e. no relevant
  ground truth).  Single-relevant ground truths work out of the box.

All functions are deterministic and allocation-light; they never touch the
retrieval backend, so they are trivially unit-testable in isolation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Recall@K for one query: |top-K ∩ relevant| / |relevant|."""
    if not relevant:
        # Honest zero: the query has no ground-truth relevant chunk.  We keep
        # it in the denominator (it contributes 0.0) rather than dropping it.
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """RR for one query: 1/rank of the first relevant result, else 0.0."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += rel / math.log2(i + 1)
    return total


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
    *,
    ranking_relevances: Sequence[float] | None = None,
) -> float:
    """NDCG@K for one query with binary (or graded) relevance.

    With the default binary relevance, ``rel_i = 1`` when the i-th retrieved
    chunk is in ``relevant``.  Pass ``ranking_relevances`` (a parallel list of
    graded relevance per retrieved position) to override for the DCG side;
    the ideal ordering always uses the binary top-|relevant| assumption,
    which is the standard graded NDCG convention.
    """
    if not relevant:
        return 0.0
    if ranking_relevances is None:
        retrieved_rels = [1.0 if c in relevant else 0.0 for c in retrieved[:k]]
    else:
        retrieved_rels = [float(r) for r in list(ranking_relevances)[:k]]

    # Ideal DCG: all relevant items first.  Binary case: k ideal items, each
    # with relevance 1, capped at |relevant| (you cannot rank more relevants
    # than exist).
    n_relevant = min(len(relevant), k)
    ideal_rels = [1.0] * n_relevant
    idcg = _dcg(ideal_rels, k)
    if idcg == 0.0:
        return 0.0
    return _dcg(retrieved_rels, k) / idcg


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sequence (never NaN)."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of non-negative latencies; 0.0 if empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[rank]


__all__ = [
    "mean",
    "ndcg_at_k",
    "percentile",
    "recall_at_k",
    "reciprocal_rank",
]
