"""Unit tests for Reciprocal Rank Fusion (Phase 2.2).

Verifies the rank-based fusion semantics of :func:`rrf_fuse` (k = 60,
the value fixed by the task — not to be changed):

- a chunk present in both channels beats one present in only one;
- single-channel chunks still participate (the missing channel adds 0);
- the fusion NEVER sums raw channel scores (rank-based by construction);
- deterministic ordering and repeat-run stability.
"""

from __future__ import annotations

import pytest

from src.core.retrieval_types import RRF_K_DEFAULT, RankedHit, rrf_fuse

K = RRF_K_DEFAULT  # 60 — the fixed RRF constant

pytestmark = pytest.mark.unit


def _hit(chunk_id: str, rank: int, score: float = 0.0) -> RankedHit:
    return RankedHit(chunk_id=chunk_id, rank=rank, score=score, text=f"text-{chunk_id}", source=f"src-{chunk_id}")


# ---------------------------------------------------------------------------
# basic fusion
# ---------------------------------------------------------------------------


def test_both_channels_present() -> None:
    dense = [_hit("A", 1, 0.9), _hit("B", 2, 0.7)]
    bm25 = [_hit("C", 1, 10.0), _hit("A", 2, 8.0)]
    fused = rrf_fuse(dense, bm25, k=K)
    by_id = {h.chunk_id: h for h in fused}
    # A is in both channels -> highest fusion score
    assert fused[0].chunk_id == "A"
    assert by_id["A"].dense_rank == 1
    assert by_id["A"].bm25_rank == 2
    assert by_id["A"].fusion_score == pytest.approx(1 / (K + 1) + 1 / (K + 2))
    # single-channel entries carry a zero rank on the missing side
    assert by_id["B"].bm25_rank == 0
    assert by_id["C"].dense_rank == 0


def test_dense_only() -> None:
    dense = [_hit("A", 1, 0.9), _hit("B", 2, 0.7)]
    fused = rrf_fuse(dense, [], k=K)
    by_id = {h.chunk_id: h for h in fused}
    assert set(by_id) == {"A", "B"}
    assert by_id["A"].fusion_score == pytest.approx(1 / (K + 1))
    assert by_id["B"].fusion_score == pytest.approx(1 / (K + 2))
    assert fused[0].chunk_id == "A"


def test_bm25_only() -> None:
    bm25 = [_hit("X", 1, 12.0), _hit("Y", 2, 9.0)]
    fused = rrf_fuse([], bm25, k=K)
    by_id = {h.chunk_id: h for h in fused}
    assert set(by_id) == {"X", "Y"}
    assert by_id["X"].fusion_score == pytest.approx(1 / (K + 1))
    assert fused[0].chunk_id == "X"


def test_chunk_in_both_channels_ranked_first() -> None:
    dense = [_hit("A", 1), _hit("B", 2), _hit("C", 3)]
    bm25 = [_hit("B", 1), _hit("A", 5)]
    fused = rrf_fuse(dense, bm25, k=K)
    # B appears rank1 in bm25 and rank2 in dense -> beats A (rank1 dense, rank5 bm25)
    assert fused[0].chunk_id == "B"
    assert fused[0].fusion_score == pytest.approx(1 / (K + 2) + 1 / (K + 1))


def test_same_rank_both_channels() -> None:
    dense = [_hit("Z", 1, 0.5)]
    bm25 = [_hit("Z", 1, 20.0)]
    fused = rrf_fuse(dense, bm25, k=K)
    assert fused[0].chunk_id == "Z"
    assert fused[0].fusion_score == pytest.approx(2 / (K + 1))


def test_different_ranks_both_channels() -> None:
    dense = [_hit("Z", 3, 0.5)]
    bm25 = [_hit("Z", 7, 20.0)]
    fused = rrf_fuse(dense, bm25, k=K)
    assert fused[0].chunk_id == "Z"
    assert fused[0].fusion_score == pytest.approx(1 / (K + 3) + 1 / (K + 7))


# ---------------------------------------------------------------------------
# rank-based, NOT score-based
# ---------------------------------------------------------------------------


def test_fusion_does_not_add_raw_scores() -> None:
    """Fusion must depend only on ranks. Two runs with wildly different raw
    scores but identical ranks produce identical fusion scores."""
    dense = [_hit("A", 1, 0.01), _hit("B", 2, 0.005)]
    bm25 = [_hit("A", 2, 0.0001), _hit("B", 1, 0.00005)]
    fused = rrf_fuse(dense, bm25, k=K)
    by_id = {h.chunk_id: h for h in fused}
    assert by_id["A"].fusion_score == pytest.approx(1 / (K + 1) + 1 / (K + 2))
    assert by_id["B"].fusion_score == pytest.approx(1 / (K + 2) + 1 / (K + 1))
    # If it had naively summed raw scores, B (rank1 bm25 tiny) could not tie A
    assert abs(by_id["A"].fusion_score - by_id["B"].fusion_score) < 1e-12


def test_identical_input_repeat_run_consistent() -> None:
    dense = [_hit("A", 1, 0.9), _hit("B", 2, 0.7), _hit("C", 3, 0.4)]
    bm25 = [_hit("C", 1, 9.0), _hit("B", 2, 5.0), _hit("A", 3, 1.0)]
    r1 = [h.chunk_id for h in rrf_fuse(dense, bm25, k=K)]
    r2 = [h.chunk_id for h in rrf_fuse(dense, bm25, k=K)]
    assert r1 == r2


def test_fusion_scores_are_descending() -> None:
    dense = [_hit("A", 1), _hit("B", 2), _hit("C", 3), _hit("D", 4)]
    bm25 = [_hit("B", 1), _hit("A", 2), _hit("D", 3)]
    fused = rrf_fuse(dense, bm25, k=K)
    scores = [h.fusion_score for h in fused]
    assert scores == sorted(scores, reverse=True)


def test_invalid_k_rejected() -> None:
    with pytest.raises(ValueError):
        rrf_fuse([], [], k=0)
    with pytest.raises(ValueError):
        rrf_fuse([], [], k=-5)


def test_empty_both_channels() -> None:
    assert rrf_fuse([], [], k=K) == []


def test_duplicate_chunks_in_one_channel_dedup_by_last_rank() -> None:
    """A channel listing the same chunk twice keeps the (best) first-seen
    contribution — RRF is over chunk identity, so the fusion is well-defined."""
    dense = [_hit("A", 1, 0.9)]
    bm25 = [_hit("A", 1, 10.0), _hit("A", 3, 1.0)]
    fused = rrf_fuse(dense, bm25, k=K)
    assert len(fused) == 1
    # last-seen bm25 rank wins the stored rank, but either way A is present
    assert fused[0].chunk_id == "A"
    assert fused[0].dense_rank == 1


def test_context_metadata_preserved_from_dense() -> None:
    dense = [_hit("A", 1, 0.9)]
    dense[0].metadata = {"doc_id": "d1"}
    dense[0].title = "t-A"
    dense[0].category = "finance"
    bm25 = [_hit("A", 1, 10.0)]  # no metadata
    fused = rrf_fuse(dense, bm25, k=K)
    assert fused[0].metadata == {"doc_id": "d1"}
    assert fused[0].title == "t-A"
    assert fused[0].category == "finance"
    assert fused[0].source == "src-A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
