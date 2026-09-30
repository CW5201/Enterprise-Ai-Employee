"""Unified retrieval result types + Reciprocal Rank Fusion (Phase 2.2).

``HybridHit`` is the single structure returned by every retrieval mode
(``dense`` / ``bm25`` / ``hybrid``), so upper layers (RAG tool, RAG node,
answer generation) never need to know which channel produced a result:

    {
      "chunk_id": "...",
      "dense_score": 0.0,      # -1.0 when the chunk is absent from that channel
      "dense_rank": 0,         # 1-based; 0 = not in this channel's results
      "bm25_score": 0.0,
      "bm25_rank": 0,
      "fusion_score": 0.0,
      # + preserved context (never dropped)
      "text", "source", "title", "category", "document_id", "metadata"
    }

Fusion is **Reciprocal Rank Fusion** (Cormack et al., 2009)::

    RRF(d) = sum over channels c of  1 / (k + rank_c(d))

Ranks (not raw scores) are fused, so the two channels' incomparable
score scales (inner-product vs BM25) never get added directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RRF_K_DEFAULT = 60  # the RRF constant used in the original paper


@dataclass
class RankedHit:
    """One ranked result from a single retrieval channel."""

    chunk_id: str
    score: float = 0.0
    rank: int = 0
    text: str = ""
    source: str = ""
    title: str = ""
    category: str = ""
    document_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridHit:
    """Fused result across two ranked channels, with context preserved.

    Phase 2.3 adds the optional cross-encoder fields ``rerank_score`` /
    ``rerank_rank``; they are only populated by the ``hybrid_rerank`` mode
    and stay at their defaults (-1.0 / 0 = absent) for ``dense`` /
    ``bm25`` / ``hybrid`` so existing behaviour is unchanged.
    """

    chunk_id: str
    dense_score: float = -1.0
    dense_rank: int = 0
    bm25_score: float = 0.0
    bm25_rank: int = 0
    fusion_score: float = 0.0
    rerank_score: float = -1.0
    rerank_rank: int = 0
    text: str = ""
    source: str = ""
    title: str = ""
    category: str = ""
    document_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "dense_score": self.dense_score,
            "dense_rank": self.dense_rank,
            "bm25_score": self.bm25_score,
            "bm25_rank": self.bm25_rank,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "category": self.category,
            "document_id": self.document_id,
            "metadata": self.metadata,
        }


def rrf_fuse(
    dense_hits: list[RankedHit],
    bm25_hits: list[RankedHit],
    *,
    k: int = RRF_K_DEFAULT,
) -> list[HybridHit]:
    """Fuse two ranked lists with RRF and return the merged, ranked list.

    A chunk present in only one channel still participates: its missing
    channel contributes ``0`` to the fusion sum.  Ties in the fused score
    are broken deterministically (higher rank in a channel, then chunk_id)
    so results are reproducible.
    """
    if k < 1:
        raise ValueError(f"RRF k must be >= 1, got {k}")

    by_id: dict[str, HybridHit] = {}

    def _upsert(hit: RankedHit, channel: str) -> None:
        entry = by_id.get(hit.chunk_id)
        if entry is None:
            entry = HybridHit(chunk_id=hit.chunk_id, text=hit.text, source=hit.source, title=hit.title,
                              category=hit.category, document_id=hit.document_id, metadata=dict(hit.metadata))
            by_id[hit.chunk_id] = entry
        # prefer the first-seen context (dense carries the canonical text)
        if not entry.text and hit.text:
            entry.text, entry.source = hit.text, hit.source
        if channel == "dense":
            entry.dense_score = hit.score
            entry.dense_rank = hit.rank
        else:
            entry.bm25_score = hit.score
            entry.bm25_rank = hit.rank

    for hit in dense_hits:
        _upsert(hit, "dense")
    for hit in bm25_hits:
        _upsert(hit, "bm25")

    for entry in by_id.values():
        score = 0.0
        if entry.dense_rank > 0:
            score += 1.0 / (k + entry.dense_rank)
        if entry.bm25_rank > 0:
            score += 1.0 / (k + entry.bm25_rank)
        entry.fusion_score = score

    fused = sorted(
        by_id.values(),
        key=lambda h: (-h.fusion_score, h.dense_rank or 1 << 30, h.bm25_rank or 1 << 30, h.chunk_id),
    )
    return fused


__all__ = ["HybridHit", "RankedHit", "RRF_K_DEFAULT", "rrf_fuse"]
