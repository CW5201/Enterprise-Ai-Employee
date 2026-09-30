"""Hybrid retriever — unified Dense + BM25 + RRF (+ Reranker) interface.

This is the ONLY interface upper layers (RAG tool / RAG node / answer
generation) should call.  It wraps:

- dense:  a real vector store (:class:`MilvusVectorStore` formal,
  :class:`FakeVectorStore` test-only) over BGE-M3 embeddings;
- bm25:   an in-process :class:`BM25Index` over the same chunk texts;
- fusion: :func:`rrf_fuse` (Reciprocal Rank Fusion, rank-based — never
  adds raw dense + BM25 scores).

Phase 2.3 adds an optional cross-encoder rerank stage:

    Query -> Dense + BM25 -> RRF -> Candidate Top-N -> BGE-Reranker-v2-M3 -> Final Top-K

The reranker ONLY re-orders the RRF candidate list; it never participates in
the original dense / BM25 recall.  It is off by default (``rerank=False``)
so the three legacy modes (``dense`` / ``bm25`` / ``hybrid``) are unchanged.

All retrieval modes share one result type (:class:`HybridHit`) so callers
are agnostic to which channel produced a hit.  The formal dense backend is
still the real Milvus server; BM25 is an in-memory complement to it
(keyword recall), not a substitute.  The reranker is a formal
:class:`Reranker` (BGE-Reranker-v2-M3) that raises
:class:`RerankerUnavailableError` when its model cannot load — no silent
fallback to fake scoring.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.bm25_store import BM25Hit, BM25Index
from src.core.exceptions import RerankerUnavailableError, RetrievalError
from src.core.reranker import Reranker
from src.core.retrieval_types import RRF_K_DEFAULT, HybridHit, RankedHit, rrf_fuse
from src.core.vector_store import SearchHit, VectorStore, create_vector_store

logger = logging.getLogger("eae.hybrid")

_VALID_MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def _configured_rrf_k() -> int:
    try:
        from src.core.config_loader import get_settings

        return int(get_settings().rag.rrf_k)
    except Exception:  # noqa: BLE001 - config is optional at import time
        return RRF_K_DEFAULT


def _configured_reranker() -> tuple[Reranker | None, int, int]:
    """Read reranker settings; return ``(reranker, candidate_k, final_k)``.

    The reranker is only built when config has it enabled AND the model is
    available.  When disabled, a ``None`` is returned (no rerank happens) and
    the caller simply skips the stage — this is *not* a silent fallback, it
    is the configured default.  A model that *fails to load while enabled*
    is surfaced by the caller via :class:`RerankerUnavailableError`.
    """
    try:
        from src.core.config_loader import get_settings

        raw = get_settings().raw
        rr = raw.get("reranker", {}) or {}
    except Exception:  # noqa: BLE001 - config is optional
        return (None, 20, 5)
    candidate_k = int(rr.get("candidate_k", 20))
    final_k = int(rr.get("final_k", 5))
    enabled = bool(rr.get("enabled", False))
    if not enabled:
        return (None, candidate_k, final_k)
    # Enabled: build the real model now so a misconfiguration fails here,
    # never silently.
    model_source = rr.get("model_path") or rr.get("model_name") or None
    device = rr.get("device", "cpu")
    batch_size = int(rr.get("batch_size", 8))
    reranker = Reranker.load_default(
        model_source=model_source, device=device, batch_size=batch_size,
    )
    return (reranker, candidate_k, final_k)


class HybridRetriever:
    """Dense + BM25 + RRF hybrid retrieval over the enterprise KB.

    The BM25 index is built from the same chunk records that back the
    dense store, so ``chunk_id`` is aligned across both channels.  A
    fresh formal instance builds its BM25 index from the knowledge-base
    corpus; pass pre-built stores explicitly for tests.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStore | None = None,
        bm25: BM25Index | None = None,
        backend: str | None = None,
        kb_root: Any | None = None,
        rrf_k: int | None = None,
        reranker: Reranker | None = None,
        candidate_k: int = 20,
        final_k: int = 5,
    ) -> None:
        self._vector_store = vector_store or create_vector_store(backend=backend)
        self._rrf_k = rrf_k if rrf_k is not None else _configured_rrf_k()
        self._bm25 = bm25 if bm25 is not None else BM25Index()
        # An explicit ``bm25`` index (even an empty one) is authoritative:
        # pass ``BM25Index()`` to opt out of the auto-built corpus index.
        # When no index is supplied, build it from the knowledge-base corpus
        # so both channels share the same chunk set.
        if bm25 is None:
            self._build_bm25(kb_root=kb_root)
        # Phase 2.3: an explicit reranker (even a fake one) is authoritative;
        # ``None`` means "defer to config on first hybrid_rerank call".
        self._reranker = reranker
        self._candidate_k = max(1, int(candidate_k))
        self._final_k = max(1, int(final_k))

    # -- reranker access -----------------------------------------------------

    @property
    def reranker(self) -> Reranker | None:
        """Return the reranker, building one from config on first access.

        ``None`` is returned only when config has the reranker *disabled* —
        that is the configured default, not a failure.  When config has it
        *enabled* but the model cannot load, this raises
        :class:`RerankerUnavailableError` (no silent fallback).
        """
        if self._reranker is None and not getattr(self, "_reranker_resolved", False):
            cfg_reranker, cfg_candidate_k, cfg_final_k = _configured_reranker()
            self._reranker = cfg_reranker
            if self._reranker is not None:
                self._candidate_k = cfg_candidate_k
                self._final_k = cfg_final_k
            self._reranker_resolved = True
        return self._reranker

    def _rerank_mode_guard(self) -> Reranker:
        """Require a usable reranker for ``hybrid_rerank``; raise if absent.

        A missing/disabled reranker in hybrid_rerank mode is a *misuse* of the
        mode, not a configurable default, so it raises rather than silently
        degrading to plain hybrid.
        """
        reranker = self.reranker
        if reranker is None:
            raise RerankerUnavailableError(
                "hybrid_rerank mode requested but no reranker is available. "
                "Enable `reranker.enabled` in config/settings.yaml (or pass a "
                "reranker= to the retriever). Formal mode does not silently "
                "fall back to a plain hybrid ranking.",
            )
        return reranker

    # -- index ---------------------------------------------------------------

    def _build_bm25(self, *, kb_root: Any | None = None) -> int:
        """Build the BM25 index from the knowledge-base chunk corpus."""
        from src.tools.rag_tool import iter_kb_chunks_for_store

        records = iter_kb_chunks_for_store(kb_root=kb_root)
        count = self._bm25.build(records)
        logger.info(
            "BM25 index built: %d chunks (dense store '%s' has %d)",
            count, self._vector_store.backend, self._vector_store.size(),
        )
        return count

    def build_index(self, *, kb_root: Any | None = None) -> int:
        """(Re)build the BM25 index; the dense store is managed by build_kb.py."""
        return self._build_bm25(kb_root=kb_root)

    # -- channel adapters -----------------------------------------------------

    def _to_ranked(self, hits: list[SearchHit]) -> list[RankedHit]:
        out: list[RankedHit] = []
        for i, h in enumerate(hits, start=1):
            out.append(
                RankedHit(
                    chunk_id=h.chunk_id,
                    score=h.score,
                    rank=i,
                    text=h.text,
                    source=h.source,
                    title=h.title,
                    category=h.category,
                    document_id=h.document_id,
                    metadata=h.metadata,
                )
            )
        return out

    @staticmethod
    def _bm25_to_ranked(hits: list[BM25Hit]) -> list[RankedHit]:
        return [
            RankedHit(
                chunk_id=h.chunk_id,
                score=h.score,
                rank=h.rank,
                text=h.text,
                source=h.source,
                title=h.title,
                category=h.category,
                metadata=h.metadata,
            )
            for h in hits
        ]

    # -- unified search interface --------------------------------------------

    def search_dense(self, query: str, top_k: int = 5) -> list[HybridHit]:
        """Dense-only channel (BGE-M3 + vector store)."""
        ranked = self._to_ranked(self._vector_store.search(query, top_k=max(1, top_k)))
        return [self._single_channel_to_hybrid(r, "dense") for r in ranked]

    def search_bm25(self, query: str, top_k: int = 5) -> list[HybridHit]:
        """BM25-only channel (keyword recall)."""
        ranked = self._bm25_to_ranked(self._bm25.search(query, top_k=max(1, top_k)))
        return [self._single_channel_to_hybrid(r, "bm25") for r in ranked]

    def hybrid_search(self, query: str, top_k: int = 5) -> list[HybridHit]:
        """Dense + BM25 fused with RRF.  Fetch a wider window per channel
        before truncating to ``top_k`` so both channels get a fair say."""
        top_k = max(1, int(top_k))
        window = max(top_k * 2, 10)
        dense = self._to_ranked(self._vector_store.search(query, top_k=window))
        bm25 = self._bm25_to_ranked(self._bm25.search(query, top_k=window))
        fused = rrf_fuse(dense, bm25, k=self._rrf_k)[:top_k]
        return fused

    def hybrid_rerank_search(
        self,
        query: str,
        *,
        candidate_k: int | None = None,
        final_k: int | None = None,
    ) -> list[HybridHit]:
        """Dense + BM25 -> RRF -> Candidate Top-N -> BGE-Reranker -> Final Top-K.

        Fetches ``candidate_k`` candidates per channel, fuses with RRF,
        then hands the fused candidate list to the cross-encoder reranker and
        returns the top ``final_k`` by rerank score.  The original RRF
        scores / ranks / context are preserved on every returned hit
        (``rerank_score`` / ``rerank_rank`` are added on top; ``fusion_score``
        is never overwritten).  When no reranker is configured or available
        this raises :class:`RerankerUnavailableError` — it never silently
        degrades to plain hybrid ranking.
        """
        candidate_k = max(1, int(candidate_k if candidate_k is not None else self._candidate_k))
        final_k = max(1, int(final_k if final_k is not None else self._final_k))
        reranker = self._rerank_mode_guard()
        dense = self._to_ranked(self._vector_store.search(query, top_k=candidate_k))
        bm25 = self._bm25_to_ranked(self._bm25.search(query, top_k=candidate_k))
        fused = rrf_fuse(dense, bm25, k=self._rrf_k)[:candidate_k]
        candidates = [h.to_payload() for h in fused]
        reranked = reranker.rerank(query, candidates, top_k=final_k)
        # Map RerankResult back to the unified HybridHit so callers stay
        # channel-agnostic (the rerank fields ride along).
        return [
            HybridHit(
                chunk_id=r.chunk_id,
                dense_score=r.dense_score,
                dense_rank=r.dense_rank,
                bm25_score=r.bm25_score,
                bm25_rank=r.bm25_rank,
                fusion_score=r.fusion_score,
                rerank_score=r.rerank_score,
                rerank_rank=r.rerank_rank,
                text=r.text,
                source=r.source,
                title=r.title,
                category=r.category,
                document_id=r.document_id,
                metadata=r.metadata,
            )
            for r in reranked
        ]

    # -- mode dispatcher ------------------------------------------------------

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[HybridHit]:
        """Dispatch on ``retrieval_mode``:
        ``dense`` | ``bm25`` | ``hybrid`` | ``hybrid_rerank``."""
        if mode not in _VALID_MODES:
            raise RetrievalError(f"Unknown retrieval mode {mode!r} (expected one of {_VALID_MODES})")
        if mode == "dense":
            return self.search_dense(query, top_k=top_k)
        if mode == "bm25":
            return self.search_bm25(query, top_k=top_k)
        if mode == "hybrid_rerank":
            return self.hybrid_rerank_search(query, final_k=top_k)
        return self.hybrid_search(query, top_k=top_k)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _single_channel_to_hybrid(r: RankedHit, channel: str) -> HybridHit:
        hit = HybridHit(
            chunk_id=r.chunk_id,
            text=r.text,
            source=r.source,
            title=r.title,
            category=r.category,
            document_id=r.document_id,
            metadata=dict(r.metadata),
        )
        if channel == "dense":
            hit.dense_score = r.score
            hit.dense_rank = r.rank
            hit.fusion_score = r.score
        else:
            hit.bm25_score = r.score
            hit.bm25_rank = r.rank
            hit.fusion_score = r.score
        return hit

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def bm25(self) -> BM25Index:
        return self._bm25

    @property
    def rerank_enabled(self) -> bool:
        """True when a reranker is (or will be) available for hybrid_rerank."""
        return self.reranker is not None

    @property
    def candidate_k(self) -> int:
        return self._candidate_k

    @property
    def final_k(self) -> int:
        return self._final_k

    @property
    def backend(self) -> str:
        return self._vector_store.backend

    def health_check(self) -> dict[str, Any]:
        return {
            "backend": self._vector_store.backend,
            "bm25_doc_count": self._bm25.size(),
            "dense_size": self._vector_store.size(),
            "rerank_enabled": self.rerank_enabled,
            "candidate_k": self._candidate_k,
            "final_k": self._final_k,
        }


__all__ = ["HybridRetriever"]
