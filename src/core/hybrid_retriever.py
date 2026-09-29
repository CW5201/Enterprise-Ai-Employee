"""Hybrid retriever — unified Dense + BM25 + RRF interface (Phase 2.2).

This is the ONLY interface upper layers (RAG tool / RAG node / answer
generation) should call.  It wraps:

- dense:  a real vector store (:class:`MilvusVectorStore` formal,
  :class:`FakeVectorStore` test-only) over BGE-M3 embeddings;
- bm25:   an in-process :class:`BM25Index` over the same chunk texts;
- fusion: :func:`rrf_fuse` (Reciprocal Rank Fusion, rank-based — never
  adds raw dense + BM25 scores).

All three retrieval modes share one result type (:class:`HybridHit`) so
callers are agnostic to which channel produced a hit.  The formal dense
backend is still the real Milvus server; BM25 is an in-memory complement
to it (keyword recall), not a substitute.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.bm25_store import BM25Hit, BM25Index
from src.core.exceptions import RetrievalError
from src.core.retrieval_types import RRF_K_DEFAULT, HybridHit, RankedHit, rrf_fuse
from src.core.vector_store import SearchHit, VectorStore, create_vector_store

logger = logging.getLogger("eae.hybrid")

_VALID_MODES = ("dense", "bm25", "hybrid")


def _configured_rrf_k() -> int:
    try:
        from src.core.config_loader import get_settings

        return int(get_settings().rag.rrf_k)
    except Exception:  # noqa: BLE001 - config is optional at import time
        return RRF_K_DEFAULT


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
    ) -> None:
        self._vector_store = vector_store or create_vector_store(backend=backend)
        self._rrf_k = rrf_k if rrf_k is not None else _configured_rrf_k()
        self._bm25 = bm25 or BM25Index()
        if bm25 is None:
            # A fresh retriever (no explicit bm25 index) builds its keyword
            # index from the current knowledge-base corpus so both channels
            # share the same chunk set.
            self._build_bm25(kb_root=kb_root)

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

    # -- mode dispatcher ------------------------------------------------------

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[HybridHit]:
        """Dispatch on ``retrieval_mode``: ``dense`` | ``bm25`` | ``hybrid``."""
        if mode not in _VALID_MODES:
            raise RetrievalError(f"Unknown retrieval mode {mode!r} (expected one of {_VALID_MODES})")
        if mode == "dense":
            return self.search_dense(query, top_k=top_k)
        if mode == "bm25":
            return self.search_bm25(query, top_k=top_k)
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
    def backend(self) -> str:
        return self._vector_store.backend

    def health_check(self) -> dict[str, Any]:
        return {
            "backend": self._vector_store.backend,
            "bm25_doc_count": self._bm25.size(),
            "dense_size": self._vector_store.size(),
        }


__all__ = ["HybridRetriever"]
