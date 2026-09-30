"""BGE-Reranker-v2-M3 — real cross-encoder relevance scoring (Phase 2.3).

Sits **after** hybrid recall, never before it::

    Query
    -> Dense + BM25
    -> RRF
    -> Candidate Top-N            (candidate_k, e.g. 20)
    -> BGE-Reranker-v2-M3         (this module)
    -> Final Top-K                (final_k, e.g. 5)

The reranker takes a query plus a list of *candidate* chunks that a
retriever has already recalled, scores each (query, chunk) pair with a
cross-encoder, and returns the candidates re-ordered by relevance.  It
does NOT recall on its own and does NOT scan the whole knowledge base.

Design constraints (Phase 2.3):

- The concrete model runtime (sentence-transformers ``CrossEncoder``) is
  hidden behind this module.  Business code never imports the model class
  directly; swapping the reranker model only touches this file + config.
- ``Formal`` backend loads the **real** BGE-Reranker-v2-M3.  When the model
  cannot be loaded, the factory raises :class:`RerankerUnavailableError` —
  there is **no silent fallback** to a fake / hash / random scorer.
- A ``FakeReranker`` (deterministic, test-only) is available for unit
  tests and is always explicitly labelled ``backend = "fake"``; it is never
  substituted when the formal model fails.

The cross-encoder returns one relevance logit per (query, document) pair.
For the BGE-Reranker-v2-M3 (``num_labels=1``) the value is passed through a
sigmoid, so scores are in ``[0, 1]`` and higher means more relevant.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.core.exceptions import RerankerUnavailableError

logger = logging.getLogger("eae.reranker")

BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class Scorer(Protocol):
    """Low-level pair scoring callable.

    A scorer takes one query string and a list of document strings and
    returns a list of one relevance score per document, in input order.
    """

    def __call__(self, query: str, documents: list[str]) -> list[float]: ...


# ---------------------------------------------------------------------------
# Rerank result
# ---------------------------------------------------------------------------


@dataclass
class RerankResult:
    """One re-ranked candidate, with the original retrieval context preserved.

    ``rerank_score`` / ``rerank_rank`` are the new cross-encoder fields; the
    original hybrid fields (``dense_*`` / ``bm25_*`` / ``fusion_score``) and
    context (``text`` / ``source`` / ``title`` / ``metadata``) are carried
    through untouched so downstream answer generation keeps its citations.
    """

    chunk_id: str
    rerank_score: float
    rerank_rank: int  # 1-based position in the final (post-rerank) list
    dense_score: float = -1.0
    dense_rank: int = 0
    bm25_score: float = 0.0
    bm25_rank: int = 0
    fusion_score: float = 0.0
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


# ---------------------------------------------------------------------------
# Real cross-encoder scorer (formal backend)
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    # stable numeric sigmoid used to map raw logits into [0, 1]
    if x >= 0:
        z = 1.0 + __import__("math").exp(-x)
        return 1.0 / z
    z = __import__("math").exp(x)
    return z / (1.0 + z)


class CrossEncoderScorer:
    """Wraps a loaded BGE-Reranker-v2-M3 ``CrossEncoder`` with batch scoring.

    ``predict()`` batches the (query, doc) pairs in one model call — never a
    per-document loop — and records the per-call latency in milliseconds so
    a caller can surface ``rerank_latency_ms`` without a monitoring stack.
    """

    backend = "bge-reranker-v2-m3"

    def __init__(self, model: Any, *, batch_size: int = 8) -> None:
        self._model = model
        self._batch_size = max(1, int(batch_size))
        self.rerank_latency_ms: float = 0.0

    def __call__(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        start = time.perf_counter()
        raw = self._model.predict(pairs, batch_size=self._batch_size)
        # predict() returns a 1-D numpy array (or list) of per-pair scores.
        self.rerank_latency_ms = (time.perf_counter() - start) * 1000.0
        return [float(v) for v in raw]


# ---------------------------------------------------------------------------
# Fake scorer (TEST MODE ONLY — always explicitly labelled)
# ---------------------------------------------------------------------------


class FakeScorer:
    """Deterministic lexical-overlap scorer for offline unit tests.

    Scores a (query, doc) pair by the fraction of query tokens present in the
    document.  It is NOT a semantic reranker and must only be used where a
    real model is unavailable in test contexts; it is clearly labelled
    ``backend = "fake"`` and is never a fallback for a real model failure.
    """

    backend = "fake"

    def __init__(self, k: int = 1) -> None:
        self.k = k
        self.rerank_latency_ms: float = 0.0

    def __call__(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        q_tokens = _tokenise(query)
        out: list[float] = []
        for doc in documents:
            if not q_tokens:
                out.append(0.0)
                continue
            d_tokens = _tokenise(doc)
            hits = len(q_tokens & d_tokens)
            out.append(round(hits / len(q_tokens), 6))
        return out


def _tokenise(text: str) -> set[str]:
    """Lower-cased whitespace/punctuation tokens — simple + deterministic."""
    import re

    return {t for t in re.split(r"[^\w一-鿿]+", text.lower()) if t}


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """Re-rank a list of candidate chunks for a query with a cross-encoder.

    Pass a ``scorer`` (``CrossEncoderScorer`` for formal, ``FakeScorer`` for
    tests) to keep this fully decoupled from the model runtime.  When no
    scorer is given, :meth:`load_default` is used to build a real one from
    ``RERANKER_MODEL`` / config — which RAISES on failure rather than falling
    back.
    """

    def __init__(self, scorer: Scorer | None = None, *, batch_size: int = 8) -> None:
        self._scorer = scorer
        self._batch_size = batch_size
        self._loaded = scorer is not None
        # model provenance, filled when a real scorer is built
        self.model_source: str = ""
        self.device: str = ""
        self.backend: str = getattr(scorer, "backend", "unknown")

    # -- model loading ------------------------------------------------------

    @classmethod
    def load_default(cls, *, model_source: str | None = None,
                     device: str | None = None, batch_size: int = 8,
                     max_length: int = 512) -> Reranker:
        """Build a Reranker around a **real** BGE-Reranker-v2-M3.

        Raises :class:`RerankerUnavailableError` if the model or runtime is
        unavailable — formal mode never silently downgrades to a fake.
        """
        source = model_source or os.environ.get("RERANKER_MODEL", BGE_RERANKER_MODEL)
        device = device or os.environ.get("RERANKER_DEVICE", "cpu")
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(source, device=device, max_length=max_length)
        except Exception as exc:  # noqa: BLE001 - surface as unified reranker error
            raise RerankerUnavailableError(
                f"BGE-Reranker-v2-M3 model '{source}' cannot be loaded "
                f"({type(exc).__name__}: {exc}). Check RERANKER_MODEL points to a "
                "valid local model directory or a reachable HF repo id, and that the "
                "runtime (torch / sentence-transformers / transformers) is healthy. "
                "Formal mode will NOT fall back to a fake scorer.",
                details={"model_source": source, "device": device},
            ) from exc
        scorer = CrossEncoderScorer(model, batch_size=batch_size)
        reranker = cls(scorer, batch_size=batch_size)
        reranker.model_source = source
        reranker.device = device
        reranker.backend = scorer.backend
        logger.info("reranker loaded: source=%s device=%s backend=%s", source, device, scorer.backend)
        return reranker

    # -- scoring ------------------------------------------------------------

    @property
    def scorer(self) -> Scorer:
        if not self._loaded:
            raise RerankerUnavailableError(
                "No reranker model is loaded. Call Reranker.load_default() "
                "or pass an explicit scorer; formal mode does not auto-fallback.",
            )
        assert self._scorer is not None
        return self._scorer

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document (input order preserved)."""
        if not isinstance(query, str):
            raise RerankerUnavailableError(
                f"query must be a str, got {type(query).__name__}",
            )
        docs = _normalise_documents(documents)
        if not docs:
            return []
        return self.scorer(query, docs)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[RerankResult]:
        """Re-rank candidate dicts and return the top ``top_k`` as RerankResult.

        Each candidate is a dict carrying at least ``chunk_id`` and ``text``;
        any hybrid fields (``dense_*`` / ``bm25_*`` / ``fusion_score`` /
        ``source`` / ``title`` / ``metadata``) are carried through verbatim.
        ``top_k <= 0`` is treated as "keep all candidates".
        """
        docs = [str(c.get("text", "")) for c in candidates]
        scores = self.score(query, docs) if docs else []

        # Deduplicate by chunk_id keeping the highest score; keep first-seen
        # context. This matches the "last wins" rebuild semantics elsewhere.
        by_id: dict[str, dict[str, Any]] = {}
        for cand, score in zip(candidates, scores, strict=True):
            cid = str(cand.get("chunk_id", ""))
            entry = {**cand, "rerank_score": score}
            if cid not in by_id or score > by_id[cid].get("rerank_score", -1.0):
                by_id[cid] = entry
        ordered = sorted(
            by_id.values(),
            key=lambda e: (-e["rerank_score"], str(e.get("chunk_id", ""))),
        )
        results: list[RerankResult] = []
        limit = len(ordered) if top_k is None or top_k <= 0 else int(top_k)
        for rank, entry in enumerate(ordered[:limit], start=1):
            results.append(
                RerankResult(
                    chunk_id=str(entry.get("chunk_id", "")),
                    rerank_score=float(entry["rerank_score"]),
                    rerank_rank=rank,
                    dense_score=float(entry.get("dense_score", -1.0)),
                    dense_rank=int(entry.get("dense_rank", 0)),
                    bm25_score=float(entry.get("bm25_score", 0.0)),
                    bm25_rank=int(entry.get("bm25_rank", 0)),
                    fusion_score=float(entry.get("fusion_score", 0.0)),
                    text=str(entry.get("text", "")),
                    source=str(entry.get("source", "")),
                    title=str(entry.get("title", "")),
                    category=str(entry.get("category", "")),
                    document_id=str(entry.get("document_id", "")),
                    metadata=dict(entry.get("metadata", {}) or {}),
                )
            )
        return results

    # -- introspection ------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_source": self.model_source,
            "device": self.device,
            "loaded": self._loaded,
        }


def _normalise_documents(documents: list[str]) -> list[str]:
    """Validate / coerce a document list; empty text becomes ""."""
    if documents is None:
        raise RerankerUnavailableError("documents must be a list, got None")
    if isinstance(documents, str):
        raise RerankerUnavailableError(
            f"documents must be a list of strings, got a bare str "
            f"({len(documents)} chars). Pass a list even for one document.",
        )
    return [str(d) for d in documents]


__all__ = [
    "BGE_RERANKER_MODEL",
    "CrossEncoderScorer",
    "FakeScorer",
    "RerankResult",
    "Reranker",
    "Scorer",
]
