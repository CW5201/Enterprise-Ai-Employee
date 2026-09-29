"""Embedder — unified BGE-M3 embedding interface (Phase 1).

Single entry point for text -> vector.  Business code must never import a
model runtime directly.

Phase 1 strategy:
- Use ``BAAI/bge-m3`` (1024-dim) through ``sentence-transformers`` when the
  model can be loaded (network or cached weights available).
- When the model is unavailable (offline CI, no HF cache), fall back to a
  deterministic hashing embedder of the same dimension so the full pipeline
  stays runnable.  The fallback is clearly recorded in the embedder's
  ``backend`` attribute so results can be attributed honestly.

The vector DB (Milvus) is handled by :mod:`src.core.vector_store`; the
embedder only produces vectors.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import Protocol

logger = logging.getLogger("eae.embedder")

BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_DIM = 1024


class EmbedFn(Protocol):
    """Callable returning a unit-length vector for a text."""

    def __call__(self, text: str) -> list[float]: ...


def _hash_embed(text: str, dim: int = BGE_M3_DIM) -> list[float]:
    """Deterministic character n-gram hashing embedding (offline fallback).

    Not a semantic embedding — it keeps the whole pipeline testable offline.
    """
    vec = [0.0] * dim
    n = 3
    tokens = text.lower()
    for i in range(len(tokens) - n + 1):
        gram = tokens[i : i + n]
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Embedder:
    """One-shot embedder: BGE-M3 when available, hashing fallback otherwise."""

    def __init__(self, dim: int = BGE_M3_DIM, force_offline: bool = False) -> None:
        self.dim = dim
        self.backend: str = "hash-fallback"
        self._model = None
        # EMBEDDING_FORCE_OFFLINE=1 skips model loading entirely (fast CI path).
        if not force_offline and os.environ.get("EMBEDDING_FORCE_OFFLINE") != "1":
            try:
                from sentence_transformers import (
                    SentenceTransformer,  # type: ignore[import-not-found]
                )

                # EMBEDDING_MODEL may be a local directory (HF snapshot or
                # plain HF repo) so offline CI and local GPU boxes can use it
                # without hitting the network.
                model_source = os.environ.get("EMBEDDING_MODEL", BGE_M3_MODEL)
                self._model = SentenceTransformer(model_source)
                self.backend = "bge-m3"
            except Exception as exc:  # noqa: BLE001 - model load failure is not fatal
                logger.warning("BGE-M3 unavailable, using hash fallback: %s", exc)

    def __call__(self, text: str) -> list[float]:
        if self._model is not None:
            return list(self._model.encode(text, normalize_embeddings=True))
        return _hash_embed(text, self.dim)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            return [list(v) for v in self._model.encode(texts, normalize_embeddings=True)]
        return [_hash_embed(t, self.dim) for t in texts]


def create_embedder(force_offline: bool = False) -> Embedder:
    """Factory used by the vector store / RAG tool."""
    return Embedder(force_offline=force_offline)


__all__ = ["Embedder", "create_embedder", "_hash_embed", "BGE_M3_DIM", "BGE_M3_MODEL"]
