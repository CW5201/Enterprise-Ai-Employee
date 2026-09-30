"""Unified BGE-M3 embedding interface — formal real-model embeddings.

Single entry point for text -> vector.  Business code must never import a
model runtime directly; it goes through :func:`create_embedder`.

Phase 2.1 formal behaviour:

- ``bge-m3`` backend: loads a **real** BGE-M3 model (sentence-transformers)
  from ``EMBEDDING_MODEL`` — an HF repo id or a local model directory
  (e.g. a local model directory).  The output dimension is measured
  from the model itself (never hardcoded), and vectors are L2-normalised.
- When the model cannot be loaded (missing weights, broken runtime) the
  factory raises :class:`EmbeddingUnavailableError` with a clear message.
  **Formal mode never falls back to hashing** — a hash vector is not a
  semantic embedding and must not masquerade as BGE-M3.
- ``force_offline=True`` (used by unit tests only) yields the deterministic
  hashing embedder of the same dimension, clearly labelled
  ``backend = "fake"``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import Protocol

from src.core.exceptions import EmbeddingUnavailableError

logger = logging.getLogger("eae.embedder")

BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_DIM = 1024


class EmbedFn(Protocol):
    """Callable returning a unit-length vector for a text."""

    def __call__(self, text: str) -> list[float]: ...


def _hash_embed(text: str, dim: int = BGE_M3_DIM) -> list[float]:
    """Deterministic character n-gram hashing embedding.

    TEST-MODE ONLY.  Not a semantic embedding — it keeps unit tests and
    offline pipelines fast and reproducible.  It is NEVER used as the
    formal BGE-M3 backend.
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


class HashEmbedder:
    """Test-mode deterministic embedder (same dimension, non-semantic)."""

    backend = "fake"

    def __init__(self, dim: int = BGE_M3_DIM) -> None:
        self.dim = dim
        self.model_path = "hash-fallback (test mode)"
        self.status = "hash-fallback"

    def __call__(self, text: str) -> list[float]:
        return _hash_embed(text, self.dim)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_hash_embed(t, self.dim) for t in texts]


class Embedder:
    """Real BGE-M3 embedder; raises EmbeddingUnavailableError when the model
    cannot be loaded (formal mode never substitutes a hash vector)."""

    backend = "bge-m3"

    def __init__(self, model_source: str | None = None, dim: int = BGE_M3_DIM) -> None:
        source = model_source or os.environ.get("EMBEDDING_MODEL", BGE_M3_MODEL)
        device = os.environ.get("EMBEDDING_DEVICE", "cpu")
        # CPU-only builds of torch do not honour an env var named cuda; fail fast
        # with a clear error instead of a cryptic device-mismatch crash.
        try:
            import torch

            if device.lower().startswith("cuda") and not torch.cuda.is_available():
                logger.warning("EMBEDDING_DEVICE=cuda requested but CUDA is unavailable; using cpu")
                device = "cpu"
        except Exception:  # noqa: BLE001 - torch optional in odd environments
            pass
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(source, device=device)
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"BGE-M3 model '{source}' cannot be loaded ({type(exc).__name__}: {exc}). "
                "Check EMBEDDING_MODEL points to a valid local model directory "
                "or a reachable HF repo id, and that the runtime (torch / "
                "sentence-transformers / transformers / tokenizers versions) is healthy.",
                details={"model_source": source, "device": device},
            ) from exc
        # Support a local directory so the model path is always attributable.
        self.model_path: str = source
        self.dim: int = dim
        self.status: str = "bge-m3"

    def _encode(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(v) for v in vector]

    def __call__(self, text: str) -> list[float]:
        vec = self._encode(text)
        # Keep the declared dimension honest: if the model outputs a
        # different size than configured, surface it via the instance attr
        # so callers can align the Milvus schema with the actual output.
        self.dim = len(vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        out = [[float(v) for v in vec] for vec in vectors]
        if out:
            self.dim = len(out[0])
        return out


def create_embedder(force_offline: bool = False) -> Embedder | HashEmbedder:
    """Factory used by the vector store / RAG tool / build scripts.

    - ``force_offline=True`` (or ``EMBEDDING_FORCE_OFFLINE=1``): the
      deterministic hashing embedder — unit tests / test mode only,
      labelled ``backend = "fake"``.
    - otherwise: a real BGE-M3 embedder; raises
      :class:`EmbeddingUnavailableError` when the model cannot load.
      Formal runs NEVER silently fall back to hashing.
    """
    if force_offline or os.environ.get("EMBEDDING_FORCE_OFFLINE") == "1":
        return HashEmbedder()
    return Embedder()


__all__ = [
    "BGE_M3_DIM",
    "BGE_M3_MODEL",
    "Embedder",
    "EmbedFn",
    "HashEmbedder",
    "create_embedder",
]
