"""Vector store - unified Milvus interface (Phase 1: in-process minimal index).

Phase 1 implements the retrieval path end to end WITHOUT a running Milvus
server:

- :class:`InMemoryMilvusIndex` - a pure-Python cosine index that mimics the
  small subset of the Milvus API used by the rag tool (``insert`` /
  ``search`` with top-k and metadata filter).  It is used in tests and in
  local single-machine runs.
- :func:`create_vector_store` - returns an in-memory index by default;
  when a live Milvus server is configured (``MILVUS_HOST`` reachable),
  Phase 2 will switch this to a ``pymilvus``-backed index.

The embedding function is decoupled: callers inject ``embed_fn`` so the
store itself does not own the model.  In Phase 1 the default embed_fn is
a deterministic character n-gram hash embedding (no model download),
which is sufficient to exercise the hybrid retrieval + rerank pipeline;
the BGE-M3 based embedder (``src/core/embedder.py``) will replace it in
Phase 2.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Protocol


class EmbedFn(Protocol):
    """Callable returning a unit-length vector for a text."""

    def __call__(self, text: str) -> list[float]: ...


@dataclass
class Record:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    id: str
    score: float
    record: Record
    vector: list[float] | None = None


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    """Deterministic character n-gram hashing embedding (Phase 1 only).

    Not a semantic embedding - it is a placeholder that keeps the whole
    pipeline testable offline.  Replaced by BGE-M3 in Phase 2.
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


class InMemoryMilvusIndex:
    """Cosine similarity index with metadata filtering."""

    def __init__(self, embed_fn: EmbedFn = _hash_embed) -> None:
        self._embed_fn = embed_fn
        self._records: dict[str, Record] = {}
        self._vectors: dict[str, list[float]] = {}

    def insert(self, records: list[Record]) -> int:
        for record in records:
            if record.id in self._records:
                continue
            self._records[record.id] = record
            self._vectors[record.id] = self._embed_fn(record.text)
        return len(records)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,  # noqa: A002 (matches Milvus API)
    ) -> list[SearchHit]:
        if not self._records:
            return []
        qv = self._embed_fn(query)
        hits: list[SearchHit] = []
        for record_id, record in self._records.items():
            if filter:
                meta = record.metadata
                if any(meta.get(k) != v for k, v in filter.items()):
                    continue
            rv = self._vectors[record_id]
            score = sum(a * b for a, b in zip(qv, rv, strict=True))
            hits.append(SearchHit(id=record_id, score=score, record=record, vector=rv))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def size(self) -> int:
        return len(self._records)


def create_vector_store(embed_fn: EmbedFn | None = None) -> InMemoryMilvusIndex:
    """Factory used by the RAG tool.  Phase 2 adds the pymilvus backend here."""
    return InMemoryMilvusIndex(embed_fn=embed_fn or _hash_embed)


__all__ = [
    "EmbedFn",
    "InMemoryMilvusIndex",
    "Record",
    "SearchHit",
    "_hash_embed",
    "create_vector_store",
]
