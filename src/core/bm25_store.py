"""BM25 keyword index — in-process sparse retrieval over KB chunks (Phase 2.2).

Complements the dense vector store (:mod:`src.core.vector_store`):
BM25 recalls chunks by exact / near-exact keyword overlap, dense vectors
recall by semantic similarity.  Both operate over the **same chunk set**
(``chunk_id`` must stay aligned with the Milvus collection so results
from the two channels can be fused).

Design:
- :class:`BM25Index` — build / search / delete over chunk texts.
- Tokenisation: Chinese character bigrams + CJK single chars + ASCII
  word tokens.  Deliberately simple and deterministic — no external
  tokenizer dependency (jieba not in the locked stack).
- Standard BM25 (Robertson / Okapi) with ``k1=1.5``, ``b=0.75``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """Deterministic tokeniser for mixed Chinese / English KB text.

    - ASCII letters/digits -> lowercased word tokens
    - CJK -> single characters AND character bigrams (captures compound
      words like 报销/审批 without a dictionary)
    """
    text = text.lower()
    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isascii():
            if ch.isalnum():
                j = i
                while j < len(text) and text[j].isascii() and text[j].isalnum():
                    j += 1
                tokens.append(text[i:j])
                i = j
                continue
            i += 1
            continue
        if "一" <= ch <= "鿿":
            # single char + bigram with next CJK char
            tokens.append(ch)
            if i + 1 < len(text) and "一" <= text[i + 1] <= "鿿":
                tokens.append(text[i : i + 2])
            i += 1
            continue
        i += 1
    return tokens


@dataclass
class BM25Hit:
    """One BM25 result.  ``rank`` is 1-based position in the returned list."""

    chunk_id: str
    score: float
    rank: int
    metadata: dict = field(default_factory=dict)
    text: str = ""
    source: str = ""
    title: str = ""
    category: str = ""

    def to_payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "rank": self.rank,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "category": self.category,
            "metadata": self.metadata,
        }


class BM25Index:
    """In-memory BM25 index over chunk records.

    Records have the same shape as :func:`iter_kb_chunks_for_store` output
    (``chunk_id / text / source / title / category / metadata``).  Duplicate
    ``chunk_id`` entries: the **last** one wins, so rebuilding from an
    updated corpus never returns stale rows.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, dict] = {}
        self._doc_tokens: dict[str, list[str]] = {}
        self._doc_freq: Counter = Counter()
        self._avg_dl: float = 0.0
        self._total_docs: int = 0

    # -- build ---------------------------------------------------------------

    def build(self, records: list[dict]) -> int:
        """Build the index from a list of chunk records (replaces current state)."""
        self._docs = {}
        self._doc_tokens = {}
        for record in records:
            chunk_id = str(record.get("chunk_id", ""))
            if not chunk_id:
                continue
            self._docs[chunk_id] = record
            self._doc_tokens[chunk_id] = tokenize(str(record.get("text", "")))
        self._rebuild_stats()
        return self.size()

    def add(self, records: list[dict]) -> int:
        """Append/update records (upsert by chunk_id) and refresh statistics."""
        for record in records:
            chunk_id = str(record.get("chunk_id", ""))
            if not chunk_id:
                continue
            self._docs[chunk_id] = record
            self._doc_tokens[chunk_id] = tokenize(str(record.get("text", "")))
        self._rebuild_stats()
        return len(records)

    def _rebuild_stats(self) -> None:
        self._doc_freq = Counter()
        total_len = 0
        for _chunk_id, tokens in self._doc_tokens.items():
            total_len += len(tokens)
            for term in set(tokens):
                self._doc_freq[term] += 1
        self._total_docs = len(self._docs)
        self._avg_dl = (total_len / self._total_docs) if self._total_docs else 0.0

    # -- query ---------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[BM25Hit]:
        """Rank all chunks by BM25 score for ``query``; return top ``top_k``.

        1-based ``rank`` in the returned list; only documents with at least
        one query-term hit are returned.  Empty query / empty index -> [].
        """
        if not self._docs:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scored: list[tuple[str, float]] = []
        for chunk_id, doc_tokens in self._doc_tokens.items():
            tf = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            score = 0.0
            for q in set(q_tokens):
                if q not in tf:
                    continue
                f = tf[q]
                df = self._doc_freq.get(q, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (self._total_docs - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (1.0 - self.b + self.b * doc_len / max(self._avg_dl, 1e-9))
                score += idf * f * (self.k1 + 1) / denom
            if score > 0.0:
                scored.append((chunk_id, score))

        scored.sort(key=lambda x: (-x[1], x[0]))
        hits: list[BM25Hit] = []
        for rank, (chunk_id, score) in enumerate(scored[: max(0, int(top_k))], start=1):
            record = self._docs[chunk_id]
            hits.append(
                BM25Hit(
                    chunk_id=chunk_id,
                    score=score,
                    rank=rank,
                    metadata=record.get("metadata", {}),
                    text=str(record.get("text", "")),
                    source=str(record.get("source", "")),
                    title=str(record.get("title", "")),
                    category=str(record.get("category", "")),
                )
            )
        return hits

    # -- maintenance ----------------------------------------------------------

    def delete(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._docs:
                del self._docs[chunk_id]
                self._doc_tokens.pop(chunk_id, None)
                removed += 1
        if removed:
            self._rebuild_stats()
        return removed

    def size(self) -> int:
        return len(self._docs)

    def health_check(self) -> dict:
        return {"backend": "bm25", "doc_count": self.size(), "k1": self.k1, "b": self.b}


__all__ = ["BM25Hit", "BM25Index", "tokenize"]
