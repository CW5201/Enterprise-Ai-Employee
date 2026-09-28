"""Tool: RAG - enterprise knowledge retrieval (Phase 1 minimal version).

Pipeline (subset of the full hybrid design in docs/ARCHITECTURE.md #11):

    query
      -> dense retrieval (in-process vector index, Phase 1)
      -> BM25 sparse retrieval (rank-bm25, optional when a corpus exists)
      -> RRF fusion
      -> top-k

The Milvus **server** and the BGE-M3 embedding model come in Phase 2
(ADR-003); Phase 1 uses the in-process index from
:mod:`src.core.vector_store` plus a filesystem-backed chunk corpus under
``data/knowledge_base/`` so the RAG path is testable end-to-end without
any external service.

Corpus layout (per data/knowledge_base/*/README.md):

    data/knowledge_base/<category>/<doc_id>.md
    # --- frontmatter ---
    # doc_id: finance-0001
    # title: 报销制度
    # department: finance
    ...
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.core.observability import observe
from src.core.vector_store import InMemoryMilvusIndex, Record, create_vector_store
from src.tools.base import ToolResult

_KB_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base"

_FRONTMATTER_RE = re.compile(r"^#\s*---\s+frontmatter\s+---\n(.*?)(?:\n# --- frontmatter end ---|\Z)", re.DOTALL)


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Simple character-window chunking (good enough for Phase 1 demo corpus)."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(text), step):
        chunks.append(text[i : i + chunk_size])
        if i + chunk_size >= len(text):
            break
    return chunks


def _tokenize(text: str) -> list[str]:
    """Whitespace + CJK bigram tokens for BM25 (no tokenizer dependency)."""
    tokens: list[str] = []
    for part in re.findall(r"[\w]+|[一-鿿]", text.lower()):
        if re.fullmatch(r"[一-鿿]", part):
            continue
        tokens.append(part)
    for part in re.findall(r"[一-鿿]{2}", text.lower()):
        tokens.extend(part)
    return tokens or ([text.lower()] if text else [])


@dataclass
class _Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeCorpus:
    """Loads markdown documents from the knowledge_base directory."""

    def __init__(self, root: Path = _KB_ROOT) -> None:
        self.root = root

    def load(self, *, chunk_size: int = 512, overlap: int = 64) -> list[_Chunk]:
        if not self.root.exists():
            return []
        chunks: list[_Chunk] = []
        for md_file in sorted(self.root.rglob("*.md")):
            if md_file.name == "README.md" or md_file.name.startswith("_"):
                continue
            content = md_file.read_text(encoding="utf-8")
            metadata = self._parse_frontmatter(content)
            metadata.setdefault("doc_id", md_file.stem)
            metadata.setdefault("department", md_file.parent.name)
            body = _FRONTMATTER_RE.sub("", content).strip()
            if not body:
                # doc has frontmatter only - no body text to index
                continue
            for index, chunk in enumerate(_chunk_text(body, chunk_size, overlap)):
                doc_hash = hashlib.md5(f"{metadata.get('doc_id')}:{index}".encode()).hexdigest()[:8]
                chunks.append(_Chunk(chunk_id=f"chunk-{doc_hash}", doc_id=str(metadata.get("doc_id")),
                                     text=chunk, metadata=metadata))
        return chunks

    @staticmethod
    def _parse_frontmatter(content: str) -> dict[str, Any]:
        match = re.match(r"^#\s*---\s+frontmatter\s+---\n(.*?)(?:\n# --- frontmatter end ---)", content, re.DOTALL)
        if not match:
            return {}
        data: dict[str, Any] = {}
        for line in match.group(1).splitlines():
            line = line.lstrip("#").strip()
            if ":" in line:
                key, _, value = line.partition(":")
                data[key.strip()] = value.strip()
        return data


class RAGTool:
    """Hybrid (dense + BM25 + RRF) retrieval over the local knowledge base."""

    name: str = "rag"

    def __init__(
        self,
        corpus: KnowledgeCorpus | None = None,
        index: InMemoryMilvusIndex | None = None,
        *,
        top_k: int = 5,
    ) -> None:
        self.corpus = corpus or KnowledgeCorpus()
        self._index = index or create_vector_store()
        self._top_k = top_k
        self._loaded = False

    # -- indexing ----------------------------------------------------------

    def build_index(self, *, chunk_size: int = 512, overlap: int = 64) -> int:
        chunks = self.corpus.load(chunk_size=chunk_size, overlap=overlap)
        records = [
            Record(id=c.chunk_id, text=c.text, metadata={**c.metadata, "source_type": "milvus"})
            for c in chunks
        ]
        self._index.insert(records)
        self._chunks = {c.chunk_id: c for c in chunks}
        corpus_docs = [
            " ".join(_tokenize(c.text)) for c in chunks
        ]
        self._bm25 = BM25Okapi(corpus_docs) if corpus_docs else None
        self._chunk_order = [c.chunk_id for c in chunks]
        self._loaded = True
        return len(records)

    @property
    def size(self) -> int:
        return self._index.size()

    # -- retrieval ---------------------------------------------------------

    @observe("rag_retrieval")
    def run(self, *, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> ToolResult:
        k = top_k or self._top_k
        if not self._loaded:
            self.build_index()
        if self._index.size() == 0:
            return ToolResult(
                tool_name=self.name,
                ok=False,
                error="knowledge base is empty; place markdown docs under data/knowledge_base/<category>/",
            )
        dense = self._dense(query, k * 2, filters)
        sparse = self._sparse(query, k * 2)
        fused = self._fuse_rrf(dense, sparse)
        hits = [(chunk, score) for chunk, score in fused[:k]]
        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text[:600],
                "metadata": chunk.metadata,
                "score": round(score, 4),
            }
            for chunk, score in hits
        ]
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data=payload,
            evidence_ref=f"milvus:{','.join(chunk.chunk_id for chunk, _ in hits)}" if hits else "milvus:",
            metadata={"top_k": k, "n_dense": len(dense), "n_sparse": len(sparse),
                      "index_size": self._index.size()},
        )

    # -- retrieval legs ------------------------------------------------------

    def _dense(self, query: str, k: int, filters: dict[str, Any] | None) -> list[_Chunk]:
        hits = self._index.search(query, top_k=k, filter=filters)
        return [self._chunks[h.id] for h in hits if h.id in self._chunks]

    def _sparse(self, query: str, k: int) -> list[_Chunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._chunks[self._chunk_order[i]] for i in ranked if scores[i] > 0]

    @staticmethod
    def _fuse_rrf(dense: list[_Chunk], sparse: list[_Chunk], k: int = 60) -> list[tuple[_Chunk, float]]:
        """Reciprocal-rank fusion (Phase 1: equal weight; RRF + rerank tuned in Phase 2)."""
        ranks: dict[str, float] = {}
        chunk_map: dict[str, _Chunk] = {}
        for leg, weight in ((dense, 1.0), (sparse, 1.0)):
            for rank, chunk in enumerate(leg):
                chunk_map[chunk.chunk_id] = chunk
                ranks[chunk.chunk_id] = ranks.get(chunk.chunk_id, 0.0) + weight / (k + rank + 1)
        ordered = sorted(ranks.items(), key=lambda kv: kv[1], reverse=True)
        return [(chunk_map[cid], score) for cid, score in ordered]


__all__ = ["KnowledgeCorpus", "RAGTool"]
