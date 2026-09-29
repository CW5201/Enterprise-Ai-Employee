"""RAG tool — Dense + BM25 + RRF hybrid retrieval with source preserved.

Pipeline (Phase 2.2): query -> (BGE-M3 -> Milvus dense) + (BM25 keyword)
-> RRF fusion -> Top-K -> RetrievalResult, with ``retrieval_mode`` selecting
``dense`` / ``bm25`` / ``hybrid``.

Upper layers only call this tool / the :class:`HybridRetriever`; they never
touch Milvus or the BM25 index internals.  The formal dense backend is still
the real Milvus server (Phase 2.1); the fake backend remains test-mode only.

Every returned chunk keeps ``source`` / ``title`` / ``category`` /
``document_id`` / ``metadata`` so answers can cite their origin.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.hybrid_retriever import HybridRetriever
from src.core.observability import observe
from src.core.retrieval_types import HybridHit
from src.core.vector_store import VectorStore

_KB_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base"
_CHUNKS_CACHE: list[dict[str, Any]] | None = None


def _chunk_records(
    kb_root: Path | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict[str, Any]]:
    """Chunk the knowledge base with build_kb's chunking (single source of
    truth so the in-process chunk ids always match the ones in Milvus).

    The default chunking result is cached in-module; explicit sizes fall
    back to a fresh compute.  This keeps the helper import-safe even when
    invoked before the repo root is on ``sys.path`` (e.g. from a test).
    """
    global _CHUNKS_CACHE
    if kb_root is None and chunk_size == 512 and chunk_overlap == 64 and _CHUNKS_CACHE is not None:
        return _CHUNKS_CACHE
    try:
        import importlib

        kb_mod = importlib.import_module("scripts.build_kb")
        root = Path(kb_root) if kb_root else _KB_ROOT
        docs = kb_mod.iter_documents(root)
        records: list[dict[str, Any]] = []
        for doc in docs:
            for chunk in kb_mod.chunk_text(doc.body, chunk_size=chunk_size, overlap=chunk_overlap, doc_id=doc.doc_id):
                records.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "document_id": doc.doc_id,
                        "title": doc.title,
                        "source": doc.source,
                        "category": doc.category,
                        "text": chunk["text"],
                        "metadata": doc.metadata,
                    }
                )
        if kb_root is None and chunk_size == 512 and chunk_overlap == 64:
            _CHUNKS_CACHE = records
        return records
    except ModuleNotFoundError:
        # Fall back to the in-module chunking (legacy behaviour) when the
        # scripts package is not importable.  Chunk ids may diverge from
        # Milvus in this mode — acceptable only for offline test helpers.
        root = Path(kb_root) if kb_root else _KB_ROOT
        records: list[dict[str, Any]] = []
        for doc in _iter_documents(root):
            for chunk in _chunk_text(doc.body, chunk_size, chunk_overlap):
                chunk_id = _chunk_id(doc.doc_id, chunk, doc.metadata)
                records.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": doc.doc_id,
                        "title": doc.title,
                        "source": doc.source,
                        "category": doc.category,
                        "text": chunk,
                        "metadata": doc.metadata,
                    }
                )
        return records

_VALID_MODES = ("dense", "bm25", "hybrid")


@dataclass
class RetrievalResult:
    """One RAG tool result: the query plus its ordered top-k hits."""

    query: str
    results: list[dict[str, Any]] = field(default_factory=list)
    backend: str = ""
    retrieval_mode: str = "hybrid"
    index_size: int = 0
    execution_time_ms: float = 0.0


class RAGTool:
    """Read-only retrieval over the enterprise KB (dense + BM25 + RRF).

    The tool assumes the knowledge base has been built
    (``scripts/build_kb.py`` -> Milvus; BM25 is built in-process over the
    same chunk corpus).  An empty result is a normal, honest outcome.
    """

    name: str = "rag"
    description: str = (
        "Retrieve enterprise policy / procedure documents "
        "(Dense + BM25 + RRF hybrid, top-k, source cited)."
    )
    input_schema: dict[str, Any] = {
        "query": "str",
        "top_k": "int (default 5)",
        "retrieval_mode": "str in {dense, bm25, hybrid} (default hybrid)",
    }
    timeout: int = 20

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        *,
        store: VectorStore | None = None,
        backend: str | None = None,
        retrieval_mode: str = "hybrid",
    ) -> None:
        if retrieval_mode not in _VALID_MODES:
            raise ValueError(f"retrieval_mode must be one of {_VALID_MODES}, got {retrieval_mode!r}")
        self._retrieval_mode = retrieval_mode
        if retriever is not None:
            self._retriever = retriever
        else:
            self._retriever = HybridRetriever(vector_store=store, backend=backend)

    @observe("rag_retrieval")
    def run(self, *, query: str, top_k: int = 5, retrieval_mode: str | None = None) -> dict[str, Any]:
        if not query or not query.strip():
            return {
                "tool": self.name,
                "ok": False,
                "error": "empty query",
                "results": [],
                "backend": self._retriever.backend,
                "retrieval_mode": self._retrieval_mode,
            }
        mode = retrieval_mode or self._retrieval_mode
        top_k = max(1, int(top_k))
        hits: list[HybridHit] = self._retriever.search(query, top_k=top_k, mode=mode)
        results = [self._hit_to_payload(h) for h in hits]
        return {
            "tool": self.name,
            "ok": True,
            "results": results,
            "backend": self._retriever.backend,
            "retrieval_mode": mode,
            "index_size": self.index_size,
        }

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _hit_to_payload(hit: HybridHit) -> dict[str, Any]:
        return {
            "chunk_id": hit.chunk_id,
            "document_id": hit.document_id,
            "text": hit.text,
            "title": hit.title,
            "source": hit.source,
            "category": hit.category,
            "metadata": hit.metadata,
            # unified hybrid fields
            "dense_score": hit.dense_score,
            "dense_rank": hit.dense_rank,
            "bm25_score": hit.bm25_score,
            "bm25_rank": hit.bm25_rank,
            "fusion_score": hit.fusion_score,
            # Phase 2.1 compatibility: the primary score is the fusion value.
            "score": hit.fusion_score,
        }

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    @property
    def store(self) -> VectorStore:
        return self._retriever.vector_store

    @property
    def index_size(self) -> int:
        return self._retriever.vector_store.size()


# ---------------------------------------------------------------------------
# Offline / test helper — populate a fake store without BGE-M3 / Milvus
# ---------------------------------------------------------------------------


def iter_kb_chunks_for_store(
    kb_root: Path | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict[str, Any]]:
    """Yield pre-chunked, hash-embedded records for test-mode stores.

    Chunking is delegated to ``scripts/build_kb`` so the in-process chunk
    ids are identical to the ones inside Milvus — a BM25 hit can always be
    matched to the same dense collection.  Hash embeddings keep this helper
    fully offline (no model, no server).
    """
    from src.core.embedder import HashEmbedder

    embed = HashEmbedder()
    records = []
    for record in _chunk_records(kb_root=kb_root, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
        record = dict(record)
        record["vector"] = embed(str(record["text"]))
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Document parsing (shared with scripts/build_kb.py)
# ---------------------------------------------------------------------------


@dataclass
class _Document:
    doc_id: str
    title: str
    body: str
    source: str
    category: str
    metadata: dict[str, Any]


_FRONTMATTER_RE = re.compile(
    r"^#\s*---\s+frontmatter\s+---\n(.*?)(?:\n# --- frontmatter end ---|\Z)",
    re.DOTALL,
)


def _iter_documents(root: Path) -> list[_Document]:
    docs: list[_Document] = []
    if not root.exists():
        return docs
    for md_file in sorted(root.rglob("*.md")):
        if md_file.name == "README.md" or md_file.name.startswith("_"):
            continue
        content = md_file.read_text(encoding="utf-8")
        metadata = _parse_frontmatter(content)
        doc_id = str(metadata.get("doc_id", md_file.stem))
        title = str(metadata.get("title", _first_h1(content) or md_file.stem))
        category = str(metadata.get("department", md_file.parent.name))
        metadata.setdefault("doc_id", doc_id)
        metadata.setdefault("department", category)
        body = _FRONTMATTER_RE.sub("", content).strip()
        source = f"{md_file.parent.name}/{md_file.stem}"
        if body:
            docs.append(_Document(doc_id=doc_id, title=title, body=body, source=source, category=category, metadata=metadata))
    return docs


def _first_h1(content: str) -> str | None:
    m = re.search(r"^#\s+(.+?)\s*$", content, re.MULTILINE)
    return m.group(1).strip() if m else None


def _parse_frontmatter(content: str) -> dict[str, Any]:
    match = re.match(
        r"^#\s*---\s+frontmatter\s+---\n(.*?)(?:\n# --- frontmatter end ---)",
        content,
        re.DOTALL,
    )
    if not match:
        return {}
    data: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        line = line.lstrip("#").strip()
        if ":" in line:
            key, _, value = line.partition(":")
            data[key.strip()] = value.strip()
    return data


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Heading/paragraph-first chunking (same algorithm as scripts/build_kb.py)."""
    text = text.strip()
    if not text:
        return []
    sections = _split_by_heading(text)
    chunks: list[str] = []
    for section in sections:
        chunks.extend(_split_section(section, chunk_size, overlap))
    return chunks


def _split_by_heading(text: str) -> list[str]:
    lines = text.splitlines()
    sections: list[list[str]] = [[]]
    for line in lines:
        if re.match(r"^#{2,3}\s", line):
            sections.append([line])
        else:
            sections[-1].append(line)
    out: list[str] = []
    for s in sections:
        body = "\n".join(s).strip()
        if body:
            out.append(body)
    return out


def _split_section(section: str, chunk_size: int, overlap: int) -> list[str]:
    section = section.strip()
    if not section:
        return []
    if len(section) <= chunk_size:
        return [section]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
    result: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                result.append(current)
                current = ""
            result.extend(_hard_split(para, chunk_size, overlap))
            continue
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                result.append(current)
            current = para
    if current:
        result.append(current)
    return result


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = max(chunk_size - overlap, 1)
    out: list[str] = []
    for i in range(0, len(text), step):
        out.append(text[i : i + chunk_size])
        if i + chunk_size >= len(text):
            break
    return out


def _chunk_id(doc_id: str, text: str, metadata: dict[str, Any]) -> str:
    digest = hashlib.md5(f"{doc_id}:{text[:128]}:{metadata.get('department', '')}".encode()).hexdigest()[:10]
    return f"kb-{digest}"


__all__ = [
    "RAGTool",
    "RetrievalResult",
    "iter_kb_chunks_for_store",
]
