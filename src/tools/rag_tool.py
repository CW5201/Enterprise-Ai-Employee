"""RAG tool — BGE-M3 + Milvus dense retrieval with source preserved.

Pipeline (Phase 2.1): query -> BGE-M3 -> Milvus Top-K -> RetrievalResult.

Every returned chunk carries ``source`` / ``title`` / ``category`` /
``score`` so answers can cite their origin.  The formal backend is
Milvus; the fake (in-memory) backend is for unit tests only and must be
selected explicitly (``create_vector_store(backend="fake")``).

``iter_kb_chunks_for_store()`` is a test/offline helper that produces
pre-chunked records (with hash vectors) so the fake backend can be
populated without running BGE-M3 or Milvus.  It is NOT used by the
formal pipeline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.observability import observe
from src.core.vector_store import SearchHit, VectorStore, create_vector_store

_KB_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base"


@dataclass
class RetrievalResult:
    """One RAG tool result: the query plus its ordered top-k hits."""

    query: str
    results: list[dict[str, Any]] = field(default_factory=list)
    backend: str = ""
    index_size: int = 0
    execution_time_ms: float = 0.0


class RAGTool:
    """Read-only retrieval over the enterprise knowledge base in Milvus.

    The tool assumes the knowledge base has been built (``scripts/build_kb.py``)
    and the Milvus collection is populated.  Retrieval that finds nothing is a
    normal, honest result — it is never padded with fake hits.
    """

    name: str = "rag"
    description: str = "Retrieve enterprise policy / procedure documents (BGE-M3 + Milvus, top-k, source cited)."
    input_schema: dict[str, Any] = {"query": "str", "top_k": "int (default 5)"}
    timeout: int = 20

    def __init__(self, store: VectorStore | None = None, backend: str | None = None) -> None:
        self._store = store or create_vector_store(backend=backend)

    @observe("rag_retrieval")
    def run(self, *, query: str, top_k: int = 5) -> dict[str, Any]:
        if not query or not query.strip():
            return {
                "tool": self.name,
                "ok": False,
                "error": "empty query",
                "results": [],
                "backend": self._store.backend,
            }
        top_k = max(1, int(top_k))
        hits: list[SearchHit] = self._store.search(query, top_k=top_k)
        results = [
            {
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "text": hit.text,
                "title": hit.title,
                "source": hit.source,
                "category": hit.category,
                "metadata": hit.metadata,
                "score": hit.score,
            }
            for hit in hits
        ]
        return {
            "tool": self.name,
            "ok": True,
            "results": results,
            "backend": self._store.backend,
            "index_size": self._store.size(),
        }

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def index_size(self) -> int:
        return self._store.size()


# ---------------------------------------------------------------------------
# Offline / test helper — populate a fake store without BGE-M3 / Milvus
# ---------------------------------------------------------------------------


def iter_kb_chunks_for_store(
    kb_root: Path | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict[str, Any]]:
    """Yield pre-chunked, hash-embedded records for test-mode stores.

    Uses the deterministic hash embedder (test mode only) so no model or
    Milvus server is required.  Each record has the full Milvus schema
    fields so ``FakeVectorStore`` can store and filter them.
    """
    from src.core.embedder import HashEmbedder

    root = kb_root or _KB_ROOT
    embed = HashEmbedder()
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
                    "vector": embed(chunk),
                }
            )
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
