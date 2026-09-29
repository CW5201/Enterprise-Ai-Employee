"""RAG tool — minimal enterprise knowledge retrieval (Phase 1).

Pipeline: document parse -> chunk -> embed (BGE-M3, hashing fallback) ->
vector store (Milvus when reachable, persistent local index otherwise) ->
top-k search -> results with source preserved.

Deliberately NOT in Phase 1 (interfaces kept open for Phase 2): BM25
hybrid, reranker, incremental updates, complex metadata filtering.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.observability import observe
from src.core.vector_store import VectorStore, create_vector_store

_KB_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base"


class RAGTool:
    """Read-only retrieval over the local knowledge base via the vector store."""

    name: str = "rag_tool"
    description: str = "Retrieve enterprise policy / procedure documents (top-k, source cited)."
    input_schema: dict[str, Any] = {"query": "str", "top_k": "int (default 5)"}
    timeout: int = 20

    def __init__(self, store: VectorStore | None = None, kb_root: Path | None = None) -> None:
        self._store = store or create_vector_store()
        self._kb_root = kb_root or _KB_ROOT
        self._built = False

    # -- indexing -----------------------------------------------------------

    def build_index(
        self, *, chunk_size: int = 512, overlap: int = 64, rebuild: bool = False
    ) -> int:
        """Parse + chunk + embed + insert into the vector store.

        Idempotent by default (skips when the store already has data);
        ``rebuild=True`` drops and reindexes from scratch.
        """
        if self._store.size() > 0 and not rebuild:
            self._built = True
            return self._store.size()
        records: list[dict[str, Any]] = []
        for doc in _iter_documents(self._kb_root):
            for chunk in _chunk_text(doc.body, chunk_size, overlap):
                chunk_id = _chunk_id(doc.doc_id, chunk, doc.metadata)
                records.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "source": doc.source,
                        "metadata": doc.metadata,
                    }
                )
        from src.core.embedder import create_embedder

        embed = create_embedder()
        for record in records:
            record["vector"] = embed(record["text"])
        self._store.insert(records)
        self._built = True
        return len(records)

    # -- retrieval ----------------------------------------------------------

    @observe("rag_retrieval")
    def run(self, *, query: str, top_k: int = 5) -> dict[str, Any]:
        if not self._built:
            self.build_index()
        start = time.perf_counter()
        hits = self._store.search(query, top_k=top_k)
        results = [hit.to_payload() for hit in hits]
        return {
            "tool": self.name,
            "ok": True,
            "results": results,
            "backend": self._store.backend,
            "index_size": self._store.size(),
            "execution_time_ms": round((time.perf_counter() - start) * 1000.0, 2),
        }

    @property
    def store(self) -> VectorStore:
        return self._store

    @property
    def index_size(self) -> int:
        return self._store.size()


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------


@dataclass
class _Document:
    doc_id: str
    title: str
    body: str
    source: str
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
        title = str(metadata.get("title", md_file.stem))
        metadata.setdefault("doc_id", doc_id)
        metadata.setdefault("department", md_file.parent.name)
        body = _FRONTMATTER_RE.sub("", content).strip()
        source = f"{md_file.parent.name}/{md_file.stem}"
        if body:
            docs.append(_Document(doc_id, title, body, source, metadata))
    return docs


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
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(text), step):
        chunks.append(text[i : i + chunk_size])
        if i + chunk_size >= len(text):
            break
    return chunks


def _chunk_id(doc_id: str, text: str, metadata: dict[str, Any]) -> str:
    digest = hashlib.md5(f"{doc_id}:{text[:128]}:{metadata.get('department', '')}".encode()).hexdigest()[:10]
    return f"kb-{digest}"


__all__ = ["RAGTool"]
