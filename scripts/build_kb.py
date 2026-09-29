"""Phase 2.1：知识库构建 — 文档 -> chunk -> BGE-M3 -> Milvus 入库。

流程：

    data/knowledge_base/*.md（真实存在的合成企业知识文档）
      -> 按标题/段落切分（稳定、可重复）
      -> 保留 metadata（chunk_id / document_id / title / source / category / text）
      -> BGE-M3 真实 embedding
      -> Milvus insert（collection 名称来自 config/settings.yaml）

正式模式（默认）：BGE-M3 + Milvus，任何一步失败直接抛异常，不做 hash /
fake 兜底。

用法（项目根目录）::

    python scripts/build_kb.py            # 增量入库（已有文档跳过）
    python scripts/build_kb.py --rebuild  # 重建：先删旧 collection 再全部重新入库
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 让 .env 的变量生效（EMBEDDING_MODEL / EMBEDDING_DEVICE / MILVUS_HOST 等）
from src.core.llm_client import _load_dotenv  # noqa: E402

_load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_KB_ROOT = _PROJECT_ROOT / "data" / "knowledge_base"


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------


@dataclass
class Document:
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


def _first_h1(content: str) -> str | None:
    m = re.search(r"^#\s+(.+?)\s*$", content, re.MULTILINE)
    return m.group(1).strip() if m else None


def iter_documents(root: Path) -> list[Document]:
    docs: list[Document] = []
    if not root.exists():
        return docs
    for md_file in sorted(root.rglob("*.md")):
        if md_file.name == "README.md" or md_file.name.startswith("_"):
            continue
        content = md_file.read_text(encoding="utf-8")
        metadata = _parse_frontmatter(content)
        doc_id = str(metadata.get("doc_id", md_file.stem))
        category = str(metadata.get("department", md_file.parent.name))
        # The doc title is only read from frontmatter when it exists;
        # otherwise it comes from the first markdown H1 heading.  This keeps
        # chunk metadata independent of the file stem so display titles stay
        # in sync with the human-readable document title.
        if "title" in metadata:
            title = str(metadata["title"])
        else:
            h1 = _first_h1(content)
            title = h1 if h1 else md_file.stem
        metadata.setdefault("doc_id", doc_id)
        metadata.setdefault("department", category)
        body = _FRONTMATTER_RE.sub("", content).strip()
        source = f"{md_file.parent.name}/{md_file.stem}"
        if body:
            docs.append(Document(doc_id=doc_id, title=title, body=body, source=source, category=category, metadata=metadata))
    return docs


# ---------------------------------------------------------------------------
# Chunking — stable, heading/paragraph-first, then character cap with overlap
# ---------------------------------------------------------------------------


def chunk_text(
    text: str, *, chunk_size: int = 512, overlap: int = 64, doc_id: str = ""
) -> list[dict[str, Any]]:
    """Split text into chunks.

    Strategy (stable + repeatable):
    1. split on markdown headings (## / ###) first, keeping the heading line
       attached to its section;
    2. sections longer than ``chunk_size`` are split on paragraphs;
    3. anything still too long is hard-cut to the character cap with overlap;
    4. each chunk gets a deterministic ``chunk_id`` derived from doc + index +
       first characters.
    """
    text = text.strip()
    if not text:
        return []
    sections = _split_by_heading(text)
    chunks: list[str] = []
    for section in sections:
        chunks.extend(_split_section(section, chunk_size, overlap))
    records: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        digest = hashlib.md5(f"{doc_id}:{i}:{chunk[:128]}".encode()).hexdigest()[:10]
        records.append(
            {
                "chunk_id": f"kb-{digest}",
                "text": chunk,
                "section_index": i,
            }
        )
    return records


def _split_by_heading(text: str) -> list[str]:
    lines = text.splitlines()
    sections: list[list[str]] = [[]]
    for line in lines:
        if re.match(r"^#{2,3}\s", line):  # ## or ###
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


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def build(
    *,
    kb_root: Path | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Parse + chunk + embed (BGE-M3) + insert into Milvus.

    Returns a summary dict; raises on any formal-step failure (no fallback).
    """
    from src.core.embedder import create_embedder
    from src.core.vector_store import create_vector_store

    root = kb_root or _KB_ROOT
    embedder = create_embedder()
    print(f"[build-kb] embedder backend = {embedder.backend} dim={embedder.dim}")

    store = create_vector_store(backend="milvus")
    print(f"[build-kb] milvus uri = {getattr(store, '_uri', 'unknown')} collection = {store.collection}")

    if rebuild:
        print("[build-kb] rebuild=True: dropping existing collection")
        store.drop()

    store.create_collection()
    t0 = time.time()
    docs = iter_documents(root)
    print(f"[build-kb] found {len(docs)} documents under {root.name}/")

    records: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_text(
            doc.body,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            doc_id=doc.doc_id,
        )
        texts = [c["text"] for c in chunks]
        vectors = embedder.embed_batch(texts)
        for chunk, vector in zip(chunks, vectors, strict=True):
            records.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": doc.doc_id,
                    "title": doc.title,
                    "source": doc.source,
                    "category": doc.category,
                    "text": chunk["text"],
                    "metadata": doc.metadata,
                    "vector": vector,
                }
            )
    print(f"[build-kb] produced {len(records)} chunks; embedding done in {time.time() - t0:.1f}s")

    if not records:
        raise RuntimeError(
            f"knowledge base is empty (no markdown docs under {root}); "
            "nothing to index"
        )

    inserted = store.insert(records)
    entity_count = store.size()
    print(f"[build-kb] inserted {inserted} chunks; Milvus entity_count = {entity_count}")
    return {
        "documents": len(docs),
        "chunks": len(records),
        "inserted": inserted,
        "entity_count": entity_count,
        "collection": store.collection,
        "dimension": embedder.dim,
        "backend": "milvus",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="构建企业知识库并写入 Milvus (Phase 2.1)")
    parser.add_argument("--rebuild", action="store_true", help="删除旧 collection 后全量重建")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()
    summary = build(
        rebuild=args.rebuild,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print("[build-kb] summary:", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
