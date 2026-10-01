"""API route: knowledge - document listing and search.

Phase 6.5: minimal knowledge base API backed by the local Markdown
knowledge base under ``data/knowledge_base/``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(tags=["knowledge"])

KB_ROOT = Path(__file__).resolve().parents[3] / "data" / "knowledge_base"


class KnowledgeDocumentOut(BaseModel):
    id: str
    title: str
    department: str
    source: str
    effective_date: str | None = None
    synthetic: bool = True


def _scan_kb() -> list[dict[str, Any]]:
    """Walk the knowledge-base directory and build a flat document list."""
    docs: list[dict[str, Any]] = []
    if not KB_ROOT.is_dir():
        return docs

    for dept_dir in sorted(KB_ROOT.iterdir()):
        if not dept_dir.is_dir() or dept_dir.name.startswith("."):
            continue
        dept = dept_dir.name

        for f in sorted(dept_dir.iterdir()):
            if f.suffix != ".md" or f.name.upper() == "README.MD":
                continue
            doc_id = f.stem  # e.g. finance-0001-travel-reimbursement-policy
            title = re.sub(r"[-_]+", " ", f.stem).title()
            docs.append({
                "id": doc_id,
                "title": title,
                "department": dept,
                "source": f"knowledge_base/{dept}/{f.name}",
                "effective_date": None,
                "synthetic": True,
            })
    return docs


@router.get("/knowledge/documents", response_model=list[KnowledgeDocumentOut])
def list_documents(
    q: str | None = Query(default=None, description="Search by title / id"),
    department: str | None = Query(default=None, description="Filter by department"),
) -> list[KnowledgeDocumentOut]:
    docs = _scan_kb()
    if department:
        docs = [d for d in docs if d["department"].lower() == department.lower()]
    if q:
        ql = q.lower()
        docs = [d for d in docs if ql in d["id"].lower() or ql in d["title"].lower()]
    return docs


@router.get("/knowledge/documents/{doc_id}", response_model=KnowledgeDocumentOut)
def get_document(doc_id: str) -> KnowledgeDocumentOut:
    docs = [d for d in _scan_kb() if d["id"] == doc_id]
    if not docs:
        raise HTTPException(status_code=404, detail=f"document {doc_id!r} not found")
    return docs[0]


@router.get("/knowledge/documents/{doc_id}/content")
def get_document_content(doc_id: str) -> dict[str, Any]:
    """Return the raw Markdown text of one KB document."""
    docs = [d for d in _scan_kb() if d["id"] == doc_id]
    if not docs:
        raise HTTPException(status_code=404, detail=f"document {doc_id!r} not found")
    dept = docs[0]["department"]
    suffix = f"{doc_id}.md"
    candidates = [p for p in (KB_ROOT / dept).glob("*.md") if p.stem == doc_id]
    if not candidates:
        # fall back to matching by filename stem across departments
        candidates = list(KB_ROOT.rglob(f"*{suffix}"))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"file not found for {doc_id!r}")
    text = candidates[0].read_text(encoding="utf-8")
    return {
        "id": doc_id,
        "title": docs[0]["title"],
        "content": text,
    }


__all__ = ["router"]
