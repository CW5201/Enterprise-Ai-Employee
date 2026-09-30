"""Evidence adapter — uniform RAG/SQL/KG/Analysis results -> Evidence[].

Phase 4 Commit 2.  The verification node must not know the internal
payload shape of any individual tool (RAG hit dicts, SQL rows, KG
template results, analysis outputs).  This module is the *only* place
that translates each tool's native result into the Phase 4
:class:`src.core.verification_types.Evidence` model, with full
provenance.

Adapters provided:

- :func:`adapt_rag`       — RAGTool hit dict -> rag evidence
- :func:`adapt_sql`       — SQL tool / node payload -> sql evidence
- :func:`adapt_kg`        — KGTool result dict -> kg evidence
- :func:`adapt_analysis`  — AnalysisTool result dict -> analysis evidence
- :func:`adapt_tool_result` — dispatch by tool name
- :func:`adapt_state_evidence` — legacy :class:`EvidenceItem` channel
  -> Phase 4 evidence (used by the evidence_collection node on top of
  ``state["evidence"]``)

Each adapter returns a list of ``Evidence`` objects.  Callers may pass
either a single payload dict (per-hit) or a list of them; the adapters
normalise to a list either way.  Duplicate source refs are deduped by
the collection node, not here.
"""

from __future__ import annotations

from typing import Any

from src.core.verification_types import Evidence, normalize_source_type

__all__ = [
    "adapt_rag",
    "adapt_sql",
    "adapt_kg",
    "adapt_analysis",
    "adapt_tool_result",
    "adapt_state_evidence",
]


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


def adapt_rag(hit: dict[str, Any] | Any, *, evidence_id: str = "") -> Evidence:
    """One RAG hit -> one rag evidence with doc/chunk provenance.

    ``hit`` is either the RAGTool payload dict (has ``chunk_id``) or a
    :class:`src.core.state.RetrievedChunk`-shaped object (attribute access).
    """
    if hasattr(hit, "chunk_id"):  # RetrievedChunk model
        chunk_id = str(getattr(hit, "chunk_id", ""))
        text = str(getattr(hit, "text", ""))
        source = str(getattr(hit, "source", ""))
        score = float(getattr(hit, "score", 0.0) or 0.0)
        metadata = dict(hit.metadata or {})
        doc_id = str(metadata.get("document_id", ""))
        title = str(metadata.get("title", ""))
    else:
        h = dict(hit)
        chunk_id = str(h.get("chunk_id", ""))
        text = str(h.get("text", ""))
        source = str(h.get("source", ""))
        score = float(h.get("score", h.get("fusion_score", 0.0)) or 0.0)
        metadata = dict(h.get("metadata") or {})
        doc_id = str(h.get("document_id", ""))
        title = str(h.get("title", ""))

    provenance = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "title": title or source,
        "source": source,
    }
    return Evidence(
        evidence_id=evidence_id or f"ev-rag-{chunk_id}",
        source_type=normalize_source_type("rag"),
        source_ref=f"rag:{chunk_id}",
        content=text[:500],
        provenance=provenance,
        retrieved_score=score,
        metadata=metadata,
        structured_value=None,
    )


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------


def adapt_sql(payload: dict[str, Any], *, evidence_id: str = "", database: str = "duckdb") -> Evidence:
    """One SQL result -> one sql evidence with query + result provenance.

    ``payload`` keys (from the SQL node / SQL tool):
    ``sql`` (str), ``columns`` (list), ``rows`` (list of lists),
    ``row_count`` (int), ``execution_time_ms`` (float, optional).

    The executed query is ALWAYS kept in provenance — never just the
    scalar answer (spec requirement).
    """
    sql = str(payload.get("sql", "") or "")
    columns = [str(c) for c in (payload.get("columns") or [])]
    rows = list(payload.get("rows") or [])
    row_count = int(payload.get("row_count", len(rows)))

    # structured_value: the scalar / tabular value the evidence asserts.
    # Keep the full table when multi-row; a single value when it's one.
    if rows and len(rows) == 1 and len(rows[0]) == 1:
        structured_value: Any = rows[0][0]
    else:
        structured_value = {"columns": columns, "rows": rows, "row_count": row_count}

    provenance: dict[str, Any] = {
        "database": database,
        "table": _guess_table(sql),
        "query": sql,
        "columns": columns,
        "row_count": row_count,
    }
    content = f"SQL: {sql}\nRows ({row_count}): {rows[:10]}"
    return Evidence(
        evidence_id=evidence_id or "ev-sql",
        source_type=normalize_source_type("sql"),
        source_ref=f"sql:{sql}",
        content=content,
        provenance=provenance,
        structured_value=structured_value,
        metadata={"execution_time_ms": float(payload.get("execution_time_ms", 0.0) or 0.0)},
    )


def _guess_table(sql: str) -> str:
    """Best-effort table reference from a SQL statement (for provenance)."""
    import re

    m = re.search(r"(?:FROM|JOIN)\s+([A-Za-z_][\w]*)", sql, re.IGNORECASE)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# KG
# ---------------------------------------------------------------------------


def adapt_kg(result: dict[str, Any], *, evidence_id: str = "") -> Evidence:
    """One KG tool result -> kg evidence with template + entity provenance.

    ``result`` is the :class:`src.tools.kg_tool.KGTool` envelope:
    ``success`` / ``data`` (records) / ``template_id`` / ``query_type`` /
    ``source``.
    """
    ok = bool(result.get("success"))
    records = result.get("data") or []
    template_id = str(result.get("template_id", result.get("query_type", "")))
    structured_value = records if ok else None
    provenance = {
        "template_id": template_id,
        "query_type": str(result.get("query_type", "")),
        "result_ref": f"kg:{template_id}",
        "rows": len(records) if ok else 0,
    }
    content = f"KG[{template_id}]: {len(records)} record(s)" if ok else f"KG[{template_id}]: failed"
    return Evidence(
        evidence_id=evidence_id or f"ev-kg-{template_id}",
        source_type=normalize_source_type("kg"),
        source_ref=f"kg:{template_id}",
        content=content[:500],
        provenance=provenance,
        structured_value=structured_value,
        metadata={"source": result.get("source", "neo4j")},
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def adapt_analysis(result: dict[str, Any], *, evidence_id: str = "", input_evidence_ids: list[str] | None = None) -> Evidence:
    """One analysis result -> analysis evidence with operation + inputs.

    ``result`` is the :class:`src.tools.analysis_tool.AnalysisTool` payload:
    ``ok`` / ``data`` (``operation``, ``result``, ``input_rows``) /
    ``metadata`` (``operation``).  ``input_evidence_ids`` links back to the
    evidence the analysis was computed over.
    """
    ok = bool(result.get("ok", result.get("success")))
    data = result.get("data") or {}
    operation = str(data.get("operation") or (result.get("metadata") or {}).get("operation") or "")
    op_result = data.get("result")
    structured_value = op_result if ok else None
    provenance = {
        "operation": operation,
        "input_evidence_ids": list(input_evidence_ids or []),
        "input_rows": int(data.get("input_rows", 0) or 0),
    }
    content = f"analysis[{operation}]: {op_result}" if ok else f"analysis[{operation}]: failed"
    return Evidence(
        evidence_id=evidence_id or f"ev-analysis-{operation}",
        source_type=normalize_source_type("analysis"),
        source_ref=f"analysis:{operation}",
        content=content[:500],
        provenance=provenance,
        structured_value=structured_value,
        metadata={"operation": operation},
    )


# ---------------------------------------------------------------------------
# Dispatch + legacy state adapter
# ---------------------------------------------------------------------------


def adapt_tool_result(tool: str, payload: Any, *, evidence_id: str = "", input_evidence_ids: list[str] | None = None) -> list[Evidence]:
    """Dispatch a tool result payload to the right adapter."""
    if tool == "rag":
        hits = payload.get("results") if isinstance(payload, dict) else None
        out: list[Evidence] = []
        for i, h in enumerate(hits or []):
            out.append(adapt_rag(h, evidence_id=evidence_id or f"ev-rag-{i}"))
        return out
    if tool == "sql":
        p = payload if isinstance(payload, dict) else {}
        return [adapt_sql(p, evidence_id=evidence_id, database=str(p.get("database", "duckdb")))]
    if tool == "kg":
        return [adapt_kg(payload, evidence_id=evidence_id)] if payload else []
    if tool == "analysis":
        return [adapt_analysis(payload, evidence_id=evidence_id, input_evidence_ids=input_evidence_ids)]
    # unknown tool: no silent fabrication of evidence
    return []


def adapt_state_evidence(items: list[Any]) -> list[Evidence]:
    """Legacy :class:`EvidenceItem` channel -> Phase 4 Evidence.

    Items may be :class:`EvidenceItem` (attribute access) or dicts.
    Legacy ``source_type`` (duckdb/milvus/fake/neo4j) is normalised onto
    the Phase 4 vocabulary (sql/rag/kg/analysis).
    """
    out: list[Evidence] = []
    for item in items or []:
        d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        raw_type = str(d.get("source_type", ""))
        stype = normalize_source_type(raw_type)
        if stype not in ("sql", "rag", "kg", "analysis"):
            # unknown legacy source: keep it but don't invent a category
            continue
        payload = d.get("payload") or {}
        provenance: dict[str, Any] = dict(payload.get("provenance") or {})
        # enrich with what we can recover from the item's payload fields
        if stype == "rag":
            provenance.setdefault("chunk_id", str(payload.get("chunk_id", "")))
            provenance.setdefault("doc_id", str(payload.get("document_id", "")))
            provenance.setdefault("title", str((payload.get("title") or (d.get("metadata") or {}).get("title")) or ""))
        elif stype == "sql":
            provenance.setdefault("query", str(payload.get("sql", "")))
            provenance.setdefault("table", _guess_table(str(payload.get("sql", ""))))
            provenance.setdefault("database", "duckdb")
        elif stype == "kg":
            provenance.setdefault("template_id", str(payload.get("template_id", "")))
        # structured value: prefer an explicit one, else the payload's rows
        structured = payload.get("structured_value")
        if structured is None and stype == "sql" and "rows" in payload:
            cols = payload.get("columns") or []
            rows = payload.get("rows") or []
            if rows and len(rows) == 1 and len(rows[0]) == 1:
                structured = rows[0][0]
            else:
                structured = {"columns": cols, "rows": rows, "row_count": len(rows)}
        out.append(Evidence(
            evidence_id=str(d.get("evidence_id", f"ev-{len(out)}")),
            source_type=stype,  # type: ignore[arg-type]
            source_ref=str(d.get("source_ref", "")),
            content=str(d.get("content", ""))[:500],
            metadata=dict(d.get("metadata") or {}),
            provenance=provenance,
            retrieved_score=d.get("score"),
            structured_value=structured,
        ))
    return out
