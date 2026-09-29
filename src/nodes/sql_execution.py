"""Node: SQL execution — Text-to-SQL generation + safe DuckDB execution.

Pipeline (docs/ROADMAP.md Phase 1, no hardcoded per-question SQL):

    user question
      -> schema context   (data/schemas/database_schema.yaml)
      -> data dictionary  (data/schemas/data_dictionary.yaml)
      -> few-shot examples (data/schemas/sql_fewshots.yaml)
      -> LLM generates one read-only statement (structured, validated)
      -> SQL guard (SELECT/WITH only)
      -> DuckDB execution (read-only, timed, row-capped)
      -> structured result into AgentState.sql / sql_result / evidence

On LLM or execution failure the error is recorded in ``state.errors`` and the
node still returns so the pipeline can produce an honest answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from src.core.exceptions import LLMError, SQLExecutionError, SQLGuardError
from src.core.llm_client import LLMClient
from src.core.observability import observe
from src.core.sql_executor import SQLExecutor, default_database_path
from src.core.state import AgentState, ErrorRecord, EvidenceItem, SQLResultPayload, ToolCallRecord

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCHEMAS_DIR = _PROJECT_ROOT / "data" / "schemas"


class TextToSQLResult(BaseModel):
    """Structured output of the Text-to-SQL LLM call (always Pydantic-checked)."""

    sql: str = Field(min_length=8)
    rationale: str = ""


class SQLExecutionNode:
    def __init__(
        self,
        llm: LLMClient | None = None,
        executor: SQLExecutor | None = None,
        schemas_dir: Path | None = None,
    ) -> None:
        self._llm = llm or LLMClient()
        self._executor = executor or SQLExecutor(db_path=default_database_path())
        self._schemas_dir = schemas_dir or _SCHEMAS_DIR

    @observe("sql_execution")
    def run(self, state: AgentState) -> AgentState:
        question = state.get("user_query", "")
        intent = state.get("intent", "")
        errors: list[ErrorRecord] = []
        tool_calls: list[ToolCallRecord] = []
        evidence: list[EvidenceItem] = []

        # 1. Build the prompt context (schema + dictionary + few-shots)
        context = _build_context(self._schemas_dir)
        system = _SYSTEM_PROMPT.format(schema=context["schema"], dictionary=context["dictionary"], fewshots=context["fewshots"])
        user = _USER_TEMPLATE.format(question=question, intent=intent, entities=json.dumps(state.get("entities") or [], ensure_ascii=False))

        # 2. LLM generates SQL (validated by Pydantic)
        sql = ""
        try:
            parsed: TextToSQLResult = self._llm.generate_structured(system, user, TextToSQLResult)
            sql = parsed.sql
        except LLMError as exc:
            errors.append(ErrorRecord(stage="text_to_sql", message=exc.message))
            sql = ""

        # 3. Guard + execute (only when we actually have SQL)
        sql_result: SQLResultPayload | None = None
        if sql:
            try:
                payload: dict[str, Any] = self._executor.execute(sql)
                sql_result = SQLResultPayload(**payload)
                tool_calls.append(
                    ToolCallRecord(
                        tool="sql_tool",
                        ok=True,
                        input={"sql": sql},
                        evidence_ref=f"duckdb:{sql}",
                        duration_ms=payload.get("execution_time_ms", 0.0),
                    )
                )
                evidence.append(
                    EvidenceItem(
                        evidence_id=f"ev-{state.get('request_id', 'req')}-sql",
                        source_type="duckdb",
                        source_ref=f"duckdb:{sql}",
                        content=_summarise(sql, payload),
                        payload={"columns": payload["columns"], "rows": payload["rows"], "row_count": payload["row_count"]},
                        metadata={"intent": intent, "rationale": ""},
                    )
                )
            except SQLGuardError as exc:
                errors.append(ErrorRecord(stage="sql_guard", message=exc.message))
                tool_calls.append(ToolCallRecord(tool="sql_tool", ok=False, input={"sql": sql}, error=exc.message))
            except SQLExecutionError as exc:
                errors.append(ErrorRecord(stage="sql_execution", message=exc.message, details=exc.details))
                tool_calls.append(ToolCallRecord(tool="sql_tool", ok=False, input={"sql": sql}, error=exc.message))

        return {
            "sql": sql,
            "sql_result": sql_result,
            "tool_calls": tool_calls,
            "evidence": evidence,
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是企业数据的 Text-to-SQL 生成器，目标数据库是 DuckDB（WideWorldImporters，只读）。
可用的表结构:
{schema}
数值口径与业务语义:
{dictionary}
参考示例:
{fewshots}
规则:
- 只生成一条只读 SELECT 或 WITH...SELECT 语句;
- 禁止 INSERT/UPDATE/DELETE/DROP/CREATE/ALTER 等任何写操作;
- 数值口径必须参考数据字典; 表名/列名必须与上面的结构一致;
- 对 "订单数量最多的客户" 这类问题, 用 COUNT + GROUP BY + ORDER BY 实现, 不要硬编码。
返回 JSON: {{"sql": str, "rationale": str}}"""

_USER_TEMPLATE = "用户问题: {question}\n意图: {intent}\n实体: {entities}"


def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# 业务核心表优先入 prompt（WWI 52 张表里真正被 Text-to-SQL 问题引用的），
# 其余表按顺序补齐；上限 max_tables 控制 prompt 长度。
_PREFERRED_TABLES = [
    "Sales_Customers", "Sales_Orders", "Sales_OrderLines",
    "Sales_Invoices", "Sales_InvoiceLines",
    "Purchasing_Suppliers", "Purchasing_SupplierCategories",
    "Warehouse_StockItems", "Warehouse_StockGroups", "Warehouse_StockItemStockGroups",
    "Application_Cities", "Application_StateProvinces", "Application_Countries",
    "Application_People", "Application_DeliveryMethods", "Application_PaymentMethods",
    "Application_TransactionTypes", "Sales_BuyingGroups", "Sales_CustomerCategories",
    "Purchasing_PurchaseOrders", "Purchasing_PurchaseOrderLines",
]


def _render_schema(doc: Any, max_tables: int = 22) -> str:
    """Compact the generated database_schema.yaml into prompt-ready text.

    优先保留业务核心表（见 _PREFERRED_TABLES），再按 YAML 原顺序补齐到
    max_tables——避免 52 张表被前 12 张字母序表占满、真正的 Sales_* 业务表
    永远进不了 prompt。
    """
    if not doc or "tables" not in doc:
        return "(schema not available)"
    tables = doc["tables"]
    by_name = {t["name"]: t for t in tables}
    ordered = [by_name[n] for n in _PREFERRED_TABLES if n in by_name]
    seen = {t["name"] for t in ordered}
    ordered += [t for t in tables if t["name"] not in seen]
    lines: list[str] = []
    for table in ordered[:max_tables]:
        cols = table.get("columns", [])
        pk = table.get("primary_key") or []
        col_desc = ", ".join(
            f"{c['name']}:{c['type']}" + (" (pk)" if c["name"] in pk else "") for c in cols
        )
        lines.append(f"table {table['name']}: {col_desc}")
    return "\n".join(lines) if lines else "(no tables)"


def _render_dictionary(doc: Any) -> str:
    if not doc or "metrics" not in doc:
        return "(dictionary not available)"
    out: list[str] = []
    for name, spec in (doc.get("metrics") or {}).items():
        out.append(f"- {name} ({spec.get('chinese', '')}): {spec.get('definition', '')} [{spec.get('formula', '')}]")
    for note in doc.get("notes", []) or []:
        out.append(f"note: {note}")
    return "\n".join(out) if out else "(empty)"


def _render_fewshots(doc: Any, max_examples: int = 3) -> str:
    if not doc or "few_shots" not in doc:
        return "(no examples)"
    out: list[str] = []
    for ex in doc.get("few_shots", [])[:max_examples]:
        out.append(f"Q: {ex.get('question', '')}\nSQL:\n{ex.get('sql', '')}")
    return "\n".join(out) if out else "(none)"


def _build_context(schemas_dir: Path) -> dict[str, str]:
    schema_doc = _load_yaml(schemas_dir / "database_schema.yaml")
    dict_doc = _load_yaml(schemas_dir / "data_dictionary.yaml")
    few_doc = _load_yaml(schemas_dir / "sql_fewshots.yaml")
    return {
        "schema": _render_schema(schema_doc),
        "dictionary": _render_dictionary(dict_doc),
        "fewshots": _render_fewshots(few_doc),
    }


def _summarise(sql: str, payload: dict[str, Any]) -> str:
    columns = payload.get("columns", [])
    rows = payload.get("rows", [])
    head = " | ".join(str(c) for c in columns)
    preview = "\n".join(" | ".join(str(v) for v in row) for row in rows[:5])
    return f"SQL: {sql}\nColumns: {head}\nRows ({len(rows)} shown):\n{preview}"


__all__ = ["SQLExecutionNode", "TextToSQLResult"]
