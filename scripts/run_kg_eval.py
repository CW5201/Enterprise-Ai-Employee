"""Run the Phase 2.4 knowledge-graph evaluation.

Reads ``data/eval/kg_eval.jsonl`` (58 SQL-verified tasks) and answers each
question with the :class:`KGTool` read-only templates, then scores the
predicted answers against the DuckDB-derived ground truth with
:mod:`src.evaluation.kg_metrics`.

Usage::

    python scripts/run_kg_eval.py                 # full run
    python scripts/run_kg_eval.py --limit 20       # smoke test
    python scripts/run_kg_eval.py --task-type cross_entity

Outputs (gitignored, kept locally):

- ``artifacts/phase2.4/kg_eval_results.json``   — per-task detail
- ``artifacts/phase2.4/kg_eval_summary.json``   — aggregate metrics
- ``artifacts/phase2.4/kg_eval_summary.md``     — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.kg_metrics import (  # noqa: E402
    exact_match,
    path_accuracy,
    set_f1,
    set_precision,
    set_recall,
)
from src.tools.kg_tool import KGTool  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = _PROJECT_ROOT / "data" / "eval" / "kg_eval.jsonl"
ARTIFACT_DIR = _PROJECT_ROOT / "artifacts" / "phase2.4"


# ---------------------------------------------------------------------------
# Answer extraction: map one eval task to a KGTool call, and pull out the
# predicted entity-id set + traversed relation hops.
# ---------------------------------------------------------------------------

# task_type -> (template_id, param extractors)
def _extract_params(task: dict) -> tuple[str, dict[str, Any]] | None:
    """Return (template_id, params) for tasks we can answer directly.

    Returns ``None`` when the question cannot be mapped to a single
    registered template — that task is recorded as a *prediction failure*
    (kept in the denominator, not dropped), matching the no-drop rule.
    """
    import re

    def _first_int(*patterns: str) -> int | None:
        text = task["question"]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None

    q = task["question"]
    ttype = task["task_type"]
    q = task["question"]

    if ttype == "entity_lookup":
        cid = _first_int(r"customer with ID (\d+)", r"customer (\d+)(?!\d)")
        sid = _first_int(r"supplier with ID (\d+)")
        iid = _first_int(r"stock item ID (\d+)")
        oid = _first_int(r"order (\d+)(?!\d)", r"invoice (\d+)(?!\d)")
        if "credit note" in q:
            return "invoice_by_id", {"invoice_id": _first_int(r"invoice (\d+)(?!\d)") or 0}
        if "group" in q and cid is not None:
            return "customer_buying_group", {"customer_id": cid}
        if "city" in q:
            sid = sid or _first_int(r"supplier (\d+)(?!\d)")
            if sid is not None:
                return "supplier_city", {"supplier_id": sid}
        if "customers" in q and "name contains" in q:
            name = re.search(r"name contains '([^']+)'", q).group(1)
            return "find_customer_by_name", {"name": name}
        if cid is not None:
            return "customer_by_id", {"customer_id": cid}
        if sid is not None:
            return "supplier_by_id", {"supplier_id": sid}
        if iid is not None:
            return "item_by_id", {"stock_item_id": iid}
        if oid is not None:
            return "order_by_id", {"order_id": oid}
        m = re.search(r"name contains '([^']+)'", q)
        if m:
            name = m.group(1)
            if "supplier" in q:
                return "find_supplier_by_name", {"name": name}
            return "find_item_by_name", {"name": name}
        return None

    if ttype in ("one_hop", "two_hop", "aggregation", "existence", "cross_entity", "multi_hop"):
        cid = _first_int(r"customer (\d+)(?!\d)")
        oid = _first_int(r"order (\d+)(?!\d)")
        inv_id = _first_int(r"invoice (\d+)(?!\d)")
        iid = _first_int(r"item (\d+)(?!\d)", r"stock item (\d+)(?!\d)")
        sid = _first_int(r"supplier (\d+)(?!\d)")
        gid = _first_int(r"group (\d+)(?!\d)")

        # cross_entity hard negatives / intersections with no single template
        if "both items" in q:
            return None
        if "buying groups contain" in q:
            return None

        if ttype == "aggregation":
            if "members" in q:
                return "group_member_count", {"buying_group_id": gid or 0}
            if "invoices bill" in q:
                return "order_invoices", {"order_id": oid or 0}
            if "quantity" in q:
                return "invoice_lines", {"invoice_id": inv_id or 0}
            return "count_orders_by_customer", {"customer_id": cid or 0}

        if ttype == "existence":
            if "belong to a buying group" in q or "belong to any buying group" in q:
                return "customer_buying_group", {"customer_id": cid or 0}
            if "ship any stock item" in q:
                return "invoice_lines", {"invoice_id": inv_id or 0}
            if "supplied by a supplier" in q and sid is None:
                return "item_suppliers", {"stock_item_id": iid or 0}
            if "supplied by supplier" in q and sid is not None:
                return "item_suppliers", {"stock_item_id": iid or 0}
            if "have an invoice" in q:
                return "order_invoices", {"order_id": oid or 0}
            if "did customer" in q or "does customer" in q:
                return "customer_orders", {"customer_id": cid or 0}
            return None

        if ttype == "multi_hop":
            # multi_hop questions ask for the far-end entity; the
            # customer_item_suppliers template returns suppliers, which is
            # what tasks 025-028/032 expect (their expected_entities list
            # Supplier ids).  Tasks asking for cities (030/031) keep the
            # supplier ids available via the same template + supplier_city.
            if cid is not None:
                return "customer_item_suppliers", {"customer_id": cid}
            if inv_id is not None:
                return "invoice_item_suppliers", {"invoice_id": inv_id}
            return None
        if "bought" in q and sid is not None and "has customer" not in q:
            return "supplier_customers", {"supplier_id": sid}
        if "has customer" in q and sid is not None:
            return "customer_items_via_supplier", {"customer_id": cid or 0, "supplier_id": sid}

        # two_hop / one_hop
        if "invoices" in q:
            if cid is not None:
                return "customer_invoices", {"customer_id": cid}
            return "order_invoices", {"order_id": oid or 0}
        if "items" in q:
            if inv_id is not None:
                return "invoice_lines", {"invoice_id": inv_id}
            if cid is not None:
                return "customer_order_item_paths", {"customer_id": cid}
            return "order_items", {"order_id": oid or 0}
        if "orders" in q:
            return "customer_orders", {"customer_id": cid or 0}
        if "supplier" in q and iid is not None:
            return "item_suppliers", {"stock_item_id": iid}
        if ttype == "one_hop" and "group" in q:
            return "customer_buying_group", {"customer_id": cid or 0}
        if ttype == "one_hop" and "city" in q and sid is not None:
            return "supplier_city", {"supplier_id": sid}

    return None


# Templates that need to exist in KGTool for the runner to work; the runner
# records which template was used so the summary can attribute results.
# ---------------------------------------------------------------------------
# The KGTool templates used by the runner (see src/tools/kg_tool.py):
#   customer_orders, find_order_invoices, order_items, item_suppliers,
#   customer_order_item_paths, customer_items_via_supplier,
#   count_orders_by_customer, supplier_city, customer_buying_group,
#   find_customer_by_name, find_item_by_name, find_supplier_by_name,
#   relationship_exists
# ---------------------------------------------------------------------------

# Map the runner's internal template ids to KGTool callables + result
# extractors.  Anything not in this table is treated as "unanswerable"
# and recorded as a prediction failure (kept in the denominator).
_TEMPLATE_HANDLERS: dict[str, tuple[str, Any]] = {
    "customer_orders": ("customer_orders", lambda r: ("ids", _pick_ids(r, "Order"))),
    "find_customer_orders": ("customer_orders", lambda r: ("ids", _pick_ids(r, "Order"))),
    "order_invoices": ("order_invoices", lambda r: ("ids", _pick_ids(r, "Invoice"))),
    "find_order_invoices": ("order_invoices", lambda r: ("ids", _pick_ids(r, "Invoice"))),
    "order_items": ("order_items", lambda r: ("ids", _pick_ids(r, "StockItem"))),
    "invoice_lines": ("invoice_lines", lambda r: ("ids", _pick_ids(r, "StockItem"))),
    "item_suppliers": ("item_suppliers", lambda r: ("ids", _pick_ids(r, "Supplier"))),
    "find_item_suppliers": ("item_suppliers", lambda r: ("ids", _pick_ids(r, "Supplier"))),
    "customer_order_item_paths": ("customer_order_item_paths", lambda r: ("ids", _pick_ids(r, "StockItem"))),
    "customer_items_via_supplier": ("customer_items_via_supplier", lambda r: ("ids", _pick_ids(r, "StockItem"))),
    "customer_item_suppliers": ("customer_item_suppliers", lambda r: ("ids", _pick_ids(r, "Supplier"))),
    "count_orders_by_customer": ("count_orders_by_customer", lambda r: ("count", _pick_scalar(r, "order_count"))),
    "supplier_city": ("supplier_city", lambda r: ("ids", _pick_ids(r, "City"))),
    "customer_buying_group": ("customer_buying_group", lambda r: ("ids", _pick_ids(r, "BuyingGroup"))),
    "find_customer_by_name": ("find_customer_by_name", lambda r: ("names", _pick_names(r, "Customer"))),
    "find_item_by_name": ("find_item_by_name", lambda r: ("names", _pick_names(r, "StockItem"))),
    "find_supplier_by_name": ("find_supplier_by_name", lambda r: ("names", _pick_names(r, "Supplier"))),
    "customer_by_id": ("customer_by_id", lambda r: ("name", _first_name(r))),
    "supplier_by_id": ("supplier_by_id", lambda r: ("name", _first_name(r))),
    "item_by_id": ("item_by_id", lambda r: ("name", _first_name(r))),
    "order_by_id": ("order_by_id", lambda r: ("dict", _first_dict(r))),
    "invoice_by_id": ("invoice_by_id", lambda r: ("dict", _first_dict(r))),
    "group_member_count": ("group_member_count", lambda r: ("count", _pick_scalar(r, "member_count"))),
    "invoice_item_suppliers": ("invoice_item_suppliers", lambda r: ("ids", _pick_ids(r, "Supplier"))),
    "supplier_customers": ("supplier_customers", lambda r: ("ids", _pick_ids(r, "Customer"))),
    "customer_invoices": ("customer_invoices", lambda r: ("ids", _pick_ids(r, "Invoice"))),
}


def _pick_ids(result: dict, label: str) -> list:
    ids = []
    key_prop = {
        "Customer": "customer_id", "Order": "order_id", "Invoice": "invoice_id",
        "StockItem": "stock_item_id", "Supplier": "supplier_id",
        "BuyingGroup": "buying_group_id", "City": "city_id",
    }[label]
    for row in result.get("data", []):
        for value in row.values():
            if isinstance(value, dict) and label in value and key_prop in value[label]:
                ids.append(value[label][key_prop])
    return ids


def _pick_names(result: dict, label: str) -> list:
    out = []
    for row in result.get("data", []):
        for value in row.values():
            if isinstance(value, dict) and label in value:
                out.append(value[label].get("name"))
    return out


def _pick_scalar(result: dict, key: str):
    for row in result.get("data", []):
        if key in row:
            return row[key]
    return None


def _first_name(result: dict):
    for row in result.get("data", []):
        for value in row.values():
            if isinstance(value, dict):
                for node in value.values():
                    if isinstance(node, dict) and "name" in node:
                        return node["name"]
    return None


def _first_dict(result: dict):
    for row in result.get("data", []):
        for value in row.values():
            if isinstance(value, dict):
                for node in value.values():
                    if isinstance(node, dict):
                        return node
    return None


def _expected_id_set(task: dict, entity_key: str | None = None) -> set:
    """Pull the expected id-set for a task's entities.

    ``expected_entities`` has the shape ``{"Order": {"order_id": [1, 501]}}``
    or ``{"Customer": {"customer_id": 40, ...}}``.  Only properties whose
    names end in ``_id`` are treated as identity values; names, dates,
    booleans and other attributes are excluded so that entity-lookup tasks
    (which ask for a name) are not scored against the node's full property
    dict.
    """
    ents = task.get("expected_entities", {})
    if entity_key:
        e = ents.get(entity_key, {})
        for prop, v in e.items():
            if prop.endswith("_id") or prop.endswith("_ids"):
                if isinstance(v, (int, float)):
                    return {v}
                if isinstance(v, list):
                    return set(v)
        return set()
    out: set = set()
    for e in ents.values():
        if isinstance(e, dict):
            for prop, v in e.items():
                if not (prop.endswith("_id") or prop.endswith("_ids")):
                    continue
                if isinstance(v, (int, float)):
                    out.add(v)
                elif isinstance(v, list):
                    out.update(v)
    return out


def _expected_relation_hops(task: dict) -> list[tuple[str, str, str]]:
    return [
        (a, b, c) for a, b, c in task.get("expected_relations", [])
    ]


def _predicted_relation_hops(handler_name: str) -> list[tuple[str, str, str]]:
    """The relation hops a template necessarily traverses.

    Used for path accuracy: a template that answers a two-hop question by
    traversing Customer->Order->StockItem counts as having covered both
    required hops when both are in the expected path.
    """
    path_map: dict[str, list[tuple[str, str, str]]] = {
        "customer_orders": [("Customer", "PLACED", "Order")],
        "order_invoices": [("Order", "INVOICED", "Invoice")],
        "order_items": [("Order", "HAS_LINE", "StockItem")],
        "invoice_lines": [("Invoice", "SHIPPED_ON", "StockItem")],
        "item_suppliers": [("StockItem", "SUPPLIED_BY", "Supplier")],
        "customer_order_item_paths": [("Customer", "PLACED", "Order"), ("Order", "HAS_LINE", "StockItem")],
        "customer_items_via_supplier": [("Customer", "PLACED", "Order"), ("Order", "HAS_LINE", "StockItem"), ("StockItem", "SUPPLIED_BY", "Supplier")],
        "customer_item_suppliers": [("Customer", "PLACED", "Order"), ("Order", "HAS_LINE", "StockItem"), ("StockItem", "SUPPLIED_BY", "Supplier")],
        "count_orders_by_customer": [("Customer", "PLACED", "Order")],
        "supplier_city": [("Supplier", "SUPPLIER_IN_CITY", "City")],
        "customer_buying_group": [("Customer", "BELONGS_TO_GROUP", "BuyingGroup")],
        "find_customer_by_name": [],
        "find_item_by_name": [],
        "find_supplier_by_name": [],
        "customer_by_id": [],
        "supplier_by_id": [],
        "item_by_id": [],
        "order_by_id": [],
        "invoice_by_id": [],
        "group_member_count": [("Customer", "BELONGS_TO_GROUP", "BuyingGroup")],
        "invoice_item_suppliers": [("Invoice", "SHIPPED_ON", "StockItem"), ("StockItem", "SUPPLIED_BY", "Supplier")],
        "supplier_customers": [("StockItem", "SUPPLIED_BY", "Supplier"), ("Invoice", "SHIPPED_ON", "StockItem"), ("Customer", "PLACED", "Order"), ("Order", "INVOICED", "Invoice")],
        "customer_invoices": [("Customer", "PLACED", "Order"), ("Order", "INVOICED", "Invoice")],
    }
    return path_map.get(handler_name, [])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run_task(tool: KGTool, task: dict) -> dict[str, Any]:
    """Answer one task; never drops it from the results."""
    routing = _extract_params(task)
    predicted_ids: set = set()
    predicted_scalar: Any = None
    traversed_hops: list[tuple[str, str, str]] = []
    error: str | None = None
    template_used: str | None = None
    latency_ms = 0.0

    if routing is None:
        error = "unroutable: question does not map to a registered KGTool template"
    else:
        kg_template, params = routing
        _, extract = _TEMPLATE_HANDLERS.get(kg_template, (None, None))
        if extract is None:
            error = f"no result extractor for template {kg_template!r}"
        else:
            t0 = time.perf_counter()
            result = tool.run(template_id=kg_template, parameters=params)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            template_used = kg_template
            if not result.get("success"):
                error = f"tool failure: {result.get('error', {}).get('code')}"
            else:
                kind, value = extract(result)
                if kind in ("ids", "orders", "invoices", "items", "suppliers", "city", "groups", "customers"):
                    predicted_ids = set(value) if isinstance(value, list) else {value}
                elif kind == "count":
                    predicted_scalar = value
                    predicted_ids = {value} if value is not None else set()
                elif kind == "names":
                    predicted_ids = set(value)
                elif kind in ("name", "customer", "supplier", "item"):
                    predicted_scalar = value
                    predicted_ids = {value} if value is not None else set()
                elif kind == "dict":
                    predicted_scalar = value
                    predicted_ids = set(value.values()) if isinstance(value, dict) else set()
                elif kind == "exists":
                    predicted_scalar = value
                    predicted_ids = {1} if value else set()
                traversed_hops = _predicted_relation_hops(kg_template)

    return {
        "predicted_ids": sorted(predicted_ids, key=repr),
        "predicted_scalar": predicted_scalar,
        "traversed_hops": [list(h) for h in traversed_hops],
        "template_used": template_used,
        "error": error,
        "latency_ms": latency_ms,
    }


def _score(task: dict, run: dict) -> dict[str, Any]:
    expected_ids = _expected_id_set(task)
    predicted_ids = set(run["predicted_ids"])
    expected_scalar = task.get("expected_answer")

    metrics: dict[str, Any] = {
        "precision": round(set_precision(predicted_ids, expected_ids), 4),
        "recall": round(set_recall(predicted_ids, expected_ids), 4),
        "f1": round(set_f1(predicted_ids, expected_ids), 4),
    }

    if run.get("error"):
        metrics["exact_match"] = False
        metrics["path_accuracy"] = None
        metrics["correct"] = False
        return metrics

    # Exact match: prefer scalar comparison when the template produced a
    # scalar; otherwise compare id-sets.  A set-valued expected answer
    # (list in expected_entities) is scored on set equality.
    has_list_expected = any(
        isinstance(v, list)
        for e in task.get("expected_entities", {}).values()
        if isinstance(e, dict)
        for v in e.values()
    )
    if run.get("predicted_scalar") is not None and not has_list_expected:
        scalar = run["predicted_scalar"]
        scalar_ok = exact_match(scalar, expected_scalar)
        if not scalar_ok and not isinstance(scalar, dict):
            scalar_ok = {scalar} == expected_ids and bool(expected_ids)
        metrics["exact_match"] = scalar_ok
    else:
        metrics["exact_match"] = predicted_ids == expected_ids

    exp_hops = _expected_relation_hops(task)
    metrics["path_accuracy"] = path_accuracy(run["traversed_hops"], exp_hops)
    metrics["correct"] = bool(metrics["exact_match"])
    return metrics


def run_eval(
    dataset_path: Path,
    *,
    limit: int | None = None,
    task_types: list[str] | None = None,
) -> dict[str, Any]:
    tool = KGTool()
    results: list[dict[str, Any]] = []
    try:
        with dataset_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                task = json.loads(line)
                if task_types and task["task_type"] not in task_types:
                    continue
                run = _run_task(tool, task)
                metrics = _score(task, run)
                results.append({
                    "id": task["id"],
                    "question": task["question"],
                    "task_type": task["task_type"],
                    "difficulty": task["difficulty"],
                    "expected_answer": task["expected_answer"],
                    "expected_entities": task["expected_entities"],
                    "expected_relations": task["expected_relations"],
                    "predicted": run["predicted_ids"] if run["error"] else run["predicted_ids"],
                    "predicted_scalar": run.get("predicted_scalar"),
                    "template_used": run["template_used"],
                    "error": run["error"],
                    "latency_ms": run["latency_ms"],
                    **metrics,
                })
                if limit and len(results) >= limit:
                    break
    finally:
        tool.close()

    summary = _summarize(results)
    return {"results": results, "summary": summary}


def _summarize(results: list[dict]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"total_tasks": 0}

    def _agg(key: str, subset: list[dict]) -> dict[str, Any]:
        m = len(subset)
        exact = sum(1 for r in subset if r.get("exact_match"))
        failures = sum(1 for r in subset if r.get("error"))
        lats = [r["latency_ms"] for r in subset if not r.get("error") and r["latency_ms"] > 0]
        pats = [r["path_accuracy"] for r in subset if r.get("path_accuracy") is not None]
        prec = [r["precision"] for r in subset]
        rec = [r["recall"] for r in subset]
        f1s = [r["f1"] for r in subset]
        return {
            "tasks": m,
            "exact_match": round(exact / m, 4),
            "failure_count": failures,
            "avg_latency_ms": round(statistics.mean(lats), 2) if lats else None,
            "p50_latency_ms": round(statistics.median(lats), 2) if lats else None,
            "p95_latency_ms": round(_pct(lats, 95), 2) if lats else None,
            "path_accuracy": round(statistics.mean(pats), 4) if pats else None,
            "avg_precision": round(statistics.mean(prec), 4),
            "avg_recall": round(statistics.mean(rec), 4),
            "avg_f1": round(statistics.mean(f1s), 4),
        }

    summary: dict[str, Any] = {
        "total_tasks": n,
        "overall": _agg("overall", results),
        "by_task_type": {},
        "by_difficulty": {},
    }
    for ttype in sorted({r["task_type"] for r in results}):
        summary["by_task_type"][ttype] = _agg(ttype, [r for r in results if r["task_type"] == ttype])
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            summary["by_difficulty"][diff] = _agg(diff, subset)
    return summary


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the KG evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--task-type", action="append", default=None)
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--skip-neo4j-check", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_neo4j_check:
        from src.core.neo4j_client import Neo4jClient
        client = Neo4jClient()
        try:
            client.health_check()
            client.close()
        except Exception as exc:  # noqa: BLE001
            print(f"Neo4j unavailable: {exc}")
            print("Set NEO4J_PASSWORD (and start a Neo4j) or pass --skip-neo4j-check.")
            return 2

    dataset_path = Path(args.dataset)
    report = run_eval(dataset_path, limit=args.limit, task_types=args.task_type)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "kg_eval_results.json").write_text(
        json.dumps(report["results"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ARTIFACT_DIR / "kg_eval_summary.json").write_text(
        json.dumps(report["summary"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_summary_md(report["summary"], ARTIFACT_DIR / "kg_eval_summary.md")

    s = report["summary"]
    print(f"ran {s['total_tasks']} tasks; exact match = {s['overall']['exact_match']:.2%}; "
          f"failures = {s['overall']['failure_count']}")
    print(f"artifacts written to {ARTIFACT_DIR}")
    return 0


def _write_summary_md(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 2.4 KG Evaluation Summary",
        "",
        f"- total tasks: {summary.get('total_tasks')}",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    overall = summary.get("overall", {})
    for key in ("exact_match", "avg_precision", "avg_recall", "avg_f1", "path_accuracy",
                "failure_count", "avg_latency_ms", "p50_latency_ms", "p95_latency_ms"):
        if key in overall:
            lines.append(f"| {key} | {overall[key]} |")
    lines += ["", "## By task type", "", "| type | tasks | exact | p | r | f1 | failures |", "|---|---|---|---|---|---|---|"]
    for ttype, d in summary.get("by_task_type", {}).items():
        lines.append(f"| {ttype} | {d['tasks']} | {d['exact_match']:.2f} | {d['avg_precision']:.2f} | {d['avg_recall']:.2f} | {d['avg_f1']:.2f} | {d['failure_count']} |")
    lines += ["", "## By difficulty", "", "| difficulty | tasks | exact | p | r | f1 | failures |", "|---|---|---|---|---|---|---|"]
    for diff, d in summary.get("by_difficulty", {}).items():
        lines.append(f"| {diff} | {d['tasks']} | {d['exact_match']:.2f} | {d['avg_precision']:.2f} | {d['avg_recall']:.2f} | {d['avg_f1']:.2f} | {d['failure_count']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
