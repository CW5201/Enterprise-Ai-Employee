"""Validate the unified Enterprise Task Benchmark (Phase 5 Commit 1).

Independent structural checks over ``data/eval/enterprise_tasks.jsonl``.
This script is a *guard* — it never rewrites the dataset, it only asserts
that the committed benchmark is internally consistent and that no ground
truth has leaked a system-output value.

Checks performed:

- task id unique
- expected tools legal (subset of the tool vocabulary)
- expected route legal (subset of the route vocabulary)
- expected evidence non-empty (unless the task is a clarification)
- expected answer non-empty
- difficulty legal (easy/medium/hard)
- task_type legal
- no duplicate question
- no accidental leakage:
  * GT evidence sources are a *subset* of the tools the task declares
    (a claim of KG evidence on a pure-SQL task is a leakage bug);
  * clarification tasks carry no tools / no evidence;
  * multi_tool tasks declare >= 2 distinct tools;
  * single-tool routes declare exactly one tool;
  * tool order, when asserted, is a permutation of the declared tools.

Exit code 0 = all checks passed; non-zero = a check failed (output lists
every offending task id).

Usage::

    python scripts/validate_enterprise_tasks.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = _PROJECT_ROOT / "data" / "eval" / "enterprise_tasks.jsonl"

_ROUTE_VOCAB = {"rag", "sql", "kg", "analysis", "multi_tool", "clarification"}
_TOOL_VOCAB = {"rag", "sql", "kg", "analysis", "chart", "report"}
_TASK_TYPES = {
    "knowledge_lookup", "structured_lookup", "aggregation", "relationship_query",
    "statistical_analysis", "trend_analysis", "comparison", "report_generation",
    "multi_source_analysis", "ambiguous_task",
}
_DIFFICULTIES = {"easy", "medium", "hard"}
_SOURCE_TOOLS = {"rag", "sql", "kg", "analysis"}


def load(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def validate(tasks: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_questions: dict[str, list[str]] = {}

    for tk in tasks:
        tid = tk.get("id", "<missing-id>")

        # --- schema presence ------------------------------------------------
        required = ("id", "question", "domain", "task_type", "difficulty",
                    "expected_route", "expected_tools", "expected_tool_order",
                    "expected_answer", "expected_evidence", "expected_claims",
                    "provenance")
        for field in required:
            if field not in tk:
                errors.append(f"{tid}: missing field {field!r}")

        # --- id uniqueness --------------------------------------------------
        if tid in seen_ids:
            errors.append(f"{tid}: duplicate task id")
        seen_ids.add(tid)

        # --- question dedup -------------------------------------------------
        q = str(tk.get("question", ""))
        seen_questions.setdefault(q, []).append(tid)

        # --- vocabularies ---------------------------------------------------
        if tk.get("task_type") not in _TASK_TYPES:
            errors.append(f"{tid}: illegal task_type {tk.get('task_type')!r}")
        if tk.get("difficulty") not in _DIFFICULTIES:
            errors.append(f"{tid}: illegal difficulty {tk.get('difficulty')!r}")
        route = tk.get("expected_route")
        if route not in _ROUTE_VOCAB:
            errors.append(f"{tid}: illegal expected_route {route!r}")
        tools = list(tk.get("expected_tools") or [])
        for tool in tools:
            if tool not in _TOOL_VOCAB:
                errors.append(f"{tid}: illegal tool {tool!r}")

        # --- answer / evidence / claims ------------------------------------
        if not str(tk.get("expected_answer", "")).strip():
            errors.append(f"{tid}: empty expected_answer")
        evidence = list(tk.get("expected_evidence") or [])
        if not tk.get("expected_clarification") and not evidence:
            errors.append(f"{tid}: non-clarification task has no expected_evidence")

        # --- leakage / consistency guards -----------------------------------
        # clarification: no tools, no evidence
        if route == "clarification":
            if tools:
                errors.append(f"{tid}: clarification task must declare no tools")
            if evidence:
                errors.append(f"{tid}: clarification task must declare no evidence")
        # single-tool route: exactly one tool
        elif route in ("rag", "sql", "kg", "analysis"):
            if len(tools) != 1:
                errors.append(f"{tid}: single-tool route {route!r} declares {len(tools)} tools")
        # multi_tool: >= 2 distinct source tools
        elif route == "multi_tool":
            distinct_sources = set(tools) & _SOURCE_TOOLS
            if len(distinct_sources) < 2:
                errors.append(f"{tid}: multi_tool route declares < 2 distinct source tools {tools}")

        # evidence must be a subset of declared tools (no phantom sources)
        declared = set(tools)
        for ev in evidence:
            if ev not in declared:
                errors.append(f"{tid}: evidence source {ev!r} not among declared tools {sorted(declared)}")

        # tool order is a permutation of the declared tools when asserted
        order = list(tk.get("expected_tool_order") or [])
        if order and sorted(order) != sorted(tools):
            errors.append(f"{tid}: expected_tool_order {order} is not a permutation of tools {tools}")

        # expected claims must be well-formed
        for claim in tk.get("expected_claims") or []:
            if "text" not in claim or not str(claim.get("text", "")).strip():
                errors.append(f"{tid}: malformed expected claim {claim!r}")

    # --- cross-task duplicate questions -------------------------------------
    dup = {q: ids for q, ids in seen_questions.items() if len(ids) > 1}
    for q, ids in dup.items():
        errors.append(f"duplicate question in {ids}: {q!r}")

    return errors


def _report(tasks: list[dict]) -> None:
    c_type = Counter(tk["task_type"] for tk in tasks)
    c_diff = Counter(tk["difficulty"] for tk in tasks)
    c_route = Counter(tk["expected_route"] for tk in tasks)
    c_tools = Counter(len(tk.get("expected_tools") or []) for tk in tasks)
    verifiers = Counter((tk.get("provenance") or {}).get("verifier", "?") for tk in tasks)
    print(f"total tasks: {len(tasks)}")
    print(f"by_task_type:  {dict(c_type)}")
    print(f"by_difficulty: {dict(c_diff)}")
    print(f"by_route:      {dict(c_route)}")
    print(f"by_num_tools:  {dict(sorted(c_tools.items()))}")
    print(f"gt_verifiers:  {dict(verifiers)}")


def main() -> int:
    if not DATASET.exists():
        print(f"ERROR: dataset not found at {DATASET} — run scripts/build_enterprise_tasks.py first")
        return 2
    tasks = load(DATASET)
    errors = validate(tasks)
    _report(tasks)
    if errors:
        print(f"\nVALIDATION FAILED — {len(errors)} problem(s):")
        for e in errors:
            print("  -", e)
        return 1
    print("\nVALIDATION PASSED — benchmark is internally consistent (id unique, GT clean, no leakage).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
