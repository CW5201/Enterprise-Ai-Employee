"""Phase 5 — End-to-End Enterprise Evaluation Runner.

Runs a configured *system* (or baseline / ablation) over the unified
Enterprise Task Benchmark and writes three gitignored artifacts to
``artifacts/phase5/``:

- ``enterprise_eval_results.json`` — one record per task × per system
  (expected / predicted / route / tools / answer / evidence / verification /
  latency / errors / failure categories).  Failures stay in the denominator.
- ``enterprise_eval_summary.json`` — per-system and per-task-type /
  per-difficulty aggregations (all ten spec metrics + latency percentiles +
  LLM-call counts + error categories).
- ``enterprise_eval_summary.md`` — the human-readable table(s) for the report.

Design rules (docs/RESEARCH.md, ADR discipline):

- **Real components.**  Every graph-based system drives the real
  :func:`src.graph.builder.build_graph` pipeline (router + tools +
  verification).  Tool-direct baselines (B1/B2/B3) call the real
  :mod:`src.tools` implementations.  No fake scores, no mock answers.
- **Failures are recorded, never skipped.**  A routing / tool / verification /
  LLM-429 / timeout failure produces a 0-scored record *and* a failure
  category; it is not dropped from any denominator.
- **Offline-deterministic when no live service is present.**  The runner is
  *honest about its mode*: it records ``live_services`` (which of Milvus /
  Neo4j / LLM were reachable) and a ``deterministic_mode`` flag in every
  artifact.  A fully-offline run is an **offline harness** result; an
  online run additionally exercises the real RAG/KG/LLM chain.  The two are
  labelled distinctly, never conflated.
- **No GT is read from system output.**  Ground truth comes only from
  ``data/eval/enterprise_tasks.jsonl`` (Commit 1).

Usage::

    # run a single system over the whole benchmark
    python scripts/run_enterprise_eval.py --system A_full
    # run several systems (repeatable)
    python scripts/run_enterprise_eval.py --system A_full --system B_no_verification
    # a cap for a quick smoke run
    python scripts/run_enterprise_eval.py --system A_full --limit 20
    # offline-deterministic mode (no live LLM/Neo4j) — harness only
    python scripts/run_enterprise_eval.py --system A_full --offline

``--offline`` sets ``LLM_FORCE_OFFLINE=1`` and disables live Neo4j so the run
is reproducible without a network; it is clearly marked in the artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.llm_client import LLMClient, _load_dotenv  # noqa: E402
from src.evaluation import enterprise_metrics as em  # noqa: E402
from src.evaluation.enterprise_systems import SystemConfig, all_systems, get_system  # noqa: E402

DATASET = _PROJECT_ROOT / "data" / "eval" / "enterprise_tasks.jsonl"
ARTIFACT_DIR = _PROJECT_ROOT / "artifacts" / "phase5"


# ---------------------------------------------------------------------------
# Service-availability probe (honest mode reporting)
# ---------------------------------------------------------------------------


def probe_services(force_offline: bool) -> dict[str, Any]:
    """Return which live services are reachable, so artifacts can state the
    run's mode (offline harness vs. online end-to-end)."""
    info: dict[str, Any] = {
        "force_offline": force_offline,
        "llm": False,
        "milvus": False,
        "neo4j": False,
        "duckdb": True,
        "deterministic_mode": force_offline,
    }
    if force_offline:
        os.environ["LLM_FORCE_OFFLINE"] = "1"
        os.environ.pop("NEO4J_INTEGRATION", None)
        return info

    _load_dotenv()
    # LLM: a real, non-offline client means a live endpoint is configured.
    try:
        info["llm"] = not LLMClient().offline
    except Exception:  # noqa: BLE001
        info["llm"] = False
    # Milvus: cheap socket probe (no model load).
    try:
        import socket

        host = os.environ.get("MILVUS_HOST", "localhost")
        port = int(os.environ.get("MILVUS_PORT", "19530"))
        s = socket.socket()
        s.settimeout(2)
        s.connect((host, port))
        s.close()
        info["milvus"] = True
    except Exception:  # noqa: BLE001
        info["milvus"] = False
    # Neo4j: probe only when integration creds are expected.
    try:
        from src.core.neo4j_client import Neo4jClient

        c = Neo4jClient()
        c.health_check()
        c.close()
        info["neo4j"] = True
    except Exception:  # noqa: BLE001
        info["neo4j"] = False
    return info


# ---------------------------------------------------------------------------
# Prediction builders — one per system kind.  Each returns the ``pred`` dict
# that :func:`em.score_task` consumes, with honest failure categories.
# ---------------------------------------------------------------------------


def _fail(category: str, message: str) -> dict[str, Any]:
    return {
        "route": "", "tools": [], "order": [], "answer": "",
        "evidence": [], "verification": None, "guard": None,
        "any_error": True, "failure_categories": [category],
        "errors": [message], "llm_calls": 0,
        "latency_ms": {"total": 0.0},
    }


def _run_graph_system(
    config: SystemConfig, task: dict[str, Any], runner: Any,
) -> dict[str, Any]:
    """Drive one graph-based system end to end on a task.

    ``runner`` is a callable ``(query) -> state`` built from
    :func:`build_graph` for *config*.  Failures are caught and recorded —
    they never abort the benchmark.
    """
    t0 = time.perf_counter()
    try:
        state = runner(task["question"])
    except Exception as exc:  # noqa: BLE001 — record, keep in denominator
        return _fail("pipeline_error", f"pipeline raised: {exc}")

    t_total = time.perf_counter() - t0
    decision = state.get("routing_decision") or {}
    route = str(state.get("route") or decision.get("route_type") or "")
    tools = list(decision.get("tools") or state.get("tools_selected") or [])
    plan = decision.get("execution_plan") or []
    order = [s.get("tool") for s in sorted(plan, key=lambda x: x.get("step", 0))] if plan else list(tools)

    # evidence source types actually produced by the tools that ran
    evidence_sources = _evidence_sources(state)

    # verification payload (None when the system has no verification layer)
    verification = None
    guard = None
    if config.enable_verification:
        results = state.get("verification_results") or []
        claims = state.get("claims") or []
        if results:
            verification = {
                "claims": [
                    {
                        "claim_id": r.get("claim_id"),
                        "supported": r.get("supported"),
                        "status": r.get("status"),
                        "verifier_type": r.get("verifier_type"),
                    }
                    for r in results
                ],
                "n_claims": len(claims) or len(results),
                "n_critical_unsupported": int(_count_critical_unsupported(results, claims)),
            }
            guard = state.get("guard_decision") or {}

    # LLM calls in graph systems: intent classification + routing profile +
    # text-to-sql + answer generation + claim extraction (+ verifier layer 3)
    llm_calls = _count_graph_llm_calls(state, config)

    categories: list[str] = []
    for e in state.get("errors") or []:
        msg = str(e.get("message", "")).lower()
        stage = str(e.get("stage", ""))
        if "429" in msg or "rate" in msg:
            categories.append("llm_429")
        elif "timeout" in msg or "timed out" in msg:
            categories.append("timeout")
        elif stage in ("routing",) and "rejected" in msg:
            categories.append("routing_failure")
        elif stage == "sql_execution":
            categories.append("sql_error")
        elif "kg" in msg or "neo4j" in msg:
            categories.append("kg_coverage")
        elif "verification" in msg or "claim" in msg:
            categories.append("verification_failure")
        elif "retriev" in msg:
            categories.append("rag_retrieval_miss")
    # dedupe, keep order
    categories = list(dict.fromkeys(c for c in categories if c))

    pred = {
        "route": route,
        "tools": tools,
        "order": order,
        "answer": str(state.get("answer", "")),
        "evidence": evidence_sources,
        "verification": verification,
        "guard": guard,
        "any_error": bool(state.get("errors")) or route == "" ,
        "failure_categories": categories,
        "errors": [f"{e.get('stage')}: {e.get('message')}" for e in (state.get("errors") or [])][:5],
        "llm_calls": llm_calls,
        "latency_ms": _stage_latencies(state, t_total),
    }
    return pred


def _evidence_sources(state: dict[str, Any]) -> list[str]:
    """Which source types actually produced evidence this run."""
    srcs: list[str] = []
    for tr in state.get("tool_results") or []:
        if tr.get("success") and tr.get("tool") in ("rag", "sql", "kg", "analysis"):
            srcs.append(tr["tool"])
    # legacy channels
    if state.get("sql_result") is not None and "sql" not in srcs:
        srcs.append("sql")
    if (state.get("retrieved_context") or state.get("evidence")) and "rag" not in srcs:
        srcs.append("rag")
    return srcs


def _count_graph_llm_calls(state: dict[str, Any], config: SystemConfig) -> int:
    """Best-effort count of LLM calls this system made for the task.

    Graph systems call the LLM at: intent understanding, routing profile
    (dynamic router only), text-to-sql, answer generation, claim extraction,
    and the semantic verification layer (when verification is on and the
    claim set is non-empty).  We infer the count from the pipeline stages
    that actually ran, which is stable and inspectable.
    """
    n = 1  # intent understanding
    decision = state.get("routing_decision") or {}
    if decision and config.router == "dynamic":
        n += 1  # LLM task-profile proposal
    if state.get("sql") or any(getattr(tc, "tool", "") == "sql_tool" for tc in state.get("tool_calls") or []):
        n += 1  # text-to-sql
    if state.get("answer"):
        n += 1  # answer generation
    if config.enable_verification and state.get("claims"):
        n += 1  # claim extraction
        if any(r.get("verifier_type") == "semantic" for r in state.get("verification_results") or []):
            n += 1  # semantic verification layer
    return n


def _count_critical_unsupported(results: list[dict], claims: list[dict]) -> int:
    claim_imp = {c.get("claim_id"): c.get("importance") for c in claims}
    n = 0
    for r in results:
        if not r.get("supported") and not r.get("conflict") and                 claim_imp.get(r.get("claim_id")) == "critical":
            n += 1
    return n


def _stage_latencies(state: dict[str, Any], t_total: float) -> dict[str, float]:
    lat = state.get("latency") or {}
    out = {k: round(float(v), 2) for k, v in lat.items()}
    out["routing"] = out.get("routing", 0.0)
    out["total"] = round(t_total * 1000.0, 2)
    # derive a tool phase if the pipeline didn't tag it
    if "tool_execution" not in out and "execute" in out:
        out["tool_execution"] = out.get("execute", 0.0)
    if "verification" not in out and "guard" not in out:
        out.setdefault("verification", 0.0)
    return out


# --- tool-direct baselines (B1 / B2 / B3) ---------------------------------


def _run_static_tool(config: SystemConfig, task: dict[str, Any]) -> dict[str, Any]:
    """Baseline: always run one fixed tool and answer from its result."""
    t0 = time.perf_counter()
    from src.core.state import make_state
    from src.nodes.answer_generation import AnswerGenerationNode
    from src.nodes.sql_execution import SQLExecutionNode
    from src.tools.rag_tool import RAGTool

    state = make_state(task["question"])
    llm_calls = 0
    categories: list[str] = []
    evidence_sources: list[str] = []

    tool = _baseline_tool(config)
    try:
        if tool == "rag":
            rtool = RAGTool(retrieval_mode=config.retrieval_mode)
            res = rtool.run(query=task["question"])
            ok = bool(res.get("ok"))
            hits = res.get("results", [])
            _populate_rag_state(state, hits)
            if ok:
                evidence_sources.append("rag")
            else:
                categories.append("rag_retrieval_miss")
        elif tool == "sql":
            sql_node = SQLExecutionNode()
            out = dict(sql_node.run(state))
            _merge_sql_state(state, out)
            if out.get("sql_result") is not None:
                evidence_sources.append("sql")
            else:
                categories.append("sql_error")
        state["route"] = tool
        state["tools_selected"] = [tool]
        llm_calls += 1
        # generate an answer from the gathered evidence (real LLM or offline)
        ans_node = AnswerGenerationNode()
        out = dict(ans_node.run(state))
        state["answer"] = out.get("answer", state.get("answer", ""))
        state["errors"] = list(state.get("errors") or []) + out.get("errors", [])
    except Exception as exc:  # noqa: BLE001
        categories.append("pipeline_error")
        state["answer"] = ""
        state["errors"] = [f"baseline tool error: {exc}"]

    t_total = time.perf_counter() - t0
    pred = {
        "route": tool,
        "tools": [tool],
        "order": [tool],
        "answer": str(state.get("answer", "")),
        "evidence": evidence_sources,
        "verification": None,
        "guard": None,
        "any_error": bool(categories),
        "failure_categories": categories,
        "errors": [str(e) for e in (state.get("errors") or [])][:5],
        "llm_calls": llm_calls,
        "latency_ms": {"routing": 0.0, "tool_execution": round((t_total) * 1000, 2),
                       "verification": 0.0, "total": round(t_total * 1000, 2)},
    }
    return pred


def _baseline_tool(config: SystemConfig) -> str:
    if config.name == "B2_static_rag":
        return "rag"
    return "sql"  # B3_static_sql


def _populate_rag_state(state: dict[str, Any], hits: list[dict]) -> None:
    from src.core.state import RetrievedChunk

    chunks = []
    for h in hits:
        chunks.append(RetrievedChunk(
            chunk_id=str(h.get("chunk_id", "")),
            text=str(h.get("text", "")),
            source=str(h.get("source", "")),
            metadata=h.get("metadata") or {},
            score=float(h.get("score", 0.0) or 0.0),
        ))
    state["retrieved_context"] = chunks
    state["evidence"] = [
        dict(item.__class__ for item in [])  # placeholder, filled by answer node
    ] if False else state.get("evidence", [])


def _merge_sql_state(state: dict[str, Any], out: dict[str, Any]) -> None:
    for key in ("sql", "sql_result", "tool_calls", "evidence", "errors"):
        if key in out:
            if key in ("tool_calls", "evidence", "errors"):
                state[key] = list(state.get(key) or []) + list(out.get(key) or [])
            else:
                state[key] = out[key]


def _run_llm_only(config: SystemConfig, task: dict[str, Any]) -> dict[str, Any]:
    """B1: a single grounded LLM call with NO tool evidence."""
    t0 = time.perf_counter()
    llm = LLMClient()
    answer = ""
    categories: list[str] = []
    try:
        system = (
            "你是企业 AI 数字员工。只依据你掌握的信息作答，若不确定请说明。"
            "用简洁中文作答。"
        )
        answer = llm.generate(system, f"用户任务: {task['question']}")
        answer = _plain_answer(answer)
    except Exception as exc:  # noqa: BLE001
        categories.append("llm_error")
        answer = ""
        err_msg = f"llm_error: {exc}"
    else:
        err_msg = ""
    t_total = time.perf_counter() - t0
    return {
        "route": "multi_tool",  # LLM-only is not a routing route; mark as best-effort
        "tools": [],
        "order": [],
        "answer": answer,
        "evidence": [],
        "verification": None,
        "guard": None,
        "any_error": bool(categories),
        "failure_categories": categories,
        "errors": [err_msg] if categories else [],
        "llm_calls": 1,
        "latency_ms": {"routing": 0.0, "tool_execution": 0.0,
                       "verification": 0.0, "total": round(t_total * 1000, 2)},
    }


def _plain_answer(raw: str) -> str:
    import json as _json

    s = (raw or "").strip()
    if s.startswith("{"):
        try:
            d = _json.loads(s)
            if isinstance(d, dict) and d.get("text"):
                return str(d["text"])
        except _json.JSONDecodeError:
            pass
    return s


# ---------------------------------------------------------------------------
# Per-system dispatcher
# ---------------------------------------------------------------------------


def _make_graph_runner(config: SystemConfig, force_offline: bool) -> Any:
    """Build the graph pipeline for a config and return ``query -> state``."""
    backend = "fake" if force_offline else None
    if config.router == "rule":
        # Deterministic keyword profile (no LLM task profile): build the graph
        # with the real dynamic router wired in, but short-circuit the LLM
        # profile with a keyword TaskProfile, then run the real multi-tool
        # orchestrator + answer node on top of it.
        from src.core.routing_rules import RoutingPolicy
        from src.core.routing_rules import decide as _decide_rule
        from src.nodes.answer_generation import AnswerGenerationNode
        from src.nodes.multi_tool_execution import MultiToolExecutionNode
        from src.tools.rag_tool import RAGTool

        policy = RoutingPolicy.load()
        answer_node = AnswerGenerationNode(llm=LLMClient())
        kg_tool = None if config.enable_kg else _DisabledKG()

        def runner(query: str) -> dict[str, Any]:
            from src.core.state import make_state

            state = make_state(query)
            state.update({"intent": "", "confidence": 0.0})
            profile = _keyword_profile(query)
            decision = _decide_rule(profile, policy)
            out = {"routing_decision": decision.model_dump(),
                   "route": decision.route_type,
                   "tools_selected": list(decision.tools)}
            state.update(out)
            # route clarification honestly (no tools), else run the orchestrator
            if decision.route_type == "clarification":
                state["answer"] = "需要澄清：该任务的关键信息不足以确定执行路径。"
                state["status"] = "clarification"
                return state
            executor = MultiToolExecutionNode(
                rag_tool=RAGTool(retrieval_mode=config.retrieval_mode),
                kg_tool=kg_tool,
            )
            ex = dict(executor.run(state))
            state.update(ex)
            an = dict(answer_node.run(state))
            state.update({k: v for k, v in an.items() if k in ("answer", "status", "evidence")})
            return state

        return runner

    # dynamic / supervisor: build the real graph.
    # When enable_kg=False (the w/o-KG ablation / System C), replace the
    # orchestrator's KG tool with an honest _DisabledKG so any kg step is
    # recorded as a skipped/failed step instead of silently falling back.
    kwargs = config.build_kwargs()
    from src.graph.builder import build_graph
    from src.nodes.multi_tool_execution import MultiToolExecutionNode
    from src.nodes.sql_execution import SQLExecutionNode
    from src.tools.rag_tool import RAGTool

    llm = LLMClient()
    rag_tool = RAGTool(backend=backend, retrieval_mode=config.retrieval_mode)
    sql_node = SQLExecutionNode(llm=llm)
    kg_tool = None if config.enable_kg else _DisabledKG()
    executor = MultiToolExecutionNode(
        sql_node=sql_node, rag_tool=rag_tool, kg_tool=kg_tool, llm=llm,
    )
    if kwargs["phase3"]:
        from src.nodes.task_router import TaskRouterNode

        router_node = TaskRouterNode(llm=llm)
        employee_graph = build_graph(
            backend=backend, rag_top_k=5, retrieval_mode=config.retrieval_mode,
            phase3=True, phase4=kwargs["phase4"],
            router_node=router_node, executor_node=executor,
        )
    else:
        from src.nodes.supervisor_router import SupervisorRouterNode

        router_node = SupervisorRouterNode()
        employee_graph = build_graph(
            backend=backend, rag_top_k=5, retrieval_mode=config.retrieval_mode,
            phase3=False, phase4=kwargs["phase4"],
            router_node=router_node, executor_node=executor,
        )

    def runner(query: str) -> dict[str, Any]:
        from src.core.observability import set_task_latencies
        from src.core.state import make_state
        from src.nodes.rag_retrieval import RAGRetrievalNode

        state = make_state(query)
        set_task_latencies(state["latency"])
        out = employee_graph.invoke(state)
        # The phase3 orchestrator's RAG step returns the hit list in the
        # uniform tool_results payload; surface it into retrieved_context /
        # evidence so the answer + verification nodes see RAG evidence even
        # when the dense backend is the fake (offline) one.
        rag_hits = _rag_hits_from_tool_results(out.get("tool_results") or [])
        if rag_hits:
            node = RAGRetrievalNode(tool=rag_tool)
            state["user_query"] = query
            rag_out = dict(node.run(state))
            out["retrieved_context"] = list(out.get("retrieved_context") or []) + list(
                rag_out.get("retrieved_context") or [])
            out["evidence"] = list(out.get("evidence") or []) + list(rag_out.get("evidence") or [])
        return out

    # Expose the compiled graph so the runner can drive it directly.
    runner._graph = employee_graph  # type: ignore[attr-defined]
    return runner


def _rag_hits_from_tool_results(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lift the RAG tool's hit dicts out of the orchestrator's uniform payload.

    The orchestrator's ``_run_rag`` stores the full hit list under ``results``;
    this returns it so the runner can re-feed it through the real
    RAGRetrievalNode (which produces ``retrieved_context`` + evidence).
    Empty when no RAG step ran — the runner then leaves evidence untouched.
    """
    for tr in tool_results:
        if tr.get("tool") == "rag" and tr.get("success"):
            return list(tr.get("results") or [])
    return []


def _keyword_profile(query: str):
    """Deterministic keyword TaskProfile (no LLM) — the B4 rule-routing lane."""
    from src.core.routing_types import TaskProfile

    q = query
    has_struct = any(k in q for k in ("统计", "多少", "数量", "金额", "客户", "订单", "排名", "前", "平均", "合计", "占比"))
    has_unstruct = any(k in q for k in ("政策", "制度", "规定", "标准", "流程", "手册", "要求", "报销", "审批"))
    has_rel = any(k in q for k in ("关系", "路径", "涉及", "哪些供应商", "哪些商品", "下过哪些", "属于", "由哪个"))
    has_stat = any(k in q for k in ("增长率", "趋势", "环比", "同比", "占比", "均值"))
    has_ambig = any(k in q for k in ("最近", "情况", "看看", "分析一下", "重要的事", "那个")) and not (
        has_struct or has_unstruct or has_rel)
    ttype = ("ambiguous_task" if has_ambig else
             "relationship_query" if has_rel else
             "multi_source_analysis" if (has_struct and has_unstruct) else
             "knowledge_lookup" if has_unstruct else
             "aggregation" if has_struct else "ambiguous_task")
    return TaskProfile(
        task_id="rule", task_type=ttype,
        requires_structured_data=has_struct,
        requires_unstructured_knowledge=has_unstruct,
        requires_relationship_reasoning=has_rel,
        requires_statistical_analysis=has_stat,
        requires_multi_source=(has_struct and has_unstruct),
        ambiguity=0.9 if has_ambig else 0.0,
    )


def _finish_graph(state, out, config):
    state.update(out)
    return state


class _DisabledKG:
    """KG tool that honestly refuses to run (the w/o-KG ablation).

    Every call returns a failure envelope so the orchestrator records a
    ``skipped``/failed step instead of silently substituting another source.
    """

    name = "kg"

    def __init__(self, reason: str = "kg disabled for this ablation") -> None:
        self._reason = reason

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "success": False,
            "error": {"code": "disabled", "message": self._reason},
            "source": "neo4j", "data": None, "rows": 0,
        }

    def close(self) -> None:
        return None



# ---------------------------------------------------------------------------
# Runner + persistence
# ---------------------------------------------------------------------------


def run_system(config: SystemConfig, tasks: list[dict[str, Any]], force_offline: bool) -> dict[str, Any]:
    dispatch = _make_graph_runner(config, force_offline) if not config.llm_only else None
    records: list[dict[str, Any]] = []
    for task in tasks:
        if config.llm_only:
            pred = _run_llm_only(config, task)
        elif config.kind == "baseline" and config.name in ("B2_static_rag", "B3_static_sql"):
            pred = _run_static_tool(config, task)
        else:
            pred = _run_graph_system(config, task, dispatch)
        records.append(em.score_task(task, pred))
    summary = em.summarise(records)
    return {"system": config.name, "config": _config_payload(config), "results": records,
            "summary": summary}


def _config_payload(config: SystemConfig) -> dict[str, Any]:
    return {
        "kind": config.kind, "router": config.router,
        "retrieval_mode": config.retrieval_mode,
        "enable_verification": config.enable_verification,
        "enable_kg": config.enable_kg, "llm_only": config.llm_only,
        "description": config.description,
    }


def load_tasks(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks[:limit] if limit else tasks


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Phase 5 Enterprise End-to-End Evaluation", ""]
    svc = report.get("live_services") or {}
    mode = "offline-harness (deterministic)" if svc.get("deterministic_mode") else "online end-to-end"
    lines += [f"- mode: **{mode}**  (llm={svc.get('llm')} milvus={svc.get('milvus')} neo4j={svc.get('neo4j')})",
              f"- tasks per system: {report.get('tasks')}", ""]
    lines.append("## Overall (per system)")
    lines.append("| system | success | ans-exact | ans-semantic | route | tool F1 | plan EM | evidence | leakage | p95 ms | llm calls |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, payload in report["systems"].items():
        o = payload["summary"]["overall"]
        sem = o.get("answer_semantic")
        sem_cell = f"{sem:.3f}" if sem is not None else "—"
        leakage = o.get("critical_unsupported_leakage", 0)
        lines.append(
            f"| {name} | {o['task_success_rate']:.3f} | {o['answer_exact']:.3f} | "
            f"{sem_cell} | {o['route_accuracy']:.3f} | "
            f"{o['tool_f1']:.3f} | {o['plan_exact_match']:.3f} | "
            f"{o['evidence_attribution_accuracy']:.3f} | {leakage} | "
            f"{o['latency_total_ms_p95']:.0f} | {o['llm_calls_avg']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 5 enterprise end-to-end evaluation")
    parser.add_argument("--system", action="append", default=None,
                        help="system name to run (repeatable); default = all")
    parser.add_argument("--limit", type=int, default=None, help="cap tasks per system")
    parser.add_argument("--offline", action="store_true",
                        help="deterministic offline harness (no live LLM/Neo4j)")
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR,
                        help="artifact dir (default artifacts/phase5)")
    args = parser.parse_args(argv)

    tasks = load_tasks(DATASET, args.limit)
    if not tasks:
        print(f"ERROR: no tasks at {DATASET}")
        return 2

    names = args.system or [s.name for s in all_systems()]
    for n in names:
        get_system(n)  # validate early

    services = probe_services(args.offline)
    report: dict[str, Any] = {
        "dataset": str(DATASET), "tasks": len(tasks),
        "live_services": services, "systems": {},
    }
    for name in names:
        config = get_system(name)
        print(f"running system {name} over {len(tasks)} tasks ({'offline' if args.offline else 'live'}) …")
        t0 = time.perf_counter()
        payload = run_system(config, tasks, args.offline)
        payload["elapsed_s"] = round(time.perf_counter() - t0, 1)
        report["systems"][name] = payload
        o = payload["summary"]["overall"]
        print(f"  success={o['task_success_rate']:.3f} tool_f1={o['tool_f1']:.3f} "
              f"route={o['route_accuracy']:.3f} p95={o['latency_total_ms_p95']:.0f}ms")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "enterprise_eval_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    summary_doc = {k: v["summary"] for k, v in report["systems"].items()}
    summary_doc["_meta"] = {"tasks": report["tasks"], "live_services": services}
    (args.out / "enterprise_eval_summary.json").write_text(
        json.dumps(summary_doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_markdown(report, args.out / "enterprise_eval_summary.md")

    print(f"artifacts written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
