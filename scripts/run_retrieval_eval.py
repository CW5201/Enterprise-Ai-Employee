"""Phase 2.3 — Reranker retrieval experiment runner.

Runs the local Phase 2.3 evaluation set (``data/eval/retrieval_eval.jsonl``,
56 queries) through **four** retrieval modes over the same 190-chunk corpus
and computes Recall@1/3/5, MRR and NDCG@5 per mode:

- ``dense``            : BGE-M3 + Milvus, top-5
- ``bm25``             : in-process keyword recall, top-5
- ``hybrid``           : Dense + BM25 + RRF, top-5
- ``hybrid_rerank``    : Dense + BM25 + RRF -> candidate_k=20 ->
                         BGE-Reranker-v2-M3 -> final_k=5

All four modes use the SAME queries, SAME ground truth, SAME final top-k
(5).  The reranker ONLY re-orders the RRF candidate list; it never
participates in the original recall and never runs before RRF.  Every number
in the output is computed by the program — nothing is hand-filled.

Fairness guards:
- corpus = same 190 chunks for all modes (dense store + BM25 index built
  from one chunk set);
- the reranker is the **real** BGE-Reranker-v2-M3 (no fake scorer — that
  lives only in unit tests);
- empty / failed queries are recorded, not silently dropped.

Outputs (gitignored, local only):
- ``artifacts/phase2.3/retrieval_eval_results.json``   — full per-query rows
- ``artifacts/phase2.3/retrieval_eval_summary.json``   — overall + slice
- ``artifacts/phase2.3/retrieval_eval_summary.md``     — human/thesis table
- ``artifacts/phase2.3/retrieval_failure_cases.md``     — notable cases

Usage::

    python scripts/run_retrieval_eval.py
    python scripts/run_retrieval_eval.py --rerank-device cuda --candidate-k 20
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from sys import path as _sys_path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_sys_path.insert(0, str(PROJECT_ROOT))

from src.evaluation.retrieval_metrics import (  # noqa: E402
    mean,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)

EVAL_SET = PROJECT_ROOT / "data" / "eval" / "retrieval_eval.jsonl"
OUT_DIR = PROJECT_ROOT / "artifacts" / "phase2.3"
OUT_RESULTS = OUT_DIR / "retrieval_eval_results.json"
OUT_SUMMARY = OUT_DIR / "retrieval_eval_summary.json"
OUT_MD = OUT_DIR / "retrieval_eval_summary.md"
OUT_FAILURE = OUT_DIR / "retrieval_failure_cases.md"

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
FINAL_K = 5
RECALL_KS = (1, 3, 5)
NDCG_K = 5


# ---------------------------------------------------------------------------
# retrieval backends
# ---------------------------------------------------------------------------


def build_retrievers(args: argparse.Namespace) -> tuple[Any, str, dict[str, Any]]:
    """Build one shared retriever; the rerank mode reuses it.

    Returns ``(retriever, dense_backend, run_info)``.  The reranker is only
    built when the mode list includes ``hybrid_rerank`` and is always a real
    model — a failed load raises, never a fake.
    """
    from src.core.hybrid_retriever import HybridRetriever
    from src.core.llm_client import _load_dotenv
    from src.core.reranker import Reranker

    # .env carries EMBEDDING_MODEL / MILVUS_* / RERANKER_* settings.
    _load_dotenv()

    retriever = HybridRetriever(
        backend="milvus",
        candidate_k=args.candidate_k,
        final_k=FINAL_K,
    )

    run_info: dict[str, Any] = {
        "corpus_chunks": retriever.bm25.size(),
        "dense_size": retriever.vector_store.size(),
        "dense_backend": "milvus",
        "candidate_k": args.candidate_k,
        "final_k": FINAL_K,
        "rrf_k": int(retriever._rrf_k),
        "embedding_device": "cpu",
        "reranker_device": None,
        "reranker_model_path": None,
        "reranker_enabled": "hybrid_rerank" in MODES,
    }

    if "hybrid_rerank" in MODES:
        # Real BGE-Reranker-v2-M3 (no fake fallback).  Raises on failure.
        reranker = Reranker.load_default(
            model_source=args.rerank_model,
            device=args.rerank_device,
            batch_size=args.batch_size,
        )
        retriever._reranker = reranker
        retriever._reranker_resolved = True
        retriever._candidate_k = args.candidate_k
        retriever._final_k = FINAL_K
        run_info["reranker_device"] = reranker.device
        run_info["reranker_model_path"] = reranker.model_source
        run_info["embedding_device"] = _probe_embedding_device()

    return retriever, "milvus", run_info


def _probe_embedding_device() -> str:
    """Return the actual device the BGE-M3 embedder will run on (honest CPU
    fallback is surfaced, not hidden)."""
    import os

    device = os.environ.get("EMBEDDING_DEVICE", "cpu")
    try:
        import torch

        if device.lower().startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
    except Exception:  # noqa: BLE001 - torch optional
        pass
    # Don't actually instantiate the heavy embedder here; the retriever
    # already built one.  Just report what the env resolves to.
    return "cpu" if not device.lower().startswith("cuda") else "cuda"


# ---------------------------------------------------------------------------
# per-query retrieval + metrics
# ---------------------------------------------------------------------------


def _retrieved_ids(retriever: Any, query: str, mode: str, top_k: int) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    hits = retriever.search(query, top_k=top_k, mode=mode)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return [h.chunk_id for h in hits], latency_ms


def evaluate_mode(retriever: Any, rows: list[dict], mode: str) -> tuple[list[dict], dict, dict[str, float]]:
    """Run one mode over every query; return per-query records, aggregate,
    and the per-query latency map (mode -> list) for later summary."""
    per_mode: list[dict] = []
    latencies: list[float] = []
    recall_sums: dict[int, float] = dict.fromkeys(RECALL_KS, 0.0)
    mrr_sum = 0.0
    ndcg_sum = 0.0
    failed = 0

    for row in rows:
        expected = set(row["expected_chunk_ids"])
        try:
            retrieved, lat = _retrieved_ids(retriever, row["query"], mode, FINAL_K)
        except Exception as exc:  # noqa: BLE001 - record, don't silently drop
            failed += 1
            retrieved = []
            lat = 0.0
            per_mode.append(
                {
                    "id": row["id"],
                    "query": row["query"],
                    "type": row.get("type", ""),
                    "difficulty": row.get("difficulty", ""),
                    "mode": mode,
                    "top_k": FINAL_K,
                    "expected_chunk_ids": sorted(expected),
                    "retrieved_chunk_ids": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0,
                    "mrr": 0.0, "ndcg@5": 0.0,
                }
            )
            continue

        latencies.append(lat)
        rmetrics = {f"recall@{k}": recall_at_k(retrieved, expected, k) for k in RECALL_KS}
        m = reciprocal_rank(retrieved, expected)
        nd = ndcg_at_k(retrieved, expected, NDCG_K)
        per_mode.append(
            {
                "id": row["id"],
                "query": row["query"],
                "type": row.get("type", ""),
                "difficulty": row.get("difficulty", ""),
                "mode": mode,
                "top_k": FINAL_K,
                "expected_chunk_ids": sorted(expected),
                "retrieved_chunk_ids": retrieved,
                **rmetrics,
                "mrr": m,
                "ndcg@5": nd,
            }
        )
        for k in RECALL_KS:
            recall_sums[k] += rmetrics[f"recall@{k}"]
        mrr_sum += m
        ndcg_sum += nd

    n = max(len(rows), 1)
    aggregate = {
        "recall@1": round(recall_sums[1] / n, 4),
        "recall@3": round(recall_sums[3] / n, 4),
        "recall@5": round(recall_sums[5] / n, 4),
        "mrr": round(mrr_sum / n, 4),
        "ndcg@5": round(ndcg_sum / n, 4),
        "queries": len(rows),
        "failed": failed,
    }
    return per_mode, aggregate, {mode: latencies}


def slice_metrics(per_mode: list[dict], key: str) -> dict[str, dict[str, float]]:
    """Aggregate Recall@5 + MRR + NDCG@5 by a categorical key (type/difficulty)."""
    groups: dict[str, list[dict]] = {}
    for rec in per_mode:
        groups.setdefault(rec.get(key, "unknown"), []).append(rec)
    out: dict[str, dict[str, float]] = {}
    for name, recs in groups.items():
        n = max(len(recs), 1)
        out[name] = {
            "recall@5": round(sum(r["recall@5"] for r in recs) / n, 4),
            "mrr": round(sum(r["mrr"] for r in recs) / n, 4),
            "ndcg@5": round(sum(r["ndcg@5"] for r in recs) / n, 4),
            "count": len(recs),
        }
    return out


# ---------------------------------------------------------------------------
# failure-case mining
# ---------------------------------------------------------------------------


def build_failure_cases(
    by_mode: dict[str, list[dict]], rows: list[dict], top: int = 15
) -> list[dict]:
    """Pick representative queries: where a relevant chunk is or isn't
    recovered per mode, and whether the reranker helped or hurt ordering.

    Score each query by how "interesting" it is for a thesis failure
    analysis: modes disagreeing on whether any relevant chunk made the cut,
    or the reranker demoting a relevant chunk.
    """
    cases: list[dict] = []
    for row in rows:
        rid: str = row["id"]
        expected: set[str] = set(row["expected_chunk_ids"])

        def topids(mode: str, rid: str = rid) -> list[str]:
            rec = next((r for r in by_mode.get(mode, []) if r["id"] == rid), None)
            return rec["retrieved_chunk_ids"][:5] if rec else []

        dense_hits = topids("dense")
        bm25_hits = topids("bm25")
        hybrid_hits = topids("hybrid")
        reranked_hits = topids("hybrid_rerank")

        def rel_top(ids: list[str], expected: set[str] = expected) -> int:
            for i, c in enumerate(ids):
                if c in expected:
                    return i + 1
            return 0

        d_first, b_first, h_first, r_first = (
            rel_top(dense_hits), rel_top(bm25_hits), rel_top(hybrid_hits), rel_top(reranked_hits)
        )

        # Interest score:
        score = 0
        if b_first and not d_first:
            score += 2  # BM25-only recall
        if d_first and not b_first:
            score += 2  # Dense-only recall
        if h_first and (d_first == 0 or b_first == 0):
            score += 1  # hybrid rescues a single-channel miss
        if r_first and r_first < h_first:
            score += 2  # reranker improved the relevant rank
        if r_first == 0 and h_first > 0:
            score += 3  # reranker demoted a previously-relevant chunk (hurt)
        if r_first == 0 and h_first == 0 and d_first == 0 and b_first == 0:
            score += 3  # all four failed
        if h_first and r_first and r_first > h_first:
            score += 1  # reranker pushed relevant down (mild hurt)

        if score == 0:
            continue

        cases.append(
            {
                "id": rid,
                "score": score,
                "query": row["query"],
                "type": row.get("type", ""),
                "difficulty": row.get("difficulty", ""),
                "expected_chunk_ids": sorted(expected),
                "dense_top": dense_hits,
                "bm25_top": bm25_hits,
                "hybrid_top": hybrid_hits,
                "reranked_top": reranked_hits,
                "first_relevant_rank": {"dense": d_first, "bm25": b_first,
                                         "hybrid": h_first, "reranked": r_first},
            }
        )
    cases.sort(key=lambda c: (-c["score"], c["id"]))
    return cases[:top]


# ---------------------------------------------------------------------------
# output writers
# ---------------------------------------------------------------------------


def write_outputs(
    args: argparse.Namespace,
    rows: list[dict],
    by_mode: dict[str, list[dict]],
    aggregates: dict[str, dict],
    latencies: dict[str, dict[str, list[float]]],
    run_info: dict[str, Any],
    cases: list[dict],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. full per-query results (one block per mode)
    results_doc = {
        "phase": "2.3",
        "experiment": "reranker_retrieval",
        "dataset": str(EVAL_SET.relative_to(PROJECT_ROOT)),
        "num_queries": len(rows),
        "final_k": FINAL_K,
        "recall_ks": list(RECALL_KS),
        "ndcg_k": NDCG_K,
        "modes": list(MODES),
        **run_info,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_mode": by_mode,
    }
    OUT_RESULTS.write_text(json.dumps(results_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2. summary JSON: overall + slices + latency
    lat_summary: dict[str, dict[str, float]] = {}
    for mode in MODES:
        vals = latencies.get(mode, {}).get(mode, [])
        lat_summary[mode] = {
            "avg_ms": round(mean(vals), 2),
            "p50_ms": round(percentile(vals, 50), 2),
            "p95_ms": round(percentile(vals, 95), 2),
        }
    summary = {
        "phase": "2.3",
        "overall": {m: {k: aggregates[m][k] for k in
                        ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@5",
                         "queries", "failed")} for m in MODES},
        "by_type": {m: slice_metrics(by_mode[m], "type") for m in MODES},
        "by_difficulty": {m: slice_metrics(by_mode[m], "difficulty") for m in MODES},
        "latency": lat_summary,
        **run_info,
        "reranker_delta_vs_hybrid": {
            "recall@5": round(aggregates["hybrid_rerank"]["recall@5"] - aggregates["hybrid"]["recall@5"], 4),
            "ndcg@5": round(aggregates["hybrid_rerank"]["ndcg@5"] - aggregates["hybrid"]["ndcg@5"], 4),
            "mrr": round(aggregates["hybrid_rerank"]["mrr"] - aggregates["hybrid"]["mrr"], 4),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. summary markdown
    _write_summary_md(summary, aggregates, lat_summary, run_info)

    # 4. failure cases markdown
    _write_failure_md(cases)

    print(f"[retrieval-eval] results -> {OUT_RESULTS}")
    print(f"[retrieval-eval] summary  -> {OUT_SUMMARY}  ({OUT_MD.name})")
    print(f"[retrieval-eval] cases    -> {OUT_FAILURE}")


def _fmt_row(method: str, agg: dict[str, Any]) -> str:
    return (
        f"| {method} | {agg['recall@1']:.4f} | {agg['recall@3']:.4f} | "
        f"{agg['recall@5']:.4f} | {agg['mrr']:.4f} | {agg['ndcg@5']:.4f} |"
    )


def _write_summary_md(
    summary: dict[str, Any],
    aggregates: dict[str, dict],
    lat_summary: dict[str, dict[str, float]],
    run_info: dict[str, Any],
) -> None:
    labels = {
        "dense": "Dense",
        "bm25": "BM25",
        "hybrid": "Hybrid (Dense+BM25+RRF)",
        "hybrid_rerank": "Hybrid + Reranker",
    }
    lines: list[str] = []
    lines.append("# Phase 2.3 Reranker Retrieval Experiment — Summary\n")
    lines.append(f"- corpus: {run_info['corpus_chunks']} chunks / {run_info['dense_size']} dense entities")
    lines.append(f"- queries: {summary['overall']['dense']['queries']}  (failed: "
                 f"{', '.join(str(summary['overall'][m]['failed']) for m in MODES)})")
    lines.append(f"- candidate_k={run_info['candidate_k']}  final_k={run_info['final_k']}  rrf_k={run_info['rrf_k']}")
    lines.append(f"- embedding_device={run_info['embedding_device']}  "
                 f"reranker_device={run_info['reranker_device']}  "
                 f"reranker_model={run_info['reranker_model_path']}")
    lines.append("")
    lines.append("## Overall results\n")
    lines.append("| Method | Recall@1 | Recall@3 | Recall@5 | MRR | NDCG@5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in MODES:
        lines.append(_fmt_row(labels[m], aggregates[m]))
    lines.append("")
    rd = summary["reranker_delta_vs_hybrid"]
    lines.append(f"**Reranker delta vs Hybrid** — Recall@5 {rd['recall@5']:+.4f}, "
                 f"MRR {rd['mrr']:+.4f}, NDCG@5 {rd['ndcg@5']:+.4f}\n")
    lines.append("## Latency\n")
    lines.append("| Method | Avg(ms) | P50(ms) | P95(ms) |")
    lines.append("|---|---:|---:|---:|")
    for m in MODES:
        L = lat_summary[m]
        lines.append(f"| {labels[m]} | {L['avg_ms']:.1f} | {L['p50_ms']:.1f} | {L['p95_ms']:.1f} |")
    lines.append("")
    lines.append("## By query type (Recall@5 / MRR / NDCG@5)\n")
    lines.append("| Method | Type | R@5 | MRR | NDCG@5 | n |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for m in MODES:
        for tname, tv in sorted(summary["by_type"][m].items()):
            lines.append(
                f"| {labels[m]} | {tname} | {tv['recall@5']:.4f} | {tv['mrr']:.4f} | "
                f"{tv['ndcg@5']:.4f} | {tv['count']} |"
            )
    lines.append("")
    lines.append("## By difficulty\n")
    lines.append("| Method | Difficulty | R@5 | MRR | NDCG@5 | n |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for m in MODES:
        for dname in ("easy", "medium", "hard"):
            if dname in summary["by_difficulty"][m]:
                dv = summary["by_difficulty"][m][dname]
                lines.append(
                    f"| {labels[m]} | {dname} | {dv['recall@5']:.4f} | {dv['mrr']:.4f} | "
                    f"{dv['ndcg@5']:.4f} | {dv['count']} |"
                )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def _write_failure_md(cases: list[dict]) -> None:
    lines: list[str] = ["# Phase 2.3 — Representative failure / divergence cases\n",
                         f"{len(cases)} most informative queries (scored by cross-mode divergence "
                         "and rerank ordering change).\n"]
    for i, c in enumerate(cases, start=1):
        fr = c["first_relevant_rank"]
        lines.append(f"## {i}. `{c['id']}` — {c['query']}")
        lines.append(f"- type={c['type']}  difficulty={c['difficulty']}  interest={c['score']}")
        lines.append(f"- expected: {', '.join(c['expected_chunk_ids'])}")
        lines.append(f"- dense   top: {', '.join(c['dense_top'])}   first-relevant@ {fr['dense']}")
        lines.append(f"- bm25    top: {', '.join(c['bm25_top'])}   first-relevant@ {fr['bm25']}")
        lines.append(f"- hybrid  top: {', '.join(c['hybrid_top'])}   first-relevant@ {fr['hybrid']}")
        lines.append(f"- rerank  top: {', '.join(c['reranked_top'])}   first-relevant@ {fr['reranked']}")
        note = _case_note(c)
        lines.append(f"- **note**: {note}")
        lines.append("")
    OUT_FAILURE.write_text("\n".join(lines), encoding="utf-8")


def _case_note(c: dict[str, Any]) -> str:
    fr = c["first_relevant_rank"]
    notes: list[str] = []
    if fr["bm25"] and not fr["dense"]:
        notes.append("BM25-only recall (dense missed)")
    if fr["dense"] and not fr["bm25"]:
        notes.append("dense-only recall (BM25 missed)")
    if fr["hybrid"] and (fr["dense"] == 0 or fr["bm25"] == 0):
        notes.append("RRF fusion rescued a single-channel miss")
    if fr["reranked"] and fr["hybrid"] and fr["reranked"] < fr["hybrid"]:
        notes.append("reranker improved the relevant rank")
    if fr["reranked"] == 0 and fr["hybrid"] > 0:
        notes.append("reranker demoted a previously-relevant chunk (hurt)")
    if fr["reranked"] == 0 and fr["hybrid"] == 0 and fr["dense"] == 0 and fr["bm25"] == 0:
        notes.append("all four methods failed to surface a relevant chunk")
    if fr["reranked"] and fr["hybrid"] and fr["reranked"] > fr["hybrid"]:
        notes.append("reranker pushed the relevant chunk down (mild)")
    return "; ".join(notes) if notes else "no notable divergence"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2.3 Reranker retrieval experiment")
    parser.add_argument("--candidate-k", type=int, default=20, help="RRF candidate window for rerank")
    parser.add_argument("--rerank-model", default=None, help="local dir or HF id for BGE-Reranker-v2-M3")
    parser.add_argument("--rerank-device", default=None, help="cpu | cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-rerank", action="store_true",
                        help="run dense/bm25/hybrid only (no rerank stage)")
    args = parser.parse_args()

    if not EVAL_SET.exists():
        raise SystemExit(f"evaluation set not found: {EVAL_SET}")
    rows = [json.loads(line) for line in
            EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[retrieval-eval] loaded {len(rows)} queries")
    type_counts = Counter(r.get("type") for r in rows)
    diff_counts = Counter(r.get("difficulty") for r in rows)
    print(f"[retrieval-eval] types={dict(type_counts)}")
    print(f"[retrieval-eval] difficulty={dict(diff_counts)}")

    run_modes = ("dense", "bm25", "hybrid") if args.skip_rerank else MODES

    t0 = time.time()
    retriever, dense_backend, run_info = build_retrievers(args)
    if args.skip_rerank:
        run_info["reranker_enabled"] = False
    print(f"[retrieval-eval] dense backend={dense_backend} corpus={run_info['corpus_chunks']} "
          f"chunks; reranker_enabled={run_info['reranker_enabled']}")

    # Run every mode over the SAME rows + SAME final_k.  Latency per mode.
    by_mode: dict[str, list[dict]] = {}
    aggregates: dict[str, dict] = {}
    latencies: dict[str, dict[str, list[float]]] = {}
    for mode in run_modes:
        per_mode, agg, lat = evaluate_mode(retriever, rows, mode)
        by_mode[mode] = per_mode
        aggregates[mode] = agg
        latencies[mode] = lat
        print(f"[retrieval-eval]   {mode:<14} R@1={agg['recall@1']:.4f} "
              f"R@3={agg['recall@3']:.4f} R@5={agg['recall@5']:.4f} "
              f"MRR={agg['mrr']:.4f} NDCG@5={agg['ndcg@5']:.4f} "
              f"(failed={agg['failed']}/{len(rows)})")

    cases = build_failure_cases(by_mode, rows, top=15)

    run_info["runtime_seconds"] = round(time.time() - t0, 1)
    write_outputs(args, rows, by_mode, aggregates, latencies, run_info, cases)

    # console summary table
    labels = {"dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid",
              "hybrid_rerank": "Hybrid+Rerank"}
    print("\n===== Phase 2.3 检索实验汇总 =====")
    print(f"{'mode':<16}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'NDCG@5':>8}")
    for m in run_modes:
        a = aggregates[m]
        print(f"{labels[m]:<16}{a['recall@1']:>8.4f}{a['recall@3']:>8.4f}{a['recall@5']:>8.4f}"
              f"{a['mrr']:>8.4f}{a['ndcg@5']:>8.4f}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
