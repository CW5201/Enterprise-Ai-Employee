#!/usr/bin/env python3
"""Phase 2.2 — Hybrid RAG retrieval experiment runner.

Runs the local Phase 2.2 retrieval experiment set
(``data/eval/hybrid_eval.jsonl``) through three retrieval modes —
``dense`` (BGE-M3 + vector store), ``bm25`` (keyword), ``hybrid``
(dense + BM25 + RRF) — and computes Recall@1/3/5 and MRR per mode.

Everything is produced by the program; no metric is hand-filled.  The
result is written to ``artifacts/evaluation/phase2.2_hybrid_results.json``
(a small file, safe to commit).  The dense channel can target a real
Milvus server or the in-process fake store; whichever is used is
recorded in the output as ``dense_backend`` so results can never be
mistaken for a different backend.

Usage (from the repo root)::

    python scripts/run_hybrid_eval.py                 # real BGE-M3 + Milvus dense
    python scripts/run_hybrid_eval.py --fake-dense     # offline, fake dense backend

Fairness: every mode uses the SAME query, SAME corpus (the same 35
chunks feed both the dense store and the BM25 index) and SAME
top_k (5).  Only the fusion window inside hybrid_search is wider, and
that window is recorded in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_SET = PROJECT_ROOT / "data" / "eval" / "hybrid_eval.jsonl"
OUT_DIR = PROJECT_ROOT / "artifacts" / "evaluation"
OUT_JSON = OUT_DIR / "phase2.2_hybrid_results.json"
OUT_CSV = OUT_DIR / "phase2.2_hybrid_results.csv"
TOP_K = 5
ROUNDS = (1, 3, 5)
MODES = ("dense", "bm25", "hybrid")


# ---------------------------------------------------------------------------
# retrieval backends
# ---------------------------------------------------------------------------


def build_retriever(fake_dense: bool) -> tuple[Any, str]:
    from src.core.hybrid_retriever import HybridRetriever
    from src.core.llm_client import _load_dotenv

    # .env carries EMBEDDING_MODEL (local BGE-M3 dir) and MILVUS_* settings.
    _load_dotenv()

    if fake_dense:
        import os

        os.environ["EMBEDDING_FORCE_OFFLINE"] = "1"
        from src.core.embedder import HashEmbedder
        from src.core.vector_store import FakeVectorStore
        from src.tools.rag_tool import iter_kb_chunks_for_store

        embed = HashEmbedder()
        store = FakeVectorStore("kb_eval_fake", dim=embed.dim, embed_fn=embed)
        records = iter_kb_chunks_for_store()
        store.insert(records)
        retriever = HybridRetriever(vector_store=store, backend="fake")
        return retriever, "fake"
    else:
        # Formal dense backend: real BGE-M3 + real Milvus.  The BM25 index is
        # built over the same chunk corpus so both channels stay aligned.
        retriever = HybridRetriever(backend="milvus")
        return retriever, "milvus"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _retrieved_ids(retriever: Any, query: str, mode: str) -> list[str]:
    hits = retriever.search(query, top_k=TOP_K, mode=mode)
    return [h.chunk_id for h in hits]


def recall_at(retrieved: list[str], expected: set[str], k: int) -> float:
    top = set(retrieved[:k])
    return len(top & expected) / len(expected) if expected else 0.0


def mrr(retrieved: list[str], expected: set[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in expected:
            return 1.0 / rank
    return 0.0


def evaluate(retriever: Any, rows: list[dict]) -> tuple[list[dict], dict]:
    """Run every row x mode, return per-query records and per-mode aggregates."""
    records: list[dict] = []
    aggregates: dict[str, dict] = {}
    for mode in MODES:
        per_mode: list[dict] = []
        recall_sums: dict[int, float] = dict.fromkeys(ROUNDS, 0.0)
        mrr_sum = 0.0
        for row in rows:
            expected = set(row["expected_chunk_ids"])
            retrieved = _retrieved_ids(retriever, row["query"], mode)
            rmetrics = {f"recall@{k}": recall_at(retrieved, expected, k) for k in ROUNDS}
            m = mrr(retrieved, expected)
            rec = {
                "id": row["id"],
                "query": row["query"],
                "type": row.get("type", ""),
                "mode": mode,
                "top_k": TOP_K,
                "expected_chunk_ids": sorted(expected),
                "retrieved_chunk_ids": retrieved,
                "retrieval_mode": mode,
                **rmetrics,
                "mrr": m,
            }
            per_mode.append(rec)
            for k in ROUNDS:
                recall_sums[k] += rmetrics[f"recall@{k}"]
            mrr_sum += m
        n = max(len(rows), 1)
        aggregates[mode] = {
            "recall@1": round(recall_sums[1] / n, 4),
            "recall@3": round(recall_sums[3] / n, 4),
            "recall@5": round(recall_sums[5] / n, 4),
            "mrr": round(mrr_sum / n, 4),
            "queries": len(rows),
        }
        records.extend(per_mode)
    return records, aggregates


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2.2 Hybrid RAG retrieval experiment")
    parser.add_argument("--fake-dense", action="store_true", help="use the in-process fake dense store (offline)")
    args = parser.parse_args()

    if not EVAL_SET.exists():
        raise SystemExit(
            f"evaluation set not found: {EVAL_SET}\n"
            "run `python scripts/build_hybrid_eval.py` first"
        )
    rows = [json.loads(line) for line in EVAL_SET.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[run-hybrid-eval] loaded {len(rows)} queries")

    started = time.time()
    retriever, dense_backend = build_retriever(args.fake_dense)
    print(f"[run-hybrid-eval] dense backend = {dense_backend}")

    records, aggregates = evaluate(retriever, rows)

    result: dict[str, Any] = {
        "phase": "2.2",
        "experiment": "hybrid_rag_retrieval",
        "dataset": str(EVAL_SET.relative_to(PROJECT_ROOT)),
        "num_queries": len(rows),
        "top_k": TOP_K,
        "rounds": list(ROUNDS),
        "modes": list(MODES),
        "dense_backend": dense_backend,
        "dense_top_k": TOP_K,
        "bm25_top_k": TOP_K,
        "hybrid_final_top_k": TOP_K,
        # hybrid_search internally fuses a wider per-channel window so both
        # channels contribute before truncating to the final top_k; record it.
        "hybrid_fusion_window": max(TOP_K * 2, 10),
        "rrf_k": int(retriever._rrf_k),
        "corpus_chunks": retriever.bm25.size(),
        "dense_size": retriever.vector_store.size(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_seconds": round(time.time() - started, 2),
        "aggregates": aggregates,
        "per_query": records,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV: one row per (query, mode)
    csv_lines = [
        "id,query,mode,top_k,expected_chunk_ids,retrieved_chunk_ids,recall@1,recall@3,recall@5,mrr"
    ]
    for rec in records:
        exp = ";".join(rec["expected_chunk_ids"])
        ret = ";".join(rec["retrieved_chunk_ids"])
        csv_lines.append(
            f"{rec['id']},{rec['query'].replace(',', ';')},{rec['mode']},{rec['top_k']},"
            f"{exp},{ret},{rec['recall@1']:.4f},{rec['recall@3']:.4f},"
            f"{rec['recall@5']:.4f},{rec['mrr']:.4f}"
        )
    OUT_CSV.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    # Human-readable summary
    print("\n===== Phase 2.2 Hybrid RAG 检索实验汇总 =====")
    print(f"dense backend : {dense_backend}   corpus chunks: {retriever.bm25.size()}")
    print(f"{'mode':<8} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7}")
    for mode in MODES:
        a = aggregates[mode]
        print(f"{mode:<8} {a['recall@1']:>7.4f} {a['recall@3']:>7.4f} {a['recall@5']:>7.4f} {a['mrr']:>7.4f}")
    print(f"\nresults -> {OUT_JSON}")
    print(f"        -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
