"""Evaluation metrics - the six core metrics defined in docs/EVALUATION.md

Status
------
Phase 1: Basic implementation of the six core evaluation metrics.

See docs/ROADMAP.md for the phase in which this module is implemented,
docs/ARCHITECTURE.md for its role in the task pipeline, and ADR.md for the
design decisions behind it.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Set, Tuple

from src.core.exceptions import MetricsError


def task_success_rate(
    results: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate task success rate - the proportion of tasks where the conclusion
    reaches the determination standard.

    Args:
        results: List of system results, each containing at least 'task_id' and 'answer'
        ground_truth: List of ground truth entries, each containing at least 'task_id' and 'expected_answer'

    Returns:
        Dictionary with overall success rate and breakdowns by task type and difficulty
    """
    if len(results) != len(ground_truth):
        raise MetricsError("Results and ground truth must have the same length")

    # Create lookup for ground truth
    gt_lookup = {item["task_id"]: item for item in ground_truth}

    # Track successes
    total = len(results)
    success_count = 0

    # Breakdowns
    by_type = defaultdict(lambda: {"total": 0, "success": 0})
    by_difficulty = defaultdict(lambda: {"total": 0, "success": 0})

    for result in results:
        task_id = result.get("task_id")
        if task_id not in gt_lookup:
            continue

        gt_item = gt_lookup[task_id]
        task_type = gt_item.get("task_type", "unknown")
        difficulty = gt_item.get("difficulty", "unknown")

        # Update breakdown totals
        by_type[task_type]["total"] += 1
        by_difficulty[difficulty]["total"] += 1

        # Determine if successful based on task type
        is_successful = _is_task_successful(result, gt_item)

        if is_successful:
            success_count += 1
            by_type[task_type]["success"] += 1
            by_difficulty[difficulty]["success"] += 1

    # Calculate rates
    overall_rate = success_count / total if total > 0 else 0.0

    by_type_rates = {}
    for task_type, counts in by_type.items():
        rate = counts["success"] / counts["total"] if counts["total"] > 0 else 0.0
        by_type_rates[task_type] = rate

    by_difficulty_rates = {}
    for difficulty, counts in by_difficulty.items():
        rate = counts["success"] / counts["total"] if counts["total"] > 0 else 0.0
        by_difficulty_rates[difficulty] = rate

    return {
        "overall": overall_rate,
        "by_task_type": by_type_rates,
        "by_difficulty": by_difficulty_rates,
        "total_tasks": total,
        "successful_tasks": success_count
    }


def _is_task_successful(result: Dict[str, Any], ground_truth: Dict[str, Any]) -> bool:
    """Determine if a task is successful based on its type and ground truth."""
    task_type = ground_truth.get("task_type")

    if task_type == "information_query":
        # Information Query: result value equals Ground Truth
        return _values_equal(result.get("answer"), ground_truth.get("expected_answer"))

    elif task_type == "knowledge_qa":
        # Knowledge QA: conclusion matches reference points and cites correct documents
        return (
            _conclusion_matches_reference(result.get("answer"), ground_truth.get("expected_answer"))
            and _citations_correct(result.get("citations", []), ground_truth.get("expected_citations", []))
        )

    elif task_type == "text_to_sql":
        # Text-to-SQL: execution result matches reference result (order-independent)
        return _result_sets_equal(
            result.get("execution_result", []),
            ground_truth.get("expected_result", [])
        )

    elif task_type == "data_analysis":
        # Data Analysis: key statistics and conclusion direction match
        return _stats_and_direction_match(
            result.get("analysis", {}),
            ground_truth.get("expected_analysis", {})
        )

    elif task_type in ["multi_source_reasoning", "complex_agent_task"]:
        # Multi-source Reasoning: used necessary sources and conclusion matches Ground Truth
        # Complex Agent Task: delivered output satisfies task requirements
        return _sources_used_correctly(result.get("sources_used", []), ground_truth.get("required_sources", [])) \
            and _conclusion_matches_reference(result.get("answer"), ground_truth.get("expected_answer"))

    else:
        # Default: simple answer match
        return _values_equal(result.get("answer"), ground_truth.get("expected_answer"))


def routing_accuracy(
    results: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate routing accuracy - consistency between system's tool set and expected tool set.

    Reports:
    - Exact Match: tool sets are identical
    - Over-selection: system selected extra tools (wasteful, may introduce noise)
    - Under-selection: system missed required tools (capability gap, usually leads to failure)
    """
    if len(results) != len(ground_truth):
        raise MetricsError("Results and ground truth must have the same length")

    total = len(results)
    exact_match = 0
    over_selection = 0
    under_selection = 0

    # Breakdowns
    by_type_exact = defaultdict(lambda: {"total": 0, "exact": 0})
    by_type_over = defaultdict(lambda: {"total": 0, "over": 0})
    by_type_under = defaultdict(lambda: {"total": 0, "under": 0})

    for result, gt in zip(results, ground_truth):
        task_type = gt.get("task_type", "unknown")

        # Update breakdown totals
        by_type_exact[task_type]["total"] += 1
        by_type_over[task_type]["total"] += 1
        by_type_under[task_type]["total"] += 1

        expected_tools: Set[str] = set(gt.get("expected_tools", []))
        actual_tools: Set[str] = set(result.get("selected_tools", []))

        if actual_tools == expected_tools:
            exact_match += 1
            by_type_exact[task_type]["exact"] += 1
        elif actual_tools.issuperset(expected_tools):
            # Over-selection: has all expected tools plus extras
            over_selection += 1
            by_type_over[task_type]["over"] += 1
        else:
            # Under-selection: missing some expected tools
            under_selection += 1
            by_type_under[task_type]["under"] += 1

    # Calculate rates
    exact_rate = exact_match / total if total > 0 else 0.0
    over_rate = over_selection / total if total > 0 else 0.0
    under_rate = under_selection / total if total > 0 else 0.0

    # Breakdown rates
    by_type_exact_rates = {}
    for task_type, counts in by_type_exact.items():
        rate = counts["exact"] / counts["total"] if counts["total"] > 0 else 0.0
        by_type_exact_rates[task_type] = rate

    by_type_over_rates = {}
    for task_type, counts in by_type_over.items():
        rate = counts["over"] / counts["total"] if counts["total"] > 0 else 0.0
        by_type_over_rates[task_type] = rate

    by_type_under_rates = {}
    for task_type, counts in by_type_under.items():
        rate = counts["under"] / counts["total"] if counts["total"] > 0 else 0.0
        by_type_under_rates[task_type] = rate

    return {
        "exact_match": exact_rate,
        "over_selection": over_rate,
        "under_selection": under_rate,
        "by_task_type": {
            "exact_match": by_type_exact_rates,
            "over_selection": by_type_over_rates,
            "under_selection": by_type_under_rates
        },
        "total_tasks": total
    }


def retrieval_recall_at_k(
    results: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """
    Calculate Retrieval Recall@K - proportion of reference evidence covered in Top-K results.

    Reports separately for:
    - Dense / BM25 / Hybrid / Hybrid+Rerank configurations
    - Document-level hits and chunk-level hits (chunk-level is stricter)
    """
    if len(results) != len(ground_truth):
        raise MetricsError("Results and ground truth must have the same length")

    # Initialize results structure
    metrics = {
        "k_values": k_values,
        "configurations": ["dense", "bm25", "hybrid", "hybrid_rerank"],
        "levels": ["document", "chunk"]
    }

    # Prepare recall storage: config -> level -> k -> list of recall scores per query
    recall_scores = {
        config: {
            level: {k: [] for k in k_values}
            for level in ["document", "chunk"]
        }
        for config in ["dense", "bm25", "hybrid", "hybrid_rerank"]
    }

    for result, gt in zip(results, ground_truth):
        expected_docs: Set[str] = set(gt.get("expected_document_ids", []))
        expected_chunks: Set[str] = set(gt.get("expected_chunk_ids", []))

        # Process each configuration
        for config in ["dense", "bm25", "hybrid", "hybrid_rerank"]:
            config_key = f"retrieval_mode_{config}" if config != "hybrid_rerank" else "retrieval_mode_hybrid_rerank"
            retrieved_docs: Set[str] = set()
            retrieved_chunks: Set[str] = set()

            # Extract retrieved items based on config
            retrieval_results = result.get("retrieval_results", {})
            if config in retrieval_results:
                config_results = retrieval_results[config]
                # Get documents
                retrieved_docs.update(doc.get("document_id") for doc in config_results.get("documents", []) if doc.get("document_id"))
                # Get chunks
                retrieved_chunks.update(chunk.get("chunk_id") for chunk in config_results.get("chunks", []) if chunk.get("chunk_id"))

            # Calculate recall at each k for document level
            for k in k_values:
                if expected_docs:
                    retrieved_top_k_docs = set(list(retrieved_docs)[:k]) if len(retrieved_docs) > k else retrieved_docs
                    doc_recall = len(retrieved_top_k_docs & expected_docs) / len(expected_docs)
                    recall_scores[config]["document"][k].append(doc_recall)
                else:
                    recall_scores[config]["document"][k].append(0.0)  # No expected docs

                # Calculate recall at each k for chunk level
                if expected_chunks:
                    retrieved_top_k_chunks = set(list(retrieved_chunks)[:k]) if len(retrieved_chunks) > k else retrieved_chunks
                    chunk_recall = len(retrieved_top_k_chunks & expected_chunks) / len(expected_chunks)
                    recall_scores[config]["chunk"][k].append(chunk_recall)
                else:
                    recall_scores[config]["chunk"][k].append(0.0)  # No expected chunks

    # Calculate average recall for each config, level, and k
    recall_by_config = {}
    for config in ["dense", "bm25", "hybrid", "hybrid_rerank"]:
        recall_by_config[config] = {}
        for level in ["document", "chunk"]:
            recall_by_config[config][level] = {}
            for k in k_values:
                scores = recall_scores[config][level][k]
                avg_recall = mean(scores) if scores else 0.0
                recall_by_config[config][level][f"recall@{k}"] = avg_recall

    # Also calculate MRR (Mean Reciprocal Rank) if needed
    mrr_scores = {
        config: {"document": [], "chunk": []}
        for config in ["dense", "bm25", "hybrid", "hybrid_rerank"]
    }

    for result, gt in zip(results, ground_truth):
        expected_docs: Set[str] = set(gt.get("expected_document_ids", []))
        expected_chunks: Set[str] = set(gt.get("expected_chunk_ids", []))

        for config in ["dense", "bm25", "hybrid", "hybrid_rerank"]:
            retrieval_results = result.get("retrieval_results", {})
            if config in retrieval_results:
                config_results = retrieval_results[config]

                # Document MRR
                doc_rankings = []
                for i, doc in enumerate(config_results.get("documents", [])):
                    if doc.get("document_id") in expected_docs:
                        doc_rankings.append(i + 1)  # 1-indexed rank
                        break
                if doc_rankings:
                    mrr_scores[config]["document"].append(1.0 / doc_rankings[0])
                else:
                    mrr_scores[config]["document"].append(0.0)

                # Chunk MRR
                chunk_rankings = []
                for i, chunk in enumerate(config_results.get("chunks", [])):
                    if chunk.get("chunk_id") in expected_chunks:
                        chunk_rankings.append(i + 1)
                        break
                if chunk_rankings:
                    mrr_scores[config]["chunk"].append(1.0 / chunk_rankings[0])
                else:
                    mrr_scores[config]["chunk"].append(0.0)

    # Calculate average MRR
    mrr_by_config = {}
    for config in ["dense", "bm25", "hybrid", "hybrid_rerank"]:
        mrr_by_config[config] = {
            "mrr_document": mean(mrr_scores[config]["document"]) if mrr_scores[config]["document"] else 0.0,
            "mrr_chunk": mean(mrr_scores[config]["chunk"]) if mrr_scores[config]["chunk"] else 0.0
        }

    return {
        "recall_by_config": recall_by_config,
        "mrr_by_config": mrr_by_config,
        "total_queries": len(results)
    }


def sql_execution_accuracy(
    results: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate SQL Execution Accuracy - proportion of generated SQL that executes correctly
    and produces expected results.

    Distinguishes between:
    - Execution failure: syntax/field errors
    - Result mismatch: semantic errors (correct syntax but wrong results)
    """
    if len(results) != len(ground_truth):
        raise MetricsError("Results and ground truth must have the same length")

    total = len(results)
    execution_success = 0
    execution_failure = 0
    result_mismatch = 0

    # Breakdowns
    by_type = defaultdict(lambda: {"total": 0, "execution_success": 0, "execution_failure": 0, "result_mismatch": 0})

    for result, gt in zip(results, ground_truth):
        task_type = gt.get("task_type", "unknown")
        by_type[task_type]["total"] += 1

        sql_result = result.get("sql_execution_result", {})
        expected_result = gt.get("expected_sql_result", {})

        if not sql_result.get("success", False):
            # Execution failed
            execution_failure += 1
            by_type[task_type]["execution_failure"] += 1
        elif _result_sets_equal(sql_result.get("data", []), expected_result.get("data", [])):
            # Execution succeeded and results match
            execution_success += 1
            by_type[task_type]["execution_success"] += 1
        else:
            # Execution succeeded but results don't match
            result_mismatch += 1
            by_type[task_type]["result_mismatch"] += 1

    # Calculate rates
    execution_success_rate = execution_success / total if total > 0 else 0.0
    execution_failure_rate = execution_failure / total if total > 0 else 0.0
    result_mismatch_rate = result_mismatch / total if total > 0 else 0.0

    # Breakdown rates
    by_type_rates = {}
    for task_type, counts in by_type.items():
        total_t = counts["total"]
        if total_t > 0:
            by_type_rates[task_type] = {
                "execution_success_rate": counts["execution_success"] / total_t,
                "execution_failure_rate": counts["execution_failure"] / total_t,
                "result_mismatch_rate": counts["result_mismatch"] / total_t
            }
        else:
            by_type_rates[task_type] = {
                "execution_success_rate": 0.0,
                "execution_failure_rate": 0.0,
                "result_mismatch_rate": 0.0
            }

    return {
        "execution_success_rate": execution_success_rate,
        "execution_failure_rate": execution_failure_rate,
        "result_mismatch_rate": result_mismatch_rate,
        "by_task_type": by_type_rates,
        "total_tasks": total
    }


def evidence_support_rate(
    results: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate Evidence Support Rate - proportion of claims that are supported by evidence.

    Evidence Support Rate = supported claims / total claims

    Also tracks:
    - Partially supported claims (listed separately, not mixed with supported)
    - Unsupported + contradicted claims (unsupported-claim leakage)
    """
    if len(results) != len(ground_truth):
        raise MetricsError("Results and ground truth must have the same length")

    total_claims = 0
    supported_claims = 0
    partially_supported_claims = 0
    unsupported_claims = 0
    contradicted_claims = 0

    # Breakdowns
    by_type = defaultdict(lambda: {
        "total_claims": 0,
        "supported": 0,
        "partially_supported": 0,
        "unsupported": 0,
        "contradicted": 0
    })

    for result, gt in zip(results, ground_truth):
        task_type = gt.get("task_type", "unknown")
        claims = result.get("claims", [])
        expected_claims = gt.get("expected_claims", [])

        # For simplicity, we'll compare each claim with expected verdict
        # In practice, this would use the verification module's output
        for claim in claims:
            total_claims += 1
            by_type[task_type]["total_claims"] += 1

            verdict = claim.get("verdict", "unsupported")  # Default to unsupported
            if verdict == "supported":
                supported_claims += 1
                by_type[task_type]["supported"] += 1
            elif verdict == "partially_supported":
                partially_supported_claims += 1
                by_type[task_type]["partially_supported"] += 1
            elif verdict == "unsupported":
                unsupported_claims += 1
                by_type[task_type]["unsupported"] += 1
            elif verdict == "contradicted":
                contradicted_claims += 1
                by_type[task_type]["contradicted"] += 1

    # Calculate rates
    evidence_support_rate = supported_claims / total_claims if total_claims > 0 else 0.0
    partial_support_rate = partially_supported_claims / total_claims if total_claims > 0 else 0.0
    unsupported_claim_leakage = (unsupported_claims + contradicted_claims) / total_claims if total_claims > 0 else 0.0

    # Breakdown rates
    by_type_rates = {}
    for task_type, counts in by_type.items():
        total_t = counts["total_claims"]
        if total_t > 0:
            by_type_rates[task_type] = {
                "evidence_support_rate": counts["supported"] / total_t,
                "partial_support_rate": counts["partially_supported"] / total_t,
                "unsupported_claim_leakage": (counts["unsupported"] + counts["contradicted"]) / total_t
            }
        else:
            by_type_rates[task_type] = {
                "evidence_support_rate": 0.0,
                "partial_support_rate": 0.0,
                "unsupported_claim_leakage": 0.0
            }

    return {
        "evidence_support_rate": evidence_support_rate,
        "partial_support_rate": partial_support_rate,
        "unsupported_claim_leakage": unsupported_claim_leakage,
        "breakdown": {
            "supported_claims": supported_claims,
            "partially_supported_claims": partially_supported_claims,
            "unsupported_claims": unsupported_claims,
            "contradicted_claims": contradicted_claims
        },
        "by_task_type": by_type_rates,
        "total_claims": total_claims
    }


def average_latency(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate Average Latency - end-to-end average latency in milliseconds.

    Must report:
    - Overall average (P50), P90, P95
    - Broken down by component: intent understanding, routing, each tool, fusion, verification
    """
    if not results:
        return {
            "average_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p90_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "component_latencies": {}
        }

    # Extract latencies
    latencies = [r.get("total_latency_ms", 0.0) for r in results]

    # Calculate percentiles
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    def percentile(p):
        idx = (p / 100) * (n - 1)
        if idx.is_integer():
            return sorted_latencies[int(idx)]
        else:
            low = sorted_latencies[int(idx)]
            high = sorted_latencies[int(idx) + 1]
            return low + (high - low) * (idx - int(idx))

    p50 = percentile(50)
    p90 = percentile(90)
    p95 = percentile(95)
    avg = mean(latencies)

    # Component latencies
    components = ["intent_understanding", "routing", "sql_tool", "rag_tool", "kg_tool",
                  "analysis_tool", "fusion", "verification", "answer_generation"]

    component_latencies = {}
    for component in components:
        comp_latencies = [r.get(f"{component}_latency_ms", 0.0) for r in results if f"{component}_latency_ms" in r]
        if comp_latencies:
            comp_sorted = sorted(comp_latencies)
            comp_n = len(comp_sorted)
            comp_avg = mean(comp_latencies)
            comp_p50 = percentile(50) if comp_n > 0 else 0.0
            comp_p90 = percentile(90) if comp_n > 0 else 0.0
            comp_p95 = percentile(95) if comp_n > 0 else 0.0

            component_latencies[component] = {
                "average_ms": comp_avg,
                "p50_ms": comp_p50,
                "p90_ms": comp_p90,
                "p95_ms": comp_p95
            }

    return {
        "average_latency_ms": avg,
        "p50_latency_ms": p50,
        "p90_latency_ms": p90,
        "p95_latency_ms": p95,
        "component_latencies": component_latencies,
        "total_samples": len(results)
    }


# Helper functions
def _values_equal(val1: Any, val2: Any) -> bool:
    """Check if two values are equal, handling None and type differences."""
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False
    try:
        return val1 == val2
    except Exception:
        return str(val1) == str(val2)


def _conclusion_matches_reference(conclusion: Any, reference: Any) -> bool:
    """Check if conclusion matches reference (for knowledge QA)."""
    return _values_equal(conclusion, reference)


def _citations_correct(actual: List[Any], expected: List[Any]) -> bool:
    """Check if actual citations match expected citations."""
    if len(actual) != len(expected):
        return False
    # Simple order-independent comparison
    return sorted([str(c) for c in actual]) == sorted([str(c) for c in expected])


def _result_sets_equal(result1: List[Any], result2: List[Any]) -> bool:
    """Check if two result sets are equal (order-independent)."""
    try:
        # Convert to sets of tuples for comparison, handling unhashable types
        set1 = set(tuple(sorted(item.items())) if isinstance(item, dict) else str(item) for item in result1)
        set2 = set(tuple(sorted(item.items())) if isinstance(item, dict) else str(item) for item in result2)
        return set1 == set2
    except Exception:
        # Fallback to string comparison
        return str(sorted(result1)) == str(sorted(result2))


def _stats_and_direction_match(actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Check if statistics and conclusion direction match."""
    # Simple implementation - in practice would be more sophisticated
    return _values_equal(actual, expected)


def _sources_used_correctly(actual: List[Any], expected: List[Any]) -> bool:
    """Check if required sources were used."""
    if not expected:
        return True  # No sources required
    actual_set = set(str(s) for s in actual)
    expected_set = set(str(s) for s in expected)
    return expected_set.issubset(actual_set)


__all__: list[str] = [
    "task_success_rate",
    "routing_accuracy",
    "retrieval_recall_at_k",
    "sql_execution_accuracy",
    "evidence_support_rate",
    "average_latency"
]
