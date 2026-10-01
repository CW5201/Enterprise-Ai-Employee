"""Evaluation runner - run baselines, ablations and the full system

Status
------
Phase 1: Basic implementation for running evaluations.

See docs/ROADMAP.md for the phase in which this module is implemented,
docs/ARCHITECTURE.md for its role in the task pipeline, and ADR.md for the
design decisions behind it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.exceptions import RunnerError
from src.evaluation.dataset import (
    get_dataset_info,
    load_jsonl_dataset,
)
from src.evaluation.metrics import (
    average_latency,
    evidence_support_rate,
    retrieval_recall_at_k,
    routing_accuracy,
    sql_execution_accuracy,
    task_success_rate,
)
from src.evaluation.reporter import (
    generate_html_report,
    generate_markdown_report,
    save_results_json,
)


def run_system(
    query: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Placeholder for running the actual system on a query.
    In a real implementation, this would invoke the LangGraph or API.
    For now, returns a mock result.

    Args:
        query: The user query
        config: Optional configuration for the system run

    Returns:
        A dictionary representing the system result
    """
    # This is a mock implementation. In reality, this would call the system.
    # We return a structure that matches what the evaluation expects.
    return {
        "request_id": "mock",
        "user_query": query,
        "timestamp": datetime.now().isoformat(),
        "intent": "unknown",
        "confidence": 0.0,
        "route": "clarification",
        "selected_tools": [],
        "answer": "This is a mock answer. The system is not implemented.",
        "citations": [],
        "claims": [],
        "evidence": [],
        "retrieval_results": {},
        "sql_execution_result": {"success": False, "error": "Not implemented"},
        "total_latency_ms": 0.0,
        # Component latencies (mock)
        "intent_understanding_latency_ms": 0.0,
        "routing_latency_ms": 0.0,
        "sql_tool_latency_ms": 0.0,
        "rag_tool_latency_ms": 0.0,
        "kg_tool_latency_ms": 0.0,
        "analysis_tool_latency_ms": 0.0,
        "fusion_latency_ms": 0.0,
        "verification_latency_ms": 0.0,
        "answer_generation_latency_ms": 0.0,
    }


def run_evaluation(
    dataset_path: Path,
    output_dir: Path,
    config: dict[str, Any] | None = None,
    system_name: str = "ours",
) -> dict[str, Any]:
    """
    Run evaluation on a dataset.

    Args:
        dataset_path: Path to the JSONL dataset
        output_dir: Directory to save results
        config: Optional system configuration
        system_name: Name of the system being evaluated (for reporting)

    Returns:
        Dictionary containing the evaluation results and metrics
    """
    # Load dataset
    try:
        dataset = load_jsonl_dataset(dataset_path)
    except Exception as e:
        raise RunnerError(f"Failed to load dataset: {e}") from e

    dataset_info = get_dataset_info(dataset)

    # Run each item in the dataset
    results = []
    ground_truth = []  # We'll extract ground truth from the dataset

    for item in dataset:
        # In a real dataset, we would have ground truth fields.
        # For now, we assume the dataset has the necessary fields for evaluation.
        # We'll separate the input (query) from the expected outputs.
        query = item.get("user_query", "")
        if not query:
            continue

        # Run the system
        try:
            system_result = run_system(query, config)
        except Exception as e:
            # If the system fails, we record the error but continue
            system_result = {
                "request_id": item.get("request_id", "unknown"),
                "user_query": query,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "answer": "",
                # ... other fields with default/error values
            }

        results.append(system_result)
        # We assume the dataset item contains the ground truth for evaluation
        ground_truth.append(item)

    # Calculate metrics
    try:
        metrics = {}
        metrics["task_success_rate"] = task_success_rate(results, ground_truth)
        metrics["routing_accuracy"] = routing_accuracy(results, ground_truth)
        metrics["retrieval_recall_at_k"] = retrieval_recall_at_k(results, ground_truth)
        metrics["sql_execution_accuracy"] = sql_execution_accuracy(results, ground_truth)
        metrics["evidence_support_rate"] = evidence_support_rate(results, ground_truth)
        metrics["average_latency"] = average_latency(results)
    except Exception as e:
        raise RunnerError(f"Failed to calculate metrics: {e}") from e

    # Prepare final results
    eval_results = {
        "system_name": system_name,
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_info,
        "config": config or {},
        "metrics": metrics,
        "results": results,  # In practice, we might not save all results to avoid huge files
        "ground_truth": ground_truth,  # Similarly, we might not save all
        "notes": "Evaluation completed. This is a basic implementation.",
    }

    # Save results
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_json_path = output_dir / "results.json"
        save_results_json(eval_results, results_json_path)

        # Generate reports
        generate_markdown_report(eval_results, output_dir / "report.md")
        generate_html_report(eval_results, output_dir / "report.html")
    except Exception as e:
        raise RunnerError(f"Failed to save results or generate reports: {e}") from e

    return eval_results


def main() -> None:
    """Main entry point for the evaluation runner."""
    parser = argparse.ArgumentParser(description="Run evaluation for the Enterprise AI Employee system")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the JSONL dataset file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results and reports",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="ours",
        help="Name of the system being evaluated (default: ours)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="JSON string or path to JSON file containing system configuration",
    )

    args = parser.parse_args()

    # Parse config if provided
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            try:
                with config_path.open(encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_path}: {e}")
                config = None
        else:
            # Try to parse as JSON string
            try:
                config = json.loads(args.config)
            except json.JSONDecodeError:
                print(f"Warning: --config argument is not a valid JSON string or file: {args.config}")
                config = None

    # Run evaluation
    try:
        run_evaluation(
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output-dir),
            config=config,
            system_name=args.system,
        )
        print(f"Evaluation completed. Results saved to {args.output-dir}")
    except RunnerError as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__: list[str] = [
    "run_evaluation",
    "main"
]
