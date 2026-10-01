"""Evaluation reporter - Markdown / HTML report generation

Status
------
Phase 1: Basic implementation for generating evaluation reports.

See docs/ROADMAP.md for the phase in which this module is implemented,
docs/ARCHITECTURE.md for its role in the task pipeline, and ADR.md for the
design decisions behind it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.core.exceptions import ReporterError


def generate_markdown_report(results: Dict[str, Any], output_path: Path | str) -> None:
    """
    Generate a Markdown report from evaluation results.

    Args:
        results: Dictionary containing evaluation results and metadata
        output_path: Path where the Markdown report should be saved

    Raises:
        ReporterError: If the report cannot be generated
    """
    path = Path(output_path)
    try:
        with path.open("w", encoding="utf-8") as f:
            f.write("# Evaluation Report\n\n")

            # Dataset Information
            f.write("## Dataset Information\n")
            dataset_info = results.get("dataset", {})
            f.write(f"- Version: {dataset_info.get('version', 'N/A')}\n")
            f.write(f"- Size: {dataset_info.get('size', 'N/A')} entries\n")
            f.write(f"- Fields: {', '.join(dataset_info.get('fields', []))}\n\n")

            # Configuration
            f.write("## Configuration\n")
            config_info = results.get("config", {})
            f.write(f"- Version: {config_info.get('version', 'N/A')}\n")
            f.write(f"- LLM Model: {config_info.get('llm_model', 'N/A')}\n")
            f.write(f"- Embedding Model: {config_info.get('embedding_model', 'N/A')}\n\n")

            # Environment
            f.write("## Environment\n")
            env_info = results.get("environment", {})
            f.write(f"- CPU: {env_info.get('cpu', 'N/A')}\n")
            f.write(f"- Memory: {env_info.get('memory', 'N/A')}\n")
            f.write(f"- OS: {env_info.get('os', 'N/A')}\n")
            f.write(f"- Python Version: {env_info.get('python_version', 'N/A')}\n\n")

            # Metrics
            f.write("## Metrics\n")
            metrics = results.get("metrics", {})
            for metric_name, metric_value in metrics.items():
                f.write(f"- {metric_name.replace('_', ' ').title()}: {metric_value}\n")
            f.write("\n")

            # Component Latencies (if available)
            if "component_latencies" in results:
                f.write("## Component Latencies (ms)\n")
                comp_lat = results["component_latencies"]
                for component, latencies in comp_lat.items():
                    f.write(f"### {component}\n")
                    for lat_name, lat_value in latencies.items():
                        f.write(f"- {lat_name}: {lat_value}\n")
                    f.write("\n")

            # Additional Notes
            f.write("## Notes\n")
            notes = results.get("notes", "No additional notes.")
            f.write(f"{notes}\n")

    except Exception as e:
        raise ReporterError(f"Failed to generate Markdown report: {e}") from e


def generate_html_report(results: Dict[str, Any], output_path: Path | str) -> None:
    """
    Generate an HTML report from evaluation results.

    Args:
        results: Dictionary containing evaluation results and metadata
        output_path: Path where the HTML report should be saved

    Raises:
        ReporterError: If the report cannot be generated
    """
    # For simplicity, we'll convert the markdown report to HTML.
    # In a real implementation, we might use a template engine.
    try:
        import markdown
    except ImportError:
        raise ReporterError(
            "Markdown package is required for HTML report generation. "
            "Install it with: pip install markdown"
        )

    # First generate markdown content
    from io import StringIO
    import markdown

    md_buffer = StringIO()
    # We'll reuse the markdown generation logic by writing to a buffer
    # For now, we'll call generate_markdown_report to a temporary string
    # But since we don't want to write to file, we'll duplicate the logic or refactor.
    # Let's refactor: create a helper that returns the markdown string.

    # Instead, let's create a helper function that generates the markdown string.
    # However, to avoid changing the interface of generate_markdown_report too much,
    # we'll create an internal helper.

    # Given time constraints, we'll do a simple conversion: assume we have markdown string.
    # We'll create a temporary markdown string by mimicking the above.

    # Alternatively, we can change generate_markdown_report to return the string
    # and then have a separate function to write to file. But to minimize changes,
    # we'll do:

    # Generate markdown to a temporary file then read and convert? Not efficient.

    # Let's create a helper function that returns the markdown string.
    # We'll refactor the markdown generation into a helper.

    # Since we are already in the middle of the task, we'll do a simple approach:
    # We'll write the markdown to a temporary string buffer and then convert.

    # We'll copy the logic from generate_markdown_report but write to a StringIO.

    try:
        md_buffer = StringIO()
        md_buffer.write("# Evaluation Report\n\n")

        # Dataset Information
        md_buffer.write("## Dataset Information\n")
        dataset_info = results.get("dataset", {})
        md_buffer.write(f"- Version: {dataset_info.get('version', 'N/A')}\n")
        md_buffer.write(f"- Size: {dataset_info.get('size', 'N/A')} entries\n")
        md_buffer.write(f"- Fields: {', '.join(dataset_info.get('fields', []))}\n\n")

        # Configuration
        md_buffer.write("## Configuration\n")
        config_info = results.get("config", {})
        md_buffer.write(f"- Version: {config_info.get('version', 'N/A')}\n")
        md_buffer.write(f"- LLM Model: {config_info.get('llm_model', 'N/A')}\n")
        md_buffer.write(f"- Embedding Model: {config_info.get('embedding_model', 'N/A')}\n\n")

        # Environment
        md_buffer.write("## Environment\n")
        env_info = results.get("environment", {})
        md_buffer.write(f"- CPU: {env_info.get('cpu', 'N/A')}\n")
        md_buffer.write(f"- Memory: {env_info.get('memory', 'N/A')}\n")
        md_buffer.write(f"- OS: {env_info.get('os', 'N/A')}\n")
        md_buffer.write(f"- Python Version: {env_info.get('python_version', 'N/A')}\n\n")

        # Metrics
        md_buffer.write("## Metrics\n")
        metrics = results.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            md_buffer.write(f"- {metric_name.replace('_', ' ').title()}: {metric_value}\n")
        md_buffer.write("\n")

        # Component Latencies (if available)
        if "component_latencies" in results:
            md_buffer.write("## Component Latencies (ms)\n")
            comp_lat = results["component_latencies"]
            for component, latencies in comp_lat.items():
                md_buffer.write(f"### {component}\n")
                for lat_name, lat_value in latencies.items():
                    md_buffer.write(f"- {lat_name}: {lat_value}\n")
                md_buffer.write("\n")

        # Additional Notes
        md_buffer.write("## Notes\n")
        notes = results.get("notes", "No additional notes.")
        md_buffer.write(f"{notes}\n")

        md_content = md_buffer.getvalue()

        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

        # Write HTML file
        path = Path(output_path)
        with path.open("w", encoding="utf-8") as f:
            f.write(html_content)

    except Exception as e:
        raise ReporterError(f"Failed to generate HTML report: {e}") from e


def save_results_json(results: Dict[str, Any], output_path: Path | str) -> None:
    """
    Save evaluation results as JSON file.

    Args:
        results: Dictionary containing evaluation results
        output_path: Path where the JSON file should be saved

    Raises:
        ReporterError: If the results cannot be saved
    """
    try:
        path = Path(output_path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise ReporterError(f"Failed to save results JSON: {e}") from e


__all__: list[str] = [
    "generate_markdown_report",
    "generate_html_report",
    "save_results_json"
]
