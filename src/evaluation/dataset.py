"""Evaluation dataset - loading and version validation

Status
------
Phase 1: Basic implementation for loading evaluation datasets.

See docs/ROADMAP.md for the phase in which this module is implemented,
docs/ARCHITECTURE.md for its role in the task pipeline, and ADR.md for the
design decisions behind it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.core.exceptions import DatasetError


def load_jsonl_dataset(file_path: Path | str) -> list[dict[str, Any]]:
    """
    Load a JSONL dataset file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries representing the dataset entries

    Raises:
        DatasetError: If the file cannot be read or parsed
    """
    path = Path(file_path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    try:
        with path.open(encoding="utf-8") as f:
            lines = f.readlines()

        dataset = []
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                dataset.append(entry)
            except json.JSONDecodeError as e:
                raise DatasetError(
                    f"Invalid JSON on line {i} in {path}: {e}"
                ) from e

        return dataset
    except Exception as e:
        if isinstance(e, DatasetError):
            raise
        raise DatasetError(f"Failed to load dataset from {path}: {e}") from e


def iterate_jsonl_dataset(file_path: Path | str) -> Iterator[dict[str, Any]]:
    """
    Iterate over a JSONL dataset file lazily.

    Args:
        file_path: Path to the JSONL file

    Yields:
        Dictionary representing each dataset entry

    Raises:
        DatasetError: If the file cannot be read or parsed
    """
    path = Path(file_path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    try:
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise DatasetError(
                        f"Invalid JSON on line {i} in {path}: {e}"
                    ) from e
    except Exception as e:
        if isinstance(e, DatasetError):
            raise
        raise DatasetError(f"Failed to iterate dataset from {path}: {e}") from e


def get_dataset_info(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Get basic information about a dataset.

    Args:
        dataset: List of dataset entries

    Returns:
        Dictionary with dataset information
    """
    if not dataset:
        return {
            "size": 0,
            "fields": [],
            "sample": None
        }

    # Get all unique keys across all entries
    fields = set()
    for entry in dataset:
        if isinstance(entry, dict):
            fields.update(entry.keys())

    return {
        "size": len(dataset),
        "fields": sorted(fields),
        "sample": dataset[0] if dataset else None
    }


def validate_dataset_version(
    dataset: list[dict[str, Any]],
    expected_version: str,
    version_field: str = "version"
) -> bool:
    """
    Validate that all entries in the dataset have the expected version.

    Args:
        dataset: List of dataset entries
        expected_version: Expected version string
        version_field: Field name containing the version (default: "version")

    Returns:
        True if all entries have the expected version, False otherwise
    """
    for _i, entry in enumerate(dataset):
        if not isinstance(entry, dict):
            continue
        actual_version = entry.get(version_field)
        if actual_version != expected_version:
            return False
    return True


__all__: list[str] = [
    "load_jsonl_dataset",
    "iterate_jsonl_dataset",
    "get_dataset_info",
    "validate_dataset_version"
]
