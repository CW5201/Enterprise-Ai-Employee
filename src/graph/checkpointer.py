"""Checkpointer - task state persistence for the graph

Status
------
Phase 1: Basic file-based checkpointer implementation.

See docs/ROADMAP.md for the phase in which this module is implemented,
docs/ARCHITECTURE.md for its role in the task pipeline, and ADR.md for the
design decisions behind it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.core.exceptions import CheckpointerError


class Checkpointer:
    """
    A simple file-based checkpointer for LangGraph.

    Each checkpoint is saved as a JSON file in a specified directory.
    The checkpoint ID is used as the filename.
    """

    def __init__(self, checkpoint_dir: str | Path):
        """
        Initialize the checkpointer.

        Args:
            checkpoint_dir: Directory where checkpoints will be saved
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """
        Get the latest checkpoint for the given config.

        Args:
            config: The configuration (usually contains thread_id, checkpoint_id, etc.)

        Returns:
            The checkpoint data if found, None otherwise
        """
        try:
            checkpoint_id = self._get_checkpoint_id(config)
            if not checkpoint_id:
                return None

            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            if not checkpoint_file.exists():
                return None

            with checkpoint_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise CheckpointerError(f"Failed to load checkpoint: {e}") from e

    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Save a checkpoint.

        Args:
            config: The configuration
            checkpoint: The checkpoint data to save
            metadata: Metadata about the checkpoint

        Returns:
            The checkpoint data that was saved (may be modified)
        """
        try:
            checkpoint_id = self._get_checkpoint_id(config)
            if not checkpoint_id:
                # Generate a checkpoint ID if not provided
                import uuid
                checkpoint_id = str(uuid.uuid4())
                config = {**config, "checkpoint_id": checkpoint_id}

            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            with checkpoint_file.open("w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)

            # Return the config with the checkpoint ID
            return {**config, "checkpoint_id": checkpoint_id}
        except Exception as e:
            raise CheckpointerError(f"Failed to save checkpoint: {e}") from e

    def list(self, config: dict[str, Any] | None) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
        """
        List existing checkpoints.

        Args:
            config: Optional configuration to filter by (e.g., thread_id)

        Yields:
            Tuples of (checkpoint, metadata, config)
        """
        try:
            # For simplicity, we'll list all checkpoint files
            # In a more advanced implementation, we would filter by config
            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                try:
                    with checkpoint_file.open("r", encoding="utf-8") as f:
                        checkpoint_data = json.load(f)

                    # Extract metadata and config from the checkpoint data
                    # This is a simplified implementation
                    metadata = checkpoint_data.get("metadata", {})
                    config_data = checkpoint_data.get("config", {})

                    yield checkpoint_data, metadata, config_data
                except Exception:
                    # Skip corrupted files
                    continue
        except Exception as e:
            raise CheckpointerError(f"Failed to list checkpoints: {e}") from e

    def _get_checkpoint_id(self, config: dict[str, Any]) -> str | None:
        """
        Extract the checkpoint ID from config.

        Args:
            config: The configuration

        Returns:
            The checkpoint ID if present, None otherwise
        """
        # Look for checkpoint_id in various places
        if "checkpoint_id" in config:
            return config["checkpoint_id"]
        if "configurable" in config and "checkpoint_id" in config["configurable"]:
            return config["configurable"]["checkpoint_id"]
        return None


# For compatibility with LangGraph's expected interface
# We'll also provide a simple in-memory checkpointer option


class InMemoryCheckpointer:
    """
    A simple in-memory checkpointer for testing.
    """

    def __init__(self):
        self.checkpoints: dict[str, dict[str, Any]] = {}

    def get(self, config: dict[str, Any]) -> dict[str, Any] | None:
        checkpoint_id = self._get_checkpoint_id(config)
        if checkpoint_id and checkpoint_id in self.checkpoints:
            return self.checkpoints[checkpoint_id]
        return None

    def put(
        self,
        config: dict[str, Any],
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        checkpoint_id = self._get_checkpoint_id(config)
        if not checkpoint_id:
            import uuid
            checkpoint_id = str(uuid.uuid4())
            config = {**config, "checkpoint_id": checkpoint_id}

        self.checkpoints[checkpoint_id] = checkpoint
        return {**config, "checkpoint_id": checkpoint_id}

    def list(self, config: dict[str, Any] | None) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
        for _checkpoint_id, checkpoint_data in self.checkpoints.items():
            metadata = checkpoint_data.get("metadata", {})
            config_data = checkpoint_data.get("config", {})
            yield checkpoint_data, metadata, config_data

    def _get_checkpoint_id(self, config: dict[str, Any]) -> str | None:
        if "checkpoint_id" in config:
            return config["checkpoint_id"]
        if "configurable" in config and "checkpoint_id" in config["configurable"]:
            return config["configurable"]["checkpoint_id"]
        return None


__all__: list[str] = [
    "Checkpointer",
    "InMemoryCheckpointer"
]
