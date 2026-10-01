"""Node: Feedback Learning - collect feedback and persist failure cases

Status
------
Phase 1: Basic implementation for collecting feedback and persisting failure cases.

See docs/ROADMAP.md for the phase in which this module is implemented,
docs/ARCHITECTURE.md for its role in the task pipeline, and ADR.md for the
design decisions behind it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.exceptions import FeedbackError
from src.core.state import AgentState


class FeedbackLearningNode:
    """
    Collects feedback from the system execution and persists failure cases
    for future improvement.
    """

    def __init__(self, feedback_dir: Optional[Path] = None):
        """
        Initialize the feedback learning node.

        Args:
            feedback_dir: Directory to store feedback and failure cases.
                         If None, defaults to 'data/feedback'
        """
        if feedback_dir is None:
            feedback_dir = Path("data/feedback")
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        # File for storing feedback
        self.feedback_file = self.feedback_dir / "feedback.jsonl"
        # File for storing failure cases
        self.failure_file = self.feedback_dir / "failures.jsonl"

    def run(self, state: AgentState) -> AgentState:
        """
        Process the agent state to extract feedback and failure information.

        This node should be placed at the end of the graph to capture the final
        outcome of a task execution.

        Args:
            state: The current agent state

        Returns:
            The same state (this node doesn't modify the state for downstream
            processing, but it persists feedback for future use)
        """
        try:
            self._process_feedback(state)
        except Exception as e:
            # Don't let feedback processing break the main flow
            # In a real system, we might want to log this error
            pass
        return state

    def _process_feedback(self, state: AgentState) -> None:
        """
        Extract feedback from the state and persist it.

        Args:
            state: The agent state after task execution
        """
        # Prepare feedback entry
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": state.get("request_id", "unknown"),
            "user_query": state.get("user_query", ""),
            "intent": state.get("intent", ""),
            "route": state.get("route", ""),
            "selected_tools": state.get("tools_selected", []),
            "success": self._determine_success(state),
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "claims": state.get("claims", []),
            "evidence": state.get("evidence", []),
            # Add any error information
            "errors": state.get("errors", []),
            # Add latency information if available
            "total_latency_ms": state.get("latency", {}).get("total_ms", 0.0),
        }

        # Persist feedback
        self._append_to_jsonl(self.feedback_file, feedback_entry)

        # If the task failed, also persist as a failure case
        if not feedback_entry["success"]:
            failure_entry = {
                **feedback_entry,
                "failure_type": self._classify_failure(state),
                "failure_reason": self._extract_failure_reason(state),
            }
            self._append_to_jsonl(self.failure_file, failure_entry)

    def _determine_success(self, state: AgentState) -> bool:
        """
        Determine if the task was successful based on the state.

        Args:
            state: The agent state

        Returns:
            True if the task succeeded, False otherwise
        """
        # Check if we have an answer and no critical errors
        answer = state.get("answer", "")
        if not answer:
            return False

        # Check for critical errors that would indicate failure
        errors = state.get("errors", [])
        for error in errors:
            if error.get("stage") in ["answer_guard", "verification"]:
                # If the answer guard or verification marked it as unsupported/critical
                if error.get("message", "").lower().find("unsupported") != -1:
                    return False

        # Check if the route was clarification (might indicate inability to answer)
        if state.get("route") == "clarification":
            # Clarification might be success if it was honest, but for simplicity
            # we'll treat it as not a full success
            # In a more sophisticated system, we'd check if clarification was appropriate
            return False

        return True

    def _classify_failure(self, state: AgentState) -> str:
        """
        Classify the type of failure based on the state.

        Args:
            state: The agent state

        Returns:
            A string representing the failure type
        """
        errors = state.get("errors", [])
        if not errors:
            return "unknown"

        # Check the first error stage
        first_error_stage = errors[0].get("stage", "") if errors else ""
        if first_error_stage in ["intent_understanding", "routing"]:
            return "routing_or_intent_error"
        elif first_error_stage in ["sql_execution", "sql_guard"]:
            return "sql_error"
        elif first_error_stage in ["rag_retrieval", "evidence_collection"]:
            return "retrieval_error"
        elif first_error_stage in ["claim_extraction", "verification", "answer_guard"]:
            return "verification_error"
        elif first_error_stage in ["multi_tool_execution"]:
            return "multi_tool_error"
        else:
            return "other_error"

    def _extract_failure_reason(self, state: AgentState) -> str:
        """
        Extract a human-readable reason for the failure.

        Args:
            state: The agent state

        Returns:
            A string describing the failure reason
        """
        errors = state.get("errors", [])
        if errors:
            # Return the first error message
            return errors[0].get("message", "Unknown error")
        return "No error information available"

    def _append_to_jsonl(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Append a dictionary as a JSON line to a file.

        Args:
            file_path: Path to the JSONL file
            data: Dictionary to append
        """
        try:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            raise FeedbackError(f"Failed to write to {file_path}: {e}") from e


__all__: list[str] = ["FeedbackLearningNode"]
