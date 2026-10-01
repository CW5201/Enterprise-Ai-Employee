"""Node: Tool Invocation - tool selection and execution via the registry

Status
------
Phase 1: Basic implementation for tool selection and execution.

See docs/ROADMAP.md for the phase in which this module is implemented,
docs/ARCHITECTURE.md for its role in the task pipeline, and ADR.md for the
design decisions behind it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.exceptions import ToolError
from src.core.observability import observe
from src.core.state import AgentState
from src.core.tool_registry import build_registry
from src.tools.base import ToolResult


class ToolInvocationNode:
    """
    Invokes tools based on the tool registry and routing decisions.

    This node is responsible for:
    1. Looking up tools in the registry
    2. Validating tool inputs
    3. Executing the tool
    4. Recording the tool call in the state
    5. Handling tool execution errors
    """

    def __init__(self):
        """Initialize the tool invocation node with the tool registry."""
        self.registry = build_registry()

    @observe("tool_invocation")
    def run(self, state: AgentState) -> AgentState:
        """
        Process the agent state to invoke tools based on routing decisions.

        Args:
            state: The current agent state containing routing information

        Returns:
            Updated agent state with tool call results
        """
        # Make a copy of the state to avoid modifying the original during processing
        new_state = dict(state)

        # Get the selected tools from routing decision or tools_selected
        selected_tools = state.get("tools_selected", [])
        if not selected_tools:
            # Try to get from routing decision if available
            routing_decision = state.get("routing_decision", {})
            selected_tools = routing_decision.get("tools", [])

        # If still no tools selected, check the route field
        if not selected_tools:
            route = state.get("route", "")
            if route == "sql":
                selected_tools = ["sql"]
            elif route == "rag":
                selected_tools = ["rag"]
            elif route == "kg":
                selected_tools = ["kg"]
            # Note: clarification route doesn't invoke tools

        # Initialize tool_calls and errors if not present
        if "tool_calls" not in new_state:
            new_state["tool_calls"] = []
        if "errors" not in new_state:
            new_state["errors"] = []

        # Invoke each selected tool
        for tool_name in selected_tools:
            try:
                self._invoke_tool(new_state, tool_name)
            except Exception as e:
                # Record the error but continue with other tools
                error_record = {
                    "stage": "tool_invocation",
                    "message": f"Failed to invoke tool {tool_name}: {str(e)}",
                    "details": {"tool_name": tool_name}
                }
                new_state["errors"] = new_state.get("errors", []) + [error_record]

        return new_state

    def _invoke_tool(self, state: AgentState, tool_name: str) -> None:
        """
        Invoke a single tool and record the result.

        Args:
            state: The agent state to update
            tool_name: The name of the tool to invoke
        """
        # Get the tool from the registry
        tool = self.registry.get_tool(tool_name)
        if not tool:
            raise ToolError(f"Tool {tool_name} not found in registry")

        # Prepare tool inputs based on the tool's input schema
        tool_inputs = self._prepare_tool_inputs(state, tool)

        # Execute the tool
        try:
            result: ToolResult = tool.run(**tool_inputs)

            # Record the tool call
            tool_call_record = {
                "tool": tool_name,
                "ok": result.ok,
                "input": tool_inputs,
                "error": result.error,
                "evidence_ref": result.evidence_ref,
                "metadata": result.metadata or {},
            }

            # Add to tool_calls
            state["tool_calls"] = state.get("tool_calls", []) + [tool_call_record]

            # If the tool produced evidence, add it to the evidence list
            if result.data:
                evidence_item = {
                    "evidence_id": f"{tool_name}-{len(state.get('evidence', []))}",
                    "source_type": tool_name,
                    "source_ref": result.evidence_ref or f"{tool_name}:{tool_inputs}",
                    "content": str(result.data)[:500],  # Truncate for storage
                    "payload": result.data,
                    "metadata": result.metadata or {},
                }
                state["evidence"] = state.get("evidence", []) + [evidence_item]

        except Exception as e:
            # If the tool execution fails, record the error
            error_record = {
                "stage": f"tool_{tool_name}",
                "message": f"Tool {tool_name} execution failed: {str(e)}",
                "details": {"tool_name": tool_name, "inputs": tool_inputs}
            }
            state["errors"] = state.get("errors", []) + [error_record]

            # Still record the tool call as failed
            tool_call_record = {
                "tool": tool_name,
                "ok": False,
                "input": tool_inputs,
                "error": str(e),
                "evidence_ref": None,
                "metadata": {},
            }
            state["tool_calls"] = state.get("tool_calls", []) + [tool_call_record]

    def _prepare_tool_inputs(self, state: AgentState, tool: Any) -> Dict[str, Any]:
        """
        Prepare inputs for a tool based on its input schema and current state.

        Args:
            state: The current agent state
            tool: The tool instance

        Returns:
            Dictionary of inputs for the tool
        """
        # Get the tool's input schema
        input_schema = getattr(tool, "input_schema", {})

        # Map state fields to tool inputs based on common patterns
        inputs = {}

        # Common mappings
        if "question" in input_schema:
            inputs["question"] = state.get("user_query", "")
        if "query" in input_schema:
            inputs["query"] = state.get("user_query", "")
        if "sql" in input_schema:
            inputs["sql"] = state.get("sql", "")
        if "template_id" in input_schema:
            # This would come from routing decision or state
            inputs["template_id"] = state.get("template_id", "customer_orders")  # Default
        if "parameters" in input_schema:
            inputs["parameters"] = state.get("tool_parameters", {})
        if "top_k" in input_schema:
            inputs["top_k"] = state.get("top_k", 5)
        if "retrieval_mode" in input_schema:
            inputs["retrieval_mode"] = state.get("retrieval_mode", "hybrid")
        if "operation" in input_schema:
            inputs["operation"] = state.get("analysis_operation", "describe")
        if "dataframe_ref" in input_schema:
            inputs["dataframe_ref"] = state.get("last_sql_result", None)
        if "chart_type" in input_schema:
            inputs["chart_type"] = state.get("chart_type", "bar")
        if "mapping" in input_schema:
            inputs["mapping"] = state.get("chart_mapping", {})
        if "answer" in input_schema:
            inputs["answer"] = state.get("answer", "")
        if "evidence" in input_schema:
            inputs["evidence"] = state.get("evidence", [])
        if "charts" in input_schema:
            inputs["charts"] = state.get("charts", [])

        # Filter to only include inputs that are in the schema
        filtered_inputs = {
            k: v for k, v in inputs.items() if k in input_schema
        }

        return filtered_inputs


__all__: list[str] = ["ToolInvocationNode"]
