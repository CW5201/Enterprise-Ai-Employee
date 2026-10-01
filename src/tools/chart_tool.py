"""Tool: Chart - build ECharts-ready data payloads

Status
------
Phase 1: Basic implementation for building ECharts-ready data payloads.

See docs/ROADMAP.md for the phase in which this module is implemented,
docs/ARCHITECTURE.md for its role in the task pipeline, and ADR.md for the
design decisions behind it.
"""

from __future__ import annotations

from typing import Any


class ChartTool:
    """Tool for building ECharts-ready data payloads from tabular data.

    This tool converts structured data (like SQL query results) into
    format suitable for ECharts visualization.
    """

    name: str = "chart"
    description: str = "Build an ECharts-ready data payload from a result table."
    timeout: int = 10

    def __init__(self):
        """Initialize the chart tool."""
        pass

    def run(self, *, chart_type: str, dataframe_ref: Any, mapping: dict[str, Any]) -> dict[str, Any]:
        """
        Build an ECharts-ready data payload.

        Args:
            chart_type: Type of chart to generate (e.g., 'bar', 'line', 'pie', 'scatter')
            dataframe_ref: Reference to the data (typically SQL query results)
            mapping: Mapping configuration for how to map data to chart axes/series

        Returns:
            Dictionary containing the echarts_option and metadata
        """
        try:
            option = self._build_chart_option(chart_type, dataframe_ref, mapping)

            return {
                "tool": self.name,
                "ok": True,
                "data": option,
                "evidence_ref": f"chart:{chart_type}",
                "metadata": {
                    "chart_type": chart_type,
                    "data_shape": self._get_data_shape(dataframe_ref)
                }
            }
        except Exception as e:
            return {
                "tool": self.name,
                "ok": False,
                "error": f"Chart generation failed: {str(e)}",
                "data": None,
                "evidence_ref": None,
                "metadata": {}
            }

    def _build_chart_option(self, chart_type: str, dataframe_ref: Any, mapping: dict[str, Any]) -> dict[str, Any]:
        """Build the ECharts option based on chart type and data."""
        # Handle different dataframe formats
        data = self._extract_data(dataframe_ref)

        if not data:
            # Return empty chart option
            return self._get_empty_option(chart_type)

        # Build option based on chart type
        if chart_type == "bar":
            return self._build_bar_chart(data, mapping)
        elif chart_type == "line":
            return self._build_line_chart(data, mapping)
        elif chart_type == "pie":
            return self._build_pie_chart(data, mapping)
        elif chart_type == "scatter":
            return self._build_scatter_chart(data, mapping)
        else:
            # Default to bar chart for unknown types
            return self._build_bar_chart(data, mapping)

    def _extract_data(self, dataframe_ref: Any) -> list:
        """Extract data from various reference formats."""
        if isinstance(dataframe_ref, list):
            return dataframe_ref
        elif isinstance(dataframe_ref, dict):
            # Handle common SQL result format
            if "rows" in dataframe_ref and "columns" in dataframe_ref:
                rows = dataframe_ref["rows"]
                columns = dataframe_ref["columns"]
                # Convert rows to list of dicts
                return [
                    dict(zip(columns, row, strict=False))
                    for row in rows
                ]
            elif "data" in dataframe_ref:
                return dataframe_ref["data"]
            else:
                # Assume it's already a list of dicts or similar
                return list(dataframe_ref.values()) if hasattr(dataframe_ref, 'values') else []
        else:
            # Try to convert to list
            try:
                return list(dataframe_ref)
            except Exception:
                return []

    def _get_data_shape(self, dataframe_ref: Any) -> dict[str, Any]:
        """Get shape information about the data."""
        data = self._extract_data(dataframe_ref)
        if not data:
            return {"rows": 0, "columns": 0}

        if isinstance(data[0], dict):
            columns = list(data[0].keys())
            return {"rows": len(data), "columns": len(columns), "column_names": columns}
        else:
            return {"rows": len(data), "columns": 1 if data and not isinstance(data[0], (list, dict)) else len(data[0]) if isinstance(data[0], list) else 0}

    def _get_empty_option(self, chart_type: str) -> dict[str, Any]:
        """Return an empty chart option."""
        return {
            "title": {"text": "No Data Available"},
            "tooltip": {},
            "legend": {},
            "xAxis": {},
            "yAxis": {},
            "series": []
        }

    def _build_bar_chart(self, data: list, mapping: dict[str, Any]) -> dict[str, Any]:
        """Build a bar chart option."""
        if not data:
            return self._get_empty_option("bar")

        # Extract x and y data based on mapping
        x_field = mapping.get("x", list(data[0].keys())[0] if data and isinstance(data[0], dict) else 0)
        y_field = mapping.get("y", list(data[0].keys())[1] if len(data) > 0 and isinstance(data[0], dict) and len(data[0]) > 1 else (list(data[0].keys())[0] if data and isinstance(data[0], dict) else 0))

        # Handle case where fields might not exist
        x_data = []
        y_data = []

        for item in data:
            if isinstance(item, dict):
                x_data.append(item.get(x_field, ""))
                y_data.append(item.get(y_field, 0))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                x_data.append(item[0])
                y_data.append(item[1])
            else:
                x_data.append(str(item))
                y_data.append(0)

        option = {
            "title": {"text": mapping.get("title", "Bar Chart")},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": [mapping.get("series_name", "Value")]},
            "toolbox": {"show": True},
            "xAxis": {
                "type": "category",
                "data": x_data,
                "name": mapping.get("x_axis_name", ""),
                "axisLabel": {"rotate": 45 if (x_data and len(str(x_data[0])) > 5) else 0},
            },
            "yAxis": {
                "type": "value",
                "name": mapping.get("y_axis_name", "")
            },
            "series": [{
                "name": mapping.get("series_name", "Value"),
                "type": "bar",
                "data": y_data,
                "label": {
                    "show": mapping.get("show_label", False),
                    "position": "top"
                }
            }]
        }

        return option

    def _build_line_chart(self, data: list, mapping: dict[str, Any]) -> dict[str, Any]:
        """Build a line chart option."""
        if not data:
            return self._get_empty_option("line")

        # Extract x and y data based on mapping
        x_field = mapping.get("x", list(data[0].keys())[0] if data and isinstance(data[0], dict) else 0)
        y_field = mapping.get("y", list(data[0].keys())[1] if len(data) > 0 and isinstance(data[0], dict) and len(data[0]) > 1 else (list(data[0].keys())[0] if data and isinstance(data[0], dict) else 0))

        # Handle case where fields might not exist
        x_data = []
        y_data = []

        for item in data:
            if isinstance(item, dict):
                x_data.append(item.get(x_field, ""))
                y_data.append(item.get(y_field, 0))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                x_data.append(item[0])
                y_data.append(item[1])
            else:
                x_data.append(str(item))
                y_data.append(0)

        option = {
            "title": {"text": mapping.get("title", "Line Chart")},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": [mapping.get("series_name", "Value")]},
            "toolbox": {"show": True},
            "xAxis": {
                "type": "category",
                "data": x_data,
                "name": mapping.get("x_axis_name", ""),
                "axisLabel": {"rotate": 45 if (x_data and len(str(x_data[0])) > 5) else 0},
            },
            "yAxis": {
                "type": "value",
                "name": mapping.get("y_axis_name", "")
            },
            "series": [{
                "name": mapping.get("series_name", "Value"),
                "type": "line",
                "data": y_data,
                "smooth": mapping.get("smooth", True),
                "label": {
                    "show": mapping.get("show_label", False),
                    "position": "top"
                }
            }]
        }

        return option

    def _build_pie_chart(self, data: list, mapping: dict[str, Any]) -> dict[str, Any]:
        """Build a pie chart option."""
        if not data:
            return self._get_empty_option("pie")

        # Extract name and value data based on mapping
        name_field = mapping.get("name", list(data[0].keys())[0] if data and isinstance(data[0], dict) else 0)
        value_field = mapping.get("value", list(data[0].keys())[1] if len(data) > 0 and isinstance(data[0], dict) and len(data[0]) > 1 else (list(data[0].keys())[0] if data and isinstance(data[0], dict) else 0))

        # Handle case where fields might not exist
        name_data = []
        value_data = []

        for item in data:
            if isinstance(item, dict):
                name_data.append(str(item.get(name_field, "")))
                value_data.append(item.get(value_field, 0))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                name_data.append(str(item[0]))
                value_data.append(item[1])
            else:
                name_data.append(str(item))
                value_data.append(1)

        # Filter out zero or negative values for pie chart
        filtered_data = [(n, v) for n, v in zip(name_data, value_data, strict=False) if v > 0]
        if not filtered_data:
            return self._get_empty_option("pie")

        name_data, value_data = zip(*filtered_data, strict=False) if filtered_data else ([], [])

        option = {
            "title": {"text": mapping.get("title", "Pie Chart")},
            "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b}: {c} ({d}%)"},
            "legend": {
                "orient": "vertical",
                "left": "left",
                "data": list(name_data)
            },
            "toolbox": {"show": True},
            "series": [{
                "name": mapping.get("series_name", ""),
                "type": "pie",
                "radius": ["40%", "70%"],
                "avoidLabelOverlap": False,
                "label": {
                    "show": False,
                    "position": "center"
                },
                "emphasis": {
                    "label": {
                        "show": True,
                        "fontSize": "18",
                        "fontWeight": "bold"
                    }
                },
                "labelLine": {
                    "show": False
                },
                "data": [
                    {"value": v, "name": n}
                    for v, n in zip(value_data, name_data, strict=False)
                ]
            }]
        }

        return option

    def _build_scatter_chart(self, data: list, mapping: dict[str, Any]) -> dict[str, Any]:
        """Build a scatter chart option."""
        if not data:
            return self._get_empty_option("scatter")

        # Extract x and y data based on mapping
        x_field = mapping.get("x", list(data[0].keys())[0] if data and isinstance(data[0], dict) else 0)
        y_field = mapping.get("y", list(data[0].keys())[1] if len(data) > 0 and isinstance(data[0], dict) and len(data[0]) > 1 else (list(data[0].keys())[0] if data and isinstance(data[0], dict) else 0))

        # Handle case where fields might not exist
        x_data = []
        y_data = []

        for item in data:
            if isinstance(item, dict):
                x_val = item.get(x_field, 0)
                y_val = item.get(y_field, 0)
                # Only include numeric values
                try:
                    x_data.append(float(x_val))
                    y_data.append(float(y_val))
                except (ValueError, TypeError):
                    pass  # Skip non-numeric data
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    x_data.append(float(item[0]))
                    y_data.append(float(item[1]))
                except (ValueError, TypeError):
                    pass  # Skip non-numeric data

        if not x_data:
            return self._get_empty_option("scatter")

        option = {
            "title": {"text": mapping.get("title", "Scatter Chart")},
            "tooltip": {
                "trigger": "item",
                "formatter": "{a} <br/>{b} : {c}"
            },
            "legend": {"data": [mapping.get("series_name", "Value")]},
            "toolbox": {"show": True},
            "xAxis": {
                "type": "value",
                "name": mapping.get("x_axis_name", ""),
                "axisLabel": {
                    "formatter": "{value}"
                }
            },
            "yAxis": {
                "type": "value",
                "name": mapping.get("y_axis_name", ""),
                "axisLabel": {
                    "formatter": "{value}"
                }
            },
            "series": [{
                "name": mapping.get("series_name", "Value"),
                "type": "scatter",
                "data": [[x, y] for x, y in zip(x_data, y_data, strict=False)],
                "symbolSize": mapping.get("symbol_size", 10)
            }]
        }

        return option


__all__: list[str] = ["ChartTool"]
