"""
Vega-Lite visualization tool for generating chart specs from structured data.

This module provides a factory function for creating tools that generate
Vega-Lite JSON specifications from data. The specs are portable and can
be rendered by any Vega-Lite compatible client.

Specs are delivered to clients via custom_outputs.visualizations (see models.py).
"""

import json
from datetime import datetime
from typing import Any, Callable, Literal

from langchain.tools import ToolRuntime, tool
from loguru import logger

from dao_ai.state import Context

VEGA_LITE_SCHEMA: str = "https://vega.github.io/schema/vega-lite/v6.json"

ChartType = Literal["bar", "line", "scatter", "area", "arc", "heatmap"]
AggregateType = Literal["sum", "mean", "count", "min", "max", "median"]
SortOrder = Literal["ascending", "descending"]

CHART_TYPE_TO_MARK: dict[str, str | dict[str, Any]] = {
    "bar": "bar",
    "line": {"type": "line", "point": True},
    "scatter": "point",
    "area": {"type": "area", "opacity": 0.7},
    "arc": "arc",
    "heatmap": "rect",
}


def _infer_field_type(value: Any) -> str:
    """Infer a Vega-Lite field type from a Python value."""
    if isinstance(value, bool):
        return "nominal"
    if isinstance(value, (int, float)):
        return "quantitative"
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return "temporal"
        except (ValueError, AttributeError):
            pass
    return "nominal"


def _build_encoding(
    x_field: str,
    y_field: str,
    x_type: str,
    y_type: str,
    chart_type: ChartType,
    color_field: str | None,
    color_type: str | None,
    aggregate: AggregateType | None,
    sort: SortOrder | None,
    color_scheme: str,
) -> dict[str, Any]:
    """Build the Vega-Lite encoding specification."""
    x_encoding: dict[str, Any] = {"field": x_field, "type": x_type}
    y_encoding: dict[str, Any] = {"field": y_field, "type": y_type}

    if sort:
        x_encoding["sort"] = sort

    if aggregate:
        y_encoding["aggregate"] = aggregate

    if chart_type == "arc":
        encoding: dict[str, Any] = {
            "theta": {"field": y_field, "type": y_type},
            "color": {
                "field": x_field,
                "type": x_type,
                "scale": {"scheme": color_scheme},
            },
        }
        if aggregate:
            encoding["theta"]["aggregate"] = aggregate
        encoding["tooltip"] = [
            {"field": x_field, "type": x_type},
            {"field": y_field, "type": y_type},
        ]
        return encoding

    if chart_type == "heatmap":
        encoding = {
            "x": x_encoding,
            "y": y_encoding,
            "color": {
                "field": y_field,
                "type": y_type,
                "scale": {"scheme": color_scheme},
            },
        }
        if aggregate:
            encoding["color"]["aggregate"] = aggregate
        encoding["tooltip"] = [
            {"field": x_field, "type": x_type},
            {"field": y_field, "type": y_type},
        ]
        return encoding

    encoding = {"x": x_encoding, "y": y_encoding}

    if color_field and color_type:
        encoding["color"] = {
            "field": color_field,
            "type": color_type,
            "scale": {"scheme": color_scheme},
        }

    encoding["tooltip"] = [
        {"field": x_field, "type": x_type},
        {"field": y_field, "type": y_type},
    ]
    if color_field:
        encoding["tooltip"].append(
            {"field": color_field, "type": color_type or "nominal"}
        )

    return encoding


def create_visualization_tool(
    name: str = "visualization_tool",
    description: str | None = None,
    default_chart_type: ChartType = "bar",
    width: int | str = "container",
    height: int = 400,
    color_scheme: str = "tableau10",
) -> Callable:
    """
    Create a tool for generating Vega-Lite visualization specs from data.

    This factory function creates a tool that accepts structured data and
    chart parameters from the LLM and produces a portable Vega-Lite JSON
    specification.

    Args:
        name: Tool name visible to the LLM. Defaults to "visualization_tool".
        description: Tool description for the LLM. Auto-generated if None.
        default_chart_type: Default chart type when not specified at runtime.
        width: Chart width. "container" for responsive, or an int for fixed.
        height: Chart height in pixels.
        color_scheme: Vega-Lite named color scheme.

    Returns:
        A LangChain tool that generates Vega-Lite specs from structured data.

    Example:
        ```python
        tool = create_visualization_tool(width=800, height=500)
        result = tool.invoke({
            "data": '[{"region": "North", "revenue": 100}]',
            "x_field": "region",
            "y_field": "revenue",
        })
        ```
    """
    tool_name: str = name
    tool_description: str = description or (
        "Generate a Vega-Lite visualization spec from structured data. "
        "Supported chart types: bar, line, scatter, area, arc (pie), heatmap. "
        "Pass data as a JSON array string and specify x/y field names."
    )

    logger.debug(
        "Creating visualization tool",
        tool_name=tool_name,
        default_chart_type=default_chart_type,
        width=width,
        height=height,
        color_scheme=color_scheme,
    )

    @tool(name_or_callable=tool_name, description=tool_description)
    def visualization_tool(
        data: str,
        x_field: str,
        y_field: str,
        chart_type: ChartType | None = None,
        color_field: str | None = None,
        title: str | None = None,
        aggregate: AggregateType | None = None,
        sort: SortOrder | None = None,
        runtime: ToolRuntime[Context] = None,
    ) -> str:
        """
        Generate a Vega-Lite visualization spec from structured data.

        Args:
            data: JSON array string of data records.
            x_field: Field name for the x-axis (or category for arc charts).
            y_field: Field name for the y-axis (or value for arc charts).
            chart_type: Chart type override. Uses factory default if not specified.
            color_field: Optional field for color/grouping encoding.
            title: Optional chart title.
            aggregate: Optional aggregation operation for the y-axis.
            sort: Optional sort order for the x-axis.

        Returns:
            A JSON string containing the Vega-Lite specification.
        """
        resolved_chart_type: ChartType = chart_type or default_chart_type

        logger.info(
            "Generating visualization spec",
            tool_name=tool_name,
            chart_type=resolved_chart_type,
            x_field=x_field,
            y_field=y_field,
            color_field=color_field,
        )

        # Parse and validate input data
        try:
            records: list[dict[str, Any]] = json.loads(data)
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON data for visualization", error=str(e))
            return f"Error: Invalid JSON data - {e}"

        if not records or not isinstance(records, list):
            return "Error: Data must be a non-empty JSON array of records."

        if not isinstance(records[0], dict):
            return "Error: Each record in the data array must be a JSON object."

        # Infer field types from first record
        first_record: dict[str, Any] = records[0]
        x_type: str = _infer_field_type(first_record.get(x_field))
        y_type: str = _infer_field_type(first_record.get(y_field))
        color_type: str | None = (
            _infer_field_type(first_record.get(color_field)) if color_field else None
        )

        # Build encoding
        encoding: dict[str, Any] = _build_encoding(
            x_field=x_field,
            y_field=y_field,
            x_type=x_type,
            y_type=y_type,
            chart_type=resolved_chart_type,
            color_field=color_field,
            color_type=color_type,
            aggregate=aggregate,
            sort=sort,
            color_scheme=color_scheme,
        )

        # Build spec
        spec: dict[str, Any] = {
            "$schema": VEGA_LITE_SCHEMA,
            "data": {"values": records},
            "mark": CHART_TYPE_TO_MARK[resolved_chart_type],
            "encoding": encoding,
            "width": width,
            "height": height,
        }

        if title:
            spec["title"] = title

        logger.debug(
            "Visualization spec generated",
            tool_name=tool_name,
            chart_type=resolved_chart_type,
            record_count=len(records),
        )

        return json.dumps(spec)

    return visualization_tool
