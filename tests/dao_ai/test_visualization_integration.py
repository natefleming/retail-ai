"""Integration tests for Vega-Lite visualization tool.

These tests validate the visualization tool works end-to-end,
including spec generation and custom_outputs extraction.
They are marked with @pytest.mark.integration.
"""

import json

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from dao_ai.models import _extract_visualizations_from_messages
from dao_ai.tools.visualization import VEGA_LITE_SCHEMA, create_visualization_tool

# ---------------------------------------------------------------------------
# Tool invocation integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_tool_generates_valid_vega_lite_spec() -> None:
    """Verify the tool output is a well-formed Vega-Lite spec."""
    tool = create_visualization_tool(
        name="integration_chart",
        width=600,
        height=400,
    )

    data = json.dumps(
        [
            {"month": "Jan", "sales": 1200},
            {"month": "Feb", "sales": 1500},
            {"month": "Mar", "sales": 900},
        ]
    )

    result: str = tool.invoke(
        {
            "data": data,
            "x_field": "month",
            "y_field": "sales",
            "chart_type": "bar",
            "title": "Monthly Sales",
        }
    )

    spec: dict = json.loads(result)
    assert spec["$schema"] == VEGA_LITE_SCHEMA
    assert spec["mark"] == "bar"
    assert spec["width"] == 600
    assert spec["height"] == 400
    assert spec["title"] == "Monthly Sales"
    assert len(spec["data"]["values"]) == 3


@pytest.mark.integration
def test_all_chart_types_produce_valid_specs() -> None:
    """Ensure every supported chart type yields parseable JSON with $schema."""
    tool = create_visualization_tool()
    data = json.dumps([{"x": "a", "y": 1}, {"x": "b", "y": 2}])

    for chart_type in ("bar", "line", "scatter", "area", "arc", "heatmap"):
        result = tool.invoke(
            {
                "data": data,
                "x_field": "x",
                "y_field": "y",
                "chart_type": chart_type,
            }
        )
        spec = json.loads(result)
        assert spec["$schema"] == VEGA_LITE_SCHEMA, f"Failed for {chart_type}"


@pytest.mark.integration
def test_tool_reusable_across_invocations() -> None:
    """A single tool instance can be called multiple times."""
    tool = create_visualization_tool()
    data = json.dumps([{"cat": "A", "val": 10}])

    for _ in range(3):
        result = tool.invoke(
            {
                "data": data,
                "x_field": "cat",
                "y_field": "val",
            }
        )
        spec = json.loads(result)
        assert spec["$schema"] == VEGA_LITE_SCHEMA


# ---------------------------------------------------------------------------
# custom_outputs extraction integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_extract_visualizations_from_tool_messages() -> None:
    """Verify _extract_visualizations_from_messages finds specs in ToolMessages."""
    spec = json.dumps(
        {
            "$schema": VEGA_LITE_SCHEMA,
            "data": {"values": [{"x": 1}]},
            "mark": "bar",
            "encoding": {},
        }
    )

    messages = [
        HumanMessage(content="show me a chart"),
        ToolMessage(content=spec, tool_call_id="tc_1"),
    ]

    visualizations = _extract_visualizations_from_messages(messages, "msg_abc123")

    assert len(visualizations) == 1
    assert visualizations[0]["message_id"] == "msg_abc123"
    assert visualizations[0]["spec"]["$schema"] == VEGA_LITE_SCHEMA


@pytest.mark.integration
def test_extract_ignores_non_vegalite_tool_messages() -> None:
    """Non Vega-Lite tool outputs should be ignored."""
    messages = [
        ToolMessage(content="plain text result", tool_call_id="tc_1"),
        ToolMessage(content='{"key": "value"}', tool_call_id="tc_2"),
    ]

    visualizations = _extract_visualizations_from_messages(messages, "msg_xyz")
    assert len(visualizations) == 0


@pytest.mark.integration
def test_extract_handles_multiple_specs() -> None:
    """Multiple Vega-Lite tool outputs in one turn should all be captured."""
    spec_template = {
        "$schema": VEGA_LITE_SCHEMA,
        "data": {"values": []},
        "mark": "bar",
        "encoding": {},
    }

    messages = [
        ToolMessage(content=json.dumps(spec_template), tool_call_id="tc_1"),
        ToolMessage(content="not a spec", tool_call_id="tc_2"),
        ToolMessage(content=json.dumps(spec_template), tool_call_id="tc_3"),
    ]

    visualizations = _extract_visualizations_from_messages(messages, "msg_multi")
    assert len(visualizations) == 2
    assert all(v["message_id"] == "msg_multi" for v in visualizations)


@pytest.mark.integration
def test_end_to_end_tool_to_extraction() -> None:
    """Full round-trip: create tool, invoke, wrap in ToolMessage, extract."""
    tool = create_visualization_tool()
    data = json.dumps(
        [
            {"product": "Widget", "sales": 50},
            {"product": "Gadget", "sales": 75},
        ]
    )

    raw_result: str = tool.invoke(
        {
            "data": data,
            "x_field": "product",
            "y_field": "sales",
            "chart_type": "bar",
            "title": "Product Sales",
        }
    )

    messages = [
        ToolMessage(content=raw_result, tool_call_id="tc_viz"),
    ]

    message_id = "msg_e2e_123"
    visualizations = _extract_visualizations_from_messages(messages, message_id)

    assert len(visualizations) == 1
    viz = visualizations[0]
    assert viz["message_id"] == message_id
    assert viz["spec"]["title"] == "Product Sales"
    assert viz["spec"]["mark"] == "bar"
    assert len(viz["spec"]["data"]["values"]) == 2
