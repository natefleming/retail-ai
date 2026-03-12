"""Tests for Vega-Lite visualization tool."""

import json

import pytest

from dao_ai.tools.visualization import (
    VEGA_LITE_SCHEMA,
    _build_encoding,
    _infer_field_type,
    create_visualization_tool,
)

# ---------------------------------------------------------------------------
# _infer_field_type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInferFieldType:
    def test_int_is_quantitative(self) -> None:
        assert _infer_field_type(42) == "quantitative"

    def test_float_is_quantitative(self) -> None:
        assert _infer_field_type(3.14) == "quantitative"

    def test_string_is_nominal(self) -> None:
        assert _infer_field_type("hello") == "nominal"

    def test_iso_date_is_temporal(self) -> None:
        assert _infer_field_type("2024-01-15") == "temporal"

    def test_iso_datetime_is_temporal(self) -> None:
        assert _infer_field_type("2024-01-15T10:30:00") == "temporal"

    def test_bool_is_nominal(self) -> None:
        assert _infer_field_type(True) == "nominal"

    def test_none_is_nominal(self) -> None:
        assert _infer_field_type(None) == "nominal"


# ---------------------------------------------------------------------------
# Factory creation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateVisualizationTool:
    def test_default_creation(self) -> None:
        tool = create_visualization_tool()
        assert tool is not None
        assert tool.name == "visualization_tool"
        assert hasattr(tool, "invoke")

    def test_custom_name(self) -> None:
        tool = create_visualization_tool(name="my_chart")
        assert tool.name == "my_chart"

    def test_custom_description(self) -> None:
        desc = "Custom chart tool"
        tool = create_visualization_tool(description=desc)
        assert tool.description == desc

    def test_default_description_mentions_chart_types(self) -> None:
        tool = create_visualization_tool()
        for ct in ("bar", "line", "scatter", "area", "arc", "heatmap"):
            assert ct in tool.description


# ---------------------------------------------------------------------------
# Spec generation
# ---------------------------------------------------------------------------


SAMPLE_DATA: str = json.dumps(
    [
        {"region": "North", "revenue": 100},
        {"region": "South", "revenue": 200},
        {"region": "East", "revenue": 150},
    ]
)


@pytest.mark.unit
class TestSpecGeneration:
    def test_basic_bar_chart(self) -> None:
        tool = create_visualization_tool()
        result: str = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
            }
        )
        spec: dict = json.loads(result)
        assert spec["$schema"] == VEGA_LITE_SCHEMA
        assert spec["mark"] == "bar"
        assert spec["encoding"]["x"]["field"] == "region"
        assert spec["encoding"]["y"]["field"] == "revenue"
        assert spec["data"]["values"] == json.loads(SAMPLE_DATA)

    def test_line_chart(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
                "chart_type": "line",
            }
        )
        spec = json.loads(result)
        assert spec["mark"]["type"] == "line"
        assert spec["mark"]["point"] is True

    def test_scatter_chart(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
                "chart_type": "scatter",
            }
        )
        spec = json.loads(result)
        assert spec["mark"] == "point"

    def test_area_chart(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
                "chart_type": "area",
            }
        )
        spec = json.loads(result)
        assert spec["mark"]["type"] == "area"

    def test_arc_chart(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
                "chart_type": "arc",
            }
        )
        spec = json.loads(result)
        assert spec["mark"] == "arc"
        assert "theta" in spec["encoding"]
        assert "color" in spec["encoding"]

    def test_heatmap_chart(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
                "chart_type": "heatmap",
            }
        )
        spec = json.loads(result)
        assert spec["mark"] == "rect"
        assert spec["encoding"]["color"]["field"] == "revenue"

    def test_title_is_included(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
                "title": "Revenue by Region",
            }
        )
        spec = json.loads(result)
        assert spec["title"] == "Revenue by Region"

    def test_title_omitted_when_not_provided(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
            }
        )
        spec = json.loads(result)
        assert "title" not in spec

    def test_color_field_encoding(self) -> None:
        data = json.dumps(
            [
                {"region": "North", "revenue": 100, "category": "A"},
                {"region": "South", "revenue": 200, "category": "B"},
            ]
        )
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": data,
                "x_field": "region",
                "y_field": "revenue",
                "color_field": "category",
            }
        )
        spec = json.loads(result)
        assert spec["encoding"]["color"]["field"] == "category"

    def test_aggregate(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
                "aggregate": "sum",
            }
        )
        spec = json.loads(result)
        assert spec["encoding"]["y"]["aggregate"] == "sum"

    def test_sort_order(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
                "sort": "descending",
            }
        )
        spec = json.loads(result)
        assert spec["encoding"]["x"]["sort"] == "descending"

    def test_factory_defaults_applied(self) -> None:
        tool = create_visualization_tool(
            width=800, height=600, color_scheme="category20"
        )
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
            }
        )
        spec = json.loads(result)
        assert spec["width"] == 800
        assert spec["height"] == 600

    def test_default_chart_type_override(self) -> None:
        tool = create_visualization_tool(default_chart_type="line")
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
            }
        )
        spec = json.loads(result)
        assert spec["mark"]["type"] == "line"

    def test_tooltip_present(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
            }
        )
        spec = json.loads(result)
        assert "tooltip" in spec["encoding"]

    def test_type_inference_quantitative(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": SAMPLE_DATA,
                "x_field": "region",
                "y_field": "revenue",
            }
        )
        spec = json.loads(result)
        assert spec["encoding"]["y"]["type"] == "quantitative"
        assert spec["encoding"]["x"]["type"] == "nominal"

    def test_type_inference_temporal(self) -> None:
        data = json.dumps(
            [
                {"date": "2024-01-01", "value": 10},
                {"date": "2024-02-01", "value": 20},
            ]
        )
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": data,
                "x_field": "date",
                "y_field": "value",
            }
        )
        spec = json.loads(result)
        assert spec["encoding"]["x"]["type"] == "temporal"


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidation:
    def test_invalid_json_data(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": "not json",
                "x_field": "x",
                "y_field": "y",
            }
        )
        assert "Error" in result

    def test_empty_array(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": "[]",
                "x_field": "x",
                "y_field": "y",
            }
        )
        assert "Error" in result

    def test_non_object_array(self) -> None:
        tool = create_visualization_tool()
        result = tool.invoke(
            {
                "data": "[1, 2, 3]",
                "x_field": "x",
                "y_field": "y",
            }
        )
        assert "Error" in result


# ---------------------------------------------------------------------------
# _build_encoding edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildEncoding:
    def test_arc_has_theta_and_color(self) -> None:
        encoding = _build_encoding(
            x_field="cat",
            y_field="val",
            x_type="nominal",
            y_type="quantitative",
            chart_type="arc",
            color_field=None,
            color_type=None,
            aggregate=None,
            sort=None,
            color_scheme="tableau10",
        )
        assert "theta" in encoding
        assert "color" in encoding
        assert encoding["theta"]["field"] == "val"
        assert encoding["color"]["field"] == "cat"

    def test_heatmap_color_matches_y_field(self) -> None:
        encoding = _build_encoding(
            x_field="x",
            y_field="y",
            x_type="nominal",
            y_type="quantitative",
            chart_type="heatmap",
            color_field=None,
            color_type=None,
            aggregate=None,
            sort=None,
            color_scheme="tableau10",
        )
        assert encoding["color"]["field"] == "y"
