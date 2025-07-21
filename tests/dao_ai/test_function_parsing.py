"""Unit tests for function model parsing and discrimination."""

import pytest
import yaml
from pydantic import ValidationError

from dao_ai.config import (
    FactoryFunctionModel,
    FunctionType,
    McpFunctionModel,
    PythonFunctionModel,
    ToolModel,
    TransportType,
    UnityCatalogFunctionModel,
)


class TestFunctionModelParsing:
    """Test parsing of different function model types from YAML."""

    @pytest.mark.unit
    def test_factory_function_parsing(self):
        """Test that factory function YAML correctly creates FactoryFunctionModel."""
        yaml_data = {
            "name": "reservation_tool",
            "function": {
                "type": "factory",
                "name": "retail.tools.create_reservation_tool",
                "human_in_the_loop": {
                    "review_prompt": "Would you like to confirm your reservation?"
                },
            },
        }

        tool = ToolModel(**yaml_data)

        assert tool.name == "reservation_tool"
        assert isinstance(tool.function, FactoryFunctionModel)
        assert tool.function.type == FunctionType.FACTORY
        assert tool.function.name == "retail.tools.create_reservation_tool"
        assert tool.function.human_in_the_loop is not None
        assert (
            tool.function.human_in_the_loop.review_prompt
            == "Would you like to confirm your reservation?"
        )

    @pytest.mark.unit
    def test_python_function_parsing(self):
        """Test that python function YAML correctly creates PythonFunctionModel."""
        yaml_data = {
            "name": "python_tool",
            "function": {"type": "python", "name": "some.module.function_name"},
        }

        tool = ToolModel(**yaml_data)

        assert tool.name == "python_tool"
        assert isinstance(tool.function, PythonFunctionModel)
        assert tool.function.type == FunctionType.PYTHON
        assert tool.function.name == "some.module.function_name"
        assert tool.function.human_in_the_loop is None

    @pytest.mark.unit
    def test_unity_catalog_function_parsing(self):
        """Test that unity catalog function YAML correctly creates UnityCatalogFunctionModel."""
        yaml_data = {
            "name": "uc_tool",
            "function": {
                "type": "unity_catalog",
                "name": "my_function",
                "schema": {"catalog_name": "my_catalog", "schema_name": "my_schema"},
            },
        }

        tool = ToolModel(**yaml_data)

        assert tool.name == "uc_tool"
        assert isinstance(tool.function, UnityCatalogFunctionModel)
        assert tool.function.type == FunctionType.UNITY_CATALOG
        assert tool.function.name == "my_function"
        assert tool.function.schema_model is not None
        assert tool.function.schema_model.catalog_name == "my_catalog"
        assert tool.function.schema_model.schema_name == "my_schema"

    @pytest.mark.unit
    def test_mcp_function_parsing_stdio(self):
        """Test that MCP function YAML with stdio transport correctly creates McpFunctionModel."""
        yaml_data = {
            "name": "mcp_tool",
            "function": {
                "type": "mcp",
                "name": "mcp_function",
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "some_mcp_server"],
            },
        }

        tool = ToolModel(**yaml_data)

        assert tool.name == "mcp_tool"
        assert isinstance(tool.function, McpFunctionModel)
        assert tool.function.type == FunctionType.MCP
        assert tool.function.name == "mcp_function"
        assert tool.function.transport == TransportType.STDIO
        assert tool.function.command == "python"
        assert tool.function.args == ["-m", "some_mcp_server"]

    @pytest.mark.unit
    def test_mcp_function_parsing_http(self):
        """Test that MCP function YAML with HTTP transport correctly creates McpFunctionModel."""
        yaml_data = {
            "name": "mcp_http_tool",
            "function": {
                "type": "mcp",
                "name": "http_mcp_function",
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            },
        }

        tool = ToolModel(**yaml_data)

        assert tool.name == "mcp_http_tool"
        assert isinstance(tool.function, McpFunctionModel)
        assert tool.function.type == FunctionType.MCP
        assert tool.function.name == "http_mcp_function"
        assert tool.function.transport == TransportType.STREAMABLE_HTTP
        assert tool.function.url == "http://localhost:8000/mcp"

    @pytest.mark.unit
    def test_factory_function_with_args(self):
        """Test factory function with additional arguments."""
        yaml_data = {
            "name": "factory_with_args",
            "function": {
                "type": "factory",
                "name": "tools.create_complex_tool",
                "args": {"param1": "value1", "param2": 42, "param3": True},
            },
        }

        tool = ToolModel(**yaml_data)

        assert isinstance(tool.function, FactoryFunctionModel)
        assert tool.function.args == {"param1": "value1", "param2": 42, "param3": True}

    @pytest.mark.unit
    def test_string_function_parsing(self):
        """Test that string function names are handled correctly."""
        yaml_data = {"name": "string_tool", "function": "simple.function.name"}

        tool = ToolModel(**yaml_data)

        assert tool.name == "string_tool"
        assert isinstance(tool.function, str)
        assert tool.function == "simple.function.name"

    @pytest.mark.unit
    def test_invalid_function_type_raises_error(self):
        """Test that invalid function type raises ValidationError."""
        yaml_data = {
            "name": "invalid_tool",
            "function": {"type": "invalid_type", "name": "some.function"},
        }

        with pytest.raises(ValidationError, match="Input should be"):
            ToolModel(**yaml_data)

    @pytest.mark.unit
    def test_mcp_validation_error_missing_url(self):
        """Test that MCP function with HTTP transport missing URL raises ValidationError."""
        yaml_data = {
            "name": "invalid_mcp_tool",
            "function": {
                "type": "mcp",
                "name": "mcp_function",
                "transport": "streamable_http",
                # Missing required 'url' field
            },
        }

        with pytest.raises(
            ValidationError, match="url must be provided for STREAMABLE_HTTP transport"
        ):
            ToolModel(**yaml_data)

    @pytest.mark.unit
    def test_human_in_the_loop_model_parsing(self):
        """Test parsing of HumanInTheLoopModel."""
        yaml_data = {
            "name": "hitl_tool",
            "function": {
                "type": "python",
                "name": "tools.example_function",
                "human_in_the_loop": {
                    "review_prompt": "Please review this action:",
                    "interrupt_config": {
                        "allow_accept": True,
                        "allow_edit": False,
                        "allow_respond": True,
                        "allow_decline": True,
                    },
                },
            },
        }

        tool = ToolModel(**yaml_data)

        hitl = tool.function.human_in_the_loop
        assert hitl is not None
        assert hitl.review_prompt == "Please review this action:"
        assert hitl.interrupt_config == {
            "allow_accept": True,
            "allow_edit": False,
            "allow_respond": True,
            "allow_decline": True,
        }


class TestFunctionModelFromYAML:
    """Test parsing function models from actual YAML strings."""

    @pytest.mark.unit
    def test_parse_from_yaml_string(self):
        """Test parsing tools from YAML string."""
        yaml_string = """
        tools:
          reservation_tool:
            name: reservation_tool
            function:
              type: factory
              name: retail.tools.create_reservation_tool
              human_in_the_loop:
                review_prompt: |
                  Would you like to confirm your reservation?
          
          python_tool:
            name: python_tool
            function:
              type: python
              name: utils.helper_function
        """

        data = yaml.safe_load(yaml_string)

        # Test factory tool
        factory_tool = ToolModel(**data["tools"]["reservation_tool"])
        assert isinstance(factory_tool.function, FactoryFunctionModel)
        assert factory_tool.function.type == FunctionType.FACTORY

        # Test python tool
        python_tool = ToolModel(**data["tools"]["python_tool"])
        assert isinstance(python_tool.function, PythonFunctionModel)
        assert python_tool.function.type == FunctionType.PYTHON

    @pytest.mark.unit
    def test_full_name_properties(self):
        """Test full_name properties for different function types."""
        # Test simple function
        python_tool = ToolModel(
            name="test_tool", function={"type": "python", "name": "module.function"}
        )
        assert python_tool.function.full_name == "module.function"

        # Test Unity Catalog function with schema
        uc_tool = ToolModel(
            name="uc_tool",
            function={
                "type": "unity_catalog",
                "name": "my_function",
                "schema": {"catalog_name": "catalog", "schema_name": "schema"},
            },
        )
        assert uc_tool.function.full_name == "catalog.schema.my_function"

    @pytest.mark.unit
    def test_model_serialization(self):
        """Test that function models can be serialized and deserialized."""
        original_data = {
            "name": "test_tool",
            "function": {
                "type": "factory",
                "name": "tools.create_tool",
                "args": {"param": "value"},
            },
        }

        # Create tool from data
        tool = ToolModel(**original_data)

        # Serialize to dict
        serialized = tool.model_dump()

        # Deserialize back
        recreated_tool = ToolModel(**serialized)

        assert isinstance(recreated_tool.function, FactoryFunctionModel)
        assert recreated_tool.function.type == FunctionType.FACTORY
        assert recreated_tool.function.name == "tools.create_tool"
        assert recreated_tool.function.args == {"param": "value"}
