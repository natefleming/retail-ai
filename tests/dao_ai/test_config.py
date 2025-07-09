import sys

import pytest
import yaml
from mlflow.models import ModelConfig

from dao_ai.config import (
    AppConfig,
    CompositeVariableModel,
    McpFunctionModel,
    PrimitiveVariableModel,
    TransportType,
)


@pytest.mark.unit
def test_app_config(model_config: ModelConfig) -> None:
    app_config = AppConfig(**model_config.to_dict())
    print(app_config.model_dump_json(indent=2), file=sys.stderr)
    assert app_config is not None


@pytest.mark.unit
def test_app_config_should_serialize(config: AppConfig) -> None:
    yaml.safe_dump(config.model_dump())
    assert True


@pytest.mark.unit
def test_app_config_tools_should_be_correct_type(
    model_config: ModelConfig, config: AppConfig
) -> None:
    for tool_name, tool in config.tools.items():
        assert tool_name in model_config.get("tools"), (
            f"Tool {tool_name} not found in model_config"
        )
        expected_type = None
        for _, expected_tool in model_config.get("tools").items():
            if expected_tool["name"] == tool.name:
                expected_type = expected_tool["function"]["type"]
                break
        assert expected_type is not None, (
            f"Expected type for tool '{tool_name}' not found in model_config"
        )
        actual_type = tool.function.type
        assert actual_type == expected_type, (
            f"Function type mismatch for tool '{tool_name}': "
            f"expected '{expected_type}', got '{actual_type}'"
        )


@pytest.mark.unit
def test_app_config_should_initialize(config: AppConfig) -> None:
    config.initialize()


@pytest.mark.unit
def test_app_config_should_shutdown(config: AppConfig) -> None:
    config.shutdown()


@pytest.mark.unit
def test_mcp_function_model_validate_bearer_header_preserves_existing_prefix() -> None:
    """Test that validate_bearer_header preserves existing 'Bearer ' prefix."""
    mcp_function = McpFunctionModel(
        name="test_mcp",
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={"Authorization": "Bearer abc123token"},
    )

    assert mcp_function.headers["Authorization"] == "Bearer abc123token"


@pytest.mark.unit
def test_mcp_function_model_validate_bearer_header_with_composite_variable() -> None:
    """Test that validate_bearer_header works with CompositeVariableModel."""
    # Create a CompositeVariableModel that resolves to a token without Bearer prefix
    token_variable = CompositeVariableModel(
        options=[PrimitiveVariableModel(value="Bearer secret123")]
    )

    mcp_function = McpFunctionModel(
        name="test_mcp",
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={"Authorization": token_variable},
    )

    # The validator should have converted the CompositeVariableModel to its resolved value with Bearer prefix
    assert mcp_function.headers["Authorization"] == "Bearer secret123"


@pytest.mark.unit
def test_mcp_function_model_validate_bearer_header_with_composite_variable_existing_prefix() -> (
    None
):
    """Test that validate_bearer_header preserves Bearer prefix in CompositeVariableModel."""
    # Create a CompositeVariableModel that already has Bearer prefix
    token_variable = CompositeVariableModel(
        options=[PrimitiveVariableModel(value="Bearer secret123")]
    )

    mcp_function = McpFunctionModel(
        name="test_mcp",
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={"Authorization": token_variable},
    )

    assert mcp_function.headers["Authorization"] == "Bearer secret123"


@pytest.mark.unit
def test_mcp_function_model_validate_bearer_header_no_authorization_header() -> None:
    """Test that validate_bearer_header does nothing when no Authorization header exists."""
    mcp_function = McpFunctionModel(
        name="test_mcp",
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={"Content-Type": "application/json"},
    )

    # Should not add Authorization header if it doesn't exist
    assert "Authorization" not in mcp_function.headers
    assert mcp_function.headers["Content-Type"] == "application/json"


@pytest.mark.unit
def test_mcp_function_model_validate_bearer_header_empty_headers() -> None:
    """Test that validate_bearer_header works with empty headers dict."""
    mcp_function = McpFunctionModel(
        name="test_mcp",
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={},
    )

    # Should not modify empty headers
    assert len(mcp_function.headers) == 0


@pytest.mark.unit
def test_mcp_function_model_validate_bearer_header_with_other_headers() -> None:
    """Test that validate_bearer_header only modifies Authorization header."""
    mcp_function = McpFunctionModel(
        name="test_mcp",
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={
            "Authorization": "Bearer mytoken",
            "Content-Type": "application/json",
            "X-Custom-Header": "custom-value",
        },
    )

    # Only Authorization header should be modified
    assert mcp_function.headers["Authorization"] == "Bearer mytoken"
    assert mcp_function.headers["Content-Type"] == "application/json"
    assert mcp_function.headers["X-Custom-Header"] == "custom-value"
