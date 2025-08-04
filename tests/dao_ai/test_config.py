import sys
from unittest.mock import Mock, patch

import pytest
import yaml
from conftest import has_retail_ai_env
from mlflow.models import ModelConfig

from dao_ai.config import (
    AppConfig,
    CompositeVariableModel,
    EnvironmentVariableModel,
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
    """Test that validate_bearer_header creates default auth when no Authorization header exists."""
    mcp_function = McpFunctionModel(
        name="test_mcp",
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={"Content-Type": "application/json"},
    )

    # Should add Authorization header with default auth
    assert "Authorization" in mcp_function.headers
    assert mcp_function.headers["Authorization"].startswith("Bearer ")
    assert "Content-Type" in mcp_function.headers
    assert mcp_function.headers["Content-Type"] == "application/json"


@pytest.mark.unit
def test_mcp_function_model_validate_bearer_header_empty_headers() -> None:
    """Test that validate_bearer_header works with empty headers dict and creates default auth."""
    mcp_function = McpFunctionModel(
        name="test_mcp",
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={},
    )

    # Should add Authorization header with default auth
    assert len(mcp_function.headers) == 1
    assert "Authorization" in mcp_function.headers
    assert mcp_function.headers["Authorization"].startswith("Bearer ")


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


# Authentication Tests
@pytest.mark.unit
def test_mcp_function_model_oauth_authentication() -> None:
    """Test OAuth authentication with client_id, client_secret, and workspace_host."""
    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        # Mock the provider instance and its create_token method
        mock_provider = Mock()
        mock_provider.create_token.return_value = "test_oauth_token"
        mock_provider_class.return_value = mock_provider

        mcp_function = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com",
            workspace_host="https://test.databricks.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
        )

        # Verify DatabricksProvider was called with correct parameters
        mock_provider_class.assert_called_once_with(
            workspace_host="https://test.databricks.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            pat=None,
        )

        # Verify create_token was called
        mock_provider.create_token.assert_called_once()

        # Verify Authorization header was set
        assert mcp_function.headers["Authorization"] == "Bearer test_oauth_token"


@pytest.mark.unit
def test_mcp_function_model_pat_authentication() -> None:
    """Test PAT authentication."""
    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        # Mock the provider instance and its create_token method
        mock_provider = Mock()
        mock_provider.create_token.return_value = "test_pat_token"
        mock_provider_class.return_value = mock_provider

        mcp_function = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com",
            pat="test_pat",
            workspace_host="https://test-workspace.cloud.databricks.com",
        )

        # Verify DatabricksProvider was called with correct parameters
        mock_provider_class.assert_called_once_with(
            workspace_host="https://test-workspace.cloud.databricks.com",
            client_id=None,
            client_secret=None,
            pat="test_pat",
        )

        # Verify create_token was called
        mock_provider.create_token.assert_called_once()

        # Verify Authorization header was set
        assert mcp_function.headers["Authorization"] == "Bearer test_pat_token"


@pytest.mark.unit
def test_mcp_function_model_no_authentication() -> None:
    """Test that when no authentication is provided, it creates a default WorkspaceClient."""
    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        # Mock the provider instance and its create_token method
        mock_provider = Mock()
        mock_provider.create_token.return_value = "default_token"
        mock_provider_class.return_value = mock_provider

        mcp_function = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com",
        )

        # Verify DatabricksProvider was called with all None parameters (default auth)
        mock_provider_class.assert_called_once_with(
            workspace_host=None,
            client_id=None,
            client_secret=None,
            pat=None,
        )

        # Verify create_token was called
        mock_provider.create_token.assert_called_once()

        # Verify Authorization header was set
        assert mcp_function.headers["Authorization"] == "Bearer default_token"


@pytest.mark.unit
def test_mcp_function_model_authentication_with_environment_variables() -> None:
    """Test authentication using environment variables."""
    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        # Mock the provider instance and its create_token method
        mock_provider = Mock()
        mock_provider.create_token.return_value = "env_token"
        mock_provider_class.return_value = mock_provider

        mcp_function = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com",
            workspace_host=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_HOST"),
            client_id=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_CLIENT_ID"),
            client_secret=EnvironmentVariableModel(
                env="RETAIL_AI_DATABRICKS_CLIENT_SECRET"
            ),
        )

        # The actual environment variable values will be resolved during validation
        mock_provider_class.assert_called_once()
        mock_provider.create_token.assert_called_once()
        assert mcp_function.headers["Authorization"] == "Bearer env_token"


@pytest.mark.unit
def test_mcp_function_model_mixed_auth_methods_error() -> None:
    """Test that providing both OAuth and PAT authentication raises an error."""
    with pytest.raises(
        ValueError, match="Cannot use both OAuth and user authentication methods"
    ):
        McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com",
            workspace_host="https://test.databricks.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            pat="test_pat",
        )


@pytest.mark.unit
def test_mcp_function_model_partial_oauth_credentials() -> None:
    """Test that partial OAuth credentials still trigger default authentication."""
    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mock_provider = Mock()
        mock_provider.create_token.return_value = "fallback_token"
        mock_provider_class.return_value = mock_provider

        # Only provide client_id and client_secret, with workspace_host
        mcp_function = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            workspace_host="https://test-workspace.cloud.databricks.com",
        )

        # Should create DatabricksProvider with OAuth credentials
        mock_provider_class.assert_called_once_with(
            workspace_host="https://test-workspace.cloud.databricks.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            pat=None,
        )

        # Verify create_token was called
        mock_provider.create_token.assert_called_once()

        # Verify Authorization header was set
        assert mcp_function.headers["Authorization"] == "Bearer fallback_token"


@pytest.mark.unit
def test_mcp_function_model_existing_authorization_header() -> None:
    """Test that existing Authorization header is preserved and authentication is skipped."""
    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        mcp_function = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com",
            headers={"Authorization": "Bearer existing_token"},
            pat="test_pat",
            workspace_host="https://test-workspace.cloud.databricks.com",
        )

        # DatabricksProvider should not be called since Authorization header exists
        mock_provider_class.assert_not_called()

        # Existing header should be preserved
        assert mcp_function.headers["Authorization"] == "Bearer existing_token"


@pytest.mark.unit
def test_mcp_function_model_authentication_failure() -> None:
    """Test that authentication failure is handled gracefully."""
    with patch("dao_ai.providers.databricks.DatabricksProvider") as mock_provider_class:
        with patch("dao_ai.config.logger") as mock_logger:
            # Mock the provider to raise an exception
            mock_provider = Mock()
            mock_provider.create_token.side_effect = Exception("Authentication failed")
            mock_provider_class.return_value = mock_provider

            mcp_function = McpFunctionModel(
                name="test_mcp",
                transport=TransportType.STREAMABLE_HTTP,
                url="https://example.com",
                pat="invalid_pat",
                workspace_host="https://test-workspace.cloud.databricks.com",
            )

            # Should log the error
            mock_logger.error.assert_called_once_with(
                "Failed to create initial token: Authentication failed"
            )  # Should not set Authorization header on failure
            assert "Authorization" not in mcp_function.headers


@pytest.mark.system
@pytest.mark.skipif(
    not has_retail_ai_env(), reason="Missing Retail AI environment variables"
)
def test_mcp_function_model_real_authentication() -> None:
    """Integration test with real Retail AI environment variables."""

    mcp_function = McpFunctionModel(
        name="test_mcp",
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        workspace_host=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_HOST"),
        client_id=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_CLIENT_ID"),
        client_secret=EnvironmentVariableModel(
            env="RETAIL_AI_DATABRICKS_CLIENT_SECRET"
        ),
    )

    # Should have Authorization header set
    assert "Authorization" in mcp_function.headers
    assert mcp_function.headers["Authorization"].startswith("Bearer ")

    # Token should not be empty
    token = mcp_function.headers["Authorization"].replace("Bearer ", "")
    assert len(token) > 0


@pytest.mark.system
@pytest.mark.skipif(
    not has_retail_ai_env(), reason="Missing Retail AI environment variables"
)
def test_mcp_function_model_real_pat_authentication() -> None:
    """Integration test with real PAT authentication."""

    mcp_function = McpFunctionModel(
        name="test_mcp",
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        pat=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_TOKEN"),
        workspace_host=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_HOST"),
    )

    # Should have Authorization header set
    assert "Authorization" in mcp_function.headers
    assert mcp_function.headers["Authorization"].startswith("Bearer ")

    # Token should not be empty
    token = mcp_function.headers["Authorization"].replace("Bearer ", "")
    assert len(token) > 0
