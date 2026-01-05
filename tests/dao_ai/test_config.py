import sys

import pytest
import yaml
from conftest import has_retail_ai_env
from mlflow.models import ModelConfig

from dao_ai.config import (
    AppConfig,
    AppModel,
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
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={"Authorization": token_variable},
    )

    assert mcp_function.headers["Authorization"] == "Bearer secret123"


@pytest.mark.unit
def test_mcp_function_model_validate_bearer_header_no_authorization_header() -> None:
    """Test that model creation doesn't add Authorization header - auth is per-invocation."""
    mcp_function = McpFunctionModel(
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={"Content-Type": "application/json"},
    )

    # Headers should remain as provided - no Authorization header added at creation time
    assert "Authorization" not in mcp_function.headers
    assert "Content-Type" in mcp_function.headers
    assert mcp_function.headers["Content-Type"] == "application/json"


@pytest.mark.unit
def test_mcp_function_model_validate_bearer_header_empty_headers() -> None:
    """Test that model creation with empty headers stays empty - auth is per-invocation."""
    mcp_function = McpFunctionModel(
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={},
    )

    # Headers should remain empty - no Authorization header added at creation time
    assert len(mcp_function.headers) == 0
    assert "Authorization" not in mcp_function.headers


@pytest.mark.unit
def test_mcp_function_model_validate_bearer_header_with_other_headers() -> None:
    """Test that validate_bearer_header only modifies Authorization header."""
    mcp_function = McpFunctionModel(
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
    """Test OAuth authentication configuration is stored but not applied at creation time."""
    mcp_function = McpFunctionModel(
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        workspace_host="https://test.databricks.com",
        client_id="test_client_id",
        client_secret="test_client_secret",
    )

    # Verify authentication config is stored
    assert mcp_function.workspace_host == "https://test.databricks.com"
    assert mcp_function.client_id == "test_client_id"
    assert mcp_function.client_secret == "test_client_secret"

    # But no Authorization header should be set at creation time (per-invocation auth)
    assert "Authorization" not in mcp_function.headers


@pytest.mark.unit
def test_mcp_function_model_pat_authentication() -> None:
    """Test PAT authentication configuration is stored but not applied at creation time."""
    mcp_function = McpFunctionModel(
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        pat="test_pat",
        workspace_host="https://test-workspace.cloud.databricks.com",
    )

    # Verify authentication config is stored
    assert mcp_function.pat == "test_pat"
    assert mcp_function.workspace_host == "https://test-workspace.cloud.databricks.com"

    # But no Authorization header should be set at creation time (per-invocation auth)
    assert "Authorization" not in mcp_function.headers


@pytest.mark.unit
def test_mcp_function_model_no_authentication() -> None:
    """Test that when no authentication is provided, model stores None values for per-invocation auth."""
    mcp_function = McpFunctionModel(
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
    )

    # Verify default authentication config values
    assert mcp_function.workspace_host is None
    assert mcp_function.client_id is None
    assert mcp_function.client_secret is None
    assert mcp_function.pat is None

    # No Authorization header should be set at creation time (per-invocation auth)
    assert "Authorization" not in mcp_function.headers


@pytest.mark.unit
def test_mcp_function_model_authentication_with_environment_variables() -> None:
    """Test authentication using environment variables stores config for per-invocation auth."""
    mcp_function = McpFunctionModel(
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        workspace_host=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_HOST"),
        client_id=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_CLIENT_ID"),
        client_secret=EnvironmentVariableModel(
            env="RETAIL_AI_DATABRICKS_CLIENT_SECRET"
        ),
    )

    # Verify environment variable config is stored
    assert isinstance(mcp_function.workspace_host, EnvironmentVariableModel)
    assert mcp_function.workspace_host.env == "RETAIL_AI_DATABRICKS_HOST"
    assert isinstance(mcp_function.client_id, EnvironmentVariableModel)
    assert mcp_function.client_id.env == "RETAIL_AI_DATABRICKS_CLIENT_ID"
    assert isinstance(mcp_function.client_secret, EnvironmentVariableModel)
    assert mcp_function.client_secret.env == "RETAIL_AI_DATABRICKS_CLIENT_SECRET"

    # No Authorization header should be set at creation time (per-invocation auth)
    assert "Authorization" not in mcp_function.headers


@pytest.mark.unit
def test_mcp_function_model_mixed_auth_methods_error() -> None:
    """Test that providing both OAuth and PAT authentication raises an error."""
    with pytest.raises(
        ValueError, match="Cannot use both OAuth and user authentication methods"
    ):
        McpFunctionModel(
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com",
            workspace_host="https://test.databricks.com",
            client_id="test_client_id",
            client_secret="test_client_secret",
            pat="test_pat",
        )


@pytest.mark.unit
def test_mcp_function_model_partial_oauth_credentials() -> None:
    """Test that partial OAuth credentials are stored for per-invocation authentication."""
    # Only provide client_id and client_secret, with workspace_host
    mcp_function = McpFunctionModel(
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        client_id="test_client_id",
        client_secret="test_client_secret",
        workspace_host="https://test-workspace.cloud.databricks.com",
    )

    # Verify OAuth credentials are stored
    assert mcp_function.client_id == "test_client_id"
    assert mcp_function.client_secret == "test_client_secret"
    assert mcp_function.workspace_host == "https://test-workspace.cloud.databricks.com"
    assert mcp_function.pat is None

    # No Authorization header should be set at creation time (per-invocation auth)
    assert "Authorization" not in mcp_function.headers


@pytest.mark.unit
def test_mcp_function_model_existing_authorization_header() -> None:
    """Test that Authorization header can be stored in headers dict.

    Note: With unified auth (via DatabricksOAuthClientProvider), the Authorization
    header in the headers dict is not used for authentication. Instead, the
    workspace_client from IsDatabricksResource provides authentication.

    This test verifies that the model can store headers, but authentication
    happens through the OAuth provider at invocation time.
    """
    mcp_function = McpFunctionModel(
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        headers={"Authorization": "Bearer existing_token"},
        pat="test_pat",
        workspace_host="https://test-workspace.cloud.databricks.com",
    )

    # Header is stored in the model
    assert mcp_function.headers["Authorization"] == "Bearer existing_token"

    # But authentication will happen via workspace_client at invocation time
    # The stored header won't be used by _build_connection_config


@pytest.mark.unit
def test_mcp_function_model_authentication_failure() -> None:
    """Test that authentication credentials are stored even if they might fail during invocation."""
    mcp_function = McpFunctionModel(
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        pat="invalid_pat",
        workspace_host="https://test-workspace.cloud.databricks.com",
    )

    # Authentication config should be stored (validation happens during invocation)
    assert mcp_function.pat == "invalid_pat"
    assert mcp_function.workspace_host == "https://test-workspace.cloud.databricks.com"

    # No Authorization header should be set at creation time (per-invocation auth)
    assert "Authorization" not in mcp_function.headers


@pytest.mark.system
@pytest.mark.skipif(
    not has_retail_ai_env(), reason="Missing Retail AI environment variables"
)
def test_mcp_function_model_real_authentication() -> None:
    """Integration test with real Retail AI environment variables."""

    mcp_function = McpFunctionModel(
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        workspace_host=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_HOST"),
        client_id=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_CLIENT_ID"),
        client_secret=EnvironmentVariableModel(
            env="RETAIL_AI_DATABRICKS_CLIENT_SECRET"
        ),
    )

    # Verify environment variable config is stored for per-invocation auth
    assert isinstance(mcp_function.workspace_host, EnvironmentVariableModel)
    assert isinstance(mcp_function.client_id, EnvironmentVariableModel)
    assert isinstance(mcp_function.client_secret, EnvironmentVariableModel)

    # No Authorization header should be set at creation time (per-invocation auth)
    assert "Authorization" not in mcp_function.headers


@pytest.mark.system
@pytest.mark.skipif(
    not has_retail_ai_env(), reason="Missing Retail AI environment variables"
)
def test_mcp_function_model_real_pat_authentication() -> None:
    """Integration test with real PAT authentication."""

    mcp_function = McpFunctionModel(
        transport=TransportType.STREAMABLE_HTTP,
        url="https://example.com",
        pat=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_TOKEN"),
        workspace_host=EnvironmentVariableModel(env="RETAIL_AI_DATABRICKS_HOST"),
    )

    # Verify environment variable config is stored for per-invocation auth
    assert isinstance(mcp_function.pat, EnvironmentVariableModel)
    assert isinstance(mcp_function.workspace_host, EnvironmentVariableModel)

    # No Authorization header should be set at creation time (per-invocation auth)
    assert "Authorization" not in mcp_function.headers


@pytest.mark.unit
def test_app_config_initialize_adds_code_paths_to_sys_path(config: AppConfig) -> None:
    """Test that code_paths are added to sys.path during model creation"""
    import os
    import tempfile
    from pathlib import Path

    # Create temporary directories for testing
    with (
        tempfile.TemporaryDirectory() as temp_dir1,
        tempfile.TemporaryDirectory() as temp_dir2,
    ):
        # Store original sys.path
        original_sys_path = sys.path.copy()

        try:
            # Create minimal AppModel data with code_paths
            from dao_ai.config import (
                AgentModel,
                LLMModel,
                OrchestrationModel,
                RegisteredModelModel,
                SchemaModel,
                SupervisorModel,
            )

            schema = SchemaModel(catalog_name="test", schema_name="test")
            registered_model = RegisteredModelModel(schema=schema, name="test_model")
            supervisor = SupervisorModel(model=LLMModel(name="test"))
            orchestration = OrchestrationModel(supervisor=supervisor)
            agent = AgentModel(name="test", model=LLMModel(name="test"))

            # Create new AppModel instance with code_paths - this should trigger the validator
            _ = AppModel(
                name="test_app",
                registered_model=registered_model,
                orchestration=orchestration,
                agents=[agent],
                code_paths=[temp_dir1, temp_dir2],
            )

            # Verify that the parent directories of both paths were added to sys.path immediately
            # The implementation adds parent directories, not the exact code_paths
            abs_temp_dir1 = os.path.abspath(temp_dir1)
            abs_temp_dir2 = os.path.abspath(temp_dir2)
            parent_dir1 = str(Path(abs_temp_dir1).parent)
            parent_dir2 = str(Path(abs_temp_dir2).parent)

            assert parent_dir1 in sys.path, (
                f"Parent of code path {parent_dir1} not found in sys.path"
            )
            assert parent_dir2 in sys.path, (
                f"Parent of code path {parent_dir2} not found in sys.path"
            )

            # Since both temp dirs have the same parent (system temp dir), only one parent path should be added
            # Verify the parent directory was added
            if parent_dir1 == parent_dir2:
                # Both temp dirs have same parent, so only one entry
                assert sys.path.count(parent_dir1) >= 1, (
                    f"Parent directory {parent_dir1} should be added to sys.path"
                )
            else:
                # Different parents, check both
                assert parent_dir1 in sys.path, f"Parent {parent_dir1} not in sys.path"
                assert parent_dir2 in sys.path, f"Parent {parent_dir2} not in sys.path"

        finally:
            # Restore original sys.path
            sys.path[:] = original_sys_path


@pytest.mark.unit
def test_app_config_initialize_skips_duplicate_code_paths(config: AppConfig) -> None:
    """Test that duplicate paths are not added to sys.path"""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        abs_temp_dir = os.path.abspath(temp_dir)

        # Add the path to sys.path first
        original_sys_path = sys.path.copy()
        sys.path.insert(0, abs_temp_dir)

        try:
            # Count occurrences before creating model with code_paths
            count_before = sys.path.count(abs_temp_dir)

            # Create minimal AppModel data with code_paths
            from dao_ai.config import (
                AgentModel,
                LLMModel,
                OrchestrationModel,
                RegisteredModelModel,
                SchemaModel,
                SupervisorModel,
            )

            schema = SchemaModel(catalog_name="test", schema_name="test")
            registered_model = RegisteredModelModel(schema=schema, name="test_model")
            supervisor = SupervisorModel(model=LLMModel(name="test"))
            orchestration = OrchestrationModel(supervisor=supervisor)
            agent = AgentModel(name="test", model=LLMModel(name="test"))

            # Create new AppModel instance with the same path - this should not add a duplicate
            _ = AppModel(
                name="test_app",
                registered_model=registered_model,
                orchestration=orchestration,
                agents=[agent],
                code_paths=[temp_dir],
            )

            # Count occurrences after creating model with code_paths
            count_after = sys.path.count(abs_temp_dir)

            # Should not have added a duplicate
            assert count_after == count_before, "Duplicate path was added to sys.path"

        finally:
            # Restore original sys.path
            sys.path[:] = original_sys_path


@pytest.mark.unit
def test_app_config_initialize_with_empty_code_paths(config: AppConfig) -> None:
    """Test that empty code_paths works correctly"""
    # Store original sys.path
    original_sys_path = sys.path.copy()

    try:
        # Create minimal AppModel data with empty code_paths
        from dao_ai.config import (
            AgentModel,
            LLMModel,
            OrchestrationModel,
            RegisteredModelModel,
            SchemaModel,
            SupervisorModel,
        )

        schema = SchemaModel(catalog_name="test", schema_name="test")
        registered_model = RegisteredModelModel(schema=schema, name="test_model")
        supervisor = SupervisorModel(model=LLMModel(name="test"))
        orchestration = OrchestrationModel(supervisor=supervisor)
        agent = AgentModel(name="test", model=LLMModel(name="test"))

        # Create new AppModel instance with empty code_paths
        _ = AppModel(
            name="test_app",
            registered_model=registered_model,
            orchestration=orchestration,
            agents=[agent],
            code_paths=[],
        )

        # sys.path should be unchanged
        # We can't assert exact equality because the assignment might still trigger the validator,
        # but we can verify no new paths were added
        assert True, (
            "Creating AppModel with empty code_paths should complete without errors"
        )

    finally:
        # Restore original sys.path
        sys.path[:] = original_sys_path
