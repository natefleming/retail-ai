"""Integration tests for Unity Catalog tool creation functionality."""

import os
from unittest.mock import Mock, patch

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from dao_ai.config import (
    CompositeVariableModel,
    FunctionType,
    PrimitiveVariableModel,
    SchemaModel,
    ToolModel,
    UnityCatalogFunctionModel,
)
from dao_ai.tools.unity_catalog import (
    _create_auth_fields,
    _execute_uc_function,
    _get_base_fields,
    _grant_function_permissions,
    _merge_schema_fields,
    _normalize_inputs,
    create_authenticating_uc_tool,
)


class MockFunctionInfo:
    """Mock Unity Catalog function info for testing."""

    def __init__(self, name: str, comment: str = None):
        self.name = name
        self.comment = comment


class MockSchemaInfo:
    """Mock schema info for testing."""

    def __init__(self, pydantic_model: type[BaseModel]):
        self.pydantic_model = pydantic_model


class MockExecutionResult:
    """Mock execution result for testing."""

    def __init__(self, value: str = "success", error: str = None):
        self.value = value
        self.error = error


@pytest.mark.integration
class TestCreateAuthenticatingUCTool:
    """Integration tests for create_authenticating_uc_tool function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_schema = SchemaModel(
            catalog_name="test_catalog", schema_name="test_schema"
        )
        self.test_uc_function = UnityCatalogFunctionModel(
            name="test_function",
            type=FunctionType.UNITY_CATALOG,
            schema=self.test_schema,
        )
        self.test_tool = ToolModel(name="test_tool", function=self.test_uc_function)
        self.test_host = CompositeVariableModel(value="https://test.databricks.com")
        self.test_client_id = CompositeVariableModel(value="test_client_id")
        self.test_client_secret = CompositeVariableModel(value="test_secret")

    def test_create_authenticating_uc_tool_with_all_auth_params(self):
        """Test creating authenticated UC tool with all authentication parameters."""
        with (
            patch(
                "dao_ai.tools.unity_catalog.DatabricksFunctionClient"
            ) as mock_client_class,
            patch(
                "dao_ai.tools.unity_catalog.generate_function_input_params_schema"
            ) as mock_gen_schema,
            patch("dao_ai.tools.unity_catalog.get_tool_name") as mock_get_name,
            patch("dao_ai.tools.unity_catalog.as_human_in_the_loop") as mock_human_loop,
        ):
            # Setup mocks
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Create a simple test schema
            class TestSchema(BaseModel):
                input_param: str = Field(description="Test input parameter")

            mock_function_info = MockFunctionInfo(
                name="test_catalog.test_schema.test_function",
                comment="Test function for integration testing",
            )
            mock_client.get_function.return_value = mock_function_info

            mock_schema_info = MockSchemaInfo(TestSchema)
            mock_gen_schema.return_value = mock_schema_info

            mock_get_name.return_value = "test_function"

            # Mock human-in-the-loop wrapper to return the tool unchanged
            mock_human_loop.side_effect = lambda tool, function: tool

            # Test the function
            result_tool = create_authenticating_uc_tool(
                tool=self.test_tool,
                host=self.test_host,
                client_id=self.test_client_id,
                client_secret=self.test_client_secret,
            )

            # Assertions
            assert isinstance(result_tool, StructuredTool)
            assert result_tool.name == "test_function"
            assert "Test function for integration testing" in result_tool.description

            # Verify DatabricksFunctionClient was called
            mock_client_class.assert_called_once()
            mock_client.get_function.assert_called_once_with(
                "test_catalog.test_schema.test_function"
            )

            # Verify human-in-the-loop wrapper was called
            mock_human_loop.assert_called_once()

    def test_create_authenticating_uc_tool_with_dict_inputs(self):
        """Test creating authenticated UC tool with dictionary inputs."""
        with (
            patch(
                "dao_ai.tools.unity_catalog.DatabricksFunctionClient"
            ) as mock_client_class,
            patch("dao_ai.tools.unity_catalog.as_human_in_the_loop") as mock_human_loop,
        ):
            # Setup mocks for fallback scenario
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get_function.side_effect = Exception("Function not found")

            mock_human_loop.side_effect = lambda tool, function: tool

            # Test with dictionary inputs
            result_tool = create_authenticating_uc_tool(
                tool={
                    "name": "dict_tool",
                    "function": {"name": "dict_func", "type": "unity_catalog"},
                },
                host={"value": "https://dict.databricks.com"},
                client_id={"value": "dict_client"},
                client_secret={"value": "dict_secret"},
            )

            # Assertions
            assert isinstance(result_tool, StructuredTool)
            assert result_tool.name == "dict_tool"
            assert "Unity Catalog function: dict_func" in result_tool.description

            # Verify human-in-the-loop wrapper was called
            mock_human_loop.assert_called_once()

    def test_create_authenticating_uc_tool_without_client_credentials(self):
        """Test creating authenticated UC tool without client credentials."""
        with (
            patch(
                "dao_ai.tools.unity_catalog.DatabricksFunctionClient"
            ) as mock_client_class,
            patch("dao_ai.tools.unity_catalog.as_human_in_the_loop") as mock_human_loop,
        ):
            # Setup mocks for fallback scenario
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get_function.side_effect = Exception("Function not found")

            mock_human_loop.side_effect = lambda tool, function: tool

            # Test without client credentials
            result_tool = create_authenticating_uc_tool(
                tool=self.test_tool,
                host=self.test_host,
                client_id=None,
                client_secret=None,
            )

            # Assertions
            assert isinstance(result_tool, StructuredTool)
            assert result_tool.name == "test_tool"

            # Verify human-in-the-loop wrapper was called
            mock_human_loop.assert_called_once()

    def test_create_authenticating_uc_tool_fallback_behavior(self):
        """Test fallback behavior when function introspection fails."""
        with (
            patch(
                "dao_ai.tools.unity_catalog.DatabricksFunctionClient"
            ) as mock_client_class,
            patch("dao_ai.tools.unity_catalog.as_human_in_the_loop") as mock_human_loop,
        ):
            # Setup mocks for fallback scenario
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get_function.side_effect = Exception("Catalog does not exist")

            mock_human_loop.side_effect = lambda tool, function: tool

            # Test fallback behavior
            result_tool = create_authenticating_uc_tool(
                tool=self.test_tool,
                host=self.test_host,
                client_id=self.test_client_id,
                client_secret=self.test_client_secret,
            )

            # Assertions
            assert isinstance(result_tool, StructuredTool)
            assert result_tool.name == "test_tool"
            assert (
                "Unity Catalog function: test_catalog.test_schema.test_function"
                in result_tool.description
            )

            # Verify fallback schema is used (should have 'parameters' field)
            assert hasattr(result_tool.args_schema, "model_fields")
            assert "parameters" in result_tool.args_schema.model_fields

            # Verify human-in-the-loop wrapper was called
            mock_human_loop.assert_called_once()

    @pytest.mark.skipif(
        not pytest.importorskip(
            "tests.conftest", reason="conftest not available"
        ).has_retail_ai_env(),
        reason="Retail AI environment variables not set",
    )
    def test_create_authenticating_uc_tool_grants_permissions_real(self):
        """Test that creating authenticated UC tool grants permissions using real credentials."""

        # Get credentials from environment variables
        databricks_host = os.environ["RETAIL_AI_DATABRICKS_HOST"]
        databricks_client_id = os.environ["RETAIL_AI_DATABRICKS_CLIENT_ID"]
        databricks_client_secret = os.environ["RETAIL_AI_DATABRICKS_CLIENT_SECRET"]

        # Create CompositeVariableModel instances using environment variables
        host = CompositeVariableModel(
            options=[PrimitiveVariableModel(value=databricks_host)]
        )
        client_id = CompositeVariableModel(
            options=[PrimitiveVariableModel(value=databricks_client_id)]
        )
        client_secret = CompositeVariableModel(
            options=[PrimitiveVariableModel(value=databricks_client_secret)]
        )

        # Test with a real Unity Catalog function from the config
        real_tool = ToolModel(
            name="insert_coffee_order_tool_uc",
            function=UnityCatalogFunctionModel(
                name="insert_coffee_order",
                type=FunctionType.UNITY_CATALOG,
                schema=SchemaModel(
                    catalog_name="retail_consumer_goods",
                    schema_name="quick_serve_restaurant",
                ),
            ),
        )

        with patch(
            "dao_ai.tools.unity_catalog.as_human_in_the_loop"
        ) as mock_human_loop:
            mock_human_loop.side_effect = lambda tool, function: tool

            try:
                # This should attempt to grant real permissions
                result_tool = create_authenticating_uc_tool(
                    tool=real_tool,
                    host=host,
                    client_id=client_id,
                    client_secret=client_secret,
                )

                # Verify tool was created successfully
                assert isinstance(result_tool, StructuredTool)
                assert "insert_coffee_order" in result_tool.name.lower()

            except Exception as e:
                # Log the error but don't fail the test if it's just a permission issue
                # This is expected if the service principal doesn't have admin rights
                if (
                    "insufficient privileges" in str(e).lower()
                    or "permission" in str(e).lower()
                ):
                    pytest.skip(f"Permission granting failed as expected: {e}")
                else:
                    # Re-raise if it's a different error
                    raise

    def test_create_authenticating_uc_tool_grants_permissions_mocked(self):
        """Test that creating authenticated UC tool grants permissions to client_id (mocked)."""

        # Create properly structured CompositeVariableModel instances
        host = CompositeVariableModel(
            options=[PrimitiveVariableModel(value="https://test.databricks.com")]
        )
        client_id = CompositeVariableModel(
            options=[PrimitiveVariableModel(value="test_client_id")]
        )
        client_secret = CompositeVariableModel(
            options=[PrimitiveVariableModel(value="test_secret")]
        )

        with (
            patch(
                "dao_ai.tools.unity_catalog.DatabricksFunctionClient"
            ) as mock_client_class,
            patch("dao_ai.tools.unity_catalog.as_human_in_the_loop") as mock_human_loop,
            patch(
                "dao_ai.tools.unity_catalog._grant_function_permissions"
            ) as mock_grant_permissions,
        ):
            # Setup mocks for successful function introspection
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock successful function info retrieval
            mock_function_info = MockFunctionInfo("test_function", "Test function")
            mock_client.get_function.return_value = mock_function_info

            # Mock schema generation
            class TestSchema(BaseModel):
                test_param: str = Field(description="Test parameter")

            mock_schema_info = MockSchemaInfo(TestSchema)

            with patch(
                "dao_ai.tools.unity_catalog.generate_function_input_params_schema",
                return_value=mock_schema_info,
            ):
                with patch(
                    "dao_ai.tools.unity_catalog.get_tool_name",
                    return_value="test_function_name",
                ):
                    mock_human_loop.side_effect = lambda tool, function: tool

                    result_tool = create_authenticating_uc_tool(
                        tool=self.test_tool,
                        host=host,
                        client_id=client_id,
                        client_secret=client_secret,
                    )

                    # Verify permission granting was called with correct parameters
                    mock_grant_permissions.assert_called_once_with(
                        "test_catalog.test_schema.test_function",
                        "test_client_id",
                        "https://test.databricks.com",
                    )

                    # Verify tool was created successfully
                    assert isinstance(result_tool, StructuredTool)


@pytest.mark.unit
class TestHelperFunctions:
    """Unit tests for helper functions in unity_catalog module."""

    def test_normalize_inputs_with_objects(self):
        """Test _normalize_inputs with object inputs."""
        test_schema = SchemaModel(
            catalog_name="test_catalog", schema_name="test_schema"
        )
        test_uc_function = UnityCatalogFunctionModel(
            name="test_function", type=FunctionType.UNITY_CATALOG, schema=test_schema
        )
        test_tool = ToolModel(name="test_tool", function=test_uc_function)
        test_host = CompositeVariableModel(value="https://test.databricks.com")
        test_client_id = CompositeVariableModel(value="test_client")
        test_client_secret = CompositeVariableModel(value="test_secret")

        tool, host, client_id, client_secret, uc_function = _normalize_inputs(
            test_tool, test_host, test_client_id, test_client_secret
        )

        assert isinstance(tool, ToolModel)
        assert isinstance(host, CompositeVariableModel)
        assert isinstance(client_id, CompositeVariableModel)
        assert isinstance(client_secret, CompositeVariableModel)
        assert isinstance(uc_function, UnityCatalogFunctionModel)

    def test_normalize_inputs_with_dicts(self):
        """Test _normalize_inputs with dictionary inputs."""
        tool_dict = {
            "name": "dict_tool",
            "function": {"name": "dict_func", "type": "unity_catalog"},
        }
        host_dict = {"options": [{"value": "https://dict.databricks.com"}]}
        client_id_dict = {"options": [{"value": "dict_client"}]}
        client_secret_dict = {"options": [{"value": "dict_secret"}]}

        tool, host, client_id, client_secret, uc_function = _normalize_inputs(
            tool_dict, host_dict, client_id_dict, client_secret_dict
        )

        assert isinstance(tool, ToolModel)
        assert isinstance(host, CompositeVariableModel)
        assert isinstance(client_id, CompositeVariableModel)
        assert isinstance(client_secret, CompositeVariableModel)
        assert tool.name == "dict_tool"
        assert host.as_value() == "https://dict.databricks.com"

    def test_get_base_fields(self):
        """Test _get_base_fields function."""

        class TestSchema(BaseModel):
            field1: str = Field(description="Test field 1")
            field2: int = Field(description="Test field 2")

        fields = _get_base_fields(TestSchema)

        assert isinstance(fields, dict)
        assert "field1" in fields
        assert "field2" in fields

    def test_create_auth_fields(self):
        """Test _create_auth_fields function."""
        base_fields = {"existing_field": "some_value"}
        host = CompositeVariableModel(value="https://test.databricks.com")
        client_id = CompositeVariableModel(value="test_client")
        client_secret = CompositeVariableModel(value="test_secret")

        auth_fields = _create_auth_fields(base_fields, host, client_id, client_secret)

        assert isinstance(auth_fields, dict)
        assert "host" in auth_fields
        assert "client_id" in auth_fields
        assert "client_secret" in auth_fields

        # Each field should be a tuple of (type, Field)
        assert isinstance(auth_fields["host"], tuple)
        assert len(auth_fields["host"]) == 2
        assert auth_fields["host"][0] is str

    def test_create_auth_fields_no_conflicts(self):
        """Test _create_auth_fields doesn't create conflicting fields."""
        # Base fields already has 'host' field
        base_fields = {"host": "existing_host_field", "other_field": "value"}
        host = CompositeVariableModel(value="https://test.databricks.com")
        client_id = CompositeVariableModel(value="test_client")
        client_secret = CompositeVariableModel(value="test_secret")

        auth_fields = _create_auth_fields(base_fields, host, client_id, client_secret)

        # Should not include 'host' since it already exists in base_fields
        assert "host" not in auth_fields
        assert "client_id" in auth_fields
        assert "client_secret" in auth_fields

    def test_merge_schema_fields(self):
        """Test _merge_schema_fields function."""

        class TestSchema(BaseModel):
            existing_field: str = Field(description="Existing field")

        auth_fields = {
            "host": (str, Field(default="https://test.com", description="Host URL")),
            "client_id": (str, Field(default="test_client", description="Client ID")),
        }

        merged_fields = _merge_schema_fields(TestSchema, auth_fields)

        assert isinstance(merged_fields, dict)
        assert "existing_field" in merged_fields
        assert "host" in merged_fields
        assert "client_id" in merged_fields

    def test_execute_uc_function_success(self):
        """Test _execute_uc_function with successful execution."""
        with patch("dao_ai.tools.unity_catalog.logger"):
            mock_client = Mock()
            mock_result = MockExecutionResult(value="test_result")
            mock_client.execute_function.return_value = mock_result

            host = CompositeVariableModel(value="https://test.databricks.com")
            client_id = CompositeVariableModel(value="test_client")
            client_secret = CompositeVariableModel(value="test_secret")

            result = _execute_uc_function(
                client=mock_client,
                function_name="test.schema.function",
                host=host,
                client_id=client_id,
                client_secret=client_secret,
                test_param="test_value",
            )

            assert result == "test_result"
            mock_client.execute_function.assert_called_once()

            # Verify parameters were passed correctly
            call_args = mock_client.execute_function.call_args
            assert call_args[1]["function_name"] == "test.schema.function"
            assert "host" in call_args[1]["parameters"]
            assert "client_id" in call_args[1]["parameters"]
            assert "client_secret" in call_args[1]["parameters"]
            assert "test_param" in call_args[1]["parameters"]

    def test_execute_uc_function_with_error(self):
        """Test _execute_uc_function with execution error."""
        with patch("dao_ai.tools.unity_catalog.logger"):
            mock_client = Mock()
            mock_result = MockExecutionResult(
                value="", error="Function execution failed"
            )
            mock_client.execute_function.return_value = mock_result

            host = CompositeVariableModel(value="https://test.databricks.com")

            with pytest.raises(RuntimeError, match="Function execution failed"):
                _execute_uc_function(
                    client=mock_client,
                    function_name="test.schema.function",
                    host=host,
                    client_id=None,
                    client_secret=None,
                )

    def test_grant_function_permissions_success(self):
        """Test _grant_function_permissions with comprehensive permission granting."""
        with patch(
            "dao_ai.tools.unity_catalog.WorkspaceClient"
        ) as mock_workspace_client_class:
            mock_workspace_client = Mock()
            mock_workspace_client_class.return_value = mock_workspace_client

            # Mock the grants.update method
            mock_grants = Mock()
            mock_workspace_client.grants = mock_grants

            _grant_function_permissions(
                function_name="test_catalog.test_schema.test_function",
                client_id="test_service_principal",
                host="https://test.databricks.com",
            )

            # Verify WorkspaceClient was initialized with correct host
            mock_workspace_client_class.assert_called_once_with(
                host="https://test.databricks.com"
            )

            # Verify grants.update was called 3 times (function, schema, catalog)
            assert mock_grants.update.call_count == 3

            # Check the calls were made for all securable objects
            call_args_list = [call[1] for call in mock_grants.update.call_args_list]
            securable_types = [call["securable_type"] for call in call_args_list]
            full_names = [call["full_name"] for call in call_args_list]

            # Should have 1 schema call, 1 function call, 1 catalog call
            assert securable_types.count("function") == 1
            assert securable_types.count("schema") == 1  # USE_SCHEMA only
            assert securable_types.count("catalog") == 1
            assert "test_catalog.test_schema.test_function" in full_names
            assert "test_catalog.test_schema" in full_names
            assert "test_catalog" in full_names

    def test_grant_function_permissions_with_error(self):
        """Test _grant_function_permissions handles errors gracefully."""
        with patch(
            "dao_ai.tools.unity_catalog.WorkspaceClient"
        ) as mock_workspace_client_class:
            mock_workspace_client_class.side_effect = Exception("Permission denied")

            # Should not raise exception even if permission granting fails
            _grant_function_permissions(
                function_name="test.schema.function",
                client_id="test_service_principal",
                host="https://test.databricks.com",
            )


@pytest.mark.integration
class TestEndToEndBehavior:
    """End-to-end integration tests for Unity Catalog tool functionality."""

    def test_tool_creation_and_structure(self):
        """Test complete tool creation and verify its structure."""
        with (
            patch(
                "dao_ai.tools.unity_catalog.DatabricksFunctionClient"
            ) as mock_client_class,
            patch("dao_ai.tools.unity_catalog.as_human_in_the_loop") as mock_human_loop,
        ):
            # Setup mocks for fallback scenario (simulating non-existent catalog)
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get_function.side_effect = Exception("Catalog does not exist")

            mock_human_loop.side_effect = lambda tool, function: tool

            # Create tool
            test_schema = SchemaModel(
                catalog_name="integration_test", schema_name="test_schema"
            )
            test_uc_function = UnityCatalogFunctionModel(
                name="integration_function",
                type=FunctionType.UNITY_CATALOG,
                schema=test_schema,
            )
            test_tool = ToolModel(name="integration_tool", function=test_uc_function)
            test_host = CompositeVariableModel(
                value="https://integration.databricks.com"
            )

            result_tool = create_authenticating_uc_tool(
                tool=test_tool, host=test_host, client_id=None, client_secret=None
            )

            # Comprehensive structure verification
            assert isinstance(result_tool, StructuredTool)
            assert result_tool.name == "integration_tool"
            assert result_tool.description is not None
            assert (
                "integration_test.test_schema.integration_function"
                in result_tool.description
            )

            # Verify schema structure
            assert hasattr(result_tool, "args_schema")
            assert hasattr(result_tool.args_schema, "model_fields")

            # Verify the tool has expected fallback schema structure
            schema_fields = result_tool.args_schema.model_fields
            assert "parameters" in schema_fields

            # Verify the tool is callable (has invoke method)
            assert hasattr(result_tool, "invoke") or hasattr(result_tool, "ainvoke")

            # Verify human-in-the-loop integration
            mock_human_loop.assert_called_once()
            call_args = mock_human_loop.call_args
            assert (
                call_args[1]["function"]
                == "integration_test.test_schema.integration_function"
            )
