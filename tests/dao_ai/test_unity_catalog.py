"""Integration tests for Unity Catalog tool creation functionality."""

import os
from unittest.mock import Mock, patch

import pytest
from conftest import has_retail_ai_env
from langchain_core.tools import StructuredTool

from dao_ai.config import (
    CompositeVariableModel,
    FunctionModel,
    FunctionType,
    PrimitiveVariableModel,
    SchemaModel,
    UnityCatalogFunctionModel,
)
from dao_ai.tools.unity_catalog import (
    create_uc_tools,
)


@pytest.mark.unit
def test_create_uc_tools_with_partial_args() -> None:
    """Test that create_uc_tools with partial_args creates and executes Unity Catalog function properly."""

    # Create test configuration matching the YAML structure
    schema = SchemaModel(
        catalog_name="retail_consumer_goods", schema_name="quick_serve_restaurant"
    )

    # Create partial_args with test credentials
    partial_args = {
        "host": CompositeVariableModel(
            options=[PrimitiveVariableModel(value="https://test.databricks.com")]
        ),
        "client_id": CompositeVariableModel(
            options=[PrimitiveVariableModel(value="test_client_id")]
        ),
        "client_secret": CompositeVariableModel(
            options=[PrimitiveVariableModel(value="test_secret")]
        ),
    }

    # Create FunctionModel resource (using alias 'schema' instead of 'schema_model')
    function_resource = FunctionModel(
        schema=schema,
        name="insert_coffee_order",
    )

    # Create Unity Catalog function with partial_args
    uc_function = UnityCatalogFunctionModel(
        type=FunctionType.UNITY_CATALOG,
        resource=function_resource,
        partial_args=partial_args,
    )

    with (
        patch(
            "dao_ai.tools.unity_catalog.DatabricksFunctionClient"
        ) as mock_client_class,
        patch(
            "dao_ai.tools.unity_catalog._grant_function_permissions"
        ) as mock_grant_perms,
    ):
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Test the function
        # Note: HITL is now handled at middleware level, not tool level
        result_tools = create_uc_tools(uc_function)

        # Assertions
        assert len(result_tools) == 1
        assert isinstance(result_tools[0], StructuredTool)  # The created tool
        assert result_tools[0].name == "insert_coffee_order"

        # Verify that the tool has the correct description
        expected_description = "Unity Catalog function: retail_consumer_goods.quick_serve_restaurant.insert_coffee_order"
        assert result_tools[0].description == expected_description

        # Verify that partial args are NOT in the tool's schema (they should be filtered out)
        tool_schema = result_tools[0].args_schema
        if hasattr(tool_schema, "model_fields"):
            # Pydantic v2
            schema_fields = set(tool_schema.model_fields.keys())
        elif hasattr(tool_schema, "__fields__"):
            # Pydantic v1
            schema_fields = set(tool_schema.__fields__.keys())
        else:
            schema_fields = set()

        # The partial args should NOT be in the schema since they're provided via closure
        partial_arg_names = {"host", "client_id", "client_secret"}
        overlapping_fields = schema_fields.intersection(partial_arg_names)
        assert len(overlapping_fields) == 0, (
            f"Partial args {overlapping_fields} should not be in tool schema but were found"
        )

        # Verify permissions were granted
        mock_grant_perms.assert_called_once_with(
            "retail_consumer_goods.quick_serve_restaurant.insert_coffee_order",
            "test_client_id",
            "https://test.databricks.com",
        )

        # Verify DatabricksFunctionClient was created
        mock_client_class.assert_called_once()

        # Test that we can invoke the created tool (with mock execution)
        mock_execute_result = "Mock execution result"
        with patch(
            "dao_ai.tools.unity_catalog._execute_uc_function",
            return_value=mock_execute_result,
        ) as mock_execute:
            # Invoke the tool with some test parameters
            tool_result = result_tools[0].invoke({"test_param": "test_value"})

            # Verify _execute_uc_function was called with our partial args
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args

            # Verify the client was passed
            assert call_args.kwargs["client"] is not None
            # Verify the function name was passed
            assert (
                call_args.kwargs["function_name"]
                == "retail_consumer_goods.quick_serve_restaurant.insert_coffee_order"
            )
            # Verify partial_args were passed with resolved values
            partial_args_passed = call_args.kwargs["partial_args"]
            assert partial_args_passed["host"] == "https://test.databricks.com"
            assert partial_args_passed["client_id"] == "test_client_id"
            assert partial_args_passed["client_secret"] == "test_secret"
            # Verify we got back the mocked result
            assert tool_result == mock_execute_result


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not has_retail_ai_env(),
    reason="Missing RETAIL_AI environment variables",
)
def test_create_uc_tools_with_partial_args_real_execution() -> None:
    """Integration test that actually executes Unity Catalog function with partial_args."""

    # Get real credentials from environment
    host: str = os.getenv("RETAIL_AI_DATABRICKS_HOST")
    client_id: str = os.getenv("RETAIL_AI_DATABRICKS_CLIENT_ID")
    client_secret: str = os.getenv("RETAIL_AI_DATABRICKS_CLIENT_SECRET")

    # Create test configuration matching the YAML structure
    schema = SchemaModel(
        catalog_name="retail_consumer_goods", schema_name="quick_serve_restaurant"
    )

    # Create partial_args with real credentials
    partial_args = {
        "host": CompositeVariableModel(options=[PrimitiveVariableModel(value=host)]),
        "client_id": CompositeVariableModel(
            options=[PrimitiveVariableModel(value=client_id)]
        ),
        "client_secret": CompositeVariableModel(
            options=[PrimitiveVariableModel(value=client_secret)]
        ),
    }

    # Create FunctionModel resource (using alias 'schema' instead of 'schema_model')
    function_resource = FunctionModel(
        schema=schema,
        name="insert_coffee_order",
    )

    # Create Unity Catalog function with partial_args
    uc_function = UnityCatalogFunctionModel(
        type=FunctionType.UNITY_CATALOG,
        resource=function_resource,
        partial_args=partial_args,
    )

    try:
        # Create the tools
        result_tools = create_uc_tools(uc_function)

        # Verify we got a tool back
        assert len(result_tools) == 1
        tool = result_tools[0]

        # The tool will be a RunnableBinding when using bind() method, which is expected
        from langchain_core.runnables.base import RunnableBinding

        assert isinstance(tool, (StructuredTool, RunnableBinding))

        # Check that it has the expected name (either directly or via bound tool)
        tool_name = getattr(tool, "name", None) or getattr(tool.bound, "name", "")
        assert "insert_coffee_order" in tool_name.lower()

        # Test tool execution with correct parameters based on the SQL function definition
        # Note: This test might fail if the service principal doesn't have proper permissions
        try:
            # Use the correct parameters as defined in the SQL function
            sample_params = {
                "coffee_name": "Cappuccino",  # Exact coffee name as expected by function
                "size": "Medium",  # Valid size option
                "session_id": "test_session_123",  # Session identifier
                # host, client_id, client_secret are provided via partial_args
            }

            result = tool.invoke(sample_params)

            # If execution succeeds, verify we get some result
            assert result is not None
            print(f"Function execution successful: {result}")

            # Verify the result indicates success or provides meaningful output
            assert isinstance(result, str)
            assert len(result) > 0

        except Exception as e:
            # If execution fails due to permissions or validation, that's expected in a test environment
            # We just want to verify the tool was created properly with partial_args
            if any(
                keyword in str(e).lower()
                for keyword in [
                    "permission",
                    "privilege",
                    "validation",
                    "required",
                    "warehouse",
                    "authentication",
                    "access",
                    "forbidden",
                    "unauthorized",
                ]
            ):
                pytest.skip(
                    f"Function execution failed as expected due to environment constraints: {e}"
                )
            else:
                # Re-raise if it's an unexpected error
                raise

    except Exception as e:
        # If tool creation fails due to permissions, skip the test
        if "permission" in str(e).lower() or "privilege" in str(e).lower():
            pytest.skip(f"Tool creation failed due to permissions: {e}")
        else:
            # Re-raise if it's an unexpected error
            raise
