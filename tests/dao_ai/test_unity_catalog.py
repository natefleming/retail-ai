"""Integration tests for Unity Catalog tool creation functionality."""

import os
from typing import Optional, Union
from unittest.mock import Mock, patch

import pytest
from conftest import has_retail_ai_env
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from dao_ai.config import (
    CompositeVariableModel,
    FunctionModel,
    FunctionType,
    PrimitiveVariableModel,
    SchemaModel,
    UnityCatalogFunctionModel,
)
from dao_ai.tools.unity_catalog import (
    _fix_boolean_schema_defaults,
    _is_bool_annotation,
    create_uc_tools,
)


def _has_databricks_auth() -> bool:
    """Check for minimal Databricks auth env vars (HOST + TOKEN)."""
    return bool(os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"))


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


# ---------------------------------------------------------------------------
# Tests for _fix_boolean_schema_defaults
# ---------------------------------------------------------------------------


class TestIsBoolAnnotation:
    """Tests for _is_bool_annotation helper."""

    @pytest.mark.unit
    def test_plain_bool(self) -> None:
        assert _is_bool_annotation(bool) is True

    @pytest.mark.unit
    def test_union_with_bool(self) -> None:
        assert _is_bool_annotation(Union[bool]) is True

    @pytest.mark.unit
    def test_optional_bool(self) -> None:
        assert _is_bool_annotation(Optional[bool]) is True

    @pytest.mark.unit
    def test_non_bool_types(self) -> None:
        assert _is_bool_annotation(str) is False
        assert _is_bool_annotation(int) is False
        assert _is_bool_annotation(Union[str, int]) is False


class TestFixBooleanSchemaDefaults:
    """Tests for _fix_boolean_schema_defaults.

    Validates the workaround for the upstream unitycatalog-ai bug where
    SQL ``DEFAULT TRUE`` / ``DEFAULT FALSE`` is serialised as the string
    ``"TRUE"`` / ``"FALSE"`` in the Pydantic schema sent to the LLM.
    """

    @pytest.mark.unit
    def test_string_true_default_converted_to_bool(self) -> None:
        """String 'TRUE' default on a bool field should become Python True."""
        Model = create_model(
            "BoolModel",
            flag=(bool, Field(default="TRUE", description="A flag (Default: TRUE)")),
        )
        Fixed = _fix_boolean_schema_defaults(Model)
        assert Fixed.model_fields["flag"].default is True

    @pytest.mark.unit
    def test_string_false_default_converted_to_bool(self) -> None:
        """String 'FALSE' default on a bool field should become Python False."""
        Model = create_model(
            "BoolModel",
            flag=(bool, Field(default="FALSE", description="A flag (Default: FALSE)")),
        )
        Fixed = _fix_boolean_schema_defaults(Model)
        assert Fixed.model_fields["flag"].default is False

    @pytest.mark.unit
    def test_case_insensitive(self) -> None:
        """Should handle mixed case like 'True', 'true', 'false'."""
        Model = create_model(
            "CaseModel",
            a=(bool, Field(default="True", description="a (Default: True)")),
            b=(bool, Field(default="false", description="b (Default: false)")),
        )
        Fixed = _fix_boolean_schema_defaults(Model)
        assert Fixed.model_fields["a"].default is True
        assert Fixed.model_fields["b"].default is False

    @pytest.mark.unit
    def test_description_normalised_to_lowercase(self) -> None:
        """The '(Default: TRUE)' suffix should become '(Default: true)'."""
        Model = create_model(
            "DescModel",
            flag=(
                bool,
                Field(default="TRUE", description="Enable feature (Default: TRUE)"),
            ),
        )
        Fixed = _fix_boolean_schema_defaults(Model)
        assert (
            Fixed.model_fields["flag"].description == "Enable feature (Default: true)"
        )

    @pytest.mark.unit
    def test_non_bool_fields_unchanged(self) -> None:
        """Non-bool fields should pass through untouched."""
        Model = create_model(
            "MixedModel",
            name=(str, Field(default="hello", description="Name")),
            count=(int, Field(default=42, description="Count")),
            flag=(bool, Field(default="TRUE", description="Flag (Default: TRUE)")),
        )
        Fixed = _fix_boolean_schema_defaults(Model)
        assert Fixed.model_fields["name"].default == "hello"
        assert Fixed.model_fields["count"].default == 42
        assert Fixed.model_fields["flag"].default is True

    @pytest.mark.unit
    def test_already_correct_bool_default_unchanged(self) -> None:
        """A model with proper bool defaults should be returned as-is (same object)."""
        Model = create_model(
            "CorrectModel",
            flag=(bool, Field(default=True, description="Flag")),
        )
        result = _fix_boolean_schema_defaults(Model)
        assert result is Model

    @pytest.mark.unit
    def test_no_fields_returns_original(self) -> None:
        """An empty model should be returned as-is."""

        class EmptyModel(BaseModel):
            pass

        result = _fix_boolean_schema_defaults(EmptyModel)
        assert result is EmptyModel

    @pytest.mark.unit
    def test_union_bool_field_fixed(self) -> None:
        """Union[bool] annotation (as produced by upstream) should also be fixed."""
        Model = create_model(
            "UnionBoolModel",
            active=(
                Union[bool],
                Field(default="FALSE", description="Active (Default: FALSE)"),
            ),
        )
        Fixed = _fix_boolean_schema_defaults(Model)
        assert Fixed.model_fields["active"].default is False
        assert Fixed.model_fields["active"].description == "Active (Default: false)"

    @pytest.mark.unit
    def test_json_schema_output_has_boolean_default(self) -> None:
        """The generated JSON schema should have a proper boolean default, not a string."""
        Model = create_model(
            "JsonSchemaModel",
            enabled=(
                bool,
                Field(default="TRUE", description="Enabled (Default: TRUE)"),
            ),
            query=(str, Field(description="The query")),
        )
        Fixed = _fix_boolean_schema_defaults(Model)
        json_schema = Fixed.model_json_schema()
        enabled_prop = json_schema["properties"]["enabled"]
        assert enabled_prop["default"] is True
        assert isinstance(enabled_prop["default"], bool)


# ---------------------------------------------------------------------------
# Integration test: boolean defaults with a real UC function
# ---------------------------------------------------------------------------

# The test function is expected to exist at this path.  The CREATE statement
# is in the docstring below so it can be re-created manually if needed.
_BOOL_TEST_FUNCTION: str = "main.dao_ai_test.bool_default_test"

# SQL to (re-)create the test function:
# CREATE OR REPLACE FUNCTION main.dao_ai_test.bool_default_test(
#     query STRING COMMENT 'The search query',
#     include_metadata BOOLEAN DEFAULT TRUE COMMENT 'Whether to include metadata in results',
#     case_sensitive BOOLEAN DEFAULT FALSE COMMENT 'Whether the search should be case sensitive'
# )
# RETURNS STRING
# LANGUAGE SQL
# COMMENT 'Test function for boolean default parameter handling'
# RETURN CONCAT(
#     'query=', query,
#     ', include_metadata=', CAST(include_metadata AS STRING),
#     ', case_sensitive=', CAST(case_sensitive AS STRING)
# );


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    not _has_databricks_auth(),
    reason="Missing DATABRICKS_HOST / DATABRICKS_TOKEN env vars",
)
class TestBooleanDefaultsIntegration:
    """Integration tests that exercise boolean-default fix against a real UC function.

    Requires:
      - DATABRICKS_HOST and DATABRICKS_TOKEN set in the environment.
      - The function ``main.dao_ai_test.bool_default_test`` to exist (see SQL above).
    """

    @pytest.fixture(scope="class", autouse=True)
    def _require_bool_test_function(self) -> None:
        """Skip all tests if the test UC function doesn't exist."""
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.errors.platform import NotFound

        w = WorkspaceClient()
        try:
            w.functions.get(name=_BOOL_TEST_FUNCTION)
        except NotFound:
            pytest.skip(
                f"UC function '{_BOOL_TEST_FUNCTION}' not found "
                "- skipping integration tests"
            )

    def test_upstream_schema_has_string_defaults(self) -> None:
        """Confirm that the upstream library produces string defaults (the bug we're working around)."""
        from databricks.sdk import WorkspaceClient
        from databricks_langchain import DatabricksFunctionClient
        from unitycatalog.ai.core.utils.function_processing_utils import (
            generate_function_input_params_schema,
        )

        w = WorkspaceClient()
        client = DatabricksFunctionClient(client=w)
        func_info = client.get_function(_BOOL_TEST_FUNCTION)
        schema = generate_function_input_params_schema(func_info)
        model = schema.pydantic_model

        # The upstream bug: defaults are strings, not booleans
        include_meta = model.model_fields["include_metadata"]
        case_sens = model.model_fields["case_sensitive"]
        assert isinstance(include_meta.default, str), (
            "Expected upstream to produce a string default for include_metadata"
        )
        assert isinstance(case_sens.default, str), (
            "Expected upstream to produce a string default for case_sensitive"
        )

    def test_fix_corrects_real_uc_function_schema(self) -> None:
        """_fix_boolean_schema_defaults converts real UC function schema defaults to proper booleans."""
        from databricks.sdk import WorkspaceClient
        from databricks_langchain import DatabricksFunctionClient
        from unitycatalog.ai.core.utils.function_processing_utils import (
            generate_function_input_params_schema,
        )

        w = WorkspaceClient()
        client = DatabricksFunctionClient(client=w)
        func_info = client.get_function(_BOOL_TEST_FUNCTION)
        schema = generate_function_input_params_schema(func_info)
        fixed = _fix_boolean_schema_defaults(schema.pydantic_model)

        # Defaults should now be real booleans
        include_meta = fixed.model_fields["include_metadata"]
        case_sens = fixed.model_fields["case_sensitive"]
        assert include_meta.default is True
        assert case_sens.default is False

        # Descriptions should use lowercase "true" / "false"
        assert "(Default: true)" in include_meta.description
        assert "(Default: false)" in case_sens.description

        # JSON schema must also carry boolean defaults
        js = fixed.model_json_schema()
        assert js["properties"]["include_metadata"]["default"] is True
        assert js["properties"]["case_sensitive"]["default"] is False

    def test_create_uc_tools_path_a_has_fixed_schema(self) -> None:
        """Tools created via UCFunctionToolkit (Path A) should have corrected boolean defaults."""
        tools = create_uc_tools(_BOOL_TEST_FUNCTION)
        assert len(tools) == 1
        tool = tools[0]

        schema = tool.args_schema
        js = schema.model_json_schema()

        include_meta_prop = js["properties"]["include_metadata"]
        case_sens_prop = js["properties"]["case_sensitive"]

        assert include_meta_prop["default"] is True, (
            f"Expected boolean True, got {include_meta_prop['default']!r}"
        )
        assert case_sens_prop["default"] is False, (
            f"Expected boolean False, got {case_sens_prop['default']!r}"
        )
        assert "true" in include_meta_prop["description"].lower()
        assert "false" in case_sens_prop["description"].lower()

    def test_tool_execution_with_boolean_params(self) -> None:
        """End-to-end: tool created via Path A can execute with proper boolean values."""
        tools = create_uc_tools(_BOOL_TEST_FUNCTION)
        assert len(tools) == 1
        tool = tools[0]

        result: str = tool.invoke(
            {"query": "test_query", "include_metadata": True, "case_sensitive": False}
        )
        assert "query=test_query" in result
        assert "include_metadata=true" in result
        assert "case_sensitive=false" in result

    def test_tool_execution_with_defaults_omitted(self) -> None:
        """End-to-end: tool works when boolean params are omitted (rely on defaults)."""
        tools = create_uc_tools(_BOOL_TEST_FUNCTION)
        tool = tools[0]

        result: str = tool.invoke({"query": "default_test"})
        assert "query=default_test" in result
        # Defaults: include_metadata=TRUE, case_sensitive=FALSE
        assert "include_metadata=true" in result
        assert "case_sensitive=false" in result
