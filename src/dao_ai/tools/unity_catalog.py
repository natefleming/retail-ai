from typing import Any, Dict, Optional, Sequence, Union

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import PermissionsChange, Privilege
from databricks_langchain import DatabricksFunctionClient, UCFunctionToolkit
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import StructuredTool
from loguru import logger
from unitycatalog.ai.core.base import FunctionExecutionResult

from dao_ai.config import (
    AnyVariable,
    CompositeVariableModel,
    ToolModel,
    UnityCatalogFunctionModel,
    value_of,
)
from dao_ai.utils import normalize_host


def create_uc_tools(
    function: UnityCatalogFunctionModel | str,
) -> Sequence[RunnableLike]:
    """
    Create LangChain tools from Unity Catalog functions.

    This factory function wraps Unity Catalog functions as LangChain tools,
    making them available for use by agents. Each UC function becomes a callable
    tool that can be invoked by the agent during reasoning.

    Args:
        function: UnityCatalogFunctionModel instance containing the function details

    Returns:
        A sequence of BaseTool objects that wrap the specified UC functions
    """

    logger.debug(f"create_uc_tools: {function}")

    original_function_model = None
    if isinstance(function, UnityCatalogFunctionModel):
        original_function_model = function
        function_name = function.full_name
    else:
        function_name = function

    # Determine which tools to create
    if original_function_model and original_function_model.partial_args:
        logger.debug("Found partial_args, creating custom tool with partial arguments")
        # Create a ToolModel wrapper for the with_partial_args function
        tool_model = ToolModel(
            name=original_function_model.name, function=original_function_model
        )

        # Use with_partial_args to create the authenticated tool
        tools = [with_partial_args(tool_model, original_function_model.partial_args)]
    else:
        # Fallback to standard UC toolkit approach
        client: DatabricksFunctionClient = DatabricksFunctionClient()

        toolkit: UCFunctionToolkit = UCFunctionToolkit(
            function_names=[function_name], client=client
        )

        tools = toolkit.tools or []
        logger.debug(f"Retrieved tools: {tools}")

    # HITL is now handled at middleware level via HumanInTheLoopMiddleware
    return list(tools)


def _execute_uc_function(
    client: DatabricksFunctionClient,
    function_name: str,
    partial_args: Dict[str, str] = None,
    **kwargs: Any,
) -> str:
    """Execute Unity Catalog function with partial args and provided parameters."""

    # Start with partial args if provided
    all_params: Dict[str, Any] = dict(partial_args) if partial_args else {}

    # Add any additional kwargs
    all_params.update(kwargs)

    logger.debug(
        f"Calling UC function {function_name} with parameters: {list(all_params.keys())}"
    )

    result: FunctionExecutionResult = client.execute_function(
        function_name=function_name, parameters=all_params
    )

    # Handle errors and extract result
    if result.error:
        logger.error(f"Unity Catalog function error: {result.error}")
        raise RuntimeError(f"Function execution failed: {result.error}")

    result_value: str = result.value if result.value is not None else str(result)
    logger.debug(f"UC function result: {result_value}")
    return result_value


def _grant_function_permissions(
    function_name: str,
    client_id: str,
    host: Optional[str] = None,
) -> None:
    """
    Grant comprehensive permissions to the service principal for Unity Catalog function execution.

    This includes:
    - EXECUTE permission on the function itself
    - USE permission on the containing schema
    - USE permission on the containing catalog
    """
    try:
        # Initialize workspace client
        workspace_client = WorkspaceClient(host=host) if host else WorkspaceClient()

        # Parse the function name to get catalog and schema
        parts = function_name.split(".")
        if len(parts) != 3:
            logger.warning(
                f"Invalid function name format: {function_name}. Expected catalog.schema.function"
            )
            return

        catalog_name, schema_name, func_name = parts
        schema_full_name = f"{catalog_name}.{schema_name}"

        logger.debug(
            f"Granting comprehensive permissions on function {function_name} to principal {client_id}"
        )

        # 1. Grant EXECUTE permission on the function
        try:
            workspace_client.grants.update(
                securable_type="function",
                full_name=function_name,
                changes=[
                    PermissionsChange(principal=client_id, add=[Privilege.EXECUTE])
                ],
            )
            logger.debug(f"Granted EXECUTE on function {function_name}")
        except Exception as e:
            logger.warning(f"Failed to grant EXECUTE on function {function_name}: {e}")

        # 2. Grant USE_SCHEMA permission on the schema
        try:
            workspace_client.grants.update(
                securable_type="schema",
                full_name=schema_full_name,
                changes=[
                    PermissionsChange(
                        principal=client_id,
                        add=[Privilege.USE_SCHEMA],
                    )
                ],
            )
            logger.debug(f"Granted USE_SCHEMA on schema {schema_full_name}")
        except Exception as e:
            logger.warning(
                f"Failed to grant USE_SCHEMA on schema {schema_full_name}: {e}"
            )

        # 3. Grant USE_CATALOG and BROWSE permissions on the catalog
        try:
            workspace_client.grants.update(
                securable_type="catalog",
                full_name=catalog_name,
                changes=[
                    PermissionsChange(
                        principal=client_id,
                        add=[Privilege.USE_CATALOG, Privilege.BROWSE],
                    )
                ],
            )
            logger.debug(f"Granted USE_CATALOG and BROWSE on catalog {catalog_name}")
        except Exception as e:
            logger.warning(
                f"Failed to grant catalog permissions on {catalog_name}: {e}"
            )

        logger.debug(
            f"Successfully granted comprehensive permissions on {function_name} to {client_id}"
        )

    except Exception as e:
        logger.warning(
            f"Failed to grant permissions on function {function_name} to {client_id}: {e}"
        )
        # Don't fail the tool creation if permission granting fails
        pass


def _create_filtered_schema(original_schema: type, exclude_fields: set[str]) -> type:
    """
    Create a new Pydantic model that excludes specified fields from the original schema.

    Args:
        original_schema: The original Pydantic model class
        exclude_fields: Set of field names to exclude from the schema

    Returns:
        A new Pydantic model class with the specified fields removed
    """
    from pydantic import BaseModel, Field, create_model
    from pydantic.fields import PydanticUndefined

    try:
        # Get the original model's fields (Pydantic v2)
        original_fields = original_schema.model_fields
        filtered_field_definitions = {}

        for name, field in original_fields.items():
            if name not in exclude_fields:
                # Reconstruct the field definition for create_model
                field_type = field.annotation
                field_default = (
                    field.default if field.default is not PydanticUndefined else ...
                )
                field_info = Field(default=field_default, description=field.description)
                filtered_field_definitions[name] = (field_type, field_info)

        # If no fields remain after filtering, return a generic empty schema
        if not filtered_field_definitions:

            class EmptySchema(BaseModel):
                """Unity Catalog function with all parameters provided via partial args."""

                pass

            return EmptySchema

        # Create the new model dynamically
        model_name = f"Filtered{original_schema.__name__}"
        docstring = getattr(
            original_schema, "__doc__", "Filtered Unity Catalog function parameters."
        )

        filtered_model = create_model(
            model_name, __doc__=docstring, **filtered_field_definitions
        )
        return filtered_model

    except Exception as e:
        logger.warning(f"Failed to create filtered schema: {e}")

        # Fallback to generic schema
        class GenericFilteredSchema(BaseModel):
            """Generic filtered schema for Unity Catalog function."""

            pass

        return GenericFilteredSchema


def with_partial_args(
    tool: Union[ToolModel, Dict[str, Any]],
    partial_args: dict[str, AnyVariable] = {},
) -> StructuredTool:
    """
    Create a Unity Catalog tool with partial arguments pre-filled.

    This function creates a wrapper tool that calls the UC function with partial arguments
    already resolved, so the caller only needs to provide the remaining parameters.

    Args:
        tool: ToolModel containing the Unity Catalog function configuration
        partial_args: Dictionary of arguments to pre-fill in the tool.
            Supports:
            - client_id, client_secret: OAuth credentials directly
            - service_principal: ServicePrincipalModel with client_id and client_secret
            - host or workspace_host: Databricks workspace host

    Returns:
        StructuredTool: A LangChain tool with partial arguments pre-filled
    """
    from unitycatalog.ai.langchain.toolkit import generate_function_input_params_schema

    from dao_ai.config import ServicePrincipalModel

    logger.debug(f"with_partial_args: {tool}")

    # Convert dict-based variables to CompositeVariableModel and resolve their values
    resolved_args: dict[str, Any] = {}
    for k, v in partial_args.items():
        if isinstance(v, dict):
            resolved_args[k] = value_of(CompositeVariableModel(**v))
        else:
            resolved_args[k] = value_of(v)

    # Handle service_principal - expand into client_id and client_secret
    if "service_principal" in resolved_args:
        sp = resolved_args.pop("service_principal")
        if isinstance(sp, dict):
            sp = ServicePrincipalModel(**sp)
        if isinstance(sp, ServicePrincipalModel):
            if "client_id" not in resolved_args:
                resolved_args["client_id"] = value_of(sp.client_id)
            if "client_secret" not in resolved_args:
                resolved_args["client_secret"] = value_of(sp.client_secret)

    # Normalize host/workspace_host - accept either key, ensure https:// scheme
    if "workspace_host" in resolved_args and "host" not in resolved_args:
        resolved_args["host"] = normalize_host(resolved_args.pop("workspace_host"))
    elif "host" in resolved_args:
        resolved_args["host"] = normalize_host(resolved_args["host"])

    # Default host from WorkspaceClient if not provided
    if "host" not in resolved_args:
        from dao_ai.utils import get_default_databricks_host

        host: str | None = get_default_databricks_host()
        if host:
            resolved_args["host"] = host

    logger.debug(f"Resolved partial args: {resolved_args.keys()}")

    if isinstance(tool, dict):
        tool = ToolModel(**tool)

    unity_catalog_function = tool.function
    if isinstance(unity_catalog_function, dict):
        unity_catalog_function = UnityCatalogFunctionModel(**unity_catalog_function)

    function_name: str = unity_catalog_function.full_name
    logger.debug(f"Creating UC tool with partial args for: {function_name}")

    # Grant permissions if we have credentials
    if "client_id" in resolved_args:
        client_id: str = resolved_args["client_id"]
        host: Optional[str] = resolved_args.get("host")
        try:
            _grant_function_permissions(function_name, client_id, host)
        except Exception as e:
            logger.warning(f"Failed to grant permissions: {e}")

    # Create the client for function execution
    client: DatabricksFunctionClient = DatabricksFunctionClient()

    # Try to get the function schema for better tool definition
    try:
        function_info = client.get_function(function_name)
        schema_info = generate_function_input_params_schema(function_info)
        tool_description = (
            function_info.comment or f"Unity Catalog function: {function_name}"
        )

        logger.debug(
            f"Generated schema for function {function_name}: {schema_info.pydantic_model}"
        )
        logger.debug(f"Tool description: {tool_description}")

        # Create a modified schema that excludes partial args
        original_schema = schema_info.pydantic_model
        schema_model = _create_filtered_schema(original_schema, resolved_args.keys())
        logger.debug(
            f"Filtered schema excludes partial args: {list(resolved_args.keys())}"
        )

    except Exception as e:
        logger.warning(f"Could not introspect function {function_name}: {e}")
        # Fallback to a generic schema
        from pydantic import BaseModel

        class GenericUCParams(BaseModel):
            """Generic parameters for Unity Catalog function."""

            pass

        schema_model = GenericUCParams
        tool_description = f"Unity Catalog function: {function_name}"

    # Create a wrapper function that calls _execute_uc_function with partial args
    def uc_function_wrapper(**kwargs) -> str:
        """Wrapper function that executes Unity Catalog function with partial args."""
        return _execute_uc_function(
            client=client,
            function_name=function_name,
            partial_args=resolved_args,
            **kwargs,
        )

    # Set the function name for the decorator
    uc_function_wrapper.__name__ = tool.name or function_name.replace(".", "_")

    # Create the tool using LangChain's StructuredTool
    from langchain_core.tools import StructuredTool

    partial_tool = StructuredTool.from_function(
        func=uc_function_wrapper,
        name=tool.name or function_name.replace(".", "_"),
        description=tool_description,
        args_schema=schema_model,
    )

    return partial_tool
