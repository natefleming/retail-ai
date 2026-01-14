from typing import Annotated, Any, Dict, Optional, Sequence, Set

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import FunctionInfo, PermissionsChange, Privilege
from databricks_langchain import DatabricksFunctionClient, UCFunctionToolkit
from langchain.tools import ToolRuntime
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import InjectedToolArg, StructuredTool
from loguru import logger
from pydantic import BaseModel
from unitycatalog.ai.core.base import FunctionExecutionResult

from dao_ai.config import (
    AnyVariable,
    CompositeVariableModel,
    UnityCatalogFunctionModel,
    value_of,
)
from dao_ai.state import Context
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
    original_function_model: UnityCatalogFunctionModel | None = None
    function_name: str

    if isinstance(function, UnityCatalogFunctionModel):
        original_function_model = function
        function_name = function.resource.full_name
    else:
        function_name = function

    logger.trace("Creating UC tools", function_name=function_name)

    # Determine which tools to create
    tools: list[RunnableLike]
    if original_function_model and original_function_model.partial_args:
        logger.debug(
            "Creating custom tool with partial arguments", function_name=function_name
        )
        # Use with_partial_args directly with UnityCatalogFunctionModel
        tools = [with_partial_args(original_function_model)]
    else:
        # For standard UC toolkit, we need workspace_client at creation time
        # Use the resource's workspace_client (will use ambient auth if no OBO)
        workspace_client: WorkspaceClient | None = None
        if original_function_model:
            workspace_client = original_function_model.resource.workspace_client

        # Fallback to standard UC toolkit approach
        client: DatabricksFunctionClient = DatabricksFunctionClient(
            client=workspace_client
        )

        toolkit: UCFunctionToolkit = UCFunctionToolkit(
            function_names=[function_name], client=client
        )

        tools = toolkit.tools or []
        logger.trace("Retrieved tools", tools_count=len(tools))

    # HITL is now handled at middleware level via HumanInTheLoopMiddleware
    return list(tools)


def _execute_uc_function(
    client: DatabricksFunctionClient,
    function_name: str,
    partial_args: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> str:
    """Execute Unity Catalog function with partial args and provided parameters."""

    # Start with partial args if provided
    all_params: Dict[str, Any] = dict(partial_args) if partial_args else {}

    # Add any additional kwargs
    all_params.update(kwargs)

    logger.debug(
        "Calling UC function",
        function_name=function_name,
        parameters=list(all_params.keys()),
    )

    result: FunctionExecutionResult = client.execute_function(
        function_name=function_name, parameters=all_params
    )

    # Handle errors and extract result
    if result.error:
        logger.error(
            "Unity Catalog function error",
            function_name=function_name,
            error=result.error,
        )
        raise RuntimeError(f"Function execution failed: {result.error}")

    result_value: str = result.value if result.value is not None else str(result)
    logger.trace(
        "UC function result",
        function_name=function_name,
        result_length=len(str(result_value)),
    )
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
        workspace_client: WorkspaceClient = (
            WorkspaceClient(host=host) if host else WorkspaceClient()
        )

        # Parse the function name to get catalog and schema
        parts: list[str] = function_name.split(".")
        if len(parts) != 3:
            logger.warning(
                "Invalid function name format, expected catalog.schema.function",
                function_name=function_name,
            )
            return

        catalog_name: str
        schema_name: str
        func_name: str
        catalog_name, schema_name, func_name = parts
        schema_full_name: str = f"{catalog_name}.{schema_name}"

        logger.debug(
            "Granting comprehensive permissions",
            function_name=function_name,
            principal=client_id,
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
            logger.trace("Granted EXECUTE permission", function_name=function_name)
        except Exception as e:
            logger.warning(
                "Failed to grant EXECUTE permission",
                function_name=function_name,
                error=str(e),
            )

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
            logger.trace("Granted USE_SCHEMA permission", schema=schema_full_name)
        except Exception as e:
            logger.warning(
                "Failed to grant USE_SCHEMA permission",
                schema=schema_full_name,
                error=str(e),
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
            logger.trace(
                "Granted USE_CATALOG and BROWSE permissions", catalog=catalog_name
            )
        except Exception as e:
            logger.warning(
                "Failed to grant catalog permissions",
                catalog=catalog_name,
                error=str(e),
            )

        logger.debug(
            "Successfully granted comprehensive permissions",
            function_name=function_name,
            principal=client_id,
        )

    except Exception as e:
        logger.warning(
            "Failed to grant permissions",
            function_name=function_name,
            principal=client_id,
            error=str(e),
        )
        # Don't fail the tool creation if permission granting fails
        pass


def _create_filtered_schema(original_schema: type, exclude_fields: Set[str]) -> type:
    """
    Create a new Pydantic model that excludes specified fields from the original schema.

    Args:
        original_schema: The original Pydantic model class
        exclude_fields: Set of field names to exclude from the schema

    Returns:
        A new Pydantic model class with the specified fields removed
    """
    from pydantic import Field, create_model
    from pydantic.fields import FieldInfo, PydanticUndefined

    try:
        # Get the original model's fields (Pydantic v2)
        original_fields: dict[str, FieldInfo] = original_schema.model_fields
        filtered_field_definitions: dict[str, tuple[type, FieldInfo]] = {}

        field_name: str
        field: FieldInfo
        for field_name, field in original_fields.items():
            if field_name not in exclude_fields:
                # Reconstruct the field definition for create_model
                field_type: type = field.annotation
                field_default: Any = (
                    field.default if field.default is not PydanticUndefined else ...
                )
                field_info: FieldInfo = Field(
                    default=field_default, description=field.description
                )
                filtered_field_definitions[field_name] = (field_type, field_info)

        # If no fields remain after filtering, return a generic empty schema
        if not filtered_field_definitions:

            class EmptySchema(BaseModel):
                """Unity Catalog function with all parameters provided via partial args."""

                pass

            return EmptySchema

        # Create the new model dynamically
        model_name: str = f"Filtered{original_schema.__name__}"
        docstring: str = getattr(
            original_schema, "__doc__", "Filtered Unity Catalog function parameters."
        )

        filtered_model: type[BaseModel] = create_model(
            model_name, __doc__=docstring, **filtered_field_definitions
        )
        return filtered_model

    except Exception as e:
        logger.warning("Failed to create filtered schema", error=str(e))

        # Fallback to generic schema
        class GenericFilteredSchema(BaseModel):
            """Generic filtered schema for Unity Catalog function."""

            pass

        return GenericFilteredSchema


def with_partial_args(
    uc_function: UnityCatalogFunctionModel,
) -> StructuredTool:
    """
    Create a Unity Catalog tool with partial arguments pre-filled.

    This function creates a wrapper tool that calls the UC function with partial arguments
    already resolved, so the caller only needs to provide the remaining parameters.

    Args:
        uc_function: UnityCatalogFunctionModel containing the function configuration
            and partial_args to pre-fill.

    Returns:
        StructuredTool: A LangChain tool with partial arguments pre-filled
    """
    from unitycatalog.ai.langchain.toolkit import generate_function_input_params_schema

    from dao_ai.config import ServicePrincipalModel

    partial_args: dict[str, AnyVariable] = uc_function.partial_args or {}

    # Convert dict-based variables to CompositeVariableModel and resolve their values
    resolved_args: dict[str, Any] = {}
    k: str
    v: AnyVariable
    for k, v in partial_args.items():
        if isinstance(v, dict):
            resolved_args[k] = value_of(CompositeVariableModel(**v))
        else:
            resolved_args[k] = value_of(v)

    # Handle service_principal - expand into client_id and client_secret
    if "service_principal" in resolved_args:
        sp: Any = resolved_args.pop("service_principal")
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

    # Get function info from the resource
    function_name: str = uc_function.resource.full_name
    tool_name: str = uc_function.resource.name or function_name.replace(".", "_")

    logger.debug(
        "Creating UC tool with partial args",
        function_name=function_name,
        tool_name=tool_name,
        partial_args=list(resolved_args.keys()),
    )

    # Grant permissions if we have credentials (using ambient auth for setup)
    if "client_id" in resolved_args:
        client_id: str = resolved_args["client_id"]
        host: Optional[str] = resolved_args.get("host")
        try:
            _grant_function_permissions(function_name, client_id, host)
        except Exception as e:
            logger.warning(
                "Failed to grant permissions", function_name=function_name, error=str(e)
            )

    # Get workspace client for schema introspection (uses ambient auth at definition time)
    # Actual execution will use OBO via context
    setup_workspace_client: WorkspaceClient = uc_function.resource.workspace_client
    setup_client: DatabricksFunctionClient = DatabricksFunctionClient(
        client=setup_workspace_client
    )

    # Try to get the function schema for better tool definition
    schema_model: type[BaseModel]
    tool_description: str
    try:
        function_info: FunctionInfo = setup_client.get_function(function_name)
        schema_info = generate_function_input_params_schema(function_info)
        tool_description = (
            function_info.comment or f"Unity Catalog function: {function_name}"
        )

        logger.trace(
            "Generated function schema",
            function_name=function_name,
            schema=schema_info.pydantic_model.__name__,
        )

        # Create a modified schema that excludes partial args
        original_schema: type = schema_info.pydantic_model
        schema_model = _create_filtered_schema(original_schema, resolved_args.keys())
        logger.trace(
            "Filtered schema to exclude partial args",
            function_name=function_name,
            excluded_args=list(resolved_args.keys()),
        )

    except Exception as e:
        logger.warning(
            "Could not introspect function", function_name=function_name, error=str(e)
        )

        # Fallback to a generic schema
        class GenericUCParams(BaseModel):
            """Generic parameters for Unity Catalog function."""

            pass

        schema_model = GenericUCParams
        tool_description = f"Unity Catalog function: {function_name}"

    # Create a wrapper function that calls _execute_uc_function with partial args
    # Uses InjectedToolArg to ensure runtime is injected but hidden from the LLM
    def uc_function_wrapper(
        runtime: Annotated[ToolRuntime[Context], InjectedToolArg] = None,
        **kwargs: Any,
    ) -> str:
        """Wrapper function that executes Unity Catalog function with partial args."""
        # Get workspace client with OBO support via context
        context: Context | None = runtime.context if runtime else None
        workspace_client: WorkspaceClient = uc_function.resource.workspace_client_from(
            context
        )
        client: DatabricksFunctionClient = DatabricksFunctionClient(
            client=workspace_client
        )

        return _execute_uc_function(
            client=client,
            function_name=function_name,
            partial_args=resolved_args,
            **kwargs,
        )

    # Set the function name for the decorator
    uc_function_wrapper.__name__ = tool_name

    # Create the tool using LangChain's StructuredTool
    partial_tool: StructuredTool = StructuredTool.from_function(
        func=uc_function_wrapper,
        name=tool_name,
        description=tool_description,
        args_schema=schema_model,
    )

    return partial_tool
