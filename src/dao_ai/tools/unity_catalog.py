from typing import Any, Dict, Optional, Sequence, Tuple, Union

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import PermissionsChange, Privilege
from databricks_langchain import DatabricksFunctionClient, UCFunctionToolkit
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import StructuredTool
from loguru import logger
from pydantic import BaseModel, Field, create_model
from unitycatalog.ai.langchain.toolkit import (
    generate_function_input_params_schema,
    get_tool_name,
)

from dao_ai.config import (
    CompositeVariableModel,
    ToolModel,
    UnityCatalogFunctionModel,
)
from dao_ai.tools.human_in_the_loop import as_human_in_the_loop


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

    if isinstance(function, UnityCatalogFunctionModel):
        function = function.full_name

    client: DatabricksFunctionClient = DatabricksFunctionClient()

    toolkit: UCFunctionToolkit = UCFunctionToolkit(
        function_names=[function], client=client
    )

    tools = toolkit.tools or []

    logger.debug(f"Retrieved tools: {tools}")

    tools = [as_human_in_the_loop(tool=tool, function=function) for tool in tools]

    return tools


def _normalize_inputs(
    tool: Union[ToolModel, Dict[str, Any]],
    host: Union[CompositeVariableModel, Dict[str, Any]],
    client_id: Optional[Union[CompositeVariableModel, Dict[str, Any]]],
    client_secret: Optional[Union[CompositeVariableModel, Dict[str, Any]]],
) -> Tuple[
    ToolModel,
    CompositeVariableModel,
    Optional[CompositeVariableModel],
    Optional[CompositeVariableModel],
    Any,
]:
    """Normalize input parameters to their proper types."""
    if isinstance(tool, dict):
        tool = ToolModel(**tool)
    if isinstance(host, dict):
        host = CompositeVariableModel(**host)
    if isinstance(client_id, dict) and client_id:
        client_id = CompositeVariableModel(**client_id)
    if isinstance(client_secret, dict) and client_secret:
        client_secret = CompositeVariableModel(**client_secret)

    unity_catalog_function = tool.function
    if isinstance(unity_catalog_function, dict):
        unity_catalog_function = UnityCatalogFunctionModel(**unity_catalog_function)

    return tool, host, client_id, client_secret, unity_catalog_function


def _get_base_fields(schema: type[BaseModel]) -> Dict[str, Any]:
    """Extract fields from Pydantic v2 schema."""
    return schema.model_fields


def _create_auth_fields(
    base_fields: Dict[str, Any],
    host: Optional[CompositeVariableModel],
    client_id: Optional[CompositeVariableModel],
    client_secret: Optional[CompositeVariableModel],
) -> Dict[str, Tuple[type, Any]]:
    """Create authentication fields that don't conflict with existing function parameters."""
    auth_fields: Dict[str, Tuple[type, Any]] = {}

    if "host" not in base_fields and host:
        auth_fields["host"] = (
            str,
            Field(default=host.as_value(), description="Host URL for authentication"),
        )
    if "client_id" not in base_fields and client_id:
        auth_fields["client_id"] = (
            str,
            Field(
                default=client_id.as_value(), description="Client ID for authentication"
            ),
        )
    if "client_secret" not in base_fields and client_secret:
        auth_fields["client_secret"] = (
            str,
            Field(
                default=client_secret.as_value(),
                description="Client secret for authentication",
            ),
        )

    return auth_fields


def _merge_schema_fields(
    base_schema: type[BaseModel], auth_fields: Dict[str, Tuple[type, Any]]
) -> Dict[str, Tuple[type, Any]]:
    """Merge base schema fields with auth fields (Pydantic v2)."""
    base_fields = _get_base_fields(base_schema)

    all_fields = {
        **{name: (field.annotation, field) for name, field in base_fields.items()},
        **auth_fields,
    }

    return all_fields


def _execute_uc_function(
    client: DatabricksFunctionClient,
    function_name: str,
    host: Optional[CompositeVariableModel],
    client_id: Optional[CompositeVariableModel],
    client_secret: Optional[CompositeVariableModel],
    **kwargs: Any,
) -> str:
    """Execute Unity Catalog function with authentication and provided parameters."""
    # Prepare authentication parameters
    auth_params: Dict[str, Optional[str]] = {
        k: v.as_value() if v else None
        for k, v in [
            ("host", host),
            ("client_id", client_id),
            ("client_secret", client_secret),
        ]
        if v is not None
    }

    # Merge all parameters
    all_params: Dict[str, Any] = {**auth_params, **kwargs}

    logger.debug(
        f"Calling UC function {function_name} with parameters: {list(all_params.keys())}"
    )

    result = client.execute_function(function_name=function_name, parameters=all_params)

    # Handle errors and extract result
    if hasattr(result, "error") and result.error:
        logger.error(f"Unity Catalog function error: {result.error}")
        raise RuntimeError(f"Function execution failed: {result.error}")

    result_value: str = result.value if hasattr(result, "value") else str(result)
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


def create_authenticating_uc_tool(
    tool: Union[ToolModel, Dict[str, Any]],
    host: Union[CompositeVariableModel, Dict[str, Any]],
    client_id: Optional[Union[CompositeVariableModel, Dict[str, Any]]] = None,
    client_secret: Optional[Union[CompositeVariableModel, Dict[str, Any]]] = None,
) -> StructuredTool:
    """
    Create a generalized authenticated Unity Catalog tool.

    This function dynamically introspects the Unity Catalog function schema
    to create a StructuredTool with the correct parameters and description.

    Args:
        tool: ToolModel containing the Unity Catalog function configuration
        host: Host URL for authentication
        client_id: Client ID for authentication (optional)
        client_secret: Client secret for authentication (optional)

    Returns:
        StructuredTool: A LangChain tool that can invoke the UC function with authentication
    """

    logger.debug(f"create_authenticating_uc_tool: {tool}")
    # Normalize inputs
    (
        normalized_tool,
        normalized_host,
        normalized_client_id,
        normalized_client_secret,
        unity_catalog_function,
    ) = _normalize_inputs(tool, host, client_id, client_secret)

    logger.debug(f"Creating authenticated UC tool: {normalized_tool.name}")

    client: DatabricksFunctionClient = DatabricksFunctionClient()
    function_name: str = unity_catalog_function.full_name

    # Grant permissions to the service principal if client_id is provided
    if normalized_client_id:
        client_id_value = normalized_client_id.as_value()
        host_value = normalized_host.as_value() if normalized_host else None
        _grant_function_permissions(function_name, client_id_value, host_value)

    # Try to get function schema
    try:
        function_info = client.get_function(function_name)
        schema_info = generate_function_input_params_schema(function_info)
        base_schema: type[BaseModel] = schema_info.pydantic_model

        # Create enhanced schema with auth parameters
        base_fields: Dict[str, Any] = _get_base_fields(base_schema)
        auth_fields: Dict[str, Tuple[type, Any]] = _create_auth_fields(
            base_fields, normalized_host, normalized_client_id, normalized_client_secret
        )
        all_fields: Dict[str, Tuple[type, Any]] = _merge_schema_fields(
            base_schema, auth_fields
        )

        enhanced_schema_name: str = (
            f"{get_tool_name(function_name)}_AuthenticatedParams"
        )
        EnhancedSchema = create_model(enhanced_schema_name, **all_fields)  # type: ignore[call-overload]

        tool_name: str = get_tool_name(function_name)
        description: str = (
            function_info.comment or f"Unity Catalog function: {function_name}"
        )

    except Exception as e:
        logger.warning(f"Could not introspect function {function_name}: {e}")
        return _create_fallback_authenticated_tool(
            normalized_tool,
            unity_catalog_function,
            normalized_host,
            normalized_client_id,
            normalized_client_secret,
        )

    # Create the tool function
    async def authenticated_tool_func(**kwargs: Any) -> str:
        """Execute the Unity Catalog function with authentication parameters."""
        return _execute_uc_function(
            client,
            function_name,
            normalized_host,
            normalized_client_id,
            normalized_client_secret,
            **kwargs,
        )

    structured_tool = StructuredTool.from_function(
        coroutine=authenticated_tool_func,
        name=tool_name,
        description=description,
        args_schema=EnhancedSchema,
        parse_docstring=False,
    )

    return as_human_in_the_loop(tool=structured_tool, function=function_name)


def _create_fallback_authenticated_tool(
    tool: ToolModel,
    unity_catalog_function: UnityCatalogFunctionModel,
    host: CompositeVariableModel,
    client_id: Optional[CompositeVariableModel],
    client_secret: Optional[CompositeVariableModel],
) -> StructuredTool:
    """Create a fallback tool when function introspection fails."""

    class FallbackSchema(BaseModel):
        """Fallback schema for when function introspection fails"""

        parameters: Dict[str, Any] = Field(
            description="Function parameters as key-value pairs", default_factory=dict
        )

    client: DatabricksFunctionClient = DatabricksFunctionClient()
    function_name: str = unity_catalog_function.full_name

    async def fallback_tool_func(parameters: Dict[str, Any]) -> str:
        """Execute the Unity Catalog function with fallback parameter handling."""
        logger.debug(f"Calling UC function {function_name} with fallback parameters")
        return _execute_uc_function(
            client, function_name, host, client_id, client_secret, **parameters
        )

    structured_tool = StructuredTool.from_function(
        coroutine=fallback_tool_func,
        name=tool.name,
        description=f"Unity Catalog function: {unity_catalog_function.full_name}",
        args_schema=FallbackSchema,
        parse_docstring=False,
    )

    return as_human_in_the_loop(tool=structured_tool, function=function_name)
