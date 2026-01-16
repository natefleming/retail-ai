"""
MCP (Model Context Protocol) tool creation for LangChain agents.

This module provides tools for connecting to MCP servers using the
MCP SDK and langchain-mcp-adapters library.

For compatibility with Databricks APIs, we use manual tool wrappers
that give us full control over the response format.

Public API:
- list_mcp_tools(): List available tools from an MCP server (for discovery/UI)
- create_mcp_tools(): Create LangChain tools for agent execution

Reference: https://docs.langchain.com/oss/python/langchain/mcp
"""

import asyncio
import fnmatch
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import tool as create_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger
from mcp.types import CallToolResult, TextContent, Tool

from dao_ai.config import (
    IsDatabricksResource,
    McpFunctionModel,
    TransportType,
)
from dao_ai.state import Context


@dataclass
class MCPToolInfo:
    """
    Information about an MCP tool for display and selection.

    This is a simplified representation of an MCP tool that contains
    only the information needed for UI display and tool selection.
    It's designed to be easily serializable for use in web UIs.

    Attributes:
        name: The unique identifier/name of the tool
        description: Human-readable description of what the tool does
        input_schema: JSON Schema describing the tool's input parameters
    """

    name: str
    description: str | None
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def _matches_pattern(tool_name: str, patterns: list[str]) -> bool:
    """
    Check if tool name matches any of the provided patterns.

    Supports glob patterns:
    - * matches any characters
    - ? matches single character
    - [abc] matches any char in set
    - [!abc] matches any char NOT in set

    Args:
        tool_name: Name of the tool to check
        patterns: List of exact names or glob patterns

    Returns:
        True if tool name matches any pattern

    Examples:
        >>> _matches_pattern("query_sales", ["query_*"])
        True
        >>> _matches_pattern("list_tables", ["query_*"])
        False
        >>> _matches_pattern("tool_a", ["tool_?"])
        True
    """
    for pattern in patterns:
        if fnmatch.fnmatch(tool_name, pattern):
            return True
    return False


def _should_include_tool(
    tool_name: str,
    include_tools: list[str] | None,
    exclude_tools: list[str] | None,
) -> bool:
    """
    Determine if a tool should be included based on include/exclude filters.

    Logic:
    1. If exclude_tools specified and tool matches: EXCLUDE (highest priority)
    2. If include_tools specified and tool matches: INCLUDE
    3. If include_tools specified and tool doesn't match: EXCLUDE
    4. If no filters specified: INCLUDE (default)

    Args:
        tool_name: Name of the tool
        include_tools: Optional list of tools/patterns to include
        exclude_tools: Optional list of tools/patterns to exclude

    Returns:
        True if tool should be included

    Examples:
        >>> _should_include_tool("query_sales", ["query_*"], None)
        True
        >>> _should_include_tool("drop_table", None, ["drop_*"])
        False
        >>> _should_include_tool("query_sales", ["query_*"], ["*_sales"])
        False  # exclude takes precedence
    """
    # Exclude has highest priority
    if exclude_tools and _matches_pattern(tool_name, exclude_tools):
        logger.debug("Tool excluded by exclude_tools", tool_name=tool_name)
        return False

    # If include list exists, tool must match it
    if include_tools:
        if _matches_pattern(tool_name, include_tools):
            logger.debug("Tool included by include_tools", tool_name=tool_name)
            return True
        else:
            logger.debug(
                "Tool not in include_tools",
                tool_name=tool_name,
                include_patterns=include_tools,
            )
            return False

    # Default: include all tools
    return True


def _has_auth_configured(resource: IsDatabricksResource) -> bool:
    """Check if a resource has explicit authentication configured."""
    return bool(
        resource.on_behalf_of_user
        or resource.service_principal
        or resource.client_id
        or resource.pat
    )


def _get_auth_resource(function: McpFunctionModel) -> IsDatabricksResource:
    """
    Get the IsDatabricksResource to use for authentication.

    Follows a priority hierarchy:
    1. Nested resource with explicit auth (app, connection, genie_room, vector_search)
    2. McpFunctionModel itself (which also inherits from IsDatabricksResource)

    Only uses a nested resource if it has authentication configured.
    Otherwise falls back to McpFunctionModel which may have credentials set at the tool level.

    Returns the resource whose workspace_client should be used for authentication.
    """
    # Check each possible resource source - only use if it has auth configured
    if function.app and _has_auth_configured(function.app):
        return function.app
    if function.connection and _has_auth_configured(function.connection):
        return function.connection
    if function.genie_room and _has_auth_configured(function.genie_room):
        return function.genie_room
    if function.vector_search and _has_auth_configured(function.vector_search):
        return function.vector_search
    # SchemaModel (functions) doesn't have auth - always fall through

    # Fall back to McpFunctionModel itself (it inherits from IsDatabricksResource)
    # This allows credentials to be set at the tool level
    return function


def _build_connection_config(
    function: McpFunctionModel,
    context: Context | None = None,
) -> dict[str, Any]:
    """
    Build the connection configuration dictionary for MultiServerMCPClient.

    Authentication Strategy:
    -----------------------
    For HTTP transport, authentication is handled consistently using
    DatabricksOAuthClientProvider with the workspace_client from the appropriate
    IsDatabricksResource. The auth resource is selected in this priority:

    1. Nested resource (app, connection, genie_room, vector_search) if it has auth
    2. McpFunctionModel itself (inherits from IsDatabricksResource)

    This approach ensures:
    - Consistent auth handling across all MCP sources
    - Automatic token refresh for long-running connections
    - Support for OBO, service principal, PAT, and ambient auth

    Args:
        function: The MCP function model configuration.
        context: Optional runtime context with headers for OBO auth.

    Returns:
        A dictionary containing the transport-specific connection settings.
    """
    if function.transport == TransportType.STDIO:
        return {
            "command": function.command,
            "args": function.args,
            "transport": function.transport.value,
        }

    # For HTTP transport, use DatabricksOAuthClientProvider with unified auth
    from databricks.sdk import WorkspaceClient
    from databricks_mcp import DatabricksOAuthClientProvider

    # Get the resource to use for authentication
    auth_resource: IsDatabricksResource = _get_auth_resource(function)

    # Get workspace client from the auth resource with OBO support via context
    workspace_client: WorkspaceClient = auth_resource.workspace_client_from(context)
    auth_provider: DatabricksOAuthClientProvider = DatabricksOAuthClientProvider(
        workspace_client
    )

    # Log which resource is providing auth
    resource_name = (
        getattr(auth_resource, "name", None) or auth_resource.__class__.__name__
    )
    logger.trace(
        "Using DatabricksOAuthClientProvider for authentication",
        auth_resource=resource_name,
        resource_type=auth_resource.__class__.__name__,
    )

    return {
        "url": function.mcp_url,
        "transport": "http",
        "auth": auth_provider,
    }


def _extract_text_content(result: CallToolResult) -> str:
    """
    Extract text content from an MCP CallToolResult.

    Converts the MCP result content to a plain string format that is
    compatible with all LLM APIs (avoiding extra fields like 'id').

    Args:
        result: The MCP tool call result.

    Returns:
        A string containing the concatenated text content.
    """
    if not result.content:
        return ""

    text_parts: list[str] = []
    for item in result.content:
        if isinstance(item, TextContent):
            text_parts.append(item.text)
        elif hasattr(item, "text"):
            # Handle other content types that have text
            text_parts.append(str(item.text))
        else:
            # Fallback: convert to string representation
            text_parts.append(str(item))

    return "\n".join(text_parts)


async def _afetch_tools_from_server(function: McpFunctionModel) -> list[Tool]:
    """
    Async version: Fetch raw MCP tools from the server.

    This is the primary async implementation that handles the actual MCP connection
    and tool listing. It's used by both alist_mcp_tools() and acreate_mcp_tools().

    Args:
        function: The MCP function model configuration.

    Returns:
        List of raw MCP Tool objects from the server.

    Raises:
        RuntimeError: If connection to MCP server fails.
    """
    connection_config = _build_connection_config(function)
    client = MultiServerMCPClient({"mcp_function": connection_config})

    try:
        async with client.session("mcp_function") as session:
            result = await session.list_tools()
            return result.tools if hasattr(result, "tools") else list(result)
    except Exception as e:
        if function.connection:
            logger.error(
                "Failed to get tools from MCP server via UC Connection",
                connection_name=function.connection.name,
                error=str(e),
            )
            raise RuntimeError(
                f"Failed to list MCP tools via UC Connection "
                f"'{function.connection.name}': {e}"
            ) from e
        else:
            logger.error(
                "Failed to get tools from MCP server",
                transport=function.transport,
                url=function.url,
                error=str(e),
            )
            raise RuntimeError(
                f"Failed to list MCP tools with transport '{function.transport}' "
                f"and URL '{function.url}': {e}"
            ) from e


def _fetch_tools_from_server(function: McpFunctionModel) -> list[Tool]:
    """
    Sync wrapper: Fetch raw MCP tools from the server.

    For async contexts, use _afetch_tools_from_server() directly.

    Args:
        function: The MCP function model configuration.

    Returns:
        List of raw MCP Tool objects from the server.

    Raises:
        RuntimeError: If connection to MCP server fails.
    """
    return asyncio.run(_afetch_tools_from_server(function))


async def alist_mcp_tools(
    function: McpFunctionModel,
    apply_filters: bool = True,
) -> list[MCPToolInfo]:
    """
    Async version: List available tools from an MCP server.

    This is the primary async implementation for tool discovery.
    For sync contexts, use list_mcp_tools() instead.

    Args:
        function: The MCP function model configuration.
        apply_filters: Whether to apply include_tools/exclude_tools filters.

    Returns:
        List of MCPToolInfo objects describing available tools.

    Raises:
        RuntimeError: If connection to MCP server fails.
    """
    mcp_url = function.mcp_url
    logger.debug(
        "Listing MCP tools (async)", mcp_url=mcp_url, apply_filters=apply_filters
    )

    # Log connection type
    if function.connection:
        logger.debug(
            "Using UC Connection for MCP",
            connection_name=function.connection.name,
            mcp_url=mcp_url,
        )
    else:
        logger.debug(
            "Using direct connection for MCP",
            transport=function.transport,
            mcp_url=mcp_url,
        )

    # Fetch tools from server (async)
    mcp_tools: list[Tool] = await _afetch_tools_from_server(function)

    # Log discovered tools
    logger.info(
        "Discovered MCP tools from server",
        tools_count=len(mcp_tools),
        tool_names=[t.name for t in mcp_tools],
        mcp_url=mcp_url,
    )

    # Apply filtering if requested and configured
    if apply_filters and (function.include_tools or function.exclude_tools):
        original_count = len(mcp_tools)
        mcp_tools = [
            tool
            for tool in mcp_tools
            if _should_include_tool(
                tool.name,
                function.include_tools,
                function.exclude_tools,
            )
        ]
        filtered_count = original_count - len(mcp_tools)

        logger.info(
            "Filtered MCP tools",
            original_count=original_count,
            filtered_count=filtered_count,
            final_count=len(mcp_tools),
            include_patterns=function.include_tools,
            exclude_patterns=function.exclude_tools,
        )

    # Convert to MCPToolInfo for cleaner API
    tool_infos: list[MCPToolInfo] = []
    for mcp_tool in mcp_tools:
        tool_info = MCPToolInfo(
            name=mcp_tool.name,
            description=mcp_tool.description,
            input_schema=mcp_tool.inputSchema or {},
        )
        tool_infos.append(tool_info)

        logger.debug(
            "MCP tool available",
            tool_name=mcp_tool.name,
            tool_description=(
                mcp_tool.description[:100] if mcp_tool.description else None
            ),
        )

    return tool_infos


def list_mcp_tools(
    function: McpFunctionModel,
    apply_filters: bool = True,
) -> list[MCPToolInfo]:
    """
    Sync wrapper: List available tools from an MCP server.

    For async contexts, use alist_mcp_tools() directly.

    Args:
        function: The MCP function model configuration.
        apply_filters: Whether to apply include_tools/exclude_tools filters.

    Returns:
        List of MCPToolInfo objects describing available tools.

    Raises:
        RuntimeError: If connection to MCP server fails.
    """
    return asyncio.run(alist_mcp_tools(function, apply_filters))


async def acreate_mcp_tools(
    function: McpFunctionModel,
) -> Sequence[RunnableLike]:
    """
    Async version: Create executable LangChain tools for invoking Databricks MCP functions.

    This is the primary async implementation. For sync contexts, use create_mcp_tools().

    Args:
        function: The MCP function model configuration.

    Returns:
        A sequence of LangChain tools that can be used by agents.

    Raises:
        RuntimeError: If connection to MCP server fails.
    """
    mcp_url = function.mcp_url
    logger.debug("Creating MCP tools (async)", mcp_url=mcp_url)

    # Fetch tools from server (async)
    mcp_tools: list[Tool] = await _afetch_tools_from_server(function)

    # Log discovered tools
    logger.info(
        "Discovered MCP tools from server",
        tools_count=len(mcp_tools),
        tool_names=[t.name for t in mcp_tools],
        mcp_url=mcp_url,
    )

    # Apply filtering if configured
    if function.include_tools or function.exclude_tools:
        original_count = len(mcp_tools)
        mcp_tools = [
            tool
            for tool in mcp_tools
            if _should_include_tool(
                tool.name,
                function.include_tools,
                function.exclude_tools,
            )
        ]
        filtered_count = original_count - len(mcp_tools)

        logger.info(
            "Filtered MCP tools",
            original_count=original_count,
            filtered_count=filtered_count,
            final_count=len(mcp_tools),
            include_patterns=function.include_tools,
            exclude_patterns=function.exclude_tools,
        )

    # Log final tool list
    for mcp_tool in mcp_tools:
        logger.debug(
            "MCP tool available",
            tool_name=mcp_tool.name,
            tool_description=(
                mcp_tool.description[:100] if mcp_tool.description else None
            ),
        )

    def _create_tool_wrapper(mcp_tool: Tool) -> RunnableLike:
        """
        Create a LangChain tool wrapper for an MCP tool.

        Supports OBO authentication via context headers.
        """
        from langchain.tools import ToolRuntime

        @create_tool(
            mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            args_schema=mcp_tool.inputSchema,
        )
        async def tool_wrapper(
            runtime: ToolRuntime[Context] = None,
            **kwargs: Any,
        ) -> str:
            """Execute MCP tool with fresh session."""
            logger.trace("Invoking MCP tool", tool_name=mcp_tool.name, args=kwargs)

            # Get context for OBO support
            context: Context | None = runtime.context if runtime else None

            invocation_client: MultiServerMCPClient = MultiServerMCPClient(
                {"mcp_function": _build_connection_config(function, context)}
            )

            try:
                async with invocation_client.session("mcp_function") as session:
                    result: CallToolResult = await session.call_tool(
                        mcp_tool.name, kwargs
                    )

                    text_result: str = _extract_text_content(result)

                    logger.trace(
                        "MCP tool completed",
                        tool_name=mcp_tool.name,
                        result_length=len(text_result),
                    )

                    return text_result

            except Exception as e:
                logger.error(
                    "MCP tool failed",
                    tool_name=mcp_tool.name,
                    error=str(e),
                )
                raise

        return tool_wrapper

    return [_create_tool_wrapper(tool) for tool in mcp_tools]


def create_mcp_tools(
    function: McpFunctionModel,
) -> Sequence[RunnableLike]:
    """
    Sync wrapper: Create executable LangChain tools for invoking Databricks MCP functions.

    For async contexts, use acreate_mcp_tools() directly.

    Args:
        function: The MCP function model configuration.

    Returns:
        A sequence of LangChain tools that can be used by agents.

    Raises:
        RuntimeError: If connection to MCP server fails.
    """
    mcp_url = function.mcp_url
    logger.debug("Creating MCP tools", mcp_url=mcp_url)

    # Fetch and filter tools using shared logic
    # We need the raw Tool objects here, not MCPToolInfo
    mcp_tools: list[Tool] = _fetch_tools_from_server(function)

    # Log discovered tools
    logger.info(
        "Discovered MCP tools from server",
        tools_count=len(mcp_tools),
        tool_names=[t.name for t in mcp_tools],
        mcp_url=mcp_url,
    )

    # Apply filtering if configured
    if function.include_tools or function.exclude_tools:
        original_count = len(mcp_tools)
        mcp_tools = [
            tool
            for tool in mcp_tools
            if _should_include_tool(
                tool.name,
                function.include_tools,
                function.exclude_tools,
            )
        ]
        filtered_count = original_count - len(mcp_tools)

        logger.info(
            "Filtered MCP tools",
            original_count=original_count,
            filtered_count=filtered_count,
            final_count=len(mcp_tools),
            include_patterns=function.include_tools,
            exclude_patterns=function.exclude_tools,
        )

    # Log final tool list
    for mcp_tool in mcp_tools:
        logger.debug(
            "MCP tool available",
            tool_name=mcp_tool.name,
            tool_description=(
                mcp_tool.description[:100] if mcp_tool.description else None
            ),
        )

    def _create_tool_wrapper(mcp_tool: Tool) -> RunnableLike:
        """
        Create a LangChain tool wrapper for an MCP tool.

        This wrapper handles:
        - Fresh session creation per invocation (stateless)
        - Content extraction to plain text (avoiding extra fields)
        - OBO authentication via context headers
        """
        from langchain.tools import ToolRuntime

        @create_tool(
            mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            args_schema=mcp_tool.inputSchema,
        )
        async def tool_wrapper(
            runtime: ToolRuntime[Context] = None,
            **kwargs: Any,
        ) -> str:
            """Execute MCP tool with fresh session."""
            logger.trace("Invoking MCP tool", tool_name=mcp_tool.name, args=kwargs)

            # Get context for OBO support
            context: Context | None = runtime.context if runtime else None

            # Create a fresh client/session for each invocation with OBO support
            invocation_client: MultiServerMCPClient = MultiServerMCPClient(
                {"mcp_function": _build_connection_config(function, context)}
            )

            try:
                async with invocation_client.session("mcp_function") as session:
                    result: CallToolResult = await session.call_tool(
                        mcp_tool.name, kwargs
                    )

                    # Extract text content, avoiding extra fields
                    text_result: str = _extract_text_content(result)

                    logger.trace(
                        "MCP tool completed",
                        tool_name=mcp_tool.name,
                        result_length=len(text_result),
                    )

                    return text_result

            except Exception as e:
                logger.error(
                    "MCP tool failed",
                    tool_name=mcp_tool.name,
                    error=str(e),
                )
                raise

        return tool_wrapper

    return [_create_tool_wrapper(tool) for tool in mcp_tools]
