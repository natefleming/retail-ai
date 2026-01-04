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
    McpFunctionModel,
    TransportType,
    value_of,
)


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


def _build_connection_config(
    function: McpFunctionModel,
) -> dict[str, Any]:
    """
    Build the connection configuration dictionary for MultiServerMCPClient.

    Args:
        function: The MCP function model configuration.

    Returns:
        A dictionary containing the transport-specific connection settings.
    """
    if function.transport == TransportType.STDIO:
        return {
            "command": function.command,
            "args": function.args,
            "transport": function.transport.value,
        }

    # For HTTP transport with UC Connection, use DatabricksOAuthClientProvider
    if function.connection:
        from databricks_mcp import DatabricksOAuthClientProvider

        workspace_client = function.connection.workspace_client
        auth_provider = DatabricksOAuthClientProvider(workspace_client)

        logger.trace(
            "Using DatabricksOAuthClientProvider for authentication",
            connection_name=function.connection.name,
        )

        return {
            "url": function.mcp_url,
            "transport": "http",
            "auth": auth_provider,
        }

    # For HTTP transport with headers-based authentication
    headers: dict[str, str] = {
        key: str(value_of(val)) for key, val in function.headers.items()
    }

    if "Authorization" not in headers:
        logger.trace("Generating fresh authentication token")

        from dao_ai.providers.databricks import DatabricksProvider

        try:
            provider = DatabricksProvider(
                workspace_host=value_of(function.workspace_host),
                client_id=value_of(function.client_id),
                client_secret=value_of(function.client_secret),
                pat=value_of(function.pat),
            )
            headers["Authorization"] = f"Bearer {provider.create_token()}"
            logger.trace("Generated fresh authentication token")
        except Exception as e:
            logger.error("Failed to create fresh token", error=str(e))
    else:
        logger.trace("Using existing authentication token")

    return {
        "url": function.mcp_url,
        "transport": "http",
        "headers": headers,
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


def _fetch_tools_from_server(function: McpFunctionModel) -> list[Tool]:
    """
    Fetch raw MCP tools from the server.

    This is the core async operation that connects to the MCP server
    and retrieves the list of available tools.

    Args:
        function: The MCP function model configuration.

    Returns:
        List of raw MCP Tool objects from the server.

    Raises:
        RuntimeError: If connection to MCP server fails.
    """
    connection_config = _build_connection_config(function)
    client = MultiServerMCPClient({"mcp_function": connection_config})

    async def _list_tools_async() -> list[Tool]:
        """Async helper to list tools from MCP server."""
        async with client.session("mcp_function") as session:
            result = await session.list_tools()
            return result.tools if hasattr(result, "tools") else list(result)

    try:
        return asyncio.run(_list_tools_async())
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


def list_mcp_tools(
    function: McpFunctionModel,
    apply_filters: bool = True,
) -> list[MCPToolInfo]:
    """
    List available tools from an MCP server.

    This function connects to an MCP server and returns information about
    all available tools. It's designed for:
    - Tool discovery and exploration
    - UI-based tool selection (e.g., in DAO AI Builder)
    - Debugging and validation of MCP configurations

    The returned MCPToolInfo objects contain all information needed to
    display tools in a UI and allow users to select which tools to use.

    Args:
        function: The MCP function model configuration containing:
            - Connection details (url, connection, headers, etc.)
            - Optional filtering (include_tools, exclude_tools)
        apply_filters: Whether to apply include_tools/exclude_tools filters.
            Set to False to get the complete list of available tools
            regardless of filter configuration. Default True.

    Returns:
        List of MCPToolInfo objects describing available tools.
        Each contains name, description, and input_schema.

    Raises:
        RuntimeError: If connection to MCP server fails.

    Example:
        # List all tools from a DBSQL MCP server
        from dao_ai.config import McpFunctionModel
        from dao_ai.tools.mcp import list_mcp_tools

        function = McpFunctionModel(sql=True)
        tools = list_mcp_tools(function)

        for tool in tools:
            print(f"{tool.name}: {tool.description}")

        # Get unfiltered list (ignore include_tools/exclude_tools)
        all_tools = list_mcp_tools(function, apply_filters=False)

    Note:
        For creating executable LangChain tools, use create_mcp_tools() instead.
        This function is for discovery/display purposes only.
    """
    mcp_url = function.mcp_url
    logger.debug("Listing MCP tools", mcp_url=mcp_url, apply_filters=apply_filters)

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

    # Fetch tools from server
    mcp_tools: list[Tool] = _fetch_tools_from_server(function)

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


def create_mcp_tools(
    function: McpFunctionModel,
) -> Sequence[RunnableLike]:
    """
    Create executable LangChain tools for invoking Databricks MCP functions.

    Supports both direct MCP connections and UC Connection-based MCP access.
    Uses manual tool wrappers to ensure response format compatibility with
    Databricks APIs (which reject extra fields in tool results).

    This function:
    1. Fetches available tools from the MCP server
    2. Applies include_tools/exclude_tools filters
    3. Wraps each tool for LangChain agent execution

    For tool discovery without creating executable tools, use list_mcp_tools().

    Based on: https://docs.databricks.com/aws/en/generative-ai/mcp/external-mcp

    Args:
        function: The MCP function model configuration containing:
            - Connection details (url, connection, headers, etc.)
            - Optional filtering (include_tools, exclude_tools)

    Returns:
        A sequence of LangChain tools that can be used by agents.

    Raises:
        RuntimeError: If connection to MCP server fails.

    Example:
        from dao_ai.config import McpFunctionModel
        from dao_ai.tools.mcp import create_mcp_tools

        function = McpFunctionModel(sql=True)
        tools = create_mcp_tools(function)

        # Use tools in an agent
        agent = create_agent(model=model, tools=tools)
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
        """

        @create_tool(
            mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            args_schema=mcp_tool.inputSchema,
        )
        async def tool_wrapper(**kwargs: Any) -> str:
            """Execute MCP tool with fresh session."""
            logger.trace("Invoking MCP tool", tool_name=mcp_tool.name, args=kwargs)

            # Create a fresh client/session for each invocation
            invocation_client = MultiServerMCPClient(
                {"mcp_function": _build_connection_config(function)}
            )

            try:
                async with invocation_client.session("mcp_function") as session:
                    result: CallToolResult = await session.call_tool(
                        mcp_tool.name, kwargs
                    )

                    # Extract text content, avoiding extra fields
                    text_result = _extract_text_content(result)

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
