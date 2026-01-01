"""
MCP (Model Context Protocol) tool creation for LangChain agents.

This module provides tools for connecting to MCP servers using the
MCP SDK and langchain-mcp-adapters library.

For compatibility with Databricks APIs, we use manual tool wrappers
that give us full control over the response format.

Reference: https://docs.langchain.com/oss/python/langchain/mcp
"""

import asyncio
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


def create_mcp_tools(
    function: McpFunctionModel,
) -> Sequence[RunnableLike]:
    """
    Create tools for invoking Databricks MCP functions.

    Supports both direct MCP connections and UC Connection-based MCP access.
    Uses manual tool wrappers to ensure response format compatibility with
    Databricks APIs (which reject extra fields in tool results).

    Based on: https://docs.databricks.com/aws/en/generative-ai/mcp/external-mcp

    Args:
        function: The MCP function model configuration.

    Returns:
        A sequence of LangChain tools that can be used by agents.
    """
    mcp_url = function.mcp_url
    logger.debug("Creating MCP tools", mcp_url=mcp_url)

    connection_config = _build_connection_config(function)

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

    # Create client to list available tools
    client = MultiServerMCPClient({"mcp_function": connection_config})

    async def _list_tools() -> list[Tool]:
        """List available MCP tools from the server."""
        async with client.session("mcp_function") as session:
            result = await session.list_tools()
            return result.tools if hasattr(result, "tools") else list(result)

    try:
        mcp_tools: list[Tool] = asyncio.run(_list_tools())

        # Log discovered tools
        logger.info(
            "Discovered MCP tools",
            tools_count=len(mcp_tools),
            mcp_url=mcp_url,
        )
        for mcp_tool in mcp_tools:
            logger.debug(
                "MCP tool discovered",
                tool_name=mcp_tool.name,
                tool_description=(
                    mcp_tool.description[:100] if mcp_tool.description else None
                ),
            )

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
