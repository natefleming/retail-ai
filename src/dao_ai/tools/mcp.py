import asyncio
from typing import Any, Sequence

from databricks_mcp import DatabricksOAuthClientProvider
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import tool as create_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import ListToolsResult, Tool

from dao_ai.config import (
    McpFunctionModel,
    TransportType,
)


def create_mcp_tools(
    function: McpFunctionModel,
) -> Sequence[RunnableLike]:
    """
    Create tools for invoking Databricks MCP functions.

    Supports both direct MCP connections and UC Connection-based MCP access.
    Uses session-based approach to handle authentication token expiration properly.

    Based on: https://docs.databricks.com/aws/en/generative-ai/mcp/external-mcp
    """
    # Get MCP URL - handles all convenience objects (connection, genie_room, warehouse, etc.)
    mcp_url = function.mcp_url
    logger.debug("Creating MCP tools", function_name=function.name, mcp_url=mcp_url)

    # Check if using UC Connection or direct MCP connection
    if function.connection:
        # Use UC Connection approach with DatabricksOAuthClientProvider
        logger.debug(
            "Using UC Connection for MCP", connection_name=function.connection.name
        )

        async def _list_tools_with_connection():
            """List available tools using DatabricksOAuthClientProvider."""
            workspace_client = function.connection.workspace_client

            async with streamablehttp_client(
                mcp_url, auth=DatabricksOAuthClientProvider(workspace_client)
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize and list tools
                    await session.initialize()
                    return await session.list_tools()

        try:
            mcp_tools: list[Tool] | ListToolsResult = asyncio.run(
                _list_tools_with_connection()
            )
            if isinstance(mcp_tools, ListToolsResult):
                mcp_tools = mcp_tools.tools

            logger.debug(
                "Retrieved MCP tools via UC Connection", tools_count=len(mcp_tools)
            )

        except Exception as e:
            logger.error(
                "Failed to get tools from MCP server via UC Connection",
                connection_name=function.connection.name,
                error=str(e),
            )
            raise RuntimeError(
                f"Failed to list MCP tools for function '{function.name}' via UC Connection '{function.connection.name}': {e}"
            )

        # Create wrapper tools with fresh session per invocation
        def _create_tool_wrapper_with_connection(mcp_tool: Tool) -> RunnableLike:
            @create_tool(
                mcp_tool.name,
                description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
                args_schema=mcp_tool.inputSchema,
            )
            async def tool_wrapper(**kwargs):
                """Execute MCP tool with fresh UC Connection session."""
                logger.trace(
                    "Invoking MCP tool with UC Connection", tool_name=mcp_tool.name
                )
                workspace_client = function.connection.workspace_client

                try:
                    async with streamablehttp_client(
                        mcp_url, auth=DatabricksOAuthClientProvider(workspace_client)
                    ) as (read_stream, write_stream, _):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            result = await session.call_tool(mcp_tool.name, kwargs)
                            logger.trace("MCP tool completed", tool_name=mcp_tool.name)
                            return result
                except Exception as e:
                    logger.error(
                        "MCP tool failed", tool_name=mcp_tool.name, error=str(e)
                    )
                    raise

            # HITL is now handled at middleware level via HumanInTheLoopMiddleware
            return tool_wrapper

        return [_create_tool_wrapper_with_connection(tool) for tool in mcp_tools]

    else:
        # Use direct MCP connection with MultiServerMCPClient
        logger.debug("Using direct MCP connection", function_name=function.name)

        def _create_fresh_connection() -> dict[str, Any]:
            """Create connection config with fresh authentication headers."""
            logger.trace("Creating fresh MCP connection", function_name=function.name)

            if function.transport == TransportType.STDIO:
                return {
                    "command": function.command,
                    "args": function.args,
                    "transport": function.transport,
                }

            # For HTTP transport, generate fresh headers
            headers = function.headers.copy() if function.headers else {}

            if "Authorization" not in headers:
                logger.trace(
                    "Generating fresh authentication token", function_name=function.name
                )

                from dao_ai.config import value_of
                from dao_ai.providers.databricks import DatabricksProvider

                try:
                    provider = DatabricksProvider(
                        workspace_host=value_of(function.workspace_host),
                        client_id=value_of(function.client_id),
                        client_secret=value_of(function.client_secret),
                        pat=value_of(function.pat),
                    )
                    headers["Authorization"] = f"Bearer {provider.create_token()}"
                    logger.trace(
                        "Generated fresh authentication token",
                        function_name=function.name,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to create fresh token",
                        function_name=function.name,
                        error=str(e),
                    )
            else:
                logger.trace(
                    "Using existing authentication token", function_name=function.name
                )

            return {
                "url": mcp_url,  # Use the resolved MCP URL
                "transport": function.transport,
                "headers": headers,
            }

        # Get available tools from MCP server
        async def _list_mcp_tools():
            connection = _create_fresh_connection()
            client = MultiServerMCPClient({function.name: connection})

            try:
                async with client.session(function.name) as session:
                    return await session.list_tools()
            except Exception as e:
                logger.error(
                    "Failed to list MCP tools",
                    function_name=function.name,
                    error=str(e),
                )
                return []

        # Note: This still needs to run sync during tool creation/registration
        # The actual tool execution will be async
        try:
            mcp_tools: list[Tool] | ListToolsResult = asyncio.run(_list_mcp_tools())
            if isinstance(mcp_tools, ListToolsResult):
                mcp_tools = mcp_tools.tools

            logger.debug(
                "Retrieved MCP tools",
                function_name=function.name,
                tools_count=len(mcp_tools),
            )
        except Exception as e:
            logger.error(
                "Failed to get tools from MCP server",
                function_name=function.name,
                transport=function.transport,
                error=str(e),
            )
            raise RuntimeError(
                f"Failed to list MCP tools for function '{function.name}' with transport '{function.transport}' and URL '{function.url}': {e}"
            )

        # Create wrapper tools with fresh session per invocation
        def _create_tool_wrapper(mcp_tool: Tool) -> RunnableLike:
            @create_tool(
                mcp_tool.name,
                description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
                args_schema=mcp_tool.inputSchema,
            )
            async def tool_wrapper(**kwargs):
                """Execute MCP tool with fresh session and authentication."""
                logger.trace(
                    "Invoking MCP tool with fresh session", tool_name=mcp_tool.name
                )

                connection = _create_fresh_connection()
                client = MultiServerMCPClient({function.name: connection})

                try:
                    async with client.session(function.name) as session:
                        return await session.call_tool(mcp_tool.name, kwargs)
                except Exception as e:
                    logger.error(
                        "MCP tool failed", tool_name=mcp_tool.name, error=str(e)
                    )
                    raise

            # HITL is now handled at middleware level via HumanInTheLoopMiddleware
            return tool_wrapper

        return [_create_tool_wrapper(tool) for tool in mcp_tools]
