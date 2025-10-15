import asyncio
from typing import Any, Sequence

from databricks_mcp import DatabricksOAuthClientProvider
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import tool as create_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from loguru import logger
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import ListToolsResult, Tool

from dao_ai.config import (
    McpFunctionModel,
    TransportType,
)
from dao_ai.tools.human_in_the_loop import as_human_in_the_loop


def create_mcp_tools(
    function: McpFunctionModel,
) -> Sequence[RunnableLike]:
    """
    Create tools for invoking Databricks MCP functions.

    Supports both direct MCP connections and UC Connection-based MCP access.
    Uses session-based approach to handle authentication token expiration properly.

    Based on: https://docs.databricks.com/aws/en/generative-ai/mcp/external-mcp
    """
    logger.debug(f"create_mcp_tools: {function}")

    # Check if using UC Connection or direct MCP connection
    if function.connection:
        # Use UC Connection approach with DatabricksOAuthClientProvider
        logger.debug(f"Using UC Connection for MCP: {function.connection.name}")

        # Construct URL if not provided
        if function.url:
            mcp_url = function.url
            logger.debug(f"Using provided MCP URL: {mcp_url}")
        else:
            # Construct URL from workspace host and connection name
            # Pattern: https://{workspace_host}/api/2.0/mcp/external/{connection_name}
            workspace_client = function.connection.workspace_client
            workspace_host = workspace_client.config.host
            connection_name = function.connection.name
            mcp_url = f"{workspace_host}/api/2.0/mcp/external/{connection_name}"
            logger.debug(f"Constructed MCP URL from connection: {mcp_url}")

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
            mcp_tools: list[Tool] | ListToolsResult = asyncio.run(_list_tools_with_connection())
            if isinstance(mcp_tools, ListToolsResult):
                mcp_tools = mcp_tools.tools

            logger.debug(
                f"Retrieved {len(mcp_tools)} MCP tools via UC Connection"
            )

        except Exception as e:
            logger.error(f"Failed to get tools from MCP server via UC Connection: {e}")
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
                logger.debug(f"Invoking MCP tool {mcp_tool.name} with fresh UC Connection session")
                workspace_client = function.connection.workspace_client

                try:
                    async with streamablehttp_client(
                        mcp_url, auth=DatabricksOAuthClientProvider(workspace_client)
                    ) as (read_stream, write_stream, _):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            result = await session.call_tool(mcp_tool.name, kwargs)
                            logger.debug(f"MCP tool {mcp_tool.name} completed successfully")
                            return result
                except Exception as e:
                    logger.error(f"MCP tool {mcp_tool.name} failed: {e}")
                    raise

            return as_human_in_the_loop(tool_wrapper, function)

        return [_create_tool_wrapper_with_connection(tool) for tool in mcp_tools]

    else:
        # Use direct MCP connection with MultiServerMCPClient
        logger.debug("Using direct MCP connection with MultiServerMCPClient")

        def _create_fresh_connection() -> dict[str, Any]:
            """Create connection config with fresh authentication headers."""
            logger.debug("Creating fresh connection...")

            if function.transport == TransportType.STDIO:
                return {
                    "command": function.command,
                    "args": function.args,
                    "transport": function.transport,
                }

            # For HTTP transport, generate fresh headers
            headers = function.headers.copy() if function.headers else {}

            if "Authorization" not in headers:
                logger.debug("Generating fresh authentication token for MCP function")

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
                    logger.debug("Generated fresh authentication token")
                except Exception as e:
                    logger.error(f"Failed to create fresh token: {e}")
            else:
                logger.debug("Using existing authentication token")

            return {
                "url": function.url,
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
                logger.error(f"Failed to list MCP tools: {e}")
                return []

        # Note: This still needs to run sync during tool creation/registration
        # The actual tool execution will be async
        try:
            mcp_tools: list[Tool] | ListToolsResult = asyncio.run(_list_mcp_tools())
            if isinstance(mcp_tools, ListToolsResult):
                mcp_tools = mcp_tools.tools

            logger.debug(f"Retrieved {len(mcp_tools)} MCP tools")
        except Exception as e:
            logger.error(f"Failed to get tools from MCP server: {e}")
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
                logger.debug(f"Invoking MCP tool {mcp_tool.name} with fresh session")

                connection = _create_fresh_connection()
                client = MultiServerMCPClient({function.name: connection})

                try:
                    async with client.session(function.name) as session:
                        return await session.call_tool(mcp_tool.name, kwargs)
                except Exception as e:
                    logger.error(f"MCP tool {mcp_tool.name} failed: {e}")
                    raise

            return as_human_in_the_loop(tool_wrapper, function)

        return [_create_tool_wrapper(tool) for tool in mcp_tools]
