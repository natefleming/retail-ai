import asyncio
from typing import Any, Sequence

from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import tool as create_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger
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

    Uses session-based approach to handle authentication token expiration properly.
    """
    logger.debug(f"create_mcp_tools: {function}")

    def _create_fresh_connection() -> dict[str, Any]:
        logger.debug("Creating fresh connection...")
        """Create connection config with fresh authentication headers."""
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

        response = {
            "url": function.url,
            "transport": function.transport,
            "headers": headers,
        }

        return response

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

    try:
        mcp_tools: list | ListToolsResult = asyncio.run(_list_mcp_tools())
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
        def tool_wrapper(**kwargs):
            """Execute MCP tool with fresh session and authentication."""
            logger.debug(f"Invoking MCP tool {mcp_tool.name} with fresh session")

            async def _invoke():
                connection = _create_fresh_connection()
                client = MultiServerMCPClient({function.name: connection})

                try:
                    async with client.session(function.name) as session:
                        return await session.call_tool(mcp_tool.name, kwargs)
                except Exception as e:
                    logger.error(f"MCP tool {mcp_tool.name} failed: {e}")
                    raise

            return asyncio.run(_invoke())

        return as_human_in_the_loop(tool_wrapper, function)

    return [_create_tool_wrapper(tool) for tool in mcp_tools]
