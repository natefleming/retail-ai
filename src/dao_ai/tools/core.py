import asyncio
from collections import OrderedDict
from typing import Any, Callable, Optional, Sequence

from databricks_langchain import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig
from langgraph.types import interrupt
from loguru import logger
from mcp.types import ListToolsResult, Tool

from dao_ai.config import (
    AnyTool,
    BaseFunctionModel,
    FactoryFunctionModel,
    HumanInTheLoopModel,
    McpFunctionModel,
    PythonFunctionModel,
    ToolModel,
    TransportType,
    UnityCatalogFunctionModel,
)
from dao_ai.hooks.core import create_hooks
from dao_ai.utils import load_function


def add_human_in_the_loop(
    tool: RunnableLike,
    *,
    interrupt_config: HumanInterruptConfig | None = None,
    review_prompt: Optional[str] = "Please review the tool call",
) -> BaseTool:
    """
    Wrap a tool with human-in-the-loop functionality.
    This function takes a tool (either a callable or a BaseTool instance) and wraps it
    with a human-in-the-loop mechanism. When the tool is invoked, it will first
    request human review before executing the tool's logic. The human can choose to
    accept, edit the input, or provide a custom response.

    Args:
        tool (Callable[..., Any] | BaseTool): _description_
        interrupt_config (HumanInterruptConfig | None, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        BaseTool: _description_
    """
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    logger.debug(f"Wrapping tool {tool} with human-in-the-loop functionality")

    @create_tool(tool.name, description=tool.description, args_schema=tool.args_schema)
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input) -> Any:
        logger.debug(f"call_tool_with_interrupt: {tool.name} with input: {tool_input}")
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input,
            },
            "config": interrupt_config,
            "description": review_prompt,
        }

        logger.debug(f"Human interrupt request: {request}")
        response: dict[str, Any] = interrupt([request])[0]
        logger.debug(f"Human interrupt response: {response}")

        if response["type"] == "accept":
            tool_response = tool.invoke(tool_input, config=config)
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            tool_response = tool.invoke(tool_input, config=config)
        elif response["type"] == "response":
            user_feedback = response["args"]
            tool_response = user_feedback
        else:
            raise ValueError(f"Unknown interrupt response type: {response['type']}")

        return tool_response

    return call_tool_with_interrupt


def as_human_in_the_loop(
    tool: RunnableLike, function: BaseFunctionModel | str
) -> RunnableLike:
    if isinstance(function, BaseFunctionModel):
        human_in_the_loop: HumanInTheLoopModel | None = function.human_in_the_loop
        if human_in_the_loop:
            logger.debug(f"Adding human-in-the-loop to tool: {tool.name}")
            tool = add_human_in_the_loop(
                tool=tool,
                interrupt_config=human_in_the_loop.interupt_config,
                review_prompt=human_in_the_loop.review_prompt,
            )
    return tool


tool_registry: dict[str, Sequence[RunnableLike]] = {}


def create_tools(tool_models: Sequence[ToolModel]) -> Sequence[RunnableLike]:
    """
    Create a list of tools based on the provided configuration.

    This factory function generates a list of tools based on the specified configurations.
    Each tool is created according to its type and parameters defined in the configuration.

    Args:
        tool_configs: A sequence of dictionaries containing tool configurations

    Returns:
        A sequence of BaseTool objects created from the provided configurations
    """

    tools: OrderedDict[str, Sequence[RunnableLike]] = OrderedDict()

    for tool_config in tool_models:
        name: str = tool_config.name
        if name in tools:
            logger.warning(f"Tools already registered for: {name}, skipping creation.")
            continue
        registered_tools: Sequence[RunnableLike] = tool_registry.get(name)
        if registered_tools is None:
            logger.debug(f"Creating tools for: {name}...")
            function: AnyTool = tool_config.function
            registered_tools = create_hooks(function)
            logger.debug(f"Registering tools for: {tool_config}")
            tool_registry[name] = registered_tools
        else:
            logger.debug(f"Tools already registered for: {name}")

        tools[name] = registered_tools

    all_tools: Sequence[RunnableLike] = [
        t for tool_list in tools.values() for t in tool_list
    ]
    logger.debug(f"Created tools: {all_tools}")
    return all_tools


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


def create_factory_tool(
    function: FactoryFunctionModel,
) -> RunnableLike:
    """
    Create a factory tool from a FactoryFunctionModel.
    This factory function dynamically loads a Python function and returns it as a callable tool.
    Args:
        function: FactoryFunctionModel instance containing the function details
    Returns:
        A callable tool function that wraps the specified factory function
    """
    logger.debug(f"create_factory_tool: {function}")

    factory: Callable[..., Any] = load_function(function_name=function.full_name)
    tool: Callable[..., Any] = factory(**function.args)
    tool = as_human_in_the_loop(
        tool=tool,
        function=function,
    )
    return tool


def create_python_tool(
    function: PythonFunctionModel | str,
) -> RunnableLike:
    """
    Create a Python tool from a Python function model.
    This factory function wraps a Python function as a callable tool that can be
    invoked by agents during reasoning.
    Args:
        function: PythonFunctionModel instance containing the function details
    Returns:
        A callable tool function that wraps the specified Python function
    """
    logger.debug(f"create_python_tool: {function}")

    if isinstance(function, PythonFunctionModel):
        function = function.full_name

    # Load the Python function dynamically
    tool: Callable[..., Any] = load_function(function_name=function)

    tool = as_human_in_the_loop(
        tool=tool,
        function=function,
    )
    return tool


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


def search_tool() -> RunnableLike:
    logger.debug("search_tool")
    return DuckDuckGoSearchRun(output_format="list")
