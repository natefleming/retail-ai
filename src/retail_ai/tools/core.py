import asyncio
from collections import OrderedDict
from typing import Any, Callable, Sequence

from databricks_langchain import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger

from retail_ai.config import (
    AgentEndpointFunctionModel,
    AnyTool,
    FactoryFunctionModel,
    McpFunctionModel,
    PythonFunctionModel,
    ToolModel,
    TransportType,
    UnityCatalogFunctionModel,
)
from retail_ai.utils import load_function

tool_registry: dict[str, BaseTool] = {}


def create_tools(tool_models: Sequence[ToolModel]) -> Sequence[BaseTool]:
    """
    Create a list of tools based on the provided configuration.

    This factory function generates a list of tools based on the specified configurations.
    Each tool is created according to its type and parameters defined in the configuration.

    Args:
        tool_configs: A sequence of dictionaries containing tool configurations

    Returns:
        A sequence of BaseTool objects created from the provided configurations
    """

    tools: OrderedDict[str, BaseTool] = OrderedDict()

    for tool_config in tool_models:
        name: str = tool_config.name
        if name in tools:
            logger.warning(f"Tool {name} already exists, skipping creation.")
            continue
        tool: BaseTool = tool_registry.get(name)
        if tool is None:
            logger.debug(f"Creating tool: {name}...")
            function: AnyTool = tool_config.function
            if isinstance(function, str):
                function = PythonFunctionModel(name=function)
            tool = function.as_tool()
            logger.debug(f"Registering tool: {tool_config}")
            tool_registry[name] = tool
        else:
            logger.debug(f"Tool {name} already registered.")

        tools[name] = tool

    return list(tools.values())


def create_mcp_tool(
    function: McpFunctionModel,
) -> Callable[..., Any]:
    """
    Create a tool for invoking a Databricks MCP function.

    This factory function wraps a Databricks MCP function as a callable tool that can be
    invoked by agents during reasoning.

    Args:
        function: McpFunctionModel instance containing the function details

    Returns:
        A callable tool function that wraps the specified MCP function
    """
    logger.debug(f"create_mcp_tool: {function}")

    connection: dict[str, Any]
    match function.transport:
        case TransportType.STDIO:
            connection = {
                "command": function.command,
                "args": function.args,
                "transport": function.transport,
            }
        case TransportType.STREAMABLE_HTTP:
            connection = {
                "url": function.url,
                "transport": function.transport,
            }

    client: MultiServerMCPClient = MultiServerMCPClient({function.name: connection})

    tools = asyncio.run(client.get_tools())
    tool = next(iter(tools or []), None)
    return tool


def create_factory_tool(
    function: FactoryFunctionModel,
) -> Callable[..., Any]:
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
    return tool


def create_python_tool(
    function: PythonFunctionModel | str,
) -> Callable[..., Any]:
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
    return tool


def create_uc_tool(function: UnityCatalogFunctionModel | str) -> Sequence[BaseTool]:
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

    logger.debug(f"create_uc_tool: {function}")

    if isinstance(function, UnityCatalogFunctionModel):
        function = function.full_name

    client: DatabricksFunctionClient = DatabricksFunctionClient()

    toolkit: UCFunctionToolkit = UCFunctionToolkit(
        function_names=[function], client=client
    )

    tool = next(iter(toolkit.tools or []), None)
    return tool


def create_agent_endpoint_tool(
    function: AgentEndpointFunctionModel,
) -> Callable[..., Any]:
    logger.debug(f"create_agent_endpoint_tool: {function}")

    def agent_endpoint(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        logger.debug(f"agent_endpoint: {function}, state: {state}, config: {config}")
        messages: Sequence[BaseMessage] = state["messages"]
        model: LanguageModelLike = function.model.as_chat_model()
        response: AIMessage = model.invoke(messages, config)
        return {"messages": [response]}

    return agent_endpoint


def search_tool() -> BaseTool:
    logger.debug("search_tool")
    return DuckDuckGoSearchRun(output_format="list")
