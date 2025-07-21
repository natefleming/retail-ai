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
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools import tool as create_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig
from langgraph.types import interrupt
from loguru import logger

from dao_ai.config import (
    AnyTool,
    BaseFunctionModel,
    FactoryFunctionModel,
    HumanInTheLoopActionType,
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
    decline_message: str = "Tool call declined by user",
    custom_actions: Optional[dict[str, str]] = None,
) -> BaseTool:
    """
    Wrap a tool with enhanced human-in-the-loop functionality.

    This function takes a tool and wraps it with a human-in-the-loop mechanism that
    supports multiple actions including accept, edit, response, and decline.

    Args:
        tool: The tool to wrap (callable or BaseTool instance)
        interrupt_config: LangGraph interrupt configuration
        review_prompt: Custom prompt for the human reviewer
        decline_message: Message to return when user declines
        custom_actions: Custom action handlers

    Returns:
        Enhanced BaseTool with human-in-the-loop functionality
    """
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
            "allow_decline": True,
        }

    if custom_actions is None:
        custom_actions = {}

    logger.debug(
        f"Wrapping tool {tool.name} with enhanced human-in-the-loop functionality"
    )

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

        response_type = response.get("type")

        if response_type == HumanInTheLoopActionType.ACCEPT:
            logger.info(f"Tool {tool.name} accepted by user")
            return tool.invoke(tool_input, config=config)

        elif response_type == HumanInTheLoopActionType.EDIT:
            logger.info(f"Tool {tool.name} edited by user")
            edited_args = response.get("args", {}).get("args", tool_input)
            return tool.invoke(edited_args, config=config)

        elif response_type == HumanInTheLoopActionType.RESPONSE:
            logger.info(f"Tool {tool.name} responded to by user")
            return response.get("args", "User provided custom response")

        elif response_type == HumanInTheLoopActionType.DECLINE:
            logger.info(f"Tool {tool.name} declined by user")
            return decline_message

        elif response_type in custom_actions:
            logger.info(f"Tool {tool.name} handled with custom action: {response_type}")
            return custom_actions[response_type]

        else:
            error_message = f"Unknown interrupt response type: {response_type}. Supported types: {', '.join([t.value for t in HumanInTheLoopActionType])}"
            logger.error(error_message)
            raise ValueError(error_message)

    return call_tool_with_interrupt


def as_human_in_the_loop(
    tool: RunnableLike, function: BaseFunctionModel | str
) -> RunnableLike:
    """
    Apply human-in-the-loop configuration to a tool based on function model settings.

    Args:
        tool: The tool to potentially wrap
        function: Function model that may contain human-in-the-loop configuration

    Returns:
        The tool, potentially wrapped with human-in-the-loop functionality
    """
    if isinstance(function, BaseFunctionModel):
        human_in_the_loop: HumanInTheLoopModel | None = function.human_in_the_loop
        if human_in_the_loop:
            logger.debug(f"Adding human-in-the-loop to tool: {tool.name}")
            tool = add_human_in_the_loop(
                tool=tool,
                interrupt_config=human_in_the_loop.interrupt_config,
                review_prompt=human_in_the_loop.review_prompt,
                decline_message=human_in_the_loop.decline_message,
                custom_actions=human_in_the_loop.custom_actions,
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
    Create a tool for invoking a Databricks MCP function.

    This factory function wraps a Databricks MCP function as callable tools that can be
    invoked by agents during reasoning. Handles both sync and async invocation patterns.

    Args:
        function: McpFunctionModel instance containing the function details

    Returns:
        A callable tool function that wraps the specified MCP function
    """
    logger.debug(f"create_mcp_tools: {function}")

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
                "headers": function.headers or {},
            }

    client: MultiServerMCPClient = MultiServerMCPClient({function.name: connection})

    async_tools: Sequence[RunnableLike] = asyncio.run(client.get_tools()) or []

    logger.debug(f"Retrieved async tools: {async_tools}")

    # Wrap StructuredTool instances to support sync invocation
    sync_tools: list[RunnableLike] = []
    for async_tool in async_tools:
        if isinstance(async_tool, StructuredTool):

            @create_tool(
                async_tool.name,
                description=async_tool.description,
                args_schema=async_tool.args_schema,
            )
            def sync_wrapper(**kwargs):
                return asyncio.run(async_tool.ainvoke(kwargs))

            sync_tools.append(sync_wrapper)
        else:
            sync_tools.append(async_tool)

    tools = [
        as_human_in_the_loop(
            tool=tool,
            function=function,
        )
        for tool in sync_tools
    ]

    return tools


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

    factory: Callable[..., Any] = load_function(function.full_name)
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

    function_name: str = (
        function.full_name if isinstance(function, PythonFunctionModel) else function
    )

    # Load the Python function dynamically
    tool: Callable[..., Any] = load_function(function_name)

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

    function_name: str = (
        function.full_name
        if isinstance(function, UnityCatalogFunctionModel)
        else function
    )

    client: DatabricksFunctionClient = DatabricksFunctionClient()

    toolkit: UCFunctionToolkit = UCFunctionToolkit(
        function_names=[function_name], client=client
    )

    tools = toolkit.tools or []

    logger.debug(f"Retrieved tools: {tools}")

    tools = [as_human_in_the_loop(tool=tool, function=function) for tool in tools]

    return tools


def search_tool() -> RunnableLike:
    logger.debug("search_tool")
    return DuckDuckGoSearchRun(output_format="list")
