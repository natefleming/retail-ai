from collections import OrderedDict
from typing import Sequence

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables.base import RunnableLike
from loguru import logger

from dao_ai.config import (
    AnyTool,
    ToolModel,
)
from dao_ai.hooks.core import create_hooks

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


def search_tool() -> RunnableLike:
    logger.debug("search_tool")
    return DuckDuckGoSearchRun(output_format="list")
