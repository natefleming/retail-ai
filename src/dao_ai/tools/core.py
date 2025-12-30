"""
Core tool creation infrastructure for DAO AI.

This module provides the foundational tool creation and registration system:
- Tool registry for caching created tools
- Factory function for creating tools from configuration
- Example tools demonstrating runtime context usage

This is "core" because it contains the essential infrastructure that all
tool creation flows through, not because it contains all tools.
"""

from collections import OrderedDict
from typing import Sequence

from langchain.tools import ToolRuntime, tool
from langchain_core.runnables.base import RunnableLike
from loguru import logger

from dao_ai.config import (
    AnyTool,
    ToolModel,
)
from dao_ai.hooks.core import create_hooks
from dao_ai.state import Context

# Module-level tool registry for caching created tools
tool_registry: dict[str, Sequence[RunnableLike]] = {}


def create_tools(tool_models: Sequence[ToolModel]) -> Sequence[RunnableLike]:
    """
    Create a list of tools based on the provided configuration.

    This factory function generates a list of tools based on the specified configurations.
    Each tool is created according to its type and parameters defined in the configuration.

    Args:
        tool_models: A sequence of ToolModel configurations

    Returns:
        A sequence of BaseTool objects created from the provided configurations
    """

    tools: OrderedDict[str, Sequence[RunnableLike]] = OrderedDict()

    for tool_config in tool_models:
        name: str = tool_config.name
        if name in tools:
            logger.warning("Tools already registered, skipping", tool_name=name)
            continue
        registered_tools: Sequence[RunnableLike] | None = tool_registry.get(name)
        if registered_tools is None:
            logger.trace("Creating tools", tool_name=name)
            function: AnyTool = tool_config.function
            registered_tools = create_hooks(function)
            logger.trace("Registering tools", tool_name=name)
            tool_registry[name] = registered_tools
        else:
            logger.trace("Tools already registered", tool_name=name)

        tools[name] = registered_tools

    all_tools: Sequence[RunnableLike] = [
        t for tool_list in tools.values() for t in tool_list
    ]
    logger.debug("Tools created", tools_count=len(all_tools))
    return all_tools


# =============================================================================
# Example Tools
# =============================================================================
# The following tools serve as examples and are included here because they
# demonstrate core patterns (like ToolRuntime usage) rather than because they
# are fundamental infrastructure. They're simple enough to colocate with the
# core tool creation logic.


@tool
def say_hello_tool(
    name: str | None = None,
    runtime: ToolRuntime[Context] = None,
) -> str:
    """
    Say hello to someone by name.

    This is an example tool demonstrating how to use ToolRuntime to access
    runtime context (like user_id) within a tool.

    If no name is provided, uses the user_id from the runtime context.

    Args:
        name: Optional name of the person to greet. If not provided,
              uses user_id from context.
        runtime: Runtime context (automatically injected, not provided by user)

    Returns:
        A greeting string
    """
    # Use provided name, or fall back to user_id from context
    if name is None:
        if runtime and runtime.context:
            user_id: str | None = runtime.context.user_id
            if user_id:
                name = user_id
            else:
                name = "there"  # Default fallback
        else:
            name = "there"  # Default fallback

    return f"Hello, {name}!"
