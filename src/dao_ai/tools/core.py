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
from langchain_core.tools import BaseTool
from loguru import logger

from dao_ai.config import (
    AnyTool,
    BaseFunctionModel,
    McpFunctionModel,
    ToolModel,
)
from dao_ai.hooks.core import create_hooks
from dao_ai.state import Context

# Module-level tool registry for caching created tools
tool_registry: dict[str, Sequence[RunnableLike]] = {}


def _tool_names(tools: Sequence[RunnableLike]) -> list[str]:
    """Extract the ``.name`` of every BaseTool in a sequence."""
    return [tool.name for tool in tools if isinstance(tool, BaseTool) and tool.name]


def resolve_tool_names(tool_model: ToolModel) -> list[str]:
    """
    Resolve a ToolModel to the runtime tool names it produces.

    Per-tool middleware (call limits, HITL, audit) must key on the actual tool
    names the LLM calls, which a single ToolModel can expand to several of
    (e.g. an MCP server or a wildcard Unity Catalog function). This is the one
    resolution path those scans should share.

    Resolution order, cheapest and most authoritative first:

    1. ``tool_registry`` — if ``create_tools`` already built this tool during
       the same agent build, reuse those exact objects. This avoids a second
       connect-and-enumerate round-trip for toolkit functions and guarantees
       the names match the tools the agent will actually run.
    2. ``function.as_tools()`` — build the tools to read their names. Used when
       the registry has no entry (e.g. resolution outside the agent-build path).
    3. ``tool_model.name`` — last-resort fallback when the function is a bare
       string reference or ``as_tools()`` yields nothing / raises.

    Args:
        tool_model: The tool configuration to resolve.

    Returns:
        A list of runtime tool-name strings (never empty; falls back to the
        ToolModel's own name).
    """
    # 1. Reuse already-built tools from this agent build when available.
    cached = tool_registry.get(tool_model.name)
    if cached is not None:
        names = _tool_names(cached)
        if names:
            return names

    function = tool_model.function

    # String function references can't be introspected.
    if not isinstance(function, BaseFunctionModel):
        return [tool_model.name]

    # 2. Build the tools to read their names.
    try:
        names = _tool_names(function.as_tools())
        if names:
            return names
    except Exception as e:
        logger.warning(
            "Error resolving tool names from ToolModel",
            tool_model_name=tool_model.name,
            error=str(e),
        )

    # 3. Fall back to the ToolModel's configured name.
    logger.debug(
        "Falling back to ToolModel.name for tool-name resolution",
        tool_model_name=tool_model.name,
    )
    return [tool_model.name]


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
            try:
                registered_tools = create_hooks(function)
            except Exception as e:
                # An on-behalf-of-user MCP server may reject discovery
                # (tools/list) under the identity present at graph-build time —
                # e.g. the app service principal has not linked the underlying
                # SaaS account, so servers that gate tools/list on a linked
                # credential (Atlassian, GitHub, Google Drive, …) 403. That must
                # not crash the whole agent: skip this tool with a loud warning
                # so the remaining tools still load. The tool becomes usable once
                # the caller's identity is linked (OBO) or its schema is supplied
                # at deploy time. Non-OBO tools (and non-MCP tools) still raise —
                # those failures are genuine misconfiguration.
                if isinstance(function, McpFunctionModel) and function.on_behalf_of_user:
                    logger.warning(
                        "Skipping OBO MCP tool that failed discovery at build time",
                        tool_name=name,
                        error=str(e),
                        note=(
                            "The agent will start without this tool. It requires "
                            "the calling identity to have linked the MCP server's "
                            "credential (OBO), or its tool schema supplied at "
                            "deploy time. See dao-ai#305."
                        ),
                    )
                    continue
                raise
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
