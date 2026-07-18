"""
Tool call limit middleware for DAO AI agents.

This module provides a factory for creating LangChain's ToolCallLimitMiddleware
from DAO AI configuration.

Example:
    from dao_ai.middleware import create_tool_call_limit_middleware

    # Global limit across all tools
    middleware = create_tool_call_limit_middleware(
        thread_limit=20,
        run_limit=10,
    )

    # Limit specific tool by name
    search_limiter = create_tool_call_limit_middleware(
        tool="search_web",
        run_limit=3,
        exit_behavior="continue",
    )
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from langchain.agents.middleware import ToolCallLimitMiddleware
from loguru import logger

from dao_ai.config import BaseFunctionModel, ToolModel

__all__ = [
    "ToolCallLimitMiddleware",
    "create_tool_call_limit_middleware",
    "create_tool_call_limit_middlewares_from_tool_models",
]


def _resolve_tool(tool: str | ToolModel | dict[str, Any]) -> list[str]:
    """
    Resolve tool argument to a list of actual tool names.

    Args:
        tool: String name, ToolModel, or dict to resolve

    Returns:
        List of tool name strings

    Raises:
        ValueError: If dict cannot be converted to ToolModel
        TypeError: If tool is not a supported type
    """
    # String: return as single-item list
    if isinstance(tool, str):
        return [tool]

    # Dict: convert to ToolModel first
    if isinstance(tool, dict):
        try:
            tool_model = ToolModel(**tool)
        except Exception as e:
            raise ValueError(
                f"Failed to construct ToolModel from dict: {e}\n"
                f"Dict must have 'name' and 'function' keys."
            ) from e
    elif isinstance(tool, ToolModel):
        tool_model = tool
    else:
        raise TypeError(
            f"tool must be str, ToolModel, or dict, got {type(tool).__name__}"
        )

    # Extract tool names from ToolModel
    return _extract_tool_names(tool_model)


def _extract_tool_names(tool_model: ToolModel) -> list[str]:
    """
    Extract actual tool names from a ToolModel.

    A single ToolModel can produce multiple tools (e.g., UC functions, MCP
    servers). Delegates to the shared ``resolve_tool_names`` so tool-name
    resolution reuses already-built tools from the agent-build registry
    (avoiding a second connect-and-enumerate for toolkit functions) and stays
    consistent with the HITL/audit scans. Falls back to ``ToolModel.name``.
    """
    from dao_ai.tools import resolve_tool_names

    return resolve_tool_names(tool_model)


def create_tool_call_limit_middleware(
    tool: str | ToolModel | dict[str, Any] | None = None,
    thread_limit: int | None = None,
    run_limit: int | None = None,
    exit_behavior: Literal["continue", "error", "end"] = "continue",
) -> ToolCallLimitMiddleware:
    """
    Create a ToolCallLimitMiddleware with graceful termination support.

    Factory for LangChain's ToolCallLimitMiddleware that supports DAO AI
    configuration types.

    Args:
        tool: Tool to limit. Can be:
            - None: Global limit on all tools
            - str: Limit specific tool by name
            - ToolModel: Limit tool(s) from DAO AI config
            - dict: Tool config dict (converted to ToolModel)
        thread_limit: Max calls per thread (conversation). Requires checkpointer.
        run_limit: Max calls per run (single invocation).
        exit_behavior: What to do when limit hit:
            - "continue": Block tool with error message, let agent continue
            - "error": Raise ToolCallLimitExceededError immediately
            - "end": Stop execution gracefully (single-tool only)

    Returns:
        A ToolCallLimitMiddleware instance. If ToolModel produces multiple tools,
        only the first tool is used (with a warning logged).

    Raises:
        ValueError: If no limits specified, or invalid dict
        TypeError: If tool is unsupported type

    Example:
        # Global limit
        limiter = create_tool_call_limit_middleware(run_limit=10)

        # Tool-specific limit
        limiter = create_tool_call_limit_middleware(
            tool="search_web",
            run_limit=3,
            exit_behavior="continue",
        )
    """
    if thread_limit is None and run_limit is None:
        raise ValueError("At least one of thread_limit or run_limit must be specified.")

    # Global limit: no tool parameter
    if tool is None:
        logger.debug(
            "Creating global tool call limit",
            thread_limit=thread_limit,
            run_limit=run_limit,
            exit_behavior=exit_behavior,
        )
        return ToolCallLimitMiddleware(
            thread_limit=thread_limit,
            run_limit=run_limit,
            exit_behavior=exit_behavior,
        )

    # Resolve to list of tool names
    names = _resolve_tool(tool)

    # Use first tool name (warn if multiple)
    tool_name = names[0]
    if len(names) > 1:
        logger.warning(
            "ToolModel resolved to multiple tool names, using first only",
            tool_names=names,
            using=tool_name,
        )

    logger.debug(
        "Creating tool call limit middleware",
        tool_name=tool_name,
        thread_limit=thread_limit,
        run_limit=run_limit,
        exit_behavior=exit_behavior,
    )

    return ToolCallLimitMiddleware(
        tool_name=tool_name,
        thread_limit=thread_limit,
        run_limit=run_limit,
        exit_behavior=exit_behavior,
    )


def create_tool_call_limit_middlewares_from_tool_models(
    tool_models: Sequence[ToolModel],
) -> list[ToolCallLimitMiddleware]:
    """
    Create ToolCallLimitMiddleware instances from ToolModel configurations.

    Scans tool_models for functions that declare a ``call_limit`` and creates
    one ToolCallLimitMiddleware per resolved runtime tool name. This is the
    primary entry point used by agent node creation — it lets a limit follow a
    tool (declared once on the tool's ``function``) rather than being repeated
    on every agent's ``middleware`` list.

    Unlike passing a ToolModel straight to ``create_tool_call_limit_middleware``
    (which limits only the first resolved tool name), this scan resolves every
    name a function expands to via ``function.as_tools()`` and limits each one.

    Args:
        tool_models: List of ToolModel configurations from agent config

    Returns:
        A list of ToolCallLimitMiddleware, one per resolved tool name that has
        a call_limit configured. Empty if no tool declares a call_limit.

    Example:
        from dao_ai.config import ToolModel, PythonFunctionModel, ToolCallLimitModel

        tool_models = [
            ToolModel(
                name="search",
                function=SearchToolModel(call_limit=ToolCallLimitModel(run_limit=3)),
            ),
        ]

        middlewares = create_tool_call_limit_middlewares_from_tool_models(tool_models)
    """
    middlewares: list[ToolCallLimitMiddleware] = []

    for tool_model in tool_models:
        function = tool_model.function

        if not isinstance(function, BaseFunctionModel):
            continue

        call_limit = function.call_limit
        if call_limit is None:
            continue

        # Resolve every runtime tool name this function expands to and limit
        # each one independently (each gets a unique
        # ToolCallLimitMiddleware[<tool>] name in LangChain).
        for tool_name in _extract_tool_names(tool_model):
            middleware = create_tool_call_limit_middleware(
                tool=tool_name,
                thread_limit=call_limit.thread_limit,
                run_limit=call_limit.run_limit,
                exit_behavior=call_limit.exit_behavior,
            )
            middlewares.append(middleware)
            logger.trace(
                "Registered tool call limit from tool model",
                tool_model_name=tool_model.name,
                tool_name=tool_name,
                run_limit=call_limit.run_limit,
                thread_limit=call_limit.thread_limit,
                exit_behavior=call_limit.exit_behavior,
            )

    return middlewares
