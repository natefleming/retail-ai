"""
Tool retry middleware for DAO AI agents.

Automatically retries failed tool calls with configurable exponential backoff.

Example:
    from dao_ai.middleware import create_tool_retry_middleware

    # Retry failed tool calls with exponential backoff
    middleware = create_tool_retry_middleware(
        max_retries=3,
        backoff_factor=2.0,
        initial_delay=1.0,
    )
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.tools import BaseTool
from loguru import logger

from dao_ai.config import BaseFunctionModel, ToolModel

__all__ = [
    "ToolRetryMiddleware",
    "create_tool_retry_middleware",
]


def _resolve_tools(
    tools: list[str | ToolModel | dict[str, Any]] | None,
) -> list[str] | None:
    """
    Resolve tool specs to a list of tool name strings.

    Returns None if tools is None (apply to all tools).
    """
    if tools is None:
        return None

    result: list[str] = []
    for tool in tools:
        if isinstance(tool, str):
            result.append(tool)
        elif isinstance(tool, dict):
            try:
                tool_model = ToolModel(**tool)
                result.extend(_extract_tool_names(tool_model))
            except Exception as e:
                raise ValueError(f"Failed to construct ToolModel from dict: {e}") from e
        elif isinstance(tool, ToolModel):
            result.extend(_extract_tool_names(tool))
        else:
            raise TypeError(
                f"Tool must be str, ToolModel, or dict, got {type(tool).__name__}"
            )

    return result if result else None


def _extract_tool_names(tool_model: ToolModel) -> list[str]:
    """Extract tool names from ToolModel, falling back to ToolModel.name."""
    function = tool_model.function

    if not isinstance(function, BaseFunctionModel):
        return [tool_model.name]

    try:
        tool_names = [
            tool.name
            for tool in function.as_tools()
            if isinstance(tool, BaseTool) and tool.name
        ]
        return tool_names if tool_names else [tool_model.name]
    except Exception:
        return [tool_model.name]


def create_tool_retry_middleware(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float | None = None,
    jitter: bool = False,
    tools: list[str | ToolModel | dict[str, Any]] | None = None,
    retry_on: tuple[type[Exception], ...] | Callable[[Exception], bool] | None = None,
    on_failure: Literal["continue", "error"] | Callable[[Exception], str] = "continue",
) -> ToolRetryMiddleware:
    """
    Create a ToolRetryMiddleware for automatic tool call retries.

    Handles transient failures in external API calls with exponential backoff.

    Args:
        max_retries: Max retry attempts after initial call. Default 3.
        backoff_factor: Multiplier for exponential backoff. Default 2.0.
            Delay = initial_delay * (backoff_factor ** retry_number)
            Set to 0.0 for constant delay.
        initial_delay: Initial delay in seconds before first retry. Default 1.0.
        max_delay: Max delay in seconds (caps exponential growth). None = no cap.
        jitter: Add ±25% random jitter to avoid thundering herd. Default False.
        tools: List of tools to apply retry to. Can be:
            - None: Apply to all tools (default)
            - list of str: Tool names
            - list of ToolModel: DAO AI tool models
            - list of dict: Tool config dicts
        retry_on: When to retry:
            - None: Retry on all errors (default)
            - tuple of Exception types: Retry only on these
            - callable: Function(exception) -> bool
        on_failure: Behavior when all retries exhausted:
            - "continue": Return error message, let agent continue (default)
            - "error": Re-raise exception, stop execution
            - callable: Function(exception) -> str for custom message

    Returns:
        List containing ToolRetryMiddleware instance

    Example:
        # Basic retry with defaults
        retry = create_tool_retry_middleware()

        # Retry specific tools with custom backoff
        retry = create_tool_retry_middleware(
            max_retries=5,
            backoff_factor=1.5,
            initial_delay=0.5,
            tools=["search_web", "query_database"],
        )

        # Retry only on specific exceptions
        retry = create_tool_retry_middleware(
            max_retries=3,
            retry_on=(TimeoutError, ConnectionError),
            on_failure="error",
        )
    """
    tool_names = _resolve_tools(tools)

    logger.debug(
        "Creating tool retry middleware",
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        max_delay=max_delay,
        jitter=jitter,
        tools=tool_names or "all",
        on_failure=on_failure if isinstance(on_failure, str) else "custom",
    )

    # Build kwargs
    kwargs: dict[str, Any] = {
        "max_retries": max_retries,
        "backoff_factor": backoff_factor,
        "initial_delay": initial_delay,
        "on_failure": on_failure,
    }

    if tool_names is not None:
        kwargs["tools"] = tool_names

    if max_delay is not None:
        kwargs["max_delay"] = max_delay

    if jitter:
        kwargs["jitter"] = jitter

    if retry_on is not None:
        kwargs["retry_on"] = retry_on

    return ToolRetryMiddleware(**kwargs)
