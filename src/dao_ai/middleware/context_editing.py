"""
Context editing middleware for DAO AI agents.

Manages conversation context by clearing older tool call outputs when token limits
are reached, while preserving recent results.

Example:
    from dao_ai.middleware import create_context_editing_middleware

    # Clear old tool outputs when context exceeds 100k tokens
    middleware = create_context_editing_middleware(
        trigger=100000,
        keep=3,
    )
"""

from __future__ import annotations

from typing import Any, Literal

from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware
from langchain_core.tools import BaseTool
from loguru import logger

from dao_ai.config import BaseFunctionModel, ToolModel

__all__ = [
    "ContextEditingMiddleware",
    "ClearToolUsesEdit",
    "create_context_editing_middleware",
    "create_clear_tool_uses_edit",
]


def _resolve_tool_names(
    tools: list[str | ToolModel | dict[str, Any]] | None,
) -> list[str]:
    """Resolve tool specs to a list of tool name strings."""
    if tools is None:
        return []

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

    return result


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


def create_clear_tool_uses_edit(
    trigger: int = 100000,
    keep: int = 3,
    clear_at_least: int = 0,
    clear_tool_inputs: bool = False,
    exclude_tools: list[str | ToolModel | dict[str, Any]] | None = None,
    placeholder: str = "[cleared]",
) -> ClearToolUsesEdit:
    """
    Create a ClearToolUsesEdit for use with ContextEditingMiddleware.

    This edit strategy clears older tool results when the conversation exceeds
    a token threshold, while preserving recent results.

    Args:
        trigger: Token count that triggers the edit. When conversation exceeds
            this, older tool outputs are cleared. Default 100000.
        keep: Number of most recent tool results to preserve. These are never
            cleared. Default 3.
        clear_at_least: Minimum tokens to reclaim when edit runs.
            0 means clear as much as needed. Default 0.
        clear_tool_inputs: Whether to clear tool call arguments on AI messages.
            When True, tool call arguments are replaced with empty objects.
            Default False.
        exclude_tools: Tools to never clear. Can be:
            - list of str: Tool names
            - list of ToolModel: DAO AI tool models
            - list of dict: Tool config dicts
            Default None (no exclusions).
        placeholder: Text inserted for cleared tool outputs.
            Default "[cleared]".

    Returns:
        ClearToolUsesEdit instance

    Example:
        edit = create_clear_tool_uses_edit(
            trigger=50000,
            keep=5,
            clear_tool_inputs=True,
            exclude_tools=["important_tool"],
        )
    """
    excluded = _resolve_tool_names(exclude_tools) if exclude_tools else []

    logger.debug(
        "Creating ClearToolUsesEdit",
        trigger=trigger,
        keep=keep,
        clear_at_least=clear_at_least,
        clear_tool_inputs=clear_tool_inputs,
        exclude_tools=excluded or "none",
        placeholder=placeholder,
    )

    return ClearToolUsesEdit(
        trigger=trigger,
        keep=keep,
        clear_at_least=clear_at_least,
        clear_tool_inputs=clear_tool_inputs,
        exclude_tools=excluded,
        placeholder=placeholder,
    )


def create_context_editing_middleware(
    trigger: int = 100000,
    keep: int = 3,
    clear_at_least: int = 0,
    clear_tool_inputs: bool = False,
    exclude_tools: list[str | ToolModel | dict[str, Any]] | None = None,
    placeholder: str = "[cleared]",
    token_count_method: Literal["approximate", "model"] = "approximate",
) -> ContextEditingMiddleware:
    """
    Create a ContextEditingMiddleware with ClearToolUsesEdit.

    Manages conversation context by clearing older tool call outputs when token
    limits are reached. Useful for long conversations with many tool calls that
    exceed context window limits.

    Use cases:
    - Long conversations with many tool calls exceeding token limits
    - Reducing token costs by removing older irrelevant tool outputs
    - Maintaining only the most recent N tool results in context

    Args:
        trigger: Token count that triggers clearing. When conversation exceeds
            this threshold, older tool outputs are cleared. Default 100000.
        keep: Number of most recent tool results to always preserve.
            These are never cleared. Default 3.
        clear_at_least: Minimum tokens to reclaim when edit runs.
            0 means clear as much as needed. Default 0.
        clear_tool_inputs: Whether to also clear tool call arguments on AI
            messages. When True, replaces arguments with empty objects.
            Default False (preserves tool call context).
        exclude_tools: Tools to never clear outputs from. Can be:
            - list of str: Tool names
            - list of ToolModel: DAO AI tool models
            - list of dict: Tool config dicts
            Default None (no exclusions).
        placeholder: Text inserted for cleared tool outputs.
            Default "[cleared]".
        token_count_method: How to count tokens:
            - "approximate": Fast estimation (default)
            - "model": Accurate count using model tokenizer

    Returns:
        List containing ContextEditingMiddleware instance

    Example:
        # Basic usage - clear old tool outputs after 100k tokens
        middleware = create_context_editing_middleware(
            trigger=100000,
            keep=3,
        )

        # Aggressive clearing with exclusions
        middleware = create_context_editing_middleware(
            trigger=50000,
            keep=5,
            clear_tool_inputs=True,
            exclude_tools=["important_tool", "critical_search"],
            placeholder="[output cleared to save context]",
        )

        # Accurate token counting
        middleware = create_context_editing_middleware(
            trigger=100000,
            keep=3,
            token_count_method="model",
        )
    """
    edit = create_clear_tool_uses_edit(
        trigger=trigger,
        keep=keep,
        clear_at_least=clear_at_least,
        clear_tool_inputs=clear_tool_inputs,
        exclude_tools=exclude_tools,
        placeholder=placeholder,
    )

    logger.debug(
        "Creating ContextEditingMiddleware",
        token_count_method=token_count_method,
    )

    return ContextEditingMiddleware(
        edits=[edit],
        token_count_method=token_count_method,
    )
