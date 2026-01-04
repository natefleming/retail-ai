"""
Tool selector middleware for intelligently filtering tools before LLM calls.

This middleware uses an LLM to select relevant tools from a large set, improving
performance and accuracy by reducing context size and improving focus.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import LLMToolSelectorMiddleware
from langchain_core.language_models import LanguageModelLike
from loguru import logger

from dao_ai.config import ToolModel


def create_llm_tool_selector_middleware(
    model: LanguageModelLike,
    max_tools: int = 3,
    always_include: list[str | ToolModel | dict[str, Any]] | None = None,
) -> LLMToolSelectorMiddleware:
    """
    Create an LLMToolSelectorMiddleware for intelligent tool selection.

    Uses an LLM to analyze the current query and select the most relevant tools
    before calling the main model. This is particularly useful for agents with
    many tools (10+) where most aren't relevant for any given query.

    Benefits:
    - Reduces token usage by filtering irrelevant tools
    - Improves model focus and accuracy
    - Optimizes cost for agents with large tool sets
    - Maintains context window efficiency

    Args:
        model: The LLM to use for tool selection. Typically a smaller, faster
            model like "gpt-4o-mini" or similar.
        max_tools: Maximum number of tools to select for each query.
            Default 3. Adjust based on your use case - higher values
            increase context but improve tool coverage.
        always_include: List of tools that should always be included regardless
            of the LLM's selection. Can be:
            - str: Tool name
            - ToolModel: Full tool configuration
            - dict: Tool configuration dictionary
            Use this for critical tools that should always be available.

    Returns:
        LLMToolSelectorMiddleware configured with the specified parameters

    Example:
        from dao_ai.middleware import create_llm_tool_selector_middleware
        from dao_ai.llms import create_llm

        # Use a fast, cheap model for tool selection
        selector_llm = create_llm("databricks-gpt-4o-mini")

        middleware = create_llm_tool_selector_middleware(
            model=selector_llm,
            max_tools=3,
            always_include=["search_web"],  # Always include search
        )

    Use Cases:
        - Large tool sets (10+ tools) where most are specialized
        - Cost optimization by reducing tokens in main model calls
        - Improved accuracy by reducing tool confusion
        - Dynamic tool filtering based on query relevance

    Note:
        The selector model makes an additional LLM call for each agent turn.
        Choose a fast, inexpensive model to minimize latency and cost overhead.
    """
    # Extract tool names from always_include
    always_include_names: list[str] = []
    if always_include:
        always_include_names = _resolve_tool_names(always_include)

    logger.debug(
        "Creating LLM tool selector middleware",
        max_tools=max_tools,
        always_include_count=len(always_include_names),
        always_include=always_include_names,
    )

    return LLMToolSelectorMiddleware(
        model=model,
        max_tools=max_tools,
        always_include=always_include_names if always_include_names else None,
    )


def _resolve_tool_names(tools: list[str | ToolModel | dict[str, Any]]) -> list[str]:
    """
    Extract tool names from a list of tool specifications.

    Args:
        tools: List of tool specifications (strings, ToolModels, or dicts)

    Returns:
        List of tool names as strings
    """
    names: list[str] = []

    for tool_spec in tools:
        if isinstance(tool_spec, str):
            # Simple string tool name
            names.append(tool_spec)
        elif isinstance(tool_spec, ToolModel):
            # ToolModel - use its name
            names.append(tool_spec.name)
        elif isinstance(tool_spec, dict):
            # Dictionary - try to extract name
            if "name" in tool_spec:
                names.append(tool_spec["name"])
            else:
                logger.warning(
                    "Tool dict missing 'name' field, skipping",
                    tool_spec=tool_spec,
                )
        else:
            logger.warning(
                "Unknown tool specification type, skipping",
                tool_spec_type=type(tool_spec).__name__,
            )

    return names
