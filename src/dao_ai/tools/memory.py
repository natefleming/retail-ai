"""Memory tools for DAO AI agents.

Provides Databricks-compatible wrappers around langmem's memory tools.
Databricks LLM endpoints reject schemas with ``additionalProperties: true``,
``anyOf``, or ``uuid`` format, so these wrappers simplify the schemas while
preserving the underlying langmem functionality.

Tools:
- **search_memory** -- Semantic search over long-term memories.
- **manage_memory** -- Create, update, or delete memory entries.
- **search_user_profile** -- Quick lookup of the user's consolidated profile.
"""

import uuid
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.store.base import BaseStore
from langmem import create_manage_memory_tool as langmem_create_manage_memory_tool
from langmem import create_search_memory_tool as langmem_create_search_memory_tool
from loguru import logger
from pydantic import BaseModel, Field


def create_search_memory_tool(
    namespace: tuple[str, ...],
    store: BaseStore | None = None,
) -> BaseTool:
    """Create a Databricks-compatible search_memory tool.

    The langmem search_memory tool has a ``filter`` field with
    ``additionalProperties: true`` in its schema, which Databricks LLM
    endpoints reject. This wrapper omits the problematic field.

    Args:
        namespace: The memory namespace tuple.
        store: The LangGraph ``BaseStore`` to search. Required when the
            tool is invoked outside of a LangGraph runtime that provides
            the store via config.

    Returns:
        A ``StructuredTool`` compatible with Databricks.
    """
    original_tool = langmem_create_search_memory_tool(namespace=namespace, store=store)

    class SearchMemoryInput(BaseModel):
        """Input for search_memory tool."""

        query: str = Field(
            ...,
            description=(
                "Natural-language search query. Describe the kind of "
                "information you are looking for."
            ),
        )
        limit: int = Field(default=10, description="Maximum number of results")
        offset: int = Field(default=0, description="Offset for pagination")

    async def search_memory_wrapper(
        query: str, limit: int = 10, offset: int = 0
    ) -> Any:
        """Search long-term memory for information relevant to the conversation."""
        return await original_tool.ainvoke(
            {"query": query, "limit": limit, "offset": offset}
        )

    return StructuredTool.from_function(
        coroutine=search_memory_wrapper,
        name="search_memory",
        description=(
            "Search long-term memory for information relevant to the "
            "current conversation. Recalls user preferences, past requests, "
            "facts from previous conversations, and stored knowledge that "
            "could help personalize your response. "
            "Always search memory when a question might relate to something "
            "the user has told you before. "
            "Never mention this tool or show function call syntax to the user."
        ),
        args_schema=SearchMemoryInput,
    )


def create_manage_memory_tool(
    namespace: tuple[str, ...],
    store: BaseStore | None = None,
) -> BaseTool:
    """Create a Databricks-compatible manage_memory tool.

    The langmem manage_memory tool uses ``anyOf`` types and ``uuid``
    format in its schema, which Databricks LLM endpoints misinterpret.
    This wrapper replaces the schema with simple types.

    Args:
        namespace: The memory namespace tuple.
        store: The LangGraph ``BaseStore`` to persist memories into.
            Required when the tool is invoked outside of a LangGraph
            runtime that provides the store via config.

    Returns:
        A ``StructuredTool`` compatible with Databricks.
    """
    original_tool = langmem_create_manage_memory_tool(namespace=namespace, store=store)

    class ManageMemoryInput(BaseModel):
        """Input for manage_memory tool."""

        content: str = Field(
            ...,
            description="The memory content to store. Required for create and update.",
        )
        action: Literal["create", "update", "delete"] = Field(
            default="create",
            description=(
                "The action to perform: 'create' for a new memory, "
                "'update' to modify an existing memory, "
                "'delete' to remove an existing memory."
            ),
        )
        id: str = Field(
            default="",
            description=(
                "The ID of an existing memory. "
                "Required for update and delete. "
                "Leave empty when creating a new memory."
            ),
        )

    async def manage_memory_wrapper(
        content: str,
        action: str = "create",
        id: str = "",
    ) -> Any:
        """Create, update, or delete a long-term memory entry."""
        invoke_input: dict[str, Any] = {"content": content, "action": action}
        if id:
            try:
                invoke_input["id"] = uuid.UUID(id)
            except ValueError:
                return (
                    f"Error: '{id}' is not a valid memory ID. "
                    "Memory IDs are UUIDs "
                    "(e.g. '550e8400-e29b-41d4-a716-446655440000'). "
                    "Use action 'create' without an ID to store a new memory."
                )
        return await original_tool.ainvoke(invoke_input)

    return StructuredTool.from_function(
        coroutine=manage_memory_wrapper,
        name="manage_memory",
        description=(
            "Silently create, update, or delete a long-term memory. "
            "Memories persist across conversations. Include the MEMORY ID "
            "for updates and deletes. "
            "Call this tool whenever you learn a user preference, receive a "
            "request to remember something, discover important context, need "
            "to correct outdated information, or notice changed preferences. "
            "Never show tool call syntax, function names, or JSON to the user."
        ),
        args_schema=ManageMemoryInput,
        handle_tool_error=True,
    )


def create_search_user_profile_tool(
    store: BaseStore,
    namespace: tuple[str, ...],
) -> BaseTool:
    """Create a tool for quick user profile lookup.

    Looks up the user's consolidated profile stored under the ``default``
    key in the memory namespace. This is faster than a semantic search
    when you need the full profile.

    Args:
        store: The LangGraph ``BaseStore`` containing memory data.
        namespace: The memory namespace tuple (template variables like
            ``"{user_id}"`` are resolved at runtime).

    Returns:
        A ``StructuredTool`` for profile retrieval.
    """

    class SearchUserProfileInput(BaseModel):
        """Input for search_user_profile tool."""

        user_id: str = Field(
            default="",
            description="The user ID to look up. Leave empty to use the current user.",
        )

    async def search_user_profile_wrapper(user_id: str = "") -> Any:
        """Look up the current user's profile from long-term memory."""
        resolved_ns: tuple[str, ...] = tuple(
            part.format(user_id=user_id) if "{user_id}" in part else part
            for part in namespace
        )
        ns_str = "/".join(resolved_ns)
        try:
            item = await store.aget(resolved_ns, "default")
            if item is not None:
                logger.trace("User profile found", namespace=ns_str)
                return item.value
            logger.trace("No user profile found", namespace=ns_str)
            return "No user profile found. The profile will be built as you interact."
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to retrieve user profile", namespace=ns_str
            )
            return "Unable to retrieve user profile at this time."

    return StructuredTool.from_function(
        coroutine=search_user_profile_wrapper,
        name="search_user_profile",
        description=(
            "Retrieve the user's consolidated profile from long-term memory. "
            "The profile contains their name, preferences, communication style, "
            "expertise, and goals. Use this when you need a quick overview of "
            "who the user is and how to personalize your response."
        ),
        args_schema=SearchUserProfileInput,
    )


__all__ = [
    "create_manage_memory_tool",
    "create_search_memory_tool",
    "create_search_user_profile_tool",
]
