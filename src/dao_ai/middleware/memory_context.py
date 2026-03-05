"""
Memory context injection middleware for DAO AI agents.

Automatically searches long-term memory before each model call and
injects relevant memories into the system prompt. This ensures the
agent has access to personalized context even without explicitly
calling ``search_memory``.

The middleware:

1. Extracts the latest user message as a search query.
2. Searches the configured ``MemoryStoreManager`` for relevant memories.
3. Prepends a ``## Memories`` section to the system message.

YAML Config::

    middleware:
      - name: dao_ai.middleware.memory_context.create_memory_context_middleware
        args:
          limit: 5

Example::

    from dao_ai.middleware.memory_context import MemoryContextMiddleware

    middleware = MemoryContextMiddleware(manager=manager, limit=5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.runtime import Runtime
from loguru import logger

if TYPE_CHECKING:
    from langmem.knowledge.extraction import MemoryStoreManager

    from dao_ai.state import AgentState, Context


MEMORY_SECTION_HEADER = "## Memories"


def _format_memories(memories: list[Any]) -> str:
    """Format memory items into a prompt section.

    Each memory item has a ``.value`` dict with a ``content`` key. The
    content can be a string or a dict (for structured schemas).
    """
    if not memories:
        return ""

    lines: list[str] = [MEMORY_SECTION_HEADER]
    for item in memories:
        value = item.value if hasattr(item, "value") else item
        if isinstance(value, dict):
            content = value.get("content", value)
            if isinstance(content, dict):
                parts = [f"- **{k}**: {v}" for k, v in content.items() if v]
                lines.append("\n".join(parts))
            else:
                lines.append(f"- {content}")
        else:
            lines.append(f"- {value}")
    return "\n".join(lines)


def _extract_query(messages: list[BaseMessage]) -> str | None:
    """Extract the latest user message content as a search query."""
    for msg in reversed(messages):
        if msg.type == "human":
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


class MemoryContextMiddleware(AgentMiddleware):
    """Injects long-term memory context into the system prompt.

    Searches the memory store for memories relevant to the current user
    message and prepends them to the system prompt before each model
    call.

    Args:
        manager: A ``MemoryStoreManager`` configured with the
            appropriate namespace and store.
        limit: Maximum number of memories to inject. Defaults to 5.
    """

    def __init__(
        self,
        manager: MemoryStoreManager,
        limit: int = 5,
    ) -> None:
        self._manager: MemoryStoreManager = manager
        self._limit: int = limit

    async def abefore_model(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Search memory and inject into system prompt before model call."""
        messages: list[BaseMessage] = state.get("messages", [])
        if not messages:
            return None

        query: str | None = _extract_query(messages)
        if not query:
            return None

        config: dict[str, Any] = {}
        if runtime.context:
            config = {"configurable": runtime.context.model_dump()}

        try:
            memories = await self._manager.asearch(
                query=query,
                limit=self._limit,
                config=config,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Memory search failed, continuing without memory context"
            )
            return None

        if not memories:
            logger.trace("No relevant memories found", query=query[:80])
            return None

        memory_text: str = _format_memories(memories)

        logger.debug(
            "Injecting memory context",
            memories_count=len(memories),
            query=query[:80],
        )

        updated_messages: list[BaseMessage] = list(messages)
        if updated_messages and isinstance(updated_messages[0], SystemMessage):
            original_content: str = updated_messages[0].content
            updated_messages[0] = SystemMessage(
                content=f"{original_content}\n\n{memory_text}"
            )
        else:
            updated_messages.insert(0, SystemMessage(content=memory_text))

        return {"messages": updated_messages}

    def before_model(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Sync fallback -- memory injection requires async."""
        logger.trace("Skipping memory injection in sync mode")
        return None


def create_memory_context_middleware(
    manager: MemoryStoreManager,
    limit: int = 5,
) -> MemoryContextMiddleware:
    """Factory function for creating a ``MemoryContextMiddleware``.

    Args:
        manager: A ``MemoryStoreManager`` for searching memories.
        limit: Maximum number of memories to inject. Defaults to 5.

    Returns:
        A configured ``MemoryContextMiddleware``.
    """
    logger.debug("Creating MemoryContextMiddleware", limit=limit)
    return MemoryContextMiddleware(manager=manager, limit=limit)


__all__ = [
    "MemoryContextMiddleware",
    "create_memory_context_middleware",
]
