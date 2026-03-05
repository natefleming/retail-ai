"""
Tool call ID sanitizer middleware for DAO AI agents.

Some LLM endpoints (particularly smaller models) occasionally return tool
calls without an ``id`` field.  LangGraph's ``_validate_tool_call`` then
attempts to construct a ``ToolMessage`` with ``tool_call_id=None``, which
Pydantic rejects.

This middleware runs ``after_model`` and patches any ``None``/empty IDs with
generated UUIDs so the tool node always receives well-formed tool calls.

This middleware is applied automatically -- no YAML configuration needed.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.runtime import Runtime
from loguru import logger

from dao_ai.state import AgentState, Context

__all__ = [
    "ToolCallIdSanitizerMiddleware",
    "create_tool_call_id_sanitizer_middleware",
]


class ToolCallIdSanitizerMiddleware(AgentMiddleware[AgentState, Context]):
    """Patches ``None`` tool-call IDs with generated UUIDs.

    Stateless and zero-cost when all IDs are already present -- it only
    scans the last message in the state.
    """

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime[Context],
    ) -> dict[str, Any] | None:
        messages: list[BaseMessage] = state.get("messages", [])
        if not messages:
            return None

        last_msg: BaseMessage = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return None

        patched = False
        for tc in last_msg.tool_calls:
            if not tc.get("id"):
                tc["id"] = uuid4().hex[:24]
                patched = True

        if patched:
            logger.trace(
                "Patched missing tool_call_id(s)",
                tool_names=[tc["name"] for tc in last_msg.tool_calls],
            )
            return {"messages": [last_msg]}

        return None


def create_tool_call_id_sanitizer_middleware() -> ToolCallIdSanitizerMiddleware:
    """Factory that returns a :class:`ToolCallIdSanitizerMiddleware` instance."""
    return ToolCallIdSanitizerMiddleware()
