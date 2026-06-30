"""Context-arg-bind middleware for DAO AI agents.

Auto-fills missing tool-call arguments from runtime context. The
canonical use is binding ``customer_id`` (or ``user_id``) parameters
to ``runtime.context.user_id`` so the LLM doesn't have to emit them.

Why this exists
---------------
LangGraph v3 streaming can dispatch a tool call before the LLM has
finished emitting all argument deltas. The tool then receives ``None``
for required parameters and the UC function call rejects with:

    Invalid parameters provided: {'customer_id':
      "Parameter customer_id should be of type STRING ... got NoneType"}

Non-streaming (``ainvoke``) doesn't hit this because the LLM emits the
tool call atomically. Streaming exposes the chunk-boundary race.

The fix shape: intercept tool calls in ``wrap_tool_call``, inspect the
``args`` dict, and for any well-known context-bound parameter name
(``customer_id``, ``user_id``) that's missing or ``None``, fill from
``runtime.context.user_id``. The original LLM-emitted value wins if
present and non-empty.

This is purely additive — it doesn't override LLM-provided values,
just patches missing context-bound params. Belt-and-suspenders for
the v3 race.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger

from dao_ai.state import AgentState, Context

__all__ = [
    "ContextArgBindMiddleware",
    "create_context_arg_bind_middleware",
]


# Parameter names known to map to runtime.context.user_id.
# Extend here when new context-bound params surface.
_USER_ID_ALIASES: frozenset[str] = frozenset({"customer_id", "user_id"})


class ContextArgBindMiddleware(AgentMiddleware[AgentState, Context]):
    """Auto-fills missing tool-call args from runtime context.

    Currently binds:
    - ``customer_id`` and ``user_id`` parameters -> ``runtime.context.user_id``

    Only fills when the LLM left the arg missing or ``None``. LLM-provided
    non-empty values are never overridden.
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], "ToolMessage | Command[Any]"],
    ) -> "ToolMessage | Command[Any]":
        request = self._bind_context_args(request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable["ToolMessage | Command[Any]"]],
    ) -> "ToolMessage | Command[Any]":
        request = self._bind_context_args(request)
        return await handler(request)

    def _bind_context_args(self, request: ToolCallRequest) -> ToolCallRequest:
        tool_call: dict[str, Any] = request.tool_call
        args: dict[str, Any] = dict(tool_call.get("args") or {})
        user_id: str | None = None
        runtime = request.runtime
        if runtime is not None and runtime.context is not None:
            user_id = runtime.context.user_id
        if user_id is None:
            return request
        patched: dict[str, str] = {}
        for alias in _USER_ID_ALIASES:
            if args.get(alias) in (None, ""):
                args[alias] = user_id
                patched[alias] = user_id
        if not patched:
            return request
        logger.info(
            "ContextArgBindMiddleware: filled tool-call args from context",
            tool_name=tool_call.get("name"),
            patched=patched,
        )
        new_tool_call: dict[str, Any] = {**tool_call, "args": args}
        return request.override(tool_call=new_tool_call)


def create_context_arg_bind_middleware() -> ContextArgBindMiddleware:
    """Factory matching the dao-ai middleware FQN pattern."""
    return ContextArgBindMiddleware()
