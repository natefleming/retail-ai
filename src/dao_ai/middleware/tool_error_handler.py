"""
Tool error handler middleware for DAO AI agents.

Catches tool execution errors and returns them as ToolMessage content
so the LLM can reason about failures and respond helpfully, rather than
letting exceptions propagate and crash the streaming response.

Uses an ``AgentMiddleware`` subclass with explicit sync and async
``wrap_tool_call`` / ``awrap_tool_call`` implementations so the
middleware works in both Model Serving (sync) and Databricks Apps
(async) execution contexts.

Example YAML config::

    middleware:
      - name: dao_ai.middleware.tool_error_handler.create_tool_error_handler_middleware
        args:
          include_traceback: false
"""

from __future__ import annotations

import traceback
from typing import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from loguru import logger

__all__ = [
    "ToolErrorHandlerMiddleware",
    "create_tool_error_handler_middleware",
]


class ToolErrorHandlerMiddleware(AgentMiddleware):
    """Catch tool execution errors and return them as ``ToolMessage`` content.

    Implements both sync and async tool-call wrappers so the middleware
    works in every execution context (``stream``/``invoke`` and
    ``astream``/``ainvoke``).

    Args:
        include_traceback: If ``True``, append the full Python traceback
            to the error message returned to the LLM.
    """

    def __init__(self, include_traceback: bool = False) -> None:
        self._include_traceback = include_traceback

    def _build_error_message(
        self, tool_name: str, tool_call_id: str, exc: Exception
    ) -> ToolMessage:
        error_type: str = type(exc).__name__
        error_msg: str = str(exc)

        logger.error(
            "Tool execution failed",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            error_type=error_type,
            error=error_msg,
            exc_info=True,
        )

        content: str = f"Tool '{tool_name}' failed: {error_type}: {error_msg}"
        if self._include_traceback:
            tb: str = traceback.format_exc()
            content = f"{content}\n\nTraceback:\n{tb}"

        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            status="error",
        )

    # -- sync --------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        tool_name: str = request.tool_call.get("name", "unknown")
        tool_call_id: str = request.tool_call.get("id", "")
        try:
            return handler(request)
        except Exception as e:
            return self._build_error_message(tool_name, tool_call_id, e)

    # -- async -------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        tool_name: str = request.tool_call.get("name", "unknown")
        tool_call_id: str = request.tool_call.get("id", "")
        try:
            return await handler(request)
        except Exception as e:
            return self._build_error_message(tool_name, tool_call_id, e)


def create_tool_error_handler_middleware(
    include_traceback: bool = False,
) -> ToolErrorHandlerMiddleware:
    """Create tool error handler middleware.

    Wraps all tool calls so that exceptions are caught and returned as
    ``ToolMessage`` content with ``status="error"``.  The LLM then sees
    the error and can generate a helpful user-facing response instead of
    the request silently failing.

    Args:
        include_traceback: If ``True``, append the full Python traceback
            to the error message returned to the LLM.  Useful during
            development but should be ``False`` in production to avoid
            leaking internals.  Default ``False``.

    Returns:
        A ``ToolErrorHandlerMiddleware`` instance.

    Example YAML config::

        middleware:
          - name: dao_ai.middleware.tool_error_handler.create_tool_error_handler_middleware
            args:
              include_traceback: false
    """
    logger.debug(
        "Creating tool error handler middleware",
        include_traceback=include_traceback,
    )
    return ToolErrorHandlerMiddleware(include_traceback=include_traceback)
