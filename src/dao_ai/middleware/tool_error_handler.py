"""
Tool error handler middleware for DAO AI agents.

Catches tool execution errors and returns them as ToolMessage content
so the LLM can reason about failures and respond helpfully, rather than
letting exceptions propagate and crash the streaming response.

This follows the LangChain best practice of using ``@wrap_tool_call``
middleware with ``create_agent`` for tool-level error handling.

Example YAML config::

    middleware:
      - name: dao_ai.middleware.tool_error_handler.create_tool_error_handler_middleware
        args:
          include_traceback: false
"""

from __future__ import annotations

import traceback
from typing import Any

from langchain.agents.middleware import wrap_tool_call
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from loguru import logger

__all__ = [
    "create_tool_error_handler_middleware",
]


def _create_handler(include_traceback: bool = False) -> Any:
    """Build a ``@wrap_tool_call`` handler with the given options."""

    @wrap_tool_call
    def tool_error_handler(
        request: ToolCallRequest,
        handler: Any,
    ) -> ToolMessage:
        tool_name: str = request.tool_call.get("name", "unknown")
        tool_call_id: str = request.tool_call.get("id", "")
        try:
            return handler(request)
        except Exception as e:
            error_type: str = type(e).__name__
            error_msg: str = str(e)

            logger.error(
                "Tool execution failed",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                error_type=error_type,
                error=error_msg,
                exc_info=True,
            )

            content: str = f"Tool '{tool_name}' failed: {error_type}: {error_msg}"
            if include_traceback:
                tb: str = traceback.format_exc()
                content = f"{content}\n\nTraceback:\n{tb}"

            return ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                status="error",
            )

    return tool_error_handler


def create_tool_error_handler_middleware(
    include_traceback: bool = False,
) -> Any:
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
        A ``@wrap_tool_call`` middleware instance.

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
    return _create_handler(include_traceback=include_traceback)
