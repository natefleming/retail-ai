"""MCP client-side callbacks that bridge server-initiated notifications
into dao-ai's observability + HITL surfaces.

Wired into ``langchain_mcp_adapters.callbacks.Callbacks`` when an
``McpFunctionModel`` declares an ``McpCapabilitiesModel``. When capabilities
is absent the classic path skips this module entirely.

The three callables here match the ``LoggingMessageCallback``,
``ProgressCallback``, and ``ElicitationCallback`` Protocols defined by
``langchain-mcp-adapters`` 0.3.0. Each is a class (not a closure) so its
name surfaces in MLflow trace spans.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import mlflow
from langchain_mcp_adapters.callbacks import CallbackContext
from langgraph.types import interrupt as langgraph_interrupt
from loguru import logger
from mcp.shared.context import RequestContext as MCPRequestContext
from mcp.types import (
    ElicitRequestParams,
    ElicitResult,
    LoggingMessageNotificationParams,
)

_LOG_LEVEL_ORDER: dict[str, int] = {
    "debug": 10,
    "info": 20,
    "notice": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
    "alert": 50,
    "emergency": 50,
}


def _add_span_event(name: str, attributes: dict[str, Any]) -> None:
    """Add an event to the currently active MLflow span. Silent when no span."""
    span = mlflow.get_current_active_span()
    if span is None:
        return
    try:
        span.add_event(name, attributes=attributes)
    except Exception as exc:
        logger.debug(f"mcp callback: failed to add span event {name!r}: {exc}")


class DaoAiProgressCallback:
    """Forwards MCP ``notifications/progress`` to the active MLflow span
    as ``mcp.progress`` events."""

    async def __call__(
        self,
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ) -> None:
        _add_span_event(
            "mcp.progress",
            {
                "mcp.server_name": context.server_name,
                "mcp.tool_name": context.tool_name or "",
                "mcp.progress": progress,
                "mcp.total": total if total is not None else -1.0,
                "mcp.message": message or "",
            },
        )


class DaoAiLoggingCallback:
    """Forwards MCP ``notifications/message`` to the active MLflow span
    as ``mcp.log.<level>`` events. Records below ``min_level`` are dropped."""

    def __init__(self, min_level: Literal["debug", "info", "warning", "error"]) -> None:
        self.min_level = min_level
        self._min_severity = _LOG_LEVEL_ORDER[min_level]

    async def __call__(
        self,
        params: LoggingMessageNotificationParams,
        context: CallbackContext,
    ) -> None:
        level = str(params.level).lower()
        severity = _LOG_LEVEL_ORDER.get(level, 20)
        if severity < self._min_severity:
            return
        _add_span_event(
            f"mcp.log.{level}",
            {
                "mcp.server_name": context.server_name,
                "mcp.tool_name": context.tool_name or "",
                "mcp.logger": params.logger or "",
                "mcp.data": str(params.data)[:2000],
            },
        )


class DaoAiElicitationCallback:
    """Handles server-initiated ``elicitation/create`` requests.

    ``mode='reject'``: return ``action='cancel'`` without prompting.

    ``mode='hitl'``: raise a LangGraph interrupt whose resume value is
    interpreted as user-provided form content. The graph must be
    running under a checkpointer; without one the interrupt bubbles up.
    """

    def __init__(self, mode: Literal["hitl", "reject"]) -> None:
        self.mode = mode

    async def __call__(
        self,
        mcp_context: MCPRequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
    ) -> ElicitResult:
        if self.mode == "reject":
            _add_span_event(
                "mcp.elicitation.reject",
                {
                    "mcp.server_name": context.server_name,
                    "mcp.tool_name": context.tool_name or "",
                    "mcp.message": params.message[:500],
                },
            )
            return ElicitResult(action="cancel")

        payload: dict[str, Any] = {
            "type": "mcp.elicitation",
            "server_name": context.server_name,
            "tool_name": context.tool_name,
            "message": params.message,
            "requestedSchema": params.requestedSchema,
        }
        _add_span_event(
            "mcp.elicitation.request",
            {
                "mcp.server_name": context.server_name,
                "mcp.tool_name": context.tool_name or "",
                "mcp.message": params.message[:500],
            },
        )
        resume_value: Any = langgraph_interrupt(payload)
        return _resume_value_to_elicit_result(resume_value)


def _resume_value_to_elicit_result(resume_value: Any) -> ElicitResult:
    """Interpret a LangGraph interrupt resume value as an ElicitResult.

    Recognized shapes:
      - ``{"action": "accept" | "decline" | "cancel", "content": {...}}``
      - a bare dict → treated as ``action='accept'`` with ``content=<dict>``
      - ``None`` → ``action='cancel'``
    """
    if resume_value is None:
        return ElicitResult(action="cancel")
    if isinstance(resume_value, dict) and "action" in resume_value:
        action = resume_value["action"]
        content: Optional[dict[str, Any]] = resume_value.get("content")
        return ElicitResult(action=action, content=content)
    if isinstance(resume_value, dict):
        return ElicitResult(action="accept", content=resume_value)
    return ElicitResult(action="cancel")
