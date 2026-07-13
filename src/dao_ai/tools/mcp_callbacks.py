"""MCP client-side callbacks that bridge server-initiated notifications
into dao-ai's observability + client-stream surfaces.

Wired into ``langchain_mcp_adapters.callbacks.Callbacks`` when an
``McpFunctionModel`` declares an ``McpCapabilitiesModel``. When capabilities
is absent the classic path skips this module entirely.

The three callables here match the ``LoggingMessageCallback``,
``ProgressCallback``, and ``ElicitationCallback`` Protocols defined by
``langchain-mcp-adapters`` 0.3.0. Each is a class (not a closure) so its
name surfaces in MLflow trace spans.

Notification transport
----------------------
Each MCP notification is dual-emitted:

1. **MLflow span** — via :func:`_add_span_event`, for post-hoc tracing.
2. **LangChain callback manager** — via :func:`_dispatch_custom_event`
   which wraps ``langchain_core.callbacks.adispatch_custom_event``.

The callback-manager path is the LangChain-native mechanism to surface
events from inside a tool up through ``create_agent``'s ToolNode to a
callback handler attached on the outer ``RunnableConfig``. Any
``AsyncCallbackHandler`` registered at ``graph.astream(config={"callbacks":
[handler]})`` time receives ``on_custom_event(name, data, ...)`` calls for
each envelope.

We use this in place of ``langgraph.config.get_stream_writer()`` because
the LangGraph streaming system (``stream_mode="custom"``) has open issue
#6447 — writes made from inside a create_agent subgraph don't bubble up to
the outer ``astream``'s custom channel. The callback-manager path is
proven end-to-end by langchain-core's own test
``test_custom_event_root_dispatch_with_in_tool``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import mlflow
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.runnables.config import RunnableConfig, ensure_config
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
    """Add an event to the currently active MLflow span. Silent when no span.

    MLflow's ``LiveSpan.add_event`` takes a single ``SpanEvent`` object
    (``mlflow/entities/span.py`` — ``def add_event(self, event: SpanEvent)``).
    Constructing the SpanEvent here keeps the caller-side ergonomics
    ``(name, attributes)`` intact.
    """
    span = mlflow.get_current_active_span()
    if span is None:
        return
    try:
        from mlflow.entities import SpanEvent

        span.add_event(SpanEvent(name=name, attributes=attributes))
    except Exception as exc:
        logger.debug(f"mcp callback: failed to add span event {name!r}: {exc}")


def capture_runnable_config() -> RunnableConfig | None:
    """Capture the current LangChain ``RunnableConfig`` for later use.

    Called at callback-construction time (in the tool wrapper's task, which
    inherited the outer graph's ``var_child_runnable_config`` ContextVar).
    The returned config is passed explicitly to ``adispatch_custom_event``
    when the MCP client's background listener task fires our callback —
    that background task may not have the ContextVar, so we can't rely on
    ``ensure_config(None)`` inside the callback.

    Returns None outside a runnable context (e.g. batch predict without a
    graph.astream, or import-time construction) — callers should treat that
    as "no client forwarding; MLflow span events only".
    """
    try:
        cfg = ensure_config(None)
    except Exception:
        return None
    # ensure_config always returns a valid dict; treat empty ``callbacks`` as
    # "no handlers registered" — dispatching would still be safe but a no-op.
    return cfg


async def _dispatch_custom_event(
    channel: str,
    envelope: dict[str, Any],
    config: RunnableConfig | None,
) -> None:
    """Dispatch an MCP envelope via LangChain's callback manager.

    Any ``AsyncCallbackHandler`` registered on the outer runnable's config
    receives ``on_custom_event(channel, envelope, ...)``.
    """
    if config is None:
        return
    try:
        await adispatch_custom_event(channel, envelope, config=config)
    except Exception as exc:
        logger.debug(
            f"mcp callback: failed to dispatch {channel!r} custom event: {exc}"
        )


async def _emit(
    span_event_name: str,
    envelope: dict[str, Any],
    config: RunnableConfig | None,
) -> None:
    """Dual-emit an MCP notification envelope to MLflow span + callback manager.

    The span event uses the canonical ``mcp.progress`` / ``mcp.log.<level>``
    name; the callback-manager event uses the ``channel`` field of the
    envelope (``mcp.progress`` / ``mcp.log``). Both receive the same
    envelope so downstream consumers see identical shape.
    """
    _add_span_event(span_event_name, envelope)
    await _dispatch_custom_event(envelope["channel"], envelope, config)


class DaoAiProgressCallback:
    """Forwards MCP ``notifications/progress`` to the active MLflow span
    and — when a callback handler is registered on the outer runnable —
    to that handler as an ``on_custom_event("mcp.progress", envelope)``.

    The RunnableConfig is captured at construction time (in the tool
    wrapper's task, which has the outer ContextVar). MCP client notifications
    fire in a background listener task where that ContextVar may not be
    inherited, so we pass the captured config explicitly to
    ``adispatch_custom_event``.
    """

    def __init__(self) -> None:
        self._config = capture_runnable_config()

    async def __call__(
        self,
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ) -> None:
        envelope: dict[str, Any] = {
            "channel": "mcp.progress",
            "server_name": context.server_name,
            "tool_name": context.tool_name or "",
            "progress": progress,
            "total": total if total is not None else -1.0,
            "message": message or "",
        }
        await _emit("mcp.progress", envelope, self._config)


class DaoAiLoggingCallback:
    """Forwards MCP ``notifications/message`` to the active MLflow span
    and — when a callback handler is registered on the outer runnable —
    to that handler as an ``on_custom_event("mcp.log", envelope)``. Records
    below ``min_level`` are dropped.

    Config capture semantics match ``DaoAiProgressCallback``.
    """

    def __init__(
        self,
        min_level: Literal["debug", "info", "warning", "error"],
    ) -> None:
        self.min_level = min_level
        self._min_severity = _LOG_LEVEL_ORDER[min_level]
        self._config = capture_runnable_config()

    async def __call__(
        self,
        params: LoggingMessageNotificationParams,
        context: CallbackContext,
    ) -> None:
        level = str(params.level).lower()
        severity = _LOG_LEVEL_ORDER.get(level, 20)
        if severity < self._min_severity:
            return
        envelope: dict[str, Any] = {
            "channel": "mcp.log",
            "server_name": context.server_name,
            "tool_name": context.tool_name or "",
            "level": level,
            "logger": params.logger or "",
            "data": str(params.data)[:2000],
        }
        await _emit(f"mcp.log.{level}", envelope, self._config)


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
