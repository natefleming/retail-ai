"""Raw ClientSession-based MCP client for sampling + roots (PR 3).

``langchain-mcp-adapters 0.3.0`` ``Callbacks`` surfaces
``on_logging_message`` / ``on_progress`` / ``on_elicitation`` — but not
``sampling_callback`` or ``list_roots_callback``. This module opens
``mcp.client.streamable_http.streamablehttp_client`` +
``mcp.client.session.ClientSession`` directly so those two callbacks can
be wired. Everything else (logging, progress, elicitation, structured
output, retries) is still handled by PR 1's ``MultiServerMCPClient`` path
— this module is only reached when ``McpFunctionModel.capabilities``
declares ``sampling`` or ``roots``.

Sampling flow: the MCP server issues ``sampling/createMessage``; dao-ai
translates the request into LangChain messages and routes it to the
configured ``InferenceEndpointModel`` (AI Gateway if declared). The
completion is returned as a ``CreateMessageResult``.

Roots flow: the server calls ``roots/list``; dao-ai returns the URIs
declared under ``capabilities.roots``.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Sequence

import mlflow
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import tool as create_tool
from loguru import logger
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.context import RequestContext
from mcp.types import (
    CallToolResult,
    CreateMessageRequestParams,
    CreateMessageResult,
    ErrorData,
    ListRootsResult,
    Root,
    SamplingMessage,
    TextContent,
    Tool,
)

from dao_ai.config import (
    McpCapabilitiesModel,
    McpFunctionModel,
    value_of,
)
from dao_ai.state import Context as DaoAiContext
from dao_ai.tools.mcp_callbacks import (
    DaoAiNotificationCallback,
    _emit,
    capture_runnable_config,
)


# ---------------------------------------------------------------------------
# Raw-MCP progress + logging adapters
#
# ``langchain-mcp-adapters`` supplies a ``CallbackContext`` (with server_name /
# tool_name) to its ``Callbacks`` protocol. The raw ``mcp.client.session``
# hooks have no such context object — ``LoggingFnT`` takes ``(params,)`` and
# ``ProgressFnT`` takes ``(progress, total, message)``. These thin wrappers
# build the same normalized envelope used by ``dao_ai.tools.mcp_callbacks``
# and dispatch through the shared ``_emit`` helper so span events + outer
# stream events look identical across the MultiServer path and the raw
# ClientSession path.
# ---------------------------------------------------------------------------


class _RawProgressAdapter:
    """MCP ``ProgressFnT`` adapter that mirrors ``DaoAiProgressCallback``.

    Captures the outer RunnableConfig at construction time so background
    MCP listener tasks can dispatch through LangChain's callback manager
    even when their own ContextVar has drifted.
    """

    def __init__(
        self,
        server_name: str,
        tool_name: str,
    ) -> None:
        self._server_name = server_name
        self._tool_name = tool_name
        self._config = capture_runnable_config()

    async def __call__(
        self,
        progress: float,
        total: float | None,
        message: str | None,
    ) -> None:
        envelope: dict[str, Any] = {
            "channel": "mcp.progress",
            "server_name": self._server_name,
            "tool_name": self._tool_name,
            "progress": progress,
            "total": total if total is not None else -1.0,
            "message": message or "",
        }
        await _emit("mcp.progress", envelope, self._config)


# ---------------------------------------------------------------------------
# Notification handling on the raw ClientSession path
#
# The raw path uses ``mcp.ClientSession(message_handler=...)`` directly, and
# ``DaoAiNotificationCallback`` (from ``mcp_callbacks``) is a MessageHandlerFnT
# out of the box — no local adapter needed.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sampling callback — converts MCP CreateMessageRequest → LangChain call
# ---------------------------------------------------------------------------


class DaoAiSamplingCallback:
    """Handles server-initiated ``sampling/createMessage`` by routing to the
    configured ``InferenceEndpointModel``.

    Instance is scoped to a single tool invocation so ``max_iterations`` can
    be enforced via an instance counter — the server may sample multiple
    times inside one ``tools/call``.
    """

    def __init__(self, function: McpFunctionModel) -> None:
        caps: McpCapabilitiesModel | None = function.capabilities
        if caps is None or caps.sampling is None:
            raise ValueError(
                "DaoAiSamplingCallback requires function.capabilities.sampling "
                "to be set — check the route selection in create_mcp_tools."
            )
        self._function = function
        self._sampling_cfg = caps.sampling
        self._iteration_count: int = 0

    async def __call__(
        self,
        context: RequestContext[ClientSession, Any],
        params: CreateMessageRequestParams,
    ) -> CreateMessageResult | ErrorData:
        self._iteration_count += 1
        if self._iteration_count > self._sampling_cfg.max_iterations:
            logger.warning(
                "mcp.sampling.max_iterations_exceeded",
                count=self._iteration_count,
                cap=self._sampling_cfg.max_iterations,
            )
            return ErrorData(
                code=-32000,
                message=(
                    f"Sampling iteration cap exceeded: "
                    f"{self._iteration_count} > {self._sampling_cfg.max_iterations}"
                ),
            )

        span = mlflow.get_current_active_span()
        if span is not None:
            try:
                span.add_event(
                    "mcp.sampling.request",
                    attributes={
                        "mcp.sampling.iteration": self._iteration_count,
                        "mcp.sampling.messages_count": len(params.messages),
                        "mcp.sampling.max_tokens": params.maxTokens,
                        "mcp.sampling.has_tools": bool(params.tools),
                    },
                )
            except Exception:
                logger.trace("mcp.sampling.span_event.skip")

        lc_messages = _mcp_messages_to_langchain(params)
        chat_model = self._sampling_cfg.endpoint.as_chat_model()

        # allow_tool_use=False: drop tools from the sampling call to prevent
        # recursion. The endpoint still runs, just without the tool surface.
        # dao-ai's sampling contract is a single completion, not a nested loop.
        if params.tools and not self._sampling_cfg.allow_tool_use:
            logger.debug(
                "mcp.sampling.tools_dropped",
                count=len(params.tools),
            )

        try:
            response = await chat_model.ainvoke(
                lc_messages,
                config={
                    "max_tokens": params.maxTokens,
                    "temperature": params.temperature,
                    "stop": params.stopSequences,
                },
            )
        except Exception as exc:
            logger.exception("mcp.sampling.invoke.failed", error=str(exc))
            return ErrorData(code=-32001, message=f"sampling invocation failed: {exc}")

        content_text = _extract_text(response)
        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=content_text),
            model=self._sampling_cfg.endpoint.name,
            stopReason="endTurn",
        )


def _mcp_messages_to_langchain(
    params: CreateMessageRequestParams,
) -> list[BaseMessage]:
    """Convert MCP sampling params (systemPrompt + messages) into a list of
    LangChain messages the chat model can consume."""
    out: list[BaseMessage] = []
    if params.systemPrompt:
        out.append(SystemMessage(content=params.systemPrompt))
    for msg in params.messages:
        text = _mcp_sampling_content_text(msg)
        if msg.role == "assistant":
            out.append(AIMessage(content=text))
        else:
            out.append(HumanMessage(content=text))
    return out


def _mcp_sampling_content_text(msg: SamplingMessage) -> str:
    """Best-effort text extraction from a SamplingMessage.

    SamplingMessage.content is a union — TextContent | ImageContent |
    AudioContent | ToolUseContent | ToolResultContent | list[...]. dao-ai's
    sampling implementation is text-only for PR 3; image/audio/tool content
    is stringified. The AI Gateway image path lives in a separate PR.
    """
    content = msg.content
    if isinstance(content, TextContent):
        return content.text
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, TextContent):
                parts.append(item.text)
            else:
                parts.append(f"[{getattr(item, 'type', 'unknown')} content]")
        return "\n".join(parts)
    return f"[{getattr(content, 'type', 'unknown')} content]"


def _extract_text(response: Any) -> str:
    """Extract a plain-text completion from a LangChain AIMessage-ish."""
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Anthropic-style content blocks
            texts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                elif isinstance(block, str):
                    texts.append(block)
            return "".join(texts)
    return str(response)


# ---------------------------------------------------------------------------
# List-roots callback — returns configured roots
# ---------------------------------------------------------------------------


class DaoAiListRootsCallback:
    """Returns ``capabilities.roots`` to the server on ``roots/list``.

    Named class (not a closure) so MLflow trace lineage shows the callback
    ran, per ``feedback_subclass_for_trace_observability``.
    """

    def __init__(self, function: McpFunctionModel) -> None:
        caps: McpCapabilitiesModel | None = function.capabilities
        if caps is None:
            raise ValueError(
                "DaoAiListRootsCallback requires function.capabilities to be set."
            )
        self._function = function
        self._roots = caps.roots

    async def __call__(
        self, context: RequestContext[ClientSession, Any]
    ) -> ListRootsResult | ErrorData:
        span = mlflow.get_current_active_span()
        if span is not None:
            try:
                span.add_event(
                    "mcp.roots.list",
                    attributes={"mcp.roots.count": len(self._roots)},
                )
            except Exception:
                logger.trace("mcp.roots.span_event.skip")
        roots: list[Root] = []
        for r in self._roots:
            try:
                roots.append(Root(uri=r.uri, name=r.name))  # type: ignore[arg-type]
            except Exception as exc:
                # A configured URI that doesn't match FileUrl validation must
                # not sink the whole call — log + skip so the server sees the
                # valid subset.
                logger.warning(
                    "mcp.roots.invalid_uri",
                    uri=r.uri,
                    error=str(exc),
                )
        return ListRootsResult(roots=roots)


# ---------------------------------------------------------------------------
# Raw-session tool-creation path
# ---------------------------------------------------------------------------


def sampling_or_roots_active(function: McpFunctionModel) -> bool:
    """Route helper — True when the function needs the raw ClientSession path."""
    caps = function.capabilities
    if caps is None:
        return False
    return caps.sampling is not None or bool(caps.roots)


async def acreate_mcp_tools_with_sampling(
    function: McpFunctionModel,
) -> Sequence[RunnableLike]:
    """Async tool-creation path for sampling + roots.

    Uses raw ``streamablehttp_client`` + ``ClientSession`` so
    ``sampling_callback`` / ``list_roots_callback`` can be attached. Each
    wrapped LangChain tool opens a fresh session per invocation to inherit
    the OBO context, matching the classic path's semantics.
    """
    from dao_ai.tools.mcp import (
        _extract_text_content,
        _get_auth_resource,
        _resolve_meta,
    )
    from dao_ai.tools.tracing import ResourceInfo, set_resource_attributes

    # Discover tools via a probe session — no callbacks needed for list_tools.
    async with _open_session(function, context=None) as session:
        listed = await session.list_tools()
        mcp_tools: list[Tool] = list(listed.tools)

    logger.info(
        "mcp.sampling.tools.discovered",
        count=len(mcp_tools),
        names=[t.name for t in mcp_tools],
    )

    def _wrap(mcp_tool: Tool) -> RunnableLike:
        @create_tool(
            mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            args_schema=mcp_tool.inputSchema,
        )
        async def _tool(
            runtime: ToolRuntime[DaoAiContext] = None,
            **kwargs: Any,
        ) -> str:
            auth_resource = _get_auth_resource(function)
            set_resource_attributes(
                ResourceInfo("mcp", auth_resource.on_behalf_of_user, mcp_tool.name)
            )
            context: DaoAiContext | None = runtime.context if runtime else None
            caps = function.capabilities
            call_kwargs: dict[str, Any] = {"meta": _resolve_meta(function.meta)}
            if caps and caps.progress:
                call_kwargs["progress_callback"] = _RawProgressAdapter(
                    server_name="mcp_function",
                    tool_name=mcp_tool.name,
                )
            async with _open_session(function, context=context) as session:
                result: CallToolResult = await session.call_tool(
                    mcp_tool.name,
                    kwargs,
                    **call_kwargs,
                )
                return _extract_text_content(result)

        return _tool

    return [_wrap(t) for t in mcp_tools]


@asynccontextmanager
async def _open_session(
    function: McpFunctionModel,
    *,
    context: DaoAiContext | None,
) -> AsyncIterator[ClientSession]:
    """Async context manager yielding a raw ``ClientSession`` with sampling
    + list_roots callbacks bound from the capabilities config.

    Uses ``AsyncExitStack`` to layer ``streamablehttp_client`` +
    ``ClientSession`` so both context managers unwind inside the same event
    loop even when the outer caller is a sync-over-async ``asyncio.run``
    site — avoids the ``anyio.NoEventLoopError`` that surfaced when the
    layered generators were GC'd after the loop had already closed.
    """
    from dao_ai.tools.mcp import _build_connection_config, _get_auth_resource

    conn = _build_connection_config(function, context)
    raw_url = conn.get("url") or function.mcp_url
    url = str(raw_url) if raw_url is not None else ""
    headers: dict[str, str] = dict(conn.get("headers") or {})
    auth = conn.get("auth")

    # Databricks Apps ingress (``*.databricksapps.com``) returns a bearer
    # challenge whose ``WWW-Authenticate`` shape ``DatabricksOAuthClientProvider``
    # doesn't reliably parse — the follow-up token exchange emits a URL with
    # a missing scheme and httpcore raises ``UnsupportedProtocol``.
    # Workaround: for App URLs, resolve a bearer token from the
    # **resource-scoped** ``WorkspaceClient`` (i.e. the SP declared on the
    # McpFunctionModel via ``client_id`` / ``client_secret`` / ``workspace_host``,
    # or on a nested ``app`` / ``connection`` resource) and inject it as an
    # ``Authorization`` header, bypassing the OAuth challenge path entirely.
    #
    # Using the resource-scoped SP (not ambient) matters on Model Serving:
    # the MS-created ambient SP is a "System Service Principal" that
    # cannot be granted CAN_USE on a Databricks App via the standard
    # permissions API (Databricks IAM constraint, verified with the
    # Model Serving team — SSPs are not exposed as regular grantable
    # principals). The resource-scoped SP (config's client_id/client_secret,
    # a regular user-managed SP) can be granted CAN_USE like any other
    # principal, so this path works on both Apps and MS.
    #
    # The managed MCP endpoints on the workspace host (``/api/2.0/mcp/...``)
    # keep the OAuth provider since those authenticate via the same
    # resource-scoped SP through ``_build_connection_config``.
    if ".databricksapps.com" in url:
        try:
            auth_resource = _get_auth_resource(function)
            auth_ws = auth_resource.workspace_client_from(context)
            hdrs = auth_ws.config.authenticate()
            identity = (
                value_of(auth_resource.client_id) if auth_resource.client_id else None
            ) or auth_ws.config.client_id or "resource-scoped"
            if hdrs and "Authorization" in hdrs:
                headers["Authorization"] = hdrs["Authorization"]
                auth = None
                logger.info(
                    "mcp.sampling.app_bearer_auth",
                    url=url,
                    identity=identity,
                    resource_type=auth_resource.__class__.__name__,
                    has_auth_header=True,
                )
            else:
                logger.warning(
                    "mcp.sampling.app_bearer_auth.no_header",
                    url=url,
                    resource_type=auth_resource.__class__.__name__,
                )
        except Exception as exc:
            logger.warning(
                "mcp.sampling.app_bearer_auth.failed",
                url=url,
                error=str(exc),
            )

    logger.info(
        "mcp.sampling.open_session",
        url=url,
        url_type=type(raw_url).__name__,
        has_auth=auth is not None,
        has_headers=bool(headers),
    )

    caps = function.capabilities
    sampling_cb: DaoAiSamplingCallback | None = None
    roots_cb: DaoAiListRootsCallback | None = None
    message_handler: DaoAiNotificationCallback | None = None
    if caps and caps.sampling:
        sampling_cb = DaoAiSamplingCallback(function)
    if caps and caps.roots:
        roots_cb = DaoAiListRootsCallback(function)
    if caps and caps.logging:
        message_handler = DaoAiNotificationCallback(server_name="mcp_function")

    async with AsyncExitStack() as stack:
        read_stream, write_stream, _ = await stack.enter_async_context(
            streamablehttp_client(url=url, headers=headers or None, auth=auth)
        )
        session = await stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
                sampling_callback=sampling_cb,
                list_roots_callback=roots_cb,
                message_handler=message_handler,
            )
        )
        await session.initialize()
        yield session
