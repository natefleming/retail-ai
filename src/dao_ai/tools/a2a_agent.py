"""Google A2A (Agent-to-Agent) protocol tool.

Calls any agent speaking Google's open `A2A protocol <https://a2a-protocol.org>`_
using the official ``a2a-sdk`` Python client, drains the streaming response,
and returns the concatenated agent reply as a **single string**.

A2A is a framework-agnostic JSON-RPC + SSE standard for cross-vendor agent
interop (LangGraph, Google ADK, Crew.ai, custom) governed by the Linux
Foundation. Vertex AI Agent Engine also speaks A2A natively, so this tool
doubles as an alternate path to Vertex-hosted ADK agents (contrast with
``vertex_agent_engine`` which uses Vertex's proprietary ``:streamQuery``).

**Why streaming?** A2A's ``send_message`` returns an async iterator of
``ClientEvent | Message`` events (Task snapshots, artifact updates, status
updates, inline messages). We drain the iterator internally and aggregate
every text ``Part`` emitted by the agent so the tool presents a synchronous
surface to the LangChain caller — token-level streaming never leaks out.

DAO AI ``Context`` feeds A2A naturally:

- ``context.thread_id`` -> A2A ``Message.context_id`` (groups turns into a
  server-side conversation; the remote agent sees the same ``context_id``
  across calls for multi-turn continuity).
- ``context.user_id`` -> ``Message.metadata["dao_ai.user_id"]`` (A2A has no
  first-class user field; metadata is the spec-sanctioned extension slot).

**Auth modes.** Three, selected by the ``auth_type`` discriminator:

- ``bearer`` — resolved ``auth`` string becomes ``Authorization: Bearer ...``.
  Use for API keys and static OAuth tokens.
- ``gcp_service_account`` — resolved ``auth`` is passed to
  ``load_gcp_credentials`` (accepts file path, ``/Volumes/...``, or inline
  JSON). A fresh OAuth access token is minted per-invocation via
  ``mint_gcp_access_token`` so expired tokens refresh transparently.
- ``none`` — no ``Authorization`` header; suitable for public A2A agents.

Auth material (tokens, bearer values, raw service-account JSON) is never
logged or captured in MLflow spans — log payloads are explicitly projected
to redact sensitive fields, and the httpx client's default headers never
appear as span inputs.

**Session handling.** If a ``context_id`` is passed and the remote server
returns an empty stream or a ``failed`` Task with context-related error
text, the tool retries once **without** ``context_id`` so the server
creates a fresh context. Retry state is stamped on the tool's MLflow span
(``dao_ai.a2a.retried_without_context`` / ``dao_ai.a2a.retry_reason``).

**Sync-over-async.** The ``a2a-sdk`` is async-only. The tool body runs
``asyncio.run(_invoke_a2a_async(...))``. ``nest_asyncio.apply()`` is called
at module import so the tool also works from inside already-running event
loops (notebooks, tests).
"""

from __future__ import annotations

import asyncio
from textwrap import dedent
from typing import Annotated, Any, Callable, Literal, Optional

import httpx
import mlflow
import nest_asyncio
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.client.helpers import create_text_message_object
from a2a.types import (
    AgentCard,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    PREV_AGENT_CARD_WELL_KNOWN_PATH,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, StructuredTool
from loguru import logger
from mlflow.entities import SpanType

from dao_ai.config import AnyVariable, value_of
from dao_ai.state import Context
from dao_ai.tools._gcp_auth import (
    coerce_any_variable,
    load_gcp_credentials,
    mint_gcp_access_token,
)
from dao_ai.tools.tracing import (
    ATTR_A2A_AGENT_NAME,
    ATTR_A2A_AGENT_VERSION,
    ATTR_A2A_AUTH_MODE,
    ATTR_A2A_CARD_PATH_USED,
    ATTR_A2A_CONTEXT_ID,
    ATTR_A2A_CONTEXT_PASSED,
    ATTR_A2A_ENDPOINT_URL,
    ATTR_A2A_PROMPT_CHARS,
    ATTR_A2A_RESPONSE_CHARS,
    ATTR_A2A_RETRIED_WITHOUT_CONTEXT,
    ATTR_A2A_RETRY_REASON,
    ATTR_A2A_STREAM_AGENT_EVENTS,
    ATTR_A2A_STREAM_NON_TEXT_PARTS,
    ATTR_A2A_STREAM_TASK_EVENTS,
    ATTR_A2A_STREAM_TERMINAL_STATE,
    ATTR_A2A_STREAM_TOTAL_EVENTS,
    ATTR_A2A_STREAMING,
    ATTR_A2A_TASK_ID,
    ATTR_A2A_USER_ID,
    ATTR_HTTP_METHOD,
    ATTR_HTTP_URL,
    ResourceInfo,
    set_resource_attributes,
)

# Allow ``asyncio.run`` to nest under an already-running loop (notebooks,
# pytest-asyncio) — idempotent and cheap. Worker-thread tool invocations
# don't need this but it costs nothing and saves one class of surprise.
nest_asyncio.apply()

_AuthMode = Literal["bearer", "gcp_service_account", "none"]
_VALID_AUTH_MODES: tuple[str, ...] = ("bearer", "gcp_service_account", "none")

_DEFAULT_DESCRIPTION: str = dedent("""
    Send a message to a remote agent over the Google A2A (Agent-to-Agent)
    protocol and return the assistant's reply. Use this tool to delegate
    questions to the remote A2A agent. Pass the user's question or prompt
    as the 'prompt' argument.
""").strip()

_EMPTY_REPLY_SENTINEL: str = "(no agent response)"


def create_a2a_agent_tool(
    endpoint: AnyVariable,
    *,
    auth: Optional[AnyVariable] = None,
    auth_type: AnyVariable = "bearer",
    streaming: AnyVariable = True,
    timeout_seconds: AnyVariable = 300,
    card_path: AnyVariable = AGENT_CARD_WELL_KNOWN_PATH,
    card_fallback_path: Optional[AnyVariable] = PREV_AGENT_CARD_WELL_KNOWN_PATH,
    user_id: Optional[AnyVariable] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
    name: Optional[AnyVariable] = None,
    description: Optional[AnyVariable] = None,
) -> Callable[..., str]:
    """Create a tool that calls a remote A2A (Agent-to-Agent) agent.

    Args:
        endpoint: Base URL of the A2A agent, e.g. ``https://agent.example.com``.
            The agent card is resolved at ``<endpoint><card_path>`` with
            fallback to ``<endpoint><card_fallback_path>`` on 404.
        auth: Auth material resolved to a string. Interpretation depends on
            ``auth_type``. Any ``AnyVariable`` form supported (env, secret,
            composite, inline value).
        auth_type: One of ``"bearer"`` (default), ``"gcp_service_account"``,
            or ``"none"``. ``bearer`` uses the resolved ``auth`` verbatim as
            the bearer token. ``gcp_service_account`` passes ``auth`` through
            ``load_gcp_credentials`` and mints a fresh access token per call.
            ``none`` suppresses the ``Authorization`` header (public agents).
        streaming: If true (default) the A2A client negotiates streaming;
            responses are still aggregated internally and returned as one
            string. Set false to force request/response.
        timeout_seconds: httpx client timeout (default 300s).
        card_path: Primary agent-card discovery path relative to ``endpoint``.
            Defaults to the current spec's ``/.well-known/agent-card.json``.
        card_fallback_path: Fallback path if the primary 404s. Defaults to the
            pre-1.0 spec's ``/.well-known/agent.json``. Set to ``None`` to
            disable fallback.
        user_id: Static value forwarded as ``Message.metadata["dao_ai.user_id"]``.
            If omitted, falls back to ``runtime.context.user_id``.
        extra_metadata: Static metadata merged into ``Message.metadata`` on
            every call. Keys collide: static ``extra_metadata`` wins over the
            built-in ``dao_ai.user_id``.
        name: Tool name shown to the LLM. Defaults to ``a2a_agent``.
        description: Tool description shown to the LLM.

    Returns:
        A :class:`StructuredTool` that sends one message and returns the
        concatenated agent text as a single string.

    Notes:
        Session continuity: ``runtime.context.thread_id`` is forwarded as
        ``Message.context_id``. If the remote agent rejects or ignores the
        context id (empty stream or failed Task referencing context), the
        tool retries once without a ``context_id`` to force a fresh session.
    """
    resolved_endpoint: str = str(value_of(coerce_any_variable(endpoint))).rstrip("/")
    resolved_auth_type: str = str(value_of(coerce_any_variable(auth_type))).lower()
    if resolved_auth_type not in _VALID_AUTH_MODES:
        raise ValueError(
            f"create_a2a_agent_tool: auth_type must be one of "
            f"{_VALID_AUTH_MODES}, got {resolved_auth_type!r}"
        )
    resolved_streaming: bool = bool(value_of(coerce_any_variable(streaming)))
    resolved_timeout: int = int(value_of(coerce_any_variable(timeout_seconds)))
    resolved_card_path: str = _normalize_card_path(
        str(value_of(coerce_any_variable(card_path)))
    )
    resolved_card_fallback: Optional[str] = (
        _normalize_card_path(str(value_of(coerce_any_variable(card_fallback_path))))
        if card_fallback_path is not None
        else None
    )
    tool_name: str = (
        str(value_of(coerce_any_variable(name))) if name is not None else "a2a_agent"
    )
    doc_description: str = (
        str(value_of(coerce_any_variable(description)))
        if description is not None
        else _DEFAULT_DESCRIPTION
    )

    logger.debug(
        "Creating A2A agent tool",
        name=tool_name,
        endpoint=resolved_endpoint,
        auth_type=resolved_auth_type,
        streaming=resolved_streaming,
        card_path=resolved_card_path,
        card_fallback_path=resolved_card_fallback,
    )

    # Resolve auth once at factory time. Static bearer tokens are cached in
    # the closure; GCP credentials are held as a refreshable ``Credentials``
    # object and minted per-invocation so expired tokens refresh. None of
    # these values are ever logged or persisted to spans.
    header_provider: Callable[[], dict[str, str]]
    if resolved_auth_type == "none":
        header_provider = _no_auth_headers
    elif resolved_auth_type == "bearer":
        if auth is None:
            raise ValueError(
                "create_a2a_agent_tool: auth_type='bearer' requires auth to be set."
            )
        _resolved_token: str = str(value_of(coerce_any_variable(auth)))
        if not _resolved_token:
            raise ValueError("create_a2a_agent_tool: resolved bearer token is empty.")
        header_provider = lambda: {"Authorization": f"Bearer {_resolved_token}"}  # noqa: E731
    else:  # gcp_service_account
        if auth is None:
            raise ValueError(
                "create_a2a_agent_tool: auth_type='gcp_service_account' "
                "requires auth to be set."
            )
        _gcp_creds = load_gcp_credentials(auth)
        header_provider = lambda: {  # noqa: E731
            "Authorization": f"Bearer {mint_gcp_access_token(_gcp_creds)}"
        }

    logger.debug(
        "Resolved A2A auth mode",
        auth_type=resolved_auth_type,
        has_auth=auth is not None,
    )

    doc_signature: str = dedent("""
    Args:
        prompt (str): Message to send to the A2A agent.

    Returns:
        str: The concatenated text response from the agent.
    """)
    doc: str = doc_description + "\n" + doc_signature

    def _resolve_user_id(context: Context | None) -> Optional[str]:
        if user_id is not None:
            return str(value_of(coerce_any_variable(user_id)))
        if context and context.user_id:
            return context.user_id
        return None

    def a2a_agent(
        prompt: str,
        runtime: Annotated[ToolRuntime[Context], InjectedToolArg] = None,
    ) -> str:
        context: Context | None = runtime.context if runtime else None
        resolved_user_id: Optional[str] = _resolve_user_id(context)
        session_id: Optional[str] = context.thread_id if context else None

        set_resource_attributes(ResourceInfo("a2a_agent", False, resolved_endpoint))

        tool_span = mlflow.get_current_active_span()
        if tool_span:
            tool_span.set_attribute(ATTR_A2A_ENDPOINT_URL, resolved_endpoint)
            tool_span.set_attribute(ATTR_A2A_AUTH_MODE, resolved_auth_type)
            tool_span.set_attribute(ATTR_A2A_STREAMING, resolved_streaming)
            if resolved_user_id:
                tool_span.set_attribute(ATTR_A2A_USER_ID, resolved_user_id)
            if session_id:
                tool_span.set_attribute(ATTR_A2A_CONTEXT_ID, session_id)
            tool_span.set_attribute(ATTR_A2A_CONTEXT_PASSED, session_id is not None)
            tool_span.set_attribute(ATTR_A2A_PROMPT_CHARS, len(prompt))

        logger.info(
            "Invoking A2A agent",
            endpoint=resolved_endpoint,
            prompt_chars=len(prompt),
            context_id=session_id,
            user_id=resolved_user_id,
            streaming=resolved_streaming,
        )

        metadata: dict[str, Any] = {}
        if resolved_user_id:
            metadata["dao_ai.user_id"] = resolved_user_id
        if extra_metadata:
            metadata.update(extra_metadata)

        try:
            return asyncio.run(
                _invoke_a2a(
                    endpoint=resolved_endpoint,
                    header_provider=header_provider,
                    prompt=prompt,
                    context_id=session_id,
                    metadata=metadata or None,
                    card_path=resolved_card_path,
                    card_fallback_path=resolved_card_fallback,
                    streaming=resolved_streaming,
                    timeout_seconds=resolved_timeout,
                    tool_span=tool_span,
                )
            )
        except Exception as exc:  # noqa: BLE001 — surface failures to caller
            logger.error(
                "A2A agent call failed",
                error_type=type(exc).__name__,
                detail=str(exc)[:500],
            )
            return f"A2A agent call failed: {type(exc).__name__}: {str(exc)[:500]}"

    structured_tool: StructuredTool = StructuredTool.from_function(
        func=a2a_agent,
        name=tool_name,
        description=doc,
        parse_docstring=False,
    )
    return structured_tool


def _normalize_card_path(path: str) -> str:
    """Ensure the card path starts with exactly one leading slash."""
    stripped: str = path.strip()
    if not stripped:
        return AGENT_CARD_WELL_KNOWN_PATH
    return "/" + stripped.lstrip("/")


def _no_auth_headers() -> dict[str, str]:
    return {}


async def _invoke_a2a(
    *,
    endpoint: str,
    header_provider: Callable[[], dict[str, str]],
    prompt: str,
    context_id: Optional[str],
    metadata: Optional[dict[str, Any]],
    card_path: str,
    card_fallback_path: Optional[str],
    streaming: bool,
    timeout_seconds: int,
    tool_span: Any,
) -> str:
    """One round-trip to the A2A agent, with optional context-retry.

    Stage 1 sends the message with ``context_id`` (if set). Stage 2 retries
    without ``context_id`` when Stage 1 returns an empty reply or a failed
    Task whose error text references the context — mirroring the
    vertex_agent_engine session-miss retry.
    """
    reply, terminal_state = await _send_once(
        span_name="a2a_agent_initial_call",
        endpoint=endpoint,
        header_provider=header_provider,
        prompt=prompt,
        context_id=context_id,
        metadata=metadata,
        card_path=card_path,
        card_fallback_path=card_fallback_path,
        streaming=streaming,
        timeout_seconds=timeout_seconds,
        tool_span=tool_span,
    )

    retried: bool = False
    retry_reason: Optional[str] = None
    should_retry: bool = context_id is not None and (
        reply == _EMPTY_REPLY_SENTINEL
        or (terminal_state == TaskState.failed and "context" in reply.lower())
    )

    if should_retry:
        logger.warning(
            "A2A agent empty or context-failed reply; retrying without context_id",
            context_id=context_id,
            previous_state=str(terminal_state) if terminal_state else None,
        )
        retried = True
        retry_reason = (
            "empty_stream" if reply == _EMPTY_REPLY_SENTINEL else "context_rejected"
        )
        reply, terminal_state = await _send_once(
            span_name="a2a_agent_retry_without_context",
            endpoint=endpoint,
            header_provider=header_provider,
            prompt=prompt,
            context_id=None,
            metadata=metadata,
            card_path=card_path,
            card_fallback_path=card_fallback_path,
            streaming=streaming,
            timeout_seconds=timeout_seconds,
            tool_span=tool_span,
        )

    if tool_span:
        tool_span.set_attribute(ATTR_A2A_RETRIED_WITHOUT_CONTEXT, retried)
        if retry_reason:
            tool_span.set_attribute(ATTR_A2A_RETRY_REASON, retry_reason)
        tool_span.set_attribute(ATTR_A2A_RESPONSE_CHARS, len(reply))
    return reply


async def _send_once(
    *,
    span_name: str,
    endpoint: str,
    header_provider: Callable[[], dict[str, str]],
    prompt: str,
    context_id: Optional[str],
    metadata: Optional[dict[str, Any]],
    card_path: str,
    card_fallback_path: Optional[str],
    streaming: bool,
    timeout_seconds: int,
    tool_span: Any,
) -> tuple[str, Optional[TaskState]]:
    """Make one full A2A round-trip: resolve card, send message, drain stream.

    Wrapped in a caller-supplied span name so each attempt in the outer
    retry ladder shows up with distinct intent in the trace. Auth headers
    are minted fresh per call (for GCP: handles token refresh; for bearer:
    re-reads the static value). Span inputs are explicitly projected so the
    ``Authorization`` header is **never** persisted.
    """
    with mlflow.start_span(name=span_name, span_type=SpanType.TOOL) as span:
        span.set_inputs(
            {
                "endpoint": endpoint,
                "context_id_passed": context_id is not None,
                "prompt_chars": len(prompt),
                "streaming": streaming,
            }
        )
        span.set_attribute(ATTR_A2A_ENDPOINT_URL, endpoint)
        span.set_attribute(ATTR_A2A_CONTEXT_PASSED, context_id is not None)
        span.set_attribute(ATTR_A2A_PROMPT_CHARS, len(prompt))

        headers: dict[str, str] = header_provider()

        async with httpx.AsyncClient(
            headers=headers, timeout=timeout_seconds
        ) as http_client:
            card, card_path_used = await _resolve_card(
                http_client=http_client,
                endpoint=endpoint,
                card_path=card_path,
                card_fallback_path=card_fallback_path,
            )
            span.set_attribute(ATTR_A2A_CARD_PATH_USED, card_path_used)
            if card.name:
                span.set_attribute(ATTR_A2A_AGENT_NAME, card.name)
            if getattr(card, "version", None):
                span.set_attribute(ATTR_A2A_AGENT_VERSION, str(card.version))
            if tool_span:
                if card.name:
                    tool_span.set_attribute(ATTR_A2A_AGENT_NAME, card.name)
                if getattr(card, "version", None):
                    tool_span.set_attribute(ATTR_A2A_AGENT_VERSION, str(card.version))
                tool_span.set_attribute(ATTR_A2A_CARD_PATH_USED, card_path_used)

            logger.info(
                "Resolved A2A agent card",
                card_path=card_path_used,
                agent_name=card.name,
                agent_version=getattr(card, "version", None),
            )

            client = ClientFactory(
                ClientConfig(streaming=streaming, httpx_client=http_client)
            ).create(card)

            message: Message = create_text_message_object(
                role=Role.user, content=prompt
            )
            if context_id:
                message.context_id = context_id
            if metadata:
                message.metadata = dict(metadata)

            reply, terminal_state = await _drain_stream(client, message)

        span.set_outputs(
            {
                "reply_chars": len(reply),
                "terminal_state": (terminal_state.value if terminal_state else None),
            }
        )
        if terminal_state:
            span.set_attribute(ATTR_A2A_STREAM_TERMINAL_STATE, terminal_state.value)
    return reply, terminal_state


async def _resolve_card(
    *,
    http_client: httpx.AsyncClient,
    endpoint: str,
    card_path: str,
    card_fallback_path: Optional[str],
) -> tuple[AgentCard, str]:
    """Resolve the A2A agent card, trying the primary path then the fallback."""
    with mlflow.start_span(
        name="a2a_agent_card_resolve", span_type=SpanType.TOOL
    ) as span:
        span.set_inputs(
            {
                "endpoint": endpoint,
                "card_path": card_path,
                "card_fallback_path": card_fallback_path,
            }
        )
        span.set_attribute(ATTR_HTTP_METHOD, "GET")
        span.set_attribute(ATTR_HTTP_URL, endpoint + card_path)

        resolver = A2ACardResolver(
            httpx_client=http_client,
            base_url=endpoint,
            agent_card_path=card_path,
        )
        try:
            card: AgentCard = await resolver.get_agent_card()
            span.set_outputs({"card_path_used": card_path, "fallback_used": False})
            return card, card_path
        except Exception as primary_exc:  # noqa: BLE001
            if not card_fallback_path or card_fallback_path == card_path:
                raise
            logger.warning(
                "A2A primary card path missing; falling back",
                primary=card_path,
                fallback=card_fallback_path,
                error_type=type(primary_exc).__name__,
            )
            fallback_resolver = A2ACardResolver(
                httpx_client=http_client,
                base_url=endpoint,
                agent_card_path=card_fallback_path,
            )
            card = await fallback_resolver.get_agent_card()
            span.set_outputs(
                {"card_path_used": card_fallback_path, "fallback_used": True}
            )
            return card, card_fallback_path


@mlflow.trace(name="a2a_agent_parse_stream")
async def _drain_stream(
    client: Any, message: Message
) -> tuple[str, Optional[TaskState]]:
    """Drain the A2A event stream and aggregate agent text.

    The ``Client.send_message`` iterator yields either a ``Message`` (an
    inline agent message) or a ``(Task, UpdateEvent)`` tuple describing a
    Task snapshot plus the latest streaming update. Text is extracted from:

    - ``Message.parts`` when ``role == agent``,
    - ``TaskArtifactUpdateEvent.artifact.parts``,
    - ``TaskStatusUpdateEvent.status.message.parts`` when the status message
      is authored by the agent,
    - terminal ``Task.artifacts[*].parts`` and ``Task.status.message`` (in
      case the server emits only a terminal snapshot).

    Emits counters on the active span:

    - ``dao_ai.a2a.stream.total_events``
    - ``dao_ai.a2a.stream.agent_events``
    - ``dao_ai.a2a.stream.task_events``
    - ``dao_ai.a2a.stream.non_text_parts``
    - ``dao_ai.a2a.stream.terminal_state`` (last observed ``TaskState``)
    """
    pieces: list[str] = []
    total_events = 0
    agent_events = 0
    task_events = 0
    non_text_parts = 0
    terminal_state: Optional[TaskState] = None
    last_task: Optional[Task] = None

    async for event in client.send_message(message):
        total_events += 1
        if isinstance(event, Message):
            if event.role == Role.agent:
                agent_events += 1
                pieces.extend(_parts_text(event.parts))
                non_text_parts += _parts_non_text_count(event.parts)
            continue

        if isinstance(event, tuple) and len(event) == 2:
            task_obj, update = event
            task_events += 1
            if isinstance(task_obj, Task):
                last_task = task_obj
                terminal_state = task_obj.status.state if task_obj.status else None
            if isinstance(update, TaskArtifactUpdateEvent):
                pieces.extend(_parts_text(update.artifact.parts))
                non_text_parts += _parts_non_text_count(update.artifact.parts)
            elif isinstance(update, TaskStatusUpdateEvent):
                status_msg = update.status.message if update.status else None
                if status_msg and status_msg.role == Role.agent:
                    pieces.extend(_parts_text(status_msg.parts))
                    non_text_parts += _parts_non_text_count(status_msg.parts)
                if update.status and update.status.state:
                    terminal_state = update.status.state

    # If the server only sent a terminal Task (no per-event Messages or
    # artifact updates), harvest from its artifacts and status message.
    if not pieces and last_task is not None:
        for artifact in last_task.artifacts or []:
            pieces.extend(_parts_text(artifact.parts))
            non_text_parts += _parts_non_text_count(artifact.parts)
        if last_task.status and last_task.status.message:
            status_msg = last_task.status.message
            if status_msg.role == Role.agent:
                pieces.extend(_parts_text(status_msg.parts))
                non_text_parts += _parts_non_text_count(status_msg.parts)

    reply: str = "".join(pieces) if pieces else _EMPTY_REPLY_SENTINEL

    if last_task is not None and last_task.status is not None:
        if last_task.status.state == TaskState.failed:
            detail: str = reply if pieces else "(no error detail)"
            logger.warning(
                "A2A agent reported failure",
                task_id=getattr(last_task, "id", None),
                state=last_task.status.state.value,
                detail=detail[:500],
            )
            reply = f"A2A agent reported failure: {detail}"

    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute(ATTR_A2A_STREAM_TOTAL_EVENTS, total_events)
        span.set_attribute(ATTR_A2A_STREAM_AGENT_EVENTS, agent_events)
        span.set_attribute(ATTR_A2A_STREAM_TASK_EVENTS, task_events)
        span.set_attribute(ATTR_A2A_STREAM_NON_TEXT_PARTS, non_text_parts)
        if last_task is not None and getattr(last_task, "id", None):
            span.set_attribute(ATTR_A2A_TASK_ID, str(last_task.id))
        if terminal_state:
            span.set_attribute(ATTR_A2A_STREAM_TERMINAL_STATE, terminal_state.value)
        span.set_attribute(ATTR_A2A_RESPONSE_CHARS, len(reply))

    return reply, terminal_state


def _parts_text(parts: list[Part]) -> list[str]:
    out: list[str] = []
    for part in parts or []:
        inner = getattr(part, "root", part)
        if isinstance(inner, TextPart) and inner.text:
            out.append(inner.text)
    return out


def _parts_non_text_count(parts: list[Part]) -> int:
    count: int = 0
    for part in parts or []:
        inner = getattr(part, "root", part)
        if not isinstance(inner, TextPart):
            count += 1
    return count
