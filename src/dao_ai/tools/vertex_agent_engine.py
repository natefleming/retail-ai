"""Vertex AI Agent Engine tool.

Calls a Google ADK agent deployed on Google Cloud's Vertex AI Agent Engine
(formerly Reasoning Engine) via the ``:streamQuery`` REST endpoint, parses
the NDJSON response stream, and returns the concatenated model reply as a
**single string**.

Why streaming? Vertex Agent Engine maps ADK methods whose ``api_mode`` is
``stream`` / ``async_stream`` to the ``:streamQuery`` HTTP endpoint. Typical
ADK agents (including this demo's ``customer_service_agent``) only expose
``stream_query`` and ``async_stream_query`` — they have no sync ``query``
method — so ``:streamQuery`` is the only way in. The streaming is a
protocol-level requirement of ADK, not a design choice of this tool.

**Tool callers never see streaming.** ``_extract_model_text()`` drains the
NDJSON stream internally, aggregates every ``role: "model"`` text part, and
returns one concatenated string — the same shape as any other synchronous
LangChain tool. Token-by-token chunks do not leak to the calling agent.

DAO AI ``Context`` feeds the ADK agent naturally:

- ``context.user_id`` → ADK ``user_id`` (scopes sessions, memory, and
  personalization — for example, the demo agent loads customer profiles
  keyed by ``user_id``).
- ``context.thread_id`` → ADK ``session_id`` (groups turns into a single
  conversation so ``state_delta`` events carry over across calls).

**Session handling note.** If ``session_id`` is passed but the session was
never created in ADK, Vertex returns *either*:

- HTTP 404 (explicit miss), or
- HTTP 200 with ``Content-Length: 0`` (silent miss — empty stream, no events).

Both failure modes are caught here and retried without ``session_id``,
which causes ADK to auto-create a fresh session. The caller receives the
agent's reply transparently.
"""

import json
from textwrap import dedent
from typing import Annotated, Any, Callable, Optional

import mlflow
import requests
from google.oauth2.service_account import Credentials
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
    ATTR_HTTP_METHOD,
    ATTR_HTTP_RESP_LEN,
    ATTR_HTTP_STATUS,
    ATTR_HTTP_URL,
    ATTR_VERTEX_CLASS_METHOD,
    ATTR_VERTEX_ENDPOINT_URL,
    ATTR_VERTEX_HTTP_METHOD,
    ATTR_VERTEX_PROMPT_CHARS,
    ATTR_VERTEX_RESPONSE_CHARS,
    ATTR_VERTEX_RETRIED_WITHOUT_SESSION,
    ATTR_VERTEX_RETRY_REASON,
    ATTR_VERTEX_SESSION_ID,
    ATTR_VERTEX_SESSION_PASSED,
    ATTR_VERTEX_STREAM_MODEL_EVENTS,
    ATTR_VERTEX_STREAM_NONJSON_LINES,
    ATTR_VERTEX_STREAM_STATE_DELTA_EVENTS,
    ATTR_VERTEX_STREAM_TOTAL_LINES,
    ATTR_VERTEX_USER_ID,
    ResourceInfo,
    set_resource_attributes,
)

_DEFAULT_DESCRIPTION: str = dedent("""
    Send a message to a Vertex AI Agent Engine (Google ADK) agent and return
    the assistant's reply. Use this tool to delegate questions to the remote
    Vertex agent. Pass the user's question or prompt as the 'prompt' argument.
""").strip()


def create_vertex_agent_engine_tool(
    endpoint: AnyVariable,
    credentials: AnyVariable,
    user_id: Optional[AnyVariable] = None,
    class_method: str = "stream_query",
    http_method: str = "streamQuery",
    timeout_seconds: int = 300,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[..., str]:
    """Create a tool that calls a Vertex AI Agent Engine endpoint.

    Args:
        endpoint: Full URL of the reasoning engine resource, e.g.
            ``https://us-central1-aiplatform.googleapis.com/v1/projects/<p>/locations/<r>/reasoningEngines/<id>``.
            A trailing ``:query`` or ``:streamQuery`` suffix, if present, is
            stripped — the effective suffix is controlled by ``http_method``.
        credentials: Service-account credentials. Accepts a local file path,
            a Databricks volume path (``/Volumes/...``), or an inline JSON
            body. Any ``AnyVariable`` form is supported (env, secret,
            composite).
        user_id: Static ``user_id`` sent to the ADK agent. If omitted, the
            tool uses ``runtime.context.user_id`` from the DAO AI request.
        class_method: ADK class method to invoke. Defaults to
            ``stream_query``; override to ``async_stream_query`` or another
            method exposed by the agent. Call ``GET`` on the engine URL to
            see available methods under ``spec.classMethods``.
        http_method: Vertex endpoint suffix. Defaults to ``streamQuery`` to
            match ``api_mode: stream`` ADK methods. Use ``query`` only if
            the target ADK method has ``api_mode: ""`` (sync, non-stream).
        timeout_seconds: Socket timeout for the HTTP call.
        name: Custom tool name. Defaults to ``vertex_agent_engine``.
        description: Custom tool description shown to the LLM.

    Returns:
        A :class:`StructuredTool` that sends a message and returns the
        aggregated model text as a **single string** (not streamed).

    Notes:
        Session continuity: ``context.thread_id`` is forwarded as the ADK
        ``session_id`` when present. If ADK returns 404 or a 200 with empty
        body (both indicate the session_id is unknown), the tool retries
        once without ``session_id`` so ADK auto-creates a fresh session.
    """
    logger.debug(
        "Creating Vertex AI Agent Engine tool",
        name=name,
        class_method=class_method,
    )

    resolved_endpoint: str = str(value_of(coerce_any_variable(endpoint))).rstrip("/")
    last_segment: str = resolved_endpoint.rsplit("/", 1)[-1]
    if ":" in last_segment:
        resolved_endpoint, _, _ = resolved_endpoint.rpartition(":")
    call_url: str = f"{resolved_endpoint}:{http_method}"

    creds: Credentials = load_gcp_credentials(credentials)

    tool_name: str = name if name else "vertex_agent_engine"
    doc_description: str = description if description else _DEFAULT_DESCRIPTION
    doc_signature: str = dedent("""
    Args:
        prompt (str): Message to send to the Vertex agent.

    Returns:
        str: The concatenated text response from the agent.
    """)
    doc: str = doc_description + "\n" + doc_signature

    def _resolve_user_id(context: Context | None) -> str:
        if user_id is not None:
            return str(value_of(coerce_any_variable(user_id)))
        if context and context.user_id:
            return context.user_id
        raise ValueError(
            "Vertex AI Agent Engine tool requires user_id. Set it in the "
            "factory config or ensure the DAO AI Context provides user_id."
        )

    def vertex_agent_engine(
        prompt: str,
        runtime: Annotated[ToolRuntime[Context], InjectedToolArg] = None,
    ) -> str:
        context: Context | None = runtime.context if runtime else None
        resolved_user_id: str = _resolve_user_id(context)
        session_id: str | None = context.thread_id if context else None

        set_resource_attributes(ResourceInfo("vertex_agent_engine", False, call_url))

        tool_span = mlflow.get_current_active_span()
        if tool_span:
            tool_span.set_attribute(ATTR_VERTEX_ENDPOINT_URL, call_url)
            tool_span.set_attribute(ATTR_VERTEX_CLASS_METHOD, class_method)
            tool_span.set_attribute(ATTR_VERTEX_HTTP_METHOD, http_method)
            tool_span.set_attribute(ATTR_VERTEX_USER_ID, resolved_user_id)
            if session_id:
                tool_span.set_attribute(ATTR_VERTEX_SESSION_ID, session_id)
            tool_span.set_attribute(ATTR_VERTEX_SESSION_PASSED, session_id is not None)
            tool_span.set_attribute(ATTR_VERTEX_PROMPT_CHARS, len(prompt))

        token: str = mint_gcp_access_token(creds)
        headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        payload_input: dict[str, Any] = {
            "message": prompt,
            "user_id": resolved_user_id,
        }
        if session_id:
            payload_input["session_id"] = session_id

        # Stage 1 — send the initial request (may or may not carry session_id).
        response, reply = _invoke_once(
            "vertex_agent_initial_call",
            call_url,
            headers,
            class_method,
            payload_input,
            timeout_seconds,
        )

        retried: bool = False
        retry_reason: str | None = None

        # Stage 2a — explicit 404 miss. Session_id was passed but doesn't exist
        # in ADK. Retry with it stripped.
        if response.status_code == 404 and session_id:
            logger.warning(
                "Vertex agent session not found; retrying without session_id",
                session_id=session_id,
            )
            retried = True
            retry_reason = "http_404"
            payload_input.pop("session_id", None)
            response, reply = _invoke_once(
                "vertex_agent_retry_after_404",
                call_url,
                headers,
                class_method,
                payload_input,
                timeout_seconds,
            )

        if response.status_code != 200:
            error_text: str = response.text[:2000]
            logger.error(
                "Vertex AI Agent Engine call failed",
                status=response.status_code,
                body=error_text,
            )
            if tool_span:
                tool_span.set_attribute(ATTR_VERTEX_RETRIED_WITHOUT_SESSION, retried)
                if retry_reason:
                    tool_span.set_attribute(ATTR_VERTEX_RETRY_REASON, retry_reason)
            return f"Vertex agent call failed ({response.status_code}): {error_text}"

        # Stage 2b — silent miss: 200 OK but the stream produced no model
        # events. Happens when ADK doesn't recognize the session_id.
        if reply == "(no model response)" and session_id:
            logger.warning(
                "Vertex agent returned empty stream; retrying without session_id",
                session_id=session_id,
            )
            retried = True
            retry_reason = "empty_body"
            payload_input.pop("session_id", None)
            response, reply = _invoke_once(
                "vertex_agent_retry_after_empty_stream",
                call_url,
                headers,
                class_method,
                payload_input,
                timeout_seconds,
            )

        if tool_span:
            tool_span.set_attribute(ATTR_VERTEX_RETRIED_WITHOUT_SESSION, retried)
            if retry_reason:
                tool_span.set_attribute(ATTR_VERTEX_RETRY_REASON, retry_reason)
            tool_span.set_attribute(ATTR_VERTEX_RESPONSE_CHARS, len(reply))
        return reply

    structured_tool: StructuredTool = StructuredTool.from_function(
        func=vertex_agent_engine,
        name=tool_name,
        description=doc,
        parse_docstring=False,
    )
    return structured_tool


def _invoke_once(
    span_name: str,
    url: str,
    headers: dict[str, str],
    class_method: str,
    payload_input: dict[str, Any],
    timeout: int,
) -> tuple[requests.Response, str]:
    """Make one full round-trip to Vertex (HTTP POST + NDJSON parse).

    Wrapped in a caller-supplied span name so each attempt in the outer
    retry ladder shows up with distinct intent in the trace — callers pass
    e.g. ``"vertex_agent_initial_call"``, ``"vertex_agent_retry_after_404"``,
    or ``"vertex_agent_retry_after_empty_stream"``. Under this span you'll
    see two child spans: ``vertex_agent_http_post`` (the wire call) and
    ``vertex_agent_parse_stream`` (the NDJSON parse).

    Returns:
        A pair ``(response, reply)``. If the HTTP status is not 200, the
        reply is an empty string and the caller is expected to check
        ``response.status_code`` before acting on ``reply``.
    """
    with mlflow.start_span(name=span_name, span_type=SpanType.TOOL) as span:
        # Summary inputs only — no headers, no tokens, no internal state.
        span.set_inputs(
            {
                "class_method": class_method,
                "session_id_passed": "session_id" in payload_input,
                "prompt_chars": len(str(payload_input.get("message") or "")),
            }
        )
        response: requests.Response = _post(
            url, headers, class_method, payload_input, timeout
        )
        reply: str = ""
        if response.status_code == 200:
            reply = _extract_model_text(response)
        span.set_outputs(
            {
                "status_code": response.status_code,
                "reply_chars": len(reply),
            }
        )
    return response, reply


def _post(
    url: str,
    headers: dict[str, str],
    class_method: str,
    payload_input: dict[str, Any],
    timeout: int,
) -> requests.Response:
    """POST one request to the Vertex reasoning engine.

    Uses manual span management (``mlflow.start_span``) instead of
    ``@mlflow.trace`` so the caller-supplied ``headers`` dict — which
    carries the OAuth ``Authorization: Bearer <token>`` — is never
    persisted as span input. Inputs are set explicitly from a redacted
    projection.

    Nested under the caller's ``_invoke_once`` span so the HTTP hop
    appears as a child alongside the stream-parse hop.
    """
    body: dict[str, Any] = {"class_method": class_method, "input": payload_input}

    with mlflow.start_span(
        name="vertex_agent_http_post", span_type=SpanType.TOOL
    ) as span:
        # Explicit inputs — no ``headers`` key, no token leak.
        span.set_inputs(
            {
                "url": url,
                "class_method": class_method,
                "payload_input": payload_input,
                "timeout": timeout,
            }
        )
        span.set_attribute(ATTR_HTTP_METHOD, "POST")
        span.set_attribute(ATTR_HTTP_URL, url)
        span.set_attribute(ATTR_VERTEX_CLASS_METHOD, class_method)
        span.set_attribute(ATTR_VERTEX_SESSION_PASSED, "session_id" in payload_input)
        msg = payload_input.get("message")
        if isinstance(msg, str):
            span.set_attribute(ATTR_VERTEX_PROMPT_CHARS, len(msg))

        r: requests.Response = requests.post(
            url, headers=headers, json=body, timeout=timeout, stream=True
        )

        span.set_attribute(ATTR_HTTP_STATUS, r.status_code)
        content_length = r.headers.get("Content-Length")
        if content_length is not None:
            try:
                span.set_attribute(ATTR_HTTP_RESP_LEN, int(content_length))
            except ValueError:
                pass
        # Explicit outputs — a short summary, not the raw Response repr.
        span.set_outputs(
            {
                "status_code": r.status_code,
                "content_length": (
                    int(content_length)
                    if content_length and content_length.isdigit()
                    else None
                ),
            }
        )
    return r


@mlflow.trace(name="vertex_agent_parse_stream")
def _extract_model_text(response: requests.Response) -> str:
    """Drain the NDJSON stream and return the concatenated model text.

    Emits counters onto the span:

    - ``dao_ai.vertex.stream.total_lines`` — non-empty lines seen
    - ``dao_ai.vertex.stream.model_events`` — events with ``role: "model"``
    - ``dao_ai.vertex.stream.state_delta_events`` — session-state events
    - ``dao_ai.vertex.stream.non_json_lines`` — parser-skipped junk
    - ``dao_ai.vertex.response_chars`` — length of the concatenated reply
    """
    pieces: list[str] = []
    total_lines = 0
    model_events = 0
    state_delta_events = 0
    non_json_lines = 0

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        total_lines += 1
        try:
            event: dict = json.loads(line)
        except json.JSONDecodeError:
            non_json_lines += 1
            logger.debug("Non-JSON line in GCP agent stream; skipping")
            continue
        actions = event.get("actions") or {}
        if isinstance(actions, dict) and actions.get("state_delta"):
            state_delta_events += 1
        content: dict = event.get("content") or {}
        if content.get("role") != "model":
            continue
        model_events += 1
        for part in content.get("parts") or []:
            text: str | None = part.get("text")
            if text:
                pieces.append(text)

    reply: str = "".join(pieces) if pieces else "(no model response)"

    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute(ATTR_VERTEX_STREAM_TOTAL_LINES, total_lines)
        span.set_attribute(ATTR_VERTEX_STREAM_MODEL_EVENTS, model_events)
        span.set_attribute(ATTR_VERTEX_STREAM_STATE_DELTA_EVENTS, state_delta_events)
        span.set_attribute(ATTR_VERTEX_STREAM_NONJSON_LINES, non_json_lines)
        span.set_attribute(ATTR_VERTEX_RESPONSE_CHARS, len(reply))
    return reply
