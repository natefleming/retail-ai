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

import requests
from google.oauth2.service_account import Credentials
from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, StructuredTool
from loguru import logger

from dao_ai.config import AnyVariable, value_of
from dao_ai.state import Context
from dao_ai.tools._gcp_auth import (
    coerce_any_variable,
    load_gcp_credentials,
    mint_gcp_access_token,
)
from dao_ai.tools.tracing import ResourceInfo, set_resource_attributes

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

        response: requests.Response = _post(
            call_url, headers, class_method, payload_input, timeout_seconds
        )

        # When a session_id is passed that was never created in ADK, the
        # endpoint can return either 404 (explicit miss) or 200 with an
        # empty body (silent miss — no events at all). Both cases require
        # retrying without session_id so ADK auto-creates a fresh session.
        if response.status_code == 404 and session_id:
            logger.warning(
                "Vertex agent session not found; retrying without session_id",
                session_id=session_id,
            )
            payload_input.pop("session_id", None)
            response = _post(
                call_url, headers, class_method, payload_input, timeout_seconds
            )

        if response.status_code != 200:
            error_text: str = response.text[:2000]
            logger.error(
                "Vertex AI Agent Engine call failed",
                status=response.status_code,
                body=error_text,
            )
            return f"Vertex agent call failed ({response.status_code}): {error_text}"

        reply: str = _extract_model_text(response)
        if reply == "(no model response)" and session_id:
            logger.warning(
                "Vertex agent returned empty stream; retrying without session_id",
                session_id=session_id,
            )
            payload_input.pop("session_id", None)
            response = _post(
                call_url, headers, class_method, payload_input, timeout_seconds
            )
            if response.status_code == 200:
                reply = _extract_model_text(response)
        return reply

    structured_tool: StructuredTool = StructuredTool.from_function(
        func=vertex_agent_engine,
        name=tool_name,
        description=doc,
        parse_docstring=False,
    )
    return structured_tool


def _post(
    url: str,
    headers: dict[str, str],
    class_method: str,
    payload_input: dict[str, Any],
    timeout: int,
) -> requests.Response:
    body: dict[str, Any] = {"class_method": class_method, "input": payload_input}
    return requests.post(url, headers=headers, json=body, timeout=timeout, stream=True)


def _extract_model_text(response: requests.Response) -> str:
    """Concatenate text parts from ``role: "model"`` events in an NDJSON stream."""
    pieces: list[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            event: dict = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Non-JSON line in GCP agent stream; skipping")
            continue
        content: dict = event.get("content") or {}
        if content.get("role") != "model":
            continue
        for part in content.get("parts") or []:
            text: str | None = part.get("text")
            if text:
                pieces.append(text)
    return "".join(pieces) if pieces else "(no model response)"
