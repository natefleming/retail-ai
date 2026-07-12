"""Expose the full dao-ai agent as a single MCP tool.

Registers exactly one MCP tool that delegates to the whole agent graph
built by :meth:`AppConfig.as_responses_agent`. The agent's own orchestrator
picks tools internally; MCP clients see one high-level capability. Tool
name and description come from ``config.app.name`` /
``config.app.description``. Headers captured by
:class:`RequestContextMiddleware` are threaded into the request's
``custom_inputs.configurable.headers`` so downstream Genie / Vector Search /
UC function calls inherit the caller's identity via the same OBO plumbing
:mod:`dao_ai.apps.handlers` uses.

The tool returns an MCP :class:`CallToolResult` with:

* ``content[0]`` — plain-text ``final_message`` for legacy clients.
* ``structuredContent`` — a validated :class:`AgentInvocationResult`
  (``final_message``, ``trace_id``, ``confidence``) so schema-aware
  callers get typed access.
* ``_meta.databricks.*`` — trace_id, experiment_id, model, latency_ms,
  request_id. Reverse-DNS namespace per MCP `_meta` conventions.
* ``isError=True`` — set when ``apredict`` raises, so the caller LLM
  still receives the error text (unlike a JSON-RPC error which loses
  content).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import time
from typing import Any, Optional

from loguru import logger
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, TextContent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from pydantic import BaseModel, Field

from dao_ai.config import AppConfig
from dao_ai.mcp._request_context import current_request_headers, current_request_id
from dao_ai.models import LanggraphResponsesAgent

_SLUG_RE = re.compile(r"[^a-z0-9_]+")

# Progress emission ------------------------------------------------------------

_HEARTBEAT_INTERVAL_SEC = 2.0
_HEARTBEAT_STEP = 5.0
_HEARTBEAT_CEILING = 95.0


async def _heartbeat_progress(ctx: Context, tool_name: str) -> None:
    """Emit periodic in-flight progress notifications until cancelled.

    Interpolates from 5→95 over the life of the enclosing agent call so the
    client sees the tool is alive even when a Genie / VS / UC-fn hop takes
    tens of seconds. Never reaches 100 — the terminal ``agent_complete`` /
    ``agent_error`` progress is emitted by the caller after ``apredict``
    returns/raises.

    Silences all exceptions per MCP progress-callback convention: a broken
    progress channel must not sink the tool call.
    """
    progress = 0.0
    try:
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_SEC)
            progress = min(progress + _HEARTBEAT_STEP, _HEARTBEAT_CEILING)
            try:
                await ctx.report_progress(
                    progress=progress, total=100.0, message="agent_in_flight"
                )
            except Exception:
                logger.trace(
                    "mcp.agent_tool.progress.heartbeat.skip",
                    tool_name=tool_name,
                    progress=progress,
                )
                return
    except asyncio.CancelledError:
        raise


def _token_fingerprint(token: str) -> str:
    """Return a short opaque fingerprint of *token* for diagnostic logging.

    Never logs the raw token. SHA-256 hex prefix is stable per-token so
    operators can diff two calls (e.g. OBO on vs off) and see whether the
    identity changed.
    """
    if not token:
        return ""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _token_subject(token: str) -> Optional[str]:
    """Best-effort decode of a JWT's ``sub`` claim, no signature verification.

    Returns ``None`` if the token isn't a well-formed JWT or has no ``sub``.
    Used only for diagnostic logs — never for authz decisions.
    """
    try:
        _header, payload, _sig = token.split(".", 2)
        padding = "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload + padding)
        claims = json.loads(decoded)
        sub = claims.get("sub")
        if isinstance(sub, str):
            return sub
    except Exception:  # pragma: no cover — best effort only
        pass
    return None


class AgentInvocationResult(BaseModel):
    """Structured response for the agent-as-tool MCP call.

    Emitted as ``structuredContent`` on every :class:`CallToolResult`. The
    schema is advertised on the MCP tool's ``outputSchema`` so
    schema-aware clients can validate + destructure the response.
    """

    final_message: str = Field(
        description="The agent's final assistant-message text.",
    )
    trace_id: Optional[str] = Field(
        default=None,
        description=(
            "Fully qualified MLflow trace id "
            "(``trace:/<catalog>.<schema>.<prefix>/<hex>``) when the agent "
            "has ``app.trace_location`` configured. ``None`` otherwise."
        ),
    )
    confidence: Optional[float] = Field(
        default=None,
        description=(
            "Reserved. Not populated by dao-ai today; kept in the schema so "
            "future agent middleware can attach a confidence estimate "
            "without a schema break."
        ),
    )


def _slugify(name: str) -> str:
    """Slugify an app name into a valid MCP tool identifier."""
    lowered: str = name.strip().lower().replace("-", "_")
    return _SLUG_RE.sub("_", lowered).strip("_") or "invoke_agent"


def _default_description(app_name: str) -> str:
    return (
        f"Invoke the {app_name} dao-ai agent. Accepts either a user question "
        "as a string or a Responses-style input array of message dicts."
    )


def _normalize_input(user_input: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce string input into the Responses ``input`` array shape."""
    if isinstance(user_input, str):
        return [{"role": "user", "content": user_input}]
    return user_input


def _extract_final_assistant_text(response: ResponsesAgentResponse) -> str:
    """Concatenate ``output_text`` parts from the last assistant message.

    ``ResponsesAgentResponse.output`` may contain either raw dicts (as
    produced by :func:`mlflow.types.responses.create_text_output_item`) or
    pydantic ``OutputItem`` instances depending on how the response was
    constructed. Handle both shapes.
    """

    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    for item in reversed(response.output):
        if _get(item, "role") != "assistant" or _get(item, "type") != "message":
            continue
        content: Any = _get(item, "content") or []
        parts: list[str] = []
        for content_item in content:
            if _get(content_item, "type") == "output_text":
                text = _get(content_item, "text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)
    return ""


def _extract_trace_id(response: ResponsesAgentResponse) -> Optional[str]:
    """Read ``custom_outputs.trace_id`` if the ResponsesAgent surfaced it.

    dao-ai's ``LanggraphResponsesAgent.apredict`` writes ``trace_id`` into
    ``custom_outputs`` when MLflow tracing is active (see
    ``reference_dao_ai_trace_id_in_custom_outputs.md``).
    """
    outputs = getattr(response, "custom_outputs", None) or {}
    if isinstance(outputs, dict):
        tid = outputs.get("trace_id")
        if isinstance(tid, str):
            return tid
    return None


def _first_agent_model_name(config: AppConfig) -> Optional[str]:
    """Best-effort read of the primary agent's model name for _meta."""
    try:
        if config.app and config.app.agents:
            model = config.app.agents[0].model
            name = getattr(model, "name", None)
            if isinstance(name, str):
                return name
    except Exception:  # pragma: no cover — best effort only
        pass
    return None


def _build_meta(
    *,
    trace_id: Optional[str],
    model_name: Optional[str],
    latency_ms: float,
    request_id: str,
    obo_present: bool,
) -> dict[str, Any]:
    """Assemble ``_meta.databricks.*`` for observability.

    ``obo_present`` reflects whether the caller forwarded a
    ``x-forwarded-access-token`` header — useful for verifying OBO
    passthrough end-to-end without needing a diagnostic tool.
    """
    meta: dict[str, Any] = {
        "databricks.latency_ms": round(latency_ms, 3),
        "databricks.obo_present": obo_present,
    }
    if trace_id:
        meta["databricks.trace_id"] = trace_id
    experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
    if experiment_id:
        meta["databricks.experiment_id"] = experiment_id
    if model_name:
        meta["databricks.model"] = model_name
    if request_id:
        meta["databricks.request_id"] = request_id
    return meta


def register_agent_as_tool(mcp: FastMCP, config: AppConfig) -> str:
    """Register the dao-ai agent as a single MCP tool on *mcp*.

    Returns the registered tool name so ``server.build_app`` can advertise it
    on ``/healthz``.
    """
    if config.app is None or not config.app.name:
        raise ValueError(
            "generate-mcp requires config.app.name to derive the MCP tool "
            "name. Set the top-level `app.name` field in your dao-ai config."
        )

    app_name: str = str(config.app.name)
    tool_name: str = _slugify(app_name)
    description: str = (
        str(config.app.description)
        if config.app.description
        else _default_description(app_name)
    )

    agent: LanggraphResponsesAgent = config.as_responses_agent()  # type: ignore[assignment]
    model_name = _first_agent_model_name(config)

    _mcp_server_cfg = (
        getattr(config.app, "mcp_server", None) if config.app is not None else None
    )
    progress_enabled: bool = _mcp_server_cfg is not None and _mcp_server_cfg.progress
    logger.info(
        "mcp.agent_tool.progress_config",
        tool_name=tool_name,
        mcp_server_present=_mcp_server_cfg is not None,
        progress_enabled=progress_enabled,
    )

    @mcp.tool(name=tool_name, description=description)
    async def invoke_agent(
        input: str | list[dict[str, Any]],
        ctx: Context,
    ) -> AgentInvocationResult:
        """Delegate the caller's input to the dao-ai agent graph.

        String input is wrapped into a single user turn; a list of message
        dicts is passed through as the Responses ``input`` array unchanged.
        Returns the final assistant text plus MLflow ``trace_id`` for
        observability. On agent-side failure, returns a
        :class:`CallToolResult` with ``isError=True`` so the caller LLM
        still receives the error text.
        """
        normalized: list[dict[str, Any]] = _normalize_input(input)
        headers: dict[str, str] = dict(current_request_headers())
        request_id: str = current_request_id() or ""
        obo_token: str = headers.get("x-forwarded-access-token", "")
        obo_present: bool = bool(obo_token)

        # OBO diagnostic: log a stable token fingerprint + decoded `sub`
        # claim (best-effort) so operators can verify end-to-end
        # propagation from an MCP client through this server into
        # downstream Databricks resources. The raw token is never logged.
        logger.info(
            "mcp.agent_tool.invoke.headers",
            tool_name=tool_name,
            request_id=request_id,
            obo_present=obo_present,
            obo_token_fingerprint=(
                _token_fingerprint(obo_token) if obo_present else None
            ),
            obo_token_subject=(
                _token_subject(obo_token) if obo_present else None
            ),
            header_count=len(headers),
        )

        custom_inputs: dict[str, Any] = {}
        if headers:
            custom_inputs["configurable"] = {"headers": headers}

        request = ResponsesAgentRequest(
            input=normalized,
            custom_inputs=custom_inputs or None,
        )

        # The server-side trace root is ``dao_ai_apredict`` (from the
        # ``@mlflow.trace`` decorator on
        # :meth:`LanggraphResponsesAgent.apredict`) once
        # ``MLFLOW_EXPERIMENT_ID`` is bound at boot — no explicit outer
        # span needed. The apredict span already carries agent-level
        # observability; MCP-specific attributes (``request_id``,
        # ``obo_present``) live on the ``_meta`` block of the response
        # rather than on a synthetic span.
        start = time.perf_counter()
        heartbeat_task: asyncio.Task[None] | None = None
        if progress_enabled:
            try:
                await ctx.report_progress(progress=0.0, total=100.0, message="agent_start")
            except Exception as _exc:
                # Progress channel failures must not fail the tool call —
                # log at debug so operators can surface if needed.
                logger.debug(
                    "mcp.agent_tool.progress.start.skip",
                    tool_name=tool_name,
                    error=str(_exc),
                )
            heartbeat_task = asyncio.create_task(
                _heartbeat_progress(ctx, tool_name)
            )
        try:
            response: ResponsesAgentResponse = await agent.apredict(request)
            text = _extract_final_assistant_text(response)
            trace_id = _extract_trace_id(response)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "mcp.agent_tool.apredict.failed",
                tool_name=tool_name,
                request_id=request_id,
                latency_ms=latency_ms,
            )
            if heartbeat_task is not None:
                heartbeat_task.cancel()
            error_text = f"Agent invocation failed: {exc}"
            error_result = AgentInvocationResult(final_message=error_text)
            return CallToolResult(
                content=[TextContent(type="text", text=error_text)],
                structuredContent=error_result.model_dump(mode="json"),
                isError=True,
                _meta=_build_meta(
                    trace_id=None,
                    model_name=model_name,
                    latency_ms=latency_ms,
                    request_id=request_id,
                    obo_present=obo_present,
                ),
            )  # type: ignore[return-value]

        latency_ms = (time.perf_counter() - start) * 1000

        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await ctx.report_progress(
                    progress=100.0, total=100.0, message="agent_complete"
                )
            except Exception:
                logger.trace(
                    "mcp.agent_tool.progress.complete.skip", tool_name=tool_name
                )

        result_model = AgentInvocationResult(
            final_message=text,
            trace_id=trace_id,
        )
        structured = result_model.model_dump(mode="json")

        # Emit the plain-text content for legacy clients + the structured
        # JSON so schema-aware callers get typed access. Returning
        # CallToolResult directly is supported by FastMCP
        # (see mcp.server.fastmcp.utilities.func_metadata.convert_result).
        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            structuredContent=structured,
            isError=False,
            _meta=_build_meta(
                trace_id=trace_id,
                model_name=model_name,
                latency_ms=latency_ms,
                request_id=request_id,
                obo_present=obo_present,
            ),
        )  # type: ignore[return-value]

    logger.info(
        "mcp.agent_tool.registered",
        tool_name=tool_name,
        app_name=app_name,
        description_len=len(description),
        model_name=model_name,
    )
    return tool_name


__all__ = ["AgentInvocationResult", "register_agent_as_tool"]
