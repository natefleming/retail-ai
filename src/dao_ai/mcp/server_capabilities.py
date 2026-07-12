"""Server-side MCP capabilities for dao-ai's own MCP server (PR 2).

Companion to the client-side callbacks/interceptors in
``src/dao_ai/tools/mcp_{callbacks,interceptors}.py``. This module registers
static resources + prompt templates on the FastMCP instance and wires a
``logging.Handler`` that forwards Python log records into
``notifications/message`` on the active FastMCP session.

Progress emission from LangGraph ``astream_events`` lives in
``src/dao_ai/mcp/agent_tool.py`` because it's threaded through the
per-request ``Context`` inside the agent-as-tool wrapper.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from loguru import logger
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts.base import Prompt, PromptArgument
from mcp.types import PromptMessage, TextContent

from dao_ai.config import (
    McpPromptModel,
    McpResourceModel,
)


def register_resources(
    mcp: FastMCP,
    resources: Sequence[McpResourceModel],
) -> list[str]:
    """Register static resources on the FastMCP instance.

    Returns the list of URIs registered so the caller can advertise them on
    ``/healthz``.
    """
    registered: list[str] = []
    for resource in resources:
        _register_single_resource(mcp, resource)
        registered.append(resource.uri)
    if registered:
        logger.info(
            "mcp.server.resources.registered",
            count=len(registered),
            uris=registered,
        )
    return registered


def _register_single_resource(mcp: FastMCP, resource: McpResourceModel) -> None:
    """Bind one resource on the FastMCP instance via ``@mcp.resource``.

    ``@mcp.resource(uri=...)`` requires a decorated function that returns
    the payload — we close over ``resource.content`` and hand FastMCP a
    zero-arg callable.
    """
    content: str = resource.content
    mime_type: str = resource.mime_type

    @mcp.resource(
        uri=resource.uri,
        name=resource.name,
        description=resource.description,
        mime_type=mime_type,
    )
    def _read() -> str:
        return content


def register_prompts(
    mcp: FastMCP,
    prompts: Sequence[McpPromptModel],
) -> list[str]:
    """Register prompt templates on the FastMCP instance.

    Returns the list of prompt names registered so ``/healthz`` can list them.
    """
    registered: list[str] = []
    for prompt in prompts:
        _register_single_prompt(mcp, prompt)
        registered.append(prompt.name)
    if registered:
        logger.info(
            "mcp.server.prompts.registered",
            count=len(registered),
            names=registered,
        )
    return registered


def _register_single_prompt(mcp: FastMCP, prompt: McpPromptModel) -> None:
    """Bind one prompt template on the FastMCP instance.

    We build a ``mcp.server.fastmcp.prompts.Prompt`` directly (rather than
    the ``@mcp.prompt`` decorator) so we can materialize the argument list
    from config and render the template with client-supplied values.
    """
    template: str = prompt.template
    prompt_name: str = prompt.name
    prompt_description: str | None = prompt.description
    arg_schema: list[PromptArgument] = [
        PromptArgument(
            name=arg.name,
            description=arg.description,
            required=arg.required,
        )
        for arg in prompt.arguments
    ]

    async def _render(**kwargs: Any) -> list[PromptMessage]:
        # Default missing optional args to empty strings so the template
        # renders even when the client omits them.
        rendered = template.format_map(_DefaultingDict(kwargs))
        return [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=rendered),
            )
        ]

    mcp.add_prompt(
        Prompt.from_function(
            _render,
            name=prompt_name,
            description=prompt_description,
        )
        if False
        else Prompt(
            name=prompt_name,
            description=prompt_description,
            arguments=arg_schema,
            fn=_render,
        )
    )


class _DefaultingDict(dict):
    """dict that returns '' for missing keys — used by ``str.format_map``.

    Lets prompts with optional arguments render cleanly when the client
    omits them (an ``__missing__`` returning '' is idiomatic for this).
    """

    def __missing__(self, key: str) -> str:
        return ""


# ---------------------------------------------------------------------------
# Log forwarding: Python logger → MCP notifications/message
# ---------------------------------------------------------------------------


_LEVEL_MAP: dict[int, str] = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "critical",
}


class MCPSessionLoggingHandler(logging.Handler):
    """Forwards Python log records to the active FastMCP session as
    ``notifications/message``.

    Guarded for absence of session context — when there's no active FastMCP
    session (module-level logs, background threads without a request in
    flight), ``emit`` is a silent no-op.
    """

    def __init__(self, mcp: FastMCP) -> None:
        super().__init__()
        self._mcp = mcp

    def emit(self, record: logging.LogRecord) -> None:
        session = _current_fastmcp_session(self._mcp)
        if session is None:
            return
        try:
            level = _LEVEL_MAP.get(record.levelno, "info")
            message = self.format(record)
            import anyio

            anyio.from_thread.run_sync(
                session.send_log_message,
                level,
                message,
                record.name,
            )
        except Exception:
            # Never let logging break the process — silence and move on.
            return


def _current_fastmcp_session(mcp: FastMCP) -> Any | None:
    """Return the active FastMCP session, if one is bound.

    FastMCP stashes the per-request session behind ``mcp.get_context().session``.
    Both the ``get_context()`` call AND the ``.session`` property can raise
    when there's no bound request — treat either as 'no session'.
    """
    try:
        ctx: Context = mcp.get_context()
        return ctx.session
    except Exception:
        return None


def wire_log_forwarding(mcp: FastMCP) -> None:
    """Attach an ``MCPSessionLoggingHandler`` to the root logger.

    Idempotent — a second call replaces the handler rather than stacking.
    """
    handler = MCPSessionLoggingHandler(mcp)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    # Drop any existing dao-ai MCP handler before attaching (idempotent).
    for existing in list(root.handlers):
        if isinstance(existing, MCPSessionLoggingHandler):
            root.removeHandler(existing)
    root.addHandler(handler)
    logger.info("mcp.server.log_forwarding.wired")
