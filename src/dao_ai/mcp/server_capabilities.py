"""Server-side MCP capabilities for dao-ai's own MCP server (PR 2).

Companion to the client-side callbacks/interceptors in
``src/dao_ai/tools/mcp_{callbacks,interceptors}.py``. This module registers
static resources + prompt templates on the FastMCP instance.

Progress emission from LangGraph ``astream_events`` lives in
``src/dao_ai/mcp/agent_tool.py`` because it's threaded through the
per-request ``Context`` inside the agent-as-tool wrapper.
"""

from __future__ import annotations

from typing import Any, Sequence

from loguru import logger
from mcp.server.fastmcp import FastMCP
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
