"""Pluggable adapter registry for MCP tools.

Each adapter knows how to translate one dao-ai tool factory (identified by its
fully-qualified Python name, e.g. ``dao_ai.tools.create_genie_toolkit``) into
one or more MCP tools registered on a ``FastMCP`` server. Adapters self-register
at module import time by calling :func:`register_adapter`.

To support a new factory, add a new module under ``dao_ai.mcp.adapters`` that
calls ``register_adapter(McpAdapter(factory_name=..., register=...))`` at
import time, then import it from :mod:`dao_ai.mcp.service`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from mcp.server.fastmcp import FastMCP


AdapterRegister = Callable[
    ["FastMCP", str, dict[str, Any], "WorkspaceClient"], None
]


@dataclass(frozen=True)
class McpAdapter:
    """How to expose a dao-ai tool factory as one or more MCP tools.

    Attributes:
        factory_name: Fully-qualified Python name of the dao-ai factory
            function (matches ``FactoryFunctionModel.name`` in YAML).
        register: Callable invoked once per matching ``tools.<name>`` entry.
            Receives the FastMCP server, the YAML key (used as the MCP tool
            name unless the adapter overrides), the factory ``args`` dict, and
            a workspace client for ambient dao-ai calls.
    """

    factory_name: str
    register: AdapterRegister


ADAPTERS: dict[str, McpAdapter] = {}


def register_adapter(adapter: McpAdapter) -> None:
    """Add ``adapter`` to the global registry, overwriting any prior entry."""
    if adapter.factory_name in ADAPTERS:
        logger.warning(
            "mcp.adapter.replace", factory_name=adapter.factory_name
        )
    ADAPTERS[adapter.factory_name] = adapter
    logger.debug("mcp.adapter.register", factory_name=adapter.factory_name)


def get_adapter(factory_name: str) -> McpAdapter | None:
    """Return the adapter for ``factory_name``, or ``None`` if none registered."""
    return ADAPTERS.get(factory_name)


__all__ = [
    "ADAPTERS",
    "AdapterRegister",
    "McpAdapter",
    "get_adapter",
    "register_adapter",
]
