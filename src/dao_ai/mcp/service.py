"""Walks ``AppConfig.tools`` and dispatches each entry to its MCP adapter.

Importing this module triggers self-registration of the shipped adapters
(:mod:`dao_ai.mcp.adapters.genie`, :mod:`dao_ai.mcp.adapters.vector_search`).
Additional adapters added via :func:`dao_ai.mcp.adapters.register_adapter`
become available automatically — :func:`register_tools_from_config` is
discriminator-agnostic.
"""

from __future__ import annotations

from databricks.sdk import WorkspaceClient
from loguru import logger
from mcp.server.fastmcp import FastMCP

from dao_ai.config import AppConfig, FactoryFunctionModel

# Side-effect imports: each module calls register_adapter() at import time.
from dao_ai.mcp.adapters import genie as _genie_adapter  # noqa: F401
from dao_ai.mcp.adapters import get_adapter
from dao_ai.mcp.adapters import vector_search as _vector_search_adapter  # noqa: F401


def register_tools_from_config(
    mcp: FastMCP,
    config: AppConfig,
    *,
    workspace_client: WorkspaceClient | None = None,
) -> set[str]:
    """Register an MCP tool for every recognized ``config.tools.<name>`` entry.

    Returns the set of MCP tool *names* registered (i.e. the YAML keys plus any
    auxiliary tools an adapter registers — Genie adds ``<name>_feedback``).
    """
    workspace_client = workspace_client or WorkspaceClient()
    registered: set[str] = set()
    skipped: list[tuple[str, str]] = []

    for tool_name, tool_def in config.tools.items():
        fn = tool_def.function
        if not isinstance(fn, FactoryFunctionModel):
            skipped.append(
                (tool_name, f"unsupported function type {type(fn).__name__}")
            )
            continue

        adapter = get_adapter(fn.name)
        if adapter is None:
            skipped.append((tool_name, f"no MCP adapter for factory {fn.name}"))
            continue

        try:
            adapter.register(mcp, tool_name, dict(fn.args or {}), workspace_client)
            registered.add(tool_name)
        except Exception:
            logger.exception(
                "mcp.service.adapter.failed",
                tool_name=tool_name,
                factory_name=fn.name,
            )
            skipped.append((tool_name, "adapter raised — see traceback above"))

    for tool_name, reason in skipped:
        logger.warning("mcp.service.tool.skip", tool_name=tool_name, reason=reason)

    logger.info(
        "mcp.service.registered",
        count=len(registered),
        registered=sorted(registered),
        skipped=[name for name, _ in skipped],
    )

    if not registered:
        raise ValueError(
            "AppConfig.tools produced no MCP-registerable tools; need at least one "
            "entry whose function.name matches a registered MCP adapter (e.g. "
            "'dao_ai.tools.create_genie_toolkit' or "
            "'dao_ai.tools.create_vector_search_tool')."
        )

    return registered


__all__ = ["register_tools_from_config"]
