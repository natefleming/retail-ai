"""Config loader for the dao-ai MCP server.

Re-uses dao-ai's full :class:`dao_ai.config.AppConfig` schema. The MCP
server runs the same agent graph that ``dao_ai.apps.server`` runs, so we
call the full :meth:`AppConfig.initialize` pipeline — hooks, resource
resolution, everything the agent runtime expects.

``app.name`` is required: the MCP server derives its ``serverInfo.name`` and
the single-exposed tool's name from it.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from dao_ai.config import AppConfig

DEFAULT_LOG_LEVEL = "INFO"
# Server identity advertised back to MCP clients via `serverInfo.name` on the
# `initialize` handshake. The `mcp-` prefix is a discovery signal for
# Databricks Multi-Agent Supervisor (MAS), which pattern-matches it when
# enumerating MCP-hosted tools across an account's Apps.
DEFAULT_SERVER_NAME = "mcp-dao-ai"

SERVER_NAME_ENV = "DAO_AI_MCP_SERVER_NAME"
LOG_LEVEL_ENV = "DAO_AI_MCP_LOG_LEVEL"


def load_app_config(
    path: str | Path,
    *,
    params: dict[str, str] | None = None,
    initialize: bool = True,
) -> AppConfig:
    """Load a dao-ai ``AppConfig`` from YAML.

    ``initialize=True`` (default) runs :meth:`AppConfig.initialize` — resource
    resolution + initialization hooks. Runtime server boot needs this so the
    agent graph has all resources bound. Bundle-generation paths and unit
    tests pass ``initialize=False`` to avoid touching live Databricks
    resources.
    """
    resolved = Path(path).resolve()
    logger.debug("mcp.config.load.start", path=str(resolved))
    config: AppConfig = AppConfig.from_file(
        resolved, params=params, initialize=initialize
    )
    logger.info(
        "mcp.config.load.done",
        path=str(resolved),
        tool_count=len(config.tools),
        has_resources=config.resources is not None,
    )
    return config


def server_name_for(config: AppConfig) -> str:
    """Pick the FastMCP server name.

    Env wins (``DAO_AI_MCP_SERVER_NAME``), else ``config.app.name`` if declared,
    else the package default.
    """
    env = os.environ.get(SERVER_NAME_ENV)
    if env:
        return env
    if config.app is not None and config.app.name:
        return str(config.app.name)
    return DEFAULT_SERVER_NAME


def log_level_for(config: AppConfig) -> str:
    """Pick the loguru log level.

    Env wins (``DAO_AI_MCP_LOG_LEVEL``), else ``config.app.log_level`` if
    declared, else the package default.
    """
    env = os.environ.get(LOG_LEVEL_ENV)
    if env:
        return env
    if config.app is not None and config.app.log_level:
        return str(config.app.log_level)
    return DEFAULT_LOG_LEVEL
