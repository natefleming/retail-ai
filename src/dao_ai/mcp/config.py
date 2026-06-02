"""Config loader for the dao-ai MCP server.

Re-uses dao-ai's full :class:`dao_ai.config.AppConfig` schema rather than
maintaining a parallel one. The MCP server's YAML is a (subset of a) dao-ai
``AppConfig``: the same ``resources.*`` blocks the agent runtime consumes,
plus a ``tools.<name>`` block declaring which MCP tools to expose. We pass
``initialize=False`` so the server doesn't provision Genie spaces, register
MLflow experiments, or run any of the side effects ``initialize()`` has — the
MCP server only consumes configuration, it doesn't own the agent lifecycle.

The ``app:`` block is intentionally optional. dao-ai's ``AppModel`` validator
requires at least one agent, which is meaningless for an MCP-only deployment.
We let server-name and log-level fall back to env vars then defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from dao_ai.config import AppConfig

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_SERVER_NAME = "dao-ai-mcp"

SERVER_NAME_ENV = "DAO_AI_MCP_SERVER_NAME"
LOG_LEVEL_ENV = "DAO_AI_MCP_LOG_LEVEL"


def load_app_config(
    path: str | Path,
    *,
    params: dict[str, str] | None = None,
) -> AppConfig:
    """Load a dao-ai ``AppConfig`` from YAML, skipping side-effecting initialization."""
    resolved = Path(path).resolve()
    logger.debug("mcp.config.load.start", path=str(resolved))
    config: AppConfig = AppConfig.from_file(resolved, params=params, initialize=False)
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
