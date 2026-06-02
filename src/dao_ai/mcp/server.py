"""FastAPI + FastMCP entrypoint for the dao-ai MCP server.

Streamable HTTP transport with ``stateless_http=True`` and
``json_response=True`` so the App scales horizontally on Databricks Apps
without sticky sessions. dao-ai's blocking ``ask_question`` /
``send_feedback`` / VS retrieval are wrapped in ``asyncio.to_thread`` so the
FastAPI event loop stays unblocked under concurrent load.

Tools are discovered dynamically by walking ``AppConfig.tools`` and looking up
each ``function.name`` against the adapter registry.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import mlflow
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from mcp.server.fastmcp import FastMCP

from dao_ai.config import AppConfig
from dao_ai.logging import configure_logging
from dao_ai.mcp import __version__
from dao_ai.mcp._request_context import RequestContextMiddleware
from dao_ai.mcp.config import (
    LOG_LEVEL_ENV,
    SERVER_NAME_ENV,
    load_app_config,
    log_level_for,
    server_name_for,
)
from dao_ai.mcp.service import register_tools_from_config

DEFAULT_CONFIG_PATH = "dao_ai.yaml"
DEFAULT_PORT = 8000

CONFIG_PATH_ENV = "DAO_AI_MCP_CONFIG_PATH"


def build_app(config: AppConfig) -> FastAPI:
    """Build the FastAPI app with the MCP transport mounted at root.

    Kept separate from :func:`main` for testability — callers can construct an
    in-process ASGI app from a synthetic config.
    """
    server_name = server_name_for(config)

    # Keep FastMCP's default ``streamable_http_path="/mcp"`` and mount the
    # inner app at the parent's root. This makes the external endpoint
    # ``/mcp`` (no trailing slash) match the inner Starlette route exactly,
    # so requests don't hit FastAPI's trailing-slash redirect machinery —
    # which would otherwise emit a 307 to ``http://localhost:8000/mcp/``
    # (the internal host name), breaking clients that don't follow redirects.
    # Databricks Agent Bricks Supervisor is one such client: it POSTs to
    # ``<app-url>/mcp`` verbatim and treats anything other than 200 as a
    # registration failure.
    mcp = FastMCP(
        server_name,
        stateless_http=True,
        json_response=True,
    )
    registered_names = register_tools_from_config(mcp, config)

    # The streamable-HTTP transport's session manager needs to be started in a
    # task group BEFORE the first request arrives. We chain it into FastAPI's
    # lifespan so uvicorn drives it during startup/shutdown. Without this every
    # request 500s with "Task group is not initialized".
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with mcp.session_manager.run():
            logger.info("mcp.session_manager.started")
            yield
            logger.info("mcp.session_manager.stopped")

    app = FastAPI(title=server_name, version=__version__, lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "version": __version__,
                "server_name": server_name,
                "tools": sorted(registered_names),
            }
        )

    @app.get("/readyz", include_in_schema=False)
    def readyz() -> JSONResponse:
        return JSONResponse({"ok": True, "version": __version__})

    # Mount FastMCP's ASGI app at root. /healthz + /readyz remain accessible
    # because FastAPI matches @app.get-registered routes before the mount in
    # the routes list, so they win over the catch-all mount.
    app.mount("/", mcp.streamable_http_app())
    return app


def main() -> None:
    """``uv``-invoked entrypoint declared in pyproject.toml ``[project.scripts]``."""
    config_path = os.environ.get(CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH)
    config = load_app_config(config_path)
    configure_logging(level=log_level_for(config))

    logger.info(
        "mcp.server.boot",
        version=__version__,
        config_path=config_path,
        server_name=server_name_for(config),
        tool_count=len(config.tools),
        server_name_env=SERVER_NAME_ENV,
        log_level_env=LOG_LEVEL_ENV,
    )

    try:
        mlflow.set_tracking_uri("databricks")
    except Exception:
        logger.warning("mcp.server.mlflow.tracking_uri.skip", exc_info=True)

    port = int(os.environ.get("DATABRICKS_APP_PORT", DEFAULT_PORT))
    host = os.environ.get("DATABRICKS_APP_HOST", "0.0.0.0")

    app = build_app(config)
    logger.info("mcp.server.listen", host=host, port=port, transport="streamable_http")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("mcp.server.fatal")
        sys.exit(1)
