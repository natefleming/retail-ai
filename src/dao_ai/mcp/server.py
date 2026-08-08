"""FastAPI + FastMCP entrypoint for the dao-ai MCP server.

Streamable HTTP transport with ``stateless_http=True`` and
``json_response=True`` so the App scales horizontally on Databricks Apps
without sticky sessions. The MCP surface exposes exactly one tool: the
whole dao-ai agent, wired to the graph produced by
:meth:`AppConfig.as_responses_agent`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from mcp.server.fastmcp import FastMCP

from dao_ai.config import AppConfig
from dao_ai.logging import configure_logging, suppress_autolog_context_warnings
from dao_ai.mcp import __version__
from dao_ai.mcp._request_context import RequestContextMiddleware
from dao_ai.mcp.agent_tool import register_agent_as_tool
from dao_ai.mcp.config import (
    LOG_LEVEL_ENV,
    SERVER_NAME_ENV,
    load_app_config,
    log_level_for,
    server_name_for,
)
from dao_ai.mcp.server_capabilities import (
    register_prompts,
    register_resources,
)

DEFAULT_CONFIG_PATH = "dao_ai.yaml"
DEFAULT_PORT = 8000

# Aligned with ``dao_ai.apps.handlers`` and ``dao_ai.apps.server`` so a
# single env var drives config discovery whether the runtime is the chat
# proxy, the responses server, or this MCP server. ``agent build --as-mcp`` also
# emits this name via the shared ``_build_app_block`` path.
CONFIG_PATH_ENV = "DAO_AI_CONFIG_PATH"


def _configure_mlflow(config: AppConfig) -> None:
    """Mirror ``dao_ai.apps.handlers`` mlflow setup for the MCP server.

    Sets registry + tracking URIs, binds the experiment (with optional UC
    ``trace_location``), and enables LangChain autolog. Errors on
    experiment binding are logged but do not abort startup — the deploy-time
    experiment linkage will still route traces to the right place.
    """
    mlflow.set_registry_uri("databricks-uc")
    try:
        mlflow.set_tracking_uri("databricks")
    except Exception:
        logger.warning("mcp.server.mlflow.tracking_uri.skip", exc_info=True)

    experiment_id: str | None = os.environ.get("MLFLOW_EXPERIMENT_ID")
    if experiment_id:
        set_experiment_kwargs: dict[str, Any] = {"experiment_id": experiment_id}
        if config.app and config.app.trace_location:
            from mlflow.entities import UnityCatalog

            trace_loc = config.app.trace_location
            trace_loc_kwargs: dict[str, Any] = {
                "catalog_name": trace_loc.catalog_name,
                "schema_name": trace_loc.schema_name,
            }
            table_prefix = trace_loc.resolved_table_prefix
            if table_prefix:
                trace_loc_kwargs["table_prefix"] = table_prefix
            set_experiment_kwargs["trace_location"] = UnityCatalog(**trace_loc_kwargs)
        try:
            mlflow.set_experiment(**set_experiment_kwargs)
        except Exception as exc:
            logger.warning(
                "mcp.server.mlflow.set_experiment.skip",
                error=str(exc),
            )

        # Populate the client-side trace-destination ContextVar so the
        # OTEL span exporter picks the prefixed UC table. Without this,
        # the env-var-based fallback picks the deprecated UCSchemaLocation
        # whose default table name is `mlflow_experiment_trace_otel_spans`
        # and every export fails with TABLE_DOES_NOT_EXIST.
        if config.app and config.app.trace_location:
            from dao_ai.providers.databricks import apply_runtime_trace_destination

            apply_runtime_trace_destination(config)

    mlflow.langchain.autolog(run_tracer_inline=True)
    suppress_autolog_context_warnings()


def build_app(config: AppConfig) -> FastAPI:
    """Build the FastAPI app with the MCP transport mounted at root."""
    server_name = server_name_for(config)

    # Progress notifications require a stateful HTTP session so the
    # notifications channel stays open during the tool call. When progress
    # is on, opt into the stateful transport; otherwise keep the stateless
    # default (needed for horizontal scaling on Databricks Apps without
    # sticky sessions).
    caps = getattr(config.app, "mcp_server", None) if config.app is not None else None
    needs_stateful: bool = caps is not None and caps.progress

    # Keep FastMCP's default ``streamable_http_path="/mcp"`` and mount the
    # inner app at the parent's root. This makes the external endpoint
    # ``/mcp`` (no trailing slash) match the inner Starlette route exactly,
    # so requests don't hit FastAPI's trailing-slash redirect machinery.
    # Databricks Agent Bricks Supervisor POSTs to ``<app-url>/mcp`` verbatim
    # and treats anything other than 200 as a registration failure.
    mcp = FastMCP(
        server_name,
        stateless_http=not needs_stateful,
        json_response=not needs_stateful,
    )
    logger.info(
        "mcp.server.transport",
        stateless_http=not needs_stateful,
        json_response=not needs_stateful,
        progress=bool(caps and caps.progress),
    )
    registered_tool = register_agent_as_tool(mcp, config)

    # PR 2 — server-side capabilities. When absent, the loops below are
    # no-ops and the server keeps its pre-PR-2 single-tool surface.
    registered_resources: list[str] = []
    registered_prompts: list[str] = []
    if caps is not None:
        registered_resources = register_resources(mcp, caps.resources)
        registered_prompts = register_prompts(mcp, caps.prompts)

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
                "tools": [registered_tool],
                "resources": registered_resources,
                "prompts": registered_prompts,
            }
        )

    @app.get("/readyz", include_in_schema=False)
    def readyz() -> JSONResponse:
        return JSONResponse({"ok": True, "version": __version__})

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
        app_name=config.app.name if config.app else None,
        server_name_env=SERVER_NAME_ENV,
        log_level_env=LOG_LEVEL_ENV,
    )

    _configure_mlflow(config)

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
