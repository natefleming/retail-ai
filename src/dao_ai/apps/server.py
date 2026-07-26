"""App server module for running dao-ai agents as Databricks Apps.

This module provides the entry point for deploying dao-ai agents as Databricks Apps
using MLflow's AgentServer. It follows the same pattern as model_serving.py but
uses the AgentServer for the Databricks Apps runtime.

When ``app.background`` is configured, strict OpenAI Responses API routes
(``/v1/responses``, ``/v1/responses/{id}``, ``/v1/responses/{id}/cancel``) are
mounted on the underlying FastAPI app so clients can use standard Responses
clients without having to encode operations into ``custom_inputs`` by hand.

Configuration Loading:
    The config path is specified via the DAO_AI_CONFIG_PATH environment variable,
    or defaults to dao_ai.yaml in the current directory.

Usage:
    # With environment variable
    DAO_AI_CONFIG_PATH=/path/to/config.yaml python -m dao_ai.apps.server

    # With default dao_ai.yaml in current directory
    python -m dao_ai.apps.server
"""

from typing import Any, AsyncGenerator, Optional

from mlflow.genai.agent_server import AgentServer

# Import the agent handlers to register the invoke and stream decorators
# This MUST happen before creating the AgentServer instance
import dao_ai.apps.handlers  # noqa: E402, F401
from dao_ai.apps.handlers import config as _config

# Create the AgentServer instance
_enable_chat_proxy = _config.app.enable_chat_proxy if _config.app else True
agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=_enable_chat_proxy)

# Define the app as a module level variable to enable multiple workers
app = agent_server.app

# TEMPORARY: memory-footprint probe (no-op unless DAO_AI_MEMPROBE is set). Runs
# in every process — master + each worker — so summed PSS reveals COW sharing.
from dao_ai.apps import _memprobe  # noqa: E402

_memprobe.start()


def _mount_background_routes() -> None:
    """Register /v1/responses* routes when background is configured.

    These routes are sugar over ``/invocations``: they build a
    ``ResponsesAgentRequest`` whose ``custom_inputs`` carry the kickoff /
    retrieve / cancel markers, then delegate to the registered handlers.
    Model Serving clients, which only see ``/invocations``, can use the
    same ``custom_inputs`` shape directly.
    """
    if not _config.app or _config.app.background is None:
        return

    from fastapi import HTTPException, Query, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from loguru import logger
    from mlflow.types.responses import ResponsesAgentRequest

    from dao_ai.apps.handlers import non_streaming, streaming
    from dao_ai.background import (
        CUSTOM_INPUT_CURSOR,
        CUSTOM_INPUT_OPERATION,
        CUSTOM_INPUT_RESPONSE_ID,
        OPERATION_CANCEL,
        OPERATION_RETRIEVE,
        is_not_found_response,
    )

    def _build_request(
        body: dict[str, Any], *, background: Optional[bool] = None
    ) -> ResponsesAgentRequest:
        payload = dict(body)
        if background is not None:
            payload["background"] = background
        # Pydantic will validate and populate defaults.
        return ResponsesAgentRequest(**payload)

    def _inject_custom_inputs(
        request: ResponsesAgentRequest, **updates: Any
    ) -> ResponsesAgentRequest:
        custom = dict(request.custom_inputs or {})
        for key, value in updates.items():
            if value is not None:
                custom[key] = value
        request.custom_inputs = custom
        return request

    async def _sse_from_events(
        agen: AsyncGenerator[Any, None],
    ) -> AsyncGenerator[bytes, None]:
        try:
            async for event in agen:
                if hasattr(event, "model_dump_json"):
                    payload = event.model_dump_json()
                else:
                    payload = str(event)
                yield f"data: {payload}\n\n".encode()
        finally:
            try:
                await agen.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing SSE generator", error=str(exc))

    @app.post("/v1/responses")
    async def create_response(raw: Request):
        body = await raw.json()
        wants_stream = bool(body.pop("stream", False))
        # background defaults to True on this route — it's the whole point.
        request = _build_request(body, background=body.pop("background", True))

        if wants_stream:
            agen = streaming(request)
            return StreamingResponse(
                _sse_from_events(agen), media_type="text/event-stream"
            )

        response = await non_streaming(request)
        return JSONResponse(response.model_dump(mode="json"))

    @app.get("/v1/responses/{response_id}")
    async def retrieve_response(
        response_id: str,
        stream: bool = Query(default=False),
        cursor: int = Query(default=0, ge=0),
    ):
        request = ResponsesAgentRequest(input=[])
        _inject_custom_inputs(
            request,
            **{
                CUSTOM_INPUT_OPERATION: OPERATION_RETRIEVE,
                CUSTOM_INPUT_RESPONSE_ID: response_id,
                CUSTOM_INPUT_CURSOR: cursor,
            },
        )

        if stream:
            agen = streaming(request)
            return StreamingResponse(
                _sse_from_events(agen), media_type="text/event-stream"
            )

        response = await non_streaming(request)
        body = response.model_dump(mode="json")
        if is_not_found_response(body):
            raise HTTPException(status_code=404, detail="Response not found")
        return JSONResponse(body)

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel_response(response_id: str):
        request = ResponsesAgentRequest(input=[])
        _inject_custom_inputs(
            request,
            **{
                CUSTOM_INPUT_OPERATION: OPERATION_CANCEL,
                CUSTOM_INPUT_RESPONSE_ID: response_id,
            },
        )
        response = await non_streaming(request)
        body = response.model_dump(mode="json")
        if is_not_found_response(body):
            raise HTTPException(status_code=404, detail="Response not found")
        return JSONResponse(body)

    logger.info(
        "Background Responses API routes mounted",
        routes=[
            "POST /v1/responses",
            "GET /v1/responses/{id}",
            "POST /v1/responses/{id}/cancel",
        ],
    )


_mount_background_routes()


def _mount_a2a_routes() -> None:
    """Register A2A protocol routes alongside the OpenAI Responses contract.

    No-op when ``app.a2a.enabled=false``. The mount adds:

    * ``GET  /.well-known/agent-card.json``
    * ``POST /a2a``

    See :mod:`dao_ai.apps.a2a` for the executor, task store, and Agent Card
    machinery.
    """
    try:
        # Imported inside the try so a missing 'a2a' extra degrades gracefully
        # to the Responses-only contract instead of crashing the server at load.
        from dao_ai.apps.a2a import mount_a2a_routes

        mount_a2a_routes(app, _config)
    except Exception as exc:  # pragma: no cover — defensive at startup
        from loguru import logger

        logger.warning(
            f"Failed to mount A2A routes; Responses contract still served. Error: {exc}"
        )


_mount_a2a_routes()


def _parse_server_args() -> tuple[int, int]:
    """Parse ``--port`` / ``--workers`` (matching MLflow AgentServer's flags)."""
    import argparse

    parser = argparse.ArgumentParser(description="Start the dao-ai agent server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true")
    args, _ = parser.parse_known_args()
    return args.port, args.workers


def _run_gunicorn_preload(port: int, workers: int) -> None:
    """Serve with gunicorn + UvicornWorker + ``preload_app=True``.

    Preload imports ``dao_ai.apps.server`` (building the agent graph) ONCE in
    the gunicorn arbiter, then forks the workers — so the graph's read-only
    pages are copy-on-write shared instead of rebuilt per worker (uvicorn's own
    multi-worker path spawns fresh interpreters with no sharing, which OOMs on
    small Apps compute). Fork-safety: the Lakebase pool and langmem reflection
    thread are already lazy (bound on first request, post-fork), so no live DB
    connection or background thread exists at fork time.
    """
    from gunicorn.app.base import BaseApplication
    from loguru import logger

    def _post_fork(_server: Any, worker: Any) -> None:  # pragma: no cover — runtime hook
        # Restart the (daemon, lock-free) mem probe in each forked worker so we
        # capture per-worker PSS; safe to call — no-op unless DAO_AI_MEMPROBE set.
        _memprobe.start()
        logger.info("gunicorn worker forked | pid={} worker={}", os.getpid(), worker.pid)

    class _DaoAiGunicornApp(BaseApplication):
        def load_config(self) -> None:
            self.cfg.set("bind", f"0.0.0.0:{port}")
            self.cfg.set("workers", workers)
            self.cfg.set("worker_class", "uvicorn.workers.UvicornWorker")
            self.cfg.set("preload_app", True)
            self.cfg.set("post_fork", _post_fork)

        def load(self) -> Any:
            # Return the already-imported, already-built ASGI app. Under
            # preload this runs in the arbiter, so the graph is built once here.
            return app

    logger.info(
        "Starting gunicorn (preload) | workers={} port={} worker_class=UvicornWorker",
        workers,
        port,
    )
    _DaoAiGunicornApp().run()


def main() -> None:
    """Entry point for running the agent server.

    Single worker → MLflow's uvicorn launcher (unchanged). Multiple workers →
    gunicorn with ``preload_app`` so the agent graph is built once and shared
    copy-on-write across forked workers rather than rebuilt (and OOM-duplicated)
    in each spawned uvicorn worker.
    """
    port, workers = _parse_server_args()
    if workers > 1:
        _run_gunicorn_preload(port, workers)
    else:
        agent_server.run(app_import_string="dao_ai.apps.server:app")


if __name__ == "__main__":
    main()
