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

import os
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
        starting_after: int | None = Query(default=None, ge=0),
    ):
        # The OpenAI Responses API uses `starting_after` as the stream cursor;
        # dao-ai originally exposed it as `cursor`. Accept both so the stock
        # OpenAI client (`responses.retrieve(id, stream=True, starting_after=N)`)
        # resumes correctly, preferring the OpenAI-spec name when supplied.
        effective_cursor: int = starting_after if starting_after is not None else cursor
        request = ResponsesAgentRequest(input=[])
        _inject_custom_inputs(
            request,
            **{
                CUSTOM_INPUT_OPERATION: OPERATION_RETRIEVE,
                CUSTOM_INPUT_RESPONSE_ID: response_id,
                CUSTOM_INPUT_CURSOR: effective_cursor,
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


def _mount_trace_routes() -> None:
    """Register ``GET /v1/traces?trace_id=...`` for the Console Timeline view.

    Returns the MLflow trace as a nested span waterfall (durations, per-span
    I/O, events). The backend owns retrieval — it already sets the tracking
    URI and redacts bearer tokens at write time — so the browser never needs
    a raw trace API or an OBO token. The trace id is a query param (not a path
    segment) because ``trace_location`` ids are UC URIs containing slashes
    (``trace:/<catalog>.<schema>.<prefix>/<id>``) that a path param can't
    capture. Sync route so FastAPI runs the short blocking propagation poll in
    its threadpool. Always mounted (independent of ``app.background``).
    """
    from fastapi import HTTPException, Query
    from fastapi.responses import JSONResponse
    from loguru import logger

    from dao_ai.apps.traces import build_trace_ui_url, get_trace_tree

    @app.get("/v1/trace-url")
    def get_trace_url(trace_id: str = Query(...)):
        # Deep link to the trace in the Databricks workspace UI. Works even when
        # the Apps runtime can't read the trace store (the browser can reach the
        # workspace UI), so the Console can offer it alongside an empty Timeline.
        return JSONResponse({"url": build_trace_ui_url(trace_id)})

    @app.get("/v1/traces")
    def get_trace(trace_id: str = Query(...)):
        # trace_location (UC OTEL) traces can take longer to become queryable
        # than the default local-tracking window, so allow more propagation time.
        # Any retrieval failure degrades to 404 (never 500) so the Console shows
        # its empty-Timeline note instead of an error.
        try:
            tree = get_trace_tree(trace_id, timeout_seconds=12.0)
        except Exception as exc:  # noqa: BLE001 — defense in depth
            logger.warning("Trace retrieval failed", trace_id=trace_id, error=str(exc))
            tree = None
        if tree is None:
            raise HTTPException(
                status_code=404, detail="Trace not found or not yet queryable"
            )
        return JSONResponse(tree)

    logger.info("Trace route mounted", routes=["GET /v1/traces?trace_id="])


_mount_trace_routes()


def _mount_sessions_routes() -> None:
    """Register the Console session routes on the same FastAPI app as the
    agent's ``/invocations`` and ``/v1/*`` routes.

    - ``GET  /v1/sessions/{thread_id}``      reload a conversation (checkpointer)
    - ``GET  /v1/sessions/{thread_id}/meta`` checkpoint metadata (last-modified, ids)
    - ``GET  /v1/sessions``                  list the calling user's threads
    - ``POST /v1/sessions``                  register/refresh a thread in the index

    The reload/meta routes are served through the checkpointer implementation
    object (``graph.aget_state``); the list/register routes keep a user→thread
    index in the configured ``BaseStore`` (``graph.store``), addressed through
    the store's own ``aput``/``asearch`` API — no native queries — so the sidebar
    works identically across any store backend. Each group mounts only when its
    backing implementation exists; otherwise the Console degrades (no reload,
    localStorage-only sidebar). ``user_id`` is always derived server-side from
    the OBO header, so a user only sees their own sessions.
    """
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse
    from loguru import logger

    from dao_ai.apps.handlers import _responses_agent
    from dao_ai.apps.sessions import (
        list_user_sessions,
        load_session,
        load_session_meta,
        register_session,
        user_id_from_headers,
    )

    graph = getattr(_responses_agent, "graph", None)
    has_checkpointer = graph is not None and getattr(graph, "checkpointer", None)

    if has_checkpointer:

        @app.get("/v1/sessions/{thread_id}")
        async def get_session(thread_id: str):
            try:
                return JSONResponse(await load_session(graph, thread_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load session", thread_id=thread_id, error=str(exc)
                )
                raise HTTPException(status_code=404, detail="Session not found")

        @app.get("/v1/sessions/{thread_id}/meta")
        async def get_session_meta(thread_id: str):
            try:
                return JSONResponse(await load_session_meta(graph, thread_id))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load session meta",
                    thread_id=thread_id,
                    error=str(exc),
                )
                raise HTTPException(status_code=404, detail="Session not found")

    store = getattr(graph, "store", None) if graph is not None else None
    if store is not None:

        @app.get("/v1/sessions")
        async def list_sessions(request: Request, limit: int = 50):
            user_id = user_id_from_headers(request.headers)
            if not user_id:
                return JSONResponse([])
            try:
                return JSONResponse(
                    await list_user_sessions(store, user_id, limit=limit)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to list sessions", error=str(exc))
                return JSONResponse([])

        @app.post("/v1/sessions")
        async def register_session_route(request: Request):
            user_id = user_id_from_headers(request.headers)
            body = await request.json()
            thread_id = body.get("thread_id")
            if not user_id or not thread_id:
                return JSONResponse({"ok": False})
            try:
                await register_session(
                    store, user_id, thread_id, body.get("title")
                )
                return JSONResponse({"ok": True})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to register session", error=str(exc))
                return JSONResponse({"ok": False})

    logger.info(
        "Session routes mounted",
        reload=bool(has_checkpointer),
        index=store is not None,
    )


_mount_sessions_routes()


def _mount_memory_routes() -> None:
    """Register ``GET /v1/memory`` for the Console memory viewer.

    Returns the calling user's long-term memory (profile / preferences /
    episodes) from the configured store (``graph.store``). ``user_id`` is
    derived server-side from the OBO header so a viewer only sees their own
    memory. When no store is configured the route returns ``{memory: null}``
    (200) so the Console simply hides the Memory panel — config-agnostic.
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from loguru import logger

    from dao_ai.apps.handlers import _responses_agent
    from dao_ai.apps.memory import safe_load_user_memory
    from dao_ai.apps.sessions import user_id_from_headers

    graph = getattr(_responses_agent, "graph", None)
    store = getattr(graph, "store", None) if graph else None
    if store is None:
        logger.info("Memory route not mounted (no store configured)")
        return

    @app.get("/v1/memory")
    async def get_memory(request: Request):
        user_id = user_id_from_headers(request.headers)
        if not user_id:
            return JSONResponse({"memory": None})
        return JSONResponse(await safe_load_user_memory(store, user_id))

    logger.info("Memory route mounted", routes=["GET /v1/memory"])


_mount_memory_routes()


def _mount_custom_inputs_route() -> None:
    """Register ``GET /v1/custom-inputs`` advertising the configurable fields
    the agent's config requires, so the Console can prepopulate its editor.

    Read-only and non-secret (names / descriptions / example values only).
    Returns ``{fields: []}`` when nothing is discoverable — config-agnostic.
    """
    from fastapi.responses import JSONResponse
    from loguru import logger

    from dao_ai.apps.custom_inputs import discover_custom_input_fields

    @app.get("/v1/custom-inputs")
    async def custom_inputs_schema():
        return JSONResponse({"fields": discover_custom_input_fields(_config)})

    logger.info("Custom-inputs route mounted", routes=["GET /v1/custom-inputs"])


_mount_custom_inputs_route()


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
    # default=0 sentinel → "unset" → auto-size to the container's CPUs in main().
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--reload", action="store_true")
    args, _ = parser.parse_known_args()
    return args.port, args.workers


# Ceiling for auto-derived worker count. The agent is async/I/O-bound, so
# throughput past a handful of workers has diminishing returns (each still
# handles many concurrent requests on its event loop); the gunicorn preload
# path keeps the per-worker memory cheap via copy-on-write. 8 leaves headroom
# for larger Apps compute while bounding fork overhead. (Validated live at 4
# workers on MEDIUM; higher counts are within the same COW-preload mechanism.)
_MAX_AUTO_WORKERS = 8


def _auto_workers() -> int:
    """Worker count sized to the container's available CPUs (min 1, capped).

    Uses ``sched_getaffinity`` (the cgroup/affinity-limited CPU set the
    container can actually run on) rather than ``os.cpu_count()`` — the latter
    reports the HOST's CPUs, which on a small Apps container is far larger than
    the allotment and would oversubscribe (the exact over-worker condition that
    crash-looped). Falls back to ``os.cpu_count()`` then 1 where affinity is
    unavailable.
    """
    try:
        cpus = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    return max(1, min(cpus, _MAX_AUTO_WORKERS))


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

    def _post_fork(
        _server: Any, worker: Any
    ) -> None:  # pragma: no cover — runtime hook
        logger.info(
            "gunicorn worker forked | pid={} worker={}", os.getpid(), worker.pid
        )

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
    from loguru import logger

    port, workers = _parse_server_args()
    # workers==0 → unset (app.workers not configured): auto-size to the
    # container's CPUs. An explicit --workers N (from app.workers) is honored.
    if workers <= 0:
        workers = _auto_workers()
        logger.info("Auto-sized backend workers to {} (container CPUs)", workers)
    if workers > 1:
        _run_gunicorn_preload(port, workers)
    else:
        agent_server.run(app_import_string="dao_ai.apps.server:app")


if __name__ == "__main__":
    main()
