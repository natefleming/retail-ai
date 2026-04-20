"""App server module for running dao-ai agents as Databricks Apps.

This module provides the entry point for deploying dao-ai agents as Databricks Apps
using MLflow's AgentServer. It follows the same pattern as model_serving.py but
uses the AgentServer for the Databricks Apps runtime.

When ``app.long_running`` is configured, strict OpenAI Responses API routes
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


def _mount_long_running_routes() -> None:
    """Register /v1/responses* routes when long_running is configured.

    These routes are sugar over ``/invocations``: they build a
    ``ResponsesAgentRequest`` whose ``custom_inputs`` carry the kickoff /
    retrieve / cancel markers, then delegate to the registered handlers.
    Model Serving clients, which only see ``/invocations``, can use the
    same ``custom_inputs`` shape directly.
    """
    if not _config.app or _config.app.long_running is None:
        return

    from fastapi import HTTPException, Query, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from loguru import logger
    from mlflow.types.responses import ResponsesAgentRequest

    from dao_ai.apps.handlers import non_streaming, streaming
    from dao_ai.long_running import (
        CUSTOM_INPUT_CURSOR,
        CUSTOM_INPUT_OPERATION,
        CUSTOM_INPUT_RESPONSE_ID,
        OPERATION_CANCEL,
        OPERATION_RETRIEVE,
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

        try:
            response = await non_streaming(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return JSONResponse(response.model_dump(mode="json"))

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
        try:
            response = await non_streaming(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return JSONResponse(response.model_dump(mode="json"))

    logger.info(
        "Long-running Responses API routes mounted",
        routes=[
            "POST /v1/responses",
            "GET /v1/responses/{id}",
            "POST /v1/responses/{id}/cancel",
        ],
    )


_mount_long_running_routes()


def main() -> None:
    """Entry point for running the agent server."""
    agent_server.run(app_import_string="dao_ai.apps.server:app")


if __name__ == "__main__":
    main()
