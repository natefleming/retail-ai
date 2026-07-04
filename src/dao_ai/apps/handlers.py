"""
Agent request handlers for MLflow AgentServer.

This module defines the invoke and stream handlers that are registered
with the MLflow AgentServer. These handlers delegate to the ResponsesAgent
created from the dao-ai configuration.

The handlers use async methods (apredict, apredict_stream) to be compatible
with both Databricks Model Serving and Databricks Apps environments.
"""

import os
from typing import Any, AsyncGenerator

import mlflow
from dotenv import load_dotenv
from loguru import logger
from mlflow.genai.agent_server import get_request_headers, invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from dao_ai.config import AppConfig
from dao_ai.logging import configure_logging, suppress_autolog_context_warnings
from dao_ai.models import LanggraphResponsesAgent


def _inject_headers_into_request(request: ResponsesAgentRequest) -> None:
    """Inject request headers into custom_inputs for Context propagation.

    Captures headers from the MLflow AgentServer context (where they're available)
    and injects them into request.custom_inputs.configurable.headers so they
    flow through to Context and can be used for OBO authentication.
    """
    headers: dict[str, str] = get_request_headers()
    if headers:
        if request.custom_inputs is None:
            request.custom_inputs = {}
        if "configurable" not in request.custom_inputs:
            request.custom_inputs["configurable"] = {}
        request.custom_inputs["configurable"]["headers"] = headers


# Load environment variables from .env.local if it exists
load_dotenv(dotenv_path=".env.local", override=True)

# Configure MLflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Get config path from environment or use default
config_path: str = os.environ.get("DAO_AI_CONFIG_PATH", "dao_ai.yaml")

# Load configuration using AppConfig.from_file (consistent with CLI, notebook, builder)
config: AppConfig = AppConfig.from_file(config_path)

# Configure logging
if config.app and config.app.log_level:
    configure_logging(level=config.app.log_level)

# Set the active MLflow experiment BEFORE enabling autolog. If autolog is
# enabled first, its instrumentation captures the initial LangChain callbacks
# under the workspace-default experiment and creates a run there — then
# subsequent set_experiment() calls change the active experiment but the run
# is already stuck under the default. That produces
# ``Span for run_id ... not found`` at trace-write time because MLflow looks
# for the run in the current (correct) experiment but it lives in the default.
_experiment_id: str | None = os.environ.get("MLFLOW_EXPERIMENT_ID")
if _experiment_id:
    _set_experiment_kwargs: dict[str, Any] = {"experiment_id": _experiment_id}
    if config.app and config.app.trace_location:
        # Post-3.11 API: pass trace_location to set_experiment so the UC OTEL
        # tables become the export destination for spans on this experiment.
        # Replaces mlflow.tracing.set_destination + set_experiment_trace_location.
        from mlflow.entities import UnityCatalog

        _loc = config.app.trace_location
        _trace_loc_kwargs: dict[str, Any] = {
            "catalog_name": _loc.catalog_name,
            "schema_name": _loc.schema_name,
        }
        _table_prefix = _loc.resolved_table_prefix
        if _table_prefix:
            _trace_loc_kwargs["table_prefix"] = _table_prefix
        _set_experiment_kwargs["trace_location"] = UnityCatalog(**_trace_loc_kwargs)
    try:
        mlflow.set_experiment(**_set_experiment_kwargs)
    except Exception as _set_exp_err:
        # If the auto-created SP lacks USE CATALOG on the trace schema, the
        # trace_location bind will fail. The experiment linkage from deploy
        # time survives — traces land there via the UC OTEL tables that were
        # already linked. Log a warning; do not raise.
        logger.warning(
            "Could not set experiment at app startup "
            f"(experiment linkage from deploy time is still in effect): {_set_exp_err}"
        )

mlflow.langchain.autolog(run_tracer_inline=True)
suppress_autolog_context_warnings()

# Create the ResponsesAgent - cast to LanggraphResponsesAgent to access async methods
_responses_agent: LanggraphResponsesAgent = config.as_responses_agent()  # type: ignore[assignment]


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """
    Handle non-streaming requests by delegating to the ResponsesAgent.

    Uses the async apredict() method for compatibility with both
    Model Serving and Apps environments.

    Args:
        request: The incoming ResponsesAgentRequest

    Returns:
        ResponsesAgentResponse with the complete output
    """
    # Capture headers while in the AgentServer async context (before they're lost)
    _inject_headers_into_request(request)
    return await _responses_agent.apredict(request)


@stream()
async def streaming(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """
    Handle streaming requests by delegating to the ResponsesAgent.

    Uses the async apredict_stream() method for compatibility with both
    Model Serving and Apps environments.

    Args:
        request: The incoming ResponsesAgentRequest

    Yields:
        ResponsesAgentStreamEvent objects as they are generated
    """
    # Capture headers while in the AgentServer async context (before they're lost)
    _inject_headers_into_request(request)
    async for event in _responses_agent.apredict_stream(request):
        yield event
