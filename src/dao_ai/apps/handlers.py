"""
Agent request handlers for MLflow AgentServer.

This module defines the invoke and stream handlers that are registered
with the MLflow AgentServer. These handlers delegate to the ResponsesAgent
created from the dao-ai configuration.

The handlers use async methods (apredict, apredict_stream) to be compatible
with both Databricks Model Serving and Databricks Apps environments.
"""

import os
from typing import AsyncGenerator

import mlflow
from dotenv import load_dotenv
from loguru import logger
from mlflow.genai.agent_server import get_request_headers, invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from dao_ai._tracing import install_trace_redaction
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
        # Log header keys only (never values — one carries the OBO token) so the
        # available OBO identity headers can be confirmed without leaking secrets.
        logger.debug("Injected request headers", header_keys=sorted(headers.keys()))


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
    # Always set the active experiment id first — this is the load-bearing
    # call that keeps autolog-generated runs under the right experiment.
    # It never fails on "already contains traces" because it's not a link
    # operation.
    mlflow.set_experiment(experiment_id=_experiment_id)

    # If a trace_location is configured, try to link it via the idempotent
    # helper. The helper reads the experiment's current UC-destination tags
    # and skips the API call when the linkage already matches — which is
    # the common re-deploy case. When the linkage is genuinely missing and
    # the experiment already has traces (the broken state), the underlying
    # MLflow RestException surfaces here and we log loudly so the operator
    # sees the mismatch instead of silently dropping every future trace to
    # a non-existent fallback table.
    if config.app and config.app.trace_location:
        try:
            from dao_ai.providers.databricks import (
                apply_runtime_trace_destination,
                link_experiment_trace_location,
            )

            # ``mlflow.set_experiment(experiment_id, trace_location=UC(...))``
            # internally calls ``_sync_trace_destination_and_provider`` which
            # populates the client-side ``_MLFLOW_TRACE_USER_DESTINATION``
            # ContextVar. On re-deploys where the link is already in place,
            # ``link_experiment_trace_location`` short-circuits — MLflow's
            # own fallback resolver (``_resolve_experiment_uc_location``,
            # provider.py:632) then reads the experiment's linked
            # ``UnityCatalog`` from the tracking store on the first span
            # export. No explicit ContextVar manipulation needed.
            link_experiment_trace_location(config, _experiment_id)
            # Also set the client-side ContextVar so the OTEL span exporter
            # picks the prefixed UC table. Without this, MLflow's env-var
            # parser falls back to the deprecated UCSchemaLocation whose
            # default table name is `mlflow_experiment_trace_otel_spans`,
            # and every export fails with TABLE_DOES_NOT_EXIST.
            apply_runtime_trace_destination(config)
        except Exception as _link_err:  # noqa: BLE001
            logger.warning(
                "dao_ai.trace_location.link_failed "
                "experiment_id={} err={}: {} — "
                "traces from this app may not export to the configured UC "
                "schema. Re-deploy with a fresh experiment (delete the "
                "existing one, or set app.experiment.name to a new value) "
                "to recover.",
                _experiment_id,
                type(_link_err).__name__,
                _link_err,
            )

mlflow.langchain.autolog(run_tracer_inline=True)
suppress_autolog_context_warnings()

# Must follow autolog: this registers the span processor that strips the caller's
# forwarded bearer out of span payloads. `_inject_headers_into_request` below puts
# that bearer on the request precisely so tools can use it, and the traced
# `apredict` call would otherwise serialize it into the span's inputs.
install_trace_redaction()

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
