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
from mlflow.genai.agent_server import get_request_headers, invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from dao_ai.config import AppConfig
from dao_ai.logging import configure_logging
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
mlflow.langchain.autolog()

# Get config path from environment or use default
config_path: str = os.environ.get("DAO_AI_CONFIG_PATH", "dao_ai.yaml")

# Load configuration using AppConfig.from_file (consistent with CLI, notebook, builder)
config: AppConfig = AppConfig.from_file(config_path)

# Configure logging
if config.app and config.app.log_level:
    configure_logging(level=config.app.log_level)

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
