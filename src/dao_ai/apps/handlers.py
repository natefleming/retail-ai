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
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from dao_ai.config import AppConfig
from dao_ai.logging import configure_logging
from dao_ai.models import LanggraphResponsesAgent

# Load environment variables from .env.local if it exists
load_dotenv(dotenv_path=".env.local", override=True)

# Configure MLflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
mlflow.langchain.autolog()

# Get config path from environment or use default
config_path: str = os.environ.get("DAO_AI_CONFIG_PATH", "model_config.yaml")

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
    async for event in _responses_agent.apredict_stream(request):
        yield event
