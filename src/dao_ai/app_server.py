"""
App server module for running dao-ai agents as Databricks Apps.

This module provides the entry point for deploying dao-ai agents as Databricks Apps
using MLflow's AgentServer. It follows the same pattern as agent_as_code.py but
uses the AgentServer for the Databricks Apps runtime.

Configuration Loading:
    The config path is specified via the DAO_AI_CONFIG_PATH environment variable,
    or defaults to model_config.yaml in the current directory.

Usage:
    # With environment variable
    DAO_AI_CONFIG_PATH=/path/to/config.yaml python -m dao_ai.app_server

    # With default model_config.yaml in current directory
    python -m dao_ai.app_server
"""

import os
from typing import AsyncGenerator

import mlflow
from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer, invoke, stream
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from dao_ai.config import AppConfig
from dao_ai.logging import configure_logging

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

# Create the ResponsesAgent
_responses_agent: ResponsesAgent = config.as_responses_agent()


@invoke()
def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """
    Handle non-streaming requests by delegating to the ResponsesAgent.

    Args:
        request: The incoming ResponsesAgentRequest

    Returns:
        ResponsesAgentResponse with the complete output
    """
    return _responses_agent.predict(request)


@stream()
def streaming(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """
    Handle streaming requests by delegating to the ResponsesAgent.

    Args:
        request: The incoming ResponsesAgentRequest

    Yields:
        ResponsesAgentStreamEvent objects as they are generated
    """
    # The predict_stream method returns a generator, convert to async generator
    for event in _responses_agent.predict_stream(request):
        yield event


# Create the AgentServer instance
agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)

# Define the app as a module level variable to enable multiple workers
app = agent_server.app


def main() -> None:
    """Entry point for running the agent server."""
    agent_server.run(app_import_string="dao_ai.app_server:app")


if __name__ == "__main__":
    main()
