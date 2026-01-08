"""
App server module for running dao-ai agents as Databricks Apps.

This module provides the entry point for deploying dao-ai agents as Databricks Apps
using MLflow's AgentServer. It follows the same pattern as model_serving.py but
uses the AgentServer for the Databricks Apps runtime.

Configuration Loading:
    The config path is specified via the DAO_AI_CONFIG_PATH environment variable,
    or defaults to dao_ai.yaml in the current directory.

Usage:
    # With environment variable
    DAO_AI_CONFIG_PATH=/path/to/config.yaml python -m dao_ai.apps.server

    # With default dao_ai.yaml in current directory
    python -m dao_ai.apps.server
"""

from mlflow.genai.agent_server import AgentServer

# Import the agent handlers to register the invoke and stream decorators
# This MUST happen before creating the AgentServer instance
import dao_ai.apps.handlers  # noqa: E402, F401

# Create the AgentServer instance
agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)

# Define the app as a module level variable to enable multiple workers
app = agent_server.app


def main() -> None:
    """Entry point for running the agent server."""
    agent_server.run(app_import_string="dao_ai.apps.server:app")


if __name__ == "__main__":
    main()
