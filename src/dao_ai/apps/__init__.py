"""
Databricks Apps deployment module for dao-ai.

This subpackage contains all modules related to deploying dao-ai agents
as Databricks Apps or Model Serving endpoints.

Modules:
    handlers: MLflow AgentServer request handlers (@invoke, @stream)
    server: Entry point for Databricks Apps deployment
    resources: Databricks App resource configuration generation
    model_serving: Entry point for Databricks Model Serving deployment
"""

from dao_ai.apps.resources import (
    generate_app_resources,
    generate_app_yaml,
    generate_sdk_resources,
)

__all__ = [
    "generate_app_resources",
    "generate_app_yaml",
    "generate_sdk_resources",
]
