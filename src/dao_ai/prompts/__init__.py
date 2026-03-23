"""
Prompt utilities for DAO AI agents.

This module provides utilities for creating dynamic prompts using
LangChain v1's @dynamic_prompt middleware decorator pattern, as well as
paths to prompt template files.
"""

from pathlib import Path
from typing import Any, Optional

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    dynamic_prompt,
)
from langchain_core.prompts import PromptTemplate
from loguru import logger
from mlflow.genai.prompts import PromptVersion

from dao_ai.config import PromptModel
from dao_ai.state import Context

PROMPTS_DIR = Path(__file__).parent

# Module-level cache of resolved PromptVersion objects.
# Populated during graph construction (make_prompt calls) and consumed
# post-inference by LanggraphResponsesAgent for explicit trace linking.
_cached_prompt_versions: list[PromptVersion] = []


def get_cached_prompt_versions() -> list[PromptVersion]:
    """Return a snapshot of all cached PromptVersion objects."""
    return list(_cached_prompt_versions)


def get_prompt_path(name: str) -> Path:
    """Get the path to a prompt template file."""
    return PROMPTS_DIR / name


def make_prompt(
    base_system_prompt: Optional[str | PromptModel],
    prompt_model: Optional[PromptModel] = None,
) -> AgentMiddleware | None:
    """
    Create a dynamic prompt middleware from configuration.

    For LangChain v1's create_agent, this function always returns an
    AgentMiddleware instance for use with the middleware parameter.
    This provides a consistent interface regardless of whether the
    prompt template has variables or not.

    When ``prompt_model`` is supplied, the resolved ``PromptVersion`` is
    cached at init time in the module-level ``_cached_prompt_versions``
    list.  After inference, ``LanggraphResponsesAgent`` uses
    ``MlflowClient.link_prompt_versions_to_trace`` to batch-link all
    cached versions to the completed trace.  This avoids the
    ``ContextVar`` propagation issues that prevent ``load_prompt``
    auto-linking from working in model serving's async environment.

    Args:
        base_system_prompt: The system prompt string or PromptModel
        prompt_model: Optional original PromptModel reference used to
            link the prompt version to the active MLflow trace.

    Returns:
        An AgentMiddleware created by @dynamic_prompt, or None if no prompt
    """
    logger.trace("Creating prompt middleware", has_prompt=bool(base_system_prompt))

    if not base_system_prompt:
        return None

    # Extract template string from PromptModel or use string directly
    template: str
    if isinstance(base_system_prompt, PromptModel):
        template = base_system_prompt.template
    else:
        template = base_system_prompt

    # Create prompt template (handles both static and dynamic prompts)
    prompt_template: PromptTemplate = PromptTemplate.from_template(template)

    # Resolve and cache the PromptVersion for post-inference trace linking.
    if prompt_model is not None:
        try:
            from dao_ai.providers.databricks import DatabricksProvider

            resolved: PromptVersion = DatabricksProvider().get_prompt(prompt_model)
            _cached_prompt_versions.append(resolved)
            logger.trace(
                "Cached prompt version for trace linking",
                prompt_name=resolved.name,
                prompt_version=resolved.version,
            )
        except Exception:
            logger.trace(
                "Could not resolve prompt version for trace linking",
                prompt_name=prompt_model.full_name,
            )

    if prompt_template.input_variables:
        logger.trace(
            "Dynamic prompt with variables", variables=prompt_template.input_variables
        )
    else:
        logger.trace("Static prompt (no variables, using middleware for consistency)")

    @dynamic_prompt
    def dynamic_system_prompt(request: ModelRequest) -> str:
        """Generate dynamic system prompt based on runtime context."""
        # Initialize parameters for template variables
        params: dict[str, Any] = {
            input_variable: "" for input_variable in prompt_template.input_variables
        }

        # Apply context fields as template parameters
        context: Context = request.runtime.context
        if context:
            context_dict = context.model_dump()
            for key, value in context_dict.items():
                if key in params and value is not None:
                    params[key] = value

        # Format the prompt
        formatted_prompt: str = prompt_template.format(**params)
        logger.trace(
            "Formatted dynamic prompt with context",
            prompt_prefix=formatted_prompt[:200],
        )

        return formatted_prompt

    return dynamic_system_prompt
