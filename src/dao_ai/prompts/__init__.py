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

from dao_ai.config import PromptModel
from dao_ai.state import Context

PROMPTS_DIR = Path(__file__).parent


def get_prompt_path(name: str) -> Path:
    """Get the path to a prompt template file."""
    return PROMPTS_DIR / name


def make_prompt(
    base_system_prompt: Optional[str | PromptModel],
) -> AgentMiddleware | None:
    """
    Create a dynamic prompt middleware from configuration.

    For LangChain v1's create_agent, this function always returns an
    AgentMiddleware instance for use with the middleware parameter.
    This provides a consistent interface regardless of whether the
    prompt template has variables or not.

    Args:
        base_system_prompt: The system prompt string or PromptModel

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
