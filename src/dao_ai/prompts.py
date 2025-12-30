"""
Prompt utilities for DAO AI agents.

This module provides utilities for creating dynamic prompts using
LangChain v1's @dynamic_prompt middleware decorator pattern.
"""

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
        # Get parameters from runtime context
        params: dict[str, Any] = {
            input_variable: "" for input_variable in prompt_template.input_variables
        }

        # Access context from runtime
        context: Context = request.runtime.context
        if context:
            if context.user_id and "user_id" in params:
                params["user_id"] = context.user_id
            if context.thread_id and "thread_id" in params:
                params["thread_id"] = context.thread_id
            # Apply all context fields as template parameters
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


def create_prompt_middleware(
    base_system_prompt: Optional[str | PromptModel],
) -> AgentMiddleware | None:
    """
    Create a dynamic prompt middleware from configuration.

    This always returns an AgentMiddleware suitable for use with
    LangChain v1's middleware system.

    Args:
        base_system_prompt: The system prompt string or PromptModel

    Returns:
        An AgentMiddleware created by @dynamic_prompt, or None if no prompt
    """
    if not base_system_prompt:
        return None

    # Extract template string from PromptModel or use string directly
    template_str: str
    if isinstance(base_system_prompt, PromptModel):
        template_str = base_system_prompt.template
    else:
        template_str = base_system_prompt

    prompt_template: PromptTemplate = PromptTemplate.from_template(template_str)

    @dynamic_prompt
    def prompt_middleware(request: ModelRequest) -> str:
        """Generate system prompt based on runtime context."""
        # Get parameters from runtime context
        params: dict[str, Any] = {
            input_variable: "" for input_variable in prompt_template.input_variables
        }

        # Access context from runtime
        context: Context = request.runtime.context
        if context:
            # Apply all context fields as template parameters
            context_dict = context.model_dump()
            for key, value in context_dict.items():
                if key in params and value is not None:
                    params[key] = value

        # Format the prompt
        formatted_prompt: str = prompt_template.format(**params)
        logger.trace("Formatted dynamic prompt with context")

        return formatted_prompt

    return prompt_middleware
