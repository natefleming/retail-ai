"""
Node creation utilities for DAO AI agents.

This module provides factory functions for creating LangGraph nodes
that implement agent logic using LangChain v1's create_agent pattern.
"""

from typing import Any, Optional, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langmem import create_manage_memory_tool
from loguru import logger

from dao_ai.config import (
    AgentModel,
    ChatHistoryModel,
    MemoryModel,
    PromptModel,
    ToolModel,
)
from dao_ai.middleware.core import create_factory_middleware
from dao_ai.middleware.guardrails import GuardrailMiddleware
from dao_ai.middleware.human_in_the_loop import create_hitl_middleware_from_tool_models
from dao_ai.middleware.summarization import create_summarization_middleware
from dao_ai.prompts import make_prompt
from dao_ai.state import AgentState, Context
from dao_ai.tools import create_tools
from dao_ai.tools.memory import create_search_memory_tool


def _create_middleware_list(
    agent: AgentModel,
    tool_models: Sequence[ToolModel],
    chat_history: Optional[ChatHistoryModel] = None,
) -> list[Any]:
    """
    Create a list of middleware instances from agent configuration.

    Args:
        agent: AgentModel configuration
        tool_models: Tool model configurations (for HITL settings)
        chat_history: Optional chat history configuration for summarization

    Returns:
        List of middleware instances (can include both AgentMiddleware and
        LangChain built-in middleware)
    """
    logger.debug(f"Building middleware list for agent '{agent.name}'")
    middleware_list: list[Any] = []

    # Add configured middleware using factory pattern
    if agent.middleware:
        logger.debug(f"Processing {len(agent.middleware)} configured middleware")
    for middleware_config in agent.middleware:
        middleware = create_factory_middleware(
            function_name=middleware_config.name,
            args=middleware_config.args,
        )
        if middleware is not None:
            middleware_list.append(middleware)

    # Add guardrails as middleware
    if agent.guardrails:
        logger.debug(f"Adding {len(agent.guardrails)} guardrail middleware")
    for guardrail in agent.guardrails:
        # Extract template string from PromptModel if needed
        prompt_str: str
        if isinstance(guardrail.prompt, PromptModel):
            prompt_str = guardrail.prompt.template
        else:
            prompt_str = guardrail.prompt

        guardrail_middleware = GuardrailMiddleware(
            name=guardrail.name,
            model=guardrail.model.as_chat_model(),
            prompt=prompt_str,
            num_retries=guardrail.num_retries or 3,
        )
        logger.debug(f"Created guardrail middleware: {guardrail.name}")
        middleware_list.append(guardrail_middleware)

    # Add summarization middleware if chat_history is configured
    if chat_history is not None:
        logger.debug("Adding summarization middleware")
        summarization_middleware = create_summarization_middleware(chat_history)
        middleware_list.append(summarization_middleware)

    # Add human-in-the-loop middleware if any tools require it
    hitl_middleware = create_hitl_middleware_from_tool_models(tool_models)
    if hitl_middleware is not None:
        logger.debug("Added human-in-the-loop middleware")
        middleware_list.append(hitl_middleware)

    logger.debug(f"Total middleware count: {len(middleware_list)}")
    return middleware_list


def create_agent_node(
    agent: AgentModel,
    memory: Optional[MemoryModel] = None,
    chat_history: Optional[ChatHistoryModel] = None,
    additional_tools: Optional[Sequence[BaseTool]] = None,
) -> RunnableLike:
    """
    Factory function that creates a LangGraph node for a specialized agent.

    This creates an agent using LangChain v1's create_agent function with
    middleware for customization. The function configures the agent with
    the appropriate model, prompt, tools, and middleware.

    Args:
        agent: AgentModel configuration for the agent
        memory: Optional MemoryModel for memory store configuration
        chat_history: Optional ChatHistoryModel for chat history summarization
        additional_tools: Optional sequence of additional tools to add to the agent

    Returns:
        RunnableLike: An agent node that processes state and returns responses
    """
    logger.debug(f"Creating agent node for {agent.name}")

    llm: LanguageModelLike = agent.model.as_chat_model()

    tool_models: Sequence[ToolModel] = agent.tools
    if not additional_tools:
        additional_tools = []
    tools: list[BaseTool] = list(create_tools(tool_models)) + list(additional_tools)

    if memory and memory.store:
        namespace: tuple[str, ...] = ("memory",)
        if memory.store.namespace:
            namespace = namespace + (memory.store.namespace,)
        logger.debug(f"Memory store namespace: {namespace}")

        # Use Databricks-compatible search_memory tool (omits problematic filter field)
        tools += [
            create_manage_memory_tool(namespace=namespace),
            create_search_memory_tool(namespace=namespace),
        ]

    # Create middleware list from configuration
    middleware_list = _create_middleware_list(
        agent=agent,
        tool_models=tool_models,
        chat_history=chat_history,
    )

    logger.debug(f"Created {len(middleware_list)} middleware for agent {agent.name}")

    checkpointer: bool = memory is not None and memory.checkpointer is not None

    # Get the prompt as middleware (always returns AgentMiddleware or None)
    prompt_middleware: AgentMiddleware | None = make_prompt(agent.prompt)

    # Add prompt middleware at the beginning for priority
    if prompt_middleware is not None:
        middleware_list.insert(0, prompt_middleware)

    # Use LangChain v1's create_agent with middleware
    # AgentState extends MessagesState with additional DAO AI fields
    # System prompt is provided via middleware (dynamic_prompt)
    compiled_agent: CompiledStateGraph = create_agent(
        name=agent.name,
        model=llm,
        tools=tools,
        middleware=middleware_list,
        checkpointer=checkpointer,
        state_schema=AgentState,
        context_schema=Context,
    )

    compiled_agent.name = agent.name

    return compiled_agent
