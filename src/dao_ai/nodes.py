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
from dao_ai.middleware.human_in_the_loop import (
    create_hitl_middleware_from_tool_models,
)
from dao_ai.middleware.summarization import (
    create_summarization_middleware,
)
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
    logger.debug("Building middleware list for agent", agent=agent.name)
    middleware_list: list[Any] = []

    # Add configured middleware using factory pattern
    if agent.middleware:
        middleware_names: list[str] = [mw.name for mw in agent.middleware]
        logger.info(
            "Middleware configuration",
            agent=agent.name,
            middleware_count=len(agent.middleware),
            middleware_names=middleware_names,
        )
    for middleware_config in agent.middleware:
        logger.trace(
            "Creating middleware for agent",
            agent=agent.name,
            middleware_name=middleware_config.name,
        )
        middleware: AgentMiddleware[AgentState, Context] = create_factory_middleware(
            function_name=middleware_config.name,
            args=middleware_config.args,
        )
        middleware_list.append(middleware)

    # Add guardrails as middleware
    if agent.guardrails:
        guardrail_names: list[str] = [gr.name for gr in agent.guardrails]
        logger.info(
            "Guardrails configuration",
            agent=agent.name,
            guardrails_count=len(agent.guardrails),
            guardrail_names=guardrail_names,
        )
    for guardrail in agent.guardrails:
        # Extract template string from PromptModel if needed
        prompt_str: str
        if isinstance(guardrail.prompt, PromptModel):
            prompt_str = guardrail.prompt.template
        else:
            prompt_str = guardrail.prompt

        guardrail_middleware: GuardrailMiddleware = GuardrailMiddleware(
            name=guardrail.name,
            model=guardrail.model.as_chat_model(),
            prompt=prompt_str,
            num_retries=guardrail.num_retries or 3,
        )
        logger.trace(
            "Created guardrail middleware", guardrail=guardrail.name, agent=agent.name
        )
        middleware_list.append(guardrail_middleware)

    # Add summarization middleware if chat_history is configured
    if chat_history is not None:
        logger.info(
            "Chat history configuration",
            agent=agent.name,
            max_tokens=chat_history.max_tokens,
            summary_model=chat_history.model.name,
        )
        summarization_middleware = create_summarization_middleware(chat_history)
        middleware_list.append(summarization_middleware)

    # Add human-in-the-loop middleware if any tools require it
    hitl_middlewares = create_hitl_middleware_from_tool_models(tool_models)
    if hitl_middlewares:
        # Log which tools require HITL
        hitl_tool_names: list[str] = [
            tool.name
            for tool in tool_models
            if hasattr(tool.function, "human_in_the_loop")
            and tool.function.human_in_the_loop is not None
        ]
        logger.info(
            "Human-in-the-Loop configuration",
            agent=agent.name,
            hitl_tools=hitl_tool_names,
        )
        middleware_list.append(hitl_middlewares)

    logger.info(
        "Middleware summary",
        agent=agent.name,
        total_middleware_count=len(middleware_list),
    )
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
    logger.info("Creating agent node", agent=agent.name)

    # Log agent configuration details
    logger.info(
        "Agent configuration",
        agent=agent.name,
        model=agent.model.name,
        description=agent.description or "No description",
    )

    llm: LanguageModelLike = agent.model.as_chat_model()

    tool_models: Sequence[ToolModel] = agent.tools
    if not additional_tools:
        additional_tools = []

    # Log tools being created
    tool_names: list[str] = [tool.name for tool in tool_models]
    logger.info(
        "Tools configuration",
        agent=agent.name,
        tools_count=len(tool_models),
        tool_names=tool_names,
    )

    tools: list[BaseTool] = list(create_tools(tool_models)) + list(additional_tools)

    if additional_tools:
        logger.debug(
            "Additional tools added",
            agent=agent.name,
            additional_count=len(additional_tools),
        )

    if memory and memory.store:
        namespace: tuple[str, ...] = ("memory",)
        if memory.store.namespace:
            namespace = namespace + (memory.store.namespace,)
        logger.info(
            "Memory configuration",
            agent=agent.name,
            has_store=True,
            has_checkpointer=memory.checkpointer is not None,
            namespace=namespace,
        )
    elif memory:
        logger.info(
            "Memory configuration",
            agent=agent.name,
            has_store=False,
            has_checkpointer=memory.checkpointer is not None,
        )

    # Add memory tools if store is configured
    if memory and memory.store:
        # Use Databricks-compatible search_memory tool (omits problematic filter field)
        tools += [
            create_manage_memory_tool(namespace=namespace),
            create_search_memory_tool(namespace=namespace),
        ]
        logger.debug(
            "Memory tools added",
            agent=agent.name,
            tools=["manage_memory", "search_memory"],
        )

    # Create middleware list from configuration
    middleware_list = _create_middleware_list(
        agent=agent,
        tool_models=tool_models,
        chat_history=chat_history,
    )

    # Log prompt configuration
    if agent.prompt:
        if isinstance(agent.prompt, PromptModel):
            logger.info(
                "Prompt configuration",
                agent=agent.name,
                prompt_type="PromptModel",
                prompt_name=agent.prompt.name,
            )
        else:
            prompt_preview: str = (
                agent.prompt[:100] + "..." if len(agent.prompt) > 100 else agent.prompt
            )
            logger.info(
                "Prompt configuration",
                agent=agent.name,
                prompt_type="string",
                prompt_preview=prompt_preview,
            )
    else:
        logger.debug("No custom prompt configured", agent=agent.name)

    checkpointer: bool = memory is not None and memory.checkpointer is not None

    # Get the prompt as middleware (always returns AgentMiddleware or None)
    prompt_middleware: AgentMiddleware | None = make_prompt(agent.prompt)

    # Add prompt middleware at the beginning for priority
    if prompt_middleware is not None:
        middleware_list.insert(0, prompt_middleware)

    # Configure structured output if response_format is specified
    response_format: Any = None
    if agent.response_format is not None:
        try:
            response_format = agent.response_format.as_strategy()
            if response_format is not None:
                logger.info(
                    "Response format configuration",
                    agent=agent.name,
                    format_type=type(response_format).__name__,
                    structured_output=True,
                )
        except ValueError as e:
            logger.error(
                "Failed to configure structured output for agent",
                agent=agent.name,
                error=str(e),
            )
            raise

    # Use LangChain v1's create_agent with middleware
    # AgentState extends MessagesState with additional DAO AI fields
    # System prompt is provided via middleware (dynamic_prompt)
    logger.info(
        "Creating LangChain agent",
        agent=agent.name,
        tools_count=len(tools),
        middleware_count=len(middleware_list),
        has_checkpointer=checkpointer,
    )

    compiled_agent: CompiledStateGraph = create_agent(
        name=agent.name,
        model=llm,
        tools=tools,
        middleware=middleware_list,
        checkpointer=checkpointer,
        state_schema=AgentState,
        context_schema=Context,
        response_format=response_format,  # Add structured output support
    )

    compiled_agent.name = agent.name

    logger.info("Agent node created successfully", agent=agent.name)

    return compiled_agent
