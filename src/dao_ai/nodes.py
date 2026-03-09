"""
Node creation utilities for DAO AI agents.

This module provides factory functions for creating LangGraph nodes
that implement agent logic using LangChain v1's create_agent pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from loguru import logger

from dao_ai.config import (
    AgentModel,
    ChatHistoryModel,
    DecisionResponse,
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
from dao_ai.tools.memory import (
    create_manage_memory_tool,
    create_search_memory_tool,
    create_search_user_profile_tool,
)

if TYPE_CHECKING:
    from langmem.knowledge.extraction import MemoryStoreManager

MEMORY_TOOL_INSTRUCTIONS = (
    "\n\n#### Memory Tools\n"
    "You have access to long-term memory tools. Use them silently:\n"
    "- Search memory before responding when the question may relate to "
    "something the user mentioned previously\n"
    "- Store new preferences and important context without telling the user\n"
    "- Never display function call syntax, tool names, or JSON to the user\n"
    "- Present memory-related information naturally in conversation"
)


def _build_hitl_prompt_guidance(tool_models: Sequence[ToolModel]) -> str | None:
    """Build HITL decision-guidance text from tool configs.

    Scans all tool models for HITL configurations and builds a prompt
    section that instructs the LLM how to respond after each decision
    type.  Only decisions configured with ``mode == "guidance"`` are
    included; template-mode decisions are handled at resume time and
    require no prompt injection.

    Returns:
        A formatted prompt section string, or ``None`` when no tools
        have guidance-mode decisions.
    """
    sections: list[str] = []

    for tool_model in tool_models:
        hitl = tool_model.function.human_in_the_loop
        if hitl is None:
            continue

        lines: list[str] = [f"### {tool_model.name}"]
        has_guidance = False
        for decision in hitl.allowed_decisions:
            resp: DecisionResponse = hitl.decision_response.response_for(decision)
            if resp.mode == "guidance":
                lines.append(f"- If **{decision}**: {resp.guidance}")
                has_guidance = True

        if has_guidance:
            sections.append("\n".join(lines))

    if not sections:
        return None

    header: str = (
        "\n\n## Human-in-the-Loop Decision Guidance\n"
        "After a tool call that requires human approval has been decided, "
        "follow the instructions below for each tool and decision type.\n\n"
    )
    return header + "\n\n".join(sections)


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
        # Use the LLMModel's URI as the MLflow judge model endpoint
        model_endpoint: str = guardrail.model.uri

        # GuardrailMiddleware handles PromptModel resolution internally
        guardrail_middleware: GuardrailMiddleware = GuardrailMiddleware(
            name=guardrail.name,
            model=model_endpoint,
            prompt=guardrail.prompt,
            num_retries=guardrail.num_retries or 3,
            fail_open=guardrail.fail_open if guardrail.fail_open is not None else True,
            max_context_length=guardrail.max_context_length or 8000,
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


def _build_memory_namespace(memory: MemoryModel) -> tuple[str, ...]:
    """Build the memory namespace tuple from config."""
    namespace: tuple[str, ...] = ("memory",)
    if memory.store and memory.store.namespace:
        namespace = namespace + (memory.store.namespace,)
    return namespace


def _create_extraction_manager(
    memory: MemoryModel,
    store: BaseStore,
    namespace: tuple[str, ...],
    fallback_model: LanguageModelLike,
) -> MemoryStoreManager | None:
    """Create a memory extraction manager if extraction is configured."""
    extraction = memory.extraction
    if extraction is None:
        return None

    from dao_ai.memory.extraction import create_extraction_manager

    model: LanguageModelLike = (
        extraction.extraction_model.as_chat_model()
        if extraction.extraction_model
        else fallback_model
    )
    query_model: LanguageModelLike | None = (
        extraction.query_model.as_chat_model() if extraction.query_model else None
    )

    return create_extraction_manager(
        model=model,
        store=store,
        namespace=namespace,
        schemas=extraction.schemas,
        instructions=extraction.instructions,
        query_model=query_model,
    )


def create_agent_node(
    agent: AgentModel,
    memory: Optional[MemoryModel] = None,
    store: Optional[BaseStore] = None,
    chat_history: Optional[ChatHistoryModel] = None,
    additional_tools: Optional[Sequence[BaseTool]] = None,
    extraction_manager: Optional[MemoryStoreManager] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledStateGraph:
    """
    Factory function that creates a LangGraph node for a specialized agent.

    This creates an agent using LangChain v1's create_agent function with
    middleware for customization. The function configures the agent with
    the appropriate model, prompt, tools, and middleware.

    Args:
        agent: AgentModel configuration for the agent
        memory: Optional MemoryModel for memory store configuration
        store: Optional BaseStore instance (pre-created by orchestration layer)
        chat_history: Optional ChatHistoryModel for chat history summarization
        additional_tools: Optional sequence of additional tools to add to the agent
        extraction_manager: Optional shared MemoryStoreManager for memory
            context injection. When provided, this instance is reused instead
            of creating a new one per agent.
        checkpointer: Optional persistent checkpointer for HITL subgraph state.
            Required for Model Serving where multiple workers need shared state.
            Falls back to InMemorySaver if not provided.

    Returns:
        A compiled agent node that processes state and returns responses
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

    namespace: tuple[str, ...] = ("memory",)
    if memory and memory.store:
        namespace = _build_memory_namespace(memory)
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
    has_memory_tools: bool = False
    if memory and memory.store:
        tools += [
            create_manage_memory_tool(namespace=namespace, store=store),
            create_search_memory_tool(namespace=namespace, store=store),
        ]
        memory_tool_names = ["manage_memory", "search_memory"]
        has_memory_tools = True

        # Add profile lookup tool when user_profile schema is configured
        if (
            store
            and memory.extraction
            and memory.extraction.schemas
            and "user_profile" in memory.extraction.schemas
        ):
            tools.append(
                create_search_user_profile_tool(store=store, namespace=namespace)
            )
            memory_tool_names.append("search_user_profile")

        logger.debug(
            "Memory tools added",
            agent=agent.name,
            tools=memory_tool_names,
        )

    # Create middleware list from configuration
    middleware_list = _create_middleware_list(
        agent=agent,
        tool_models=tool_models,
        chat_history=chat_history,
    )

    # Always apply tool-call-id sanitizer so models that omit IDs don't crash
    from dao_ai.middleware.tool_call_id_sanitizer import ToolCallIdSanitizerMiddleware

    middleware_list.append(ToolCallIdSanitizerMiddleware())

    # Add memory context injection middleware if configured
    if memory and memory.store and memory.extraction and store:
        extraction = memory.extraction
        if extraction.auto_inject:
            mgr = extraction_manager or _create_extraction_manager(
                memory=memory,
                store=store,
                namespace=namespace,
                fallback_model=llm,
            )
            if mgr:
                from dao_ai.middleware.memory_context import MemoryContextMiddleware

                memory_middleware = MemoryContextMiddleware(
                    manager=mgr,
                    limit=extraction.auto_inject_limit,
                )
                middleware_list.append(memory_middleware)
                logger.info(
                    "Memory context injection enabled",
                    agent=agent.name,
                    auto_inject_limit=extraction.auto_inject_limit,
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

    # Append memory tool instructions to the prompt when memory tools are present
    effective_prompt: str | PromptModel | None = agent.prompt
    if has_memory_tools and effective_prompt is not None:
        if isinstance(effective_prompt, PromptModel):
            effective_prompt = PromptModel(
                **{
                    **effective_prompt.model_dump(),
                    "template": effective_prompt.template + MEMORY_TOOL_INSTRUCTIONS,
                }
            )
        else:
            effective_prompt = effective_prompt + MEMORY_TOOL_INSTRUCTIONS
        logger.debug("Memory tool instructions appended to prompt", agent=agent.name)

    # Append HITL decision guidance to the prompt when tools have HITL config
    hitl_guidance: str | None = _build_hitl_prompt_guidance(tool_models)
    if hitl_guidance is not None:
        if effective_prompt is None:
            effective_prompt = hitl_guidance.lstrip("\n")
        elif isinstance(effective_prompt, PromptModel):
            effective_prompt = PromptModel(
                **{
                    **effective_prompt.model_dump(),
                    "template": effective_prompt.template + hitl_guidance,
                }
            )
        else:
            effective_prompt = effective_prompt + hitl_guidance
        logger.debug("HITL decision guidance appended to prompt", agent=agent.name)

    # Get the prompt as middleware (always returns AgentMiddleware or None)
    prompt_middleware: AgentMiddleware | None = make_prompt(effective_prompt)

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

    # HITL agents need a checkpointer so that interrupt()/resume works
    # within the subgraph.  The handler in orchestration/core.py calls
    # agent.ainvoke() directly (not via add_node), so the subgraph
    # cannot inherit the parent's checkpointer -- it needs its own.
    # In Model Serving with multiple workers, InMemorySaver state is lost
    # across requests; a persistent checkpointer (e.g. Lakebase-backed)
    # must be used.  The sub_thread_id scoping in the handler prevents
    # conflicts between the parent and subgraph checkpointer tables.
    # Non-HITL agents keep checkpointer=False to disable checkpointing.
    has_hitl: bool = any(
        isinstance(mw, HumanInTheLoopMiddleware) for mw in middleware_list
    )
    subgraph_checkpointer: BaseCheckpointSaver | Literal[False] = (
        (checkpointer or InMemorySaver()) if has_hitl else False
    )

    logger.info(
        "Creating LangChain agent",
        agent=agent.name,
        tools_count=len(tools),
        middleware_count=len(middleware_list),
        has_hitl=has_hitl,
        subgraph_checkpointer=type(subgraph_checkpointer).__name__,
    )

    compiled_agent: CompiledStateGraph = create_agent(
        name=agent.name,
        model=llm,
        tools=tools,
        middleware=middleware_list,
        checkpointer=subgraph_checkpointer,
        state_schema=AgentState,
        context_schema=Context,
        response_format=response_format,
    )

    compiled_agent.name = agent.name

    logger.info("Agent node created successfully", agent=agent.name)

    return compiled_agent
