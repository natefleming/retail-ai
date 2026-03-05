"""
Swarm pattern for multi-agent orchestration.

The swarm pattern allows agents to directly hand off control to each other
without a central coordinator. Each agent has handoff tools for the agents
they are allowed to transfer control to. This provides decentralized,
peer-to-peer collaboration.

Supports two handoff modes:
- **Agentic** (default): The LLM decides when to transfer control via tool calls.
- **Deterministic**: Control always transfers to the specified agent after the
  source agent completes its turn, without LLM routing.

Based on: https://github.com/langchain-ai/langgraph-swarm-py
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from loguru import logger

from dao_ai.config import (
    AgentModel,
    AppConfig,
    HandoffRouteModel,
    MemoryModel,
    OrchestrationModel,
    SwarmModel,
)
from dao_ai.nodes import create_agent_node
from dao_ai.orchestration import (
    create_agent_node_handler,
    create_checkpointer,
    create_handoff_tool,
    create_store,
    get_handoff_description,
)
from dao_ai.state import AgentState, Context


@dataclass(frozen=True)
class HandoffResult:
    """
    Result of resolving handoff configuration for an agent.

    Separates agentic handoff tools (LLM-invoked) from the optional
    deterministic handoff target (always-routed).
    """

    tools: list[BaseTool] = field(default_factory=list)
    deterministic_target: str | None = None


def _resolve_agent(
    handoff_entry: AgentModel | str | HandoffRouteModel,
) -> tuple[AgentModel | str, bool]:
    """
    Normalize a handoff entry into (agent_ref, is_deterministic).

    Args:
        handoff_entry: A handoff target — may be a plain agent name,
            an ``AgentModel``, or a ``HandoffRouteModel``.

    Returns:
        A tuple of (agent reference, is_deterministic flag).
    """
    if isinstance(handoff_entry, HandoffRouteModel):
        return handoff_entry.agent, handoff_entry.is_deterministic
    return handoff_entry, False


def _handoffs_for_agent(
    agent: AgentModel,
    config: AppConfig,
) -> HandoffResult:
    """
    Resolve handoff configuration for an agent.

    Processes the swarm ``handoffs`` mapping and produces:
    - A list of agentic handoff **tools** (LLM-invoked via ``create_handoff_tool``).
    - An optional **deterministic target** agent name that the source agent
      always routes to after completing its turn.

    Handoff tools route to the parent graph since agents are subgraphs
    wrapped in handlers.

    Args:
        agent: The agent to resolve handoff configuration for.
        config: The application configuration.

    Returns:
        A ``HandoffResult`` containing agentic tools and an optional
        deterministic target.

    Raises:
        ValueError: If more than one deterministic handoff is configured for
            the same agent, or if a deterministic handoff references itself.
    """
    handoff_tools: list[BaseTool] = []
    deterministic_target: str | None = None

    handoffs: dict[str, Sequence[AgentModel | str | HandoffRouteModel] | None] = (
        config.app.orchestration.swarm.handoffs or {}
    )

    agent_handoffs: Sequence[AgentModel | str | HandoffRouteModel] | None = (
        handoffs.get(agent.name)
    )
    if agent_handoffs is None:
        agent_handoffs = config.app.agents

    for handoff_entry in agent_handoffs:
        agent_ref: AgentModel | str
        is_deterministic: bool
        agent_ref, is_deterministic = _resolve_agent(handoff_entry)

        # Resolve string references to AgentModel using the app-level agent list.
        # We search config.app.agents (not config.find_agents) because the swarm
        # should only reference agents registered in the app's agent list.
        handoff_to_agent: AgentModel | None
        if isinstance(agent_ref, str):
            handoff_to_agent = next(
                (a for a in config.app.agents if a.name == agent_ref),
                None,
            )
        else:
            handoff_to_agent = agent_ref

        if handoff_to_agent is None:
            logger.warning("Handoff agent not found in configuration", agent=agent.name)
            continue

        # Skip self-referencing handoffs
        if agent.name == handoff_to_agent.name:
            if is_deterministic:
                raise ValueError(
                    f"Agent '{agent.name}' cannot have a deterministic "
                    f"handoff to itself."
                )
            continue

        if is_deterministic:
            if deterministic_target is not None:
                raise ValueError(
                    f"Agent '{agent.name}' has multiple deterministic handoffs. "
                    f"Only one deterministic handoff is allowed per agent. "
                    f"Found targets: '{deterministic_target}' and "
                    f"'{handoff_to_agent.name}'."
                )
            deterministic_target = handoff_to_agent.name
            logger.debug(
                "Registered deterministic handoff",
                from_agent=agent.name,
                to_agent=handoff_to_agent.name,
            )
        else:
            logger.debug(
                "Creating handoff tool",
                from_agent=agent.name,
                to_agent=handoff_to_agent.name,
            )
            handoff_description: str = get_handoff_description(handoff_to_agent)
            handoff_tools.append(
                create_handoff_tool(
                    target_agent_name=handoff_to_agent.name,
                    description=handoff_description,
                )
            )

    return HandoffResult(
        tools=handoff_tools,
        deterministic_target=deterministic_target,
    )


def _create_swarm_router(
    default_agent: str,
    agent_names: list[str],
) -> Callable[[AgentState], str]:
    """
    Create a router function for the swarm pattern.

    This router checks the `active_agent` field in state to determine
    which agent should handle the next step. This enables:
    1. Resuming conversations with the last active agent (from checkpointer)
    2. Routing to the default agent for new conversations
    3. Following handoffs that set active_agent

    Args:
        default_agent: The default agent to route to if active_agent is not set
        agent_names: List of valid agent names

    Returns:
        A router function that returns the agent name to route to
    """

    def router(state: AgentState) -> str:
        active_agent: str | None = state.get("active_agent")

        # If no active agent set, use default
        if not active_agent:
            logger.trace(
                "No active agent in state, routing to default",
                default_agent=default_agent,
            )
            return default_agent

        # Validate active_agent exists
        if active_agent in agent_names:
            logger.trace("Routing to active agent", active_agent=active_agent)
            return active_agent

        # Fallback to default if active_agent is invalid
        logger.warning(
            "Invalid active agent, routing to default",
            active_agent=active_agent,
            default_agent=default_agent,
        )
        return default_agent

    return router


def _create_deterministic_handler(
    inner_handler: Callable[[AgentState, "Runtime[Context]"], "Awaitable[AgentState]"],
    target_agent_name: str,
) -> Callable[["AgentState", "Runtime[Context]"], "Awaitable[AgentState]"]:
    """
    Wrap an agent node handler to set ``active_agent`` for deterministic routing.

    After the inner handler completes, ``active_agent`` is unconditionally set
    to *target_agent_name* so that:

    1. The ``add_edge`` in the parent graph routes to the deterministic target.
    2. The swarm router correctly resumes at the target on re-entry
       (e.g. after checkpoint restore).

    If the agent invoked an agentic handoff tool during its turn, the resulting
    ``Command(graph=Command.PARENT)`` takes precedence over the static edge,
    so this wrapper is effectively a no-op in that case.

    Args:
        inner_handler: The original handler produced by ``create_agent_node_handler``.
        target_agent_name: The agent name to deterministically route to.

    Returns:
        An async handler with the same signature as *inner_handler*.
    """

    async def handler(state: AgentState, runtime: Runtime[Context]) -> AgentState:
        result: AgentState = await inner_handler(state, runtime)
        result["active_agent"] = target_agent_name
        logger.debug(
            "Deterministic handoff: setting active_agent",
            target_agent=target_agent_name,
        )
        return result

    return handler


def create_swarm_graph(config: AppConfig) -> CompiledStateGraph:
    """
    Create a swarm-based multi-agent graph.

    The swarm pattern allows agents to directly hand off control to each other
    without a central coordinator. Each agent has handoff tools for the agents
    they are allowed to transfer control to.

    Supports two handoff modes:

    - **Agentic** (default): Handoff tools are added to the agent and the LLM
      decides when to invoke them via ``Command(goto=..., graph=Command.PARENT)``.
    - **Deterministic**: A static ``add_edge`` in the parent graph routes
      control to a fixed target agent after the source agent completes its
      turn. The handler wrapper sets ``active_agent`` for checkpoint
      resumption.

    Key features:
    1. Router function checks ``active_agent`` state to resume with last active agent
    2. Handoff tools update ``active_agent`` and use ``Command(goto=...)`` to route
    3. Agents are ``CompiledStateGraph`` instances wrapped in handlers for message filtering
    4. Checkpointer persists state to enable conversation resumption
    5. Deterministic handoffs use ``add_edge`` for unconditional routing

    Args:
        config: The application configuration

    Returns:
        A compiled LangGraph state machine

    See: https://github.com/langchain-ai/langgraph-swarm-py
    """
    orchestration: OrchestrationModel = config.app.orchestration
    swarm: SwarmModel = orchestration.swarm

    # Determine the default agent name
    default_agent: str
    if isinstance(swarm.default_agent, AgentModel):
        default_agent = swarm.default_agent.name
    elif swarm.default_agent is not None:
        default_agent = swarm.default_agent
    elif len(config.app.agents) > 0:
        # Fallback to first agent if no default specified
        default_agent = config.app.agents[0].name
    else:
        raise ValueError("Swarm requires at least one agent and a default_agent")

    logger.info(
        "Creating swarm graph",
        pattern="handoff",
        default_agent=default_agent,
        agents_count=len(config.app.agents),
    )

    # Create agent subgraphs with their specific handoff tools
    # Each agent gets handoff tools only for agents they're allowed to hand off to
    agent_subgraphs: dict[str, CompiledStateGraph] = {}
    deterministic_targets: dict[str, str] = {}
    memory: MemoryModel | None = orchestration.memory

    # Set up memory store early so we can pass it to agents for auto-injection
    store: BaseStore | None = create_store(orchestration)
    checkpointer: BaseCheckpointSaver | None = create_checkpointer(orchestration)

    # Get swarm-level middleware to apply to all agents
    swarm_middleware: list = swarm.middleware if swarm.middleware else []
    if swarm_middleware:
        logger.info(
            "Applying swarm-level middleware to all agents",
            middleware_count=len(swarm_middleware),
            middleware_names=[mw.name for mw in swarm_middleware],
        )

    # Set up shared extraction manager and background reflection executor
    # before creating agents so the manager can be shared across all nodes.
    extraction_manager = None
    reflection_executor = None
    needs_extraction = (
        memory
        and memory.store
        and memory.extraction
        and store
        and (memory.extraction.background_extraction or memory.extraction.auto_inject)
    )
    if needs_extraction:
        from dao_ai.memory.extraction import (
            create_extraction_manager,
            create_reflection_executor,
        )
        from dao_ai.nodes import _build_memory_namespace

        extraction_ns = _build_memory_namespace(memory)
        extraction_model: LanguageModelLike = (
            memory.extraction.extraction_model.as_chat_model()
            if memory.extraction.extraction_model
            else config.app.agents[0].model.as_chat_model()
        )
        query_model: LanguageModelLike | None = (
            memory.extraction.query_model.as_chat_model()
            if memory.extraction.query_model
            else None
        )

        extraction_manager = create_extraction_manager(
            model=extraction_model,
            store=store,
            namespace=extraction_ns,
            schemas=memory.extraction.schemas,
            instructions=memory.extraction.instructions,
            query_model=query_model,
        )

        if memory.extraction.background_extraction:
            reflection_executor = create_reflection_executor(extraction_manager, store)
            logger.info("Background memory extraction enabled for swarm graph")

    for registered_agent in config.app.agents:
        # Resolve handoff configuration for this agent
        handoff_result: HandoffResult = _handoffs_for_agent(
            agent=registered_agent,
            config=config,
        )

        # Track deterministic targets for graph wiring
        if handoff_result.deterministic_target is not None:
            deterministic_targets[registered_agent.name] = (
                handoff_result.deterministic_target
            )

        # Merge swarm-level middleware with agent-specific middleware
        # Swarm middleware is applied first, then agent middleware
        if swarm_middleware:
            from copy import deepcopy

            # Create a copy of the agent to avoid modifying the original
            agent_with_middleware = deepcopy(registered_agent)

            # Combine swarm middleware (first) with agent middleware
            agent_with_middleware.middleware = (
                swarm_middleware + agent_with_middleware.middleware
            )

            logger.debug(
                "Merged middleware for agent",
                agent=registered_agent.name,
                swarm_middleware_count=len(swarm_middleware),
                agent_middleware_count=len(registered_agent.middleware),
                total_middleware_count=len(agent_with_middleware.middleware),
            )
        else:
            agent_with_middleware = registered_agent

        agent_subgraph: CompiledStateGraph = create_agent_node(
            agent=agent_with_middleware,
            memory=memory,
            store=store,
            chat_history=config.app.chat_history,
            additional_tools=handoff_result.tools,
            extraction_manager=extraction_manager,
        )
        agent_subgraphs[registered_agent.name] = agent_subgraph
        logger.debug(
            "Created swarm agent subgraph",
            agent=registered_agent.name,
            handoffs_count=len(handoff_result.tools),
            deterministic_target=handoff_result.deterministic_target,
        )

    # Get list of agent names for the router
    agent_names: list[str] = list(agent_subgraphs.keys())

    # Create the workflow graph
    # All agents are nodes wrapped in handlers, handoffs route via Command
    workflow: StateGraph = StateGraph(
        AgentState,
        input=AgentState,
        output=AgentState,
        context_schema=Context,
    )

    # Add agent nodes with message filtering handlers
    # This ensures consistent behavior with supervisor pattern
    for agent_name, agent_subgraph in agent_subgraphs.items():
        handler = create_agent_node_handler(
            agent_name=agent_name,
            agent=agent_subgraph,
            output_mode="last_message",
            reflection_executor=reflection_executor,
        )

        # Wrap the handler for deterministic routing:
        # - Sets active_agent so the swarm router resumes correctly
        # - The add_edge below provides the actual graph routing
        if agent_name in deterministic_targets:
            target: str = deterministic_targets[agent_name]
            handler = _create_deterministic_handler(handler, target)
            logger.debug(
                "Wrapped agent handler for deterministic handoff",
                agent=agent_name,
                deterministic_target=target,
            )

        workflow.add_node(agent_name, handler)

    # Wire deterministic edges in the parent graph
    # When the agent finishes without an agentic handoff tool firing,
    # the static edge routes control to the deterministic target.
    # If an agentic handoff tool fires, Command(graph=Command.PARENT)
    # overrides this static edge (standard LangGraph behavior).
    for source_agent, target_agent in deterministic_targets.items():
        workflow.add_edge(source_agent, target_agent)
        logger.info(
            "Added deterministic edge",
            from_agent=source_agent,
            to_agent=target_agent,
        )

    # Create the swarm router that checks active_agent state
    # This enables resuming conversations with the last active agent
    router = _create_swarm_router(default_agent, agent_names)

    # Use conditional entry point to route based on active_agent
    # This is the key pattern from langgraph-swarm-py
    workflow.set_conditional_entry_point(router)

    return workflow.compile(checkpointer=checkpointer, store=store)
