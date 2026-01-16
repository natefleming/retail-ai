"""
Swarm pattern for multi-agent orchestration.

The swarm pattern allows agents to directly hand off control to each other
without a central coordinator. Each agent has handoff tools for the agents
they are allowed to transfer control to. This provides decentralized,
peer-to-peer collaboration.

Based on: https://github.com/langchain-ai/langgraph-swarm-py
"""

from typing import Callable, Sequence

from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from loguru import logger

from dao_ai.config import (
    AgentModel,
    AppConfig,
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


def _handoffs_for_agent(
    agent: AgentModel,
    config: AppConfig,
) -> Sequence[BaseTool]:
    """
    Create handoff tools for an agent based on configuration.

    Handoff tools route to the parent graph since agents are subgraphs
    wrapped in handlers.

    Args:
        agent: The agent to create handoff tools for
        config: The application configuration

    Returns:
        List of handoff tools for the agent
    """
    handoff_tools: list[BaseTool] = []

    handoffs: dict[str, Sequence[AgentModel | str] | None] = (
        config.app.orchestration.swarm.handoffs or {}
    )
    agent_handoffs: Sequence[AgentModel | str] | None = handoffs.get(agent.name)
    if agent_handoffs is None:
        agent_handoffs = config.app.agents

    for handoff_to_agent in agent_handoffs:
        if isinstance(handoff_to_agent, str):
            handoff_to_agent = next(
                iter(config.find_agents(lambda a: a.name == handoff_to_agent)), None
            )

        if handoff_to_agent is None:
            logger.warning("Handoff agent not found in configuration", agent=agent.name)
            continue
        if agent.name == handoff_to_agent.name:
            continue
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
    return handoff_tools


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


def create_swarm_graph(config: AppConfig) -> CompiledStateGraph:
    """
    Create a swarm-based multi-agent graph.

    The swarm pattern allows agents to directly hand off control to each other
    without a central coordinator. Each agent has handoff tools for the agents
    they are allowed to transfer control to.

    Key features:
    1. Router function checks `active_agent` state to resume with last active agent
    2. Handoff tools update `active_agent` and use Command(goto=...) to route
    3. Agents are CompiledStateGraphs wrapped in handlers for message filtering
    4. Checkpointer persists state to enable conversation resumption

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
    memory: MemoryModel | None = orchestration.memory

    # Get swarm-level middleware to apply to all agents
    swarm_middleware: list = swarm.middleware if swarm.middleware else []
    if swarm_middleware:
        logger.info(
            "Applying swarm-level middleware to all agents",
            middleware_count=len(swarm_middleware),
            middleware_names=[mw.name for mw in swarm_middleware],
        )

    for registered_agent in config.app.agents:
        # Get handoff tools for this agent
        handoff_tools: Sequence[BaseTool] = _handoffs_for_agent(
            agent=registered_agent,
            config=config,
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
            chat_history=config.app.chat_history,
            additional_tools=handoff_tools,
        )
        agent_subgraphs[registered_agent.name] = agent_subgraph
        logger.debug(
            "Created swarm agent subgraph",
            agent=registered_agent.name,
            handoffs_count=len(handoff_tools),
        )

    # Set up memory store and checkpointer
    store: BaseStore | None = create_store(orchestration)
    checkpointer: BaseCheckpointSaver | None = create_checkpointer(orchestration)

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
        )
        workflow.add_node(agent_name, handler)

    # Create the swarm router that checks active_agent state
    # This enables resuming conversations with the last active agent
    router = _create_swarm_router(default_agent, agent_names)

    # Use conditional entry point to route based on active_agent
    # This is the key pattern from langgraph-swarm-py
    workflow.set_conditional_entry_point(router)

    return workflow.compile(checkpointer=checkpointer, store=store)
