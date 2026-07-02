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

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command
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
    create_extraction_manager_and_executor,
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
    deterministic handoff target (always-routed). ``is_terminal`` is
    true when either:

    * the YAML's ``handoffs`` entry for this agent is an explicit empty
      list (``composer: []``) — no tools and no deterministic target are
      generated, so this is detected structurally as
      ``not tools and deterministic_target is None``; OR
    * the agent's :attr:`AgentModel.is_terminal` flag is set to ``True``
      in YAML — used to force terminal behavior on agents that still
      have outbound handoffs available (the LLM may or may not invoke
      them, but the swarm should reset between user turns either way).

    The swarm runtime detects terminal at the handler-wrap site and
    clears ``active_agent`` after the agent completes so the next user
    turn restarts at ``default_agent`` instead of resuming at the
    terminal one.
    """

    tools: tuple[BaseTool, ...] = field(default_factory=tuple)
    deterministic_target: str | None = None
    is_terminal: bool = False


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

    # Three cases for resolving an agent's outbound handoffs:
    #   * agent name MISSING from the handoffs dict → default to all
    #     other agents (legacy peer-to-peer swarm behavior).
    #   * agent name maps to an explicit empty list ([]) → terminal node;
    #     no handoff tools, no deterministic target. The swarm runtime
    #     detects this at the handler-wrap site (via
    #     ``HandoffResult.is_terminal``) and clears ``active_agent``
    #     after the agent runs.
    #   * agent name maps to a non-empty list → use it.
    agent_handoffs: Sequence[AgentModel | str | HandoffRouteModel] | None = (
        handoffs.get(agent.name, config.app.agents)
    )
    if agent_handoffs is None:
        agent_handoffs = ()

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
                requires=handoff_to_agent.requires,
            )
            handoff_description: str = get_handoff_description(handoff_to_agent)
            handoff_tools.append(
                create_handoff_tool(
                    target_agent_name=handoff_to_agent.name,
                    description=handoff_description,
                    requires=list(handoff_to_agent.requires),
                )
            )

    # Terminal when either the agent flagged itself terminal in YAML
    # (``AgentModel.is_terminal: true``) OR the resolved handoff config
    # is structurally empty (no tools and no deterministic target — the
    # ``composer: []`` case). Both routes converge on the same runtime
    # behavior: clear ``active_agent`` after this agent runs.
    is_terminal: bool = bool(agent.is_terminal) or (
        not handoff_tools and deterministic_target is None
    )

    return HandoffResult(
        tools=tuple(handoff_tools),
        deterministic_target=deterministic_target,
        is_terminal=is_terminal,
    )


def _create_swarm_router(
    default_agent: str,
    agent_names: list[str],
    terminal_agents: frozenset[str] = frozenset(),
) -> Callable[[AgentState], str]:
    """
    Create a router function for the swarm pattern.

    This router checks the ``active_agent`` field in state to determine
    which agent should handle the next step. This enables:

    1. Resuming conversations with the last active agent (from checkpointer).
    2. Routing to the default agent for new conversations.
    3. Following handoffs that set ``active_agent``.
    4. **Restarting at ``default_agent`` whenever the prior turn ended at
       a terminal agent.** An agent is terminal when its YAML ``handoffs``
       entry is an explicit empty list, OR when
       ``AgentModel.is_terminal: true`` is set. The router detects that
       state.active_agent is in the terminal set and falls through to
       ``default_agent``, effectively breaking the sticky-active-agent
       behavior between user turns.

    Args:
        default_agent: The default agent to route to if active_agent is
            not set or has been marked terminal.
        agent_names: List of valid agent names.
        terminal_agents: Names of agents that should not be resumed —
            their presence in ``active_agent`` triggers a fall-through
            to ``default_agent``.

    Returns:
        A router function that returns the agent name to route to.
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

        # If the prior turn ended at a terminal agent, restart at default.
        if active_agent in terminal_agents:
            logger.trace(
                "Prior turn ended at terminal agent; restarting at default",
                terminal_agent=active_agent,
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

    After the inner handler completes, ``active_agent`` is set to
    *target_agent_name* so that:

    1. The ``add_edge`` in the parent graph routes to the deterministic target.
    2. The swarm router correctly resumes at the target on re-entry
       (e.g. after checkpoint restore).

    If the agent invoked an agentic handoff tool during its turn, the inner
    handler returns a ``Command(graph=Command.PARENT)`` carrying its own
    routing and ``active_agent`` update. That Command is passed through
    unchanged — the agentic target takes precedence over the deterministic
    edge.

    Args:
        inner_handler: The original handler produced by ``create_agent_node_handler``.
        target_agent_name: The agent name to deterministically route to.

    Returns:
        An async handler with the same signature as *inner_handler*.
    """

    async def handler(state: AgentState, runtime: Runtime[Context]) -> AgentState:
        result = await inner_handler(state, runtime)
        if isinstance(result, Command):
            logger.debug(
                "Deterministic handoff overridden by agentic Command",
                deterministic_target=target_agent_name,
                agentic_goto=result.goto,
            )
            return result

        # Normalize the message tail before the static edge routes to the
        # downstream agent. Without this, the downstream LLM call receives
        # `messages[-1] = AIMessage(...)` (the upstream agent's plain
        # response), which Databricks Model Serving rejects for Claude with
        # "This model does not support assistant message prefill. The
        # conversation must end with a user message." This guard is
        # model-agnostic — any provider that enforces a non-assistant tail
        # is handled the same way.
        #
        # We append a small synthetic HumanMessage rather than a
        # tool_call/tool_result pair because the downstream agent's
        # `filter_messages_for_agent` (core.py) drops ToolMessages whose
        # tool_call_id wasn't issued by the current agent. HumanMessages
        # pass through the filter unchanged.
        messages = result.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            # Bridge content matters — a cryptic "[automated deterministic
            # handoff to X]" makes the downstream agent reply "that
            # message came through as a system handoff rather than a
            # question from you". Include the last real user query
            # verbatim so the next agent has grounding.
            last_user_query: str = ""
            for prior in reversed(messages):
                if (
                    isinstance(prior, HumanMessage)
                    and prior.name != "__deterministic_handoff__"
                    and prior.name != "__filter_bridge__"
                ):
                    content = prior.content
                    if isinstance(content, str) and content.strip():
                        last_user_query = content.strip()
                        break
            if last_user_query:
                bridge_content = (
                    f"Continue this pipeline as {target_agent_name}. "
                    f"The user's original request was: {last_user_query!r}. "
                    f"Do your job based on the prior agents' work above."
                )
            else:
                bridge_content = (
                    f"Continue this pipeline as {target_agent_name}. "
                    f"Do your job based on the prior agents' work above."
                )
            bridge: HumanMessage = HumanMessage(
                content=bridge_content,
                name="__deterministic_handoff__",
            )
            result["messages"] = list(messages) + [bridge]
            logger.debug(
                "Deterministic handoff: appended HumanMessage bridge to normalize message tail",
                target_agent=target_agent_name,
            )

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
    agent_recursion_limits: dict[str, int | None] = {}
    deterministic_targets: dict[str, str] = {}
    terminal_agents: set[str] = set()
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
    extraction_manager, reflection_executor = create_extraction_manager_and_executor(
        memory=memory,
        store=store,
        fallback_model=config.app.agents[0].model.as_chat_model(),
        graph_label="swarm graph",
    )

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

        # Track terminal agents so the parent-graph handler can reset
        # ``active_agent`` after a turn ends here. Terminal supersedes
        # nothing — an agent with a deterministic target is never terminal
        # because it always routes onward.
        if handoff_result.is_terminal and handoff_result.deterministic_target is None:
            terminal_agents.add(registered_agent.name)
            logger.debug(
                "Registered terminal agent (active_agent will reset after its turns)",
                agent=registered_agent.name,
                explicit_is_terminal=bool(registered_agent.is_terminal),
                empty_handoffs=not handoff_result.tools,
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
            checkpointer=checkpointer,
        )
        agent_subgraphs[registered_agent.name] = agent_subgraph
        agent_recursion_limits[registered_agent.name] = registered_agent.recursion_limit
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
            output_mode=orchestration.output_mode,
            reflection_executor=reflection_executor,
            recursion_limit=agent_recursion_limits.get(agent_name),
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

    # Create the swarm router that checks active_agent state.
    # Terminal agents short-circuit back to default_agent — see the
    # docstring on _create_swarm_router for the semantics.
    router = _create_swarm_router(
        default_agent, agent_names, frozenset(terminal_agents)
    )

    # Use conditional entry point to route based on active_agent
    # This is the key pattern from langgraph-swarm-py
    workflow.set_conditional_entry_point(router)

    compiled = workflow.compile(checkpointer=checkpointer, store=store)

    # Apply the cross-agent hop ceiling at the parent graph level. This is
    # the only bound on agentic ping-pong between peers; per-worker
    # recursion_limit only protects within a single agent's turn.
    return compiled.with_config({"recursion_limit": swarm.max_hops})
