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
from langgraph.types import Command, Send
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
from dao_ai.orchestration.core import PARALLEL_DISPATCH_STATE_KEY
from dao_ai.state import AgentState, Context


@dataclass(frozen=True)
class HandoffResult:
    """
    Result of resolving handoff configuration for an agent.

    Separates agentic handoff tools (LLM-invoked) from the optional
    deterministic handoff target (always-routed) and the parallel
    fan-out cohort.

    ``parallel_targets`` names the sibling agents that share this source and
    are members of the fan-out cohort. Each has its own handoff tool present
    in ``tools`` (LLM-invocable — parallel execution is achieved by the LLM
    issuing multiple tool calls in a single turn). ``parallel_join`` is the
    shared deterministic join agent the cohort converges on; the parent
    graph gets a static ``add_edge(sibling, parallel_join)`` for each
    sibling so LangGraph's superstep semantics run the join exactly once.
    """

    tools: tuple[BaseTool, ...] = field(default_factory=tuple)
    deterministic_target: str | None = None
    parallel_targets: tuple[str, ...] = field(default_factory=tuple)
    parallel_join: str | None = None


def _lookup_agent(
    ref: AgentModel | str, app_agents: Sequence[AgentModel]
) -> AgentModel | None:
    """Resolve a handoff target reference to a registered ``AgentModel``.

    Handoff entries may reference agents by name (str) or embed the model
    directly. String references are looked up against ``app.agents`` (not
    ``config.find_agents``) so the swarm only routes to agents the user
    explicitly registered on the app.

    Returns ``None`` when a name doesn't match any registered agent —
    caller logs a warning and skips.
    """
    if isinstance(ref, AgentModel):
        return ref
    return next((a for a in app_agents if a.name == ref), None)


def _handoffs_for_agent(
    agent: AgentModel,
    config: AppConfig,
) -> HandoffResult:
    """
    Resolve handoff configuration for an agent.

    Processes the swarm ``handoffs`` mapping and produces:

    - A list of agentic handoff **tools** (LLM-invoked via
      ``create_handoff_tool``). Cohort entries (``agents`` + ``join``)
      produce one *parallel* handoff tool per sibling — they behave the
      same at the LLM surface as regular agentic handoffs, but return a
      state update instead of a Command so the source's ToolNode can run
      multiple in one turn without short-circuiting.
    - An optional **deterministic target** agent name for pipeline-style
      handoffs.
    - The parallel fan-out cohort's sibling names and shared join, when
      present.

    Args:
        agent: The agent to resolve handoff configuration for.
        config: The application configuration.

    Returns:
        A ``HandoffResult`` carrying tools + deterministic target +
        parallel cohort.

    Raises:
        ValueError: If two entries on the same source both set
            ``is_deterministic`` (only one deterministic edge per source
            is meaningful), or if the source's own agent name appears as
            a handoff target with a deterministic flag.
    """
    handoff_tools: list[BaseTool] = []
    deterministic_target: str | None = None
    parallel_targets: list[str] = []
    parallel_join: str | None = None

    handoffs: dict[str, Sequence[AgentModel | str | HandoffRouteModel] | None] = (
        config.app.orchestration.swarm.handoffs or {}
    )

    # Two cases for resolving an agent's outbound handoffs:
    #   * agent name MISSING from the handoffs dict → default to all
    #     other agents (legacy peer-to-peer swarm behavior).
    #   * agent name maps to a (possibly empty) list → use it.
    agent_handoffs: Sequence[AgentModel | str | HandoffRouteModel] | None = (
        handoffs.get(agent.name, config.app.agents)
    )
    if agent_handoffs is None:
        agent_handoffs = ()

    for handoff_entry in agent_handoffs:
        # Cohort entry (``HandoffRouteModel`` with ``agents`` + ``join``).
        if (
            isinstance(handoff_entry, HandoffRouteModel)
            and handoff_entry.agents is not None
        ):
            join_agent: AgentModel | None = _lookup_agent(
                handoff_entry.join, config.app.agents
            )
            if join_agent is None:
                join_name_repr: str = (
                    handoff_entry.join.name
                    if isinstance(handoff_entry.join, AgentModel)
                    else handoff_entry.join
                )
                logger.warning(
                    "Cohort join agent not found in configuration; skipping cohort",
                    source=agent.name,
                    join=join_name_repr,
                )
                continue
            if join_agent.name == agent.name:
                # This is also caught by SwarmModel.validate_parallel_cohort_shape;
                # keep as a defense-in-depth guard.
                raise ValueError(
                    f"Agent '{agent.name}' cannot be the ``join`` of its own cohort."
                )
            # Cohorts share one join per source. Emitting a second here
            # (from a second cohort on the same source) would be a
            # SwarmModel validator failure; keep a runtime guard too.
            if parallel_join is not None and parallel_join != join_agent.name:
                raise ValueError(
                    f"Agent '{agent.name}' has multiple parallel cohorts with "
                    f"different joins ('{parallel_join}' and '{join_agent.name}'). "
                    "Only one cohort per source is supported."
                )
            parallel_join = join_agent.name

            for sibling_ref in handoff_entry.agents:
                sibling_agent: AgentModel | None = _lookup_agent(
                    sibling_ref, config.app.agents
                )
                if sibling_agent is None:
                    sibling_name_repr: str = (
                        sibling_ref.name
                        if isinstance(sibling_ref, AgentModel)
                        else sibling_ref
                    )
                    logger.warning(
                        "Cohort sibling not found in configuration; skipping",
                        source=agent.name,
                        sibling=sibling_name_repr,
                    )
                    continue
                if sibling_agent.name == agent.name:
                    raise ValueError(
                        f"Agent '{agent.name}' cannot appear in its own cohort "
                        "(``agents``). A source can't fan out to itself."
                    )
                parallel_targets.append(sibling_agent.name)
                logger.info(
                    "Registered parallel fan-out sibling",
                    from_agent=agent.name,
                    to_agent=sibling_agent.name,
                    requires=sibling_agent.requires,
                )
                sibling_description: str = get_handoff_description(sibling_agent)
                handoff_tools.append(
                    create_handoff_tool(
                        target_agent_name=sibling_agent.name,
                        description=(
                            f"{sibling_description} "
                            "You may call this alongside other parallel workers in a "
                            "single turn — they will run concurrently and results are "
                            "merged before the join agent responds."
                        ),
                        requires=list(sibling_agent.requires),
                        parallel=True,
                    )
                )
            continue

        # Single-target entry.
        target_ref: AgentModel | str
        is_deterministic: bool
        if isinstance(handoff_entry, HandoffRouteModel):
            target_ref = handoff_entry.agent
            is_deterministic = handoff_entry.is_deterministic
        else:
            target_ref = handoff_entry
            is_deterministic = False

        handoff_to_agent: AgentModel | None = _lookup_agent(
            target_ref, config.app.agents
        )
        if handoff_to_agent is None:
            logger.warning(
                "Handoff agent not found in configuration", agent=agent.name
            )
            continue

        # Skip self-referencing handoffs (deterministic self-ref is a
        # hard error; agentic self-ref is silently dropped for
        # historical peer-to-peer swarm behavior).
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

    return HandoffResult(
        tools=tuple(handoff_tools),
        deterministic_target=deterministic_target,
        parallel_targets=tuple(parallel_targets),
        parallel_join=parallel_join,
    )


def _create_swarm_router(
    default_agent: str,
    agent_names: list[str],
) -> Callable[[AgentState], str]:
    """
    Create a router function for the swarm pattern.

    This router checks the ``active_agent`` field in state to determine
    which agent should handle the next step. This enables:

    1. Resuming conversations with the last active agent (from checkpointer).
    2. Routing to the default agent for new conversations.
    3. Following handoffs that set ``active_agent``.

    Args:
        default_agent: The default agent to route to if active_agent is
            not set.
        agent_names: List of valid agent names.

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


def _create_parallel_source_handler(
    inner_handler: Callable[[AgentState, "Runtime[Context]"], "Awaitable[AgentState]"],
    cohort_targets: frozenset[str],
    join_target: str,
) -> Callable[["AgentState", "Runtime[Context]"], "Awaitable[AgentState]"]:
    """Wrap a source agent that owns a parallel fan-out cohort.

    Parallel handoff tools (``create_handoff_tool(parallel=True)``) return
    only a state update — no ``goto`` / no ``graph=PARENT`` — so the
    source's ToolNode is free to execute ALL parallel tool calls in one
    LLM turn without short-circuiting on the first ``ParentCommand``. Each
    parallel handoff tool records its target in the
    ``parallel_dispatches`` state field via a list-accumulating reducer
    (``concat_parallel_dispatches``).

    After the inner handler completes, this wrapper reads that field. If
    any siblings were dispatched, it returns a single ``Command`` whose
    ``goto`` is a list of ``Send`` calls — one per fired sibling. LangGraph
    then schedules all targeted siblings in the same superstep (true
    concurrent execution). Each sibling's static edge to the shared
    ``join_target`` runs the join exactly once via superstep fan-in
    semantics.

    If the LLM didn't invoke any parallel handoff tools (e.g. it answered
    directly, or fired a non-parallel handoff that raised ParentCommand),
    the result is returned unchanged.

    Args:
        inner_handler: The handler produced by ``create_agent_node_handler``.
        cohort_targets: The names of the parallel siblings the source is
            configured to fan out to. Used to filter dispatched targets so a
            stray write from an earlier turn never routes to an unrelated
            agent.
        join_target: The shared deterministic join for the cohort. Used
            for logging and as the ``active_agent`` value the wrapper
            writes into its update so a follow-up turn without any worker
            fires still resumes at the synthesizer.

    Returns:
        An async handler with the same signature as ``inner_handler``.
    """

    async def handler(state: AgentState, runtime: Runtime[Context]) -> AgentState:
        result = await inner_handler(state, runtime)
        # If the inner handler already returned a Command (e.g. an agentic
        # peer handoff fired, not a parallel one), respect it unchanged.
        if isinstance(result, Command):
            logger.info(
                "Parallel source: inner Command passthrough",
                agent_result_goto=result.goto,
            )
            return result

        raw_dispatches: list[str] = list(result.get(PARALLEL_DISPATCH_STATE_KEY, []))
        # Filter to targets that actually belong to this cohort — defense
        # in depth against a spurious dispatch from an earlier turn.
        # Preserve LLM tool_call order and deduplicate (if the LLM double-
        # dispatched to the same sibling, run it once).
        seen: set[str] = set()
        siblings: list[str] = []
        for target in raw_dispatches:
            if target in cohort_targets and target not in seen:
                seen.add(target)
                siblings.append(target)

        # Clear the dispatch field so subsequent turns start empty. The
        # reducer accumulates; without an explicit clear it would carry
        # stale targets across turns on a checkpointed thread.
        result_cleared: dict = dict(result)
        result_cleared[PARALLEL_DISPATCH_STATE_KEY] = []

        if not siblings:
            logger.info(
                "Parallel source: no parallel siblings invoked this turn",
                cohort_size=len(cohort_targets),
            )
            return result_cleared

        base_update: dict = {**result_cleared, "active_agent": join_target}
        # LangGraph's Send carries per-target state; each sibling sees the
        # same message history and gets its own active_agent so the
        # per-sibling deterministic handler can rewrite it to the join
        # after the sibling's turn.
        sends: list[Send] = [
            Send(sibling, {**base_update, "active_agent": sibling})
            for sibling in siblings
        ]

        logger.info(
            "Parallel fan-out dispatched",
            siblings=siblings,
            join=join_target,
            sibling_count=len(siblings),
        )

        # NOTE: we deliberately do NOT set graph=Command.PARENT here. The
        # source handler is registered on the parent StateGraph itself
        # (via workflow.add_node in create_swarm_graph), so a Command with
        # goto=[Send(...), ...] routes within the parent graph — no
        # further parent to escape to. Setting graph=PARENT would raise
        # ParentCommand out of the swarm and blow up the caller.
        return Command(
            update=base_update,
            goto=sends,
        )

    return handler


def _create_deterministic_handler(
    inner_handler: Callable[[AgentState, "Runtime[Context]"], "Awaitable[AgentState]"],
    target_agent_name: str,
    *,
    sibling_of_cohort: bool = False,
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
        sibling_of_cohort: When True, this handler wraps a parallel fan-out
            sibling (target is the cohort's join). Only affects log tagging
            so runtime traces distinguish parallel-fan-in from plain
            deterministic edges.

    Returns:
        An async handler with the same signature as *inner_handler*.
    """
    # Log tag used at runtime so operators can distinguish parallel fan-in
    # events from plain deterministic pipeline events in App / Model Serving logs.
    handoff_kind: str = "parallel_fan_in" if sibling_of_cohort else "deterministic"

    async def handler(state: AgentState, runtime: Runtime[Context]) -> AgentState:
        result = await inner_handler(state, runtime)
        if isinstance(result, Command):
            logger.info(
                f"{handoff_kind} handoff overridden by agentic Command",
                target=target_agent_name,
                agentic_goto=result.goto,
                handoff_kind=handoff_kind,
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
                f"{handoff_kind} handoff: appended HumanMessage bridge to normalize message tail",
                target_agent=target_agent_name,
                handoff_kind=handoff_kind,
            )

        result["active_agent"] = target_agent_name
        logger.info(
            f"{handoff_kind} handoff fired",
            target_agent=target_agent_name,
            handoff_kind=handoff_kind,
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
    # Reverse map: sibling_agent_name -> parallel_join_target. A sibling can
    # only belong to one cohort (validated by SwarmModel), so this is a
    # simple dict. When a sibling's handler runs, it sets active_agent to
    # the join so LangGraph resumes correctly on checkpoint restore.
    parallel_sibling_joins: dict[str, str] = {}
    # Source agents that own a parallel cohort → (frozenset[targets], join).
    # These sources need the parallel-source wrapper which fans out via
    # Send() based on which parallel-handoff tools the LLM actually invoked
    # in a single turn. Without this wrapper, parallel Command handoffs
    # short-circuit on the first ParentCommand and only one sibling runs.
    parallel_sources: dict[str, tuple[frozenset[str], str]] = {}
    # Agents marked ``internal: true`` in config. Their AIMessage outputs
    # are filtered out of the history view passed to non-internal agents to
    # prevent cross-agent context leakage (Anthropic distilled-handoff
    # pattern; LangGraph swarm private-history recipe). Internal agents
    # can still see each other's outputs.
    internal_agents: frozenset[str] = frozenset(
        a.name for a in config.app.agents if a.internal
    )
    if internal_agents:
        logger.debug(
            "Swarm internal agents (outputs hidden from non-internal peers)",
            internal_agents=sorted(internal_agents),
        )
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

        # Track deterministic targets for graph wiring. When a parallel
        # cohort is present on this source, the "deterministic target" is
        # the cohort's join — reached *through* the siblings, not from the
        # source directly. Record it separately so we skip the direct edge.
        # ``SwarmModel.validate_parallel_cohort_shape`` guarantees each
        # sibling belongs to at most one cohort, so the assignment below
        # is unconditional.
        if handoff_result.parallel_join is not None:
            parallel_sources[registered_agent.name] = (
                frozenset(handoff_result.parallel_targets),
                handoff_result.parallel_join,
            )
            for sibling_name in handoff_result.parallel_targets:
                parallel_sibling_joins[sibling_name] = handoff_result.parallel_join
        elif handoff_result.deterministic_target is not None:
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
            internal_agents=internal_agents,
        )

        # Wrap the handler for deterministic routing:
        # - Sets active_agent so the swarm router resumes correctly
        # - The add_edge below provides the actual graph routing
        #
        # A node is EITHER a plain deterministic source OR a parallel
        # sibling, never both. Parallel-sibling wrapping takes precedence:
        # the sibling always routes to its cohort's join, regardless of any
        # deterministic edge that agent might have on its own outbound
        # handoffs list.
        if agent_name in parallel_sources:
            cohort_targets, cohort_join = parallel_sources[agent_name]
            handler = _create_parallel_source_handler(
                handler, cohort_targets=cohort_targets, join_target=cohort_join
            )
            logger.info(
                "Wrapped agent handler as parallel fan-out source",
                agent=agent_name,
                cohort_targets=sorted(cohort_targets),
                parallel_join=cohort_join,
            )
        elif agent_name in parallel_sibling_joins:
            join_target: str = parallel_sibling_joins[agent_name]
            handler = _create_deterministic_handler(
                handler, join_target, sibling_of_cohort=True
            )
            logger.info(
                "Wrapped agent handler as parallel fan-out sibling",
                agent=agent_name,
                parallel_join=join_target,
            )
        elif agent_name in deterministic_targets:
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

    # Wire parallel fan-out edges: every parallel sibling gets a static
    # edge to its cohort's shared join. LangGraph's superstep semantics
    # coalesce these — the join node runs exactly once after all fired
    # siblings complete, even when N > 1 siblings all target the same
    # join in the same superstep.
    #
    # We deliberately do NOT add an edge from the source to the join.
    # The source's LLM decides which siblings (if any) to invoke; the
    # join is reached *through* the siblings. When the LLM invokes zero
    # parallel handoff tools, the source's turn terminates without
    # firing the join — that's the correct semantic (nothing to reduce).
    for sibling_agent, join_agent in parallel_sibling_joins.items():
        workflow.add_edge(sibling_agent, join_agent)
        logger.info(
            "Added parallel fan-in edge",
            from_sibling=sibling_agent,
            to_join=join_agent,
        )

    # Create the swarm router that checks active_agent state.
    router = _create_swarm_router(default_agent, agent_names)

    # Use conditional entry point to route based on active_agent
    # This is the key pattern from langgraph-swarm-py
    workflow.set_conditional_entry_point(router)

    compiled = workflow.compile(checkpointer=checkpointer, store=store)

    # Apply the cross-agent hop ceiling at the parent graph level. This is
    # the only bound on agentic ping-pong between peers; per-worker
    # recursion_limit only protects within a single agent's turn.
    return compiled.with_config({"recursion_limit": swarm.max_hops})
