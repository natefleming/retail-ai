"""
Core orchestration utilities and infrastructure.

This module provides the foundational utilities for multi-agent orchestration:
- Memory and checkpointer creation
- Message filtering and extraction
- Agent node handlers
- Handoff tools
- Main orchestration graph factory
"""

from typing import Any, Awaitable, Callable, Literal

from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_config
from langgraph.graph.state import CompiledStateGraph, ParentCommand
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Command, Interrupt
from langgraph.types import interrupt as langgraph_interrupt
from loguru import logger

from dao_ai.config import (
    AgentModel,
    AppConfig,
    MemoryModel,
    OrchestrationModel,
)
from dao_ai.messages import last_ai_message, last_ai_message_with_tool_calls
from dao_ai.state import AgentState, Context

# Constant for supervisor node name
SUPERVISOR_NODE = "supervisor"

# Output mode for agent responses
# - "full_history": Include all messages from the agent's execution
# - "last_message": Include only the final AI message from the agent
OutputMode = Literal["full_history", "last_message"]


def create_store(orchestration: OrchestrationModel) -> BaseStore | None:
    """
    Create a memory store from orchestration config.

    Args:
        orchestration: The orchestration configuration

    Returns:
        The configured store, or None if not configured
    """
    if orchestration.memory and orchestration.memory.store:
        store = orchestration.memory.store.as_store()
        logger.debug("Memory store configured", store_type=type(store).__name__)
        return store
    return None


def create_checkpointer(
    orchestration: OrchestrationModel,
) -> BaseCheckpointSaver | None:
    """
    Create a checkpointer from orchestration config.

    Args:
        orchestration: The orchestration configuration

    Returns:
        The configured checkpointer, or None if not configured
    """
    if orchestration.memory and orchestration.memory.checkpointer:
        checkpointer = orchestration.memory.checkpointer.as_checkpointer()
        logger.debug(
            "Checkpointer configured", checkpointer_type=type(checkpointer).__name__
        )
        return checkpointer
    return None


def filter_messages_for_agent(
    messages: list[BaseMessage],
    current_agent_name: str | None = None,
) -> list[BaseMessage]:
    """
    Filter messages for a worker agent to avoid tool_use/tool_result pairing errors.

    When agents share parent-graph state, each agent should only see:
    - HumanMessage (user queries)
    - AIMessage with content from any agent (peer responses, with tool_calls
      stripped so the model doesn't see orphaned tool_use blocks)
    - Its own AIMessage(tool_calls=...) and the matching ToolMessage(s), so a
      multi-turn agent can see its own prior tool results

    Ownership rules:
    - AIMessages are tagged with the agent's name by ``create_agent`` (and as
      a fallback by ``create_agent_node_handler``). ``msg.name ==
      current_agent_name`` identifies an own AIMessage.
    - ToolMessages carry the **tool's** name in ``msg.name`` (e.g.
      ``product_vector_search_tool``) — *not* the agent's. To attribute a
      ToolMessage to an agent, we pair it via ``tool_call_id`` against the
      surrounding ``AIMessage(tool_calls=…)`` we already decided to keep.

    Args:
        messages: The full message history from parent state
        current_agent_name: The name of the agent receiving the filtered
            history. When set, the agent's own tool exchanges are preserved.
            When ``None`` (legacy callers), all tool exchanges are stripped.

    Returns:
        Filtered messages safe for the agent to process
    """
    filtered: list[BaseMessage] = []
    own_tool_call_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered.append(msg)
        elif isinstance(msg, AIMessage):
            is_own = current_agent_name is not None and msg.name == current_agent_name
            if msg.tool_calls:
                if is_own:
                    filtered.append(msg)
                    for tc in msg.tool_calls:
                        tc_id = (
                            tc.get("id")
                            if isinstance(tc, dict)
                            else getattr(tc, "id", None)
                        )
                        if tc_id:
                            own_tool_call_ids.add(tc_id)
                elif msg.content:
                    # Peer/orchestration message: keep visible content but
                    # strip tool_calls so the model doesn't see orphans.
                    filtered.append(
                        AIMessage(content=msg.content, id=msg.id, name=msg.name)
                    )
                # else: drop silent tool-call-only message from peers
            elif msg.content:
                filtered.append(msg)
        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id in own_tool_call_ids:
                # Pair with a kept own AIMessage(tool_calls=…).
                filtered.append(msg)
            # else: peer / handoff / orchestration tool result — drop
    return filtered


def extract_agent_response(
    messages: list[BaseMessage],
    output_mode: OutputMode = "last_message",
) -> list[BaseMessage]:
    """
    Extract the agent's response based on the output mode.

    Args:
        messages: The agent's full message history after execution
        output_mode: How to extract the response
            - "full_history": Return all messages (may cause issues)
            - "last_message": Return only the final AI message

    Returns:
        Messages to include in the parent state update
    """
    if output_mode == "full_history":
        return messages

    # Find the last AI message with content
    final_response: AIMessage | None = last_ai_message(messages)

    if final_response:
        # Return clean AIMessage without tool_calls
        if final_response.tool_calls:
            return [AIMessage(content=final_response.content, id=final_response.id)]
        return [final_response]

    return []


def create_agent_node_handler(
    agent_name: str,
    agent: CompiledStateGraph,
    output_mode: OutputMode = "last_message",
    reflection_executor: Any | None = None,
    recursion_limit: int | None = None,
) -> Callable[[AgentState, Runtime[Context]], Awaitable[AgentState]]:
    """
    Create a handler that wraps an agent subgraph with message filtering.

    This filters messages before passing to the agent and extracts only
    the relevant response, avoiding tool_use/tool_result pairing errors.

    Used by both supervisor and swarm patterns to ensure consistent
    message handling when agents are CompiledStateGraphs.

    Based on langgraph-supervisor-py output_mode pattern.

    Args:
        agent_name: Name of the agent (for logging)
        agent: The compiled agent subgraph
        output_mode: How to extract response ("last_message" or "full_history")
        reflection_executor: Optional background reflection executor for
            automatic memory extraction after each agent turn.

    Returns:
        An async handler function for the workflow node
    """

    async def handler(state: AgentState, runtime: Runtime[Context]) -> AgentState:
        # Filter messages to avoid tool_use/tool_result pairing errors.
        # Passing agent_name preserves the agent's own tool exchanges from
        # prior turns while still hiding peers' tool exchanges.
        original_messages = state.get("messages", [])
        filtered_messages = filter_messages_for_agent(
            original_messages, current_agent_name=agent_name
        )

        logger.trace(
            "Agent receiving filtered messages",
            agent=agent_name,
            filtered_count=len(filtered_messages),
            original_count=len(original_messages),
        )

        # Create state with filtered messages for the agent
        agent_state: AgentState = {
            **state,
            "messages": filtered_messages,
        }

        # Build config with configurable from context for langmem compatibility
        # langmem tools expect user_id to be in config.configurable
        config: dict[str, Any] = {}
        if runtime.context:
            config = {"configurable": runtime.context.model_dump()}
        if recursion_limit is not None:
            config["recursion_limit"] = recursion_limit

        # --- HITL interrupt propagation & state-scoped sub-thread_id ---
        #
        # The handler invokes the subgraph via ainvoke() (not add_node),
        # so the subgraph cannot inherit the parent's checkpointer and
        # needs its own InMemorySaver.  We must:
        #   1. Use a per-task sub-thread_id so the subgraph starts fresh on
        #      each parent-graph step, preventing stale message accumulation.
        #   2. Detect __interrupt__ in the subgraph result and propagate it
        #      to the parent via langgraph_interrupt().
        #   3. On resume (same parent task), the handler re-enters with the
        #      same task_id.  We detect the interrupted subgraph state via
        #      aget_state and propagate using langgraph_interrupt(), which
        #      returns the user's decisions.  We then forward them to the
        #      subgraph via Command(resume=...).
        #
        # ParentCommand handling: because worker subgraphs are invoked via
        # ainvoke() rather than added directly as compiled-graph nodes,
        # LangGraph's ParentCommand exception (raised when a tool returns
        # Command(graph=Command.PARENT)) escapes ainvoke() instead of being
        # caught by the parent pregel engine.  We catch it here and convert
        # it to a plain Command that the parent graph can process.
        try:
            if agent.checkpointer:
                parent_conf = get_config().get("configurable", {})
                task_id: str = parent_conf.get("__pregel_task_id", "default")
                thread_id: str = config.get("configurable", {}).get(
                    "thread_id", "default"
                )
                sub_thread_id: str = f"{thread_id}__sub_{task_id}"
                sub_config: dict[str, Any] = {
                    "configurable": {
                        **config.get("configurable", {}),
                        "thread_id": sub_thread_id,
                    }
                }

                logger.debug(
                    "HITL: Invoking subgraph with scoped thread",
                    agent=agent_name,
                    sub_thread_id=sub_thread_id,
                )

                sub_state = await agent.aget_state(config=sub_config)
                is_interrupted: bool = bool(sub_state and sub_state.next)

                if is_interrupted:
                    interrupts = list(sub_state.interrupts)
                    logger.info(
                        "HITL: Subgraph is interrupted, propagating to parent",
                        agent=agent_name,
                        interrupts_count=len(interrupts),
                    )
                    resume_value: Any = None
                    for intr in interrupts:
                        resume_value = langgraph_interrupt(intr.value)

                    logger.info(
                        "HITL: Resume value received, forwarding to subgraph",
                        agent=agent_name,
                    )
                    result: dict[str, Any] = await agent.ainvoke(
                        Command(resume=resume_value),
                        context=runtime.context,
                        config=sub_config,
                    )
                else:
                    result = await agent.ainvoke(
                        agent_state, context=runtime.context, config=sub_config
                    )
                    if "__interrupt__" in result:
                        interrupts_list: list[Interrupt] = result["__interrupt__"]
                        logger.info(
                            "HITL: Subgraph returned interrupt, propagating",
                            agent=agent_name,
                            interrupts_count=len(interrupts_list),
                        )
                        for intr in interrupts_list:
                            langgraph_interrupt(intr.value)
            else:
                result = await agent.ainvoke(
                    agent_state, context=runtime.context, config=config
                )
        except ParentCommand as pc:
            cmd: Command = pc.args[0]
            logger.debug(
                "Subgraph issued parent command, propagating",
                agent=agent_name,
                goto=cmd.goto,
            )
            return cmd

        # Extract agent response based on output mode
        result_messages = result.get("messages", [])
        response_messages = extract_agent_response(result_messages, output_mode)

        # Defensive fallback: if any AIMessage came back without ``name``
        # set, tag it with the agent's name so ``filter_messages_for_agent``
        # can identify it as own on re-entry. ``create_agent(name=…)``
        # normally sets this already, so this is rarely a no-op. ToolMessages
        # carry the tool's name (not the agent's) and get attributed via
        # tool_call_id pairing in the filter.
        response_messages = [
            msg.model_copy(update={"name": agent_name})
            if isinstance(msg, AIMessage) and not msg.name
            else msg
            for msg in response_messages
        ]

        logger.debug(
            "Agent completed",
            agent=agent_name,
            response_count=len(response_messages),
            total_messages=len(result_messages),
            output_mode=output_mode,
        )

        # Schedule background memory extraction if configured
        if reflection_executor is not None and result_messages:
            try:
                reflection_executor.submit(
                    {"messages": result_messages},
                    config=config or None,
                    after_seconds=0,
                )
                logger.trace(
                    "Background memory extraction scheduled",
                    agent=agent_name,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Failed to schedule background memory extraction",
                    agent=agent_name,
                )

        # Return state update with extracted response
        return {
            **result,
            "messages": response_messages,
        }

    return handler


def create_handoff_tool(
    target_agent_name: str,
    description: str,
    requires: list[str] | None = None,
) -> BaseTool:
    """
    Create a handoff tool that transfers control to another agent.

    The tool returns a Command object with goto to directly route
    to the target agent node in the parent graph.

    Args:
        target_agent_name: The name of the agent to hand off to
        description: Description of what this agent handles
        requires: Optional list of agent names that must have appeared in
            ``state["messages"]`` (as ``AIMessage.name``) before the handoff
            is allowed. When unmet, the tool returns a refusal ``ToolMessage``
            naming the missing prereqs and ``active_agent`` stays unchanged.
            Empty/None means no constraint (existing behavior).

    Returns:
        A tool that triggers a handoff to the target agent via Command
    """
    required_agents: tuple[str, ...] = tuple(requires or ())

    @tool
    def handoff_tool(runtime: ToolRuntime[Context, AgentState]) -> Command:
        """Transfer control to another agent."""
        tool_call_id: str = runtime.tool_call_id
        logger.debug("Handoff to agent", target_agent=target_agent_name)

        # Get the AIMessage that triggered this handoff (required for tool_use/tool_result pairing)
        # LLMs expect tool calls to be paired with their responses, so we must include both
        # the AIMessage containing the tool call and the ToolMessage acknowledging it.
        messages: list[BaseMessage] = runtime.state.get("messages", [])
        triggering_ai_message: AIMessage | None = last_ai_message_with_tool_calls(
            messages
        )

        # Enforce ``requires`` (handoff constraints): refuse if any required
        # agent has not yet produced a tagged AIMessage in this conversation.
        if required_agents:
            called_agents: set[str] = {
                m.name for m in messages if isinstance(m, AIMessage) and m.name
            }
            missing: list[str] = [a for a in required_agents if a not in called_agents]
            if missing:
                logger.debug(
                    "Handoff refused: prerequisites unmet",
                    target_agent=target_agent_name,
                    missing=missing,
                    called=sorted(called_agents),
                )
                refusal_messages: list[BaseMessage] = []
                if triggering_ai_message:
                    refusal_messages.append(triggering_ai_message)
                refusal_messages.append(
                    ToolMessage(
                        content=(
                            f"Cannot hand off to '{target_agent_name}' — "
                            f"requires {list(required_agents)}; called so far: "
                            f"{sorted(called_agents)}. Missing: {missing}. "
                            f"Pick a different handoff or continue working."
                        ),
                        tool_call_id=tool_call_id,
                    )
                )
                # Return-only update: no goto, no active_agent change. Source
                # agent stays in control and sees the refusal in-band.
                return Command(update={"messages": refusal_messages})

        # Build message list with proper pairing
        update_messages: list[BaseMessage] = []
        if triggering_ai_message:
            update_messages.append(triggering_ai_message)
        update_messages.append(
            ToolMessage(
                content=f"Transferred to {target_agent_name}",
                tool_call_id=tool_call_id,
            )
        )

        return Command(
            update={
                "active_agent": target_agent_name,
                "messages": update_messages,
            },
            goto=target_agent_name,
            graph=Command.PARENT,
        )

    # Set the tool name and description
    handoff_tool.name = f"handoff_to_{target_agent_name}"
    handoff_tool.__doc__ = f"Transfer to {target_agent_name}: {description}"
    handoff_tool.description = f"Transfer to {target_agent_name}: {description}"

    return handoff_tool


def get_handoff_description(agent: AgentModel) -> str:
    """
    Get the handoff description for an agent.

    Priority: handoff_prompt > description > default message

    Args:
        agent: The agent to get the handoff description for

    Returns:
        The handoff description string
    """
    return (
        agent.handoff_prompt
        or agent.description
        or f"Handles {agent.name} related tasks and inquiries"
    )


def create_extraction_manager_and_executor(
    memory: MemoryModel | None,
    store: BaseStore | None,
    fallback_model: LanguageModelLike,
    graph_label: str,
) -> tuple[Any | None, Any | None]:
    """
    Build the shared extraction manager and optional reflection executor.

    Both the supervisor and swarm graphs need a single extraction manager
    shared across all nodes. The fallback model differs (the supervisor's
    own model vs. the first worker's model in swarm), so it's parameterised.

    Args:
        memory: The orchestration memory configuration, if any.
        store: The shared LangGraph store, if any.
        fallback_model: Chat model to use when ``memory.extraction.extraction_model``
            is not set.
        graph_label: Human-readable label used only in log messages
            (e.g. ``"supervisor graph"``).

    Returns:
        ``(extraction_manager, reflection_executor)``. Either or both may
        be ``None`` when extraction is not configured or background
        extraction is disabled.
    """
    needs_extraction = (
        memory
        and memory.store
        and memory.extraction
        and store
        and (memory.extraction.background_extraction or memory.extraction.auto_inject)
    )
    if not needs_extraction:
        return None, None

    # Lazy imports avoid an import cycle with dao_ai.memory and dao_ai.nodes.
    from dao_ai.memory.extraction import (
        create_extraction_manager,
        create_reflection_executor,
    )
    from dao_ai.nodes import _build_memory_namespace

    extraction_ns = _build_memory_namespace(memory)
    extraction_model: LanguageModelLike = (
        memory.extraction.extraction_model.as_chat_model()
        if memory.extraction.extraction_model
        else fallback_model
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

    reflection_executor = None
    if memory.extraction.background_extraction:
        reflection_executor = create_reflection_executor(extraction_manager, store)
        logger.info(f"Background memory extraction enabled for {graph_label}")

    return extraction_manager, reflection_executor


def create_orchestration_graph(config: AppConfig) -> CompiledStateGraph:
    """
    Create the main orchestration graph based on the configuration.

    Dispatches to ``supervisor``, ``swarm``, or ``deep_agent`` depending on which
    block is set under ``app.orchestration``. The three modes are mutually
    exclusive (validated at config load time).

    Args:
        config: The application configuration

    Returns:
        A compiled LangGraph state machine
    """
    from dao_ai.orchestration.deep_agent import create_deep_agent_graph
    from dao_ai.orchestration.supervisor import create_supervisor_graph
    from dao_ai.orchestration.swarm import create_swarm_graph

    orchestration: OrchestrationModel = config.app.orchestration
    if orchestration.supervisor:
        return create_supervisor_graph(config)

    if orchestration.swarm:
        return create_swarm_graph(config)

    if orchestration.deep_agent:
        return create_deep_agent_graph(config)

    raise ValueError("No valid orchestration model found in the configuration.")
