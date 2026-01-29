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
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Command
from loguru import logger

from dao_ai.config import AgentModel, AppConfig, OrchestrationModel
from dao_ai.messages import last_ai_message
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


def filter_messages_for_agent(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Filter messages for a worker agent to avoid tool_use/tool_result pairing errors.

    When the supervisor hands off to an agent, the agent should only see:
    - HumanMessage (user queries)
    - AIMessage with content (previous responses, but not tool calls)

    This prevents the agent from seeing orphaned ToolMessages or AIMessages
    with tool_calls that don't belong to the agent's context.

    Args:
        messages: The full message history from parent state

    Returns:
        Filtered messages safe for the agent to process
    """
    filtered: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # Always include user messages
            filtered.append(msg)
        elif isinstance(msg, AIMessage):
            # Include AI messages but strip tool_calls to avoid confusion
            if msg.content and not msg.tool_calls:
                filtered.append(msg)
            elif msg.content and msg.tool_calls:
                # Include content but create clean AIMessage without tool_calls
                filtered.append(AIMessage(content=msg.content, id=msg.id))
        # Skip ToolMessages - they belong to the supervisor's context
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

    Returns:
        An async handler function for the workflow node
    """

    async def handler(state: AgentState, runtime: Runtime[Context]) -> AgentState:
        # Filter messages to avoid tool_use/tool_result pairing errors
        original_messages = state.get("messages", [])
        filtered_messages = filter_messages_for_agent(original_messages)

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

        # Invoke the agent with both context and config
        result: AgentState = await agent.ainvoke(
            agent_state, context=runtime.context, config=config
        )

        # Extract agent response based on output mode
        result_messages = result.get("messages", [])
        response_messages = extract_agent_response(result_messages, output_mode)

        logger.debug(
            "Agent completed",
            agent=agent_name,
            response_count=len(response_messages),
            total_messages=len(result_messages),
            output_mode=output_mode,
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
) -> BaseTool:
    """
    Create a handoff tool that transfers control to another agent.

    The tool returns a Command object with goto to directly route
    to the target agent node in the parent graph.

    Args:
        target_agent_name: The name of the agent to hand off to
        description: Description of what this agent handles

    Returns:
        A tool that triggers a handoff to the target agent via Command
    """

    @tool
    def handoff_tool(runtime: ToolRuntime[Context, AgentState]) -> Command:
        """Transfer control to another agent."""
        tool_call_id: str = runtime.tool_call_id
        logger.debug("Handoff to agent", target_agent=target_agent_name)

        # Get the AIMessage that triggered this handoff (required for tool_use/tool_result pairing)
        # LLMs expect tool calls to be paired with their responses, so we must include both
        # the AIMessage containing the tool call and the ToolMessage acknowledging it.
        messages: list[BaseMessage] = runtime.state.get("messages", [])
        last_ai_message: AIMessage | None = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                last_ai_message = msg
                break

        # Build message list with proper pairing
        update_messages: list[BaseMessage] = []
        if last_ai_message:
            update_messages.append(last_ai_message)
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


def create_orchestration_graph(config: AppConfig) -> CompiledStateGraph:
    """
    Create the main orchestration graph based on the configuration.

    This factory function creates either a supervisor or swarm graph
    depending on the configuration.

    Args:
        config: The application configuration

    Returns:
        A compiled LangGraph state machine
    """
    from dao_ai.orchestration.supervisor import create_supervisor_graph
    from dao_ai.orchestration.swarm import create_swarm_graph

    orchestration: OrchestrationModel = config.app.orchestration
    if orchestration.supervisor:
        return create_supervisor_graph(config)

    if orchestration.swarm:
        return create_swarm_graph(config)

    raise ValueError("No valid orchestration model found in the configuration.")
