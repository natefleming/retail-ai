"""
Supervisor pattern for multi-agent orchestration.

The supervisor pattern uses a central supervisor agent that coordinates
specialized worker agents. The supervisor hands off control to agents
who then control the conversation. Agents can hand back to the supervisor
when done.

Based on: https://github.com/langchain-ai/langgraph-supervisor-py
"""

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware as LangchainAgentMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command
from loguru import logger

from dao_ai.config import (
    AppConfig,
    MemoryModel,
    OrchestrationModel,
    PromptModel,
    SupervisorModel,
)
from dao_ai.middleware.base import AgentMiddleware
from dao_ai.middleware.core import create_factory_middleware
from dao_ai.nodes import create_agent_node
from dao_ai.orchestration import (
    SUPERVISOR_NODE,
    create_agent_node_handler,
    create_checkpointer,
    create_handoff_tool,
    create_store,
    get_handoff_description,
)
from dao_ai.prompts import make_prompt
from dao_ai.state import AgentState, Context
from dao_ai.tools import create_tools
from dao_ai.tools.memory import create_manage_memory_tool, create_search_memory_tool


def _create_handoff_back_to_supervisor_tool() -> BaseTool:
    """
    Create a tool for agents to hand control back to the supervisor.

    This is used in the supervisor pattern when an agent has completed
    its task and wants to return control to the supervisor for further
    coordination or to complete the conversation.

    Returns:
        A tool that routes back to the supervisor node
    """

    @tool
    def handoff_to_supervisor(
        summary: str,
        runtime: ToolRuntime[Context, AgentState],
    ) -> Command:
        """
        Hand control back to the supervisor.

        Use this when you have completed your task and want to return
        control to the supervisor for further coordination.

        Args:
            summary: A brief summary of what was accomplished
        """
        tool_call_id: str = runtime.tool_call_id
        logger.debug("Agent handing back to supervisor", summary_preview=summary[:100])

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
                content=f"Task completed: {summary}",
                tool_call_id=tool_call_id,
            )
        )

        return Command(
            update={
                "messages": update_messages,
            },
            goto=SUPERVISOR_NODE,
            graph=Command.PARENT,
        )

    return handoff_to_supervisor


def _create_supervisor_agent(
    config: AppConfig,
    tools: list[BaseTool],
    handoff_tools: list[BaseTool],
    middlewares: list[AgentMiddleware],
    has_memory_tools: bool = False,
) -> CompiledStateGraph:
    """
    Create a supervisor agent with handoff tools for each worker agent.

    The supervisor coordinates worker agents by handing off control.
    Worker agents take over the conversation and can hand back to
    the supervisor when done.

    Args:
        config: Application configuration
        tools: Additional tools for the supervisor (e.g., memory tools)
        handoff_tools: Handoff tools to route to worker agents
        middlewares: Middleware to apply to the supervisor
        has_memory_tools: Whether memory tools are included in tools

    Returns:
        Compiled supervisor agent
    """
    orchestration: OrchestrationModel = config.app.orchestration
    supervisor: SupervisorModel = orchestration.supervisor

    all_tools: list[BaseTool] = list(tools) + list(handoff_tools)

    model: LanguageModelLike = supervisor.model.as_chat_model()

    # Capture the original PromptModel reference before any string conversion
    # so it can be forwarded for MLflow trace linking.
    effective_prompt: str | PromptModel | None = supervisor.prompt
    prompt_model_ref: PromptModel | None = (
        effective_prompt if isinstance(effective_prompt, PromptModel) else None
    )

    # Append memory tool instructions to the prompt when memory tools are present
    if has_memory_tools and effective_prompt is not None:
        from dao_ai.nodes import MEMORY_TOOL_INSTRUCTIONS

        if isinstance(effective_prompt, PromptModel):
            effective_prompt = effective_prompt.template + MEMORY_TOOL_INSTRUCTIONS
        else:
            effective_prompt = effective_prompt + MEMORY_TOOL_INSTRUCTIONS
        logger.debug("Memory tool instructions appended to supervisor prompt")

    # Get the prompt as middleware (always returns AgentMiddleware or None)
    prompt_middleware: LangchainAgentMiddleware | None = make_prompt(
        effective_prompt, prompt_model=prompt_model_ref
    )

    # Add prompt middleware at the beginning for priority
    if prompt_middleware is not None:
        middlewares.insert(0, prompt_middleware)

    # Create the supervisor agent
    # Handoff tools route to worker agents in the parent workflow graph
    supervisor_agent: CompiledStateGraph = create_agent(
        name=SUPERVISOR_NODE,
        model=model,
        tools=all_tools,
        middleware=middlewares,
        state_schema=AgentState,
        context_schema=Context,
    )

    return supervisor_agent


def create_supervisor_graph(config: AppConfig) -> CompiledStateGraph:
    """
    Create a supervisor-based multi-agent system using handoffs.

    This implements a supervisor pattern where:
    1. Supervisor receives user input and decides which agent to hand off to
    2. Agent takes control of the conversation and interacts with user
    3. Agent can hand back to supervisor or complete the task

    The supervisor and all worker agents are nodes in a workflow graph.
    Handoff tools use Command(goto=..., graph=Command.PARENT) to route
    between nodes.

    Args:
        config: The application configuration

    Returns:
        A compiled LangGraph state machine

    Based on: https://github.com/langchain-ai/langgraph-supervisor-py
    """
    orchestration: OrchestrationModel = config.app.orchestration
    supervisor_config: SupervisorModel = orchestration.supervisor

    logger.info(
        "Creating supervisor graph",
        pattern="handoff",
        agents_count=len(config.app.agents),
    )

    # Create handoff tools for supervisor to route to agents
    handoff_tools: list[BaseTool] = []
    for registered_agent in config.app.agents:
        description: str = get_handoff_description(registered_agent)
        handoff_tool: BaseTool = create_handoff_tool(
            target_agent_name=registered_agent.name,
            description=description,
        )
        handoff_tools.append(handoff_tool)
        logger.debug("Created handoff tool for supervisor", agent=registered_agent.name)

    # Create supervisor's own tools (e.g., memory tools)
    logger.debug(
        "Creating tools for supervisor", tools_count=len(supervisor_config.tools)
    )
    supervisor_tools: list[BaseTool] = list(create_tools(supervisor_config.tools))

    # Create middleware from configuration
    # All middleware factories return list[AgentMiddleware] for composability
    middlewares: list[AgentMiddleware] = []

    for middleware_config in supervisor_config.middleware:
        logger.trace(
            "Creating middleware for supervisor",
            middleware_name=middleware_config.name,
        )
        middleware: LangchainAgentMiddleware = create_factory_middleware(
            function_name=middleware_config.name,
            args=middleware_config.args,
        )
        middlewares.append(middleware)
        logger.debug(
            "Created supervisor middleware",
            middleware=middleware_config.name,
        )

    # Always apply tool-call-id sanitizer so models that omit IDs don't crash
    from dao_ai.middleware.tool_call_id_sanitizer import ToolCallIdSanitizerMiddleware

    middlewares.append(ToolCallIdSanitizerMiddleware())

    # Set up memory store and checkpointer
    store: BaseStore | None = create_store(orchestration)
    checkpointer: BaseCheckpointSaver | None = create_checkpointer(orchestration)

    # Add memory tools if store is configured with namespace
    has_memory_tools: bool = False
    if (
        orchestration.memory
        and orchestration.memory.store
        and orchestration.memory.store.namespace
    ):
        namespace: tuple[str, ...] = ("memory", orchestration.memory.store.namespace)
        logger.debug("Memory store namespace configured", namespace=namespace)
        supervisor_tools += [
            create_manage_memory_tool(namespace=namespace, store=store),
            create_search_memory_tool(namespace=namespace, store=store),
        ]
        has_memory_tools = True

    # Set up shared extraction manager and background reflection executor.
    # A single extraction manager is shared across the supervisor and all
    # worker agents to avoid creating redundant model instances.
    extraction_manager = None
    reflection_executor = None
    memory: MemoryModel | None = orchestration.memory
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
            else supervisor_config.model.as_chat_model()
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
            logger.info("Background memory extraction enabled for supervisor graph")

    if needs_extraction and extraction_manager and memory.extraction.auto_inject:
        from dao_ai.middleware.memory_context import MemoryContextMiddleware

        memory_middleware = MemoryContextMiddleware(
            manager=extraction_manager,
            limit=memory.extraction.auto_inject_limit,
        )
        middlewares.append(memory_middleware)
        logger.info(
            "Memory context injection enabled for supervisor",
            auto_inject_limit=memory.extraction.auto_inject_limit,
        )

    # Create the supervisor agent
    supervisor_agent: CompiledStateGraph = _create_supervisor_agent(
        config=config,
        tools=supervisor_tools,
        handoff_tools=handoff_tools,
        middlewares=middlewares,
        has_memory_tools=has_memory_tools,
    )

    # Create worker agent subgraphs
    # Each worker gets a handoff_to_supervisor tool to return control
    agent_subgraphs: dict[str, CompiledStateGraph] = {}
    for registered_agent in config.app.agents:
        # Create handoff back to supervisor tool
        supervisor_handoff: BaseTool = _create_handoff_back_to_supervisor_tool()

        # Create the worker agent with handoff back to supervisor
        agent_subgraph: CompiledStateGraph = create_agent_node(
            agent=registered_agent,
            memory=memory,
            store=store,
            chat_history=config.app.chat_history,
            additional_tools=[supervisor_handoff],
            extraction_manager=extraction_manager,
            checkpointer=checkpointer,
        )
        agent_subgraphs[registered_agent.name] = agent_subgraph
        logger.debug("Created worker agent subgraph", agent=registered_agent.name)

    # Build the workflow graph
    # All agents are nodes, handoffs route between them via Command
    workflow: StateGraph = StateGraph(
        AgentState,
        input=AgentState,
        output=AgentState,
        context_schema=Context,
    )

    # Add supervisor node
    workflow.add_node(SUPERVISOR_NODE, supervisor_agent)

    # Add worker agent nodes with message filtering handlers
    for agent_name, agent_subgraph in agent_subgraphs.items():
        handler = create_agent_node_handler(
            agent_name=agent_name,
            agent=agent_subgraph,
            output_mode="last_message",
            reflection_executor=reflection_executor,
        )
        workflow.add_node(agent_name, handler)

    # Supervisor is the entry point
    workflow.set_entry_point(SUPERVISOR_NODE)

    compiled: CompiledStateGraph = workflow.compile(
        checkpointer=checkpointer, store=store
    )
    logger.info(
        "Supervisor graph compiled successfully",
        nodes=list(agent_subgraphs.keys()),
        has_checkpointer=checkpointer is not None,
        has_store=store is not None,
    )
    return compiled
