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
    GenieAgentModel,
    MemoryModel,
    OrchestrationModel,
    PromptModel,
    SupervisorModel,
)
from dao_ai.messages import last_ai_message_with_tool_calls
from dao_ai.middleware.base import AgentMiddleware
from dao_ai.middleware.core import create_factory_middleware
from dao_ai.nodes import create_agent_node
from dao_ai.orchestration import (
    SUPERVISOR_NODE,
    create_agent_node_handler,
    create_checkpointer,
    create_extraction_manager_and_executor,
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
        triggering_ai_message: AIMessage | None = last_ai_message_with_tool_calls(
            messages
        )

        # Build message list with proper pairing
        update_messages: list[BaseMessage] = []
        if triggering_ai_message:
            update_messages.append(triggering_ai_message)
        update_messages.append(
            ToolMessage(
                content=f"Task completed: {summary}",
                tool_call_id=tool_call_id,
            )
        )

        return Command(
            update={
                "active_agent": SUPERVISOR_NODE,
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

    effective_prompt: str | PromptModel | None = supervisor.prompt

    # Append memory tool instructions to the prompt when memory tools are present
    if has_memory_tools and effective_prompt is not None:
        from dao_ai.nodes import MEMORY_TOOL_INSTRUCTIONS

        if isinstance(effective_prompt, PromptModel):
            effective_prompt = effective_prompt.template + MEMORY_TOOL_INSTRUCTIONS
        else:
            effective_prompt = effective_prompt + MEMORY_TOOL_INSTRUCTIONS
        logger.debug("Memory tool instructions appended to supervisor prompt")

    # Get the prompt as middleware (always returns AgentMiddleware or None)
    prompt_middleware: LangchainAgentMiddleware | None = make_prompt(effective_prompt)

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

    # Reject worker agents that would collide with the reserved supervisor
    # node name. The collision is silent at graph compile time but produces
    # confusing routing failures at runtime, so fail fast at config time.
    colliding_agents: list[str] = [
        a.name for a in config.app.agents if a.name == SUPERVISOR_NODE
    ]
    if colliding_agents:
        raise ValueError(
            f"Worker agent name(s) {colliding_agents!r} collide with the "
            f"reserved supervisor node name {SUPERVISOR_NODE!r}. Rename the "
            f"agent(s) in config.app.agents."
        )

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
    memory: MemoryModel | None = orchestration.memory
    extraction_manager, reflection_executor = create_extraction_manager_and_executor(
        memory=memory,
        store=store,
        fallback_model=supervisor_config.model.as_chat_model(),
        graph_label="supervisor graph",
    )

    # extraction_manager is non-None iff create_extraction_manager_and_executor
    # determined extraction is needed; use it as the gate for downstream wiring.
    if (
        extraction_manager
        and memory
        and memory.extraction
        and memory.extraction.auto_inject
        and memory.extraction.supervisor_auto_inject
    ):
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
    elif (
        extraction_manager
        and memory
        and memory.extraction
        and memory.extraction.auto_inject
    ):
        logger.info(
            "Memory context injection skipped for supervisor (supervisor_auto_inject=False)"
        )

    # Add OBO model middleware when supervisor LLM uses on-behalf-of-user authentication
    if supervisor_config.model.on_behalf_of_user:
        from dao_ai.middleware.obo import OBOModelMiddleware

        middlewares.append(OBOModelMiddleware(llm_model=supervisor_config.model))
        logger.info(
            "OBO model middleware enabled for supervisor",
            model=supervisor_config.model.name,
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
    agent_recursion_limits: dict[str, int | None] = {}
    # Agents marked ``internal: true`` in config. Their AIMessage outputs
    # are hidden from non-internal (customer-facing) agents' view of
    # shared history (context-leakage prevention).
    internal_agents: frozenset[str] = frozenset(
        a.name for a in config.app.agents if a.internal
    )
    if internal_agents:
        logger.debug(
            "Supervisor pattern internal agents (outputs hidden from non-internal peers)",
            internal_agents=sorted(internal_agents),
        )
    for registered_agent in config.app.agents:
        # Every worker gets a handoff back to the supervisor — except a
        # Genie-brain worker (GenieAgentModel), whose model runs its own tool
        # loop server-side and never calls a client tool. Its turn ends when it
        # answers, which is the only way control ever leaves that worker anyway,
        # so the tool would be dead weight in its graph and its logs.
        additional_tools: list[BaseTool] = (
            []
            if isinstance(registered_agent.model, GenieAgentModel)
            else [_create_handoff_back_to_supervisor_tool()]
        )

        agent_subgraph: CompiledStateGraph = create_agent_node(
            agent=registered_agent,
            memory=memory,
            store=store,
            chat_history=config.app.chat_history,
            additional_tools=additional_tools,
            extraction_manager=extraction_manager,
            checkpointer=checkpointer,
        )
        agent_subgraphs[registered_agent.name] = agent_subgraph
        agent_recursion_limits[registered_agent.name] = registered_agent.recursion_limit
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
            output_mode=orchestration.output_mode,
            reflection_executor=reflection_executor,
            recursion_limit=agent_recursion_limits.get(agent_name),
            internal_agents=internal_agents,
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
