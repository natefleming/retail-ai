from typing import Sequence

from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph_supervisor import create_handoff_tool as supervisor_handoff_tool
from langgraph_supervisor import create_supervisor
from langgraph_swarm import create_handoff_tool as swarm_handoff_tool
from langgraph_swarm import create_swarm
from langmem import create_manage_memory_tool, create_search_memory_tool
from loguru import logger

from dao_ai.config import (
    AgentModel,
    AppConfig,
    OrchestrationModel,
    SupervisorModel,
    SwarmModel,
)
from dao_ai.nodes import (
    create_agent_node,
    message_hook_node,
)
from dao_ai.prompts import make_prompt
from dao_ai.state import Context, IncomingState, OutgoingState, SharedState


def route_message(state: SharedState) -> str:
    if not state["is_valid"]:
        return END
    return "orchestration"


def _handoffs_for_agent(agent: AgentModel, config: AppConfig) -> Sequence[BaseTool]:
    handoff_tools: list[BaseTool] = []

    handoffs: dict[str, Sequence[AgentModel | str]] = (
        config.app.orchestration.swarm.handoffs or {}
    )
    agent_handoffs: Sequence[AgentModel | str] = handoffs.get(agent.name)
    if agent_handoffs is None:
        agent_handoffs = config.app.agents

    for handoff_to_agent in agent_handoffs:
        if isinstance(handoff_to_agent, str):
            handoff_to_agent = next(
                iter(config.find_agents(lambda a: a.name == handoff_to_agent)), None
            )

        if handoff_to_agent is None:
            logger.warning(
                f"Handoff agent {handoff_to_agent} not found in configuration for agent {agent.name}"
            )
            continue
        if agent.name == handoff_to_agent.name:
            continue
        logger.debug(
            f"Creating handoff tool from agent {agent.name} to {handoff_to_agent.name}"
        )
        handoff_tools.append(
            swarm_handoff_tool(
                agent_name=handoff_to_agent.name,
                description=f"Ask {handoff_to_agent.name} for help with: "
                + handoff_to_agent.handoff_prompt,
            )
        )
    return handoff_tools


def _create_supervisor_graph(config: AppConfig) -> CompiledStateGraph:
    logger.debug("Creating supervisor graph")
    agents: list[CompiledStateGraph] = []
    tools: Sequence[BaseTool] = []
    for registered_agent in config.app.agents:
        agents.append(
            create_agent_node(
                app=config.app, agent=registered_agent, additional_tools=[]
            )
        )
        tools.append(
            supervisor_handoff_tool(
                agent_name=registered_agent.name,
                description=registered_agent.handoff_prompt,
            )
        )

    orchestration: OrchestrationModel = config.app.orchestration
    supervisor: SupervisorModel = orchestration.supervisor

    store: BaseStore = None
    if orchestration.memory and orchestration.memory.store:
        store = orchestration.memory.store.as_store()
        logger.debug(f"Using memory store: {store}")
        namespace: tuple[str, ...] = ("memory",)

        if orchestration.memory.store.namespace:
            namespace = namespace + (orchestration.memory.store.namespace,)
            logger.debug(f"Memory store namespace: {namespace}")
            tools += [
                create_manage_memory_tool(namespace=namespace),
                create_search_memory_tool(namespace=namespace),
            ]

    checkpointer: BaseCheckpointSaver = None
    if orchestration.memory and orchestration.memory.checkpointer:
        checkpointer = orchestration.memory.checkpointer.as_checkpointer()
        logger.debug(f"Using checkpointer: {checkpointer}")

    prompt: str = supervisor.prompt

    model: LanguageModelLike = supervisor.model.as_chat_model()
    supervisor_workflow: StateGraph = create_supervisor(
        supervisor_name="supervisor",
        prompt=make_prompt(base_system_prompt=prompt),
        agents=agents,
        model=model,
        tools=tools,
        state_schema=SharedState,
        config_schema=RunnableConfig,
        output_mode="last_message",
        add_handoff_messages=False,
        add_handoff_back_messages=False,
        context_schema=Context,
        # output_mode="full",
        # add_handoff_messages=True,
        # add_handoff_back_messages=True,
    )

    supervisor_node: CompiledStateGraph = supervisor_workflow.compile(
        checkpointer=checkpointer, store=store
    )

    workflow: StateGraph = StateGraph(
        SharedState,
        input=IncomingState,
        output=OutgoingState,
        context_schema=Context,
    )

    workflow.add_node("message_hook", message_hook_node(config=config))

    workflow.add_node("orchestration", supervisor_node)
    workflow.add_conditional_edges(
        "message_hook",
        route_message,
        {
            "orchestration": "orchestration",
            END: END,
        },
    )
    workflow.set_entry_point("message_hook")

    return workflow.compile(checkpointer=checkpointer, store=store)


def _create_swarm_graph(config: AppConfig) -> CompiledStateGraph:
    logger.debug("Creating swarm graph")
    agents: list[CompiledStateGraph] = []
    for registered_agent in config.app.agents:
        handoff_tools: Sequence[BaseTool] = _handoffs_for_agent(
            agent=registered_agent, config=config
        )
        agents.append(
            create_agent_node(
                app=config.app, agent=registered_agent, additional_tools=handoff_tools
            )
        )

    orchestration: OrchestrationModel = config.app.orchestration
    swarm: SwarmModel = orchestration.swarm

    store: BaseStore = None
    if orchestration.memory and orchestration.memory.store:
        store = orchestration.memory.store.as_store()
        logger.debug(f"Using memory store: {store}")

    checkpointer: BaseCheckpointSaver = None
    if orchestration.memory and orchestration.memory.checkpointer:
        checkpointer = orchestration.memory.checkpointer.as_checkpointer()
        logger.debug(f"Using checkpointer: {checkpointer}")

    default_agent: AgentModel = swarm.default_agent
    if isinstance(default_agent, AgentModel):
        default_agent = default_agent.name

    swarm_workflow: StateGraph = create_swarm(
        agents=agents,
        default_active_agent=default_agent,
        state_schema=SharedState,
        context_schema=Context,
    )

    swarm_node: CompiledStateGraph = swarm_workflow.compile(
        checkpointer=checkpointer, store=store
    )

    workflow: StateGraph = StateGraph(
        SharedState,
        input=IncomingState,
        output=OutgoingState,
        context_schema=Context,
    )

    workflow.add_node("message_hook", message_hook_node(config=config))
    workflow.add_node("orchestration", swarm_node)

    workflow.add_conditional_edges(
        "message_hook",
        route_message,
        {
            "orchestration": "orchestration",
            END: END,
        },
    )

    workflow.set_entry_point("message_hook")

    return workflow.compile(checkpointer=checkpointer, store=store)


def create_dao_ai_graph(config: AppConfig) -> CompiledStateGraph:
    orchestration: OrchestrationModel = config.app.orchestration
    if orchestration.supervisor:
        return _create_supervisor_graph(config)

    if orchestration.swarm:
        return _create_swarm_graph(config)

    raise ValueError("No valid orchestration model found in the configuration.")
