from typing import Callable, Sequence

from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph_swarm import create_handoff_tool, create_swarm
from loguru import logger

from retail_ai.config import AgentModel, AppConfig, OrchestrationModel
from retail_ai.nodes import (
    create_agent_node,
    message_validation_node,
    supervisor_node,
)
from retail_ai.state import AgentConfig, AgentState


def route_message_validation(on_success: str) -> Callable:
    def _(state: AgentState) -> str:
        if not state["is_valid_config"]:
            return END
        return on_success

    return _


def _create_supervisor_graph(config: AppConfig) -> CompiledStateGraph:
    logger.debug("Creating supervisor graph")
    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    workflow.add_node("message_validation", message_validation_node(config=config))
    workflow.add_node("supervisor", supervisor_node(config=config))

    agents: Sequence[AgentModel] = config.app.agents
    for agent in agents:
        workflow.add_node(agent.name, create_agent_node(agent=agent))

    workflow.add_conditional_edges(
        "message_validation",
        route_message_validation("supervisor"),
        {
            "supervisor": "supervisor",
            END: END,
        },
    )

    workflow.add_edge("message_validation", "supervisor")

    routes: dict[str, str] = {n: n for n in [agent.name for agent in agents]}
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        routes,
    )

    workflow.set_entry_point("message_validation")

    return workflow.compile()


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
            create_handoff_tool(
                agent_name=handoff_to_agent.name,
                description=f"Ask {handoff_to_agent.name} for help with: "
                + handoff_to_agent.handoff_prompt,
            )
        )
    return handoff_tools


def _create_swarm_graph(config: AppConfig) -> CompiledStateGraph:
    logger.debug("Creating swarm graph")
    agents: list[CompiledStateGraph] = []
    for registered_agent in config.app.agents:
        handoff_tools: Sequence[BaseTool] = _handoffs_for_agent(
            agent=registered_agent, config=config
        )
        agents.append(
            create_agent_node(agent=registered_agent, additional_tools=handoff_tools)
        )

    default_agent: AgentModel = config.app.orchestration.swarm.default_agent
    if isinstance(default_agent, AgentModel):
        default_agent = default_agent.name

    swarm_workflow: StateGraph = create_swarm(
        agents=agents,
        default_active_agent=default_agent,
        state_schema=AgentState,
        config_schema=AgentConfig,
    )

    checkpointer = None  # InMemorySaver()
    store = None  # InMemoryStore()
    swarm_node: CompiledStateGraph = swarm_workflow.compile(
        checkpointer=checkpointer, store=store
    )

    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    workflow.add_node("message_validation", message_validation_node(config=config))
    workflow.add_node("swarm", swarm_node)

    workflow.add_conditional_edges(
        "message_validation",
        route_message_validation("swarm"),
        {
            "swarm": "swarm",
            END: END,
        },
    )

    workflow.add_edge("message_validation", "swarm")

    workflow.set_entry_point("message_validation")

    return workflow.compile()


def create_retail_ai_graph(config: AppConfig) -> CompiledStateGraph:
    orchestration: OrchestrationModel = config.app.orchestration
    if orchestration.supervisor:
        return _create_supervisor_graph(config)

    if orchestration.swarm:
        return _create_swarm_graph(config)

    raise ValueError("No valid orchestration model found in the configuration.")
