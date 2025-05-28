from typing import Sequence

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from retail_ai.config import AgentModel, AppConfig, OrchestrationModel
from retail_ai.messages import has_image
from retail_ai.nodes import (
    create_agent_node,
    message_validation_node,
    process_images_node,
    supervisor_node,
)
from retail_ai.state import AgentConfig, AgentState



def route_message_validation(state: AgentState) -> str:
    if not state["is_valid_config"]:
        return END
    if has_image(state["messages"]):
        return "process_images"
    return "supervisor"


def _create_supervisor_graph(config: AppConfig) -> CompiledStateGraph:
    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    workflow.add_node("message_validation", message_validation_node(config=config))
    workflow.add_node("process_images", process_images_node(config=config))
    workflow.add_node("supervisor", supervisor_node(config=config))

    agents: Sequence[AgentModel] = config.app.agents
    for agent in agents:
        workflow.add_node(agent.name, create_agent_node(agent=agent))

    workflow.add_conditional_edges(
        "message_validation",
        route_message_validation,
        {
            "supervisor": "supervisor",
            "process_images": "process_images",
            END: END,
        },
    )

    workflow.add_edge("process_images", "supervisor")

    routes: dict[str, str] = {n: n for n in [agent.name for agent in agents]}
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        routes,
    )

    workflow.set_entry_point("message_validation")

    return workflow.compile()


def create_retail_ai_graph(config: AppConfig) -> CompiledStateGraph:
    orchestration: OrchestrationModel = config.app.orchestration
    if orchestration.supervisor:
        return _create_supervisor_graph(config)
    
    if orchestration.swarm:
        raise NotImplementedError("Swarm orchestration is not implemented yet.")
    
    raise ValueError("No valid orchestration model found in the configuration.")
 

