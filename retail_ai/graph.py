from typing import Any, Sequence

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.models import ModelConfig

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


def create_retail_ai_graph(model_config: ModelConfig) -> CompiledStateGraph:
    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    workflow.add_node(
        "message_validation", message_validation_node(model_config=model_config)
    )
    workflow.add_node("process_images", process_images_node(model_config=model_config))
    workflow.add_node("supervisor", supervisor_node(model_config=model_config))

    agents: Sequence[dict[str, Any]] = model_config.get("app").get("agents")
    agent_names: Sequence[str] = [agent["name"] for agent in agents]
    for name in agent_names:
        workflow.add_node(name, create_agent_node(name=name, model_config=model_config))

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

    routes: dict[str, str] = {n: n for n in agent_names}
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        routes,
    )

    workflow.set_entry_point("message_validation")

    return workflow.compile()
