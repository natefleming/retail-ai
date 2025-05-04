
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.models import ModelConfig

from retail_ai.nodes import (comparison_node, diy_node, general_node,
                             inventory_node, message_validation_node,
                             orders_node, product_node, recommendation_node,
                             router_node, process_images_node)
from retail_ai.state import AgentConfig, AgentState
from retail_ai.messages import has_image


def route_message_validation(state: AgentState) -> str:
    if not state["is_valid_config"]:
        return END
    if has_image(state["messages"]):
        return "process_images"
    return "router"

def create_ace_arma_graph(model_config: ModelConfig) -> CompiledStateGraph:

    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    workflow.add_node("message_validation", message_validation_node(model_config=model_config))
    workflow.add_node("process_images", process_images_node(model_config=model_config))
    workflow.add_node("router", router_node(model_config=model_config))
    workflow.add_node("general", general_node(model_config=model_config))
    workflow.add_node("recommendation", recommendation_node(model_config=model_config))
    workflow.add_node("inventory", inventory_node(model_config=model_config))
    workflow.add_node("product", product_node(model_config=model_config))
    workflow.add_node("orders", orders_node(model_config=model_config))
    workflow.add_node("diy", diy_node(model_config=model_config))
    workflow.add_node("comparison", comparison_node(model_config=model_config))

    workflow.add_conditional_edges(
        "message_validation",
        route_message_validation,
        {
            "router": "router",
            "process_images": "process_images",
            END: END,
        }
    )

    workflow.add_edge("process_images", "router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],  
        {
            "general": "general",
            "recommendation": "recommendation",
            "inventory": "inventory",
            "product": "product",
            "orders": "orders",
            "diy": "diy",
            "comparison": "comparison",
        }
    )

    workflow.set_entry_point("message_validation")
    

    return workflow.compile()


