from typing import Sequence

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.models import ModelConfig

from retail_ai.nodes import (
    router_node, 
    general_node,
    inventory_node,
    product_node,
    comparison_node,
    orders_node,    
    diy_node,
    recommendation_node,
)
from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable


def create_ace_arma_graph(model_config: ModelConfig) -> CompiledStateGraph:
    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    workflow.add_node("router", router_node(model_config=model_config))
    workflow.add_node("general", general_node(model_config=model_config))
    workflow.add_node("recommendation", recommendation_node(model_config=model_config))
    workflow.add_node("inventory", inventory_node(model_config=model_config))
    workflow.add_node("product", product_node(model_config=model_config))
    workflow.add_node("orders", orders_node(model_config=model_config))
    workflow.add_node("diy", diy_node(model_config=model_config))
    workflow.add_node("comparison", comparison_node(model_config=model_config))


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

    workflow.set_entry_point("router")
    

    return workflow.compile()


