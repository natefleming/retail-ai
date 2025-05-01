from typing import Sequence

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.state import CompiledStateGraph
from mlflow.models import ModelConfig
from langgraph_reflection import create_reflection_graph
from retail_ai.nodes import (
    router_node, 
    general_node,
    inventory_node,
    product_node,
    comparison_node,
    orders_node,    
    diy_node,
    recommendation_node,
    message_validation_node,
    judge_node,
)
from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable


def with_judge(graph: CompiledStateGraph, judge: CompiledStateGraph) -> CompiledStateGraph:
    return create_reflection_graph(graph, judge, state_schema=AgentState, config_schema=AgentConfig).compile()


def create_ace_arma_graph(model_config: ModelConfig) -> CompiledStateGraph:

    judge: CompiledStateGraph = (
        StateGraph(AgentState, config_schema=AgentConfig)
        .add_node("judge", judge_node(model_config=model_config))
        .add_edge(START, "judge")
        .add_edge("judge", END)
        .compile()
    )

    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    workflow.add_node("message_validation", message_validation_node(model_config=model_config))
    workflow.add_node("router", router_node(model_config=model_config))
    workflow.add_node("general", with_judge(general_node(model_config=model_config), judge=judge))
    workflow.add_node("recommendation", with_judge(recommendation_node(model_config=model_config), judge=judge))
    workflow.add_node("inventory", with_judge(inventory_node(model_config=model_config), judge=judge))
    workflow.add_node("product", with_judge(product_node(model_config=model_config), judge=judge))
    workflow.add_node("orders", with_judge(orders_node(model_config=model_config), judge=judge))
    workflow.add_node("diy", with_judge(diy_node(model_config=model_config), judge=judge))
    workflow.add_node("comparison", with_judge(comparison_node(model_config=model_config), judge=judge))

    #reflection_app = create_reflection_graph(diy_node(model_config=model_config), judge_graph)


    workflow.add_conditional_edges(
        "message_validation",
        lambda state: state["is_valid_config"],
        {
            True: "router",
            False: END,
        }
    )

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


