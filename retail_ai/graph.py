from typing import (
    Callable,
    Sequence, 
    Literal, 
    Optional, 
    Any, 
    Generator
)

from langchain_core.messages import BaseMessage
from langchain_core.documents.base import Document

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

import mlflow
from mlflow.models import ModelConfig

from retail_ai.state import AgentState, AgentConfig
from retail_ai.models import LangGraphChatAgent, create_agent
from retail_ai.nodes import (
    route_question_node,
    retrieve_context_node,
    code_question_node,
    vector_search_question_node,
    generic_question_node,
    genie_question_node,
    summarize_response_node,
)
from retail_ai.types import AgentCallable


def create_graph(
    model_name: str,
    endpoint: str,
    index_name: str,
    search_parameters: dict[str, str],
    columns: Sequence[str],
    space_id: str
) -> CompiledStateGraph:

    route_question: AgentCallable = route_question_node(model_name)
    genie_question: AgentCallable = genie_question_node(model_name=model_name, space_id=space_id)
    code_question: AgentCallable = code_question_node(model_name)
    vector_search_question: AgentCallable = vector_search_question_node(
        model_name=model_name,
        index_name=index_name,
        columns=columns,
    )
    generic_question: AgentCallable = generic_question_node(model_name)
    summarize_response: AgentCallable = summarize_response_node(model_name)

    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    workflow.add_node("router", route_question)
    workflow.add_node("code_agent", code_question)
    workflow.add_node("generic_agent", generic_question)
    workflow.add_node("genie_agent", genie_question)
    #workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("vector_search_agent", vector_search_question)
    workflow.add_node("summarize_agent", summarize_response)

    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "code": "code_agent",
            "general": "generic_agent",
            "genie": "genie_agent",
            "vector_search": "vector_search_agent",
        }
    )

    workflow.set_entry_point("router")
    workflow.add_edge("code_agent", "summarize_agent")
    workflow.add_edge("generic_agent", "summarize_agent")
    workflow.add_edge("genie_agent", "summarize_agent")
    #workflow.add_edge("retrieve_context", "vector_search_agent")
    workflow.add_edge("vector_search_agent", "summarize_agent")
    workflow.set_finish_point("summarize_agent")
 
    
    return workflow.compile()