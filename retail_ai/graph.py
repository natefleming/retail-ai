from typing import Sequence

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from retail_ai.nodes import (code_question_node, generic_question_node,
                             genie_question_node, route_question_node,
                             summarize_response_node,
                             vector_search_question_node)
from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable


def create_graph(
    model_name: str,
    endpoint: str,
    index_name: str,
    primary_key: str,
    text_column: str,
    doc_uri: str,
    columns: Sequence[str],
    search_parameters: dict[str, str],
    space_id: str
) -> CompiledStateGraph:

    route_question: AgentCallable = route_question_node(model_name=model_name)
    genie_question: AgentCallable = genie_question_node(model_name=model_name, space_id=space_id)
    code_question: AgentCallable = code_question_node(model_name=model_name)
    vector_search_question: AgentCallable = vector_search_question_node(
        model_name=model_name,
        index_name=index_name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        columns=columns,
        search_parameters=search_parameters,
    )
    generic_question: AgentCallable = generic_question_node(model_name=model_name)
    summarize_response: AgentCallable = summarize_response_node(model_name=model_name)

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