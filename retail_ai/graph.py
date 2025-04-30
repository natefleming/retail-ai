from typing import Sequence

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from retail_ai.nodes import (code_question_node, generic_question_node,
                             genie_question_node, route_question_node,
                             summarize_response_node,
                             vector_search_question_node)
from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable


def create_retail_ai_graph(
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
    """
    Create a multi-agent orchestration graph for retail AI question answering.
    
    This function constructs a directed graph of specialized agents that work together
    to handle different types of retail-related queries. The workflow uses a router
    to determine the appropriate agent for each query and a summarizer to generate
    the final response.
    
    Args:
        model_name: Name of the foundation model to use for all agents
        endpoint: Endpoint URL for model serving
        index_name: Name of the vector search index for retrieval
        primary_key: Primary key field in the vector index
        text_column: Column containing the text content in the vector index
        doc_uri: Document URI field in the vector index
        columns: Specific columns to retrieve from the vector index
        search_parameters: Additional parameters for vector search configuration
        space_id: Databricks workspace ID for Genie integration
        
    Returns:
        A compiled LangGraph state machine that orchestrates the multi-agent workflow
    """
    # Initialize specialized agent nodes for different question types
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

    # Create the workflow graph with the defined state schema
    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    # Add all nodes to the graph
    workflow.add_node("router", route_question)  # Determines which agent should handle the query
    workflow.add_node("code_agent", code_question)  # Handles code-related questions
    workflow.add_node("generic_agent", generic_question)  # Handles general retail questions
    workflow.add_node("genie_agent", genie_question)  # Handles database queries via Genie
    # Commented out retrieve_context node - may be used in future versions
    #workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("vector_search_agent", vector_search_question)  # Searches product database
    workflow.add_node("summarize_agent", summarize_response)  # Generates final user-friendly response

    # Add conditional routing based on the question type
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],  # Use the route field in state to determine next node
        {
            "code": "code_agent",  # Route code questions to code agent
            "general": "generic_agent",  # Route general questions to generic agent
            "genie": "genie_agent",  # Route data questions to genie agent
            "vector_search": "vector_search_agent",  # Route product lookup to vector search
        }
    )

    # Set the entry point to the router node
    workflow.set_entry_point("router")
    
    # Connect each specialized agent to the summarizer
    workflow.add_edge("code_agent", "summarize_agent")
    workflow.add_edge("generic_agent", "summarize_agent")
    workflow.add_edge("genie_agent", "summarize_agent")
    # Commented out edge that may be used in future versions
    #workflow.add_edge("retrieve_context", "vector_search_agent")
    workflow.add_edge("vector_search_agent", "summarize_agent")
    
    # Set the final node in the workflow
    workflow.set_finish_point("summarize_agent")
 
    # Compile the graph into an executable state machine
    return workflow.compile()