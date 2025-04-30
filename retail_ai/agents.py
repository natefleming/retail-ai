from typing import Callable, Optional, Sequence

from databricks_ai_bridge.genie import GenieResponse
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools.base import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from retail_ai.state import AgentConfig, AgentState
from retail_ai.tools import create_genie_tool, create_vector_search_tool

from mlflow.models import ModelConfig


from databricks_langchain import ChatDatabricks

from loguru import logger

def create_arma_agent(model_config: ModelConfig, config: AgentState) -> CompiledStateGraph:
    logger.debug(f"config: {config}")
    model_name: str = model_config.get("agents").get("arma").get("model_name")
    if not model_name:
        model_name = model_config.get("llms").get("model_name")

    prompt: str = model_config.get("agents").get("arma").get("prompt")

    llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)

    agent: CompiledStateGraph = create_react_agent(
        name="arma_agent",
        model=llm,
        prompt=prompt,
        state_schema=AgentState,
        config_schema=AgentConfig,
        tools=[],
    )

    return agent    


def create_vector_search_agent(
    model: LanguageModelLike, 
    index_name: str,
    primary_key: str = "id",
    text_column: str = "content",
    doc_uri: str = "doc_uri",
    columns: Optional[Sequence[str]] = None,
    search_parameters: dict[str, str] = {},
) -> CompiledStateGraph:
    """
    Create a LangGraph agent specialized for vector search operations in the retail domain.
    
    This agent can query vector indexes to retrieve relevant product information and reviews
    using similarity search. It uses the ReAct framework (Reasoning and Acting) to determine
    when to use its vector search capabilities.
    
    Args:
        model: The language model that powers the agent's reasoning
        index_name: Name of the vector database index to query
        primary_key: Field name of the document's primary identifier
        text_column: Field name that contains the main text content
        doc_uri: Field name for storing document URI/location
        columns: Specific columns to retrieve from the vector store (None retrieves all)
        search_parameters: Additional parameters to customize the vector search behavior
        
    Returns:
        A compiled LangGraph state machine representing the vector search agent
    """

    vs_tool: BaseTool = create_vector_search_tool(
        name="vector_search_tool",
        description="find context from vector search",
        index_name=index_name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        columns=columns,
        search_parameters=search_parameters,
    )

    vector_search_agent: CompiledStateGraph = create_react_agent(
        name="vector_search_agent",
        model=model,
        tools=[vs_tool],
        prompt=f"""
            You are an intelligent agent that can answer questions about summarizing product details. You have access to a vector search index that contains product reviews. Use the `{vs_tool.name}` tool to answer the question. If the question is not related to product reviews, just say that you don't know.
        """,
        state_schema=AgentState,
        config_schema=AgentConfig,
        checkpointer=None,
    )

    return vector_search_agent
  

def create_genie_agent(
    model: LanguageModelLike, 
    space_id: str
) -> CompiledStateGraph:
    """
    Create a LangGraph agent that interfaces with Databricks Genie for retail operations.
    
    This agent uses Databricks Genie to generate SQL queries, analyze retail data, and
    provide insights about product inventory and sales information. The agent acts as an
    interface between the user's natural language requests and structured data operations.
    
    Args:
        model: The language model that powers the agent's reasoning
        space_id: The Databricks workspace ID where Genie is configured
        
    Returns:
        A compiled LangGraph state machine representing the Genie agent
    """

    genie_tool: Callable[[str], GenieResponse] = create_genie_tool(
        space_id=space_id
    )

    genie_agent: CompiledStateGraph = create_react_agent(
        name="genie_agent",
        model=model,
        tools=[genie_tool],
        prompt="""
            You are an intelligent agent that can answer questions about retail items. You have access to a warehouse that contains product information. Use tools to answer the question. If the question is not related to product details, just say that you don't know.
        """,
        state_schema=AgentState,
        config_schema=AgentConfig,
        checkpointer=None,
    )

    return genie_agent
