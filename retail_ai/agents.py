from typing import Callable, Optional, Sequence

from databricks_ai_bridge.genie import GenieResponse
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools.base import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from retail_ai.state import AgentConfig, AgentState
from retail_ai.tools import create_genie_tool, create_vector_search_tool


def create_vector_search_agent(
    model: LanguageModelLike, 
    index_name: str,
    columns: Optional[Sequence[str]] = None
) -> CompiledStateGraph:

    vs_tool: BaseTool = create_vector_search_tool(
        name="vector_search_tool",
        description="find context from vector search",
        index_name=index_name,
        columns=columns
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
