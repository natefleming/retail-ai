from typing import (
    Sequence, 
    Annotated, 
    TypedDict,
    Annotated, 
    Literal, 
    Optional, 
    Any, 
    Generator
)

from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, ChatMessageChunk
from langchain_core.runnables import RunnableSequence
from langchain_core.vectorstores.base import VectorStore
from langchain_core.documents.base import Document

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from databricks_langchain import ChatDatabricks
from databricks_langchain import DatabricksVectorSearch
from databricks_langchain.genie import Genie
from databricks_ai_bridge.genie import GenieResponse

from pydantic import BaseModel, Field

import mlflow
from mlflow.models import ModelConfig

from retail_ai.state import AgentState, AgentConfig
from retail_ai.messages import last_message, last_human_message
from retail_ai.models import LangGraphChatAgent


config: ModelConfig = ModelConfig(development_config="model_config.yaml")

allowed_routes: Sequence[str] = (
    "code", 
    "general", 
    "genie", 
    "vector_search",
)

class Router(BaseModel):
  """
  A router that will route the question to the correct agent. 
  * Questions about inventory and product information should route to `genie`. 
  * Questions that are most easy solved by code should route to `code`.
  * Questions that about recommendations should route to `vector_search`.
  * All other questions should route to `general`
  """
  route: Literal[tuple(allowed_routes)] = (
      Field(
          default="general", 
          description=f"The route to take. Must be one of {allowed_routes}"
        )
  )

# class AgentState(ChatAgentState):
# #  messages: Annotated[Sequence[BaseMessage], add_messages]
#   context: Sequence[Document]
#   route: str


@mlflow.trace()
def route_question(state: AgentState, config: AgentConfig) -> dict[str, str]:
    print("route_question")
    llm: LanguageModelLike = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
    chain: RunnableSequence = llm.with_structured_output(Router)
    messages: Sequence[BaseMessage] = state["messages"]
    last_message: BaseMessage = last_human_message(messages)
    response = chain.invoke([last_message])
    print(f"route: {response}" )
    return {"route": response.route}


@mlflow.trace()
def vector_search_question(state: AgentState, config: AgentConfig) -> dict[str, str]:

    messages: Sequence[BaseMessage] = state["messages"]
    content: str = last_message(messages).content

    vector_search: VectorStore = DatabricksVectorSearch(
        endpoint=endpoint,
        index_name=index_name,
    )

    context: Sequence[Document] = vector_search.similarity_search(
        query=content, k=1, filter={}
    )

    return {"context": context}


@mlflow.trace()
def genie_question(state: AgentState, config: AgentConfig) -> dict[str, str]:
    print("genie_question")
    print(f"config={config}")
    messages: Sequence[BaseMessage] = state["messages"]
    content: str = last_message(messages).content

    space_id = "01f01c91f1f414d59daaefd2b7ec82ea"
    genie: Genie = Genie(space_id=space_id)
    genie_response: GenieResponse = genie.ask_question(content)
    description: str = genie_response.description
    result: str = genie_response.result
    response: str = f"{description}\n{result}"
    return {"messages": [response]}
  


@mlflow.trace()
def code_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
    llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
    prompt: PromptTemplate = PromptTemplate.from_template(
        "You are a software engineer. Answer this question with step by steps details : {input}"
    )
    chain: RunnableSequence = prompt | llm
    messages: Sequence[BaseMessage] = state["messages"]
    content: str = last_message(messages).content
    response = chain.invoke({"input": content})
    return {"messages": [response]}


@mlflow.trace()
def vector_search_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
    llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
    prompt: PromptTemplate = PromptTemplate.from_template(
        "You are a Business Analyst. Answer this question with step by steps details : {input}"
    )
    chain: RunnableSequence = prompt | llm
    messages: Sequence[BaseMessage] = state["messages"]
    content: str = last_message(messages).content
    response = chain.invoke({"input": content})
    return {"messages": [response]}


@mlflow.trace()
def generic_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
    llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
    prompt: PromptTemplate = PromptTemplate.from_template(
        "Give a general and concise answer to the question: {input}"
    )
    chain: RunnableSequence = prompt | llm
    messages: Sequence[BaseMessage] = state["messages"]
    content: str = last_message(messages).content
    response = chain.invoke({"input": content})
    return {"messages": [response]}
  

@mlflow.trace()
def summarize_response(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
    llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
    prompt: PromptTemplate = PromptTemplate.from_template(
        "Summarize: {input}"
    )
    chain: RunnableSequence = prompt | llm
    messages: Sequence[BaseMessage] = state["messages"]
    response = chain.invoke(messages)
    return {"messages": [response]}


def create_graph() -> CompiledStateGraph:
    workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)

    workflow.add_node("router", route_question)
    workflow.add_node("code_agent", code_question)
    workflow.add_node("generic_agent", generic_question)
    workflow.add_node("genie_agent", genie_question)
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
    workflow.add_edge("vector_search_agent", "summarize_agent")
    workflow.set_finish_point("summarize_agent")
 
    
    return workflow.compile(checkpointer=checkpointer)
    

graph: CompiledStateGraph = create_graph()

app: LangGraphChatAgent = LangGraphChatAgent(graph)

mlflow.models.set_model(app)
