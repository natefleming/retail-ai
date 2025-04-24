from typing import Literal, Optional, Sequence

import mlflow
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain.prompts import PromptTemplate
from langchain_core.documents.base import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableSequence
from langchain_core.vectorstores.base import VectorStore
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from retail_ai.agents import create_genie_agent, create_vector_search_agent
from retail_ai.messages import last_human_message, last_message
from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable

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

def route_question_node(model_name: str) -> AgentCallable:
    @mlflow.trace()
    def route_question(state: AgentState, config: AgentConfig) -> dict[str, str]:
        llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
        chain: RunnableSequence = llm.with_structured_output(Router)
        messages: Sequence[BaseMessage] = state["messages"]
        last_message: BaseMessage = last_human_message(messages)
        response = chain.invoke([last_message])
        return {"route": response.route}

    return route_question
  

def generic_question_node(model_name: str) -> AgentCallable:
    
    @mlflow.trace()
    def generic_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
        prompt: PromptTemplate = PromptTemplate.from_template(
            "Give a general and concise answer to the question: {input}"
        )
        chain: RunnableSequence = prompt | llm
        messages: Sequence[BaseMessage] = state["messages"]
        content: str = last_message(messages).content
        response = chain.invoke({"input": content})
        return {"messages": [response]}
    
    return generic_question

  
def code_question_node(model_name: str) -> AgentCallable:
    @mlflow.trace()
    def code_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
        prompt: PromptTemplate = PromptTemplate.from_template(
            "You are a software engineer. Answer this question with step by steps details : {input}"
        )
        chain: RunnableSequence = prompt | llm
        messages: Sequence[BaseMessage] = state["messages"]
        content: str = last_message(messages).content
        response = chain.invoke({"input": content})
        return {"messages": [response]}

    return code_question
  

def vector_search_question_node(
    model_name: str, 
    index_name: str,
    primary_key: str,
    text_column: str,
    doc_uri: str,
    columns: Optional[Sequence[str]] = None,
    search_parameters: dict[str, str] = {},    
) -> AgentCallable:

    model: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)

    vector_search_agent: CompiledStateGraph = create_vector_search_agent(
        model=model,
        index_name=index_name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        columns=columns,
        search_parameters=search_parameters,
    )

    @mlflow.trace()
    def vector_search_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        return vector_search_agent.invoke(input=state, config=config)

    return vector_search_question
  

def genie_question_node(model_name: str, space_id: str) -> AgentCallable:

    model: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)

    genie_agent: CompiledStateGraph = create_genie_agent(
        model=model,
        space_id=space_id
    )

    @mlflow.trace()
    def genie_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        return genie_agent.invoke(input=state, config=config)
    
    return genie_question
  

def summarize_response_node(model_name: str) -> AgentCallable:
    @mlflow.trace()
    def summarize_response(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
        llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
        prompt: PromptTemplate = PromptTemplate.from_template(
            "Summarize: {input}"
        )
        chain: RunnableSequence = prompt | llm
        messages: Sequence[BaseMessage] = state["messages"]
        response = chain.invoke(messages)
        return {"messages": [response]}

    return summarize_response
  

def retrieve_context_node(
    endpoint: str, 
    index_name: str, 
    search_parameters: dict[str, str]
) -> AgentCallable:
    
    @mlflow.trace()
    def retrieve_context(state: AgentState, config: AgentConfig) -> dict[str, str]:
        messages: Sequence[BaseMessage] = state["messages"]
        content: str = last_message(messages).content

        vector_search: VectorStore = DatabricksVectorSearch(
            endpoint=endpoint,
            index_name=index_name,
        )

        context: Sequence[Document] = vector_search.similarity_search(
            query=content, **search_parameters
        )

        return {"context": context}

    return retrieve_context
