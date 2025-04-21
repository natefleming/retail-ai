# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "langgraph",
  "langchain",
  "databricks-langchain",
  "langgraph-checkpoint-postgres",
  "psycopg", 
  "psycopg-pool",
  "databricks-sdk",
  "mlflow",
  "pydantic",
  "python-dotenv"
)

pip_requirements: str = " ".join(pip_requirements)

%pip install --quiet --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from importlib.metadata import version

from typing import Sequence

from importlib.metadata import version

pip_requirements: Sequence[str] = [
    f"langgraph=={version('langgraph')}",
    f"langchain=={version('langchain')}"
    f"databricks-langchain=={version('databricks-langchain')}"
    f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
    f"psycopg=={version('psycopg')}", 
    f"psycopg-pool=={version('psycopg-pool')}"
    f"databricks-sdk=={version('databricks-sdk')}",
    f"mlflow=={version('mlflow')}",
    f"pydantic=={version('pydantic')}",
]
print("\n".join(pip_requirements))


# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %%writefile agent_as_code.py
# MAGIC from typing import (
# MAGIC     Sequence, 
# MAGIC     Annotated, 
# MAGIC     TypedDict,
# MAGIC     Annotated, 
# MAGIC     Literal, 
# MAGIC     Optional, 
# MAGIC     Any, 
# MAGIC     Generator
# MAGIC )
# MAGIC
# MAGIC from langchain.prompts import PromptTemplate
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.messages import AIMessage, BaseMessage, ChatMessageChunk, HumanMessage
# MAGIC from langchain_core.runnables import RunnableSequence
# MAGIC from langchain_core.vectorstores.base import VectorStore
# MAGIC from langchain_core.documents.base import Document
# MAGIC
# MAGIC from langgraph.graph import StateGraph, END
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC from databricks_langchain import DatabricksVectorSearch
# MAGIC from databricks_langchain.genie import Genie
# MAGIC from databricks_ai_bridge.genie import GenieResponse
# MAGIC
# MAGIC from pydantic import BaseModel, Field
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.models import ModelConfig
# MAGIC
# MAGIC from retail_ai.state import AgentState, AgentConfig
# MAGIC from retail_ai.messages import last_message, last_human_message
# MAGIC from retail_ai.models import LangGraphChatAgent
# MAGIC
# MAGIC
# MAGIC config: ModelConfig = ModelConfig(development_config="model_config.yaml")
# MAGIC
# MAGIC allowed_routes: Sequence[str] = (
# MAGIC     "code", 
# MAGIC     "general", 
# MAGIC     "genie", 
# MAGIC     "vector_search",
# MAGIC )
# MAGIC
# MAGIC class Router(BaseModel):
# MAGIC   """
# MAGIC   A router that will route the question to the correct agent. 
# MAGIC   * Questions about inventory and product information should route to `genie`. 
# MAGIC   * Questions that are most easy solved by code should route to `code`.
# MAGIC   * Questions that about recommendations should route to `vector_search`.
# MAGIC   * All other questions should route to `general`
# MAGIC   """
# MAGIC   route: Literal[tuple(allowed_routes)] = (
# MAGIC       Field(
# MAGIC           default="general", 
# MAGIC           description=f"The route to take. Must be one of {allowed_routes}"
# MAGIC         )
# MAGIC   )
# MAGIC
# MAGIC # class AgentState(ChatAgentState):
# MAGIC # #  messages: Annotated[Sequence[BaseMessage], add_messages]
# MAGIC #   context: Sequence[Document]
# MAGIC #   route: str
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def route_question(state: AgentState, config: AgentConfig) -> dict[str, str]:
# MAGIC     print("route_question")
# MAGIC     llm: LanguageModelLike = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
# MAGIC     chain: RunnableSequence = llm.with_structured_output(Router)
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     last_message: BaseMessage = last_human_message(messages)
# MAGIC     response = chain.invoke([last_message])
# MAGIC     print(f"route: {response}" )
# MAGIC     return {"route": response.route}
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def vector_search_question(state: AgentState, config: AgentConfig) -> dict[str, str]:
# MAGIC
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = last_message(messages).content
# MAGIC
# MAGIC     vector_search: VectorStore = DatabricksVectorSearch(
# MAGIC         endpoint=endpoint,
# MAGIC         index_name=index_name,
# MAGIC     )
# MAGIC
# MAGIC     context: Sequence[Document] = vector_search.similarity_search(
# MAGIC         query=content, k=1, filter={}
# MAGIC     )
# MAGIC
# MAGIC     return {"context": context}
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def genie_question(state: AgentState, config: AgentConfig) -> dict[str, str]:
# MAGIC     print("genie_question")
# MAGIC     print(f"config={config}")
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = last_message(messages).content
# MAGIC
# MAGIC     space_id = "01f01c91f1f414d59daaefd2b7ec82ea"
# MAGIC     genie: Genie = Genie(space_id=space_id)
# MAGIC     genie_response: GenieResponse = genie.ask_question(content)
# MAGIC     description: str = genie_response.description
# MAGIC     result: str = genie_response.result
# MAGIC     response: HumanMessage = HumanMessage(content=f"{description}\n{result}")
# MAGIC     return {"messages": [response]}
# MAGIC   
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def code_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
# MAGIC     llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
# MAGIC     prompt: PromptTemplate = PromptTemplate.from_template(
# MAGIC         "You are a software engineer. Answer this question with step by steps details : {input}"
# MAGIC     )
# MAGIC     chain: RunnableSequence = prompt | llm
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = last_message(messages).content
# MAGIC     response = chain.invoke({"input": content})
# MAGIC     return {"messages": [response]}
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def vector_search_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
# MAGIC     llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
# MAGIC     prompt: PromptTemplate = PromptTemplate.from_template(
# MAGIC         "You are a Business Analyst. Answer this question with step by steps details : {input}"
# MAGIC     )
# MAGIC     chain: RunnableSequence = prompt | llm
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = last_message(messages).content
# MAGIC     response = chain.invoke({"input": content})
# MAGIC     return {"messages": [response]}
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def generic_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
# MAGIC     llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
# MAGIC     prompt: PromptTemplate = PromptTemplate.from_template(
# MAGIC         "Give a general and concise answer to the question: {input}"
# MAGIC     )
# MAGIC     chain: RunnableSequence = prompt | llm
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = last_message(messages).content
# MAGIC     response = chain.invoke({"input": content})
# MAGIC     return {"messages": [response]}
# MAGIC   
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def summarize_response(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
# MAGIC     llm: BaseChatModel = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct")
# MAGIC     prompt: PromptTemplate = PromptTemplate.from_template(
# MAGIC         "Summarize: {input}"
# MAGIC     )
# MAGIC     chain: RunnableSequence = prompt | llm
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     response = chain.invoke(messages)
# MAGIC     return {"messages": [response]}
# MAGIC
# MAGIC
# MAGIC def create_graph() -> CompiledStateGraph:
# MAGIC     workflow: StateGraph = StateGraph(AgentState, config_schema=AgentConfig)
# MAGIC
# MAGIC     workflow.add_node("router", route_question)
# MAGIC     workflow.add_node("code_agent", code_question)
# MAGIC     workflow.add_node("generic_agent", generic_question)
# MAGIC     workflow.add_node("genie_agent", genie_question)
# MAGIC     workflow.add_node("vector_search_agent", vector_search_question)
# MAGIC     workflow.add_node("summarize_agent", summarize_response)
# MAGIC
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "router",
# MAGIC         lambda state: state["route"],
# MAGIC         {
# MAGIC             "code": "code_agent",
# MAGIC             "general": "generic_agent",
# MAGIC             "genie": "genie_agent",
# MAGIC             "vector_search": "vector_search_agent",
# MAGIC         }
# MAGIC     )
# MAGIC
# MAGIC     workflow.set_entry_point("router")
# MAGIC     workflow.add_edge("code_agent", "summarize_agent")
# MAGIC     workflow.add_edge("generic_agent", "summarize_agent")
# MAGIC     workflow.add_edge("genie_agent", "summarize_agent")
# MAGIC     workflow.add_edge("vector_search_agent", "summarize_agent")
# MAGIC     workflow.set_finish_point("summarize_agent")
# MAGIC  
# MAGIC     
# MAGIC     return workflow.compile(checkpointer=checkpointer)
# MAGIC     
# MAGIC
# MAGIC graph: CompiledStateGraph = create_graph()
# MAGIC
# MAGIC app: LangGraphChatAgent = LangGraphChatAgent(graph)
# MAGIC
# MAGIC mlflow.models.set_model(app)
# MAGIC

# COMMAND ----------

# from IPython.display import Image, display
# from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# from agent_as_code import graph

# display(
#     Image(
#         graph.get_graph().draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API,
#         )
#     )
# )

# COMMAND ----------

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# COMMAND ----------

from agent_as_code import graph

config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Could you tell me how many ottomans we have in stock?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# COMMAND ----------

from typing import Sequence
from langchain_core.messages import HumanMessage, MessageLikeRepresentation

from agent_as_code import graph



messages: Sequence[MessageLikeRepresentation] = [
    HumanMessage(content="Can you recommend an ottoman?")
]
response = graph.invoke({"messages": messages})


response["messages"][-1].content

# COMMAND ----------

from agent_as_code import graph

inputs = {
    "messages": [
        ("user", "show me total bank accounts for each country"),
    ]
}
response = graph.invoke(inputs)


response["messages"][-1].content

# COMMAND ----------

from agent_as_code import app

app.predict({"messages": [{"role": "user", "content": "show me marketing research for Japan"}]})

# COMMAND ----------

import mlflow
from agent_as_code import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint



resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))


with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent_as_code.py",
        code_paths=["retail_ai"],
        pip_requirements=pip_requirements,
        resources=resources,
    )

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "nfleming"
schema = "default"
model_name = "langgraph_chatagent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "docs"})

# COMMAND ----------

import pprint

inputs = {
    "messages": [
        ("user", "show me total bank accounts for each country"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")


