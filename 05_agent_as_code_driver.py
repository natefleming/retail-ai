# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "langgraph",
  "langchain",
  "databricks-langchain",
  "langgraph-checkpoint-postgres",
  "databricks-agents",
  "psycopg[binary,pool]", 
  "databricks-sdk",
  "mlflow",
  "pydantic",
  "python-dotenv",
  "uv"
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
    f"langchain=={version('langchain')}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
    f"databricks-sdk=={version('databricks-sdk')}",
    f"mlflow=={version('mlflow')}",
    f"psycopg[binary,pool]=={version('psycopg')}", 
    f"databricks-agents=={version('databricks-agents')}",
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
# MAGIC from langgraph.prebuilt import create_react_agent
# MAGIC
# MAGIC from langgraph.graph import StateGraph, END
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC
# MAGIC from langgraph.checkpoint.postgres import PostgresSaver
# MAGIC from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
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
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC config: ModelConfig = ModelConfig(development_config="model_config.yaml")
# MAGIC
# MAGIC model_name: str = config.get("llms").get("model_name")
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
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def route_question(state: AgentState, config: AgentConfig) -> dict[str, str]:
# MAGIC     llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
# MAGIC     chain: RunnableSequence = llm.with_structured_output(Router)
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     last_message: BaseMessage = last_human_message(messages)
# MAGIC     response = chain.invoke([last_message])
# MAGIC     return {"route": response.route}
# MAGIC
# MAGIC
# MAGIC endpoint: str = config.get("retriever").get("endpoint_name")
# MAGIC index_name: str = config.get("retriever").get("index_name")
# MAGIC search_parameters: dict[str, str] = config.get("retriever").get("search_parameters")
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def retrieve_context(state: AgentState, config: AgentConfig) -> dict[str, str]:
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = last_message(messages).content
# MAGIC
# MAGIC     vector_search: VectorStore = DatabricksVectorSearch(
# MAGIC         endpoint=endpoint,
# MAGIC         index_name=index_name,
# MAGIC     )
# MAGIC
# MAGIC     context: Sequence[Document] = vector_search.similarity_search(
# MAGIC         query=content, **search_parameters
# MAGIC     )
# MAGIC
# MAGIC     return {"context": context}
# MAGIC
# MAGIC
# MAGIC space_id: str = config.get("genie").get("space_id")
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def genie_question(state: AgentState, config: AgentConfig) -> dict[str, str]:
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = last_message(messages).content
# MAGIC
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
# MAGIC     llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
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
# MAGIC     llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
# MAGIC     prompt: PromptTemplate = PromptTemplate.from_template("""
# MAGIC         You are an expert retail recommendation assistant.
# MAGIC         Answer the following question using only the context provided:
# MAGIC         ### Question:
# MAGIC         {input}
# MAGIC
# MAGIC         ### Context: 
# MAGIC         {context}
# MAGIC     """
# MAGIC     )
# MAGIC     chain: RunnableSequence = prompt | llm
# MAGIC     messages: Sequence[BaseMessage] = state["messages"]
# MAGIC     content: str = last_message(messages).content
# MAGIC     context: str = state["context"]
# MAGIC     response = chain.invoke({"input": content, "context": context})
# MAGIC     return {"messages": [response]}
# MAGIC
# MAGIC
# MAGIC @mlflow.trace()
# MAGIC def generic_question(state: AgentState, config: AgentConfig) -> dict[str, BaseMessage]:
# MAGIC     llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
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
# MAGIC     llm: LanguageModelLike = ChatDatabricks(model=model_name, temperature=0.1)
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
# MAGIC     workflow.add_node("retrieve_context", retrieve_context)
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
# MAGIC             "vector_search": "retrieve_context",
# MAGIC         }
# MAGIC     )
# MAGIC
# MAGIC     workflow.set_entry_point("router")
# MAGIC     workflow.add_edge("code_agent", "summarize_agent")
# MAGIC     workflow.add_edge("generic_agent", "summarize_agent")
# MAGIC     workflow.add_edge("genie_agent", "summarize_agent")
# MAGIC     workflow.add_edge("retrieve_context", "vector_search_agent")
# MAGIC     workflow.add_edge("vector_search_agent", "summarize_agent")
# MAGIC     workflow.set_finish_point("summarize_agent")
# MAGIC  
# MAGIC     
# MAGIC     return workflow.compile()
# MAGIC     
# MAGIC
# MAGIC def create_app(graph: CompiledStateGraph) -> LangGraphChatAgent:
# MAGIC     return LangGraphChatAgent(graph)
# MAGIC
# MAGIC
# MAGIC graph: CompiledStateGraph = create_graph()
# MAGIC app: LangGraphChatAgent = create_app(graph)
# MAGIC
# MAGIC
# MAGIC mlflow.models.set_model(app)
# MAGIC

# COMMAND ----------

# from agent_as_code import graph

# from IPython.display import Image, display

# display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

# COMMAND ----------

from typing import Any
from agent_as_code import app, config

example_input: dict[str, Any] = config.get("app").get("example_input")

app.predict(example_input)

# COMMAND ----------

from typing import Any, Union, Sequence, Iterator
from langchain_core.messages import BaseMessage, MessageLikeRepresentation

from agent_as_code import app

def process_messages_stream(
  messages: Union[Sequence[MessageLikeRepresentation], dict[str, Any]]
) -> Iterator[BaseMessage]:
  if isinstance(messages, list):
    messages = {"messages": messages}
  for event in app.predict_stream(messages):
    yield event


def process_messages(
  messages: Union[Sequence[MessageLikeRepresentation], dict[str, Any]]
) -> Iterator[BaseMessage]:
  if isinstance(messages, list):
    messages = {"messages": messages}
  return app.predict(messages)

# COMMAND ----------

from typing import Any
from mlflow.types.agent import ChatAgentResponse
from agent_as_code import app, config

example_input: dict[str, Any] = config.get("app").get("example_input")

result: ChatAgentResponse = process_messages(example_input)
print(result)

# COMMAND ----------

from typing import Any
from mlflow.types.agent import ChatAgentChunk
from agent_as_code import app, config


example_input: dict[str, Any] = config.get("app").get("example_input")

for event in process_messages_stream(example_input["messages"]):
  event: ChatAgentChunk
  print(event, "-----------\n")


# COMMAND ----------


from typing import Sequence

from databricks_langchain import VectorSearchRetrieverTool

from mlflow.models.resources import (
    DatabricksResource,
    DatabricksVectorSearchIndex,
    DatabricksTable,
    DatabricksUCConnection,
    DatabricksSQLWarehouse,
    DatabricksGenieSpace, 
    DatabricksFunction, 
    DatabricksServingEndpoint
)
import mlflow
from mlflow.models.model import ModelInfo

from agent_as_code import config


model_name: str = config.get("llms").get("model_name")
index_name: str = config.get("retriever").get("index_name")
space_id: str = config.get("genie").get("space_id")

resources: Sequence[DatabricksResource] = [
    DatabricksServingEndpoint(endpoint_name=model_name),
    DatabricksVectorSearchIndex(index_name=index_name),
    DatabricksGenieSpace(genie_space_id=space_id),
    DatabricksFunction(function_name="nfleming.retail_ai.find_wands_product_by_id"),
    DatabricksFunction(function_name="system.ai.python_exec")
]

# tools = []
# for tool in tools:
#     if isinstance(tool, VectorSearchRetrieverTool):
#         resources.extend(tool.resources)
#     elif isinstance(tool, UnityCatalogTool):
#         resources.append(DatabricksFunction(function_name=tool.uc_function_name))


with mlflow.start_run(run_name="agent"):
    mlflow.set_tag("type", "chain")
    logged_agent_info: ModelInfo = mlflow.pyfunc.log_model(
        python_model="agent_as_code.py",
        code_paths=["retail_ai"],
        model_config=config.to_dict(),
        artifact_path="agent",
        pip_requirements=pip_requirements,
        resources=resources,
    )

# COMMAND ----------

from typing import Any
from agent_as_code import config

example_input: dict[str, Any] = config.get("app").get("example_input")

mlflow.models.predict(
    model_uri=logged_agent_info.model_uri,
    input_data=example_input,
    env_manager="uv",
)

# COMMAND ----------

import mlflow
from mlflow.models.model import ModelInfo
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.models.evaluation import EvaluationResult

import pandas as pd

from agent_as_code import config


model_info: mlflow.models.model.ModelInfo
evaluation_result: EvaluationResult

evaluation_table_name: str = config.get("evaluation").get("table_name")

evaluation_pdf: pd.DataFrame = spark.table(evaluation_table_name).toPandas()

global_guidelines = {
  "English": ["The response must be in English"],
  "Clarity": ["The response must be clear, coherent, and concise"],
}

with mlflow.start_run():
  eval_results = mlflow.evaluate(
      data=evaluation_pdf,            
      model=logged_agent_info.model_uri,    
      model_type="databricks-agent",  
      evaluator_config={
        "databricks-agent": {"global_guidelines": global_guidelines}
      }
  )


# COMMAND ----------

import mlflow
from mlflow.entities.model_registry.model_version import ModelVersion

from agent_as_code import config


mlflow.set_registry_uri("databricks-uc")

registered_model_name: str = config.get("app").get("registered_model_name")

model_version: ModelVersion = mlflow.register_model(
    name=registered_model_name,
    model_uri=logged_agent_info.model_uri
)


# COMMAND ----------

from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion

client: MlflowClient = MlflowClient()

client.set_registered_model_alias(name=registered_model_name, alias="Champion", version=model_version.version)
champion_model: ModelVersion = client.get_model_version_by_alias(registered_model_name, "Champion")
print(champion_model)

# COMMAND ----------

from databricks import agents
from databricks.sdk.service.serving import (
    ServedModelInputWorkloadSize, 
)

from agent_as_code import config
from retail_ai.models import get_latest_model_version

registered_model_name: str = config.get("app").get("registered_model_name")
endpoint_name: str = config.get("app").get("endpoint_name")
tags: dict[str, str] = config.get("app").get("tags")
latest_version: int = get_latest_model_version(registered_model_name)


agents.deploy(
  model_name=registered_model_name, 
  model_version=latest_version, 
  scale_to_zero=True,
  environment_vars={},
  workload_size=ServedModelInputWorkloadSize.SMALL,
  endpoint_name=endpoint_name,
  tags=tags
)

# COMMAND ----------

from typing import Sequence

from databricks.agents import set_permissions, PermissionLevel

from agent_as_code import config


users: Sequence[str] = config.get("app").get("users") 

if users:
  set_permissions(model_name=registered_model_name, users=users, permission_level=PermissionLevel.CAN_MANAGE)

