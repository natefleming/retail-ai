# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

from typing import Sequence

pip_requirements: Sequence[str] = (
  "langgraph",
  "langchain",
  "databricks-langchain", 
  "databricks-sdk",
  "mlflow",
  "python-dotenv"
)

pip_requirements: str = " ".join(pip_requirements)

%pip install --quiet --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from typing import Sequence
from importlib.metadata import version

pip_requirements: Sequence[str] = (
  f"langgraph=={version('langgraph')}",
  f"langchain=={version('langchain')}"
  f"databricks-langchain=={version('databricks-langchain')}",
  f"databricks-sdk=={version('databricks-sdk')}",
  f"mlflow=={version('mlflow')}",
  f"python-dotenv=={version('python-dotenv')}"
)

print("\n".join(pip_requirements))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from typing import Any, Sequence

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)

retreiver_config: dict[str, Any] = config.get("retriever")

embedding_model_endpoint_name: str = retreiver_config.get("embedding_model_endpoint_name")
endpoint_name: str = retreiver_config.get("endpoint_name")
endpoint_type: str = retreiver_config.get("endpoint_type")
index_name: str = retreiver_config.get("index_name")
primary_key: str = retreiver_config.get("primary_key")
embedding_source_column: str = retreiver_config.get("embedding_source_column")
columns: Sequence[str] = retreiver_config.get("columns", [])
search_parameters: dict[str, Any] = retreiver_config.get("search_parameters", {})

datasets_config: dict[str, Any] = config.get("datasets")
huggingface_config: dict[str, Any] = datasets_config.get("huggingface")
source_table_name: str = huggingface_config.get("table_name")

space_id = config.get("genie").get("space_id")

assert embedding_model_endpoint_name is not None
assert endpoint_name is not None
assert endpoint_type is not None
assert index_name is not None
assert primary_key is not None
assert embedding_source_column is not None
assert source_table_name is not None
assert columns is not None
assert search_parameters is not None
assert space_id is not None

# COMMAND ----------

from langchain_core.tools.base import BaseTool
from databricks_langchain.vector_search_retriever_tool import VectorSearchRetrieverTool
from databricks.sdk import WorkspaceClient




w: WorkspaceClient = WorkspaceClient()
vector_search_retriever_tool: BaseTool = (
  VectorSearchRetrieverTool(
    name="vector_search_retriever_tool",
    description="Retrieves documents from a vector search index",
    index_name=index_name,
    columns=None,
    workspace_client=w,
  )
)

# COMMAND ----------

from langgraph.prebuilt import create_react_agent
from databricks_langchain import ChatDatabricks
from retail_ai.tools import create_vector_search_tool
from retail_ai.state import AgentState, AgentConfig


vs_tool = create_vector_search_tool(
    name="vector_search_tool",
    description="find context from vector search",
    index_name=index_name,
    columns=columns
)

model_name: str = "databricks-meta-llama-3-3-70b-instruct"
vector_search_agent = create_react_agent(
    model=ChatDatabricks(model=model_name, temperature=0.1),
    tools=[vs_tool],
    prompt="You are an intelligent agent that can answer questions about summarizing product reviews. You have access to a vector search index that contains product reviews. Use the vector search index to answer the question. If the question is not related to product reviews, just say that you don't know.",
    state_schema=AgentState,
    config_schema=AgentConfig,
    checkpointer=None,
)

# COMMAND ----------

type(vector_search_agent)

# COMMAND ----------


input = config.get("app").get("example_input")

foo = vector_search_agent.invoke(input=input)

# COMMAND ----------

ffoo

# COMMAND ----------

space_id

# COMMAND ----------

from retail_ai.tools import create_genie_tool

genie_tool= create_genie_tool(
    space_id=space_id
)


# COMMAND ----------

type(genie_tool.__doc__)
