# Databricks notebook source
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


assert embedding_model_endpoint_name is not None
assert endpoint_name is not None
assert endpoint_type is not None
assert index_name is not None
assert primary_key is not None
assert embedding_source_column is not None
assert source_table_name is not None
assert columns is not None
assert search_parameters is not None


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

\
