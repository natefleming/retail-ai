# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "databricks-sdk",
  "databricks-vectorsearch",
  "python-dotenv",
  "loguru"
)

pip_requirements: str = " ".join(pip_requirements)

%pip install --quiet --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from typing import Sequence

from importlib.metadata import version


pip_requirements: Sequence[str] = (
  f"databricks-sdk=={version('databricks-sdk')}",
  f"databricks-vectorsearch=={version('databricks-vectorsearch')}",
  f"loguru=={version('loguru')}",
)
print("\n".join(pip_requirements))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from typing import Any

from databricks.vector_search.client import VectorSearchClient

import mlflow.deployments
from mlflow.deployments.databricks import DatabricksDeploymentClient

from retail_ai.vector_search import endpoint_exists, index_exists


def create_endpoint(vsc: VectorSearchClient, endpoint_name: str) -> None:
    if not endpoint_exists(vsc, endpoint_name):
        vsc.create_endpoint_and_wait(name=endpoint_name, verbose=True, endpoint_type=endpoint_type)

    print(f"Endpoint named {endpoint_name} is ready.")


def create_index(
    vsc: VectorSearchClient, 
    endpoint_name: str, 
    index_name: str,
    source_table_name: str,
    primary_key: str,
    embedding_source_column: str,
    embedding_model_endpoint_name: str,
    endpoint_type: str = "STANDARD"
) -> None:
  if not index_exists(vsc, endpoint_name, index_name):
    print(f"Creating index {index_name} on endpoint {endpoint_name}...")
    vsc.create_delta_sync_index_and_wait(
      endpoint_name=endpoint_name,
      index_name=index_name,
      source_table_name=source_table_name,
      pipeline_type="TRIGGERED",
      primary_key=primary_key,
      embedding_source_column=embedding_source_column, 
      embedding_model_endpoint_name=embedding_model_endpoint_name,
      endpoint_type=endpoint_type
    )
  else:
    vsc.get_index(endpoint_name, index_name).sync()

  print(f"index {index_name} on table {source_table_name} is ready")


def query(vsc: VectorSearchClient, question: str, endpoint_name: str, index_name: str, columns: list[str] = [], k: int = 3) -> list[str]:
  deploy_client: DatabricksDeploymentClient = mlflow.deployments.get_deploy_client("databricks")

  question = "What is Databricks?"

  index: VectorSearchIndex = vsc.get_index(endpoint_name, index_name)

  search_results: Dict[str, Any] = index.similarity_search(
    query_text=question,
    columns=columns,
    num_results=k)

  chunks: list[str] = search_results.get('result', {}).get('data_array', [])
  return chunks

# COMMAND ----------

from typing import Any, Sequence
from pathlib import Path

from databricks.vector_search.client import VectorSearchClient
from retail_ai.config import AppConfig, Retriever


file_path: str = "agent_as_config.yaml"
config: AppConfig = AppConfig(config=file_path)

vsc: VectorSearchClient = VectorSearchClient()

for alias, retriever in config.resources.retrievers.items():
  alias: str
  retriever: Retriever

  create_endpoint(vsc, retriever.endpoint_name)
  create_index(
    vsc=vsc, 
    endpoint_name=retriever.endpoint_name, 
    index_name=retriever.index_name,
    source_table_name=retriever.source_table_name,
    primary_key=retriever.primary_key,
    embedding_source_column=retriever.embedding_source_column,
    embedding_model_endpoint_name=retriever.embedding_model.model,
    endpoint_type=retriever.endpoint_type,
  )


# COMMAND ----------

from typing import Any, Sequence
from pathlib import Path

from rich import print as pprint

from databricks.vector_search.client import VectorSearchClient

from retail_ai.config import AppConfig, Retriever


file_path: str = "agent_as_config.yaml"
config: AppConfig = AppConfig(config=file_path)

vsc: VectorSearchClient = VectorSearchClient()

for alias, retriever in config.resources.retrievers.items():
  alias: str
  retriever: Retriever

  chunks: list[str] = (
    query(
      vsc=vsc, 
      question="What is Databricks?", 
      endpoint_name=retriever.endpoint_name, 
      index_name=retriever.index_name,
      columns=[retriever.primary_key, retriever.embedding_source_column],
    )
  )
  pprint(chunks)

