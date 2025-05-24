# Databricks notebook source
# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

from typing import Any, Sequence

from mlflow.models import ModelConfig
_ = load_dotenv(find_dotenv())


model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)

retreiver_config: dict[str, Any] = config.get("retriever")

embedding_model_endpoint_name: str = retreiver_config.get("embedding_model_endpoint_name")
endpoint_name: str = retreiver_config.get("endpoint_name")
endpoint_type: str = retreiver_config.get("endpoint_type")
index_name: str = retreiver_config.get("index_name")
source_table_name: str = retreiver_config.get("source_table_name")
primary_key: str = retreiver_config.get("primary_key")
embedding_source_column: str = retreiver_config.get("embedding_source_column")
columns: Sequence[str] = retreiver_config.get("columns", [])
search_parameters: dict[str, Any] = retreiver_config.get("search_parameters", {})


print(f"embedding_model_endpoint_name: {embedding_model_endpoint_name}")
print(f"endpoint_name: {endpoint_name}")
print(f"endpoint_type: {endpoint_type}")
print(f"index_name: {index_name}")
print(f"source_table_name: {source_table_name}")
print(f"primary_key: {primary_key}")
print(f"embedding_source_column: {embedding_source_column}")
print(f"columns: {columns}")
print(f"search_parameters: {search_parameters}")




# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from retail_ai.vector_search import endpoint_exists

vsc: VectorSearchClient = VectorSearchClient()

if not endpoint_exists(vsc, endpoint_name):
    vsc.create_endpoint_and_wait(name=endpoint_name, verbose=True, endpoint_type=endpoint_type)

print(f"Endpoint named {endpoint_name} is ready.")


# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.vector_search.index import VectorSearchIndex
from retail_ai.vector_search import index_exists


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
    columns=columns
  )
else:
  vsc.get_index(endpoint_name, index_name).sync()

print(f"index {index_name} on table {source_table_name} is ready")

# COMMAND ----------

from typing import Dict, Any, List

import mlflow.deployments
from databricks.vector_search.index import VectorSearchIndex
from mlflow.deployments.databricks import DatabricksDeploymentClient


question: str = "What what is the best hammer for drywall?"

index: VectorSearchIndex = vsc.get_index(endpoint_name, index_name)
k: int = search_parameters.get("k", 3)

search_results: Dict[str, Any] = index.similarity_search(
  query_text=question,
  columns=columns,
  num_results=k)

chunks: List[str] = search_results.get('result', {}).get('data_array', [])
chunks
