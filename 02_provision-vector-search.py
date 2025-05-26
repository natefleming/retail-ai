# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "databricks-sdk",
  "databricks-vectorsearch",
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
  f"databricks-sdk=={version('databricks-sdk')}",
  f"databricks-vectorsearch=={version('databricks-vectorsearch')}",
  f"mlflow=={version('mlflow')}",
)
print("\n".join(pip_requirements))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from typing import Any, Sequence

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)

vector_store_config: dict[str, Any] = config.get("resources").get("vector_stores").get("products_vector_store")

embedding_model_endpoint_name: str = vector_store_config.get("embedding_model").get("name")
endpoint_name: str = vector_store_config.get("endpoint_name")
endpoint_type: str = vector_store_config.get("endpoint_type")
index_name: str = vector_store_config.get("index_name")
source_table_name: str = vector_store_config.get("source_table_name")
primary_key: str = vector_store_config.get("primary_key")
embedding_source_column: str = vector_store_config.get("embedding_source_column")
columns: Sequence[str] = vector_store_config.get("columns", [])



print(f"embedding_model_endpoint_name: {embedding_model_endpoint_name}")
print(f"endpoint_name: {endpoint_name}")
print(f"endpoint_type: {endpoint_type}")
print(f"index_name: {index_name}")
print(f"source_table_name: {source_table_name}")
print(f"primary_key: {primary_key}")
print(f"embedding_source_column: {embedding_source_column}")
print(f"columns: {columns}")



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
    embedding_model_endpoint_name=embedding_model_endpoint_name 
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
k: int = 3

search_results: Dict[str, Any] = index.similarity_search(
  query_text=question,
  columns=columns,
  num_results=k)

chunks: List[str] = search_results.get('result', {}).get('data_array', [])
chunks
