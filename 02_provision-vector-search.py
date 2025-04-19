# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "databricks-sdk",
  "databricks-vectorsearch",
  "mlflow",
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
    embedding_source_column=embedding_source_column, #The column containing our text
    embedding_model_endpoint_name=embedding_model_endpoint_name #The embedding endpoint used to create the embeddings
  )
else:
  vsc.get_index(endpoint_name, index_name).sync()

print(f"index {index_name} on table {source_table_name} is ready")

# COMMAND ----------

from typing import Dict, Any, List

import mlflow.deployments
from databricks.vector_search.index import VectorSearchIndex
from mlflow.deployments.databricks import DatabricksDeploymentClient

deploy_client: DatabricksDeploymentClient = mlflow.deployments.get_deploy_client("databricks")

question = "What is Databricks?"

index: VectorSearchIndex = vsc.get_index(endpoint_name, index_name)
k: int = search_parameters.get("k", 3)

search_results: Dict[str, Any] = index.similarity_search(
  query_text=question,
  columns=columns,
  num_results=k)

chunks: List[str] = search_results.get('result', {}).get('data_array', [])
chunks
