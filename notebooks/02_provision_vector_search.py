# Databricks notebook source
%pip install uv

import os
os.environ["UV_PROJECT_ENVIRONMENT"] = os.environ["VIRTUAL_ENV"]

%sh uv --project ../ sync
%restart_python


# from typing import Sequence

# pip_requirements: Sequence[str] = (
#   "databricks-sdk",
#   "databricks-vectorsearch",
#   "mlflow",
#   "python-dotenv"
# )

# pip_requirements: str = " ".join(pip_requirements)

# %pip install --quiet --upgrade {pip_requirements}
# %restart_python

# COMMAND ----------

import sys
from typing import Sequence
from importlib.metadata import version

sys.path.insert(0, "..")


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
from retail_ai.config import AppConfig, VectorStoreModel
from databricks.sdk import WorkspaceClient
from databricks.vector_search.index import VectorSearchIndex
from retail_ai.vector_search import index_exists, endpoint_exists
from databricks.vector_search.client import VectorSearchClient


model_config_file: str = "model_config.yaml"
model_config: ModelConfig = ModelConfig(development_config=model_config_file)
config: AppConfig = AppConfig(**model_config.to_dict())


vector_stores: dict[str, VectorStoreModel] = config.resources.vector_stores

for _, vector_store in vector_stores.items():
  vector_store: VectorStoreModel

  print(f"vector_store: {vector_store}")

  vsc: VectorSearchClient = VectorSearchClient()

  if not endpoint_exists(vsc, vector_store.endpoint_name):
      vsc.create_endpoint_and_wait(
        name=vector_store.endpoint_name, 
        endpoint_type=vector_store.endpoint_type.name,
        verbose=True, 
      )

  print(f"Endpoint named {vector_store.endpoint_name} is ready.")


  if not index_exists(vsc, vector_store.endpoint_name, vector_store.index.full_name):
    print(f"Creating index {vector_store.index.full_name} on endpoint {vector_store.endpoint_name}...")
    vsc.create_delta_sync_index_and_wait(
      endpoint_name=vector_store.endpoint_name,
      index_name=vector_store.index.full_name,
      source_table_name=vector_store.source_table.full_name,
      pipeline_type="TRIGGERED",
      primary_key=vector_store.index.primary_key,
      embedding_source_column=vector_store.embedding_source_column, 
      embedding_model_endpoint_name=vector_store.embedding_model.name 
    )
  else:
    vsc.get_index(vector_store.endpoint_name, vector_store.index.full_name).sync()

  print(f"index {vector_store.index.full_name} on table {vector_store.source_table.full_name} is ready")


# COMMAND ----------

from typing import Dict, Any, List

from databricks.vector_search.index import VectorSearchIndex
from retail_ai.config import RetrieverModel


question: str = "What what is the best hammer for drywall?"

for name, retriever in config.retrievers.items():
  retriever: RetrieverModel
  index: VectorSearchIndex = vsc.get_index(retriever.vector_store.endpoint_name, retriever.vector_store.index.full_name)
  k: int = 3

  search_results: Dict[str, Any] = index.similarity_search(
    query_text=question,
    columns=retriever.columns,
    **retriever.search_parameters
  )

  chunks: list[str] = search_results.get('result', {}).get('data_array', [])
  chunks
