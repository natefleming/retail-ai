# Databricks notebook source
# MAGIC %pip install --quiet --upgrade -r ../requirements.txt
# MAGIC %pip uninstall --quiet -y databricks-connect pyspark pyspark-connect
# MAGIC %pip install --quiet databricks-connect
# MAGIC %restart_python

# COMMAND ----------

from typing import Sequence
import os

def find_yaml_files_os_walk(base_path: str) -> Sequence[str]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path does not exist: {base_path}")
    
    if not os.path.isdir(base_path):
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")
    
    yaml_files = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.yaml', '.yml')):
                yaml_files.append(os.path.join(root, file))
    
    return sorted(yaml_files)

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

config_files: Sequence[str] = find_yaml_files_os_walk("../config")
dbutils.widgets.dropdown(name="config-paths", choices=config_files, defaultValue=next(iter(config_files), ""))

config_path: str | None = dbutils.widgets.get("config-path") or None
project_path: str = dbutils.widgets.get("config-paths") or None

config_path: str = config_path or project_path

print(config_path)
# COMMAND ----------

import sys
from typing import Sequence
from importlib.metadata import version

sys.path.insert(0, "../src")

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

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

from dao_ai.config import VectorStoreModel

vector_stores: dict[str, VectorStoreModel] = config.resources.vector_stores

for _, vector_store in vector_stores.items():
  vector_store: VectorStoreModel

  print(f"vector_store: {vector_store}")
  vector_store.create()


# COMMAND ----------

from typing import Dict, Any, List

from databricks.vector_search.index import VectorSearchIndex
from dao_ai.config import RetrieverModel


question: str = "How many grills do we have in stock?"

for name, retriever in config.retrievers.items():
  retriever: RetrieverModel
  index: VectorSearchIndex = retriever.vector_store.as_index() 
  k: int = 3

  search_results: Dict[str, Any] = index.similarity_search(
    query_text=question,
    columns=retriever.columns,
    **retriever.search_parameters.model_dump()
  )

  chunks: list[str] = search_results.get('result', {}).get('data_array', [])
  print(len(chunks))
  print(chunks)

# COMMAND ----------

from typing import Sequence

from databricks_langchain import DatabricksVectorSearch
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStore

content = "What grills do you have in stock?"
for name, retriever in config.retrievers.items():
  vector_search: VectorStore = DatabricksVectorSearch(
      endpoint=retriever.vector_store.endpoint.name,
      index_name=retriever.vector_store.index.full_name,
      columns=retriever.columns,
      client_args={},
  )

  documents: Sequence[Document] = vector_search.similarity_search(
      query=content, **retriever.search_parameters.model_dump()
  )
  print(len(documents))
