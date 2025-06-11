# Databricks notebook source
# MAGIC %pip install --quiet uv
# MAGIC
# MAGIC import os
# MAGIC os.environ["UV_PROJECT_ENVIRONMENT"] = os.environ["VIRTUAL_ENV"]

# COMMAND ----------

# MAGIC %sh uv --project ../ sync

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="../config/model_config.yaml")
config_path: str = dbutils.widgets.get("config-path")
print(config_path)

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


model_config_file: str = config_path
model_config: ModelConfig = ModelConfig(development_config=model_config_file)
config: AppConfig = AppConfig(**model_config.to_dict())


vector_stores: dict[str, VectorStoreModel] = config.resources.vector_stores

for _, vector_store in vector_stores.items():
  vector_store: VectorStoreModel

  print(f"vector_store: {vector_store}")
  vector_store.create()


# COMMAND ----------

from typing import Dict, Any, List

from databricks.vector_search.index import VectorSearchIndex
from retail_ai.config import RetrieverModel


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

import asyncio
import os
from collections import OrderedDict
from io import StringIO
from typing import Any, Callable, Literal, Optional, Sequence

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    StatementResponse,
    StatementState,
)
from databricks_ai_bridge.genie import GenieResponse
from databricks_langchain import (
    DatabricksFunctionClient,
    DatabricksVectorSearch,
    UCFunctionToolkit,
)
from databricks_langchain.genie import Genie
from databricks_langchain.vector_search_retriever_tool import VectorSearchRetrieverTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores.base import VectorStore
from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger
from pydantic import BaseModel, Field
from unitycatalog.ai.core.base import FunctionExecutionResult


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
