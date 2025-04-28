# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "databricks-sdk",
  "delta-spark",
  "python-dotenv",
  "loguru",
  "rich",
)

pip_requirements: str = " ".join(pip_requirements)

%pip install --quiet --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from typing import Sequence
from importlib.metadata import version

pip_requirements: Sequence[str] = (
  f"databricks-sdk=={version('databricks-sdk')}",
  f"delta-spark=={version('delta-spark')}",
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

import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="TRACE")

# COMMAND ----------

from pathlib import Path
from retail_ai.config import AppConfig


file_path: Path = Path("agent_as_config.yaml")

config: AppConfig = AppConfig(config=file_path)

catalog_name: str = config.unity_catalog.catalog_name
database_name: str = config.unity_catalog.database_name
volume_name: str = config.unity_catalog.volume_name
primary_key: str = config.resources.retrievers["retail_vector_search_retriever"].primary_key
doc_uri: str = config.resources.retrievers["retail_vector_search_retriever"].doc_uri
embedding_source_column: str = config.resources.retrievers["retail_vector_search_retriever"].embedding_source_column

print(f"catalog_name: {config.unity_catalog.catalog_name}")
print(f"database_name: {config.unity_catalog.database_name}")
print(f"volume_name: {config.unity_catalog.volume_name}")
print(f"primary_key: {primary_key}")
print(f"doc_uri: {doc_uri}")
print(f"embedding_source_column: {embedding_source_column}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
  CatalogInfo, 
  SchemaInfo, 
  VolumeInfo, 
  VolumeType
)
from retail_ai.catalog import (
  get_or_create_catalog, 
  get_or_create_database, 
  get_or_create_volume
)


w: WorkspaceClient = WorkspaceClient()

catalog: CatalogInfo = get_or_create_catalog(name=catalog_name, w=w)
database: SchemaInfo = get_or_create_database(catalog=catalog, name=database_name, w=w)
volume: VolumeInfo = get_or_create_volume(catalog=catalog, database=database, name=volume_name, w=w)

print(f"catalog: {catalog.full_name}")
print(f"database: {database.full_name}")
print(f"volume: {volume.full_name}")
print(f"volume path: {volume.as_path()}")
