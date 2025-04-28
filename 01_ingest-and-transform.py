# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "databricks-sdk",
  "delta-spark",
  "mlflow",
  "datasets",
  "python-dotenv",
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
  f"mlflow=={version('mlflow')}",
  f"datasets=={version('datasets')}",
)

print("\n".join(pip_requirements))


# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

import mlflow
from mlflow.models import ModelConfig

config: ModelConfig = ModelConfig(development_config="model_config.yaml")

catalog_name: str = config.get("catalog_name")
database_name: str = config.get("database_name")
volume_name: str = config.get("volume_name")
primary_key: str = config.get("retriever").get("primary_key")
doc_uri: str = config.get("retriever").get("doc_uri")
embedding_source_column: str = config.get("retriever").get("embedding_source_column")

print(f"catalog_name: {catalog_name}")
print(f"database_name: {database_name}")
print(f"volume_name: {volume_name}")
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

# COMMAND ----------

from pathlib import Path


hf_home_path: Path = volume.as_path() / "huggingface"
hf_datasets_path: Path = hf_home_path / "datasets"


# COMMAND ----------

from retail_ai.datasets import HuggingfaceDataSource

spark.dataSource.register(HuggingfaceDataSource)

# COMMAND ----------

from typing import Any

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from delta.tables import DeltaTable, IdentityGenerator


datasets_config: dict[str, Any] = config.get("datasets")
huggingface_configs: dict[str, Any] = datasets_config.get("huggingface")

for huggingface_config in huggingface_configs:
  url: str = huggingface_config.get("url") 
  repo_id: str = huggingface_config.get("repo_id")
  primary_key: str = huggingface_config.get("primary_key")
  table_name: str = huggingface_config.get("table_name")

  source_type: str = "repo_id" if repo_id else "url"
  source_path: str = repo_id if repo_id else url

  wands_df: DataFrame = (
    spark.read
      .format("huggingface")
      .option(source_type, source_path) 
      .option("primary_key", primary_key)
      .option("cache_dir", hf_datasets_path.as_posix())
      .load()
  )

  if "product_description" in wands_df.columns:
    wands_df = wands_df.filter(F.col("product_description").isNotNull())

  additional_cols = {
      doc_uri: F.lit(source_path),
      primary_key: F.col(primary_key)
  }

  if "product_description" in wands_df.columns:
    additional_cols[embedding_source_column] = F.col("product_description")
                                                     
  wands_df = wands_df.withColumns(additional_cols)
  
  (
    DeltaTable.createOrReplace(spark)
      .tableName(table_name)
      .property("delta.enableChangeDataFeed", "true")
      .addColumns(wands_df.schema)
      .execute()
  )

  pk_constraint: str = f"{table_name.replace(".", "_")}_{primary_key}_pk"
  spark.sql(f"ALTER TABLE {table_name} ADD CONSTRAINT {pk_constraint} PRIMARY KEY ({primary_key})")

  wands_df.write.mode("append").saveAsTable(table_name)

  display(spark.table(table_name))
