# Databricks notebook source
# MAGIC %pip install uv
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
  f"python-dotenv=={version('python-dotenv')}",
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

import mlflow
from mlflow.models import ModelConfig
from retail_ai.config import AppConfig

model_config: ModelConfig = ModelConfig(development_config=config_path)
config: AppConfig = AppConfig(**model_config.to_dict())


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
from retail_ai.config import SchemaModel, VolumeModel


w: WorkspaceClient = WorkspaceClient()


for _, schema in config.schemas.items():
  schema: SchemaModel
  catalog_info: CatalogInfo = get_or_create_catalog(name=schema.catalog_name, w=w)
  schema_info: SchemaInfo = get_or_create_database(catalog=catalog_info, name=schema.schema_name, w=w)

  print(f"catalog: {catalog_info.full_name}")
  print(f"schema: {schema_info.full_name}")

for _, volume in config.resources.volumes.items():
  print(volume.name)
  volume: VolumeModel
  volume_info: VolumeInfo = get_or_create_volume(
    catalog=catalog_info,
    database=schema_info,
    name=volume.name,
    w=w
  )
  print(f"volume: {volume_info.full_name}")

# COMMAND ----------

from typing import Any, Sequence
import re
from pathlib import Path
from retail_ai.config import DatasetModel

datasets: Sequence[DatasetModel] = config.datasets

current_dir: Path = "file:///" / Path.cwd().relative_to("/")

for dataset in datasets:
  dataset: DatasetModel
  table: str = dataset.table.full_name
  ddl_path: Path = Path(dataset.ddl)
  data_path: Path = current_dir / Path(dataset.data)
  format: str = dataset.format

  statements: Sequence[str] = [s for s in re.split(r"\s*;\s*",  ddl_path.read_text()) if s]
  for statement in statements:
    spark.sql(statement, args={"database": dataset.table.schema_model.full_name})
    spark.read.format(format).load(data_path.as_posix()).write.mode("overwrite").saveAsTable(table)



# COMMAND ----------


for dataset in config.datasets:
  display(spark.table(dataset.table.full_name))
