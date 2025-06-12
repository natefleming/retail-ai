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

from retail_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from retail_ai.config import SchemaModel, VolumeModel


w: WorkspaceClient = WorkspaceClient()

for _, schema in config.schemas.items():
  schema: SchemaModel
  _ = schema.create(w=w)

  print(f"schema: {schema.full_name}")

for _, volume in config.resources.volumes.items():
  volume: VolumeModel
  
  _ = volume.create(w=w)
  print(f"volume: {volume.full_name}")

# COMMAND ----------

from retail_ai.config import DatasetModel

datasets: Sequence[DatasetModel] = config.datasets

for dataset in datasets:
    dataset: DatasetModel
    dataset.create()
