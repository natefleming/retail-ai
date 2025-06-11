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
  f"langgraph=={version('langgraph')}",
  f"langchain=={version('langchain')}",
  f"databricks-langchain=={version('databricks-langchain')}",
  f"databricks-sdk=={version('databricks-sdk')}",
  f"mlflow=={version('mlflow')}",
  f"python-dotenv=={version('python-dotenv')}",
  f"unitycatalog-langchain[databricks]=={version('unitycatalog-langchain')}",
)

print("\n".join(pip_requirements))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from typing import Any, Dict, Optional, List

from mlflow.models import ModelConfig
from retail_ai.config import AppConfig

model_config_file: str = config_path
model_config: ModelConfig = ModelConfig(development_config=model_config_file)
config: AppConfig = AppConfig(**model_config.to_dict())

# COMMAND ----------


from databricks.sdk import WorkspaceClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from databricks_langchain import DatabricksFunctionClient, UCFunctionToolkit


w: WorkspaceClient = WorkspaceClient()
client: DatabricksFunctionClient = DatabricksFunctionClient(client=w)

# COMMAND ----------

from typing import Sequence
from pathlib import Path

import pandas as pd
from io import StringIO

from databricks.sdk.service.catalog import FunctionInfo
from unitycatalog.ai.core.base import FunctionExecutionResult
from retail_ai.config import (
  AppConfig, 
  FunctionModel, 
  SchemaModel, 
  UnityCatalogFunctionSqlModel,
)


unity_catalog_functions: Sequence[UnityCatalogFunctionSqlModel] = config.unity_catalog_functions

for unity_catalog_function in unity_catalog_functions:
  unity_catalog_function: UnityCatalogFunctionSqlModel
  unity_catalog_function.create()
