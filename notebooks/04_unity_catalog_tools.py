# Databricks notebook source
# MAGIC %pip install --quiet --upgrade -r ../requirements.txt
# MAGIC %pip uninstall --quiet -y databricks-connect pyspark pyspark-connect
# MAGIC %pip install --quiet databricks-connect
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="../config/model_config.yaml")
config_path: str = dbutils.widgets.get("config-path")
print(config_path)

# COMMAND ----------

import sys
from typing import Sequence
from importlib.metadata import version

sys.path.insert(0, "../src")

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

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

from typing import Sequence
from dao_ai.config import UnityCatalogFunctionSqlModel


unity_catalog_functions: Sequence[UnityCatalogFunctionSqlModel] = (
    config.unity_catalog_functions
)

for unity_catalog_function in unity_catalog_functions:
    unity_catalog_function: UnityCatalogFunctionSqlModel
    unity_catalog_function.create()
