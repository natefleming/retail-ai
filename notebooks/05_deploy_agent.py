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
dbutils.widgets.dropdown(
    name="deployment-target",
    choices=["", "model_serving", "apps"],
    defaultValue="",
)

config_files: Sequence[str] = find_yaml_files_os_walk("../config")
dbutils.widgets.dropdown(name="config-paths", choices=config_files, defaultValue=next(iter(config_files), ""))

config_path: str | None = dbutils.widgets.get("config-path") or None
project_path: str = dbutils.widgets.get("config-paths") or None
deployment_target_str: str | None = dbutils.widgets.get("deployment-target") or None

config_path: str = config_path or project_path

print(f"Config path: {config_path}")
print(f"Deployment target: {deployment_target_str or '(using config default)'}")

# COMMAND ----------

import sys
from typing import Sequence
from importlib.metadata import version
from pkg_resources import get_distribution

sys.path.insert(0, "../src")

pip_requirements: Sequence[str] = [
    f"databricks-agents=={version('databricks-agents')}",
    f"databricks-connect=={get_distribution('databricks-connect').version}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"databricks-sdk=={version('databricks-sdk')}",
    f"ddgs=={version('ddgs')}",
    f"langchain=={version('langchain')}",
    f"langchain-mcp-adapters=={version('langchain-mcp-adapters')}",
    f"langgraph=={version('langgraph')}",
    f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
    f"langmem=={version('langmem')}",
    f"loguru=={version('loguru')}",
    f"mlflow=={version('mlflow')}",
    f"openevals=={version('openevals')}",
    f"psycopg[binary,pool]=={version('psycopg')}",
    f"pydantic=={version('pydantic')}",
    f"unitycatalog-ai[databricks]=={version('unitycatalog-ai')}",
    f"unitycatalog-langchain[databricks]=={version('unitycatalog-langchain')}",
]
print("\n".join(pip_requirements))

# COMMAND ----------

import dao_ai.providers
import dao_ai.providers.base
import dao_ai.providers.databricks

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()

# COMMAND ----------

from dao_ai.config import AppConfig, DeploymentTarget

config: AppConfig = AppConfig.from_file(path=config_path)

# Resolve deployment target from widget (hybrid resolution)
# If widget is empty/None, deploy_agent() will use config.app.deployment_target or default
deployment_target: DeploymentTarget | None = None
if deployment_target_str:
    deployment_target = DeploymentTarget(deployment_target_str)
    print(f"Using widget-specified deployment target: {deployment_target.value}")
elif config.app and config.app.deployment_target:
    print(f"Using config file deployment target: {config.app.deployment_target.value}")
else:
    print("Using default deployment target: model_serving")

# COMMAND ----------

config.display_graph()

# COMMAND ----------

config.create_agent()

# COMMAND ----------

config.deploy_agent(target=deployment_target)
