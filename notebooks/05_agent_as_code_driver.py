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
from pkg_resources import get_distribution

sys.path.insert(0, "../src")

pip_requirements: Sequence[str] = [
    f"databricks-agents=={version('databricks-agents')}",
    f"databricks-connect=={get_distribution('databricks-connect').version}",
    f"databricks-langchain=={version('databricks-langchain')}",
    f"databricks-sdk=={version('databricks-sdk')}",
    f"duckduckgo-search=={version('duckduckgo-search')}",
    f"langchain=={version('langchain')}",
    f"langchain-mcp-adapters=={version('langchain-mcp-adapters')}",
    f"langgraph=={version('langgraph')}",
    f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
    f"langgraph-supervisor=={version('langgraph-supervisor')}",
    f"langgraph-swarm=={version('langgraph-swarm')}",
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

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()

# COMMAND ----------

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

config.display_graph()

# COMMAND ----------

config.create_agent()


# COMMAND ----------

config.deploy_agent()
