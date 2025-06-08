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
  f"databricks-agents=={version('databricks-agents')}",
  f"mlflow=={version('mlflow')}",
)

print("\n".join(pip_requirements))

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

import mlflow
from mlflow.models import ModelConfig
from retail_ai.config import AppConfig

development_config: str = config_path
model_config: ModelConfig = ModelConfig(development_config=development_config)
config: AppConfig = AppConfig(**model_config.to_dict())

# COMMAND ----------

from rich import print as pprint

from databricks.agents.monitoring import (
    update_monitor, 
    AssessmentsSuiteConfig, 
    BuiltinJudge, 
    GuidelinesJudge
)
from databricks.rag_eval.monitoring.entities import Monitor

try:
  monitor: Monitor = update_monitor(
      endpoint_name = config.app.endpoint_name,
      assessments_config = AssessmentsSuiteConfig(
          sample=1.0, 
          assessments=[
              BuiltinJudge(name='safety'),  # or {'name': 'safety'}
              BuiltinJudge(name='groundedness', sample_rate=0.4), # or {'name': 'groundedness', 'sample_rate': 0.4}
              BuiltinJudge(name='relevance_to_query'), # or {'name': 'relevance_to_query'}
              BuiltinJudge(name='chunk_relevance'), # or {'name': 'chunk_relevance'}
              # Create custom judges with the guidelines judge.
              GuidelinesJudge(guidelines={
                "english": ["The response must be in English"],
                "clarity": ["The response must be clear, coherent, and concise"],
              }),
          ],
      )
  )
  pprint(monitor)
except Exception as e:
  print(f"Error updating monitor: {e}")


