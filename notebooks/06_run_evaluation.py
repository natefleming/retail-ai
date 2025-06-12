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

sys.path.insert(0, "..")

# COMMAND ----------

from retail_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

import mlflow
from mlflow.models.model import ModelInfo
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.models.evaluation import EvaluationResult
import pandas as pd


model_info: mlflow.models.model.ModelInfo
evaluation_result: EvaluationResult

registered_model_name: str = config.app.registered_model.full_name

if config.evaluation:
    
    evaluation_table_name: str = config.evaluation.table.full_name

    evaluation_pdf: pd.DataFrame = spark.table(evaluation_table_name).toPandas()

    global_guidelines = {

    }

    model_uri: str = f"models:/{registered_model_name}@Current"

    with mlflow.start_run():
        mlflow.set_tag("type", "evaluation")
        eval_results = mlflow.evaluate(
            data=evaluation_pdf,
            model=model_uri,
            model_type="databricks-agent",
            evaluator_config={"databricks-agent": {"global_guidelines": global_guidelines}},
        )
