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

config_files: Sequence[str] = find_yaml_files_os_walk("../config")
dbutils.widgets.dropdown(name="config-paths", choices=config_files, defaultValue=next(iter(config_files), ""))

config_path: str | None = dbutils.widgets.get("config-path") or None
project_path: str = dbutils.widgets.get("config-paths") or None

config_path: str = config_path or project_path

print(config_path)

# COMMAND ----------

# DBTITLE 1,Add Source Directory to System Path
import sys

sys.path.insert(0, "../src")

# COMMAND ----------

# DBTITLE 1,Load Configuration
from dao_ai.config import AppConfig, EvaluationModel

config: AppConfig = AppConfig.from_file(path=config_path)

evaluation: EvaluationModel = config.evaluation

if not evaluation:
    dbutils.notebook.exit("Missing evaluation configuration")

if not evaluation.monitoring:
    dbutils.notebook.exit("Missing evaluation.monitoring configuration")

print(f"Built-in scorer sample rate: {evaluation.monitoring.sample_rate}")
print(f"Guidelines scorer sample rate: {evaluation.monitoring.guidelines_sample_rate}")

# COMMAND ----------

# DBTITLE 1,Resolve MLflow Experiment from Registered Model
import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from dao_ai.models import get_latest_model_version

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_model_name: str = config.app.registered_model.full_name
latest_version: int = get_latest_model_version(registered_model_name)
model_version: ModelVersion = mlflow_client.get_model_version(registered_model_name, str(latest_version))

model_run = mlflow_client.get_run(model_version.run_id)
experiment_id: str = model_run.info.experiment_id

print(f"Model: {registered_model_name} v{latest_version}")
print(f"Experiment ID: {experiment_id}")

# COMMAND ----------

# DBTITLE 1,Register and Start Monitoring Scorers
from dao_ai.evaluation import register_monitoring_scorers

registered = register_monitoring_scorers(
    evaluation_config=config.evaluation,
    experiment_id=experiment_id,
)

print(f"\nRegistered {len(registered)} scorers for production monitoring")

# COMMAND ----------

# DBTITLE 1,Display Active Monitoring Scorers
from dao_ai.evaluation import get_monitoring_scorers

scorers = get_monitoring_scorers()
for s in scorers:
    print(f"  {s.name}: sample_rate={s.sample_rate}")

print(f"\nTotal active scorers: {len(scorers)}")
