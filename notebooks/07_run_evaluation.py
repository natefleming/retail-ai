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

import dao_ai.providers
import dao_ai.providers.base
import dao_ai.providers.databricks

# COMMAND ----------

# DBTITLE 1,Enable Nest Asyncio for Compatibility
import nest_asyncio
nest_asyncio.apply()

# COMMAND ----------

# DBTITLE 1,Initialize and Configure DAO AI ChatModel
import mlflow
from langgraph.graph.state import CompiledStateGraph
from mlflow.pyfunc import ChatModel
from dao_ai.graph import create_dao_ai_graph
from dao_ai.models import create_agent
from dao_ai.config import AppConfig
from dao_ai.logging import configure_logging

mlflow.langchain.autolog(log_traces=True)

config: AppConfig = AppConfig.from_file(path=config_path)
configure_logging(level=config.app.log_level)

graph: CompiledStateGraph = create_dao_ai_graph(config=config)
app: ChatModel = create_agent(graph)

# COMMAND ----------

# DBTITLE 1,Validate Evaluation Configuration
from typing import Any
from dao_ai.config import EvaluationModel

evaluation: EvaluationModel = config.evaluation

if not evaluation:
    dbutils.notebook.exit("Missing evaluation configuration")

payload_table: str = evaluation.table.full_name
custom_inputs: dict[str, Any] = evaluation.custom_inputs

print(f"Evaluation table: {payload_table}")
print(f"Custom inputs: {custom_inputs}")

# COMMAND ----------

# DBTITLE 1,Load Model Version and Define Prediction Function
from typing import Any
import time

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.types.llm import ChatCompletionResponse
from dao_ai.models import process_messages, get_latest_model_version

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_model_name: str = config.app.registered_model.full_name
latest_version: int = get_latest_model_version(registered_model_name)
model_uri: str = f"models:/{registered_model_name}/{latest_version}"
model_version: ModelVersion = mlflow_client.get_model_version(registered_model_name, str(latest_version))

PREDICT_DELAY_SECONDS = 1.0
_predict_counter = {"current": 0, "total": 0}


def _run_prediction(messages: list[dict[str, Any]], custom_inputs: dict[str, Any] | None) -> str:
    input_data: dict[str, Any] = {"messages": messages}
    if custom_inputs:
        input_data["custom_inputs"] = custom_inputs

    response: ChatCompletionResponse = process_messages(app, **input_data)
    return response.choices[0].message.content


@mlflow.trace(name="predict", span_type="CHAIN")
def predict_fn(messages: list[dict[str, Any]]) -> str:
    _predict_counter["current"] += 1
    row_num = _predict_counter["current"]
    total = _predict_counter["total"]
    print(f"[{row_num}/{total}] Predicting...")

    if row_num > 1:
        time.sleep(PREDICT_DELAY_SECONDS)

    try:
        response_content = _run_prediction(messages, custom_inputs)
    except Exception as e:
        print(f"[{row_num}/{total}] ERROR: {e}")
        response_content = f"[ERROR] {e}"

    print(f"[{row_num}/{total}] Done ({len(response_content)} chars)")
    return response_content

# COMMAND ----------

# DBTITLE 1,Load and Prepare Evaluation Data
import pandas as pd
from dao_ai.evaluation import (
    prepare_eval_dataframe,
    build_scorers,
    create_or_get_eval_dataset,
    prepare_eval_results_for_display,
)

eval_df: pd.DataFrame = prepare_eval_dataframe(
    spark_df=spark.read.table(payload_table),
    num_evals=config.evaluation.num_evals,
)

display(eval_df)

# COMMAND ----------

# DBTITLE 1,Run Evaluation
from mlflow.models.evaluation import EvaluationResult

scorers = build_scorers(config.evaluation)
print(f"Scorers: {[getattr(s, 'name', type(s).__name__) for s in scorers]}")

model_run = mlflow_client.get_run(model_version.run_id)
mlflow.set_experiment(experiment_id=model_run.info.experiment_id)

eval_dataset = create_or_get_eval_dataset(
    name=f"{payload_table}_dataset",
    experiment_id=model_run.info.experiment_id,
    source_df=eval_df,
)

_predict_counter["total"] = len(eval_df)
print(f"Starting evaluation: {len(eval_df)} rows, {len(scorers)} scorers")

with mlflow.start_run(run_id=model_version.run_id):
    eval_results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        model_id=model_version.model_id,
        scorers=scorers,
    )

# COMMAND ----------

# DBTITLE 1,Display Evaluation Results
print("Evaluation Metrics:")
for metric_name, metric_value in eval_results.metrics.items():
    print(f"  {metric_name}: {metric_value}")

eval_results_df = prepare_eval_results_for_display(eval_results)
print(f"Total evaluation results: {len(eval_results_df)} rows")
display(eval_results_df.head(100))
