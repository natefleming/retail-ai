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

import sys

sys.path.insert(0, "../src")

# COMMAND ----------

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

# DBTITLE 1,Resolve Inference Table from Serving Endpoint
from rich import print as pprint

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServingEndpointDetailed, AiGatewayConfig, AiGatewayInferenceTableConfig

w: WorkspaceClient = WorkspaceClient()

endpoint_config: ServingEndpointDetailed = w.serving_endpoints.get(config.app.endpoint_name)
ai_gateway: AiGatewayConfig = endpoint_config.ai_gateway
inference_table_config: AiGatewayInferenceTableConfig = ai_gateway.inference_table_config

catalog_name: str = inference_table_config.catalog_name
schema_name: str = inference_table_config.schema_name
table_name_prefix: str = inference_table_config.table_name_prefix

payload_table: str = f"{catalog_name}.{schema_name}.{table_name_prefix}_payload"

pprint(payload_table)

# COMMAND ----------

# DBTITLE 1,Load Model and Define Prediction Function
from typing import Any
import time

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from dao_ai.models import get_latest_model_version

mlflow.langchain.autolog(log_traces=True)

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_model_name: str = config.app.registered_model.full_name
latest_version: int = get_latest_model_version(registered_model_name)
model_uri: str = f"models:/{registered_model_name}/{latest_version}"
model_version: ModelVersion = mlflow_client.get_model_version(registered_model_name, str(latest_version))

loaded_agent = mlflow.pyfunc.load_model(model_uri)

PREDICT_DELAY_SECONDS = 1.0
_predict_counter = {"current": 0, "total": 0}


def _run_prediction(messages: list[dict[str, Any]]) -> str:
    input_data: dict[str, Any] = {"messages": messages}
    response: dict[str, Any] = loaded_agent.predict(input_data)
    return response["choices"][0]["message"]["content"]


@mlflow.trace(name="predict", span_type="CHAIN")
def predict_fn(messages: list[dict[str, Any]]) -> str:
    _predict_counter["current"] += 1
    row_num = _predict_counter["current"]
    total = _predict_counter["total"]
    print(f"[{row_num}/{total}] Predicting...")

    if row_num > 1:
        time.sleep(PREDICT_DELAY_SECONDS)

    try:
        content = _run_prediction(messages)
    except Exception as e:
        print(f"[{row_num}/{total}] ERROR: {e}")
        content = f"[ERROR] {e}"

    print(f"[{row_num}/{total}] Done ({len(content)} chars)")
    return content

# COMMAND ----------

# DBTITLE 1,Load and Prepare Inference Data for Evaluation
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pandas as pd

from dao_ai.evaluation import (
    prepare_eval_dataframe,
    build_scorers,
    create_or_get_eval_dataset,
    prepare_eval_results_for_display,
)

df: DataFrame = spark.read.table(payload_table)
df = df.select("databricks_request_id", "request", "response")
df = df.withColumns({
    "inputs": F.struct(F.col("request").alias("request")),
    "expectations": F.struct(F.col("response").alias("expected_response"))
})

eval_df: pd.DataFrame = prepare_eval_dataframe(
    spark_df=df.select("databricks_request_id", "inputs", "expectations"),
    num_evals=config.evaluation.num_evals,
)

display(eval_df)

# COMMAND ----------

# DBTITLE 1,Run Evaluation
from mlflow.models.evaluation import EvaluationResult

if not config.evaluation:
    dbutils.notebook.exit("Missing evaluation configuration")

scorers = build_scorers(config.evaluation)
print(f"Scorers: {[getattr(s, 'name', type(s).__name__) for s in scorers]}")

model_run = mlflow_client.get_run(model_version.run_id)
mlflow.set_experiment(experiment_id=model_run.info.experiment_id)

eval_dataset = create_or_get_eval_dataset(
    name=f"{payload_table}_dataset",
    experiment_id=model_run.info.experiment_id,
    source_df=eval_df,
    replace=config.evaluation.replace,
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
