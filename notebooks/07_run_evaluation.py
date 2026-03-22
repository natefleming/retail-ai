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

# DBTITLE 1,Initialize and Configure DAO AI ResponsesAgent
import mlflow
from mlflow.pyfunc import ResponsesAgent
from dao_ai.config import AppConfig
from dao_ai.logging import configure_logging

mlflow.langchain.autolog(run_tracer_inline=True)

config: AppConfig = AppConfig.from_file(path=config_path)
configure_logging(level=config.app.log_level)

if config.app and config.app.trace_location:
    os.environ.setdefault("MLFLOW_TRACING_SQL_WAREHOUSE_ID", config.app.trace_location.warehouse_id)

app: ResponsesAgent = config.as_responses_agent()

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

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from dao_ai.models import get_latest_model_version

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_model_name: str = config.app.registered_model.full_name
latest_version: int = get_latest_model_version(registered_model_name)
model_uri: str = f"models:/{registered_model_name}/{latest_version}"
model_version: ModelVersion = mlflow_client.get_model_version(registered_model_name, str(latest_version))

_predict_counter = {"current": 0, "total": 0}


def _extract_output_text(response: ResponsesAgentResponse) -> str:
    texts: list[str] = []
    for output in response.output:
        if isinstance(output, dict):
            if output.get("type") == "message":
                for content in output.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "output_text":
                        texts.append(content.get("text", ""))
                    elif isinstance(content, dict) and "text" in content:
                        texts.append(content.get("text", ""))
                    elif getattr(content, "type", None) == "output_text":
                        texts.append(content.text)
        elif getattr(output, "type", None) == "message":
            for content in output.content:
                if isinstance(content, dict) and "text" in content:
                    texts.append(content.get("text", ""))
                elif getattr(content, "type", None) == "output_text":
                    texts.append(content.text)
    return "".join(texts) if texts else str(response.output)


def _run_prediction(messages: list[dict[str, Any]], custom_inputs: dict[str, Any] | None) -> str:
    import asyncio

    request = ResponsesAgentRequest(
        input=[{"role": m["role"], "content": m["content"]} for m in messages],
        custom_inputs=custom_inputs,
    )
    loop = asyncio.get_event_loop()
    response: ResponsesAgentResponse = loop.run_until_complete(app.apredict(request))
    return _extract_output_text(response)


@mlflow.trace(name="evaluation", span_type="CHAIN")
def predict_fn(messages: list[dict[str, Any]]) -> str:
    _predict_counter["current"] += 1
    row_num = _predict_counter["current"]
    total = _predict_counter["total"]
    print(f"[{row_num}/{total}] Predicting...")

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
from datetime import datetime
from mlflow.models.evaluation import EvaluationResult

scorers = build_scorers(config.evaluation)
print(f"Scorers: {[getattr(s, 'name', type(s).__name__) for s in scorers]}")

model_run = mlflow_client.get_run(model_version.run_id)
mlflow.set_experiment(experiment_id=model_run.info.experiment_id)

eval_dataset = create_or_get_eval_dataset(
    name=f"{payload_table}_dataset",
    experiment_id=model_run.info.experiment_id,
    source_df=eval_df,
    replace=evaluation.replace,
)

experiment = mlflow.get_experiment(model_run.info.experiment_id)
print(f"Dataset name:      {eval_dataset.name}")
print(f"Dataset ID:        {eval_dataset.dataset_id}")
print(f"Dataset source:    {eval_dataset.source_type}")
print(f"Dataset records:   {len(eval_dataset.to_df())} rows")
print(f"Experiment name:   {experiment.name}")
print(f"Experiment ID:     {model_run.info.experiment_id}")

mlflow.autolog(disable=True)
mlflow.langchain.autolog(run_tracer_inline=True)

run_tags: dict[str, str] = {
    k: str(v) for k, v in (config.app.tags or {}).items()
}
run_tags["run_type"] = "evaluation"
run_name: str = f"{config.app.name}_evaluation_v{latest_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

_predict_counter["total"] = len(eval_df)
print(f"Starting evaluation: {len(eval_df)} rows, {len(scorers)} scorers")

with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
    try:
        eval_results: EvaluationResult = mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=predict_fn,
            model_id=model_version.model_id,
            scorers=scorers,
        )
        print(f"Evaluation completed. Run ID: {run.info.run_id}")
    except Exception as e:
        print(f"Evaluation raised an exception (metrics may still be logged): {e}")
        eval_results = None

# COMMAND ----------

# DBTITLE 1,Display Evaluation Results
if eval_results is not None:
    print("Evaluation Metrics:")
    for metric_name, metric_value in eval_results.metrics.items():
        print(f"  {metric_name}: {metric_value}")

    eval_results_df = prepare_eval_results_for_display(eval_results)
    print(f"Total evaluation results: {len(eval_results_df)} rows")
    if not eval_results_df.empty:
        display(eval_results_df.head(100))
    else:
        print("No detailed results table available. Available tables:", list(eval_results.tables.keys()))
else:
    print("Evaluation results not available. Check the MLflow run for logged metrics and traces.")
