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

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

import sys

sys.path.insert(0, "../src")

# COMMAND ----------

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

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

from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from dao_ai.models import get_latest_model_version

# Enable autologging with trace support - IMPORTANT: log_traces=True is required
# for trace-based scorers to receive trace objects
mlflow.langchain.autolog(log_traces=True)

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_model_name: str = config.app.registered_model.full_name
latest_version: int = get_latest_model_version(registered_model_name)
model_uri: str = f"models:/{registered_model_name}/{latest_version}"
model_version: ModelVersion = mlflow_client.get_model_version(registered_model_name, str(latest_version))

loaded_agent = mlflow.pyfunc.load_model(model_uri)


# Use @mlflow.trace decorator to ensure traces are created and linked to evaluation
# This is CRITICAL for trace-based scorers like tool_call_efficiency to work
@mlflow.trace(name="predict", span_type="CHAIN")
def predict_fn(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Prediction function wrapped with MLflow tracing.
    
    Args:
        messages: Chat messages in format [{"role": "user", "content": "..."}]
    
    Returns dict output format for MLflow 3.8+ scorer compatibility:
    {"response": "..."} instead of a plain string
    """
    # Keep the full messages payload visible in MLflow traces
    print(f"messages={messages}")
    input_data = {"messages": messages}
    response: dict[str, Any] = loaded_agent.predict(input_data)
    content: str = response["choices"][0]["message"]["content"]
    
    print(f"response_content={content}")
    
    # Return dict format for compatibility with scorers
    return {"response": content}

# COMMAND ----------

# Import reusable scorers from dao_ai.evaluation module
# These scorers are designed to work with MLflow 3.8+ patterns
from dao_ai.evaluation import (
    response_completeness,
    tool_call_efficiency,
    create_response_clarity_scorer,
    create_agent_routing_scorer,
    create_guidelines_scorers,
)
from mlflow.genai.scorers import Safety, Guidelines
from mlflow.entities import Feedback, Trace

# COMMAND ----------

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

import pandas as pd


df: DataFrame = spark.read.table(payload_table)

df = df.select("databricks_request_id", "request", "response")
df = df.withColumns({
    "inputs": F.struct(F.col("request").alias("request")),
    "expectations": F.struct(F.col("response").alias("expected_response"))
})

eval_df: pd.DataFrame = df.select("databricks_request_id", "inputs", "expectations").toPandas()
display(eval_df)

# Normalize evaluation inputs for Guidelines scorers: it expects a list of message dicts
def normalize_eval_inputs_to_input_dict(raw_inputs: Any) -> dict[str, Any]:
    """
    MLflow requires the 'inputs' column to be a dict of field names -> values.
    Guidelines scorers expect those inputs to contain chat messages. We standardize to:
        {"messages": [{"role": "user", "content": "..."}]}
    """
    # If already dict-shaped, keep ONLY the keys that match predict_fn params.
    # Our predict_fn takes only `messages`, so we must not pass extra keys.
    if isinstance(raw_inputs, dict) and "messages" in raw_inputs:
        messages_val = raw_inputs.get("messages")
        if isinstance(messages_val, list):
            return {"messages": messages_val}
        return {"messages": [{"role": "user", "content": str(messages_val)}]}

    if isinstance(raw_inputs, list) and (not raw_inputs or isinstance(raw_inputs[0], dict)):
        return {"messages": raw_inputs}

    if isinstance(raw_inputs, dict) and "request" in raw_inputs:
        return {"messages": [{"role": "user", "content": str(raw_inputs["request"])}]}

    try:
        request_val = raw_inputs["request"]  # type: ignore[index]
        return {"messages": [{"role": "user", "content": str(request_val)}]}
    except Exception:
        return {"messages": [{"role": "user", "content": str(raw_inputs)}]}


if "inputs" in eval_df.columns:
    eval_df["inputs"] = eval_df["inputs"].apply(normalize_eval_inputs_to_input_dict)

# COMMAND ----------

import mlflow
from mlflow.models.model import ModelInfo
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.models.evaluation import EvaluationResult
import pandas as pd


model_info: mlflow.models.model.ModelInfo
evaluation_result: EvaluationResult

registered_model_name: str = config.app.registered_model.full_name

if not config.evaluation:
  dbutils.notebook.exit("Missing evaluation configuration")

evaluation_table_name: str = config.evaluation.table.full_name

# Build scorer list with Safety and custom scorers from dao_ai.evaluation
judge_model = config.evaluation.judge_model_endpoint
print(f"Using judge model: {judge_model}")

scorers_list = [
    Safety(model=judge_model),
    response_completeness,
    tool_call_efficiency,
    # TODO: Re-enable when Databricks endpoints support response_schema
    # create_response_clarity_scorer(judge_model=judge_model),
    # create_agent_routing_scorer(judge_model=judge_model),
]

# Add Guidelines scorers from config with proper judge model
if config.evaluation.guidelines:
    custom_scorers = create_guidelines_scorers(
        guidelines_config=config.evaluation.guidelines,
        judge_model=judge_model,
    )
    scorers_list += custom_scorers
    print(f"Added {len(custom_scorers)} Guidelines scorers")

# Get the experiment ID from the model's run and set it as the current experiment
model_run = mlflow_client.get_run(model_version.run_id)
mlflow.set_experiment(experiment_id=model_run.info.experiment_id)

with mlflow.start_run(run_id=model_version.run_id):
  eval_results = mlflow.genai.evaluate(
      data=eval_df,
      predict_fn=predict_fn,
      model_id=model_version.model_id,
      scorers=scorers_list,
  )

# COMMAND ----------

# DBTITLE 1,Display Evaluation Results
print("Evaluation Metrics:")
for metric_name, metric_value in eval_results.metrics.items():
    print(f"  {metric_name}: {metric_value}")

# Display the evaluation results table
# Note: The 'assessments' column contains complex objects that can't be
# directly displayed in Databricks. We convert it to string representation.
eval_results_df = eval_results.tables["eval_results"].copy()

# Convert complex columns to string for display compatibility
if "assessments" in eval_results_df.columns:
    eval_results_df["assessments"] = eval_results_df["assessments"].astype(str)

# Convert any other object columns that might cause Arrow conversion issues
for col in eval_results_df.columns:
    if eval_results_df[col].dtype == "object":
        try:
            # Try to keep as-is first, only convert if it fails
            eval_results_df[col].to_list()
        except Exception:
            eval_results_df[col] = eval_results_df[col].astype(str)

# Display limited rows to avoid performance issues with large result sets
print(f"Total evaluation results: {len(eval_results_df)} rows")
display(eval_results_df.head(100))


