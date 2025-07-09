# Databricks notebook source
# MAGIC %pip install --quiet -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="../config/model_config.yaml")
config_path: str = dbutils.widgets.get("config-path")
print(config_path)

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

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_model_name: str = config.app.registered_model.full_name
model_uri: str = f"models:/{registered_model_name}@Current"
model_version: ModelVersion = mlflow_client.get_model_version_by_alias(registered_model_name, "Current")

loaded_agent = mlflow.pyfunc.load_model(model_uri)
def predict_fn(request: str) -> str:
  input = {"messages": [{"role": "user", "content": f"{request}"}]}
  response: dict[str, Any] = loaded_agent.predict(input)
  content: str = response["choices"][0]["message"]["content"]
  return content

# COMMAND ----------

from mlflow.genai.scorers import scorer, Safety, Guidelines
from mlflow.entities import Feedback, Trace


clarity = Guidelines(
    name="clarity",
    guidelines=["The response must be clear, coherent, and concise"]
)

@scorer
def response_completeness(outputs: str) -> Feedback:
    # Outputs is return value of your app. Here we assume it's a string.
    if len(outputs.strip()) < 10:
        return Feedback(
            value=False,
            rationale="Response too short to be meaningful"
        )

    if outputs.lower().endswith(("...", "etc", "and so on")):
        return Feedback(
            value=False,
            rationale="Response appears incomplete"
        )

    return Feedback(
        value=True,
        rationale="Response appears complete"
    )

@scorer
def tool_call_efficiency(trace: Trace) -> Feedback:
    """Evaluate how effectively the app uses tools"""
    # Retrieve all tool call spans from the trace
    tool_calls = trace.search_spans(span_type="TOOL")

    if not tool_calls:
        return Feedback(
            value=None,
            rationale="No tool usage to evaluate"
        )

    # Check for redundant calls
    tool_names = [span.name for span in tool_calls]
    if len(tool_names) != len(set(tool_names)):
        return Feedback(
            value=False,
            rationale=f"Redundant tool calls detected: {tool_names}"
        )

    # Check for errors
    failed_calls = [s for s in tool_calls if s.status.status_code != "OK"]
    if failed_calls:
        return Feedback(
            value=False,
            rationale=f"{len(failed_calls)} tool calls failed"
        )

    return Feedback(
        value=True,
        rationale=f"Efficient tool usage: {len(tool_calls)} successful calls"
    )

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

scorers_list = [Safety(), clarity, response_completeness, tool_call_efficiency]

with mlflow.start_run(run_id=model_version.run_id):
  eval_results = mlflow.genai.evaluate(
      data=eval_df,
      predict_fn=predict_fn,
      model_id=model_version.model_id,
      scorers=scorers_list,
  )


