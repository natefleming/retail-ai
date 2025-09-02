# Databricks notebook source
# DBTITLE 1,Install Required Python Packages and Restart Kernel
# MAGIC %pip install --quiet --upgrade -r ../requirements.txt
# MAGIC %pip uninstall --quiet -y databricks-connect pyspark pyspark-connect
# MAGIC %pip install --quiet databricks-connect
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Set and Retrieve Configuration Path
dbutils.widgets.text(name="config-path", defaultValue="../config/model_config.yaml")
config_path: str = dbutils.widgets.get("config-path")
print(config_path)

# COMMAND ----------

# DBTITLE 1,Add Source Directory to System Path
import sys

sys.path.insert(0, "../src")

# COMMAND ----------

# DBTITLE 1,Enable Nest Asyncio for Compatibility
import nest_asyncio
nest_asyncio.apply()

# COMMAND ----------

# DBTITLE 1,Initialize and Configure DAO AI ChatModel
import sys
import mlflow
from langgraph.graph.state import CompiledStateGraph
from mlflow.pyfunc import ChatModel
from dao_ai.graph import create_dao_ai_graph
from dao_ai.models import create_agent 
from dao_ai.config import AppConfig

from loguru import logger

mlflow.langchain.autolog()

config: AppConfig = AppConfig.from_file(path=config_path)

log_level: str = config.app.log_level

logger.add(sys.stderr, level=log_level)

graph: CompiledStateGraph = create_dao_ai_graph(config=config)

app: ChatModel = create_agent(graph)

# COMMAND ----------

# DBTITLE 1,Check Evaluation Configuration and Print Details
from typing import Any
from rich import print as pprint
from dao_ai.config import EvaluationModel

evaluation: EvaluationModel = config.evaluation

if not evaluation:
  dbutils.notebook.exit("Missing evaluation configuration")
  
payload_table: str = evaluation.table.full_name
custom_inputs: dict[str, Any] = evaluation.custom_inputs

pprint(payload_table)
pprint(custom_inputs)

# COMMAND ----------

# DBTITLE 1,Load Model and Process Chat Messages
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from dao_ai.models import process_messages_stream, process_messages
from mlflow.types.llm import (
    ChatCompletionResponse,
)

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_model_name: str = config.app.registered_model.full_name
model_uri: str = f"models:/{registered_model_name}@Current"
model_version: ModelVersion = mlflow_client.get_model_version_by_alias(registered_model_name, "Current")

#loaded_agent = mlflow.pyfunc.load_model(model_uri)
def predict_fn(messages: dict[str, Any]) -> dict[str, Any]:
    print(f"messages={messages}")
    input = {"messages": messages}
    if custom_inputs:
        input["custom_inputs"] = custom_inputs
    response_content = ""
    for chunk in process_messages_stream(app, **input):
        # Handle different chunk types
        if hasattr(chunk, "content") and chunk.content:
            content = chunk.content
            response_content += content
        elif hasattr(chunk, "choices") and chunk.choices:
            # Handle ChatCompletionChunk format
            for choice in chunk.choices:
                if (
                    hasattr(choice, "delta")
                    and choice.delta
                    and choice.delta.content
                ):
                    content = choice.delta.content
                    response_content += content

    print(f"response_content={response_content}")

    outputs: dict[str, Any] = {
        "outputs": {
            "response": response_content,
        }
    }
    return outputs

# COMMAND ----------

# DBTITLE 1,- Define and Evaluate Scoring Functions for Responses
from mlflow.genai.scorers import scorer, Safety, Guidelines
from mlflow.entities import Feedback, Trace


@scorer
def response_completeness(outputs: dict[str, Any]) -> Feedback:

    print(f"outputs={outputs}")

    content = outputs["outputs"]["response"]

    # Outputs is return value of your app. Here we assume it's a string.
    if len(content.strip()) < 10:
        return Feedback(
            value=False,
            rationale="Response too short to be meaningful"
        )

    if content.lower().endswith(("...", "etc", "and so on")):
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

# DBTITLE 1,Convert Spark DataFrame to Pandas DataFrame
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

import pandas as pd
import json


df: DataFrame = spark.read.table(payload_table)
df = df.select("request_id", "inputs", "expectations")

eval_df: pd.DataFrame = df.toPandas()
display(eval_df)

# COMMAND ----------

# DBTITLE 1,- Evaluate Model with Custom Scorers and Log Results
import mlflow
from mlflow.models.model import ModelInfo
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.models.evaluation import EvaluationResult
import pandas as pd


model_info: mlflow.models.model.ModelInfo
evaluation_result: EvaluationResult

registered_model_name: str = config.app.registered_model.full_name

eval_df: pd.DataFrame = spark.read.table(payload_table).toPandas()

evaluation_table_name: str = config.evaluation.table.full_name

scorers = [Safety(), response_completeness, tool_call_efficiency]
custom_scorers = []

if config.evaluation.guidelines:
    custom_scorers = [Guidelines(name=guideline.name, guidelines=guideline.guidelines) for guideline in config.evaluation.guidelines]

scorers += custom_scorers

with mlflow.start_run(run_id=model_version.run_id):
  eval_results = mlflow.genai.evaluate(
      data=eval_df,
      predict_fn=predict_fn,
      model_id=model_version.model_id,
      scorers=scorers,
  )


