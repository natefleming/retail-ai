# Databricks notebook source
# MAGIC %pip install --quiet --upgrade -r ../requirements.txt
# MAGIC %pip uninstall --quiet -y pyspark pyspark-connect
# MAGIC %restart_python

# COMMAND ----------

from typing import Sequence
import os

def find_yaml_files_os_walk(base_path: str) -> Sequence[str]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    if not os.path.isdir(base_path):
        raise NotADirectoryError(f"Base path is not a directory: {base_path}")

    yaml_files: list[str] = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith((".yaml", ".yml")):
                yaml_files.append(os.path.join(root, file))
    return sorted(yaml_files)

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

config_files: Sequence[str] = find_yaml_files_os_walk("../config")
dbutils.widgets.dropdown(name="config-paths", choices=config_files, defaultValue=next(iter(config_files), ""))

config_path: str = dbutils.widgets.get("config-path") or dbutils.widgets.get("config-paths")
print(config_path)

# COMMAND ----------

# DBTITLE 1,Add Source Directory to System Path
import sys, os, glob, subprocess

# Install dao-ai from local wheel (bundle artifact or manual build), or fall back to source path
_wheels = sorted(
    glob.glob("../dist/dao_ai-*.whl") or glob.glob("../../artifacts/.internal/dao_ai-*.whl"),
    key=os.path.getmtime,
    reverse=True,
)
if _wheels:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall", _wheels[0]])
elif os.path.isdir("../src/dao_ai"):
    sys.path.insert(0, "../src")

# COMMAND ----------

import dao_ai.providers
import dao_ai.providers.base
import dao_ai.providers.databricks
import dao_ai.memory.postgres
import dao_ai.memory.databricks

# COMMAND ----------

# DBTITLE 1,Load Application Config
from dao_ai.config import AppConfig, EvaluationModel
from dao_ai.logging import configure_logging

config: AppConfig = AppConfig.from_file(path=config_path)
configure_logging(level=config.app.log_level)

# COMMAND ----------

# DBTITLE 1,Validate Evaluation Configuration
from typing import Any

evaluation: EvaluationModel = config.evaluation
if not evaluation:
    dbutils.notebook.exit("Missing evaluation configuration")

payload_table: str = evaluation.table.full_name
# Fall back to app.input_example.custom_inputs when evaluation.custom_inputs is
# empty. Required for configs that wire validation middleware (e.g.
# store_validation in sporting_goods_store*) but omit evaluation.custom_inputs;
# without this the middleware raises on every row. app.input_example is a
# ChatPayload Pydantic model (config.py:7547) with a Dict custom_inputs field.
_eval_custom_inputs: dict[str, Any] = evaluation.custom_inputs or {}
_input_example_custom_inputs: dict[str, Any] = (
    getattr(config.app.input_example, "custom_inputs", None) or {}
)
custom_inputs: dict[str, Any] = _eval_custom_inputs or _input_example_custom_inputs or {}
_custom_inputs_source: str = (
    "evaluation" if _eval_custom_inputs
    else "input_example" if _input_example_custom_inputs
    else "empty"
)

print(f"Evaluation table:        {payload_table}")
print(f"Custom inputs (source={_custom_inputs_source}): {custom_inputs}")

# COMMAND ----------

# DBTITLE 1,Resolve UC Model Version
import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from dao_ai.models import get_latest_model_version

mlflow.set_registry_uri("databricks-uc")
# Match the Apps handler (src/dao_ai/apps/handlers.py:52) so config-mode runs
# produce complete child spans (LLM, tool, retriever). Idempotent for the
# registry path where pyfunc-load already wires it.
mlflow.langchain.autolog(run_tracer_inline=True)
mlflow_client: MlflowClient = MlflowClient()

registered_model_name: str = config.app.registered_model.full_name
latest_version: int = get_latest_model_version(registered_model_name)
model_uri: str = f"models:/{registered_model_name}/{latest_version}"
model_version: ModelVersion = mlflow_client.get_model_version(registered_model_name, str(latest_version))

print(f"Registered model: {registered_model_name}")
print(f"Latest version:   {latest_version}")
print(f"Model ID:         {model_version.model_id}")

# COMMAND ----------

# DBTITLE 1,Build Shared Agent — UC Registry (default) or Local Config
#
# Two supported eval sources:
#   - "registry" loads the actual UC-registered artifact via mlflow.pyfunc.load_model.
#     Most honest eval (you test what's deployed). Requires a successful deploy first.
#   - "config" builds in-process from the local YAML via AppConfig.as_responses_agent().
#     Fast iteration on the YAML without redeploying.
#
# Both paths reach the same LanggraphResponsesAgent code with the same async
# singletons (Lakebase checkpointer, langgraph saver). The persistent-loop
# predict_fn below pins those singletons to a single loop for the whole run.
from typing import Literal

dbutils.widgets.dropdown(name="agent-source", defaultValue="registry", choices=["registry", "config"])
agent_source: Literal["registry", "config"] = dbutils.widgets.get("agent-source")  # type: ignore[assignment]

if agent_source == "registry":
    app: Any = mlflow.pyfunc.load_model(model_uri)
else:
    app = config.as_responses_agent()

print(f"Agent source: {agent_source}")

# COMMAND ----------

# DBTITLE 1,Define predict_fn — sync wrapper over a persistent asyncio loop
#
# mlflow.genai.evaluate calls predict_fn per row. Both available paths to the
# agent create a fresh event loop per row:
#   - registry mode: pyfunc.load_model.predict -> LanggraphResponsesAgent.predict
#     -> asyncio.run(self.apredict(...)) at src/dao_ai/models.py:1599
#   - config mode (async predict_fn): the harness wraps each row in asyncio.run
#
# The shared agent's Lakebase checkpointer binds an asyncio.Lock to the loop
# that touches it first. Row 2's fresh loop then trips
#   RuntimeError: <asyncio.locks.Lock> is bound to a different event loop
# at langgraph/pregel/_loop.py:1888 (await checkpointer.aget_tuple(...)).
# Reproduced live on trace tr-c06b0a1c8d3147aded17809bd20207d3
# (experiment 2481319859454651).
#
# Fix: pin every apredict to ONE long-lived loop running in a daemon thread,
# and submit row coroutines via asyncio.run_coroutine_threadsafe. The Lock
# binds to this loop once and stays valid for the whole eval run.
# Vault memory feedback_no_eval_accommodation_in_runtime.md mandates we fix
# this in the harness, never in dao-ai runtime.
import asyncio
import threading

from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

# Idempotent across interactive cell re-runs: if the loop already exists and is
# alive, reuse it so the agent's Lakebase Lock binding stays valid. Re-running
# this cell with a fresh loop while the previous agent still binds the Lock to
# the old loop would re-introduce the very bug we're fixing.
if "_eval_loop" not in globals() or _eval_loop.is_closed():  # type: ignore[name-defined]
    _eval_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
    _eval_loop_thread: threading.Thread = threading.Thread(
        target=_eval_loop.run_forever, name="dao-ai-eval-loop", daemon=True
    )
    _eval_loop_thread.start()

# Unwrap pyfunc to drive LanggraphResponsesAgent.apredict directly and skip the
# sync .predict() wrapper at src/dao_ai/models.py:1599. ResponsesAgent-flavored
# pyfunc models use _ResponsesAgentPyfuncWrapper (mlflow/pyfunc/loaders/
# responses_agent.py) which exposes the underlying agent via get_raw_model();
# unwrap_python_model() does NOT work here (that returns _model_impl.python_model
# which only exists on the generic PythonModel wrapper).
if agent_source == "registry":
    agent: Any = app._model_impl.get_raw_model()
else:
    agent = app


def predict_fn(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sync predict_fn so mlflow.genai.evaluate does NOT wrap in asyncio.run."""
    request: ResponsesAgentRequest = ResponsesAgentRequest(
        input=[{"role": m["role"], "content": m["content"]} for m in messages],
        custom_inputs=custom_inputs,
    )
    future = asyncio.run_coroutine_threadsafe(agent.apredict(request), _eval_loop)
    response: ResponsesAgentResponse = future.result(timeout=300)
    return response.model_dump()

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
print(f"Dataset records:   {len(eval_dataset.to_df())} rows")
print(f"Experiment name:   {experiment.name}")
print(f"Experiment ID:     {model_run.info.experiment_id}")

run_tags: dict[str, str] = {k: str(v) for k, v in (config.app.tags or {}).items()}
run_tags["run_type"] = "evaluation"
run_name: str = f"{config.app.name}_evaluation_v{latest_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"Starting evaluation: {len(eval_df)} rows, {len(scorers)} scorers")

with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
    try:
        eval_results: EvaluationResult | None = mlflow.genai.evaluate(
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

    eval_results_df: pd.DataFrame = prepare_eval_results_for_display(eval_results)
    print(f"Total evaluation results: {len(eval_results_df)} rows")
    if not eval_results_df.empty:
        display(eval_results_df.head(100))
    else:
        print("No detailed results table available. Available tables:", list(eval_results.tables.keys()))
else:
    print("Evaluation results not available. Check the MLflow run for logged metrics and traces.")
