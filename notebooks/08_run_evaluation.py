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

# DBTITLE 1,Serialize Predictions With a Lock
# Evaluation runs `predict_fn` once per eval row. The dao-ai ResponsesAgent's
# sync `predict()` wraps `apredict()` with `asyncio.run` — a fresh event loop
# is created per call IN the calling thread. That keeps OpenTelemetry's
# contextvars-based span propagation intact, so langchain.autolog spans
# (LangGraph node, agent subgraphs, ChatUnityAIGateway, etc.) all nest under
# the per-row `evaluation` trace span.
#
# Earlier versions of this notebook used a dedicated event loop on a daemon
# thread plus `asyncio.run_coroutine_threadsafe`. That made traces incomplete:
# only the explicit `mlflow.start_span` calls (memory_context_search, vector
# search inner spans) survived the cross-thread context boundary. The
# autolog-instrumented spans ended up in orphan traces.
#
# A simple threading.Lock here keeps predictions serialized — mlflow.genai
# .evaluate may call predict_fn concurrently across rows, and the
# ResponsesAgent is not designed for parallel sync calls against the same
# graph instance.
import threading

_predict_lock = threading.Lock()

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

    from mlflow.entities import UCSchemaLocation

    _loc = config.app.trace_location
    mlflow.tracing.set_destination(
        destination=UCSchemaLocation(
            catalog_name=_loc.catalog_name,
            schema_name=_loc.schema_name,
        )
    )

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
from typing import Any, Optional

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from dao_ai.models import get_latest_model_version

# `deployment_target` controls how the agent reaches end users:
#   - "model_serving" / "both": agent is registered + versioned in Unity Catalog,
#     served via a Model Serving endpoint. Evaluation can `mlflow.load_model` it.
#   - "apps": agent runs in-process inside a Databricks App container ONLY. No
#     UC model is registered. The `config.as_responses_agent()` call at the top
#     of this notebook already gave us a fully-functional `app` object to
#     evaluate against — skip the UC fetch.
deployment_target: str = getattr(config.app, "deployment_target", None) or "model_serving"
print(f"Deployment target: {deployment_target}")

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_model_name: Optional[str] = None
latest_version: Optional[int] = None
model_uri: Optional[str] = None
model_version: Optional[ModelVersion] = None

if deployment_target in ("model_serving", "both"):
    registered_model_name = config.app.registered_model.full_name
    latest_version = get_latest_model_version(registered_model_name)
    model_uri = f"models:/{registered_model_name}/{latest_version}"
    model_version = mlflow_client.get_model_version(registered_model_name, str(latest_version))
    print(f"Loaded UC model: {model_uri}")
else:
    print(
        "Skipping UC model fetch — deployment_target is 'apps' so no UC model "
        "is registered. Evaluating the in-process ResponsesAgent directly."
    )

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


# Evaluation session marker — every per-row request gets a conversation_id
# prefixed with `eval-<timestamp>-<short_id>-row<N>` so the session field
# in the trace clearly identifies which traces came from this eval run vs
# interactive App use. The shared prefix groups all per-row sessions from
# one eval invocation together.
import uuid
from datetime import datetime
_EVAL_SESSION_PREFIX: str = (
    f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
)
print(f"Evaluation session prefix: {_EVAL_SESSION_PREFIX}")


def _run_prediction(messages: list[dict[str, Any]], custom_inputs: dict[str, Any] | None) -> str:
    with _predict_lock:
        # Build a fresh, eval-marked custom_inputs per row so the session.
        # conversation_id surfaces this trace as eval-origin in the UI.
        # We copy rather than mutate the shared `custom_inputs` dict so
        # successive rows don't accumulate stale values.
        row_num = _predict_counter["current"]
        eval_inputs: dict[str, Any] = dict(custom_inputs or {})
        eval_session: dict[str, Any] = dict(eval_inputs.get("session") or {})
        eval_session["conversation_id"] = f"{_EVAL_SESSION_PREFIX}-row{row_num}"
        eval_inputs["session"] = eval_session

        request = ResponsesAgentRequest(
            input=[{"role": m["role"], "content": m["content"]} for m in messages],
            custom_inputs=eval_inputs,
        )
        # Use the sync `predict()` wrapper (which calls asyncio.run() in the
        # same thread) instead of dispatching to a separate event-loop thread.
        # This preserves OpenTelemetry's contextvars-based span propagation
        # so langchain.autolog spans nest correctly under the per-row
        # `evaluation` trace.
        response: ResponsesAgentResponse = app.predict(request)
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

# Resolve the experiment to log this eval run into:
#   - model_serving / both → the experiment associated with the UC-registered
#     model run (matches the model version, so results stay attached to that
#     specific model version)
#   - apps → there's no UC model run to anchor to, so use (or create) a
#     workspace experiment named after the app. mlflow.set_experiment will
#     create it on first invocation.
if model_version is not None:
    model_run = mlflow_client.get_run(model_version.run_id)
    experiment_id = model_run.info.experiment_id
else:
    model_run = None
    # Use the app name to pick a stable experiment path under the deploying
    # user's home directory. This keeps eval runs grouped per-app for
    # apps-only deploys where no UC model anchors the experiment.
    workspace_user = (
        spark.conf.get("spark.databricks.workspaceUrl", "") and
        dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        .userName().get()
    )
    experiment_path = f"/Users/{workspace_user}/{config.app.name}_evaluation"
    experiment_id = mlflow.set_experiment(experiment_path).experiment_id
    print(f"Apps-only deploy: created/using workspace experiment {experiment_path}")

mlflow.set_experiment(experiment_id=experiment_id)

eval_dataset = create_or_get_eval_dataset(
    name=f"{payload_table}_dataset",
    experiment_id=experiment_id,
    source_df=eval_df,
    replace=evaluation.replace,
)

experiment = mlflow.get_experiment(experiment_id)
print(f"Dataset name:      {eval_dataset.name}")
print(f"Dataset ID:        {eval_dataset.dataset_id}")
print(f"Dataset source:    {eval_dataset.source_type}")
print(f"Dataset records:   {len(eval_dataset.to_df())} rows")
print(f"Experiment name:   {experiment.name}")
print(f"Experiment ID:     {experiment_id}")

# Note: a previous version disabled all autologs (`mlflow.autolog(disable=True)`)
# then re-enabled langchain. The disable-all side-effected langchain too in
# practice — autolog patches that had been registered earlier no longer fired
# during mlflow.genai.evaluate, producing eval traces that were missing the
# LangGraph subgraph + agent nodes + ChatUnityAIGateway model-call spans (only
# explicit `mlflow.start_span` markers survived). We rely on the autolog call
# at the top of the notebook (after `import mlflow`) staying active for the
# duration of the run instead.

run_tags: dict[str, str] = {
    k: str(v) for k, v in (config.app.tags or {}).items()
}
run_tags["run_type"] = "evaluation"
run_tags["deployment_target"] = deployment_target

# Run name suffix:
#   - UC-registered models → include the model version so each registered
#     version has its eval runs cleanly grouped
#   - apps-only → no version exists, so just use the timestamp
_version_suffix = f"v{latest_version}_" if latest_version is not None else ""
run_name: str = f"{config.app.name}_evaluation_{_version_suffix}{datetime.now().strftime('%Y%m%d_%H%M%S')}"

_predict_counter["total"] = len(eval_df)
print(f"Starting evaluation: {len(eval_df)} rows, {len(scorers)} scorers")

# `model_id` ties the eval to a specific registered model version in the
# MLflow UI. For apps-only deploys there's no model id to bind, so we omit
# the kwarg — mlflow.genai.evaluate treats `model_id=None` as unbound, which
# is the right semantics for an in-process agent eval.
_evaluate_kwargs = dict(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=scorers,
)
if model_version is not None:
    _evaluate_kwargs["model_id"] = model_version.model_id

with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
    try:
        eval_results: EvaluationResult = mlflow.genai.evaluate(**_evaluate_kwargs)
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
