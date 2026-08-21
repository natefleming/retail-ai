# Databricks notebook source
# Dependency bootstrap. Install dao-ai via uv, self-contained (stdlib only, no
# dao_ai import — a bootstrap must not import the package it installs). Prefer the
# deploy's pinned ``dao_ai_dep`` parameter (a ./dist wheel re-anchored to ../dist in
# development, else a version/PyPI spec); fall back to the newest local ../dist wheel,
# else PyPI, only for standalone runs. ``%restart_python`` makes the freshly installed
# package importable below. The spec is single-quoted in the magic so a dev wheel's
# ``+local`` tag and any ``[extras]`` survive shell expansion.
import glob, os

from packaging.version import Version


# Newest by *version*, not by filename: a lexical sort puts 0.2.8 above 0.2.10.
def _wheel_version(wheel: str) -> Version:
    return Version(os.path.basename(wheel).split("-")[1])


_wheels = sorted(glob.glob("../dist/dao_ai-*.whl"), key=_wheel_version, reverse=True)
dbutils.widgets.text(name="dao_ai_dep", defaultValue="")
_pin = dbutils.widgets.get("dao_ai_dep")
if _pin.endswith(".whl"):
    _dao_ai_dep = os.path.join("..", _pin.removeprefix("./"))
elif _pin:
    _dao_ai_dep = _pin
else:
    _dao_ai_dep = (_wheels[0] if _wheels else "dao-ai") + "[all]"

# MAGIC %uv pip install --quiet '{_dao_ai_dep}'
# MAGIC %restart_python

# COMMAND ----------

# Record the installed dao-ai version plus the key libraries resolved under
# it, so each run's logs capture exactly what was installed. Alphabetical; the
# list is short and hand-curated.
from importlib.metadata import version

print(f"dao-ai=={version('dao-ai')}")
print(f"databricks-langchain=={version('databricks-langchain')}")
print(f"databricks-sdk=={version('databricks-sdk')}")
print(f"langchain=={version('langchain')}")
print(f"langgraph=={version('langgraph')}")
print(f"mlflow=={version('mlflow')}")

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

# There is no `../config` discovery fallback. That directory exists only in the
# staged bundle layout, and the bundle stages exactly one config — the same one
# the job passes here — so discovery could only ever guess, and guessing is how
# the wrong config gets loaded. `config-path` is the single input.
widget_path: str | None = dbutils.widgets.get("config-path") or None
if not widget_path:
    raise ValueError(
        "No config to evaluate against: the `config-path` widget is empty. In a "
        "staged pipeline bundle the config sits beside this notebook under "
        "`../config/` and the job always passes it; running this notebook by "
        "hand, set `config-path` to an absolute workspace path, for example "
        "`/Workspace/Users/you@example.com/dao-ai/examples/04_genie/genie_basic.yaml`."
    )

config_path: str = widget_path
print(config_path)

# COMMAND ----------


# COMMAND ----------
# DBTITLE 1,Load Application Config
# Serverless v5 FIPS: select psycopg's pure-Python impl before importing
# dao_ai.config, which transitively imports psycopg via databricks-langchain;
# the binary wheel's vendored OpenSSL aborts (SIGABRT) on import. Job-scoped.
import os

os.environ["PSYCOPG_IMPL"] = "python"

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
custom_inputs: dict[str, Any] = (
    _eval_custom_inputs or _input_example_custom_inputs or {}
)
_custom_inputs_source: str = (
    "evaluation"
    if _eval_custom_inputs
    else "input_example"
    if _input_example_custom_inputs
    else "empty"
)

print(f"Evaluation table:        {payload_table}")
print(f"Custom inputs (source={_custom_inputs_source}): {custom_inputs}")

# COMMAND ----------

# DBTITLE 1,Build Shared Agent — source derived from serving mode
#
# The eval source is driven by the same ``mode`` the deploy task used, not a
# free-standing knob — only ``model_serving`` logs/registers an MLflow model
# (see 07_deploy_agent.py), so only it can be evaluated from the UC registry:
#
#   - model_serving → "registry": load the deployed UC artifact via
#     ``mlflow.pyfunc.load_model`` — the most honest eval (test what shipped).
#   - apps / mcp    → "config":   build the agent in-process from the YAML via
#     ``AppConfig.as_responses_agent()``. These modes register no model, so
#     there is no artifact to load; the app itself also builds from config at
#     runtime, so this evaluates equivalent code.
#
# ``apps + registry`` / ``mcp + registry`` are unrepresentable here by design —
# that invalid combination was the source of RESOURCE_DOES_NOT_EXIST failures
# when eval defaulted to the registry regardless of deploy mode.
#
# Both paths reach the same LanggraphResponsesAgent with the same async
# singletons (Lakebase checkpointer, langgraph saver). autolog with
# run_tracer_inline matches the Apps handler (src/dao_ai/apps/handlers.py:52)
# so child spans (LLM, tool, retriever) are complete on the config path too.
import mlflow

from dao_ai.config import ServingMode

mlflow.set_registry_uri("databricks-uc")
mlflow.langchain.autolog(run_tracer_inline=True)

dbutils.widgets.dropdown(
    name="mode", choices=["", "model_serving", "apps"], defaultValue=""
)
mode_str: str = dbutils.widgets.get("mode") or ServingMode.APPS.value
mode: ServingMode = ServingMode(mode_str)

# Downstream cells need three values regardless of source:
#   experiment_id — where the eval run + traces land
#   version_label — run-name suffix
#   eval_model_id — mlflow.genai.evaluate model_id (None on the config path;
#                   there is no logged model version to attribute to)
mlflow_client: "MlflowClient"  # noqa: F821 - set in both branches below
experiment_id: str
version_label: str
eval_model_id: str | None

if mode == ServingMode.MODEL_SERVING:
    from mlflow import MlflowClient
    from mlflow.entities.model_registry.model_version import ModelVersion

    from dao_ai.models import get_latest_model_version

    registered_model_name: str = config.app.registered_model.full_name
    latest_version: int = get_latest_model_version(registered_model_name)
    model_uri: str = f"models:/{registered_model_name}/{latest_version}"
    mlflow_client = MlflowClient()
    model_version: ModelVersion = mlflow_client.get_model_version(
        registered_model_name, str(latest_version)
    )
    print(f"Registered model: {registered_model_name}")
    print(f"Latest version:   {latest_version}")
    print(f"Model ID:         {model_version.model_id}")
    app: Any = mlflow.pyfunc.load_model(model_uri)
    agent_source = "registry"
    experiment_id = mlflow_client.get_run(model_version.run_id).info.experiment_id
    version_label = f"v{latest_version}"
    eval_model_id = model_version.model_id
else:
    from mlflow import MlflowClient

    app = config.as_responses_agent()
    agent_source = "config"
    mlflow_client = MlflowClient()
    # No registered model on the apps path — resolve the experiment the
    # app writes to: its configured MLflow experiment if set, else the
    # deploy-time bundle experiment ``/Users/<current-user>/<app_resource_name>``.
    # ``set_experiment`` is get-or-create, so the eval run + traces always have
    # a home even on the first apps deploy. Deliberately UNPREFIXED even when the
    # agent is also deployed as an MCP server: eval builds the agent in-process
    # from the config (``as_responses_agent``) rather than calling a deployed
    # App, so it belongs with the chat App's experiment.
    resolved: str | None = (
        config.app.experiment.resolved_id if config.app.experiment is not None else None
    )
    if resolved:
        experiment_id = resolved
    else:
        current_user: str = spark.sql("SELECT current_user()").collect()[0][0]
        experiment_name: str = f"/Users/{current_user}/{config.app.app_resource_name}"
        experiment_id = mlflow.set_experiment(experiment_name).experiment_id
    version_label = "config"
    eval_model_id = None

print(f"Serving mode: {mode.value} → agent source: {agent_source}")
print(f"Experiment ID:    {experiment_id}")

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
    build_scorers,
    create_or_get_eval_dataset,
    prepare_eval_dataframe,
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
print(f"Dataset records:   {len(eval_dataset.to_df())} rows")
print(f"Experiment name:   {experiment.name}")
print(f"Experiment ID:     {experiment_id}")

run_tags: dict[str, str] = {k: str(v) for k, v in (config.app.tags or {}).items()}
run_tags["run_type"] = "evaluation"
run_name: str = f"{config.app.name}_evaluation_{version_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"Starting evaluation: {len(eval_df)} rows, {len(scorers)} scorers")

with mlflow.start_run(run_name=run_name, tags=run_tags) as run:
    try:
        # model_id is None on the apps/mcp (config) path — there is no logged
        # model version to attribute the eval to; genai.evaluate accepts None.
        eval_results: EvaluationResult | None = mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=predict_fn,
            model_id=eval_model_id,
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
        print(
            "No detailed results table available. Available tables:",
            list(eval_results.tables.keys()),
        )
else:
    print(
        "Evaluation results not available. Check the MLflow run for logged metrics and traces."
    )
