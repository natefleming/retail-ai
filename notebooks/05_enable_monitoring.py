# Databricks notebook source
# Dependency bootstrap, matching the pipeline step notebooks. Install dao-ai —
# which pulls its own transitive deps — from the newest ../dist wheel if one is
# there, else the published PyPI package. This notebook only reads config and
# calls the monitoring APIs, so bare dao-ai is enough. The spec is single-quoted
# in the magic so a dev wheel's ``+local`` version tag survives shell expansion.
#
# Note: with no ../dist wheel this installs the *published* dao-ai, not your
# working tree. Run ``uv build`` from the repo root first to test local changes.
import glob, os

from packaging.version import Version

# Newest by *version*, not by filename: a lexical sort puts 0.2.8 above
# 0.2.10. ``Version`` also parses a dev wheel's ``+local`` tag correctly.
def _wheel_version(wheel: str) -> Version:
    return Version(os.path.basename(wheel).split("-")[1])

_wheels = sorted(glob.glob("../dist/dao_ai-*.whl"), key=_wheel_version, reverse=True)
_dao_ai_dep = _wheels[0] if _wheels else "dao-ai"

# MAGIC %uv pip install --quiet '{_dao_ai_dep}'
# MAGIC %pip uninstall --quiet -y pyspark pyspark-connect
# MAGIC %restart_python

# COMMAND ----------

# There is no `../config` discovery fallback. That directory does not exist in a
# repo checkout, and discovery would pick the first YAML it happened to find, so
# `config-path` is the single input. It defaults to a shipped example; point it at
# any other config under ../examples.
dbutils.widgets.text(
    name="config-path",
    defaultValue="../examples/99_complete_applications/hardware_store/hardware_store.yaml",
)

widget_path: str | None = dbutils.widgets.get("config-path") or None
if not widget_path:
    raise ValueError(
        "No config to monitor: the `config-path` widget is empty. Set it to a "
        "config YAML — relative to this notebook (e.g. "
        "`../examples/99_complete_applications/hardware_store/hardware_store.yaml`) "
        "or an absolute workspace path."
    )

config_path: str = widget_path

print(config_path)

# COMMAND ----------

# DBTITLE 1,Add Source Directory to System Path
import sys, os, glob, subprocess

from packaging.version import Version

# Newest by *version*, not by filename: a lexical sort puts 0.2.8 above
# 0.2.10. ``Version`` also parses a dev wheel's ``+local`` tag correctly.
def _wheel_version(wheel: str) -> Version:
    return Version(os.path.basename(wheel).split("-")[1])

_wheels = sorted(
    glob.glob("../dist/dao_ai-*.whl") or glob.glob("../../artifacts/.internal/dao_ai-*.whl"),
    key=_wheel_version,
    reverse=True,
)
if _wheels:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall", _wheels[0]])
elif os.path.isdir("../src/dao_ai"):
    sys.path.insert(0, "../src")

# COMMAND ----------

# DBTITLE 1,Load Configuration
from dao_ai.config import AppConfig, MonitoringModel

config: AppConfig = AppConfig.from_file(path=config_path)

if not config.app or not config.app.trace_location:
    dbutils.notebook.exit("Missing app.trace_location configuration")

if not config.app.trace_location.monitoring:
    dbutils.notebook.exit("Missing app.trace_location.monitoring configuration")

monitoring: MonitoringModel = config.app.trace_location.monitoring

print(f"Built-in scorer sample rate: {monitoring.sample_rate}")
print(f"Guidelines scorer sample rate: {monitoring.guidelines_sample_rate}")
if monitoring.scorers:
    print(f"Configured scorers: {monitoring.scorers}")
else:
    print("Using all built-in scorers (default)")

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
    monitoring_config=monitoring,
    experiment_id=experiment_id,
    sql_warehouse_id=config.app.trace_location.warehouse_id,
)

print(f"\nRegistered {len(registered)} scorers for production monitoring")

# COMMAND ----------

# DBTITLE 1,Display Active Monitoring Scorers
from dao_ai.evaluation import get_monitoring_scorers

scorers = get_monitoring_scorers()
for s in scorers:
    print(f"  {s.name}: sample_rate={s.sample_rate}")

print(f"\nTotal active scorers: {len(scorers)}")
