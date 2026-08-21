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
    _dao_ai_dep = os.path.join("..", _pin.lstrip("./"))
elif _pin:
    _dao_ai_dep = _pin
else:
    _dao_ai_dep = (_wheels[0] if _wheels else "dao-ai") + "[all]"

# MAGIC %uv pip install --quiet '{_dao_ai_dep}'
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")
dbutils.widgets.dropdown(
    name="mode",
    choices=["", "model_serving", "apps"],
    defaultValue="",
)
dbutils.widgets.dropdown(
    name="as_mcp",
    choices=["false", "true"],
    defaultValue="false",
)
dbutils.widgets.dropdown(
    name="development",
    choices=["auto", "true", "false"],
    defaultValue="auto",
)

mode_str: str | None = dbutils.widgets.get("mode") or None
as_mcp: bool = (dbutils.widgets.get("as_mcp") or "false").lower() == "true"

# There is no `../config` discovery fallback. That directory exists only in the
# staged bundle layout, and the bundle stages exactly one config — the same one
# the job passes here — so discovery could only ever guess, and guessing is how
# the wrong config gets deployed. `config-path` is the single input.
widget_path: str | None = dbutils.widgets.get("config-path") or None
if not widget_path:
    raise ValueError(
        "No config to deploy: the `config-path` widget is empty. In a staged "
        "pipeline bundle the config sits beside this notebook under `../config/` "
        "and the job always passes it; running this notebook by hand, set "
        "`config-path` to an absolute workspace path, for example "
        "`/Workspace/Users/you@example.com/dao-ai/examples/04_genie/genie_basic.yaml`."
    )

config_path: str = widget_path

print(f"Config path: {config_path}")
print(f"Serving mode: {mode_str or '(using config default)'}")

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

# COMMAND ----------
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# COMMAND ----------
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

import nest_asyncio

nest_asyncio.apply()

# COMMAND ----------

# Serverless v5 FIPS: select psycopg's pure-Python impl before importing
# dao_ai.config, which transitively imports psycopg via databricks-langchain;
# the binary wheel's vendored OpenSSL aborts (SIGABRT) on import. Job-scoped.
import os

os.environ["PSYCOPG_IMPL"] = "python"

# COMMAND ----------

from dao_ai.config import AppConfig, ServingMode

# `config_path` and `mode_str` were already resolved (and validated) in the widget
# cell above. Re-reading the widgets here would bypass that empty-path guard and
# re-introduce the `str | None` the guard exists to rule out.

# Source selection tri-state forwarded by `dao-ai generate-workflow` via the
# `development` bundle var: "true" ships local dao-ai source/wheel, "false"
# pins the published PyPI package, "auto"/"" auto-detects from the install
# type (None -> create_agent/deploy_agent resolve via is_published()).
development_str: str = (dbutils.widgets.get("development") or "auto").lower()
development: bool | None
if development_str == "true":
    development = True
elif development_str == "false":
    development = False
else:
    development = None

print(f"Config path: {config_path}")
print(f"Serving mode: {mode_str or '(using config default)'}")
print(f"Development source: {development_str}")

# Pull any resolved parameter values that upstream provisioning tasks
# (e.g. provision-genie) forwarded via job taskValues. AppConfig.from_file
# probes the declared parameters block against taskValues.get(taskKey=...,
# key=<param_name>) and folds non-empty results into substitution.
config: AppConfig = AppConfig.from_file(
    path=config_path,
    task_values=dbutils.jobs.taskValues,
    task_key="provision-genie",
)
print(f"Substituted parameters: {config.substitution_vars}")

mode: ServingMode
if mode_str:
    mode = ServingMode(mode_str)
    print(f"Using widget-specified serving platform: {mode.value}")
else:
    mode = ServingMode.APPS
    print("Using default serving platform: apps")

if as_mcp:
    if mode is not ServingMode.MODEL_SERVING:
        print("Serving over MCP instead of the chat UI (deploys as mcp-<app>)")
    else:
        # MCP runs on the Apps runtime; there is no Model Serving MCP surface.
        raise ValueError(
            "as_mcp=true requires mode=apps (MCP is served on the Databricks "
            f"Apps runtime); got mode={mode.value}"
        )

# COMMAND ----------

config.display_graph()

# COMMAND ----------

# Only log/register the MLflow model for Model Serving deployments. Apps deploy
# directly from the config + PyPI package (no MLflow model registration), whether
# they serve the chat UI or the MCP server.
if mode == ServingMode.MODEL_SERVING:
    config.create_agent(development=development)

# COMMAND ----------

config.deploy_agent(mode=mode, development=development, as_mcp=as_mcp)
