# Databricks notebook source
# Dependency bootstrap. Install dao-ai (which pulls its own transitive deps) via
# uv — the newest bundled ../dist wheel in development mode, else the published
# PyPI package. In a deployed job the serverless environment has already
# installed it; this reinstall is harmless. ``%restart_python`` makes the freshly
# installed package importable in the cells below.
# This notebook builds the agent graph (display_graph / create_agent below), so
# it must install every optional feature extra — the graph is built before the
# config is known, and any agent may use memory (langmem), a2a, deepagents,
# rerank, or search. Hence ``[all]``. The install spec is single-quoted in the
# magic so a dev wheel's ``+local`` version tag and the ``[all]`` bracket survive
# shell glob/bracket expansion.
import glob

_dao_ai_dep = (
    next(iter(sorted(glob.glob("../dist/dao_ai-*.whl"), reverse=True)), "dao-ai")
    + "[all]"
)

# MAGIC %uv pip install --quiet '{_dao_ai_dep}'
# MAGIC %restart_python

# COMMAND ----------

from dao_ai.utils import find_config_files

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")
dbutils.widgets.dropdown(
    name="mode",
    choices=["", "model_serving", "apps", "mcp"],
    defaultValue="",
)
dbutils.widgets.dropdown(
    name="development",
    choices=["auto", "true", "false"],
    defaultValue="auto",
)

config_files: list[str] = find_config_files("../config")
dbutils.widgets.dropdown(
    name="config-paths", choices=config_files, defaultValue=next(iter(config_files), "")
)

explicit_path: str | None = dbutils.widgets.get("config-path") or None
discovered_path: str | None = dbutils.widgets.get("config-paths") or None
mode_str: str | None = dbutils.widgets.get("mode") or None

# An explicit `config-path` always wins; the `../config` dropdown is the fallback
# for an interactive run. Re-binding `config_path` from `str | None` to `str` would
# make the annotation a lie (and hide that `from_file` rejects None), so the inputs
# get their own names and the result is annotated once.
resolved_path: str | None = explicit_path or discovered_path
if not resolved_path:
    raise ValueError(
        "No config to deploy: the `config-path` widget is empty and no YAML was "
        "found under ../config. Set `config-path` to the staged config (the "
        "pipeline bundle always passes it)."
    )

config_path: str = resolved_path

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
    print(f"Using widget-specified serving mode: {mode.value}")
else:
    mode = ServingMode.APPS
    print("Using default serving mode: apps")

# COMMAND ----------

config.display_graph()

# COMMAND ----------

# Only log/register the MLflow model for Model Serving deployments.
# Apps and MCP deploy directly from the config + PyPI package (no MLflow model registration).
if mode not in (ServingMode.APPS, ServingMode.MCP):
    config.create_agent(development=development)

# COMMAND ----------

config.deploy_agent(mode=mode, development=development)
