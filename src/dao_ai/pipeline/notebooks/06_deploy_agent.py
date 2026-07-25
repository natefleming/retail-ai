# Databricks notebook source
# Dependency bootstrap. Install dao-ai (which pulls its own transitive deps) via
# uv — the newest bundled ../dist wheel in development mode, else the published
# PyPI package. In a deployed job the serverless environment has already
# installed it; this reinstall is harmless. ``%restart_python`` makes the freshly
# installed package importable in the cells below.
import glob

_dao_ai_dep = next(iter(sorted(glob.glob("../dist/dao_ai-*.whl"), reverse=True)), "dao-ai")

# MAGIC %uv pip install --quiet {_dao_ai_dep}
# MAGIC %restart_python

# COMMAND ----------

from typing import Sequence
import os

def find_yaml_files_os_walk(base_path: str) -> Sequence[str]:
    # Tolerate a missing/non-dir base path: when the pipeline runs from a
    # wheel-only bundle an explicit `config-path` is always supplied, so the
    # `../config` discovery dropdown is optional. Return [] instead of raising.
    if not os.path.isdir(base_path):
        return []
    
    yaml_files = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.yaml', '.yml')):
                yaml_files.append(os.path.join(root, file))
    
    return sorted(yaml_files)

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")
dbutils.widgets.dropdown(
    name="deployment-target",
    choices=["", "model_serving", "apps", "mcp"],
    defaultValue="",
)
dbutils.widgets.dropdown(
    name="development",
    choices=["auto", "true", "false"],
    defaultValue="auto",
)

config_files: Sequence[str] = find_yaml_files_os_walk("../config")
dbutils.widgets.dropdown(name="config-paths", choices=config_files, defaultValue=next(iter(config_files), ""))

config_path: str | None = dbutils.widgets.get("config-path") or None
project_path: str = dbutils.widgets.get("config-paths") or None
deployment_target_str: str | None = dbutils.widgets.get("deployment-target") or None

config_path: str = config_path or project_path

print(f"Config path: {config_path}")
print(f"Deployment target: {deployment_target_str or '(using config default)'}")

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

import dao_ai.providers
import dao_ai.providers.base
import dao_ai.providers.databricks

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

from dao_ai.config import AppConfig, DeploymentTarget

config_path: str = dbutils.widgets.get("config-path") or dbutils.widgets.get("config-paths")
deployment_target_str: str | None = dbutils.widgets.get("deployment-target") or None

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
print(f"Deployment target: {deployment_target_str or '(using config default)'}")
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

deployment_target: DeploymentTarget
if deployment_target_str:
    deployment_target = DeploymentTarget(deployment_target_str)
    print(f"Using widget-specified deployment target: {deployment_target.value}")
else:
    deployment_target = DeploymentTarget.APPS
    print("Using default deployment target: apps")

# COMMAND ----------

config.display_graph()

# COMMAND ----------

# Only log/register the MLflow model for Model Serving deployments.
# Apps and MCP deploy directly from the config + PyPI package (no MLflow model registration).
if deployment_target not in (DeploymentTarget.APPS, DeploymentTarget.MCP):
    config.create_agent(development=development)

# COMMAND ----------

config.deploy_agent(target=deployment_target, development=development)
