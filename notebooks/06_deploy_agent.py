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
dbutils.widgets.dropdown(
    name="deployment-target",
    choices=["", "model_serving", "apps", "both"],
    defaultValue="",
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

# MAGIC %sh
# MAGIC # Build a dao-ai wheel from source if no pre-built wheel exists
# MAGIC if [ -z "$(ls ../dist/dao_ai-*.whl 2>/dev/null)" ] && [ -d "../src/dao_ai" ] && [ -f "../pyproject.toml" ]; then
# MAGIC   echo "dao-ai source: building wheel from source..."
# MAGIC   uv build --wheel --out-dir ../dist ..
# MAGIC fi

# COMMAND ----------

import sys, os, glob, subprocess

_wheels = sorted(glob.glob("../dist/dao_ai-*.whl") or glob.glob("../../artifacts/.internal/dao_ai-*.whl"))
if _wheels:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall", _wheels[-1]])
    print(f"dao-ai source: local wheel ({os.path.basename(_wheels[-1])})")
elif os.path.isdir("../src/dao_ai"):
    sys.path.insert(0, "../src")
    print("dao-ai source: source path")
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "dao-ai"])
    print("dao-ai source: PyPI")

# COMMAND ----------

try:
    from importlib.metadata import version as _meta_version
    print(f"dao-ai version: {_meta_version('dao-ai')}")
except Exception:
    print("dao-ai version: dev (source path)")

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

from typing import Sequence
from dao_ai.config import AppConfig, DeploymentTarget

from dao_ai.config import is_parameter, parameter_name

config_path: str = dbutils.widgets.get("config-path") or dbutils.widgets.get("config-paths")
deployment_target_str: str | None = dbutils.widgets.get("deployment-target") or None

print(f"Config path: {config_path}")
print(f"Deployment target: {deployment_target_str or '(using config default)'}")

# For each Genie room whose space_id is bound to a `${var.NAME}`
# parameter, pull the matching taskValue from provision-genie (key=NAME)
# and add it to the params dict. Symmetric with provision-genie, which
# sets `taskValues.set(key=NAME, value=<resolved_space_id>)` for each
# parameterized room.
config_for_lookup: AppConfig = AppConfig.from_file(path=config_path, initialize=False)
params: dict[str, str] = {}
if config_for_lookup.resources is not None and config_for_lookup.resources.genie_rooms:
    for room in config_for_lookup.resources.genie_rooms.values():
        if not is_parameter(room.raw_space_id):
            continue
        name: str = parameter_name(room.raw_space_id)
        try:
            val: str = dbutils.jobs.taskValues.get(
                taskKey="provision-genie",
                key=name,
                default="",
                debugValue="",
            )
        except Exception:
            continue
        if val:
            params[name] = val

print(f"Resolved params from provision-genie taskValues: {params}")

config: AppConfig = AppConfig.from_file(path=config_path, params=params or None)

deployment_target: DeploymentTarget
if deployment_target_str:
    deployment_target = DeploymentTarget(deployment_target_str)
    print(f"Using widget-specified deployment target: {deployment_target.value}")
elif config.app and config.app.deployment_target:
    deployment_target = config.app.deployment_target
    print(f"Using config file deployment target: {deployment_target.value}")
else:
    deployment_target = DeploymentTarget.MODEL_SERVING
    print("Using default deployment target: model_serving")

# COMMAND ----------

config.display_graph()

# COMMAND ----------

# Only log/register the MLflow model for Model Serving deployments.
# Apps deploy directly from the config + PyPI package.
if deployment_target != DeploymentTarget.APPS:
    config.create_agent()

# COMMAND ----------

if deployment_target == DeploymentTarget.BOTH:
    config.deploy_agent(target=DeploymentTarget.MODEL_SERVING)
    config.deploy_agent(target=DeploymentTarget.APPS)
else:
    config.deploy_agent(target=deployment_target)
