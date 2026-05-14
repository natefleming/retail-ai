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
dbutils.widgets.dropdown(
    name="config-paths",
    choices=config_files,
    defaultValue=next(iter(config_files), ""),
)

config_path: str | None = dbutils.widgets.get("config-path") or None
project_path: str | None = dbutils.widgets.get("config-paths") or None
config_path: str = config_path or project_path or ""

print(f"Config path: {config_path}")

# COMMAND ----------

import sys
import glob
import subprocess
from importlib.metadata import version

_wheels: list[str] = sorted(
    glob.glob("../dist/dao_ai-*.whl") or glob.glob("../../artifacts/.internal/dao_ai-*.whl"),
    key=os.path.getmtime,
    reverse=True,
)
if _wheels:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall", _wheels[0]]
    )
elif os.path.isdir("../src/dao_ai"):
    sys.path.insert(0, "../src")

pip_requirements: Sequence[str] = (
    f"databricks-sdk=={version('databricks-sdk')}",
    f"mlflow=={version('mlflow')}",
)
print("\n".join(pip_requirements))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

# For each Genie room whose space_id is backed by a ${var.NAME} parameter,
# either reuse the configured space (when from_space_id returns a hydrated
# model) or create a fresh one. Forward each resolved id back to the
# associated parameter via taskValues so deploy-agents can inject it as a
# dao-ai parameter at config load time.

import json
from databricks.sdk import WorkspaceClient
from dao_ai.config import AppConfig, GenieRoomModel, value_of

config: AppConfig = AppConfig.from_file(path=config_path, initialize=False)
room_params: dict[str, str] = config.parameterized_genie_rooms()
print(f"Genie rooms with parameterized space_id: {room_params}")

provisioned: dict[str, str] = {}
if room_params:
    w: WorkspaceClient = WorkspaceClient()
    for room_key, param_name in room_params.items():
        room: GenieRoomModel = config.resources.genie_rooms[room_key]

        existing: GenieRoomModel | None = GenieRoomModel.from_space_id(
            value_of(room.space_id), w=w
        )
        if existing is not None:
            room.space_id = existing.space_id
            print(f"[{room_key}] reusing space {room.space_id}")
        else:
            room.space_id = None
            room.create(w=w)
            print(f"[{room_key}] created space {value_of(room.space_id)}")

        provisioned[param_name] = value_of(room.space_id)

# COMMAND ----------

summary: dict = {
    "room_params_discovered": room_params,
    "provisioned": provisioned,
}

if provisioned:
    payload: str = json.dumps(provisioned)
    dbutils.jobs.taskValues.set(key="genie_space_params", value=payload)
    print(f"Set taskValue genie_space_params = {payload}")
    summary["taskvalue_set"] = True
    summary["taskvalue_payload"] = payload
else:
    print("No parameterized Genie space_ids; no taskValues set.")
    summary["taskvalue_set"] = False

# Surface the summary as notebook_output so it's visible via
# `databricks jobs get-run-output` without scraping cluster logs.
dbutils.notebook.exit(json.dumps(summary))
