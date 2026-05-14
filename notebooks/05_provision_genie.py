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

# Iterate each declared Genie room. For rooms whose space_id is bound
# to a `${var.NAME}` parameter (detected via `room.raw_space_id` and
# `is_parameter`), either reuse the configured space (via
# `from_space_id`) or create a fresh one (via `room.create()`), then
# forward the resolved id back to that parameter via taskValues so
# deploy-agents can inject it at config load time.

import json
from dao_ai.config import (
    AppConfig,
    GenieRoomModel,
    is_parameter,
    parameter_name,
    value_of,
)

config: AppConfig = AppConfig.from_file(path=config_path, initialize=False)
provisioned: dict[str, str] = {}

if config.resources is not None and config.resources.genie_rooms:
    for room_key, room in config.resources.genie_rooms.items():
        room: GenieRoomModel
        if not is_parameter(room.raw_space_id):
            print(f"[{room_key}] space_id is literal/unset; skipping")
            continue

        param: str = parameter_name(room.raw_space_id)

        # Resolution order:
        #   1. Configured space_id (e.g., operator pre-set via --var)
        #   2. Existing space matching this room's title (most-recent
        #      wins — idempotent across deploys without orphans)
        #   3. Provision a fresh space via room.create()
        existing: GenieRoomModel | None = (
            GenieRoomModel.from_space_id(value_of(room.space_id), w=room.workspace_client)
            or GenieRoomModel.from_name(value_of(room.name), w=room.workspace_client)
        )
        if existing is not None:
            room.space_id = existing.space_id
            print(f"[{room_key}] reusing space {room.space_id}")
        else:
            room.space_id = None
            room.create()
            print(f"[{room_key}] created space {value_of(room.space_id)}")

        resolved: str = value_of(room.space_id)
        provisioned[param] = resolved
        dbutils.jobs.taskValues.set(key=param, value=resolved)
        print(f"  Set taskValue {param} = {resolved}")

# COMMAND ----------

# Also set a consolidated JSON map so deploy-agents can read every
# resolved parameter via a single base_parameter (the bundle template
# can't reference N dynamic keys at template-write time).
summary: dict = {"provisioned": provisioned}
if provisioned:
    dbutils.jobs.taskValues.set(key="genie_space_params", value=json.dumps(provisioned))
    summary["taskvalue_set"] = True
else:
    print("No parameterized Genie space_ids; no taskValues set.")
    summary["taskvalue_set"] = False

dbutils.notebook.exit(json.dumps(summary))
