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

# Discover which Genie rooms in the YAML use a `${var.<param>}` reference
# for `space_id`. By the time AppConfig.from_file returns, substitution
# has already collapsed those refs to their parameter defaults — so we
# re-parse the raw YAML to map (room_key -> param_name). Rooms with
# literal or empty space_ids are skipped (nothing for provision-genie
# to inject downstream).

import re
import yaml
from dao_ai.config import AppConfig

_PARAM_REF: re.Pattern[str] = re.compile(r"^\$\{(?:var|param)\.([A-Za-z_][A-Za-z0-9_]*)\}$")

with open(config_path) as _f:
    _raw_yaml: dict = yaml.safe_load(_f) or {}

room_params: dict[str, str] = {}
for _room_key, _room_dict in (_raw_yaml.get("resources", {}).get("genie_rooms", {}) or {}).items():
    _m: re.Match[str] | None = _PARAM_REF.match(str((_room_dict or {}).get("space_id", "")).strip())
    if _m:
        room_params[_room_key] = _m.group(1)

print(f"Genie rooms with parameterized space_id: {room_params}")

config: AppConfig = AppConfig.from_file(path=config_path, initialize=False)

# COMMAND ----------

# For each parameterized Genie room: resolve the space by name via
# dao-ai's native `_resolve_space_id_by_name`, falling back to
# `room.create()` when no match exists. Both are existing dao-ai
# primitives — no duplicated lookup/create logic in this notebook.

import json
from databricks.sdk import WorkspaceClient
from dao_ai.config import value_of

if not room_params:
    print("No parameterized Genie space_ids; nothing to provision.")
    provisioned: dict[str, str] = {}
else:
    w: WorkspaceClient = WorkspaceClient()
    provisioned = {}
    for room_key, param_name in room_params.items():
        room = config.resources.genie_rooms[room_key]
        title: str = value_of(room.name)
        try:
            room.space_id = room._resolve_space_id_by_name(title)
            print(f"[{room_key}] resolved existing space '{title}' -> {room.space_id}")
        except ValueError as exc:
            if "No Genie space found" not in str(exc):
                # Multi-match: propagate so the operator disambiguates.
                raise
            # No existing space — create one via dao-ai's create() primitive.
            room.space_id = None
            room.create(w=w)
            print(f"[{room_key}] created new space '{title}' -> {value_of(room.space_id)}")
        provisioned[param_name] = value_of(room.space_id)

# COMMAND ----------

# Forward to deploy-agents via a single taskValue holding a JSON map of
# {param_name: space_id}. Skip entirely when nothing was provisioned —
# deploy-agents treats a missing taskValue as "no genie params to inject"
# and preserves the prior code path.

if provisioned:
    dbutils.jobs.taskValues.set(key="genie_space_params", value=json.dumps(provisioned))
    print(f"\nSet taskValue genie_space_params = {json.dumps(provisioned)}")
else:
    print("\nNo Genie spaces provisioned; not setting taskValues.")
