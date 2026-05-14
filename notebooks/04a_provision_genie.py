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
    for root, _, files in os.walk(base_path):
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

import sys, os, glob, subprocess
from typing import Sequence
from importlib.metadata import version

_wheels = sorted(
    glob.glob("../dist/dao_ai-*.whl") or glob.glob("../../artifacts/.internal/dao_ai-*.whl"),
    key=os.path.getmtime,
    reverse=True,
)
if _wheels:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall", _wheels[0]])
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

from dao_ai.config import AppConfig

# Skip Pydantic initialize() — that resolves space_ids by name and errors on
# multi-match. We do our own resolve-or-create logic below.
config: AppConfig = AppConfig.from_file(path=config_path, initialize=False)

# COMMAND ----------

# Provision each declared Genie room: find an existing space by name (picking
# the most recently created when multiple exist) or create a fresh one. The
# resolved space_id is forwarded to the deploy-agents task via taskValues so
# the agent config sees a populated space_id at load time and skips its own
# name-resolution (which the cluster identity may not have permissions for).

from dataclasses import dataclass
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from dao_ai.config import value_of

w = WorkspaceClient()
provisioned: dict[str, str] = {}

if config.resources is None or not config.resources.genie_rooms:
    print("No genie_rooms declared; skipping provision-genie.")
else:
    for room_key, room in config.resources.genie_rooms.items():
        title: str = value_of(room.name) if room.name else room_key
        parent_path: str = value_of(room.parent_path) if room.parent_path else None
        warehouse_id: str = value_of(room.warehouse.warehouse_id) if room.warehouse else None
        explicit_space_id: str = value_of(room.space_id) if room.space_id else ""

        if explicit_space_id:
            print(f"[{room_key}] space_id already set explicitly: {explicit_space_id}")
            provisioned[room_key] = explicit_space_id
            continue

        # Look for existing spaces by title
        matches = []
        try:
            for space in w.genie.list_spaces().spaces or []:
                if (space.title or "") == title:
                    matches.append(space)
        except Exception as e:
            print(f"[{room_key}] list_spaces failed: {e}")

        chosen_space_id: str | None = None
        if matches:
            # Pick most recently created when multiple
            matches.sort(key=lambda s: getattr(s, "created_time", 0) or 0, reverse=True)
            chosen_space_id = matches[0].space_id
            print(f"[{room_key}] found {len(matches)} existing space(s); using most recent: {chosen_space_id}")
        else:
            print(f"[{room_key}] no existing space with title='{title}'; creating new")
            # Use dao-ai's create path (handles serialization + warehouse)
            from dao_ai.providers.databricks import DatabricksProvider
            provider = DatabricksProvider(w=w)
            created = provider.create_genie_space(room)
            chosen_space_id = getattr(created, "space_id", None) or value_of(room.space_id)
            print(f"[{room_key}] created space: {chosen_space_id}")

        if not chosen_space_id:
            raise RuntimeError(f"Failed to resolve or create Genie space for '{room_key}'")
        provisioned[room_key] = chosen_space_id

print("\nProvisioned Genie space_ids:")
for k, v in provisioned.items():
    print(f"  {k} = {v}")

# COMMAND ----------

# Forward to downstream tasks via taskValues. For the single-room case
# (typical), set a flat "genie_space_id" key that deploy-agents can pass
# straight through to AppConfig.from_file(params=...).
#
# For multi-room configs, also set a JSON-keyed map under "genie_space_ids".

import json

if provisioned:
    first_space_id = next(iter(provisioned.values()))
    dbutils.jobs.taskValues.set(key="genie_space_id", value=first_space_id)
    dbutils.jobs.taskValues.set(key="genie_space_ids", value=json.dumps(provisioned))
    print(f"Set taskValue genie_space_id = {first_space_id}")
    print(f"Set taskValue genie_space_ids = {json.dumps(provisioned)}")
else:
    # No-op: nothing for downstream tasks to forward
    dbutils.jobs.taskValues.set(key="genie_space_id", value="")
    dbutils.jobs.taskValues.set(key="genie_space_ids", value="{}")
    print("No Genie rooms; set empty taskValues for downstream compatibility.")
