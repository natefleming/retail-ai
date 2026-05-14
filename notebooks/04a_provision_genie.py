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

import re
import yaml
from dao_ai.config import AppConfig

# Param-name discovery requires the *raw* YAML text — by the time
# AppConfig.from_file finishes, every `${var.<name>}` reference has been
# substituted, so the loaded model can't tell us which YAML knob backs
# each Genie room. Parse the raw YAML to map (room_key -> param_name).
_PARAM_REF: re.Pattern[str] = re.compile(r"^\$\{(?:var|param)\.([A-Za-z_][A-Za-z0-9_]*)\}$")
with open(config_path) as _f:
    _raw: dict = yaml.safe_load(_f) or {}
_room_params: dict[str, str] = {}
for _room_key, _room_dict in (_raw.get("resources", {}).get("genie_rooms", {}) or {}).items():
    _raw_space_id: str = str((_room_dict or {}).get("space_id", "")).strip()
    _m: re.Match[str] | None = _PARAM_REF.match(_raw_space_id)
    if _m:
        _room_params[_room_key] = _m.group(1)
print(f"Rooms with parameterized space_id: {_room_params}")

# Now load the AppConfig (initialize=False to skip GenieRoomModel name
# resolution, which fails on multi-match or missing Can View).
config: AppConfig = AppConfig.from_file(path=config_path, initialize=False)

# COMMAND ----------

# For each Genie room declared in the config, if its `space_id` is a
# `${var.<name>}` reference (i.e. the operator wants provision-genie to
# fill in the id), find an existing space by title (most-recent on
# duplicate-name) or create a fresh one. Map the resolved id under the
# parameter name extracted from the reference.
#
# Skips when:
#   - no genie_rooms in the YAML
#   - space_id is a literal (already set, nothing to inject)
#   - space_id is unset (no parameter to wire)

import json
from dataclasses import dataclass
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieSpace
from dao_ai.config import GenieRoomModel, value_of


@dataclass(frozen=True)
class GenieProvisioning:
    room_key: str
    param_name: str
    space_id: str


def find_space_by_title(w: WorkspaceClient, title: str) -> GenieSpace | None:
    """Return the most-recently created Genie space matching `title`, or None."""
    matches: list[GenieSpace] = []
    try:
        for space in w.genie.list_spaces().spaces or []:
            if (space.title or "") == title:
                matches.append(space)
    except Exception as exc:
        print(f"list_spaces failed for title='{title}': {exc}")
        return None
    if not matches:
        return None
    matches.sort(key=lambda s: getattr(s, "created_time", 0) or 0, reverse=True)
    return matches[0]


def provision_room(w: WorkspaceClient, room_key: str, room: GenieRoomModel) -> str:
    """Resolve or create the Genie space for `room`. Returns the space_id."""
    title: str = value_of(room.name) if room.name else room_key

    existing: GenieSpace | None = find_space_by_title(w, title)
    if existing is not None:
        print(f"[{room_key}] reusing existing space '{title}': {existing.space_id}")
        return existing.space_id

    print(f"[{room_key}] no existing space with title='{title}'; creating new")
    from dao_ai.providers.databricks import DatabricksProvider

    provider: DatabricksProvider = DatabricksProvider(w=w)
    created: GenieSpace = provider.create_genie_space(room)
    new_id: str | None = getattr(created, "space_id", None) or value_of(room.space_id)
    if not new_id:
        raise RuntimeError(f"Failed to obtain space_id for room '{room_key}'")
    print(f"[{room_key}] created space: {new_id}")
    return new_id


w: WorkspaceClient = WorkspaceClient()
provisioned: list[GenieProvisioning] = []

if config.resources is None or not config.resources.genie_rooms:
    print("No genie_rooms declared in config; nothing to provision.")
elif not _room_params:
    print("No Genie rooms reference a `${var.X}` space_id; nothing to inject.")
else:
    for room_key, room in config.resources.genie_rooms.items():
        param_name: str | None = _room_params.get(room_key)
        if param_name is None:
            literal: str = value_of(room.space_id) if room.space_id else ""
            if literal:
                print(f"[{room_key}] space_id is literal ({literal}); skipping (no param to inject)")
            else:
                print(f"[{room_key}] space_id is unset; skipping (no param to inject)")
            continue

        space_id: str = provision_room(w, room_key, room)
        provisioned.append(GenieProvisioning(room_key=room_key, param_name=param_name, space_id=space_id))

# COMMAND ----------

# Forward results to deploy-agents via taskValues. The deploy-agents
# notebook reads `genie_space_params` (a JSON map of {param_name:
# space_id}) and binds each entry as a dao-ai config parameter at
# AppConfig.from_file load time.
#
# Skip entirely when nothing was provisioned — deploy-agents reads
# the taskValue with a default of "" which it treats as "no genie
# params to inject", preserving prior behavior.

if provisioned:
    params_map: dict[str, str] = {p.param_name: p.space_id for p in provisioned}
    dbutils.jobs.taskValues.set(key="genie_space_params", value=json.dumps(params_map))
    print(f"\nSet taskValue genie_space_params = {json.dumps(params_map)}")
else:
    print("\nNo Genie spaces provisioned; not setting taskValues.")
