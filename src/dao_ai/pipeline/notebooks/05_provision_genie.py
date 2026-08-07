# Databricks notebook source
# Dependency bootstrap. Install dao-ai (which pulls its own transitive deps) via
# uv — the newest bundled ../dist wheel in development mode, else the published
# PyPI package. In a deployed job the serverless environment has already
# installed it; this reinstall is harmless. ``%restart_python`` makes the freshly
# installed package importable in the cells below.
# No extras suffix: this provisioning notebook only calls core APIs. Notebooks
# that build the agent graph (07_deploy_agent, 09_run_evaluation) install
# ``[all]``; 01_ingest_and_transform installs ``[excel]``. The install spec is
# single-quoted in the magic so a dev wheel's ``+local`` version tag and any
# ``[extras]`` survive shell glob/bracket expansion.
import glob

_dao_ai_dep = next(
    iter(sorted(glob.glob("../dist/dao_ai-*.whl"), reverse=True)), "dao-ai"
)

# MAGIC %uv pip install --quiet '{_dao_ai_dep}'
# MAGIC %restart_python

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

from dao_ai.utils import find_config_files

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

config_files: list[str] = find_config_files("../config")
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
        # NOTE: when an existing space is reused, we deliberately skip
        # room.create() to avoid etag conflicts on update_space. If a
        # room's configuration changes (e.g., new warehouse, new table
        # sources), delete the existing space first or rename the room
        # title so provision-genie creates a fresh one.
        existing: GenieRoomModel | None = GenieRoomModel.from_space_id(
            value_of(room.space_id), w=room.workspace_client
        ) or GenieRoomModel.from_name(value_of(room.name), w=room.workspace_client)
        if existing is not None:
            room.space_id = existing.space_id
            print(f"[{room_key}] reusing existing space {room.space_id}")
        else:
            room.space_id = None
            room.create()
            print(f"[{room_key}] created new space {value_of(room.space_id)}")

        resolved: str = value_of(room.space_id)
        provisioned[param] = resolved
        dbutils.jobs.taskValues.set(key=param, value=resolved)
        print(f"  Set taskValue {param} = {resolved}")

# COMMAND ----------

# Emit a short notebook_output summary for observability. The actual
# downstream contract is the per-parameter taskValues set above
# (key=<parameter_name>, value=<resolved space_id>); deploy-agents
# loads via AppConfig.from_file(task_values=dbutils.jobs.taskValues,
# task_key="provision-genie"), which probes each declared parameter
# against this task's taskValues and folds non-empty results into
# substitution.
summary: dict = {"provisioned": provisioned}
dbutils.notebook.exit(json.dumps(summary))
