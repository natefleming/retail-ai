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
    _dao_ai_dep = (_wheels[0] if _wheels else "dao-ai") + "[excel]"

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

from typing import Sequence

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

# There is no `../config` discovery fallback. That directory exists only in the
# staged bundle layout, and the bundle stages exactly one config — the same one
# the job passes here — so discovery could only ever guess, and guessing is how
# the wrong config gets loaded. `config-path` is the single input.
widget_path: str | None = dbutils.widgets.get("config-path") or None
if not widget_path:
    raise ValueError(
        "No config to load: the `config-path` widget is empty. In a staged "
        "pipeline bundle the config sits beside this notebook under `../config/` "
        "and the job always passes it; running this notebook by hand, set "
        "`config-path` to an absolute workspace path, for example "
        "`/Workspace/Users/you@example.com/dao-ai/examples/04_genie/genie_basic.yaml`."
    )

config_path: str = widget_path

print(config_path)

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

# Serverless v5 FIPS: select psycopg's pure-Python impl before importing
# dao_ai.config, which transitively imports psycopg via databricks-langchain;
# the binary wheel's vendored OpenSSL aborts (SIGABRT) on import. Job-scoped.
import os

os.environ["PSYCOPG_IMPL"] = "python"

# COMMAND ----------

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

from databricks.sdk import WorkspaceClient

from dao_ai.config import SchemaModel, VolumeModel

w: WorkspaceClient = WorkspaceClient()

for _, schema in config.schemas.items():
    schema: SchemaModel
    _ = schema.create(w=w)

    print(f"schema: {schema.full_name}")

for _, volume in config.resources.volumes.items():
    volume: VolumeModel

    _ = volume.create(w=w)
    print(f"volume: {volume.full_name}")

# COMMAND ----------

from dao_ai.config import DatasetModel

datasets: Sequence[DatasetModel] = config.datasets

for dataset in datasets:
    dataset: DatasetModel
    dataset.create()
