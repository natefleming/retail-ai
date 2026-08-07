# Databricks notebook source
# Dependency bootstrap. Install dao-ai (which pulls its own transitive deps) via
# uv — the newest bundled ../dist wheel in development mode, else the published
# PyPI package. In a deployed job the serverless environment has already
# installed it; this reinstall is harmless. ``%restart_python`` makes the freshly
# installed package importable in the cells below.
# This notebook ingests datasets, which may be EXCEL-format (pd.read_excel needs
# openpyxl). The dataset format is not known until the config loads below, so it
# installs the ``[excel]`` extra unconditionally. Notebooks that build the agent
# graph (06_deploy_agent, 08_run_evaluation) install ``[all]``. The install spec
# is single-quoted in the magic so a dev wheel's ``+local`` version tag and the
# ``[excel]`` bracket survive shell glob/bracket expansion.
import glob

_dao_ai_dep = (
    next(iter(sorted(glob.glob("../dist/dao_ai-*.whl"), reverse=True)), "dao-ai")
    + "[excel]"
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

from typing import Sequence

from dao_ai.utils import find_config_files

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

config_files: list[str] = find_config_files("../config")
dbutils.widgets.dropdown(
    name="config-paths", choices=config_files, defaultValue=next(iter(config_files), "")
)

config_path: str | None = dbutils.widgets.get("config-path") or None
project_path: str = dbutils.widgets.get("config-paths") or None

config_path: str = config_path or project_path

print(config_path)

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

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
