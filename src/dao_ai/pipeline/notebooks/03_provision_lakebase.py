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

from dao_ai.config import DatabaseModel

databases: dict[str, DatabaseModel] = config.resources.databases

for _, database in databases.items():
    database: DatabaseModel

    print(f"database: {database}")
    database.create()
