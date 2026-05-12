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

# Install dao-ai from local wheel (bundle artifact or manual build), or fall back to source path
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
  f"python-dotenv=={version('python-dotenv')}",
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
