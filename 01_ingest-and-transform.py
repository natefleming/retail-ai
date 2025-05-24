# Databricks notebook source
from typing import Sequence
import tomli
from pathlib import Path

# Read dependencies from pyproject.toml
with open("pyproject.toml", "rb") as f:
    pyproject = tomli.load(f)
    
pip_requirements = pyproject["project"]["dependencies"]
pip_requirements_str = " ".join(pip_requirements)

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade {pip_requirements_str}
# MAGIC %restart_python

# COMMAND ----------

from typing import Sequence
from importlib.metadata import version

# Print installed versions for verification
for req in pip_requirements:
    pkg_name = req.split("[")[0].split(">=")[0].split("==")[0].strip('"')
    print(f"{pkg_name}=={version(pkg_name)}")

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

import mlflow
from mlflow.models import ModelConfig

config: ModelConfig = ModelConfig(development_config="model_config.yaml")

catalog_name: str = config.get("catalog_name")
database_name: str = config.get("database_name")
volume_name: str = config.get("volume_name")

print(f"catalog_name: {catalog_name}")
print(f"database_name: {database_name}")
print(f"volume_name: {volume_name}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
  CatalogInfo, 
  SchemaInfo, 
  VolumeInfo, 
  VolumeType
)
from retail_ai.catalog import (
  get_or_create_catalog, 
  get_or_create_database, 
  get_or_create_volume
)

w: WorkspaceClient = WorkspaceClient()

catalog: CatalogInfo = get_or_create_catalog(name=catalog_name, w=w)
database: SchemaInfo = get_or_create_database(catalog=catalog, name=database_name, w=w)
volume: VolumeInfo = get_or_create_volume(catalog=catalog, database=database, name=volume_name, w=w)

print(f"catalog: {catalog.full_name}")
print(f"database: {database.full_name}")
print(f"volume: {volume.full_name}")
print(f"volume path: {volume.as_path()}")

# COMMAND ----------

from typing import Any, Sequence
import re
from pathlib import Path

datasets: Sequence[dict[str, str]] = config.get("datasets")

context = dbutils.entry_point.getDbutils().notebook().getContext()
current_dir = "file:///Workspace" / Path(context.notebookPath().get()).relative_to("/").parent

for dataset in datasets:
    table: str = dataset.get("table")
    ddl_path: Path = Path(dataset.get("ddl"))
    data_path: Path = current_dir / Path(dataset.get("data"))
    format: str = dataset.get("format")

    # Execute DDL statements
    statements: Sequence[str] = [s for s in re.split(r"\s*;\s*", ddl_path.read_text()) if s]
    for statement in statements:
        print(statement)
        spark.sql(statement, args={"database": database.full_name})

    # Load data based on format
    if format == "sql":
        # For SQL data files, execute the SQL statements
        data_statements: Sequence[str] = [s for s in re.split(r"\s*;\s*", Path(dataset.get("data")).read_text()) if s]
        for statement in data_statements:
            print(statement)
            spark.sql(statement, args={"database": database.full_name})
    else:
        # For other formats (parquet, csv, etc.), use spark.read
        spark.read.format(format).load(data_path.as_posix()).write.mode("overwrite").saveAsTable(table)

# COMMAND ----------

spark.sql(f"USE {database.full_name}")
for dataset in datasets:
    table: str = dataset.get("table")
    display(spark.table(table))
