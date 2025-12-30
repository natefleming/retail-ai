# Databricks notebook source
# MAGIC %pip install --quiet --upgrade -r ../requirements.txt
# MAGIC %pip uninstall --quiet -y databricks-connect pyspark pyspark-connect
# MAGIC %pip install --quiet --upgrade databricks-connect
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
project_path: str = dbutils.widgets.get("config-paths") or None

config_path: str = config_path or project_path

print(config_path)

# COMMAND ----------

import sys
from typing import Sequence
from importlib.metadata import version

sys.path.insert(0, "../src")

pip_requirements: Sequence[str] = (
    f"databricks-agents=={version('databricks-agents')}",
    f"mlflow=={version('mlflow')}",
    f"databricks-connect=={version('databricks-connect')}",
)
print("\n".join(pip_requirements))

# COMMAND ----------

import nest_asyncio

nest_asyncio.apply()

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from loguru import logger

from dao_ai.logging import configure_logging

configure_logging(level="DEBUG")

# COMMAND ----------

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize Prompts
# MAGIC
# MAGIC This notebook optimizes prompts using MLflow's prompt optimization capabilities.
# MAGIC It iterates through all prompt optimizations defined in the configuration and
# MAGIC runs optimization for each one, registering new versions in MLflow.

# COMMAND ----------

from dao_ai.config import PromptOptimizationModel, PromptModel, OptimizationsModel


# Get optimizations configuration from config
optimizations: OptimizationsModel = config.optimizations

if not optimizations or not optimizations.prompt_optimizations:
    dbutils.notebook.exit("No prompt optimizations configured")
    
prompt_optimizations: dict[str, PromptOptimizationModel] = optimizations.prompt_optimizations
print(f"Found {len(prompt_optimizations)} prompt optimization(s) to process:")
for name in prompt_optimizations.keys():
    print(f"  - {name}")

# COMMAND ----------

# First, ensure all training datasets are created/updated in MLflow
print("\n" + "="*80)
print("Creating/updating training datasets")
print("="*80)

if optimizations.training_datasets:
    for dataset_name, dataset_model in optimizations.training_datasets.items():
        print(f"Processing dataset: {dataset_name}")
        try:
            dataset_model.as_dataset()
            print(f"  Dataset '{dataset_name}' ready")
        except Exception as e:
            logger.error(f"Failed to create/update dataset {dataset_name}: {e}")
            print(f"  Failed to create dataset: {str(e)}")
else:
    print("No training datasets defined in configuration (will use existing MLflow datasets)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Optimizations
# MAGIC
# MAGIC For each optimization:
# MAGIC 1. Create/update training datasets in MLflow
# MAGIC 2. Run prompt optimization using the specified agent and parameters
# MAGIC 3. Register the optimized prompt as a new version in MLflow
# MAGIC 4. Display optimization results

# COMMAND ----------

import mlflow
from typing import Sequence

# Set MLflow registry
mlflow.set_registry_uri("databricks-uc")

# Track results
optimization_results: Sequence[tuple[str, PromptModel, str]] = []

# Run optimizations
for opt_name, optimization in prompt_optimizations.items():
    print(f"\n{'='*80}")
    print(f"Optimizing: {opt_name}")
    print(f"{'='*80}")
    
    try:
        # Run optimization
        optimized_prompt: PromptModel = optimization.optimize()
        
        # Track result
        optimization_results.append(
            (
                opt_name,
                optimized_prompt,
                "Success"
            )
        )
        
        print(f"\nSuccessfully optimized prompt: {optimization.prompt.name}")
        print(f"   New version URI: {optimized_prompt.uri}")
        
    except Exception as e:
        logger.error(f"Failed to optimize {opt_name}: {e}")
        optimization_results.append(
            (
                opt_name,
                None,
                f"Failed: {str(e)}"
            )
        )
        print(f"\nFailed to optimize: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary

# COMMAND ----------

import pandas as pd

# Create summary DataFrame
summary_data = [
    {
        "Optimization Name": opt_name,
        "Original Prompt": prompt_optimizations[opt_name].prompt.name,
        "Optimized Version": opt_prompt.version if opt_prompt else "N/A",
        "Optimized URI": opt_prompt.uri if opt_prompt else "N/A",
        "Status": status,
    }
    for opt_name, opt_prompt, status in optimization_results
]

summary_df = pd.DataFrame(summary_data)
display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Next Steps
# MAGIC
# MAGIC After optimizing prompts:
# MAGIC 1. Review the optimized prompt templates in MLflow
# MAGIC 2. Evaluate the optimized prompts using `07_run_evaluation.py`
# MAGIC 3. Update your configuration to use the optimized prompt versions
# MAGIC 4. Deploy the updated agent with optimized prompts

# COMMAND ----------

print("\n" + "="*80)
print("Prompt Optimization Complete")
print("="*80)
print(f"Total optimizations processed: {len(optimization_results)}")
print(f"Successful: {sum(1 for _, _, status in optimization_results if 'Success' in status)}")
print(f"Failed: {sum(1 for _, _, status in optimization_results if 'Failed' in status)}")
