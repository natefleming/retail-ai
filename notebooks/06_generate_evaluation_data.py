# Databricks notebook source
# MAGIC %pip install --quiet --upgrade -r ../requirements.txt
# MAGIC %pip uninstall --quiet -y databricks-connect pyspark pyspark-connect
# MAGIC %pip install --quiet databricks-connect
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

import sys
from typing import Sequence
from importlib.metadata import version

sys.path.insert(0, "../src")

pip_requirements: Sequence[str] = (
  f"databricks-agents=={version('databricks-agents')}",
  f"mlflow=={version('mlflow')}",
)
print("\n".join(pip_requirements))

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

from typing import Any, Dict, Optional, List

from mlflow.models import ModelConfig
from dao_ai.config import AppConfig, VectorStoreModel, EvaluationModel
from pyspark.sql import DataFrame, Column
import pyspark.sql.functions as F
import pandas as pd
from pyspark.sql import DataFrame
from databricks.agents.evals import generate_evals_df



evaluation: EvaluationModel = config.evaluation

if not evaluation:
  dbutils.notebook.exit("Missing evaluation configuration")

spark.sql(f"DROP TABLE IF EXISTS {evaluation.table.full_name}")

for _, vector_store in config.resources.vector_stores.items():
  vector_store: VectorStoreModel    

  doc_uri: Column = F.col(vector_store.doc_uri) if vector_store.doc_uri else F.lit("source")
  parsed_docs_df: DataFrame = (
    spark.table(vector_store.source_table.full_name)
    .withColumn("id", F.col(vector_store.primary_key))
    .withColumn("content", F.col(vector_store.embedding_source_column))
    .withColumn("doc_uri", doc_uri)
  )
  parsed_docs_pdf: pd.DataFrame = parsed_docs_df.toPandas()

  display(parsed_docs_pdf)

  agent_description: str = evaluation.agent_description
  if not agent_description:
      agent_description = """
  A general-purpose chatbot AI agent is designed to engage in natural conversations 
  across diverse topics and tasks, drawing from broad knowledge to answer questions, 
  assist with writing, solve problems, and provide explanations while maintaining 
  context throughout interactions. It aims to be a versatile, adaptable assistant 
  that can help with the wide spectrum of things people encounter in daily life, 
  adjusting its communication style and level of detail based on user needs.
      """

  question_guidelines: str = evaluation.question_guidelines
  if not question_guidelines:
      question_guidelines = f"""
# User personas
- A curious individual seeking information or explanations
- A student looking for homework help or learning assistance  
- A professional needing quick research or writing support
- A creative person brainstorming ideas or seeking inspiration

# Example questions
- Can you explain how photosynthesis works?
- Help me write a professional email to my boss
- What are some good books similar to Harry Potter?
- How do I fix a leaky faucet?

# Additional Guidelines  
- Questions should be conversational and natural
- Users may ask follow-up questions to dig deeper into topics
- Requests can range from simple facts to complex multi-step tasks
- Tone can vary from casual chat to formal assistance
  """

  evals_pdf: pd.DataFrame = generate_evals_df(
      docs=parsed_docs_pdf[
          :500
      ],  
      num_evals=evaluation.num_evals, 
      agent_description=agent_description,
      question_guidelines=question_guidelines,
  )

  evals_df: DataFrame = spark.createDataFrame(evals_pdf)

  evals_df.write.mode("append").saveAsTable(evaluation.table.full_name)

  display(spark.table(evaluation.table.full_name))
