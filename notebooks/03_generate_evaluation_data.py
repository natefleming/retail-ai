# Databricks notebook source
# MAGIC %pip install --quiet -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="../config/model_config.yaml")
config_path: str = dbutils.widgets.get("config-path")
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
    .withColumn("doc_uri", F.lit("source"))
  )
  parsed_docs_pdf: pd.DataFrame = parsed_docs_df.toPandas()

  display(parsed_docs_pdf)

  agent_description = f"""
  The agent is a RAG chatbot that answers questions about retail hardware and gives recommendations for purchases. 
  """
  question_guidelines = f"""
  # User personas
  - An employee or client asking about products and inventory


  # Example questions
  - What grills do you have in stock?
  - Can you recommend a accessories for my Toro lawn mower?

  # Additional Guidelines
  - Questions should be succinct, and human-like
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
