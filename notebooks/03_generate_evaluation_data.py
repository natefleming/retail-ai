# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "databricks-agents",
  "backoff",
  "mlflow",
  "python-dotenv",
)

pip_requirements: str = " ".join(pip_requirements)

%pip install --quiet --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from typing import Sequence

from importlib.metadata import version


pip_requirements: Sequence[str] = (
  f"databricks-agents=={version('databricks-agents')}",
  f"backoff=={version('backoff')}",
  f"mlflow=={version('mlflow')}",
)
print("\n".join(pip_requirements))

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from typing import Any, Dict, Optional, List

from mlflow.models import ModelConfig
from retail_ai.catalog import full_name

model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)


vector_store_config: dict[str, Any] = config.get("resources").get("vector_stores").get("product_vector_store")

source_table_name: str = full_name(vector_store_config.get("source_table"))
primary_key: str = vector_store_config.get("primary_key")
embedding_source_column: str = vector_store_config.get("embedding_source_column")
doc_uri: str = vector_store_config.get("doc_uri")

evaluation_config: Dict[str, Any] = config.get("evaluation")

evaluation_table_name: str = full_name(evaluation_config.get("table"))
num_evals: int = evaluation_config.get("num_evals")

print(f"evaluation_table_name: {evaluation_table_name}")
print(f"source_table_name: {source_table_name}")
print(f"primary_key: {primary_key}")
print(f"embedding_source_column: {embedding_source_column}")
print(f"doc_uri: {doc_uri}")
print(f"num_evals: {num_evals}")

# COMMAND ----------

from pyspark.sql import DataFrame, Column
import pyspark.sql.functions as F
import pandas as pd

doc_uri: Column = F.col(doc_uri) if doc_uri else F.lit("source")
parsed_docs_df: DataFrame = (
  spark.table(source_table_name)
  .withColumn("id", F.col(primary_key))
  .withColumn("content", F.col(embedding_source_column))
  .withColumn("doc_uri", F.lit("source"))
)
parsed_docs_pdf: pd.DataFrame = parsed_docs_df.toPandas()

display(parsed_docs_pdf)

# COMMAND ----------

from pyspark.sql import DataFrame

# Use the synthetic eval generation API to get some evals
from databricks.agents.evals import generate_evals_df

# "Ghost text" for agent description and question guidelines - feel free to modify as you see fit.
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
    num_evals=num_evals, 
    agent_description=agent_description,
    question_guidelines=question_guidelines,
)

evals_df: DataFrame = spark.createDataFrame(evals_pdf)

evals_df.write.mode("overwrite").saveAsTable(evaluation_table_name)


# COMMAND ----------

display(spark.table(evaluation_table_name))
