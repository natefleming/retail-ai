# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "databricks-agents",
  "backoff",
  "python-dotenv",
  "loguru",
  "rich"
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
  f"loguru=={version('loguru')}"
)
print("\n".join(pip_requirements))

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from typing import Any, Dict, Optional, List
from pathlib import Path

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pandas as pd

from databricks.agents.evals import generate_evals_df

from retail_ai.config import AppConfig, Retriever


file_path: Path = "agent_as_config.yaml"
config: AppConfig = AppConfig(config=file_path)

evaluation_table_name: str = config.evaluation.table_name
num_evals: int = config.evaluation.num_evals


for i, (alias, retriever) in enumerate(config.resources.retrievers.items()):
  alias: str
  retriever: Retriever

  source_table_name: str = retriever.source_table_name

  print(f"evaluation_table_name: {evaluation_table_name}")
  print(f"source_table_name: {source_table_name}")
  print(f"num_evals: {num_evals}")

  parsed_docs_df: DataFrame = spark.table(source_table_name)

  if not set(["id", "doc_uri"]).issubset(parsed_docs_df.columns):
    print("'id' and 'doc_uri' columns not found in source table")
    continue

  parsed_docs_pdf: pd.DataFrame = parsed_docs_df.toPandas()

  agent_description = f"""
  The agent is a RAG chatbot that answers questions retail furniture and gives recommendations for purchases. 
  """
  question_guidelines = f"""
  # User personas
  - An employee or client asking about furniture


  # Example questions
  - Do you have any purple leather sofas in stock?
  - Can you recommend a lamp to match my walnut side tables?

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

  mode: str = "overwrite" if i == 0 else "append"
  evals_df.write.mode(mode).saveAsTable(evaluation_table_name)
