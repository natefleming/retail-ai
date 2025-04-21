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
logger.info("\n".join(pip_requirements))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from typing import Any, Dict, Optional, List

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
model_config: ModelConfig = ModelConfig(development_config=model_config_file)

evaluation_config: Dict[str, Any] = model_config.get("evaluation")
datasets_config: Dict[str, Any] = model_config.get("datasets")
huggingface_config: Dict[str, Any] = datasets_config.get("huggingface")

evaluation_table_name: str = evaluation_config.get("table_name")
source_table_name: str = huggingface_config.get("table_name")

print(f"evaluation_table_name: {evaluation_table_name}")
print(f"source_table_name: {source_table_name}")


# COMMAND ----------

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pandas as pd

parsed_docs_df: DataFrame = spark.table(source_table_name)
parsed_docs_pdf: pd.DataFrame = parsed_docs_df.toPandas()

display(parsed_docs_pdf)

# COMMAND ----------

from pyspark.sql import DataFrame

# Use the synthetic eval generation API to get some evals
from databricks.agents.evals import generate_evals_df

# "Ghost text" for agent description and question guidelines - feel free to modify as you see fit.
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

num_evals = 25
evals_pdf: pd.DataFrame = generate_evals_df(
    docs=parsed_docs_pdf[
        :500
    ],  # Pass your docs. They should be in a Pandas or Spark DataFrame with columns `content STRING` and `doc_uri STRING`.
    num_evals=num_evals,  # How many synthetic evaluations to generate
    agent_description=agent_description,
    question_guidelines=question_guidelines,
)

evals_df: DataFrame = spark.createDataFrame(evals_pdf)

evals_df.write.mode("overwrite").saveAsTable(evaluation_table_name)

