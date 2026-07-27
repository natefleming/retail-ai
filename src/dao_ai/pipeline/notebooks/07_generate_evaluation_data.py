# Databricks notebook source
# Dependency bootstrap. Install dao-ai (which pulls its own transitive deps) via
# uv — the newest bundled ../dist wheel in development mode, else the published
# PyPI package. In a deployed job the serverless environment has already
# installed it; this reinstall is harmless. ``%restart_python`` makes the freshly
# installed package importable in the cells below.
import glob

_dao_ai_dep = next(
    iter(sorted(glob.glob("../dist/dao_ai-*.whl"), reverse=True)), "dao-ai"
)

# MAGIC %uv pip install --quiet {_dao_ai_dep}
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

import os
from typing import Sequence


def find_yaml_files_os_walk(base_path: str) -> Sequence[str]:
    # Tolerate a missing/non-dir base path: when the pipeline runs from a
    # wheel-only bundle an explicit `config-path` is always supplied, so the
    # `../config` discovery dropdown is optional. Return [] instead of raising.
    if not os.path.isdir(base_path):
        return []

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
    name="config-paths", choices=config_files, defaultValue=next(iter(config_files), "")
)

config_path: str | None = dbutils.widgets.get("config-path") or None
project_path: str = dbutils.widgets.get("config-paths") or None

config_path: str = config_path or project_path

print(config_path)

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------


import pandas as pd
import pyspark.sql.functions as F
from databricks.agents.evals import generate_evals_df
from pyspark.sql import Column, DataFrame

from dao_ai.config import AppConfig, EvaluationModel, VectorStoreModel

evaluation: EvaluationModel = config.evaluation

if not evaluation:
    dbutils.notebook.exit("Missing evaluation configuration")

if evaluation.replace:
    spark.sql(f"DROP TABLE IF EXISTS `{evaluation.table.full_name}`")
elif evaluation.table.exists():
    print(f"Table already exists, skipping generation: {evaluation.table.full_name}")
    dbutils.notebook.exit(f"Table already exists: {evaluation.table.full_name}")

for _, vector_store in config.resources.vector_stores.items():
    vector_store: VectorStoreModel

    doc_uri: Column = (
        F.col(vector_store.doc_uri) if vector_store.doc_uri else F.lit("source")
    )
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
        question_guidelines = """
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
        docs=parsed_docs_pdf[:500],
        num_evals=evaluation.num_evals,
        agent_description=agent_description,
        question_guidelines=question_guidelines,
    )

    evals_df: DataFrame = spark.createDataFrame(evals_pdf)

    evals_df.write.mode("append").saveAsTable(evaluation.table.full_name)

    display(spark.table(evaluation.table.full_name))
