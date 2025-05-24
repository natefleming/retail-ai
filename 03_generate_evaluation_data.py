# Databricks notebook source

# COMMAND ----------

from typing import Sequence

from importlib.metadata import version

# COMMAND ----------

from typing import Any, Dict, Optional, List

from mlflow.models import ModelConfig


model_config_file: str = "model_config.yaml"
config: ModelConfig = ModelConfig(development_config=model_config_file)

retreiver_config: dict[str, Any] = config.get("retriever")
source_table_name: str = retreiver_config.get("source_table_name")
primary_key: str = retreiver_config.get("primary_key")
embedding_source_column: str = retreiver_config.get("embedding_source_column")
doc_uri: str = retreiver_config.get("doc_uri")

evaluation_config: Dict[str, Any] = config.get("evaluation")

evaluation_table_name: str = evaluation_config.get("table_name")
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
from decimal import Decimal

doc_uri: Column = F.col(doc_uri) if doc_uri else F.lit("source")
parsed_docs_df: DataFrame = (
  spark.table(source_table_name)
  .withColumn("id", F.col(primary_key))
  .withColumn("content", F.col(embedding_source_column))
  .withColumn("doc_uri", F.lit("source"))
  .withColumn("base_price", F.col("base_price").cast("double"))
  .withColumn("msrp", F.col("msrp").cast("double"))
  .withColumn("weight", F.col("weight").cast("double"))
  .withColumn("attributes", F.to_json(F.col("attributes")))
)

eval_columns = ["id", "content", "doc_uri", "product_name", "short_description", "long_description"]
parsed_docs_df = parsed_docs_df.select(eval_columns)

parsed_docs_pdf: pd.DataFrame = parsed_docs_df.toPandas()

display(parsed_docs_pdf)

# COMMAND ----------

from pyspark.sql import DataFrame

# Use the synthetic eval generation API to get some evals
from databricks.agents.evals import generate_evals_df

# "Ghost text" for agent description and question guidelines - feel free to modify as you see fit.
agent_description = f"""
The agent is a retail AI assistant that helps customers and employees with product information, inventory checks, recommendations, and general shopping assistance across all store departments. The agent has access to the store's complete product catalog, real-time inventory data, and can provide personalized recommendations based on customer preferences and shopping history.
"""
question_guidelines = f"""
# User personas
- In-store customers looking for product information and recommendations
- Online shoppers checking availability and comparing products
- Store associates helping customers on the sales floor
- Customer service representatives handling product inquiries

# Example questions
- Do you have the new Apple AirPods Pro in stock?
- What size sheets fit a California King mattress?
- Can you recommend some healthy snacks for kids' lunches?
- Which stores near me have the Nintendo Switch OLED in stock?
- What's the difference between regular and organic baby formula?
- Is this coffee maker compatible with K-cups?
- Do you have any deals on school supplies this week?
- What aisle can I find cleaning supplies in?

# Additional Guidelines
- Questions should reflect everyday shopping needs
- Include queries about multiple departments (Electronics, Home, Grocery, Fashion, etc.)
- Consider seasonal shopping patterns (Back-to-School, Holiday, Summer)
- Include both specific product queries and general recommendation requests
- Questions should be conversational and natural
- Include availability, price, and comparison questions
- Consider both in-store and online shopping scenarios
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
