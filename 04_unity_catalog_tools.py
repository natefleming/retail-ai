# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "langgraph",
  "langchain",
  "databricks-langchain",
  "databricks-sdk",
  "mlflow",
  "python-dotenv"
)

pip_requirements: str = " ".join(pip_requirements)

%pip install --quiet --upgrade {pip_requirements}
%restart_python

# COMMAND ----------

from typing import Sequence
from importlib.metadata import version

pip_requirements: Sequence[str] = (
  f"langgraph=={version('langgraph')}",
  f"langchain=={version('langchain')}",
  f"databricks-langchain=={version('databricks-langchain')}",
  f"databricks-sdk=={version('databricks-sdk')}",
  f"mlflow=={version('mlflow')}",
  f"python-dotenv=={version('python-dotenv')}",
)

print("\n".join(pip_requirements))

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

catalog_name: str = model_config.get("catalog_name")
database_name: str = model_config.get("database_name")
index_name: str = model_config.get("retriever").get("index_name")

print(f"catalog_name: {catalog_name}")
print(f"database_name: {database_name}")
print(f"index_name: {index_name}")

# COMMAND ----------


from databricks.sdk import WorkspaceClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from databricks_langchain import DatabricksFunctionClient, UCFunctionToolkit


w: WorkspaceClient = WorkspaceClient()
client: DatabricksFunctionClient = DatabricksFunctionClient(client=w)

# COMMAND ----------

client.create_function(
  sql_function_body=f"""
    CREATE OR REPLACE FUNCTION {catalog_name}.{database_name}.find_wands_product_by_id(
      product_id INT COMMENT 'The product id'
    )
    RETURNS TABLE(
      id INT
      ,product_name STRING
      ,product_class STRING
      ,product_description STRING
      ,average_rating FLOAT
      ,rating_count INT
    )
    READS SQL DATA
    COMMENT 'This function returns the product details for a given product id'
    RETURN 
    SELECT 
      id
      ,product_name
      ,product_class
      ,product_description
      ,average_rating
      ,rating_count
    FROM {catalog_name}.{database_name}.wands 
    WHERE product_id = find_wands_product_by_id.product_id;
  """
)

# COMMAND ----------

import pandas as pd
from io import StringIO

from unitycatalog.ai.core.base import FunctionExecutionResult


result: FunctionExecutionResult = client.execute_function(
    function_name=f"{catalog_name}.{database_name}.find_wands_product_by_id",
    parameters={"product_id": 13771 }
)

if result.error:
  raise Exception(result.error)

pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))

display(pdf)

# COMMAND ----------

client.create_function(
  sql_function_body=f"""
    CREATE OR REPLACE FUNCTION {catalog_name}.{database_name}.find_wands_product_description(
      description STRING COMMENT 'The product description'
    ) 
    RETURNS TABLE(
      id INT
      ,product_name STRING
      ,product_class STRING
      ,product_description STRING
      ,average_rating FLOAT
      ,rating_count INT
    )
    READS SQL DATA
    COMMENT 'This function returns the product details for a given product id'
    RETURN 
    WITH search_products AS(
       SELECT product_id FROM VECTOR_SEARCH(
        index => {repr(index_name)}
        ,query_text => description
        ,num_results => 10
      )
    ) 
    SELECT 
      id
      ,product_name
      ,product_class
      ,product_description
      ,average_rating
      ,rating_count
    FROM {catalog_name}.{database_name}.wands 
    WHERE product_id IN (SELECT product_id FROM search_products)
    LIMIT 10
  """
)

# COMMAND ----------

import pandas as pd
from io import StringIO

from unitycatalog.ai.core.base import FunctionExecutionResult


result: FunctionExecutionResult = client.execute_function(
    function_name=f"{catalog_name}.{database_name}.find_wands_product_description",
    parameters={"description": "A white ottoman" }
)

if result.error:
  raise Exception(result.error)

pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))

display(pdf)
