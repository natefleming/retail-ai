# Databricks notebook source
from typing import Sequence

pip_requirements: Sequence[str] = (
  "langgraph",
  "langchain",
  "databricks-langchain",
  "unitycatalog-langchain[databricks]",
  "databricks-sdk",
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
  f"langgraph=={version('langgraph')}",
  f"langchain=={version('langchain')}",
  f"databricks-langchain=={version('databricks-langchain')}",
  f"databricks-sdk=={version('databricks-sdk')}",
  f"mlflow=={version('mlflow')}",
  f"python-dotenv=={version('python-dotenv')}",
  f"unitycatalog-langchain[databricks]=={version('unitycatalog-langchain')}",
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
CREATE OR REPLACE FUNCTION {catalog_name}.{database_name}.find_product_by_sku(
  sku STRING COMMENT 'Unique identifier for retrieve. It may help to use another tool to provide this value'
)
RETURNS TABLE(
  id INT COMMENT 'Unique identifier for the wand product',
  product_name STRING COMMENT 'Name of the magic wand product',
  product_class STRING COMMENT 'Classification category of the wand (e.g., Beginner, Expert, Professional)',
  product_description STRING COMMENT 'Detailed description of the wand including materials, properties, and special features',
  average_rating FLOAT COMMENT 'Average customer rating of the product on a scale of 1.0 to 5.0',
  rating_count INT COMMENT 'Total number of customer ratings submitted for this product'
)
READS SQL DATA
COMMENT 'Retrieves detailed information about a specific magic wand product by its ID. This function is designed for product information retrieval in retail applications and can be used for product display pages, comparison tools, and recommendation systems.'
RETURN 
SELECT 
  id,
  product_name,
  product_class,
  product_description,
  average_rating,
  rating_count
FROM {catalog_name}.{database_name}.wands 
WHERE sku = find_product_by_sku.sku;
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


column_name: str = "product_class"
table_name: str = f"{catalog_name}.{database_name}.wands_product"

client.create_function(
  sql_function_body=f"""
  CREATE OR REPLACE FUNCTION  {catalog_name}.{database_name}.find_allowable_product_classifications()
  RETURNS TABLE(
    classification STRING
  )
  READS SQL DATA
  COMMENT 'Returns a unique list of allowable product classifications'
  RETURN
    SELECT DISTINCT {column_name}
    FROM {table_name}
  """
)




# COMMAND ----------

import pandas as pd
from io import StringIO

from unitycatalog.ai.core.base import FunctionExecutionResult


result: FunctionExecutionResult = client.execute_function(
    function_name=f"{catalog_name}.{database_name}.find_allowable_product_classifications",
    parameters={}
)

if result.error:
  raise Exception(result.error)

pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))

display(pdf)

# COMMAND ----------

pdf['classification'].tolist()
