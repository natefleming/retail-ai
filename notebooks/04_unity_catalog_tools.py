# Databricks notebook source
# MAGIC %pip install --quiet uv
# MAGIC
# MAGIC import os
# MAGIC os.environ["UV_PROJECT_ENVIRONMENT"] = os.environ["VIRTUAL_ENV"]

# COMMAND ----------

# MAGIC %sh uv --project ../ sync

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="../config/model_config.yaml")
config_path: str = dbutils.widgets.get("config-path")
print(config_path)

# COMMAND ----------

import sys
from typing import Sequence
from importlib.metadata import version

sys.path.insert(0, "..")

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
from retail_ai.config import AppConfig, SchemaModel

model_config_file: str = config_path
model_config: ModelConfig = ModelConfig(development_config=model_config_file)
config: AppConfig = AppConfig(**model_config.to_dict())

schema: SchemaModel = config.schemas.get("retail_schema")
catalog_name: str = schema.catalog_name
schema_name: str = schema.schema_name


print(f"catalog_name: {catalog_name}")
print(f"schema_name: {schema_name}")

# COMMAND ----------


from databricks.sdk import WorkspaceClient
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from databricks_langchain import DatabricksFunctionClient, UCFunctionToolkit


w: WorkspaceClient = WorkspaceClient()
client: DatabricksFunctionClient = DatabricksFunctionClient(client=w)

# COMMAND ----------

client.create_function(
  sql_function_body=f"""
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_product_by_sku(
  sku ARRAY<STRING> COMMENT 'One or more unique identifiers for retrieve. It may help to use another tool to provide this value. SKU values are between 5-8 alpha numeric characters'
)
RETURNS TABLE(
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog' 
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product'
  ,product_name STRING COMMENT 'Display name of the product as shown to customers'
  ,merchandise_class STRING COMMENT 'Broad category classification of the product (e.g., Electronics, Apparel, Grocery)'
  ,class_cd STRING COMMENT 'Alphanumeric code representing the specific product subcategory'
  ,description STRING COMMENT 'Detailed text description of the product including key features and attributes'
)
READS SQL DATA
COMMENT 'Retrieves detailed information about a specific product by its SKU. This function is designed for product information retrieval in retail applications and can be used for product information, comparison, and recommendation.'
RETURN 
SELECT 
  product_id
  ,sku
  ,upc
  ,brand_name
  ,product_name
  ,merchandise_class
  ,class_cd
  ,description
FROM {catalog_name}.{schema_name}.products 
WHERE ARRAY_CONTAINS(find_product_by_sku.sku, sku)
"""
)

# COMMAND ----------

import pandas as pd
from io import StringIO

from unitycatalog.ai.core.base import FunctionExecutionResult


result: FunctionExecutionResult = client.execute_function(
    function_name=f"{catalog_name}.{schema_name}.find_product_by_sku",
    parameters={"sku": ["00176279"] }
)

if result.error:
  raise Exception(result.error)

pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))

display(pdf)

# COMMAND ----------

client.create_function(
  sql_function_body=f"""
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_product_by_upc(
  upc ARRAY<STRING> COMMENT 'One or more unique identifiers for retrieve. It may help to use another tool to provide this value. UPC values are between 10-16 alpha numeric characters'
)
RETURNS TABLE(
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog' 
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product'
  ,product_name STRING COMMENT 'Display name of the product as shown to customers'
  ,merchandise_class STRING COMMENT 'Broad category classification of the product (e.g., Electronics, Apparel, Grocery)'
  ,class_cd STRING COMMENT 'Alphanumeric code representing the specific product subcategory'
  ,description STRING COMMENT 'Detailed text description of the product including key features and attributes'
)

COMMENT 'Retrieves detailed information about a specific product by its SKU. This function is designed for product information retrieval in retail applications and can be used for product information, comparison, and recommendation.'
RETURN 
SELECT 
  product_id
  ,sku
  ,upc
  ,brand_name
  ,product_name
  ,merchandise_class
  ,class_cd
  ,description
FROM {catalog_name}.{schema_name}.products 
WHERE ARRAY_CONTAINS(find_product_by_upc.upc, upc);
  """
)

# COMMAND ----------

import pandas as pd
from io import StringIO

from unitycatalog.ai.core.base import FunctionExecutionResult


result: FunctionExecutionResult = client.execute_function(
    function_name=f"{catalog_name}.{schema_name}.find_product_by_upc",
    parameters={"upc": ["0017627748017"] }
)

if result.error:
  raise Exception(result.error)

pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))
display(pdf)


# COMMAND ----------

client.create_function(
  sql_function_body=f"""
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_inventory_by_sku(
  sku ARRAY<STRING> COMMENT 'One or more unique identifiers for retrieve. It may help to use another tool to provide this value. SKU values are between 5-8 alpha numeric characters'
)
RETURNS TABLE(
  inventory_id BIGINT COMMENT 'Unique identifier for each inventory record'
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table identifying the specific product'  
  ,store STRING COMMENT 'Store identifier where inventory is located'
  ,store_quantity INT COMMENT 'Current available quantity of product in the specified store'
  ,warehouse STRING COMMENT 'Warehouse identifier where backup inventory is stored'
  ,warehouse_quantity INT COMMENT 'Current available quantity of product in the specified warehouse'
  ,retail_amount DECIMAL(11, 2) COMMENT 'Current retail price of the product'
  ,popularity_rating STRING COMMENT 'Rating indicating how popular/frequently purchased the product is (e.g., high, medium, low)'
  ,department STRING COMMENT 'Department within the store where the product is categorized'
  ,aisle_location STRING COMMENT 'Physical aisle location identifier where the product can be found in store'
  ,is_closeout BOOLEAN COMMENT 'Flag indicating whether the product is marked for closeout/clearance'
)
READS SQL DATA
COMMENT 'Retrieves detailed information about a specific product by its SKU. This function is designed for product information retrieval in retail applications and can be used for product information, comparison, and recommendation.'
RETURN 
SELECT 
  inventory_id
  ,sku
  ,upc
  ,inventory.product_id
  ,store
  ,store_quantity
  ,warehouse
  ,warehouse_quantity
  ,retail_amount
  ,popularity_rating
  ,department
  ,aisle_location
  ,is_closeout
FROM {catalog_name}.{schema_name}.inventory inventory
JOIN {catalog_name}.{schema_name}.products products
ON inventory.product_id = products.product_id
WHERE ARRAY_CONTAINS(find_inventory_by_sku.sku, products.sku);
  """
)

# COMMAND ----------

import pandas as pd
from io import StringIO

from unitycatalog.ai.core.base import FunctionExecutionResult


result: FunctionExecutionResult = client.execute_function(
    function_name=f"{catalog_name}.{schema_name}.find_inventory_by_sku",
    parameters={"sku": ["00176279"] }
)

if result.error:
  raise Exception(result.error)

pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))

display(pdf)

# COMMAND ----------

client.create_function(
  sql_function_body=f"""
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_inventory_by_upc(
  upc ARRAY<STRING> COMMENT 'One or more unique identifiers for retrieve. It may help to use another tool to provide this value. UPC values are between 10-16 alpha numeric characters'
)
RETURNS TABLE(
  inventory_id BIGINT COMMENT 'Unique identifier for each inventory record'
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table identifying the specific product'  
  ,store STRING COMMENT 'Store identifier where inventory is located'
  ,store_quantity INT COMMENT 'Current available quantity of product in the specified store'
  ,warehouse STRING COMMENT 'Warehouse identifier where backup inventory is stored'
  ,warehouse_quantity INT COMMENT 'Current available quantity of product in the specified warehouse'
  ,retail_amount DECIMAL(11, 2) COMMENT 'Current retail price of the product'
  ,popularity_rating STRING COMMENT 'Rating indicating how popular/frequently purchased the product is (e.g., high, medium, low)'
  ,department STRING COMMENT 'Department within the store where the product is categorized'
  ,aisle_location STRING COMMENT 'Physical aisle location identifier where the product can be found in store'
  ,is_closeout BOOLEAN COMMENT 'Flag indicating whether the product is marked for closeout/clearance'
)
READS SQL DATA
COMMENT 'Retrieves detailed information about a specific product by its SKU. This function is designed for product information retrieval in retail applications and can be used for product information, comparison, and recommendation.'
RETURN 
SELECT 
  inventory_id
  ,sku
  ,upc
  ,inventory.product_id
  ,store
  ,store_quantity
  ,warehouse
  ,warehouse_quantity
  ,retail_amount
  ,popularity_rating
  ,department
  ,aisle_location
  ,is_closeout
FROM {catalog_name}.{schema_name}.inventory inventory
JOIN {catalog_name}.{schema_name}.products products
ON inventory.product_id = products.product_id
WHERE ARRAY_CONTAINS( find_inventory_by_upc.upc, products.upc)
  """
)

# COMMAND ----------

import pandas as pd
from io import StringIO

from unitycatalog.ai.core.base import FunctionExecutionResult


result: FunctionExecutionResult = client.execute_function(
    function_name=f"{catalog_name}.{schema_name}.find_inventory_by_upc",
    parameters={"upc": ["0017627748017"] }
)

if result.error:
  raise Exception(result.error)

pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))

display(pdf)

# COMMAND ----------

client.create_function(
  sql_function_body=f"""
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_store_inventory_by_sku(
  store STRING COMMENT 'The store identifier to retrieve inventory for'
  ,sku ARRAY<STRING> COMMENT 'One or more unique identifiers to retrieve. It may help to use another tool to provide this value. SKU values are between 5-8 alpha numeric characters'
)
RETURNS TABLE(
  inventory_id BIGINT COMMENT 'Unique identifier for each inventory record'
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table identifying the specific product'  
  ,store STRING COMMENT 'Store identifier where inventory is located'
  ,store_quantity INT COMMENT 'Current available quantity of product in the specified store'
  ,warehouse STRING COMMENT 'Warehouse identifier where backup inventory is stored'
  ,warehouse_quantity INT COMMENT 'Current available quantity of product in the specified warehouse'
  ,retail_amount DECIMAL(11, 2) COMMENT 'Current retail price of the product'
  ,popularity_rating STRING COMMENT 'Rating indicating how popular/frequently purchased the product is (e.g., high, medium, low)'
  ,department STRING COMMENT 'Department within the store where the product is categorized'
  ,aisle_location STRING COMMENT 'Physical aisle location identifier where the product can be found in store'
  ,is_closeout BOOLEAN COMMENT 'Flag indicating whether the product is marked for closeout/clearance'
)
READS SQL DATA
COMMENT 'Retrieves detailed information about a specific product by its SKU for a specific store. This function is designed for store inventory retrieval in retail applications and can be used for product information, comparison, and recommendation.'
RETURN 
SELECT 
  inventory_id
  ,sku
  ,upc
  ,inventory.product_id
  ,store
  ,store_quantity
  ,warehouse
  ,warehouse_quantity
  ,retail_amount
  ,popularity_rating
  ,department
  ,aisle_location
  ,is_closeout
FROM {catalog_name}.{schema_name}.inventory inventory
JOIN {catalog_name}.{schema_name}.products products
ON inventory.product_id = products.product_id
WHERE ARRAY_CONTAINS(find_store_inventory_by_sku.sku, products.sku) AND inventory.store = find_store_inventory_by_sku.store;
  """
)

# COMMAND ----------

import pandas as pd
from io import StringIO

from unitycatalog.ai.core.base import FunctionExecutionResult


result: FunctionExecutionResult = client.execute_function(
    function_name=f"{catalog_name}.{schema_name}.find_store_inventory_by_sku",
    parameters={"store": "35048", "sku": ["00176279"] }
)

if result.error:
  raise Exception(result.error)

pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))

display(pdf)

# COMMAND ----------

client.create_function(
  sql_function_body=f"""
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_store_inventory_by_upc(
  store STRING COMMENT 'The store identifier to retrieve inventory for'
  ,upc ARRAY<STRING> COMMENT 'One or more unique identifiers to retrieve. It may help to use another tool to provide this value. UPC values are between 10-16 alpha numeric characters'
)
RETURNS TABLE(
  inventory_id BIGINT COMMENT 'Unique identifier for each inventory record'
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table identifying the specific product'  
  ,store STRING COMMENT 'Store identifier where inventory is located'
  ,store_quantity INT COMMENT 'Current available quantity of product in the specified store'
  ,warehouse STRING COMMENT 'Warehouse identifier where backup inventory is stored'
  ,warehouse_quantity INT COMMENT 'Current available quantity of product in the specified warehouse'
  ,retail_amount DECIMAL(11, 2) COMMENT 'Current retail price of the product'
  ,popularity_rating STRING COMMENT 'Rating indicating how popular/frequently purchased the product is (e.g., high, medium, low)'
  ,department STRING COMMENT 'Department within the store where the product is categorized'
  ,aisle_location STRING COMMENT 'Physical aisle location identifier where the product can be found in store'
  ,is_closeout BOOLEAN COMMENT 'Flag indicating whether the product is marked for closeout/clearance'
)
READS SQL DATA
COMMENT 'Retrieves detailed information about a specific product by its UPC for a specific store. This function is designed for store inventory retrieval in retail applications and can be used for product information, comparison, and recommendation.'
RETURN 
SELECT 
  inventory_id
  ,sku
  ,upc
  ,inventory.product_id
  ,store
  ,store_quantity
  ,warehouse
  ,warehouse_quantity
  ,retail_amount
  ,popularity_rating
  ,department
  ,aisle_location
  ,is_closeout
FROM {catalog_name}.{schema_name}.inventory inventory
JOIN {catalog_name}.{schema_name}.products products
ON inventory.product_id = products.product_id
WHERE ARRAY_CONTAINS(find_store_inventory_by_upc.upc, products.upc) AND inventory.store = find_store_inventory_by_upc.store;
  """
)

# COMMAND ----------

import pandas as pd
from io import StringIO

from unitycatalog.ai.core.base import FunctionExecutionResult


result: FunctionExecutionResult = client.execute_function(
    function_name=f"{catalog_name}.{schema_name}.find_store_inventory_by_upc",
    parameters={"store": "35048", "upc": ["0017627748017"] }
)

if result.error:
  raise Exception(result.error)

pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))

display(pdf)
