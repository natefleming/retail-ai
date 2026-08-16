-- Function to find product details by SKU.
--
-- The DDL lives beside the config in git; `unity_catalog_functions[*].ddl` is
-- resolved against the config's directory inside the checkout, so this file is
-- readable with no local clone of the project.
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_product_by_sku(
  sku ARRAY<STRING> COMMENT 'One or more SKUs to retrieve. SKU values are 8 alphanumeric characters, e.g. DRL10045'
)
RETURNS TABLE(
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog'
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product'
  ,product_name STRING COMMENT 'Display name of the product as shown to customers'
  ,merchandise_class STRING COMMENT 'Broad category classification of the product'
  ,list_price DOUBLE COMMENT 'Current list price in USD'
  ,description STRING COMMENT 'Detailed text description of the product including key features'
)
READS SQL DATA
COMMENT 'Retrieves detailed information about a product by its SKU. Use this when the customer has already named an exact SKU; use product search instead for descriptive queries.'
RETURN
SELECT
  product_id
  ,sku
  ,brand_name
  ,product_name
  ,merchandise_class
  ,list_price
  ,description
FROM {catalog_name}.{schema_name}.products
WHERE ARRAY_CONTAINS(find_product_by_sku.sku, sku);
