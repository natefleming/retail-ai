CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_product(
  sku_or_id STRING COMMENT 'Product SKU (e.g. FRZ-CAKE-001) or numeric product_id as a string'
)
RETURNS TABLE(
   product_id BIGINT COMMENT 'Unique product identifier'
  ,sku STRING COMMENT 'Stock Keeping Unit'
  ,product_name STRING COMMENT 'Customer-facing product name'
  ,brand STRING COMMENT 'Brand or manufacturer'
  ,category STRING COMMENT 'Top-level merchandise category'
  ,subcategory STRING COMMENT 'Sub-category'
  ,description STRING COMMENT 'Detailed product description'
  ,price DOUBLE COMMENT 'Base list price in USD'
  ,is_b2b_only BOOLEAN COMMENT 'True when product is only sold via B2B channels'
)
READS SQL DATA
COMMENT 'Look up a single Commerce Swarm SKU by either its SKU string or numeric product_id. Returns the canonical product record including price and channel restriction.'
RETURN
SELECT
   product_id
  ,sku
  ,product_name
  ,brand
  ,category
  ,subcategory
  ,description
  ,price
  ,is_b2b_only
FROM {catalog_name}.{schema_name}.products
WHERE sku = find_product.sku_or_id
   OR CAST(product_id AS STRING) = find_product.sku_or_id
;
