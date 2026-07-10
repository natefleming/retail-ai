USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE products (
  product_id BIGINT COMMENT 'Unique product identifier' NOT NULL PRIMARY KEY
  ,sku STRING COMMENT 'Stock Keeping Unit — unique product code' NOT NULL
  ,product_name STRING COMMENT 'Customer-facing product name'
  ,brand STRING COMMENT 'Brand or manufacturer'
  ,category STRING COMMENT 'Top-level merchandise category'
  ,subcategory STRING COMMENT 'Sub-category for finer-grained classification'
  ,description STRING COMMENT 'Detailed product description — Vector Search embedding source'
  ,attributes STRING COMMENT 'JSON-encoded structured attributes (size, color, dietary, etc.)'
  ,price DOUBLE COMMENT 'Base list price in USD'
  ,is_b2b_only BOOLEAN COMMENT 'True when product is only sold via B2B channels'
  ,created_at TIMESTAMP COMMENT 'Row creation timestamp'
  ,updated_at TIMESTAMP COMMENT 'Row last-updated timestamp'
)
CLUSTER BY AUTO
COMMENT 'Commerce Swarm catalog. Source table for products_vector_store (description column embedded).'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
