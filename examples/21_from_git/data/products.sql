USE IDENTIFIER(:database);

CREATE TABLE IF NOT EXISTS products (
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog' NOT NULL PRIMARY KEY
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code' NOT NULL
  ,brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product'
  ,product_name STRING COMMENT 'Display name of the product as shown to customers'
  ,merchandise_class STRING COMMENT 'Broad category classification of the product'
  ,list_price DOUBLE COMMENT 'Current list price in USD'
  ,description STRING COMMENT 'Detailed text description of the product including key features'
)
CLUSTER BY AUTO
COMMENT 'Product catalog seeded from a CSV colocated with the config in git'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
