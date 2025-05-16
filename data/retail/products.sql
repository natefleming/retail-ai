USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE products (
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog' NOT NULL PRIMARY KEY
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code' NOT NULL 
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification' NOT NULL
  ,brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product'
  ,product_name STRING COMMENT 'Display name of the product as shown to customers'
  ,merchandise_class STRING COMMENT 'Broad category classification of the product (e.g., Electronics, Apparel, Grocery)'
  ,class_cd STRING COMMENT 'Alphanumeric code representing the specific product subcategory'
  ,description STRING COMMENT 'Detailed text description of the product including key features and attributes'
)
CLUSTER BY AUTO
COMMENT 'Master product catalog containing core product information and identifiers'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;