-- Unity Catalog Function: find_product_by_sku
-- Description: Retrieves detailed information about a specific product by its SKU
-- This function is designed for product information retrieval in retail applications

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_product_by_sku(
  sku ARRAY<STRING> COMMENT 'One or more unique identifiers for retrieve. It may help to use another tool to provide this value. SKU values are between 8-12 alpha numeric characters'
)
RETURNS TABLE(
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog' 
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product'
  ,product_name STRING COMMENT 'Display name of the product as shown to customers'
  ,short_description STRING COMMENT 'Brief description of the product'
  ,long_description STRING COMMENT 'Detailed text description of the product including key features and attributes'
  ,merchandise_class STRING COMMENT 'Broad category classification of the product (e.g., Beverages)'
  ,class_cd STRING COMMENT 'Alphanumeric code representing the specific product subcategory'
  ,department_name STRING COMMENT 'Name of the department the product belongs to'
  ,category_name STRING COMMENT 'Name of the category the product belongs to'
  ,subcategory_name STRING COMMENT 'Name of the subcategory the product belongs to'
  ,base_price DOUBLE COMMENT 'Base price of the product'
  ,msrp DOUBLE COMMENT 'MSRP (Manufacturer Suggested Retail Price)'
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
  ,short_description
  ,long_description
  ,merchandise_class
  ,class_cd
  ,department_name
  ,category_name
  ,subcategory_name
  ,base_price
  ,msrp
FROM {catalog_name}.{schema_name}.products 
WHERE ARRAY_CONTAINS(find_product_by_sku.sku, sku)
