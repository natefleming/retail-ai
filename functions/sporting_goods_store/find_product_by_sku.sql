-- Function to find product details by SKU
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_product_by_sku(
  sku ARRAY<STRING> COMMENT 'One or more SKU identifiers to retrieve. SKU format is typically 3 letters, a dash, 3 letters, a dash, and 3 digits (e.g., NKE-RUN-001)'
)
RETURNS TABLE(
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog' 
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product'
  ,product_name STRING COMMENT 'Display name of the product as shown to customers'
  ,merchandise_class STRING COMMENT 'Broad category classification of the product (e.g., Footwear, Apparel, Equipment)'
  ,class_cd STRING COMMENT 'Alphanumeric code representing the specific product subcategory'
  ,department_name STRING COMMENT 'Name of the department where product is typically located'
  ,sport_category STRING COMMENT 'Primary sport or activity the product is designed for'
  ,base_price DOUBLE COMMENT 'Standard retail price before any discounts'
  ,long_description STRING COMMENT 'Detailed text description of the product including key features and attributes'
)
READS SQL DATA
COMMENT 'Retrieves detailed information about sporting goods products by SKU. Supports product lookup, comparison, and recommendation use cases across footwear, apparel, equipment, and accessories.'
RETURN 
SELECT 
  product_id
  ,sku
  ,upc
  ,brand_name
  ,product_name
  ,merchandise_class
  ,class_cd
  ,department_name
  ,sport_category
  ,base_price
  ,long_description
FROM {catalog_name}.{schema_name}.products 
WHERE ARRAY_CONTAINS(find_product_by_sku.sku, sku);
