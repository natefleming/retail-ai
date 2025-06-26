-- Unity Catalog Function: find_product_by_upc
-- Description: Retrieves detailed information about specific products by their UPC
-- This function is designed for product information retrieval in retail applications

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_product_by_upc(
  upc ARRAY<STRING> COMMENT 'One or more unique identifiers to retrieve. UPC values are between 10-16 alphanumeric characters'
)
RETURNS TABLE(
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog'
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product'
  ,product_name STRING COMMENT 'Display name of the product as shown to customers'
  ,short_description STRING COMMENT 'Brief product description for quick reference'
  ,long_description STRING COMMENT 'Detailed text description of the product including key features and attributes'
  ,merchandise_class STRING COMMENT 'Broad category classification of the product (e.g., Electronics, Apparel, Grocery)'
  ,class_cd STRING COMMENT 'Alphanumeric code representing the specific product subcategory'
  ,department_name STRING COMMENT 'Name of the department where product is typically located'
  ,category_name STRING COMMENT 'Name of the product category'
  ,subcategory_name STRING COMMENT 'Name of the product subcategory'
  ,base_price DOUBLE COMMENT 'Standard retail price before any discounts'
  ,msrp DOUBLE COMMENT 'MSRP (Manufacturer Suggested Retail Price)'
)
READS SQL DATA
COMMENT 'Retrieves detailed information about specific products by their UPC. This function is designed for product information retrieval in retail applications and can be used for product information, comparison, and recommendation.'
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
WHERE ARRAY_CONTAINS(find_product_by_upc.upc, upc);
