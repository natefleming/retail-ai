-- Unity Catalog Function: find_inventory_by_sku
-- Description: Retrieves detailed inventory information for products by SKU across all stores
-- This function is designed for inventory management in retail applications

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_inventory_by_sku(
  sku ARRAY<STRING> COMMENT 'One or more unique identifiers for retrieve. It may help to use another tool to provide this value. SKU values are between 5-8 alpha numeric characters'
)
RETURNS TABLE(
  inventory_id BIGINT COMMENT 'Unique identifier for each inventory record'
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table identifying the specific product'  
  ,store_id INT COMMENT 'Store identifier where inventory is located'
  ,store_quantity INT COMMENT 'Current available quantity of product in the specified store'
  ,warehouse STRING COMMENT 'Warehouse identifier where backup inventory is stored'
  ,warehouse_quantity INT COMMENT 'Current available quantity of product in the specified warehouse'
  ,retail_amount DOUBLE COMMENT 'Current retail price of the product'
  ,popularity_rating STRING COMMENT 'Rating indicating how popular/frequently purchased the product is (e.g., high, medium, low)'
  ,department STRING COMMENT 'Department within the store where the product is categorized'
  ,aisle_location STRING COMMENT 'Physical aisle location identifier where the product can be found in store'
  ,is_closeout BOOLEAN COMMENT 'Flag indicating whether the product is marked for closeout/clearance'
)
READS SQL DATA
COMMENT 'Retrieves detailed inventory information for products by SKU across all stores. This function is designed for inventory management in retail applications.'
RETURN 
SELECT 
  inventory_id
  ,sku
  ,upc
  ,inventory.product_id
  ,store_id
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
