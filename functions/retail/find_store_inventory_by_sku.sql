-- Function to find store-specific inventory details by SKU
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_store_inventory_by_sku(
  store STRING COMMENT 'Store identifier to filter inventory by specific store location',
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
COMMENT 'Retrieves detailed inventory information about a specific product by its SKU for a specific store. This function is designed for store-specific product inventory retrieval in retail applications and can be used for store-level stock checking, availability, and pricing information.'
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
WHERE ARRAY_CONTAINS(find_store_inventory_by_sku.sku, products.sku)
AND inventory.store = find_store_inventory_by_sku.store;
