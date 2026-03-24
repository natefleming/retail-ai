-- Function to find inventory details by UPC across all stores
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_inventory_by_upc(
  upc ARRAY<STRING> COMMENT 'One or more UPC identifiers to retrieve inventory for. UPC values are 12 numeric characters'
)
RETURNS TABLE(
  inventory_id BIGINT COMMENT 'Unique identifier for each inventory record'
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table identifying the specific product'
  ,product_name STRING COMMENT 'Display name of the product'
  ,store STRING COMMENT 'Store identifier where inventory is located'
  ,store_name STRING COMMENT 'Name of the store'
  ,store_quantity INT COMMENT 'Current available quantity of product in the specified store'
  ,warehouse STRING COMMENT 'Warehouse identifier where backup inventory is stored'
  ,warehouse_quantity INT COMMENT 'Current available quantity of product in the specified warehouse'
  ,retail_amount DOUBLE COMMENT 'Current retail price of the product'
  ,popularity_rating STRING COMMENT 'Rating indicating how popular the product is (high, medium, low)'
  ,department STRING COMMENT 'Department within the store where the product is categorized'
  ,aisle_location STRING COMMENT 'Physical aisle location identifier where the product can be found in store'
  ,is_closeout BOOLEAN COMMENT 'Flag indicating whether the product is marked for closeout or clearance'
  ,is_out_of_stock BOOLEAN COMMENT 'Flag indicating if product is out of stock'
  ,stockout_risk_level STRING COMMENT 'Risk level of stockout (low, medium, high, critical)'
  ,trend_direction STRING COMMENT 'Current sales trend (increasing, stable, decreasing)'
)
READS SQL DATA
COMMENT 'Retrieves detailed inventory information for sporting goods products by UPC across all stores. Includes stock levels, pricing, location, and demand trend data.'
RETURN 
SELECT 
  inventory_id
  ,products.sku
  ,products.upc
  ,inventory.product_id
  ,products.product_name
  ,store
  ,store_name
  ,store_quantity
  ,warehouse
  ,warehouse_quantity
  ,retail_amount
  ,popularity_rating
  ,department
  ,aisle_location
  ,is_closeout
  ,is_out_of_stock
  ,stockout_risk_level
  ,trend_direction
FROM {catalog_name}.{schema_name}.inventory inventory
JOIN {catalog_name}.{schema_name}.products products
ON inventory.product_id = products.product_id
WHERE ARRAY_CONTAINS(find_inventory_by_upc.upc, products.upc);
