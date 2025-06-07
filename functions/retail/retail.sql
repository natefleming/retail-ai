USE IDENTIFIER(:database);

-- Function to find product details by SKU
CREATE OR REPLACE FUNCTION find_product_by_sku(
  sku ARRAY<STRING> COMMENT 'One or more unique identifiers for retrieve. It may help to use another tool to provide this value. SKU values are between 5-8 alpha numeric characters'
)
RETURNS TABLE(
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog' 
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product'
  ,product_name STRING COMMENT 'Display name of the product as shown to customers'
  ,merchandise_class STRING COMMENT 'Broad category classification of the product (e.g., Electronics, Apparel, Grocery)'
  ,class_cd STRING COMMENT 'Alphanumeric code representing the specific product subcategory'
  ,description STRING COMMENT 'Detailed text description of the product including key features and attributes'
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
  ,merchandise_class
  ,class_cd
  ,description
FROM products 
WHERE ARRAY_CONTAINS(find_product_by_sku.sku, sku);

-- Function to find product details by UPC
CREATE OR REPLACE FUNCTION find_product_by_upc(
  upc ARRAY<STRING> COMMENT 'One or more unique identifiers for retrieve. It may help to use another tool to provide this value. UPC values are between 10-16 alpha numeric characters'
)
RETURNS TABLE(
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog' 
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  ,brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product'
  ,product_name STRING COMMENT 'Display name of the product as shown to customers'
  ,merchandise_class STRING COMMENT 'Broad category classification of the product (e.g., Electronics, Apparel, Grocery)'
  ,class_cd STRING COMMENT 'Alphanumeric code representing the specific product subcategory'
  ,description STRING COMMENT 'Detailed text description of the product including key features and attributes'
)
READS SQL DATA
COMMENT 'Retrieves detailed information about a specific product by its UPC. This function is designed for product information retrieval in retail applications and can be used for product information, comparison, and recommendation.'
RETURN 
SELECT 
  product_id
  ,sku
  ,upc
  ,brand_name
  ,product_name
  ,merchandise_class
  ,class_cd
  ,description
FROM products 
WHERE ARRAY_CONTAINS(find_product_by_upc.upc, upc);

-- Function to find inventory details by SKU
CREATE OR REPLACE FUNCTION find_inventory_by_sku(
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
COMMENT 'Retrieves detailed inventory information about a specific product by its SKU. This function is designed for product inventory retrieval in retail applications and can be used for stock checking, availability, and pricing information.'
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
FROM inventory inventory
JOIN products products
ON inventory.product_id = products.product_id
WHERE ARRAY_CONTAINS(find_inventory_by_sku.sku, products.sku);

-- Function to find inventory details by UPC
CREATE OR REPLACE FUNCTION find_inventory_by_upc(
  upc ARRAY<STRING> COMMENT 'One or more unique identifiers for retrieve. It may help to use another tool to provide this value. UPC values are between 10-16 alpha numeric characters'
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
COMMENT 'Retrieves detailed inventory information about a specific product by its UPC. This function is designed for product inventory retrieval in retail applications and can be used for stock checking, availability, and pricing information.'
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
FROM inventory inventory
JOIN products products
ON inventory.product_id = products.product_id
WHERE ARRAY_CONTAINS(find_inventory_by_upc.upc, products.upc);

-- Function to find store-specific inventory by SKU
CREATE OR REPLACE FUNCTION find_store_inventory_by_sku(
  store STRING COMMENT 'The store identifier to retrieve inventory for'
  ,sku ARRAY<STRING> COMMENT 'One or more unique identifiers to retrieve. It may help to use another tool to provide this value. SKU values are between 5-8 alpha numeric characters'
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
COMMENT 'Retrieves detailed inventory information about a specific product by its SKU for a specific store. This function is designed for store-specific inventory retrieval in retail applications and can be used for product information, comparison, and recommendation.'
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
FROM inventory inventory
JOIN products products
ON inventory.product_id = products.product_id
WHERE ARRAY_CONTAINS(find_store_inventory_by_sku.sku, products.sku) AND inventory.store = find_store_inventory_by_sku.store;

-- Function to find store-specific inventory by UPC
CREATE OR REPLACE FUNCTION find_store_inventory_by_upc(
  store STRING COMMENT 'The store identifier to retrieve inventory for'
  ,upc ARRAY<STRING> COMMENT 'One or more unique identifiers to retrieve. It may help to use another tool to provide this value. UPC values are between 10-16 alpha numeric characters'
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
COMMENT 'Retrieves detailed inventory information about a specific product by its UPC for a specific store. This function is designed for store-specific inventory retrieval in retail applications and can be used for product information, comparison, and recommendation.'
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
FROM inventory inventory
JOIN products products
ON inventory.product_id = products.product_id
WHERE ARRAY_CONTAINS(find_store_inventory_by_upc.upc, products.upc) AND inventory.store = find_store_inventory_by_upc.store;


