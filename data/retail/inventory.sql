USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE inventory (
  inventory_id BIGINT COMMENT 'Unique identifier for each inventory record' NOT NULL PRIMARY KEY
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table identifying the specific product' NOT NULL 
  ,store STRING COMMENT 'Store identifier where inventory is located' NOT NULL
  ,store_quantity INT COMMENT 'Current available quantity of product in the specified store'
  ,warehouse STRING COMMENT 'Warehouse identifier where backup inventory is stored'
  ,warehouse_quantity INT COMMENT 'Current available quantity of product in the specified warehouse'
  ,retail_amount DECIMAL(11, 2) COMMENT 'Current retail price of the product'
  ,popularity_rating STRING COMMENT 'Rating indicating how popular/frequently purchased the product is (e.g., high, medium, low)'
  ,department STRING COMMENT 'Department within the store where the product is categorized'
  ,aisle_location STRING COMMENT 'Physical aisle location identifier where the product can be found in store'
  ,is_closeout BOOLEAN COMMENT 'Flag indicating whether the product is marked for closeout/clearance'
  ,FOREIGN KEY (product_id) REFERENCES products(product_id)
)
CLUSTER BY AUTO
COMMENT 'Inventory tracking table that maintains current product quantities across stores and warehouses, including location information and pricing details'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
