USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE inventory (
  inventory_id BIGINT COMMENT 'Unique identifier for each inventory record' NOT NULL PRIMARY KEY
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table identifying the specific product' NOT NULL 
  
  -- Store location information (enhanced from store)
  ,store_id BIGINT COMMENT 'Unique identifier for the store'
  ,store_name STRING COMMENT 'Name of the store' NOT NULL
  ,store_address STRING COMMENT 'Physical address of the store' NOT NULL
  ,store_city STRING COMMENT 'City where the store is located' NOT NULL
  ,store_state STRING COMMENT 'State where the store is located' NOT NULL
  ,store_zip_code STRING COMMENT 'ZIP code of the store location' NOT NULL
  ,store_phone STRING COMMENT 'Contact phone number for the store' NOT NULL
  ,store_email STRING COMMENT 'Contact email for the store' NOT NULL
  ,store_type STRING COMMENT 'Type of store (flagship, outlet, express, popup)' NOT NULL
  ,store_size_sqft INT COMMENT 'Size of the store in square feet' NOT NULL
  ,store_rating DECIMAL(2,1) COMMENT 'Store rating on a scale of 0-5' NOT NULL
  ,store_hours VARIANT COMMENT 'Store operating hours by day'
  ,latitude DECIMAL(10,8) COMMENT 'Store location latitude coordinate' NOT NULL
  ,longitude DECIMAL(11,8) COMMENT 'Store location longitude coordinate'
  
  -- Keep existing inventory fields
  ,store STRING COMMENT 'Store identifier where inventory is located' NOT NULL
  ,store_quantity INT COMMENT 'Current available quantity of product in the specified store'
  ,warehouse STRING COMMENT 'Warehouse identifier where backup inventory is stored'
  ,warehouse_quantity INT COMMENT 'Current available quantity of product in the specified warehouse'
  ,retail_amount DECIMAL(11, 2) COMMENT 'Current retail price of the product'
  ,popularity_rating STRING COMMENT 'Rating indicating how popular/frequently purchased the product is (e.g., high, medium, low)'
  ,department STRING COMMENT 'Department within the store where the product is categorized'
  ,aisle_location STRING COMMENT 'Physical aisle location identifier where the product can be found in store'
  ,is_closeout BOOLEAN COMMENT 'Flag indicating whether the product is marked for closeout/clearance'
  
  -- Additional inventory management fields from store_inventory
  ,min_stock_level INT COMMENT 'Minimum stock level before reorder' NOT NULL
  ,max_stock_level INT COMMENT 'Maximum stock level capacity' NOT NULL
  ,last_restock_date TIMESTAMP COMMENT 'Date of last inventory restock'
  ,last_count_date TIMESTAMP COMMENT 'Date of last physical inventory count'
  ,is_out_of_stock BOOLEAN COMMENT 'Flag indicating if product is out of stock' NOT NULL
  ,is_low_stock BOOLEAN COMMENT 'Flag indicating if product is below minimum stock level' NOT NULL
  ,next_restock_date TIMESTAMP COMMENT 'Expected date of next inventory restock'
  
  -- Demand prediction and analytics
  ,daily_demand_prediction INT COMMENT 'Predicted daily demand quantity' NOT NULL
  ,weekly_demand_prediction INT COMMENT 'Predicted weekly demand quantity' NOT NULL
  ,monthly_demand_prediction INT COMMENT 'Predicted monthly demand quantity' NOT NULL
  ,last_7_days_sales INT COMMENT 'Total sales in the last 7 days' NOT NULL
  ,last_30_days_sales INT COMMENT 'Total sales in the last 30 days' NOT NULL
  ,last_90_days_sales INT COMMENT 'Total sales in the last 90 days' NOT NULL
  ,days_until_stockout INT COMMENT 'Predicted days until stock depletion' NOT NULL
  ,stockout_risk_level STRING COMMENT 'Risk level of stockout (low, medium, high, critical)' NOT NULL
  
  -- Seasonality and trend analysis
  ,is_seasonal BOOLEAN COMMENT 'Flag indicating if product has seasonal demand patterns' NOT NULL
  ,season_peak_factor DECIMAL(5,2) COMMENT 'Seasonal demand multiplier' NOT NULL
  ,trend_direction STRING COMMENT 'Current sales trend (increasing, stable, decreasing)' NOT NULL
  ,trend_strength DECIMAL(5,2) COMMENT 'Strength of the current trend (0-1)' NOT NULL
  
  -- Metadata
  ,last_prediction_update TIMESTAMP COMMENT 'Timestamp of last demand prediction update' NOT NULL
  ,is_store_active BOOLEAN COMMENT 'Flag indicating if store is currently active' NOT NULL
  ,store_created_at TIMESTAMP COMMENT 'Store creation timestamp' NOT NULL
  ,store_last_updated TIMESTAMP COMMENT 'Last store update timestamp' NOT NULL
  
  -- Foreign key constraint
  ,FOREIGN KEY (product_id) REFERENCES products(product_id)
)
CLUSTER BY AUTO
COMMENT 'Enhanced inventory tracking table that maintains current product quantities across stores and warehouses, including detailed store information, location data, and advanced inventory analytics'
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.feature.variantType' = 'supported'
);
