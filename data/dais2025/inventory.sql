USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE inventory (
  inventory_id BIGINT COMMENT 'Unique identifier for each inventory record'
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table identifying the specific product'
  
  -- Store location information (enhanced from store)
  ,store_id BIGINT COMMENT 'Unique identifier for the store'
  ,store_name STRING COMMENT 'Name of the store'
  ,store_address STRING COMMENT 'Physical address of the store'
  ,store_city STRING COMMENT 'City where the store is located'
  ,store_state STRING COMMENT 'State where the store is located'
  ,store_zip_code STRING COMMENT 'ZIP code of the store location'
  ,store_phone STRING COMMENT 'Contact phone number for the store'
  ,store_email STRING COMMENT 'Contact email for the store'
  ,store_type STRING COMMENT 'Type of store (flagship, outlet, express, popup)'
  ,store_size_sqft INT COMMENT 'Size of the store in square feet'
  ,store_rating DOUBLE COMMENT 'Store rating on a scale of 0-5'
  ,store_hours STRING COMMENT 'Store operating hours by day in JSON format'
  ,latitude DOUBLE COMMENT 'Store location latitude coordinate'
  ,longitude DOUBLE COMMENT 'Store location longitude coordinate'
  
  -- Keep existing inventory fields
  ,store STRING COMMENT 'Store identifier where inventory is located'
  ,store_quantity INT COMMENT 'Current available quantity of product in the specified store'
  ,warehouse STRING COMMENT 'Warehouse identifier where backup inventory is stored'
  ,warehouse_quantity INT COMMENT 'Current available quantity of product in the specified warehouse'
  ,retail_amount DOUBLE COMMENT 'Current retail price of the product'
  ,popularity_rating STRING COMMENT 'Rating indicating how popular/frequently purchased the product is (e.g., high, medium, low)'
  ,department STRING COMMENT 'Department within the store where the product is categorized'
  ,aisle_location STRING COMMENT 'Physical aisle location identifier where the product can be found in store'
  ,is_closeout BOOLEAN COMMENT 'Flag indicating whether the product is marked for closeout/clearance'
  
  -- Additional inventory management fields from store_inventory
  ,min_stock_level INT COMMENT 'Minimum stock level before reorder'
  ,max_stock_level INT COMMENT 'Maximum stock level capacity'
  ,last_restock_date TIMESTAMP COMMENT 'Date of last inventory restock'
  ,last_count_date TIMESTAMP COMMENT 'Date of last physical inventory count'
  ,is_out_of_stock BOOLEAN COMMENT 'Flag indicating if product is out of stock'
  ,is_low_stock BOOLEAN COMMENT 'Flag indicating if product is below minimum stock level'
  ,next_restock_date TIMESTAMP COMMENT 'Expected date of next inventory restock'
  
  -- Demand prediction and analytics
  ,daily_demand_prediction INT COMMENT 'Predicted daily demand quantity'
  ,weekly_demand_prediction INT COMMENT 'Predicted weekly demand quantity'
  ,monthly_demand_prediction INT COMMENT 'Predicted monthly demand quantity'
  ,last_7_days_sales INT COMMENT 'Total sales in the last 7 days'
  ,last_30_days_sales INT COMMENT 'Total sales in the last 30 days'
  ,last_90_days_sales INT COMMENT 'Total sales in the last 90 days'
  ,days_until_stockout INT COMMENT 'Predicted days until stock depletion'
  ,stockout_risk_level STRING COMMENT 'Risk level of stockout (low, medium, high, critical)'
  
  -- Seasonality and trend analysis
  ,is_seasonal BOOLEAN COMMENT 'Flag indicating if product has seasonal demand patterns'
  ,season_peak_factor DOUBLE COMMENT 'Seasonal demand multiplier'
  ,trend_direction STRING COMMENT 'Current sales trend (increasing, stable, decreasing)'
  ,trend_strength DOUBLE COMMENT 'Strength of the current trend (0-1)'
  
  -- Metadata
  ,last_prediction_update TIMESTAMP COMMENT 'Timestamp of last demand prediction update'
  ,is_store_active BOOLEAN COMMENT 'Flag indicating if store is currently active'
  ,store_created_at TIMESTAMP COMMENT 'Store creation timestamp'
  ,store_last_updated TIMESTAMP COMMENT 'Last store update timestamp'
)
USING DELTA
COMMENT 'Enhanced inventory tracking table that maintains current product quantities across stores and warehouses, including detailed store information, location data, and advanced inventory analytics. Schema optimized for parquet compatibility.'
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact' = 'true'
);
