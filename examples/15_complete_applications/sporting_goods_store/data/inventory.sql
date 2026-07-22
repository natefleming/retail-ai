USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE inventory (
  inventory_id BIGINT COMMENT 'Unique identifier for each inventory record'
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table identifying the specific product'
  
  -- Store location information
  ,store_id BIGINT COMMENT 'Unique identifier for the store'
  ,store_name STRING COMMENT 'Name of the store'
  ,store_address STRING COMMENT 'Physical address of the store'
  ,store_city STRING COMMENT 'City where the store is located'
  ,store_state STRING COMMENT 'State where the store is located'
  ,store_zip_code STRING COMMENT 'ZIP code of the store location'
  ,store_phone STRING COMMENT 'Contact phone number for the store'
  ,store_type STRING COMMENT 'Type of store (flagship, standard, outlet, express)'
  ,latitude DOUBLE COMMENT 'Store location latitude coordinate'
  ,longitude DOUBLE COMMENT 'Store location longitude coordinate'
  
  -- Core inventory fields
  ,store STRING COMMENT 'Store identifier where inventory is located'
  ,store_quantity INT COMMENT 'Current available quantity of product in the specified store'
  ,warehouse STRING COMMENT 'Warehouse identifier where backup inventory is stored'
  ,warehouse_quantity INT COMMENT 'Current available quantity of product in the specified warehouse'
  ,retail_amount DOUBLE COMMENT 'Current retail price of the product'
  ,popularity_rating STRING COMMENT 'Rating indicating how popular the product is (high, medium, low)'
  ,department STRING COMMENT 'Department within the store where the product is categorized'
  ,aisle_location STRING COMMENT 'Physical aisle location identifier where the product can be found in store'
  ,is_closeout BOOLEAN COMMENT 'Flag indicating whether the product is marked for closeout or clearance'
  
  -- Stock management
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
)
USING DELTA
COMMENT 'Inventory tracking table for sporting goods store maintaining current product quantities across stores and warehouses with demand prediction, seasonality analysis, and stockout risk assessment.'
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact' = 'true'
);
