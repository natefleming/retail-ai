USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE dim_stores (
  store_id INT COMMENT 'Unique identifier for each store location'
  ,store_name STRING COMMENT 'Display name of the store location'
  ,store_address STRING COMMENT 'Street address of the store location'
  ,store_city STRING COMMENT 'City where the store is located'
  ,store_state STRING COMMENT 'State abbreviation (e.g., NY, CA, TX)'
  ,store_zipcode STRING COMMENT 'Postal or ZIP code for the store location'
  ,store_country STRING COMMENT 'Country where the store is located'
  ,store_phone STRING COMMENT 'Primary contact phone number for the store'
  ,store_email STRING COMMENT 'Primary contact email address for the store'
  ,store_manager_id STRING COMMENT 'Identifier of the current store manager'
  ,store_manager_name STRING COMMENT 'Name of the current store manager'
  ,opening_date DATE COMMENT 'Date when the store first opened for business'
  ,store_type STRING COMMENT 'Type of store (flagship, standard, outlet, express)'
  ,store_size_sqft INT COMMENT 'Total floor space of the store in square feet'
  ,store_rating DOUBLE COMMENT 'Customer rating of the store on a scale of 1.0 to 5.0'
  ,store_hours STRING COMMENT 'Operating hours for each day of the week in JSON format'
  ,latitude DOUBLE COMMENT 'Geographic latitude coordinate of the store location'
  ,longitude DOUBLE COMMENT 'Geographic longitude coordinate of the store location'
  ,region_id STRING COMMENT 'Identifier for the region the store belongs to'
  ,region_name STRING COMMENT 'Name of the sales region'
  ,district_id STRING COMMENT 'Identifier for the district'
  ,district_name STRING COMMENT 'Name of the district'
  ,timezone STRING COMMENT 'Time zone identifier (e.g., America/New_York)'
  ,is_active BOOLEAN COMMENT 'Flag indicating whether the store is currently operational'
  ,has_footwear BOOLEAN COMMENT 'Flag indicating if the store has a footwear department'
  ,has_apparel BOOLEAN COMMENT 'Flag indicating if the store has an apparel department'
  ,has_team_sports BOOLEAN COMMENT 'Flag indicating if the store has a team sports department'
  ,has_fitness BOOLEAN COMMENT 'Flag indicating if the store has a fitness equipment department'
  ,has_outdoor BOOLEAN COMMENT 'Flag indicating if the store has an outdoor and camping department'
  ,has_cycling BOOLEAN COMMENT 'Flag indicating if the store has a cycling department'
  ,has_golf BOOLEAN COMMENT 'Flag indicating if the store has a golf department'
  ,has_hunting_fishing BOOLEAN COMMENT 'Flag indicating if the store has a hunting and fishing department'
  ,store_details_text STRING COMMENT 'Detailed text description of the store including location, departments, hours, and services'
  ,parking_spaces INT COMMENT 'Number of customer parking spaces available'
  ,last_renovation_date DATE COMMENT 'Date of the most recent store renovation'
  ,created_at TIMESTAMP COMMENT 'Timestamp when this store record was created'
  ,updated_at TIMESTAMP COMMENT 'Timestamp when this store record was last updated'
)
USING DELTA
COMMENT 'Store dimension table for sporting goods retail chain with department availability flags and location details.'
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact' = 'true',
  'delta.enableChangeDataFeed' = 'true'
);
