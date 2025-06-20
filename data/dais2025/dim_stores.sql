USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE dim_stores (
    store_id INT COMMENT 'Unique identifier for each store location (e.g., 1, 2)',
    store_name STRING COMMENT 'Display name of the store location (e.g., Downtown, Uptown)',
    store_address STRING COMMENT 'Street address of the store location',
    store_city STRING COMMENT 'City where the store is located',
    store_state STRING COMMENT 'State or province abbreviation (e.g., NY, CA, TX)',
    store_zipcode STRING COMMENT 'Postal/ZIP code for the store location',
    store_country STRING COMMENT 'Country where the store is located (e.g., USA)',
    store_phone STRING COMMENT 'Primary contact phone number for the store',
    store_email STRING COMMENT 'Primary contact email address for the store',
    store_manager_id STRING COMMENT 'Identifier of the current store manager',
    opening_date DATE COMMENT 'Date when the store first opened for business',
    store_area_sqft DOUBLE COMMENT 'Total floor space of the store in square feet',
    is_open_24_hours BOOLEAN COMMENT 'Flag indicating if the store is open 24 hours',
    latitude DOUBLE COMMENT 'Geographic latitude coordinate of the store location',
    longitude DOUBLE COMMENT 'Geographic longitude coordinate of the store location',
    region_id STRING COMMENT 'Identifier for the region the store belongs to',
    store_details_text STRING COMMENT 'Detailed text description of the store including location, hours, and services',
    store_type STRING COMMENT 'Type/category of store (flagship, outlet, express, popup)',
    store_size_sqft INT COMMENT 'Total floor space of the store in square feet',
    store_rating DOUBLE COMMENT 'Customer rating of the store on a scale of 1.0 to 5.0',
    store_hours STRING COMMENT 'Operating hours for each day of the week in JSON format',
    timezone STRING COMMENT 'Time zone identifier for the store location (e.g., America/New_York)',
    is_active BOOLEAN COMMENT 'Flag indicating whether the store is currently operational',
    last_renovation_date DATE COMMENT 'Date of the most recent store renovation or major update',
    parking_spaces INT COMMENT 'Number of customer parking spaces available',
    has_pharmacy BOOLEAN COMMENT 'Flag indicating if the store includes a pharmacy department',
    has_grocery BOOLEAN COMMENT 'Flag indicating if the store includes a grocery department',
    has_electronics BOOLEAN COMMENT 'Flag indicating if the store includes an electronics department',
    has_clothing BOOLEAN COMMENT 'Flag indicating if the store includes a clothing department',
    has_home_goods BOOLEAN COMMENT 'Flag indicating if the store includes a home goods department',
    created_at TIMESTAMP COMMENT 'Timestamp when this store record was created',
    updated_at TIMESTAMP COMMENT 'Timestamp when this store record was last updated'
) 
USING DELTA
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true',
    'delta.enableChangeDataFeed' = 'true'
);
