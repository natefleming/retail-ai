-- Unity Catalog Function: find_store_by_number
-- Description: Retrieves detailed information about stores by their store IDs
-- This function is designed for store location and information retrieval in retail applications

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_store_by_number(
  store_ids ARRAY<INT> COMMENT 'One or more store identifiers to retrieve. Store IDs are integer values'
)
RETURNS TABLE(
  store_id INT COMMENT 'Unique identifier for each store in the system'
  ,store_name STRING COMMENT 'Display name of the store location'
  ,store_address STRING COMMENT 'Street address of the store location'
  ,store_city STRING COMMENT 'City where the store is located'
  ,store_state STRING COMMENT 'State or province where the store is located'
  ,store_zipcode STRING COMMENT 'Postal code for the store location'
  ,store_country STRING COMMENT 'Country where the store is located'
  ,store_phone STRING COMMENT 'Primary phone number for the store'
  ,store_email STRING COMMENT 'Email address for the store'
  ,store_manager_id STRING COMMENT 'Identifier for the store manager'
  ,opening_date DATE COMMENT 'Date when the store opened'
  ,store_area_sqft DOUBLE COMMENT 'Total floor space of the store in square feet'
  ,is_open_24_hours BOOLEAN COMMENT 'Flag indicating if the store is open 24 hours'
  ,latitude DOUBLE COMMENT 'Latitude coordinate of the store location'
  ,longitude DOUBLE COMMENT 'Longitude coordinate of the store location'
  ,region_id STRING COMMENT 'Identifier for the region the store belongs to'
  ,store_details_text STRING COMMENT 'Detailed text description of the store including location, hours, and services'
)
READS SQL DATA
COMMENT 'Retrieves detailed information about stores by their store IDs. This function is designed for store location and information retrieval in retail applications.'
RETURN 
SELECT 
  store_id
  ,store_name
  ,store_address
  ,store_city
  ,store_state
  ,store_zipcode
  ,store_country
  ,store_phone
  ,store_email
  ,store_manager_id
  ,opening_date
  ,store_area_sqft
  ,is_open_24_hours
  ,latitude
  ,longitude
  ,region_id
  ,store_details_text
FROM {catalog_name}.{schema_name}.dim_stores 
WHERE ARRAY_CONTAINS(find_store_by_number.store_ids, dim_stores.store_id);
