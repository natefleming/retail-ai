-- Function to find store hours by city.
--
-- Reads the `store_hours` table, which is seeded from a JSON file colocated
-- with the config. Serverless Spark cannot read the driver-local checkout, so
-- csv/parquet seeds are staged into a UC volume first — json is read on the
-- driver with pandas and needs no staging. This function proves the resulting
-- table is queryable either way.
CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_store_hours_by_city(
  city STRING COMMENT 'City name to look up store hours for, e.g. Chicago'
)
RETURNS TABLE(
  store_id BIGINT COMMENT 'Unique identifier for the store'
  ,store_name STRING COMMENT 'Customer-facing store name'
  ,city STRING COMMENT 'City the store is located in'
  ,state STRING COMMENT 'Two-letter state code'
  ,weekday_hours STRING COMMENT 'Opening hours Monday through Friday'
  ,weekend_hours STRING COMMENT 'Opening hours Saturday and Sunday'
)
READS SQL DATA
COMMENT 'Returns every store in a city along with its weekday and weekend opening hours. Use this for any question about when a store is open.'
RETURN
SELECT
  store_id
  ,store_name
  ,city
  ,state
  ,weekday_hours
  ,weekend_hours
FROM {catalog_name}.{schema_name}.store_hours
WHERE LOWER(city) = LOWER(find_store_hours_by_city.city);
