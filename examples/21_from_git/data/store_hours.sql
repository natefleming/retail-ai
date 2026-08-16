USE IDENTIFIER(:database);

CREATE TABLE IF NOT EXISTS store_hours (
  store_id BIGINT COMMENT 'Unique identifier for the store' NOT NULL PRIMARY KEY
  ,store_name STRING COMMENT 'Customer-facing store name'
  ,city STRING COMMENT 'City the store is located in'
  ,state STRING COMMENT 'Two-letter state code'
  ,weekday_hours STRING COMMENT 'Opening hours Monday through Friday'
  ,weekend_hours STRING COMMENT 'Opening hours Saturday and Sunday'
)
COMMENT 'Store hours seeded from a JSON file colocated with the config in git'
;
