USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE customers (
  customer_id STRING COMMENT 'Unique customer identifier (used as user_id for memory namespace)' NOT NULL PRIMARY KEY
  ,email STRING COMMENT 'Primary email contact' NOT NULL
  ,first_name STRING COMMENT 'Given name (B2C) or "B2B" sentinel for business accounts'
  ,last_name STRING COMMENT 'Family name (B2C) or business name (B2B)'
  ,customer_type STRING COMMENT 'B2C or B2B'
  ,segment STRING COMMENT 'Marketing segment (consumer/gifting/baker_hobbyist for B2C; foodservice/hospitality/catering/retail/stadium for B2B)'
  ,signup_date DATE COMMENT 'Date customer registered'
  ,city STRING COMMENT 'Billing city'
  ,state STRING COMMENT 'Billing state / region'
  ,country STRING COMMENT 'Two-letter ISO country code'
  ,loyalty_tier STRING COMMENT 'B2C loyalty tier (bronze/silver/gold/platinum); NULL for B2B accounts'
  ,created_at TIMESTAMP COMMENT 'Row creation timestamp'
)
CLUSTER BY AUTO
COMMENT 'Commerce Swarm customers — mix of B2C consumers and B2B foodservice/hospitality accounts.'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
