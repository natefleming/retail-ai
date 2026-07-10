USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE credit_limits (
  customer_id STRING COMMENT 'FK -> customers.customer_id (B2B accounts only)' NOT NULL PRIMARY KEY
  ,credit_limit DOUBLE COMMENT 'Total credit line in USD'
  ,credit_available DOUBLE COMMENT 'Currently available credit'
  ,credit_used DOUBLE COMMENT 'Credit currently consumed by open balances'
  ,payment_terms STRING COMMENT 'Payment terms (Net30, Net60, COD)'
  ,risk_rating STRING COMMENT 'Internal risk rating (A, B, C, D)'
  ,last_review_date DATE COMMENT 'Most recent credit review date'
  ,FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
)
CLUSTER BY AUTO
COMMENT 'Commerce Swarm B2B credit limits. Backs the credit_limit_handler agent.'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
