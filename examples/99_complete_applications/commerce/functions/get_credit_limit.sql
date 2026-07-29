CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.get_credit_limit(
  customer_id STRING COMMENT 'B2B customer identifier (B####)'
)
RETURNS TABLE(
   customer_id STRING COMMENT 'B2B customer identifier'
  ,business_name STRING COMMENT 'B2B business name'
  ,credit_limit DOUBLE COMMENT 'Total credit line in USD'
  ,credit_available DOUBLE COMMENT 'Currently available credit'
  ,credit_used DOUBLE COMMENT 'Credit currently consumed'
  ,payment_terms STRING COMMENT 'Payment terms (Net30, Net60, COD)'
  ,risk_rating STRING COMMENT 'Internal risk rating (A, B, C, D)'
  ,last_review_date DATE COMMENT 'Most recent credit review date'
)
READS SQL DATA
COMMENT 'Retrieve credit limit, availability, and payment terms for a Commerce Swarm B2B customer. Returns no rows for B2C customers.'
RETURN
SELECT
   c.customer_id
  ,c.last_name AS business_name
  ,cl.credit_limit
  ,cl.credit_available
  ,cl.credit_used
  ,cl.payment_terms
  ,cl.risk_rating
  ,cl.last_review_date
FROM {catalog_name}.{schema_name}.credit_limits cl
JOIN {catalog_name}.{schema_name}.customers c ON cl.customer_id = c.customer_id
WHERE cl.customer_id = get_credit_limit.customer_id
;
