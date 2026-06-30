CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.lookup_customer_by_user(
   user_id STRING COMMENT 'Authenticated end-user identifier (typically the SSO email, e.g. alice@example.com). Matched case-insensitively against customers.email.'
)
RETURNS TABLE(
   customer_id    STRING  COMMENT 'Internal customer identifier (C#### for B2C, B#### for B2B)'
  ,email          STRING  COMMENT 'Customer email used for the lookup'
  ,customer_type  STRING  COMMENT 'B2C or B2B'
  ,segment        STRING  COMMENT 'Customer segment (consumer, baker_hobbyist, restaurant, ...)'
  ,loyalty_tier   STRING  COMMENT 'Loyalty tier (bronze / silver / gold / platinum)'
  ,first_name     STRING  COMMENT 'First name'
  ,last_name      STRING  COMMENT 'Last name'
)
READS SQL DATA
COMMENT 'Resolve the internal customer_id for an authenticated user. Pass the authenticated user identifier (typically the SSO email) and receive the matching customer record. Returns at most one row; returns no rows if the user is not a registered Commerce Swarm customer.'
RETURN
SELECT
   customer_id
  ,email
  ,customer_type
  ,segment
  ,loyalty_tier
  ,first_name
  ,last_name
FROM {catalog_name}.{schema_name}.customers
WHERE LOWER(email) = LOWER(lookup_customer_by_user.user_id)
LIMIT 1
;
