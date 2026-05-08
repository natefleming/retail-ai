-- Unity Catalog Function: get_customer_details
-- Description: Looks up a customer profile by customer_id (preferred) or by partial name match
--              against `customer_name` or `preferred_name`.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.get_customer_details(
  customer_lookup STRING COMMENT 'Customer identifier (preferred, e.g., "CUST-005") or partial customer name to match (case-insensitive)'
)
RETURNS TABLE(
  customer_id STRING COMMENT 'Customer identifier'
  ,customer_name STRING COMMENT 'Customer full name'
  ,preferred_name STRING COMMENT 'Customer preferred name'
  ,customer_tier STRING COMMENT 'Customer tier (Premium, Gold, Silver, Standard)'
  ,member_since DATE COMMENT 'Date the customer became a member'
  ,preferred_store_id STRING COMMENT 'Preferred store identifier'
  ,style_preferences STRING COMMENT 'Fashion style preferences and styling notes'
  ,size_information STRING COMMENT 'Clothing sizes and fit preferences (JSON)'
  ,color_preferences STRING COMMENT 'Preferred colors and dislikes'
  ,brand_preferences STRING COMMENT 'Preferred and avoided brands'
  ,budget_range STRING COMMENT 'Typical spending range per visit'
  ,total_lifetime_spend DOUBLE COMMENT 'Total amount spent as customer'
  ,average_transaction_value DOUBLE COMMENT 'Average transaction amount'
  ,last_visit_date DATE COMMENT 'Date of last store visit'
  ,visit_frequency STRING COMMENT 'How often the customer visits'
  ,satisfaction_score DOUBLE COMMENT 'Average satisfaction score (1.0-5.0)'
  ,special_occasions STRING COMMENT 'Important dates and occasions (JSON)'
  ,customer_alerts STRING COMMENT 'Special alerts or flags for staff attention'
  ,requires_manager_greeting BOOLEAN COMMENT 'TRUE when the manager should personally greet the customer'
  ,next_appointment_date TIMESTAMP COMMENT 'Date and time of the next scheduled appointment'
  ,appointment_type STRING COMMENT 'Type of upcoming appointment'
)
READS SQL DATA
COMMENT 'Looks up a customer profile by customer_id (preferred) or by partial name match.'
RETURN
SELECT customer_id, customer_name, preferred_name, customer_tier, member_since,
       preferred_store_id, style_preferences, size_information, color_preferences,
       brand_preferences, budget_range, total_lifetime_spend, average_transaction_value,
       last_visit_date, visit_frequency, satisfaction_score, special_occasions,
       customer_alerts, requires_manager_greeting, next_appointment_date, appointment_type
FROM {catalog_name}.{schema_name}.customers
WHERE customer_id = get_customer_details.customer_lookup
   OR LOWER(customer_name) LIKE LOWER(CONCAT('%', get_customer_details.customer_lookup, '%'))
   OR LOWER(preferred_name) LIKE LOWER(CONCAT('%', get_customer_details.customer_lookup, '%'));
