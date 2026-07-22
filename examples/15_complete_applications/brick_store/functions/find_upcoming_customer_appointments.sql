-- Unity Catalog Function: find_upcoming_customer_appointments
-- Description: Returns customer appointments scheduled within the next N days, optionally
--              scoped to one or more preferred stores. Wraps the `upcoming_customer_appointments`
--              view, which is restricted to active customers with future appointments.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_upcoming_customer_appointments(
  store_id ARRAY<STRING> COMMENT 'One or more preferred store identifiers to scope to. Pass an empty array to include all stores.'
  ,days_ahead INT COMMENT 'Number of days into the future to include (e.g., 1 for today, 7 for the next week)'
)
RETURNS TABLE(
  customer_id STRING COMMENT 'Customer identifier'
  ,customer_name STRING COMMENT 'Customer full name'
  ,preferred_name STRING COMMENT 'Customer preferred name'
  ,customer_tier STRING COMMENT 'Customer tier (Premium, Gold, Silver, Standard)'
  ,preferred_store_id STRING COMMENT 'Preferred store identifier'
  ,store_name STRING COMMENT 'Preferred store name'
  ,preferred_stylist_id STRING COMMENT 'Preferred personal stylist employee identifier'
  ,preferred_stylist_name STRING COMMENT 'Preferred personal stylist name'
  ,next_appointment_date TIMESTAMP COMMENT 'Date and time of the next appointment'
  ,appointment_type STRING COMMENT 'Type of appointment (personal styling, wardrobe consultation, etc.)'
  ,appointment_purpose STRING COMMENT 'Purpose or occasion for the appointment'
  ,style_preferences STRING COMMENT 'Customer style preferences and styling notes'
  ,budget_range STRING COMMENT 'Customer typical spending range per visit'
  ,preparation_notes STRING COMMENT 'Special preparation notes for the upcoming visit'
  ,requires_manager_greeting BOOLEAN COMMENT 'TRUE when the manager should personally greet the customer'
  ,customer_alerts STRING COMMENT 'Special alerts or flags for staff attention'
)
READS SQL DATA
COMMENT 'Returns upcoming customer appointments within a specified number of days, optionally scoped by store.'
RETURN
SELECT customer_id, customer_name, preferred_name, customer_tier, preferred_store_id,
       store_name, preferred_stylist_id, preferred_stylist_name, next_appointment_date,
       appointment_type, appointment_purpose, style_preferences, budget_range,
       preparation_notes, requires_manager_greeting, customer_alerts
FROM {catalog_name}.{schema_name}.upcoming_customer_appointments
WHERE next_appointment_date <= DATEADD(DAY, find_upcoming_customer_appointments.days_ahead, CURRENT_TIMESTAMP())
  AND (SIZE(find_upcoming_customer_appointments.store_id) = 0
       OR ARRAY_CONTAINS(find_upcoming_customer_appointments.store_id, preferred_store_id))
ORDER BY next_appointment_date ASC;
