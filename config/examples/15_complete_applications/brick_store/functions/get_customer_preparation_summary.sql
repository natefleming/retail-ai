-- Unity Catalog Function: get_customer_preparation_summary
-- Description: Returns the comprehensive customer preparation summary for one or more customers,
--              including style preferences, sizes, stylist info, and time-until-appointment.
--              Wraps the `customer_preparation_summary` view.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.get_customer_preparation_summary(
  customer_id ARRAY<STRING> COMMENT 'One or more customer identifiers (e.g., ["CUST-005"])'
)
RETURNS TABLE(
  customer_id STRING COMMENT 'Customer identifier'
  ,customer_name STRING COMMENT 'Customer full name'
  ,preferred_name STRING COMMENT 'Customer preferred name'
  ,customer_tier STRING COMMENT 'Customer tier'
  ,preferred_store_id STRING COMMENT 'Preferred store identifier'
  ,store_name STRING COMMENT 'Preferred store name'
  ,next_appointment_date TIMESTAMP COMMENT 'Date and time of the next appointment'
  ,appointment_type STRING COMMENT 'Type of upcoming appointment'
  ,appointment_purpose STRING COMMENT 'Purpose or occasion for the appointment'
  ,style_preferences STRING COMMENT 'Customer style preferences'
  ,size_information STRING COMMENT 'Clothing sizes and fit preferences (JSON)'
  ,color_preferences STRING COMMENT 'Color preferences and dislikes'
  ,brand_preferences STRING COMMENT 'Brand preferences'
  ,budget_range STRING COMMENT 'Typical spending range per visit'
  ,preparation_notes STRING COMMENT 'Special preparation notes'
  ,service_notes STRING COMMENT 'Important service notes'
  ,special_occasions STRING COMMENT 'Important dates and occasions (JSON)'
  ,dietary_restrictions STRING COMMENT 'Dietary restrictions for refreshments'
  ,accessibility_needs STRING COMMENT 'Accessibility requirements'
  ,requires_manager_greeting BOOLEAN COMMENT 'TRUE when the manager should personally greet the customer'
  ,customer_alerts STRING COMMENT 'Special alerts or flags for staff attention'
  ,preferred_stylist_id STRING COMMENT 'Preferred stylist employee identifier'
  ,preferred_stylist_name STRING COMMENT 'Preferred stylist full name'
  ,stylist_experience INT COMMENT 'Number of personal shopping sessions completed by the stylist'
  ,stylist_rating DOUBLE COMMENT 'Stylist customer satisfaction score (1.0-5.0)'
  ,customer_satisfaction DOUBLE COMMENT 'Customer''s own satisfaction score (1.0-5.0)'
  ,total_lifetime_spend DOUBLE COMMENT 'Customer total lifetime spend'
  ,average_transaction_value DOUBLE COMMENT 'Customer average transaction value'
  ,last_visit_date DATE COMMENT 'Date of last store visit'
  ,visit_frequency STRING COMMENT 'How often the customer visits'
  ,days_since_last_visit INT COMMENT 'Days since the customer''s last visit'
  ,hours_until_appointment DOUBLE COMMENT 'Hours until the next scheduled appointment'
)
READS SQL DATA
COMMENT 'Returns the customer preparation summary including stylist details and time-until-appointment.'
RETURN
SELECT customer_id, customer_name, preferred_name, customer_tier, preferred_store_id, store_name,
       next_appointment_date, appointment_type, appointment_purpose, style_preferences,
       size_information, color_preferences, brand_preferences, budget_range, preparation_notes,
       service_notes, special_occasions, dietary_restrictions, accessibility_needs,
       requires_manager_greeting, customer_alerts, preferred_stylist_id, preferred_stylist_name,
       stylist_experience, stylist_rating, customer_satisfaction, total_lifetime_spend,
       average_transaction_value, last_visit_date, visit_frequency, days_since_last_visit,
       hours_until_appointment
FROM {catalog_name}.{schema_name}.customer_preparation_summary
WHERE ARRAY_CONTAINS(get_customer_preparation_summary.customer_id, customer_id);
