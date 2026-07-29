-- Unity Catalog Function: find_employee_manager
-- Description: Looks up the direct manager and contact details for one or more employees by
--              joining `employee_performance` to `managers` on `manager_id`.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_employee_manager(
  employee_id ARRAY<STRING> COMMENT 'One or more employee identifiers (e.g., "EMP-016")'
)
RETURNS TABLE(
  employee_id STRING COMMENT 'Employee identifier'
  ,employee_name STRING COMMENT 'Employee full name'
  ,manager_id STRING COMMENT 'Manager identifier'
  ,manager_name STRING COMMENT 'Manager full name'
  ,manager_email STRING COMMENT 'Manager email address'
  ,manager_phone STRING COMMENT 'Manager phone number'
  ,manager_slack_user_id STRING COMMENT 'Manager Slack user ID'
  ,preferred_communication_method STRING COMMENT 'Manager preferred communication method (email, slack, teams, phone)'
  ,manager_department STRING COMMENT 'Department the manager oversees'
  ,store_name STRING COMMENT 'Store where the manager works'
)
READS SQL DATA
COMMENT 'Looks up the direct manager and contact details for one or more employees.'
RETURN
SELECT DISTINCT
  ep.employee_id,
  ep.employee_name,
  m.manager_id,
  m.manager_name,
  m.email_address AS manager_email,
  m.phone_number AS manager_phone,
  m.slack_user_id AS manager_slack_user_id,
  m.preferred_communication_method,
  m.department AS manager_department,
  m.store_name
FROM {catalog_name}.{schema_name}.employee_performance ep
JOIN {catalog_name}.{schema_name}.managers m
  ON ep.manager_id = m.manager_id
WHERE ARRAY_CONTAINS(find_employee_manager.employee_id, ep.employee_id);
