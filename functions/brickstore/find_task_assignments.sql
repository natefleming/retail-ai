-- Unity Catalog Function: find_task_assignments
-- Description: Returns today's tasks for one or more stores, optionally filtered by status.
--              Wraps the `employee_daily_tasks` view, which is restricted to assignments where
--              `assigned_date = CURRENT_DATE()`.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_task_assignments(
  store_id ARRAY<STRING> COMMENT 'One or more store identifiers to filter on. Pass an empty array to include all stores.'
  ,task_status_filter STRING COMMENT 'Optional task status to filter on (Pending, In_Progress, Completed, Cancelled, On_Hold, Overdue). Pass an empty string to include all statuses.'
)
RETURNS TABLE(
  task_id STRING COMMENT 'Task identifier'
  ,employee_id STRING COMMENT 'Assigned employee identifier'
  ,store_id STRING COMMENT 'Store where the task is performed'
  ,task_title STRING COMMENT 'Brief title of the task'
  ,task_type STRING COMMENT 'Task type (BOPIS, Service, Restock, Cleaning, etc.)'
  ,priority_level STRING COMMENT 'Priority (Low, Medium, High, Critical, Urgent)'
  ,task_status STRING COMMENT 'Current status'
  ,due_time TIMESTAMP COMMENT 'Time the task is due'
  ,customer_name STRING COMMENT 'Customer name for customer-related tasks'
  ,location_details STRING COMMENT 'Specific location within the store'
  ,department STRING COMMENT 'Department where the task is performed'
  ,is_overdue BOOLEAN COMMENT 'TRUE when the task is past due and not yet completed'
)
READS SQL DATA
COMMENT 'Returns today''s task assignments, optionally filtered by store(s) and status.'
RETURN
SELECT task_id, employee_id, store_id, task_title, task_type, priority_level, task_status,
       due_time, customer_name, location_details, department, is_overdue
FROM {catalog_name}.{schema_name}.employee_daily_tasks
WHERE (SIZE(find_task_assignments.store_id) = 0
       OR ARRAY_CONTAINS(find_task_assignments.store_id, store_id))
  AND (find_task_assignments.task_status_filter = ''
       OR LOWER(task_status) = LOWER(find_task_assignments.task_status_filter));
