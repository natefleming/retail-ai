-- Unity Catalog Function: find_top_employees_by_department
-- Description: Returns the top-ranked employees for a given department for the current monthly
--              performance period. Wraps the `top_employees_by_department` view.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_top_employees_by_department(
  department_name STRING COMMENT 'Department name to filter on (e.g., "Womens Fashion", "Electronics", "Customer Service")'
  ,limit_count INT COMMENT 'Maximum number of top employees to return (e.g., 5)'
)
RETURNS TABLE(
  department STRING COMMENT 'Department name'
  ,store_name STRING COMMENT 'Store where the employee works'
  ,employee_id STRING COMMENT 'Employee identifier'
  ,employee_name STRING COMMENT 'Employee full name'
  ,position_title STRING COMMENT 'Employee job title'
  ,overall_performance_score DOUBLE COMMENT 'Overall performance score (1.0-5.0)'
  ,sales_achievement_percentage DOUBLE COMMENT 'Percentage of sales target achieved'
  ,task_completion_rate DOUBLE COMMENT 'Percentage of tasks completed on time'
  ,customer_satisfaction_score DOUBLE COMMENT 'Average customer satisfaction rating (1.0-5.0)'
  ,attendance_rate DOUBLE COMMENT 'Percentage of scheduled shifts attended'
  ,dept_rank BIGINT COMMENT 'Rank within the department (1 = top performer)'
)
READS SQL DATA
COMMENT 'Returns the top employees in a given department for the current monthly performance period.'
RETURN
SELECT department, store_name, employee_id, employee_name, position_title,
       overall_performance_score, sales_achievement_percentage, task_completion_rate,
       customer_satisfaction_score, attendance_rate, dept_rank
FROM {catalog_name}.{schema_name}.top_employees_by_department
WHERE LOWER(department) = LOWER(find_top_employees_by_department.department_name)
  AND dept_rank <= find_top_employees_by_department.limit_count
ORDER BY dept_rank ASC;
