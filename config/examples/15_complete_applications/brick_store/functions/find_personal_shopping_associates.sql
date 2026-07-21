-- Unity Catalog Function: find_personal_shopping_associates
-- Description: Returns the top personal shopping associates across all stores, ranked by a
--              composite score of session count, customer satisfaction, product knowledge,
--              and overall performance. Wraps the `top_personal_shopping_associates_all_stores` view.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.find_personal_shopping_associates(
  limit_count INT COMMENT 'Maximum number of personal shopping associates to return (e.g., 5)'
)
RETURNS TABLE(
  store_id STRING COMMENT 'Store where the associate works'
  ,store_name STRING COMMENT 'Store name'
  ,employee_id STRING COMMENT 'Employee identifier'
  ,employee_name STRING COMMENT 'Employee full name'
  ,position_title STRING COMMENT 'Employee job title'
  ,personal_shopping_sessions INT COMMENT 'Number of personal shopping sessions conducted'
  ,customer_satisfaction_score DOUBLE COMMENT 'Average customer satisfaction rating (1.0-5.0)'
  ,product_knowledge_score DOUBLE COMMENT 'Product knowledge assessment score (1.0-5.0)'
  ,overall_performance_score DOUBLE COMMENT 'Overall performance score (1.0-5.0)'
  ,comprehensive_score DOUBLE COMMENT 'Composite personal-shopping expertise score'
  ,expertise_level STRING COMMENT 'Expertise level (Expert, Advanced, Intermediate, Beginner)'
  ,overall_rank BIGINT COMMENT 'Cross-store rank (1 = top associate)'
)
READS SQL DATA
COMMENT 'Returns the top personal shopping associates across all stores for the current monthly period.'
RETURN
SELECT store_id, store_name, employee_id, employee_name, position_title,
       personal_shopping_sessions, customer_satisfaction_score, product_knowledge_score,
       overall_performance_score, comprehensive_score, expertise_level, overall_rank
FROM {catalog_name}.{schema_name}.top_personal_shopping_associates_all_stores
WHERE overall_rank <= find_personal_shopping_associates.limit_count
ORDER BY overall_rank ASC;
