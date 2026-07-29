-- Unity Catalog Function: extract_store_numbers
-- Description: Extracts store numbers from natural language text using pattern matching
-- Store numbers are typically 3-4 digit numeric values

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.extract_store_numbers(
  input_text STRING COMMENT 'Text input that may contain store numbers'
)
RETURNS ARRAY<STRING>
READS SQL DATA
COMMENT 'Extracts store numbers from natural language text using pattern matching. Store numbers are typically 3-4 digit numeric values.'
RETURN 
SELECT 
  COLLECT_LIST(DISTINCT store_id) as store_numbers
FROM {catalog_name}.{schema_name}.dim_stores
WHERE input_text RLIKE CONCAT('\\\\b', store_id, '\\\\b');
