-- Unity Catalog Function: get_offer_ranking
-- Description: Returns the most recent stored offer ranking for a customer
-- under a specific prompt version. Used by the ranking_explainer agent
-- to surface stored rationale + scores.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.get_offer_ranking(
  customer_id STRING COMMENT 'Loyalty customer id.',
  prompt_version STRING COMMENT 'Prompt version to look up (e.g., v1).'
)
RETURNS TABLE(
  customer_id    STRING,
  prompt_version STRING,
  model_endpoint STRING,
  generated_at   TIMESTAMP,
  offer_id       STRING,
  rank           INT,
  score          DOUBLE,
  reason         STRING
)
READS SQL DATA
COMMENT 'Returns the most recent ranking rows for (customer, prompt_version), exploded one row per offer. Use this to explain why specific offers were ranked the way they were.'
RETURN
WITH latest AS (
  SELECT *
  FROM {catalog_name}.{schema_name}.offer_rankings r
  WHERE r.customer_id = get_offer_ranking.customer_id
    AND r.prompt_version = get_offer_ranking.prompt_version
  QUALIFY row_number() OVER (PARTITION BY r.customer_id, r.prompt_version ORDER BY r.generated_at DESC) = 1
)
SELECT
  l.customer_id,
  l.prompt_version,
  l.model_endpoint,
  l.generated_at,
  exploded.offer_id,
  exploded.rank,
  exploded.score,
  exploded.reason
FROM latest l
LATERAL VIEW explode(l.ranking) AS exploded
ORDER BY exploded.rank;
