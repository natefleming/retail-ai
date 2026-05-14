-- Unity Catalog Function: refresh_offer_rankings
-- Description: The batch personalization path. Returns one row per customer
-- with the ranking already computed by ai_query (via rank_offers_for_customer).
--
-- To produce a refresh, INSERT the result into offer_rankings:
--
--   INSERT INTO loyalty_offers.offer_rankings
--   SELECT * FROM loyalty_offers.refresh_offer_rankings('v1', 'databricks-claude-sonnet-4-5');
--
-- Schedule the INSERT as a daily SQL task on a Pro SQL Warehouse. For 40M
-- customers, shard by customer_id % N and add a WHERE filter on the
-- function call site to run shards in parallel.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.refresh_offer_rankings(
  prompt_version STRING DEFAULT 'v1' COMMENT 'Prompt version tag stamped onto every output row.',
  model_endpoint STRING DEFAULT 'databricks-claude-sonnet-4-5' COMMENT 'Model serving endpoint tag for observability.'
)
RETURNS TABLE(
  customer_id     STRING,
  prompt_version  STRING,
  model_endpoint  STRING,
  generated_at    TIMESTAMP,
  ranking         ARRAY<STRUCT<offer_id: STRING, rank: INT, score: DOUBLE, reason: STRING>>
)
READS SQL DATA
COMMENT 'Batch personalization. Returns one row per eligible customer with the LLM-ranked offer list, ready to INSERT into offer_rankings.'
RETURN
SELECT
  cx.customer_id,
  refresh_offer_rankings.prompt_version,
  refresh_offer_rankings.model_endpoint,
  current_timestamp() AS generated_at,
  {catalog_name}.{schema_name}.rank_offers_for_customer(
    cx.customer_id,
    refresh_offer_rankings.prompt_version,
    cx.eligible_offer_ids
  ) AS ranking
FROM {catalog_name}.{schema_name}.customers_x_eligible_offers cx
WHERE cx.eligible_offer_count >= 10;
