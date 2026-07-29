-- Unity Catalog Function: top_offers_by_segment
-- Description: Returns the top redeemed offers within a segment over a
-- lookback window. Used by the redemption_outcomes agent for after-the-fact
-- performance questions.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.top_offers_by_segment(
  segment STRING COMMENT 'Loyalty tier to filter on (Standard | Silver | Gold | Premium | ALL).',
  window_days INT COMMENT 'Lookback window in days.'
)
RETURNS TABLE(
  offer_id          STRING,
  offer_name        STRING,
  brand             STRING,
  category          STRING,
  segment           STRING,
  redemption_count  BIGINT,
  unique_customers  BIGINT,
  total_value       DOUBLE
)
READS SQL DATA
COMMENT 'Returns offers ranked by redemption volume within the requested loyalty segment over the lookback window.'
RETURN
SELECT
  oc.offer_id,
  oc.name AS offer_name,
  oc.brand,
  oc.category,
  cf.loyalty_tier AS segment,
  count(*)                 AS redemption_count,
  count(DISTINCT r.customer_id) AS unique_customers,
  sum(r.redemption_value)  AS total_value
FROM {catalog_name}.{schema_name}.redemptions r
JOIN {catalog_name}.{schema_name}.offer_catalog oc USING (offer_id)
JOIN {catalog_name}.{schema_name}.customer_features cf
  ON r.customer_id = cf.customer_id
WHERE r.redeemed_ts >= current_timestamp() - make_interval(0, 0, 0, top_offers_by_segment.window_days, 0, 0, 0)
  AND (top_offers_by_segment.segment = 'ALL' OR cf.loyalty_tier = top_offers_by_segment.segment)
GROUP BY oc.offer_id, oc.name, oc.brand, oc.category, cf.loyalty_tier
ORDER BY redemption_count DESC
LIMIT 20;
