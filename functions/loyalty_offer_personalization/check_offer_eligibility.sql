-- Unity Catalog Function: check_offer_eligibility
-- Description: Returns whether a customer is eligible for a given offer
-- based on loyalty tier, lifetime spend, and offer validity window.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.check_offer_eligibility(
  customer_id STRING COMMENT 'Loyalty customer id.',
  offer_id STRING COMMENT 'Offer id to check.'
)
RETURNS TABLE(
  eligible BOOLEAN,
  reason   STRING
)
READS SQL DATA
COMMENT 'Returns (eligible, reason) for a (customer, offer) pair. Reasons cite the failing rule.'
RETURN
WITH tier_order AS (
  SELECT 'Standard' AS tier, 0 AS rank UNION ALL
  SELECT 'Silver',   1 UNION ALL
  SELECT 'Gold',     2 UNION ALL
  SELECT 'Premium',  3
),
joined AS (
  SELECT
    cf.loyalty_tier,
    cf.total_lifetime_spend,
    oc.valid_from,
    oc.valid_to,
    get_json_object(oc.eligibility_json, '$.min_tier')                           AS req_tier,
    cast(get_json_object(oc.eligibility_json, '$.min_lifetime_spend') AS DOUBLE) AS req_spend
  FROM {catalog_name}.{schema_name}.customer_features cf
  CROSS JOIN {catalog_name}.{schema_name}.offer_catalog oc
  WHERE cf.customer_id = check_offer_eligibility.customer_id
    AND oc.offer_id = check_offer_eligibility.offer_id
)
SELECT
  (current_date() BETWEEN j.valid_from AND j.valid_to)
    AND (j.req_spend IS NULL OR j.total_lifetime_spend >= j.req_spend)
    AND (cust.rank >= req.rank)
    AS eligible,
  concat_ws(' | ',
    CASE WHEN current_date() < j.valid_from THEN 'offer not yet active' END,
    CASE WHEN current_date() > j.valid_to   THEN 'offer expired' END,
    CASE WHEN j.req_spend IS NOT NULL AND j.total_lifetime_spend < j.req_spend
         THEN concat('lifetime spend $', cast(j.total_lifetime_spend AS STRING),
                     ' below required $', cast(j.req_spend AS STRING)) END,
    CASE WHEN cust.rank < req.rank
         THEN concat('tier ', j.loyalty_tier, ' below required ', coalesce(j.req_tier, 'Standard')) END
  ) AS reason
FROM joined j
LEFT JOIN tier_order req  ON req.tier  = coalesce(j.req_tier, 'Standard')
LEFT JOIN tier_order cust ON cust.tier = j.loyalty_tier;
