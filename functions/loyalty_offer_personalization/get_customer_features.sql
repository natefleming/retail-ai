-- Unity Catalog Function: get_customer_features
-- Description: Returns the Customer 360 feature row for a given customer.
-- Used by the customer_intelligence and ranking_explainer agents.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.get_customer_features(
  customer_id STRING COMMENT 'Loyalty customer id (e.g., C-00007).'
)
RETURNS TABLE(
  customer_id              STRING,
  loyalty_tier             STRING,
  enrolled_at              DATE,
  days_enrolled            INT,
  last_visit_date          DATE,
  days_since_last_visit    INT,
  visits_90d               BIGINT,
  receipts_lifetime        BIGINT,
  aov                      DOUBLE,
  total_lifetime_spend     DOUBLE,
  avg_basket_items         DOUBLE,
  price_tolerance_score    DOUBLE,
  top_brands               ARRAY<STRING>,
  top_categories           ARRAY<STRING>,
  avoided_brands           ARRAY<STRING>,
  avoided_categories       ARRAY<STRING>,
  redemptions_lifetime     BIGINT,
  redemptions_90d          BIGINT,
  last_redeemed_offer_id   STRING,
  last_redemption_date     DATE,
  promo_response_rate      DOUBLE
)
READS SQL DATA
COMMENT 'Returns the full Customer 360 feature row for one customer. Use this when an operator names a specific customer.'
RETURN
SELECT *
FROM {catalog_name}.{schema_name}.customer_features
WHERE customer_id = get_customer_features.customer_id;
