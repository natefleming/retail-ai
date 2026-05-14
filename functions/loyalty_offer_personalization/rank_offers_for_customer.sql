-- Unity Catalog Function: rank_offers_for_customer
-- Description: SINGLE SOURCE OF TRUTH for the offer-ranking prompt.
--
-- For a given customer and a candidate pool of offer_ids, builds the
-- personalization prompt (Customer 360 features + candidate offer attributes)
-- and calls `ai_query` against the configured model-serving endpoint with
-- responseFormat='json_object'. Returns the parsed 10-element ranking.
--
-- Called by:
--   * the what_if_ranker agent tool (real-time, single customer)
--   * refresh_offer_rankings(...) (batch, all customers via a JOIN)
--   * the deployed agent's real-time scoring endpoint
--
-- Edit this function to change the prompt. Every surface picks up the change.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.rank_offers_for_customer(
  customer_id STRING COMMENT 'Loyalty customer id.',
  prompt_version STRING COMMENT 'Prompt version tag stamped onto the output row (e.g., v1).',
  candidate_offer_ids ARRAY<STRING> COMMENT 'The 10-30 offers to rank for this customer. Pre-filtered for eligibility upstream.'
)
RETURNS ARRAY<STRUCT<offer_id: STRING, rank: INT, score: DOUBLE, reason: STRING>>
LANGUAGE SQL
COMMENT 'Calls the chat-completion endpoint with the personalization prompt for this customer + candidate offers. Returns the parsed 10-element ranking. The prompt body lives inline here so it is the single source of truth across batch, agentic, and real-time surfaces.'
RETURN
  from_json(
    ai_query(
      'databricks-claude-sonnet-4-5',
      concat(
        '<task>',
          'You are a retail offer-personalization ranker. Given one customer profile ',
          'and a list of candidate offers, return a JSON object with key "ranking" ',
          'whose value is an array of exactly ', cast(least(size(candidate_offer_ids), 10) AS STRING),
          ' objects. Each object has: offer_id (string), rank (1-indexed integer, lowest=best), ',
          'score (0-100 double; higher means more likely to redeem), and reason ',
          '(one sentence citing the specific customer feature(s) that drove the rank — ',
          'brand preference, category preference, price tolerance, recency, redemption history). ',
          'Do not include offers outside the candidate list. Do not invent offers. ',
          'Return ONLY the JSON object, no prose.',
        '</task>',
        '<customer_profile>',
          to_json(named_struct(
            'customer_id',            cf.customer_id,
            'loyalty_tier',           cf.loyalty_tier,
            'days_since_last_visit',  cf.days_since_last_visit,
            'visits_90d',             cf.visits_90d,
            'aov',                    cf.aov,
            'total_lifetime_spend',   cf.total_lifetime_spend,
            'avg_basket_items',       cf.avg_basket_items,
            'price_tolerance_score',  cf.price_tolerance_score,
            'top_brands',             cf.top_brands,
            'top_categories',         cf.top_categories,
            'avoided_brands',         cf.avoided_brands,
            'avoided_categories',     cf.avoided_categories,
            'redemptions_90d',        cf.redemptions_90d,
            'last_redeemed_offer_id', cf.last_redeemed_offer_id,
            'promo_response_rate',    cf.promo_response_rate
          )),
        '</customer_profile>',
        '<candidate_offers>',
          to_json(co.candidates),
        '</candidate_offers>'
      ),
      responseFormat => 'json_object',
      modelParameters => named_struct('temperature', 0.1, 'max_tokens', 1500)
    ):ranking,
    'ARRAY<STRUCT<offer_id: STRING, rank: INT, score: DOUBLE, reason: STRING>>'
  )
FROM {catalog_name}.{schema_name}.customer_features cf
CROSS JOIN (
  SELECT collect_list(named_struct(
    'offer_id',      o.offer_id,
    'name',          o.name,
    'description',   o.description,
    'brand',         o.brand,
    'category',      o.category,
    'discount_kind', o.discount_kind,
    'discount_pct',  o.discount_pct,
    'margin_class',  o.margin_class,
    'seasonal_tag',  o.seasonal_tag
  )) AS candidates
  FROM {catalog_name}.{schema_name}.offer_catalog o
  WHERE array_contains(rank_offers_for_customer.candidate_offer_ids, o.offer_id)
) co
WHERE cf.customer_id = rank_offers_for_customer.customer_id;
