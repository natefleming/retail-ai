USE IDENTIFIER(:database);

-- Per-customer narrowed candidate pool. For each customer, build an array of
-- eligible offer_ids based on:
--   * loyalty tier eligibility (parsed from offer_catalog.eligibility_json)
--   * lifetime spend thresholds
--   * date validity
--   * cap at 30 offers (the LLM gets a curated, not exhaustive, pool)
-- Then sort by simple affinity (brand match, then category match, then evergreen)
-- so the LLM sees the most plausible candidates first.
CREATE OR REPLACE TABLE customers_x_eligible_offers AS
WITH tier_order AS (
    SELECT 'Standard' AS tier, 0 AS rank UNION ALL
    SELECT 'Silver',   1 UNION ALL
    SELECT 'Gold',     2 UNION ALL
    SELECT 'Premium',  3
),
expanded AS (
    SELECT
        cf.customer_id,
        cf.loyalty_tier,
        cf.total_lifetime_spend,
        cf.top_brands,
        cf.top_categories,
        oc.offer_id,
        oc.brand                                                            AS offer_brand,
        oc.category                                                         AS offer_category,
        oc.valid_from,
        oc.valid_to,
        get_json_object(oc.eligibility_json, '$.min_tier')                  AS req_tier,
        cast(get_json_object(oc.eligibility_json, '$.min_lifetime_spend') AS DOUBLE) AS req_spend,
        CASE WHEN array_contains(cf.top_brands, oc.brand) THEN 2
             WHEN oc.brand = 'ALL_BRANDS' THEN 1
             ELSE 0 END                                                     AS brand_affinity,
        CASE WHEN array_contains(cf.top_categories, oc.category) THEN 2
             WHEN oc.category IN ('Apparel-Tops','Accessories') THEN 1
             ELSE 0 END                                                     AS category_affinity
    FROM customer_features cf
    CROSS JOIN offer_catalog oc
    WHERE current_date() BETWEEN oc.valid_from AND oc.valid_to
),
filtered AS (
    SELECT e.*
    FROM expanded e
    LEFT JOIN tier_order req ON req.tier = coalesce(e.req_tier, 'Standard')
    LEFT JOIN tier_order cust ON cust.tier = e.loyalty_tier
    WHERE cust.rank >= req.rank
      AND (e.req_spend IS NULL OR e.total_lifetime_spend >= e.req_spend)
),
ranked AS (
    SELECT
        customer_id,
        offer_id,
        offer_brand,
        offer_category,
        brand_affinity,
        category_affinity,
        row_number() OVER (
            PARTITION BY customer_id
            ORDER BY brand_affinity DESC, category_affinity DESC, offer_id
        ) AS rn
    FROM filtered
)
SELECT
    customer_id,
    collect_list(offer_id) AS eligible_offer_ids,
    count(*)               AS eligible_offer_count
FROM ranked
WHERE rn <= 30
GROUP BY customer_id;
