USE IDENTIFIER(:database);

-- Synthetic redemptions: ~1.5 per customer over 6mo. Offer choice is biased
-- toward the customer cohort's top brand so redemption history correlates
-- with brand preference — the key signal the ranker should learn.

INSERT INTO redemptions
WITH brand_offer_lookup AS (
    -- For each cohort (0..9), pick offers that match the cohort's top brand
    SELECT cohort_idx, collect_list(offer_id) AS offer_pool
    FROM (
        SELECT 0 AS cohort_idx, offer_id FROM offer_catalog WHERE brand = 'Nike'
        UNION ALL SELECT 1, offer_id FROM offer_catalog WHERE brand = 'Adidas'
        UNION ALL SELECT 2, offer_id FROM offer_catalog WHERE brand = 'Lululemon'
        UNION ALL SELECT 3, offer_id FROM offer_catalog WHERE brand = 'Patagonia'
        UNION ALL SELECT 4, offer_id FROM offer_catalog WHERE brand = 'REI'
        UNION ALL SELECT 5, offer_id FROM offer_catalog WHERE brand = 'Levis'
        UNION ALL SELECT 6, offer_id FROM offer_catalog WHERE brand = 'GAP'
        UNION ALL SELECT 7, offer_id FROM offer_catalog WHERE brand = 'JCrew'
        UNION ALL SELECT 8, offer_id FROM offer_catalog WHERE brand = 'BananaRepublic'
        UNION ALL SELECT 9, offer_id FROM offer_catalog WHERE brand = 'Puma'
    )
    GROUP BY cohort_idx
)
SELECT
    concat('RED-', lpad(cast(monotonically_increasing_id() AS STRING), 9, '0')) AS redemption_id,
    concat('C-', lpad(cast(pmod(cast(rand(601 + id) * 10000 AS BIGINT), 10000) + 1 AS STRING), 5, '0')) AS customer_id,
    element_at(bol.offer_pool, cast(rand(602 + id) * size(bol.offer_pool) AS INT) + 1) AS offer_id,
    NULL AS receipt_id,
    current_timestamp() - make_interval(0, 0, 0, cast(rand(603 + id) * 180 AS INT), 0, 0, 0) AS redeemed_ts,
    round(rand(604 + id) * 30 + 5, 2) AS redemption_value
FROM range(15000) r
JOIN brand_offer_lookup bol
    ON pmod(cast(rand(605 + r.id) * 10000 AS BIGINT), 10) = bol.cohort_idx;
