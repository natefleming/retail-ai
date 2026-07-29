USE IDENTIFIER(:database);

-- Customer 360 feature table. One row per customer with everything the
-- offer-ranking prompt needs. Built as a materialized table from
-- loyalty_events + receipts + receipt_lines + redemptions so it can be
-- recomputed by a single SQL refresh (no Spark code required).
--
-- The dao-ai dataset pipeline runs this DDL exactly once at provisioning;
-- after that, the customer_features table is treated like any other Delta
-- table. Refresh via `CREATE OR REPLACE TABLE customer_features AS ...`
-- on whatever cadence the customer needs.
CREATE OR REPLACE TABLE customer_features AS
WITH base AS (
    SELECT
        e.customer_id,
        max_by(e.tier_after, e.event_ts) AS loyalty_tier,
        min(CASE WHEN e.event_type = 'ENROLL' THEN e.event_ts END)::date AS enrolled_at
    FROM loyalty_events e
    GROUP BY e.customer_id
),
receipt_agg AS (
    SELECT
        customer_id,
        max(receipt_ts)::date AS last_visit_date,
        count(*) AS receipts_lifetime,
        count(CASE WHEN receipt_ts >= current_timestamp() - INTERVAL 90 DAYS THEN 1 END) AS visits_90d,
        avg(basket_total) AS aov_lifetime,
        avg(CASE WHEN receipt_ts >= current_timestamp() - INTERVAL 90 DAYS THEN basket_total END) AS aov_90d,
        avg(item_count) AS avg_basket_items,
        sum(basket_total) AS total_lifetime_spend,
        sum(CASE WHEN on_promo THEN 1 ELSE 0 END)::double / nullif(count(*), 0) AS pct_on_promo_lifetime,
        sum(CASE WHEN receipt_ts >= current_timestamp() - INTERVAL 180 DAYS AND on_promo THEN 1 ELSE 0 END)::double
            / nullif(sum(CASE WHEN receipt_ts >= current_timestamp() - INTERVAL 180 DAYS THEN 1 ELSE 0 END), 0) AS pct_on_promo_180d
    FROM receipts
    GROUP BY customer_id
),
line_agg AS (
    SELECT
        r.customer_id,
        slice(transform(
            sort_array(collect_list(struct(brand_count, brand)), false),
            x -> x.brand
        ), 1, 3) AS top_brands,
        slice(transform(
            sort_array(collect_list(struct(cat_count, category)), false),
            x -> x.category
        ), 1, 3) AS top_categories,
        avg(line_discount) AS avg_line_discount,
        sum(line_discount) AS total_savings_lifetime
    FROM (
        SELECT
            r.customer_id,
            rl.brand,
            rl.category,
            rl.line_price,
            rl.line_discount,
            count(*) OVER (PARTITION BY r.customer_id, rl.brand) AS brand_count,
            count(*) OVER (PARTITION BY r.customer_id, rl.category) AS cat_count
        FROM receipts r
        JOIN receipt_lines rl USING (receipt_id)
    ) r
    GROUP BY r.customer_id
),
redemption_agg AS (
    SELECT
        customer_id,
        count(*) AS redemptions_lifetime,
        count(CASE WHEN redeemed_ts >= current_timestamp() - INTERVAL 90 DAYS THEN 1 END) AS redemptions_90d,
        max_by(offer_id, redeemed_ts) AS last_redeemed_offer_id,
        max(redeemed_ts)::date AS last_redemption_date
    FROM redemptions
    GROUP BY customer_id
)
SELECT
    b.customer_id,
    coalesce(b.loyalty_tier, 'Standard') AS loyalty_tier,
    b.enrolled_at,
    datediff(current_date(), b.enrolled_at) AS days_enrolled,
    -- Recency / frequency / monetary
    ra.last_visit_date,
    coalesce(datediff(current_date(), ra.last_visit_date), 9999) AS days_since_last_visit,
    coalesce(ra.visits_90d, 0) AS visits_90d,
    coalesce(ra.receipts_lifetime, 0) AS receipts_lifetime,
    round(coalesce(ra.aov_90d, ra.aov_lifetime, 0.0), 2) AS aov,
    round(coalesce(ra.total_lifetime_spend, 0.0), 2) AS total_lifetime_spend,
    round(coalesce(ra.avg_basket_items, 0.0), 2) AS avg_basket_items,
    -- Price sensitivity (0-1; higher = more price-sensitive)
    round(coalesce(ra.pct_on_promo_180d, ra.pct_on_promo_lifetime, 0.0), 3) AS price_tolerance_score,
    -- Brand & category preferences
    coalesce(la.top_brands, array()) AS top_brands,
    coalesce(la.top_categories, array()) AS top_categories,
    -- Inverse-frequency "avoided" lists (brands/categories the customer
    -- has bought ≤1 unit of in the last 18mo despite >=10 total purchases).
    -- Empty for now; cast to ARRAY<STRING> so Delta accepts the column
    -- (a bare array() literal has NullType element type).
    cast(array() AS ARRAY<STRING>) AS avoided_brands,
    cast(array() AS ARRAY<STRING>) AS avoided_categories,
    -- Redemption history
    coalesce(red.redemptions_lifetime, 0) AS redemptions_lifetime,
    coalesce(red.redemptions_90d, 0) AS redemptions_90d,
    red.last_redeemed_offer_id,
    red.last_redemption_date,
    -- Promotion response: redemptions per receipt (proxy for how often a
    -- visit converts when an offer is in-hand)
    round(coalesce(red.redemptions_lifetime / nullif(ra.receipts_lifetime, 0), 0.0), 3) AS promo_response_rate
FROM base b
LEFT JOIN receipt_agg ra USING (customer_id)
LEFT JOIN line_agg la USING (customer_id)
LEFT JOIN redemption_agg red USING (customer_id);
