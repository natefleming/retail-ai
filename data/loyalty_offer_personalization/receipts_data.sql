USE IDENTIFIER(:database);

-- Synthetic receipts: ~6 receipts/customer over last 18 months → ~60K receipts.
-- Each receipt is tied to a customer in C-00001..C-10000.

INSERT INTO receipts
SELECT
    concat('R-', lpad(cast(id AS STRING), 9, '0'))                              AS receipt_id,
    concat('C-', lpad(cast(pmod(cast(rand(401 + id) * 10000 AS BIGINT), 10000) + 1 AS STRING), 5, '0')) AS customer_id,
    cast(rand(402 + id) * 8 + 101 AS INT)                                       AS store_id,
    CASE WHEN rand(403 + id) < 0.55 THEN 'STORE'
         WHEN rand(403 + id) < 0.90 THEN 'ONLINE'
         ELSE 'APP' END                                                         AS channel,
    round(rand(404 + id) * 240 + 25, 2)                                         AS basket_total,
    cast(rand(405 + id) * 6 + 1 AS INT)                                         AS item_count,
    cast(rand(406 + id) * 8 + 1 AS INT)                                         AS units_count,
    rand(407 + id) < 0.35                                                       AS on_promo,
    CASE WHEN rand(407 + id) < 0.35 THEN round(rand(408 + id) * 40 + 5, 2)
         ELSE 0.0 END                                                           AS promo_savings,
    current_timestamp() - make_interval(0, 0, 0, cast(rand(409 + id) * 540 AS INT), 0, 0, 0) AS receipt_ts
FROM range(60000);
