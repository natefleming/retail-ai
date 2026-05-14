-- Unity Catalog Function: get_recent_receipts
-- Description: Returns the customer's recent receipts + a flattened summary of
-- the brands and categories represented on each receipt. Used by the
-- customer_intelligence agent.

CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.get_recent_receipts(
  customer_id STRING COMMENT 'Loyalty customer id (e.g., C-00007).',
  days INT COMMENT 'How many days back to look (e.g., 90).'
)
RETURNS TABLE(
  receipt_id      STRING,
  receipt_ts      TIMESTAMP,
  channel         STRING,
  store_id        INT,
  basket_total    DOUBLE,
  item_count      INT,
  on_promo        BOOLEAN,
  promo_savings   DOUBLE,
  brands          ARRAY<STRING>,
  categories      ARRAY<STRING>
)
READS SQL DATA
COMMENT 'Returns the customer''s receipts in the requested lookback window, with the brands and categories present on each receipt.'
RETURN
SELECT
  r.receipt_id,
  r.receipt_ts,
  r.channel,
  r.store_id,
  r.basket_total,
  r.item_count,
  r.on_promo,
  r.promo_savings,
  array_distinct(collect_list(rl.brand))    AS brands,
  array_distinct(collect_list(rl.category)) AS categories
FROM {catalog_name}.{schema_name}.receipts r
LEFT JOIN {catalog_name}.{schema_name}.receipt_lines rl USING (receipt_id)
WHERE r.customer_id = get_recent_receipts.customer_id
  AND r.receipt_ts >= current_timestamp() - make_interval(0, 0, 0, get_recent_receipts.days, 0, 0, 0)
GROUP BY r.receipt_id, r.receipt_ts, r.channel, r.store_id,
         r.basket_total, r.item_count, r.on_promo, r.promo_savings
ORDER BY r.receipt_ts DESC
LIMIT 25;
