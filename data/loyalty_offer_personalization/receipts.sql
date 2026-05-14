USE IDENTIFIER(:database);

-- Receipt header (one row per checkout).
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id      STRING  COMMENT 'Unique receipt id',
    customer_id     STRING  COMMENT 'Loyalty customer id (may be null for guest)',
    store_id        INT     COMMENT 'Store where the transaction occurred',
    channel         STRING  COMMENT 'STORE | ONLINE | APP',
    basket_total    DOUBLE  COMMENT 'Total basket value in USD',
    item_count      INT     COMMENT 'Number of distinct line items',
    units_count     INT     COMMENT 'Total units sold',
    on_promo        BOOLEAN COMMENT 'Whether any line was discounted',
    promo_savings   DOUBLE  COMMENT 'Total dollars saved by promos on this receipt',
    receipt_ts      TIMESTAMP COMMENT 'Timestamp of the transaction'
)
USING DELTA
COMMENT 'Receipt headers';
