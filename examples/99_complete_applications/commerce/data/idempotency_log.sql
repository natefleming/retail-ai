USE IDENTIFIER(:database);

CREATE TABLE IF NOT EXISTS idempotency_log (
  idempotency_key STRING COMMENT 'Hash of (intent + payload + action) — primary key for short-circuit lookup' NOT NULL PRIMARY KEY
  ,intent STRING COMMENT 'Intent name (add_to_cart, place_order, refund, etc.)'
  ,action STRING COMMENT 'Concrete action executed via MCP (commercetools.add_to_cart, stripe.charge, etc.)'
  ,payload STRING COMMENT 'JSON-encoded payload that produced the key'
  ,result STRING COMMENT 'JSON-encoded result returned by the underlying service'
  ,status STRING COMMENT 'Execution status (success, failed, in_progress)'
  ,executed_at TIMESTAMP COMMENT 'When the command was first executed'
)
CLUSTER BY AUTO
COMMENT 'UCP idempotency audit log. Lookups short-circuit duplicate commerce/payment commands so repeated calls return the same result.'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
