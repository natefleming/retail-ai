CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.get_order_history(
   customer_id STRING COMMENT 'Customer identifier (C#### for B2C, B#### for B2B)'
  ,row_limit INT DEFAULT 10 COMMENT 'Maximum number of orders to return; defaults to 10'
)
RETURNS TABLE(
   order_id STRING COMMENT 'Unique order identifier'
  ,status STRING COMMENT 'Current order status'
  ,total_amount DOUBLE COMMENT 'Order total in USD'
  ,channel STRING COMMENT 'Sales channel'
  ,placed_at TIMESTAMP COMMENT 'Order placement timestamp'
  ,shipped_at TIMESTAMP COMMENT 'Shipment timestamp; NULL if not yet shipped'
  ,delivered_at TIMESTAMP COMMENT 'Delivery timestamp; NULL if not yet delivered'
  ,tracking_number STRING COMMENT 'Carrier tracking number'
)
READS SQL DATA
COMMENT 'Retrieve recent order history for a customer, sorted by most-recently placed.'
RETURN
-- Implementation note: uses ROW_NUMBER() because Spark SQL LIMIT requires a
-- foldable constant and does not accept parameterized expressions.
SELECT
   order_id
  ,status
  ,total_amount
  ,channel
  ,placed_at
  ,shipped_at
  ,delivered_at
  ,tracking_number
FROM (
  SELECT
     order_id
    ,status
    ,total_amount
    ,channel
    ,placed_at
    ,shipped_at
    ,delivered_at
    ,tracking_number
    ,ROW_NUMBER() OVER (ORDER BY placed_at DESC) AS rn
  FROM {catalog_name}.{schema_name}.orders
  WHERE customer_id = get_order_history.customer_id
)
WHERE rn <= get_order_history.row_limit
;
