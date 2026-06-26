CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.check_stock(
   sku STRING COMMENT 'Product SKU to look up across all distribution locations'
)
RETURNS TABLE(
   sku STRING COMMENT 'Product SKU'
  ,product_name STRING COMMENT 'Customer-facing product name'
  ,location_id STRING COMMENT 'Distribution location identifier'
  ,location_name STRING COMMENT 'Distribution location name'
  ,on_hand_qty INT COMMENT 'Quantity on hand'
  ,reserved_qty INT COMMENT 'Quantity reserved by open orders'
  ,available_qty INT COMMENT 'Quantity available to promise (on_hand - reserved)'
  ,reorder_threshold INT COMMENT 'Restock trigger threshold'
  ,last_counted_at TIMESTAMP COMMENT 'Most recent cycle-count timestamp'
)
READS SQL DATA
COMMENT 'Check inventory levels for a product across all distribution locations. Returns available-to-promise quantity per location.'
RETURN
SELECT
   p.sku
  ,p.product_name
  ,i.location_id
  ,i.location_name
  ,i.on_hand_qty
  ,i.reserved_qty
  ,i.on_hand_qty - i.reserved_qty AS available_qty
  ,i.reorder_threshold
  ,i.last_counted_at
FROM {catalog_name}.{schema_name}.products p
JOIN {catalog_name}.{schema_name}.inventory i ON p.product_id = i.product_id
WHERE p.sku = check_stock.sku
ORDER BY i.location_id
;
