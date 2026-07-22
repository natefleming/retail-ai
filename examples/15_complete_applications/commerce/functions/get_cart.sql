CREATE OR REPLACE FUNCTION {catalog_name}.{schema_name}.get_cart(
  customer_id STRING COMMENT 'Internal customer identifier (C#### for B2C, B#### for B2B), as returned by lookup_customer_by_user.'
)
RETURNS TABLE(
   cart_id STRING COMMENT 'Cart identifier'
  ,product_id BIGINT COMMENT 'Product identifier in this cart row'
  ,sku STRING COMMENT 'Product SKU'
  ,product_name STRING COMMENT 'Product name'
  ,quantity INT COMMENT 'Quantity in cart'
  ,unit_price DOUBLE COMMENT 'Unit price'
  ,line_total DOUBLE COMMENT 'quantity * unit_price'
  ,added_at TIMESTAMP COMMENT 'When the line was added'
)
READS SQL DATA
COMMENT 'Retrieve the current cart for a customer, joined to the product catalog so each line includes product name and unit price.'
RETURN
SELECT
   ct.cart_id
  ,ct.product_id
  ,ct.sku
  ,p.product_name
  ,ct.quantity
  ,p.price AS unit_price
  ,ct.quantity * p.price AS line_total
  ,ct.added_at
FROM {catalog_name}.{schema_name}.cart ct
JOIN {catalog_name}.{schema_name}.products p ON ct.product_id = p.product_id
WHERE ct.customer_id = get_cart.customer_id
ORDER BY ct.added_at DESC
;
