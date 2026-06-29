USE IDENTIFIER(:database);

CREATE TABLE IF NOT EXISTS order_items (
  order_item_id STRING COMMENT 'Unique order line-item identifier' NOT NULL PRIMARY KEY
  ,order_id STRING COMMENT 'FK -> orders.order_id' NOT NULL
  ,product_id BIGINT COMMENT 'FK -> products.product_id' NOT NULL
  ,sku STRING COMMENT 'Denormalized SKU for convenience'
  ,quantity INT COMMENT 'Quantity ordered'
  ,unit_price DOUBLE COMMENT 'Per-unit price at time of order'
  ,line_total DOUBLE COMMENT 'quantity * unit_price'
  ,FOREIGN KEY (order_id) REFERENCES orders(order_id)
  ,FOREIGN KEY (product_id) REFERENCES products(product_id)
)
CLUSTER BY AUTO
COMMENT 'Commerce Swarm order line items. Joined to orders for full order detail responses.'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
