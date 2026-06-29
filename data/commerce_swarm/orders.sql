USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE orders (
  order_id STRING COMMENT 'Unique order identifier' NOT NULL PRIMARY KEY
  ,customer_id STRING COMMENT 'FK -> customers.customer_id' NOT NULL
  ,status STRING COMMENT 'Order lifecycle status (placed, confirmed, shipped, delivered, cancelled, returned)'
  ,total_amount DOUBLE COMMENT 'Total order amount in USD including all line items'
  ,currency STRING COMMENT 'ISO-4217 currency code'
  ,channel STRING COMMENT 'Sales channel (web, mobile, store, b2b_portal, phone, edi)'
  ,placed_at TIMESTAMP COMMENT 'Order placement timestamp'
  ,shipped_at TIMESTAMP COMMENT 'Shipment timestamp; NULL if not yet shipped'
  ,delivered_at TIMESTAMP COMMENT 'Delivery timestamp; NULL if not yet delivered'
  ,tracking_number STRING COMMENT 'Carrier tracking number; NULL until shipped'
  ,FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
)
CLUSTER BY AUTO
COMMENT 'Commerce Swarm orders. Backs the order_history_handler agent.'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
