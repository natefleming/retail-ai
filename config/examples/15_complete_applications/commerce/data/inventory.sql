USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE inventory (
  product_id BIGINT COMMENT 'FK -> products.product_id' NOT NULL
  ,location_id STRING COMMENT 'Warehouse or store location identifier' NOT NULL
  ,location_name STRING COMMENT 'Human-readable distribution location name'
  ,on_hand_qty INT COMMENT 'Quantity available on hand'
  ,reserved_qty INT COMMENT 'Quantity reserved by open orders'
  ,reorder_threshold INT COMMENT 'Restock trigger threshold'
  ,last_counted_at TIMESTAMP COMMENT 'Most recent cycle-count timestamp'
  ,FOREIGN KEY (product_id) REFERENCES products(product_id)
)
CLUSTER BY AUTO
COMMENT 'Commerce Swarm inventory across distribution locations. Backs the stock_handler agent.'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
