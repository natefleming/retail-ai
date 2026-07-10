USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE cart (
  cart_id STRING COMMENT 'Cart identifier (one cart can contain multiple rows — one per product)' NOT NULL
  ,customer_id STRING COMMENT 'FK -> customers.customer_id' NOT NULL
  ,product_id BIGINT COMMENT 'FK -> products.product_id' NOT NULL
  ,sku STRING COMMENT 'Denormalized SKU'
  ,quantity INT COMMENT 'Quantity of this product in the cart'
  ,added_at TIMESTAMP COMMENT 'When this product was added to the cart'
  ,FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
  ,FOREIGN KEY (product_id) REFERENCES products(product_id)
)
CLUSTER BY AUTO
COMMENT 'Commerce Swarm active carts. Target table for UCP add_to_cart / update_cart commands.'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
;
