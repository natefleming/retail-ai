USE IDENTIFIER(:database);

CREATE TABLE fulfil_item_orders (
  uuid STRING,
  coffee_name STRING,
  size STRING,
  order_timestamp TIMESTAMP DEFAULT current_timestamp(),
  session_id STRING)
USING delta
TBLPROPERTIES (
  delta.enableChangeDataFeed = true
);
