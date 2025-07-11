-- Test table DDL for inventory
CREATE TABLE IF NOT EXISTS inventory (
    product_id INT,
    quantity INT,
    location STRING,
    last_updated TIMESTAMP
) USING DELTA;
