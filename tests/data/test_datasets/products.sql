-- Test table DDL for products
CREATE TABLE IF NOT EXISTS products (
    id INT,
    name STRING,
    price DECIMAL(10, 2),
    category STRING,
    description STRING,
    created_at TIMESTAMP
) USING DELTA;
