USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE orders (
    row_id BIGINT NOT NULL,
    order_id STRING NOT NULL,
    created_at TIMESTAMP NOT NULL,
    item_id STRING NOT NULL,
    quantity INT NOT NULL,
    cust_name STRING NOT NULL,
    in_or_out STRING
) 
USING delta
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.enableDeletionVectors' = 'true',
    'delta.feature.changeDataFeed' = 'supported',
    'delta.feature.deletionVectors' = 'supported',
    'delta.minReaderVersion' = '3',
    'delta.minWriterVersion' = '7'
)
COMMENT 'Coffee shop order line items with customer and delivery preferences';