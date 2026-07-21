USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE items_raw (
    item_id STRING NOT NULL,
    sku STRING NOT NULL,
    item_name STRING NOT NULL,
    item_cat STRING NOT NULL,
    item_size STRING,
    item_price DECIMAL(10,2) NOT NULL
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
COMMENT 'Coffee shop menu items with pricing and categorization';