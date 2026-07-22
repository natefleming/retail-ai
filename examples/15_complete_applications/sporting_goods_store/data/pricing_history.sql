USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE pricing_history (
  pricing_id BIGINT COMMENT 'Unique identifier for each pricing record'
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table'
  ,sku STRING COMMENT 'Stock Keeping Unit of the product'
  ,store_id BIGINT COMMENT 'Store where this price applies (NULL for chain-wide pricing)'
  ,effective_date DATE COMMENT 'Date this price becomes effective'
  ,end_date DATE COMMENT 'Date this price expires (NULL if current)'
  ,original_price DOUBLE COMMENT 'Original retail price before any changes'
  ,new_price DOUBLE COMMENT 'New price after the change'
  ,cost DOUBLE COMMENT 'Current cost of goods'
  ,margin_pct DOUBLE COMMENT 'Margin percentage at the new price'
  ,price_change_type STRING COMMENT 'Type of price change (initial, markdown, clearance, promotion, seasonal, competitive, rollback)'
  ,price_change_reason STRING COMMENT 'Reason for the price change'
  ,promotion_id STRING COMMENT 'Promotion identifier if tied to a promotion'
  ,promotion_name STRING COMMENT 'Name of the promotion if applicable'
  ,markdown_pct DOUBLE COMMENT 'Percentage markdown from original price'
  ,department STRING COMMENT 'Department the product belongs to'
  ,sport_category STRING COMMENT 'Sport or activity category'
  ,season STRING COMMENT 'Season this price change is associated with'
  ,is_active BOOLEAN COMMENT 'Whether this is the currently active price'
  ,approved_by STRING COMMENT 'Name of the person who approved the price change'
  ,created_at TIMESTAMP COMMENT 'Record creation timestamp'
  ,created_by STRING COMMENT 'User who created the pricing record'
)
USING DELTA
COMMENT 'Pricing history table tracking all price changes including markdowns, promotions, seasonal adjustments, and competitive pricing moves for sporting goods merchandising.'
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact' = 'true'
);
