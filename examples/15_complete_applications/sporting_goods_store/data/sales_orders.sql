USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE sales_orders (
  order_id BIGINT COMMENT 'Unique identifier for each sales order'
  ,order_line_id BIGINT COMMENT 'Unique identifier for each line item within an order'
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table'
  ,sku STRING COMMENT 'Stock Keeping Unit of the product sold'
  ,store_id BIGINT COMMENT 'Store where the sale occurred'
  ,store_name STRING COMMENT 'Name of the store where the sale occurred'
  ,customer_id STRING COMMENT 'Customer identifier if available'
  ,order_date DATE COMMENT 'Date the order was placed'
  ,order_timestamp TIMESTAMP COMMENT 'Exact timestamp of the order'
  ,quantity INT COMMENT 'Quantity of items sold'
  ,unit_price DOUBLE COMMENT 'Price per unit at the time of sale'
  ,discount_amount DOUBLE COMMENT 'Discount applied to this line item'
  ,discount_type STRING COMMENT 'Type of discount (percentage, fixed, promotion, markdown, clearance)'
  ,promotion_id STRING COMMENT 'Promotion identifier if a promotion was applied'
  ,total_amount DOUBLE COMMENT 'Total amount for this line item after discount'
  ,cost_amount DOUBLE COMMENT 'Cost of goods sold for this line item'
  ,margin_amount DOUBLE COMMENT 'Gross margin for this line item'
  ,margin_pct DOUBLE COMMENT 'Gross margin percentage'
  ,payment_method STRING COMMENT 'Payment method used (credit, debit, cash, gift_card)'
  ,channel STRING COMMENT 'Sales channel (in_store, online, mobile_app)'
  ,department STRING COMMENT 'Department the product belongs to'
  ,sport_category STRING COMMENT 'Sport or activity category of the product'
  ,season STRING COMMENT 'Season during which the sale occurred'
  ,is_return BOOLEAN COMMENT 'Flag indicating if this is a return transaction'
  ,return_reason STRING COMMENT 'Reason for return if applicable'
  ,sales_associate_id STRING COMMENT 'ID of the sales associate who handled the transaction'
  ,created_at TIMESTAMP COMMENT 'Record creation timestamp'
)
USING DELTA
COMMENT 'Sales transaction table capturing all sporting goods sales across stores and channels with margin analysis and seasonal tracking.'
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact' = 'true'
);
