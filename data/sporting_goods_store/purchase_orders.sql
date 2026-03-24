USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE purchase_orders (
  po_id BIGINT COMMENT 'Unique identifier for each purchase order'
  ,po_number STRING COMMENT 'Human-readable purchase order number'
  ,po_line_id BIGINT COMMENT 'Unique identifier for each line item within a purchase order'
  ,product_id BIGINT COMMENT 'Foreign key reference to the product table'
  ,sku STRING COMMENT 'Stock Keeping Unit of the product ordered'
  ,supplier_id STRING COMMENT 'Identifier of the supplier'
  ,supplier_name STRING COMMENT 'Name of the supplier'
  ,destination_store_id BIGINT COMMENT 'Store the inventory is destined for'
  ,destination_warehouse STRING COMMENT 'Warehouse the inventory is destined for'
  ,order_date DATE COMMENT 'Date the purchase order was created'
  ,expected_delivery_date DATE COMMENT 'Expected delivery date from supplier'
  ,actual_delivery_date DATE COMMENT 'Actual delivery date if delivered'
  ,quantity_ordered INT COMMENT 'Quantity of items ordered'
  ,quantity_received INT COMMENT 'Quantity of items actually received'
  ,unit_cost DOUBLE COMMENT 'Cost per unit from supplier'
  ,total_cost DOUBLE COMMENT 'Total cost for this line item'
  ,po_status STRING COMMENT 'Current status of the PO (draft, submitted, confirmed, in_transit, received, cancelled)'
  ,season STRING COMMENT 'Season this PO is buying for (Spring, Summer, Fall, Winter)'
  ,buy_plan_id STRING COMMENT 'Reference to the buy plan this PO belongs to'
  ,department STRING COMMENT 'Department the product belongs to'
  ,sport_category STRING COMMENT 'Sport or activity category of the product'
  ,lead_time_days INT COMMENT 'Lead time in days from order to delivery'
  ,is_reorder BOOLEAN COMMENT 'Whether this is a reorder or initial buy'
  ,buyer_id STRING COMMENT 'ID of the buyer who created the PO'
  ,buyer_name STRING COMMENT 'Name of the buyer'
  ,approval_status STRING COMMENT 'Approval status (pending, approved, rejected)'
  ,approved_by STRING COMMENT 'Name of the approver'
  ,notes STRING COMMENT 'Additional notes on the purchase order'
  ,created_at TIMESTAMP COMMENT 'Record creation timestamp'
  ,updated_at TIMESTAMP COMMENT 'Record last update timestamp'
)
USING DELTA
COMMENT 'Purchase order tracking table for managing vendor orders, buy plans, and receiving across the sporting goods merchandising lifecycle.'
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact' = 'true'
);
