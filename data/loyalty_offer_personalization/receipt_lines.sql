USE IDENTIFIER(:database);

-- Individual line items per receipt. Joins to receipts on receipt_id.
CREATE TABLE IF NOT EXISTS receipt_lines (
    line_id         STRING  COMMENT 'Unique line id',
    receipt_id      STRING  COMMENT 'Parent receipt id',
    sku             STRING  COMMENT 'SKU sold',
    brand           STRING  COMMENT 'Brand of the SKU',
    category        STRING  COMMENT 'Category of the SKU',
    qty             INT     COMMENT 'Units of this SKU on the line',
    line_price      DOUBLE  COMMENT 'Final line price after discount (USD)',
    line_discount   DOUBLE  COMMENT 'Discount applied to this line'
)
USING DELTA
COMMENT 'Receipt line items';
