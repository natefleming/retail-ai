USE IDENTIFIER(:database);

CREATE OR REPLACE TABLE products (
  -- Core product identification
  product_id BIGINT COMMENT 'Unique identifier for each product in the catalog'
  ,sku STRING COMMENT 'Stock Keeping Unit - unique internal product identifier code'
  ,upc STRING COMMENT 'Universal Product Code - standardized barcode number for product identification'
  
  -- Product details
  ,brand_name STRING COMMENT 'Name of the manufacturer or brand that produces the product'
  ,product_name STRING COMMENT 'Display name of the product as shown to customers'
  ,short_description STRING COMMENT 'Brief product description for quick reference'
  ,long_description STRING COMMENT 'Detailed text description of the product including key features and attributes'
  ,product_url STRING COMMENT 'URL to the product page on the e-commerce site'
  ,image_url STRING COMMENT 'URL to the primary product image'
  
  -- Product classification
  ,merchandise_class STRING COMMENT 'Broad category classification (e.g., Footwear, Apparel, Equipment, Accessories)'
  ,class_cd STRING COMMENT 'Alphanumeric code representing the specific product subcategory'
  ,department_id STRING COMMENT 'Department identifier for store organization'
  ,department_name STRING COMMENT 'Name of the department where product is typically located'
  ,category_id STRING COMMENT 'Category identifier within department'
  ,category_name STRING COMMENT 'Name of the product category'
  ,subcategory_id STRING COMMENT 'Subcategory identifier within category'
  ,subcategory_name STRING COMMENT 'Name of the product subcategory'
  ,sport_category STRING COMMENT 'Primary sport or activity the product is designed for (e.g., Running, Basketball, Camping, Cycling)'
  
  -- Product attributes
  ,base_price DOUBLE COMMENT 'Standard retail price before any discounts'
  ,msrp DOUBLE COMMENT 'MSRP (Manufacturer Suggested Retail Price)'
  ,cost DOUBLE COMMENT 'Wholesale cost from supplier'
  ,weight DOUBLE COMMENT 'Product weight'
  ,weight_unit STRING COMMENT 'Unit of measurement for weight (e.g., kg, lb)'
  ,dimensions STRING COMMENT 'Product dimensions in JSON format (length, width, height)'
  ,color STRING COMMENT 'Primary color of the product'
  ,size STRING COMMENT 'Size specification if applicable'
  ,material STRING COMMENT 'Primary material(s) used in the product'
  ,gender STRING COMMENT 'Target gender (Mens, Womens, Unisex, Youth)'
  ,attributes STRING COMMENT 'Additional product attributes in JSON format'
  
  -- Inventory management
  ,min_order_quantity INT COMMENT 'Minimum quantity for ordering from supplier'
  ,max_order_quantity INT COMMENT 'Maximum quantity for ordering from supplier'
  ,reorder_point INT COMMENT 'Quantity threshold for triggering reorder'
  ,lead_time_days INT COMMENT 'Average lead time for restocking in days'
  ,safety_stock_level INT COMMENT 'Recommended safety stock quantity'
  ,economic_order_quantity INT COMMENT 'Optimal order quantity for cost efficiency'
  
  -- Supplier information
  ,primary_supplier_id STRING COMMENT 'ID of the primary supplier'
  ,primary_supplier_name STRING COMMENT 'Name of the primary supplier'
  ,supplier_part_number STRING COMMENT 'Supplier part number for the product'
  ,alternative_suppliers STRING COMMENT 'List of alternative suppliers in JSON format'
  
  -- Product status and lifecycle
  ,product_status STRING COMMENT 'Current product status (active, discontinued, pending, seasonal)'
  ,launch_date DATE COMMENT 'Date when product was first introduced'
  ,discontinue_date DATE COMMENT 'Date when product will be or was discontinued'
  ,is_seasonal BOOLEAN COMMENT 'Whether the product is seasonal'
  ,season STRING COMMENT 'Target season (Spring, Summer, Fall, Winter, Year-Round)'
  ,season_start DATE COMMENT 'Start date of the seasonal availability'
  ,season_end DATE COMMENT 'End date of the seasonal availability'
  ,is_returnable BOOLEAN COMMENT 'Whether the product can be returned'
  ,return_policy STRING COMMENT 'Specific return policy for the product'
  
  -- Marketing and merchandising
  ,is_featured BOOLEAN COMMENT 'Whether the product is featured in promotions'
  ,promotion_eligibility BOOLEAN COMMENT 'Whether the product can be promoted'
  ,tags STRING COMMENT 'Search and classification tags as JSON array'
  ,keywords STRING COMMENT 'Search keywords as JSON array'
  ,merchandising_priority INT COMMENT 'Priority for merchandising displays (1=highest, 5=lowest)'
  ,recommended_display_location STRING COMMENT 'Recommended location for store display'
  ,planogram_position STRING COMMENT 'Position in the planogram layout'
  
  -- Compliance and regulations
  ,hazmat_flag BOOLEAN COMMENT 'Whether product is considered hazardous material'
  ,age_restriction INT COMMENT 'Minimum age requirement for purchase if applicable'
  
  -- Metadata
  ,created_at TIMESTAMP COMMENT 'Record creation timestamp'
  ,updated_at TIMESTAMP COMMENT 'Record last update timestamp'
  ,created_by STRING COMMENT 'User who created the record'
  ,updated_by STRING COMMENT 'User who last updated the record'
)
USING DELTA
COMMENT 'Master product catalog for sporting goods store containing comprehensive product information including athletic footwear, apparel, equipment, and accessories with merchandising and seasonal attributes.'
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact' = 'true'
);
