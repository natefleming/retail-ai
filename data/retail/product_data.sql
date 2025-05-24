USE IDENTIFIER(:database);
-- Insert sample product records for coffee pods
INSERT INTO products (
    -- Core identification
    product_id,
    sku,
    upc,
    
    -- Product details
    brand_name,
    product_name,
    short_description,
    long_description,
    product_url,
    image_url,
    
    -- Classification
    merchandise_class,
    class_cd,
    department_id,
    department_name,
    category_id,
    category_name,
    subcategory_id,
    subcategory_name,
    
    -- Product attributes
    base_price,
    msrp,
    weight,
    weight_unit,
    dimensions,
    attributes,
    
    -- Inventory management
    min_order_quantity,
    max_order_quantity,
    reorder_point,
    lead_time_days,
    safety_stock_level,
    economic_order_quantity,
    
    -- Supplier information
    primary_supplier_id,
    primary_supplier_name,
    supplier_part_number,
    alternative_suppliers,
    
    -- Product status
    product_status,
    launch_date,
    is_seasonal,
    is_returnable,
    return_policy,
    
    -- Marketing
    is_featured,
    promotion_eligibility,
    tags,
    keywords,
    merchandising_priority,
    recommended_display_location,
    
    -- Compliance
    hazmat_flag,
    regulatory_flags,
    age_restriction,
    
    -- Metadata
    created_at,
    updated_at,
    created_by,
    updated_by
) VALUES
    -- Starbucks Pike Place Medium Roast K-Cups
    (1, 'STB-KCP-001', '012345678901',
    'Starbucks', 'Pike Place Medium Roast K-Cup Pods',
    'Starbucks signature medium roast coffee in K-Cup pods',
    'A smooth, well-rounded blend of Latin American coffees with subtly rich flavors of chocolate and toasted nuts. Perfect for every day drinking, Pike Place Roast is served fresh in our stores every day. Each K-Cup pod contains perfect portions of ground coffee and is compatible with all Keurig K-Cup brewers.',
    'https://retail.ai/products/stb-kcp-001',
    'https://retail.ai/images/stb-kcp-001.jpg',
    
    'Beverages', 'COF-KCP',
    'BEV-01', 'Beverages',
    'COF-01', 'Coffee',
    'KCP-01', 'K-Cup Pods',
    
    18.99, 19.99,
    0.42, 'lb',
    '{"length": 7.5, "width": 4.0, "height": 4.0, "unit": "inch"}',
    '{"roast_level": "medium", "count": 24, "caffeinated": true, "kosher": true, "recyclable": true}',
    
    20, 200, 30, 3, 40, 100,
    
    'SUP001', 'Starbucks Distribution',
    'SB-PKP-24CT',
    '{"suppliers": ["Coffee Distributors Inc", "National Beverage Supply"]}',
    
    'active', '2020-01-01', false, true,
    'Returnable within 30 days if unopened',
    
    true, true,
    ARRAY('medium roast', 'everyday coffee', 'balanced', 'popular'),
    ARRAY('starbucks', 'pike place', 'medium roast', 'k-cup', 'coffee pods'),
    1, 'Eye-level shelf',
    
    false,
    '{"food_grade": true, "fda_approved": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Peet's Coffee Major Dickason's Blend K-Cups
    (2, 'PET-KCP-001', '123456789012',
    'Peet''s Coffee', 'Major Dickason''s Blend K-Cup Pods',
    'Peet''s signature dark roast blend in K-Cup pods',
    'Incomparable world blend, rich, complex, and full-bodied. Major Dickason''s Blend is the finest example of the signature Peet''s style. This coffee is expertly roasted to bring out the full flavor and complexity, creating a rich, satisfying cup with a full body and layered flavors.',
    'https://retail.ai/products/pet-kcp-001',
    'https://retail.ai/images/pet-kcp-001.jpg',
    
    'Beverages', 'COF-KCP',
    'BEV-01', 'Beverages',
    'COF-01', 'Coffee',
    'KCP-01', 'K-Cup Pods',
    
    17.99, 18.99,
    0.42, 'lb',
    '{"length": 7.5, "width": 4.0, "height": 4.0, "unit": "inch"}',
    '{"roast_level": "dark", "count": 24, "caffeinated": true, "kosher": true, "recyclable": true}',
    
    20, 200, 25, 3, 35, 90,
    
    'SUP002', 'Peet''s Coffee Distribution',
    'PT-MDB-24CT',
    '{"suppliers": ["Coffee Distributors Inc", "West Coast Coffee Supply"]}',
    
    'active', '2020-02-01', false, true,
    'Returnable within 30 days if unopened',
    
    true, true,
    ARRAY('dark roast', 'bold', 'complex', 'popular'),
    ARRAY('peets', 'major dickason', 'dark roast', 'k-cup', 'coffee pods'),
    1, 'Eye-level shelf',
    
    false,
    '{"food_grade": true, "fda_approved": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Dunkin' Original Blend Medium Roast K-Cups
    (3, 'DUN-KCP-001', '234567890123',
    'Dunkin''', 'Original Blend Medium Roast K-Cup Pods',
    'Dunkin'' signature medium roast coffee in K-Cup pods',
    'The coffee that made Dunkin'' famous. Smooth, flavorful medium roast coffee in convenient K-Cup pods. Original Blend delivers the same great taste of Dunkin'' Original Blend Coffee served in Dunkin'' stores. Each K-Cup pod is filled with the finest quality Arabica coffee and crafted to deliver consistent flavor from cup to cup.',
    'https://retail.ai/products/dun-kcp-001',
    'https://retail.ai/images/dun-kcp-001.jpg',
    
    'Beverages', 'COF-KCP',
    'BEV-01', 'Beverages',
    'COF-01', 'Coffee',
    'KCP-01', 'K-Cup Pods',
    
    16.99, 17.99,
    0.42, 'lb',
    '{"length": 7.5, "width": 4.0, "height": 4.0, "unit": "inch"}',
    '{"roast_level": "medium", "count": 24, "caffeinated": true, "kosher": true, "recyclable": true}',
    
    20, 200, 25, 3, 35, 90,
    
    'SUP003', 'Dunkin'' Distribution',
    'DN-OGB-24CT',
    '{"suppliers": ["Coffee Distributors Inc", "East Coast Coffee Supply"]}',
    
    'active', '2020-03-01', false, true,
    'Returnable within 30 days if unopened',
    
    true, true,
    ARRAY('medium roast', 'classic', 'smooth', 'popular'),
    ARRAY('dunkin', 'original blend', 'medium roast', 'k-cup', 'coffee pods'),
    1, 'Eye-level shelf',
    
    false,
    '{"food_grade": true, "fda_approved": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Green Mountain Coffee Breakfast Blend K-Cups
    (4, 'GMC-KCP-001', '345678901234',
    'Green Mountain Coffee', 'Breakfast Blend Light Roast K-Cup Pods',
    'Green Mountain''s popular light roast breakfast blend in K-Cup pods',
    'Wake up to a mild, smooth and balanced cup of coffee. Light roasted to bring out the bright, crisp flavors while maintaining a smooth, clean finish. Perfect morning coffee that''s never bitter. Each K-Cup pod is made with 100% Arabica coffee and contains no artificial ingredients.',
    'https://retail.ai/products/gmc-kcp-001',
    'https://retail.ai/images/gmc-kcp-001.jpg',
    
    'Beverages', 'COF-KCP',
    'BEV-01', 'Beverages',
    'COF-01', 'Coffee',
    'KCP-01', 'K-Cup Pods',
    
    15.99, 16.99,
    0.42, 'lb',
    '{"length": 7.5, "width": 4.0, "height": 4.0, "unit": "inch"}',
    '{"roast_level": "light", "count": 24, "caffeinated": true, "kosher": true, "recyclable": true}',
    
    20, 200, 20, 3, 30, 80,
    
    'SUP004', 'Green Mountain Distribution',
    'GM-BBL-24CT',
    '{"suppliers": ["Coffee Distributors Inc", "Mountain Coffee Supply"]}',
    
    'active', '2020-04-01', false, true,
    'Returnable within 30 days if unopened',
    
    true, true,
    ARRAY('light roast', 'breakfast blend', 'smooth', 'morning coffee'),
    ARRAY('green mountain', 'breakfast blend', 'light roast', 'k-cup', 'coffee pods'),
    2, 'Eye-level shelf',
    
    false,
    '{"food_grade": true, "fda_approved": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- The Original Donut Shop Regular Medium Roast K-Cups
    (5, 'ODS-KCP-001', '456789012345',
    'The Original Donut Shop', 'Regular Medium Roast K-Cup Pods',
    'The Original Donut Shop''s classic medium roast coffee in K-Cup pods',
    'Extra bold, medium roasted coffee that brings back memories of simpler days. This coffee has a classic, sweet flavor that''s reminiscent of your favorite diner''s coffee. Made from 100% Arabica coffee beans and specially crafted for your Keurig brewer.',
    'https://retail.ai/products/ods-kcp-001',
    'https://retail.ai/images/ods-kcp-001.jpg',
    
    'Beverages', 'COF-KCP',
    'BEV-01', 'Beverages',
    'COF-01', 'Coffee',
    'KCP-01', 'K-Cup Pods',
    
    16.99, 17.99,
    0.42, 'lb',
    '{"length": 7.5, "width": 4.0, "height": 4.0, "unit": "inch"}',
    '{"roast_level": "medium", "count": 24, "caffeinated": true, "kosher": true, "recyclable": true}',
    
    20, 200, 20, 3, 30, 80,
    
    'SUP005', 'Keurig Dr Pepper Distribution',
    'DS-RMR-24CT',
    '{"suppliers": ["Coffee Distributors Inc", "National Beverage Supply"]}',
    
    'active', '2020-05-01', false, true,
    'Returnable within 30 days if unopened',
    
    true, true,
    ARRAY('medium roast', 'classic', 'diner style', 'popular'),
    ARRAY('donut shop', 'regular', 'medium roast', 'k-cup', 'coffee pods'),
    2, 'Eye-level shelf',
    
    false,
    '{"food_grade": true, "fda_approved": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'); 