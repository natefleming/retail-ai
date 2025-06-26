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
    '["medium roast", "everyday coffee", "balanced", "popular"]',
    '["starbucks", "pike place", "medium roast", "k-cup", "coffee pods"]',
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
    '["dark roast", "bold", "complex", "popular"]',
    '["peets", "major dickason", "dark roast", "k-cup", "coffee pods"]',
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
    '["medium roast", "classic", "smooth", "popular"]',
    '["dunkin", "original blend", "medium roast", "k-cup", "coffee pods"]',
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
    '["light roast", "breakfast blend", "smooth", "morning coffee"]',
    '["green mountain", "breakfast blend", "light roast", "k-cup", "coffee pods"]',
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
    '["medium roast", "classic", "diner style", "popular"]',
    '["donut shop", "regular", "medium roast", "k-cup", "coffee pods"]',
    2, 'Eye-level shelf',
    
    false,
    '{"food_grade": true, "fda_approved": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Adidas Gazelle Sneakers
    (6, 'ADI-GAZ-001', '567890123456',
    'Adidas', 'Gazelle Classic Sneakers',
    'Iconic Adidas Gazelle retro sneakers in various colors',
    'The Adidas Gazelle is a timeless classic that has been a staple in streetwear culture for decades. Originally designed as a training shoe, the Gazelle features a suede upper, leather 3-Stripes, and a rubber outsole. This versatile sneaker offers comfort and style for everyday wear.',
    'https://retail.ai/products/adi-gaz-001',
    'https://retail.ai/images/adi-gaz-001.jpg',
    
    'Footwear', 'SNK-CAS',
    'FOO-01', 'Footwear',
    'SNK-01', 'Sneakers',
    'CAS-01', 'Casual Sneakers',
    
    89.99, 100.00,
    1.2, 'lb',
    '{"length": 12.0, "width": 4.5, "height": 5.0, "unit": "inch"}',
    '{"material": "suede", "sole": "rubber", "closure": "lace-up", "sizes": ["7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12"], "colors": ["black", "navy", "grey", "burgundy"]}',
    
    6, 60, 12, 7, 18, 36,
    
    'SUP006', 'Adidas Distribution',
    'AD-GAZ-MULTI',
    '{"suppliers": ["Athletic Footwear Inc", "Sports Direct Supply"]}',
    
    'active', '2021-01-01', false, true,
    'Returnable within 30 days with original packaging',
    
    true, true,
    '["retro", "classic", "casual", "streetwear", "comfortable"]',
    '["adidas", "gazelle", "sneakers", "casual shoes", "retro"]',
    1, 'Featured display',
    
    false,
    '{"material_safety": true, "non_toxic": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Nike Air Force 1 Low
    (7, 'NIK-AF1-001', '678901234567',
    'Nike', 'Air Force 1 Low Sneakers',
    'Classic Nike Air Force 1 Low basketball-inspired sneakers',
    'The Nike Air Force 1 Low is a basketball legend that has transcended the court to become a street style icon. Featuring premium leather construction, perforated toe box for breathability, and the classic pivot point outsole for smooth transitions. A timeless design that goes with everything.',
    'https://retail.ai/products/nik-af1-001',
    'https://retail.ai/images/nik-af1-001.jpg',
    
    'Footwear', 'SNK-CAS',
    'FOO-01', 'Footwear',
    'SNK-01', 'Sneakers',
    'CAS-01', 'Casual Sneakers',
    
    109.99, 120.00,
    1.5, 'lb',
    '{"length": 12.5, "width": 4.8, "height": 5.2, "unit": "inch"}',
    '{"material": "leather", "sole": "rubber", "closure": "lace-up", "sizes": ["7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12", "13"], "colors": ["white", "black", "white/black"]}',
    
    6, 60, 15, 7, 20, 40,
    
    'SUP007', 'Nike Distribution',
    'NK-AF1-LOW',
    '{"suppliers": ["Athletic Footwear Inc", "Nike Direct Supply"]}',
    
    'active', '2021-02-01', false, true,
    'Returnable within 30 days with original packaging',
    
    true, true,
    '["basketball", "classic", "leather", "iconic", "versatile"]',
    '["nike", "air force 1", "sneakers", "basketball shoes", "classic"]',
    1, 'Featured display',
    
    false,
    '{"material_safety": true, "non_toxic": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Converse Chuck Taylor All Star
    (8, 'CON-CHK-001', '789012345678',
    'Converse', 'Chuck Taylor All Star High Top',
    'Classic Converse Chuck Taylor All Star high-top canvas sneakers',
    'The Converse Chuck Taylor All Star is the original basketball shoe and an American icon. Featuring a timeless silhouette, durable canvas upper, and the signature rubber toe cap and outsole. These high-top sneakers have been a symbol of self-expression and creativity for generations.',
    'https://retail.ai/products/con-chk-001',
    'https://retail.ai/images/con-chk-001.jpg',
    
    'Footwear', 'SNK-CAS',
    'FOO-01', 'Footwear',
    'SNK-01', 'Sneakers',
    'CAS-01', 'Casual Sneakers',
    
    64.99, 70.00,
    1.0, 'lb',
    '{"length": 12.0, "width": 4.2, "height": 6.0, "unit": "inch"}',
    '{"material": "canvas", "sole": "rubber", "closure": "lace-up", "sizes": ["6", "6.5", "7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12"], "colors": ["black", "white", "red", "navy", "optical white"]}',
    
    6, 72, 18, 5, 24, 48,
    
    'SUP008', 'Converse Distribution',
    'CV-CHK-HI',
    '{"suppliers": ["Athletic Footwear Inc", "Canvas Shoe Supply"]}',
    
    'active', '2021-03-01', false, true,
    'Returnable within 30 days with original packaging',
    
    true, true,
    '["classic", "canvas", "high-top", "iconic", "vintage"]',
    '["converse", "chuck taylor", "all star", "canvas sneakers", "high top"]',
    2, 'Eye-level shelf',
    
    false,
    '{"material_safety": true, "non_toxic": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Vans Old Skool
    (9, 'VAN-OLD-001', '890123456789',
    'Vans', 'Old Skool Skate Sneakers',
    'Classic Vans Old Skool skateboarding sneakers with side stripe',
    'The Vans Old Skool is the original skate shoe and an icon of street culture. Featuring sturdy canvas and suede uppers, the signature side stripe, and Vans'' waffle outsole for superior grip. Built for skateboarding but loved by everyone for its timeless style and durability.',
    'https://retail.ai/products/van-old-001',
    'https://retail.ai/images/van-old-001.jpg',
    
    'Footwear', 'SNK-SKT',
    'FOO-01', 'Footwear',
    'SNK-01', 'Sneakers',
    'SKT-01', 'Skate Sneakers',
    
    69.99, 75.00,
    1.1, 'lb',
    '{"length": 12.2, "width": 4.3, "height": 4.8, "unit": "inch"}',
    '{"material": "canvas/suede", "sole": "waffle rubber", "closure": "lace-up", "sizes": ["6.5", "7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12"], "colors": ["black/white", "navy", "burgundy", "grey"]}',
    
    6, 60, 15, 6, 20, 40,
    
    'SUP009', 'Vans Distribution',
    'VN-OLD-SKOOL',
    '{"suppliers": ["Skate Supply Co", "Street Footwear Inc"]}',
    
    'active', '2021-04-01', false, true,
    'Returnable within 30 days with original packaging',
    
    true, true,
    '["skate", "street", "durable", "classic", "side stripe"]',
    '["vans", "old skool", "skate shoes", "street wear", "skateboarding"]',
    2, 'Eye-level shelf',
    
    false,
    '{"material_safety": true, "non_toxic": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Puma Suede Classic
    (10, 'PUM-SUD-001', '901234567890',
    'Puma', 'Suede Classic Sneakers',
    'Iconic Puma Suede Classic retro basketball sneakers',
    'The Puma Suede Classic is a basketball legend that became a street style staple. Originally worn by basketball players in the 1960s, this shoe features a premium suede upper, classic Puma formstrip, and a rubber outsole. A timeless design that represents the intersection of sport and culture.',
    'https://retail.ai/products/pum-sud-001',
    'https://retail.ai/images/pum-sud-001.jpg',
    
    'Footwear', 'SNK-CAS',
    'FOO-01', 'Footwear',
    'SNK-01', 'Sneakers',
    'CAS-01', 'Casual Sneakers',
    
    79.99, 85.00,
    1.1, 'lb',
    '{"length": 12.0, "width": 4.4, "height": 4.9, "unit": "inch"}',
    '{"material": "suede", "sole": "rubber", "closure": "lace-up", "sizes": ["7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12"], "colors": ["black", "navy", "red", "grey", "green"]}',
    
    6, 60, 12, 7, 18, 36,
    
    'SUP010', 'Puma Distribution',
    'PM-SUD-CLS',
    '{"suppliers": ["Athletic Footwear Inc", "Puma Direct Supply"]}',
    
    'active', '2021-05-01', false, true,
    'Returnable within 30 days with original packaging',
    
    true, true,
    '["retro", "suede", "basketball", "classic", "heritage"]',
    '["puma", "suede", "classic", "retro sneakers", "basketball"]',
    2, 'Eye-level shelf',
    
    false,
    '{"material_safety": true, "non_toxic": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Adidas Samba Classic Sneakers
    (11, 'ADI-SMB-001', '012345678902',
    'Adidas', 'Samba Classic Sneakers',
    'Iconic Adidas Samba indoor soccer shoes with gum sole',
    'The Adidas Samba is a timeless classic originally designed for indoor soccer training. Featuring a premium leather upper, suede T-toe overlay, and the signature gum rubber outsole. This versatile sneaker has transcended sports to become a street style icon, perfect for casual wear.',
    'https://retail.ai/products/adi-smb-001',
    'https://retail.ai/images/adi-smb-001.jpg',
    
    'Footwear', 'SNK-CAS',
    'FOO-01', 'Footwear',
    'SNK-01', 'Sneakers',
    'CAS-01', 'Casual Sneakers',
    
    94.99, 105.00,
    1.3, 'lb',
    '{"length": 12.0, "width": 4.5, "height": 4.8, "unit": "inch"}',
    '{"material": "leather/suede", "sole": "gum rubber", "closure": "lace-up", "sizes": ["7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12"], "colors": ["black/white", "white/green", "navy/white", "burgundy/white"]}',
    
    6, 60, 12, 7, 18, 36,
    
    'SUP006', 'Adidas Distribution',
    'AD-SMB-CLS',
    '{"suppliers": ["Athletic Footwear Inc", "Sports Direct Supply"]}',
    
    'active', '2021-06-01', false, true,
    'Returnable within 30 days with original packaging',
    
    true, true,
    '["soccer", "indoor", "gum sole", "leather", "classic"]',
    '["adidas", "samba", "soccer shoes", "indoor training", "gum sole"]',
    1, 'Featured display',
    
    false,
    '{"material_safety": true, "non_toxic": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Adidas Stan Smith Sneakers
    (12, 'ADI-STS-001', '123456789013',
    'Adidas', 'Stan Smith Classic Sneakers',
    'Iconic Adidas Stan Smith tennis shoes in clean white leather',
    'The Adidas Stan Smith is one of the most recognizable tennis shoes in the world. Originally created for tennis legend Stan Smith, this minimalist design features premium white leather construction with green accents. A timeless, versatile sneaker that works with any outfit.',
    'https://retail.ai/products/adi-sts-001',
    'https://retail.ai/images/adi-sts-001.jpg',
    
    'Footwear', 'SNK-CAS',
    'FOO-01', 'Footwear',
    'SNK-01', 'Sneakers',
    'CAS-01', 'Casual Sneakers',
    
    84.99, 90.00,
    1.2, 'lb',
    '{"length": 12.0, "width": 4.4, "height": 4.6, "unit": "inch"}',
    '{"material": "leather", "sole": "rubber", "closure": "lace-up", "sizes": ["7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12"], "colors": ["white/green", "white/navy", "white/black", "all white"]}',
    
    6, 60, 15, 7, 20, 40,
    
    'SUP006', 'Adidas Distribution',
    'AD-STS-CLS',
    '{"suppliers": ["Athletic Footwear Inc", "Sports Direct Supply"]}',
    
    'active', '2021-07-01', false, true,
    'Returnable within 30 days with original packaging',
    
    true, true,
    '["tennis", "minimalist", "white leather", "clean", "versatile"]',
    '["adidas", "stan smith", "tennis shoes", "white sneakers", "classic"]',
    1, 'Featured display',
    
    false,
    '{"material_safety": true, "non_toxic": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Adidas Superstar Sneakers
    (13, 'ADI-SUP-001', '234567890124',
    'Adidas', 'Superstar Classic Sneakers',
    'Iconic Adidas Superstar with signature shell toe design',
    'The Adidas Superstar is a basketball legend that became a cultural icon. Originally designed for the court in 1969, this shoe features the distinctive rubber shell toe, premium leather upper, and classic 3-Stripes. A symbol of street culture and self-expression.',
    'https://retail.ai/products/adi-sup-001',
    'https://retail.ai/images/adi-sup-001.jpg',
    
    'Footwear', 'SNK-CAS',
    'FOO-01', 'Footwear',
    'SNK-01', 'Sneakers',
    'CAS-01', 'Casual Sneakers',
    
    99.99, 110.00,
    1.4, 'lb',
    '{"length": 12.2, "width": 4.6, "height": 5.0, "unit": "inch"}',
    '{"material": "leather", "sole": "rubber", "closure": "lace-up", "sizes": ["7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12"], "colors": ["white/black", "black/white", "white/gold", "all black"]}',
    
    6, 60, 12, 7, 18, 36,
    
    'SUP006', 'Adidas Distribution',
    'AD-SUP-CLS',
    '{"suppliers": ["Athletic Footwear Inc", "Sports Direct Supply"]}',
    
    'active', '2021-08-01', false, true,
    'Returnable within 30 days with original packaging',
    
    true, true,
    '["basketball", "shell toe", "street culture", "iconic", "leather"]',
    '["adidas", "superstar", "shell toe", "basketball shoes", "street"]',
    1, 'Featured display',
    
    false,
    '{"material_safety": true, "non_toxic": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'),

    -- Adidas Campus Sneakers
    (14, 'ADI-CAM-001', '345678901235',
    'Adidas', 'Campus Classic Sneakers',
    'Retro Adidas Campus basketball shoes with suede upper',
    'The Adidas Campus is a vintage basketball shoe that has become a streetwear staple. Originally designed in the 1980s, this shoe features a premium suede upper, classic 3-Stripes, and a rubber cupsole. A perfect blend of retro style and modern comfort.',
    'https://retail.ai/products/adi-cam-001',
    'https://retail.ai/images/adi-cam-001.jpg',
    
    'Footwear', 'SNK-CAS',
    'FOO-01', 'Footwear',
    'SNK-01', 'Sneakers',
    'CAS-01', 'Casual Sneakers',
    
    87.99, 95.00,
    1.2, 'lb',
    '{"length": 12.0, "width": 4.5, "height": 4.9, "unit": "inch"}',
    '{"material": "suede", "sole": "rubber", "closure": "lace-up", "sizes": ["7", "7.5", "8", "8.5", "9", "9.5", "10", "10.5", "11", "11.5", "12"], "colors": ["grey/white", "navy/white", "burgundy/white", "black/white"]}',
    
    6, 60, 12, 7, 18, 36,
    
    'SUP006', 'Adidas Distribution',
    'AD-CAM-CLS',
    '{"suppliers": ["Athletic Footwear Inc", "Sports Direct Supply"]}',
    
    'active', '2021-09-01', false, true,
    'Returnable within 30 days with original packaging',
    
    true, true,
    '["retro", "basketball", "suede", "vintage", "streetwear"]',
    '["adidas", "campus", "retro sneakers", "suede shoes", "basketball"]',
    2, 'Eye-level shelf',
    
    false,
    '{"material_safety": true, "non_toxic": true}',
    0,
    
    '2023-01-01 00:00:00', CURRENT_TIMESTAMP(),
    'system', 'system'); 