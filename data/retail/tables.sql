-- Store inventory table that combines product and location information
CREATE TABLE IF NOT EXISTS store_inventory (
    inventory_id BIGINT GENERATED ALWAYS AS IDENTITY,
    sku STRING NOT NULL,
    product_name STRING NOT NULL,
    product_description STRING NOT NULL,
    product_category STRING NOT NULL,
    -- Store information
    store_id BIGINT NOT NULL,
    store_name STRING NOT NULL,
    store_address STRING NOT NULL,
    store_city STRING NOT NULL,
    store_state STRING NOT NULL,
    store_zip_code STRING NOT NULL,
    store_phone STRING NOT NULL,
    store_email STRING NOT NULL,
    -- Location coordinates
    latitude DECIMAL(10,8) NOT NULL,
    longitude DECIMAL(11,8) NOT NULL,
    -- Store details
    store_type STRING NOT NULL, -- 'flagship', 'outlet', 'express', 'popup'
    store_size_sqft INT NOT NULL,
    store_rating DECIMAL(2,1) NOT NULL,
    store_hours JSON NOT NULL, -- JSON object with opening hours for each day
    location_type STRING NOT NULL, -- 'sales_floor', 'backroom', 'warehouse'
    quantity INT NOT NULL,
    min_stock_level INT NOT NULL,
    max_stock_level INT NOT NULL,
    last_restock_date TIMESTAMP,
    last_count_date TIMESTAMP,
    is_out_of_stock BOOLEAN NOT NULL,
    is_low_stock BOOLEAN NOT NULL,
    next_restock_date TIMESTAMP,
    -- Demand prediction fields
    daily_demand_prediction INT NOT NULL,
    weekly_demand_prediction INT NOT NULL,
    monthly_demand_prediction INT NOT NULL,
    -- Historical sales data
    last_7_days_sales INT NOT NULL,
    last_30_days_sales INT NOT NULL,
    last_90_days_sales INT NOT NULL,
    -- Stockout risk metrics
    days_until_stockout INT NOT NULL,
    stockout_risk_level STRING NOT NULL, -- 'low', 'medium', 'high', 'critical'
    -- Seasonality and trend data
    is_seasonal BOOLEAN NOT NULL,
    season_peak_factor DECIMAL(5,2) NOT NULL,
    trend_direction STRING NOT NULL, -- 'increasing', 'stable', 'decreasing'
    trend_strength DECIMAL(5,2) NOT NULL,
    -- Last prediction update
    last_prediction_update TIMESTAMP NOT NULL,
    -- Store status
    is_store_active BOOLEAN NOT NULL,
    store_created_at TIMESTAMP NOT NULL,
    store_last_updated TIMESTAMP NOT NULL
);

-- Insert sample coffee pod data
INSERT INTO store_inventory (
    sku,
    product_name,
    product_description,
    product_category,
    store_id,
    store_name,
    store_address,
    store_city,
    store_state,
    store_zip_code,
    store_phone,
    store_email,
    latitude,
    longitude,
    store_type,
    store_size_sqft,
    store_rating,
    store_hours,
    location_type,
    quantity,
    min_stock_level,
    max_stock_level,
    last_restock_date,
    last_count_date,
    is_out_of_stock,
    is_low_stock,
    next_restock_date,
    daily_demand_prediction,
    weekly_demand_prediction,
    monthly_demand_prediction,
    last_7_days_sales,
    last_30_days_sales,
    last_90_days_sales,
    days_until_stockout,
    stockout_risk_level,
    is_seasonal,
    season_peak_factor,
    trend_direction,
    trend_strength,
    last_prediction_update,
    is_store_active,
    store_created_at,
    store_last_updated
) VALUES
    -- Out of stock item (Popular Nespresso - Intensity 9)
    ('NES-ORIG-001', 'Nespresso Original - Arpeggio',
    'A bold and intense coffee with a rich, full-bodied flavor. Features notes of cocoa and roasted cereal, with a smooth, velvety texture. Perfect for those who enjoy a strong, traditional espresso with a lingering finish.',
    'Coffee Pods',
    1, 'Downtown Store',
    '123 Main Street',
    'San Francisco',
    'CA',
    '94105',
    '415-555-0101',
    'downtown@store.com',
    37.7749, -122.4194,
    'flagship',
    5000,
    4.5,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    0, 30, 150, '2024-03-01 10:00:00', '2024-03-15 14:30:00',
    true, false, DATEADD(day, 5, CURRENT_TIMESTAMP()),
    12, 84, 360, 90, 360, 1080,
    0, 'critical', false, 1.0, 'stable', 0.1,
    CURRENT_TIMESTAMP(),
    true,
    '2023-01-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- New out of stock item with descriptive blend name
    ('NES-ORIG-008', 'Nespresso Original - Caramel Creme Brulee',
    'A sweet and indulgent coffee that combines the rich taste of caramel with the creamy, custard-like notes of creme brulee. Features a smooth, velvety texture and a lingering sweet finish. Perfect for those who enjoy dessert-like coffee flavors.',
    'Coffee Pods',
    1, 'Downtown Store',
    '123 Main Street',
    'San Francisco',
    'CA',
    '94105',
    '415-555-0101',
    'downtown@store.com',
    37.7749, -122.4194,
    'flagship',
    5000,
    4.5,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    0, 25, 100, '2024-03-05 09:00:00', '2024-03-15 14:30:00',
    true, false, DATEADD(day, 3, CURRENT_TIMESTAMP()),
    15, 105, 450, 95, 380, 1140,
    0, 'critical', true, 1.3, 'increasing', 0.4,
    CURRENT_TIMESTAMP(),
    true,
    '2023-01-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- Marina Store - Caramel Creme Brulee (In Stock)
    ('NES-ORIG-008', 'Nespresso Original - Caramel Creme Brulee',
    'A sweet and indulgent coffee that combines the rich taste of caramel with the creamy, custard-like notes of creme brulee. Features a smooth, velvety texture and a lingering sweet finish. Perfect for those who enjoy dessert-like coffee flavors.',
    'Coffee Pods',
    2, 'Marina Store',
    '456 Beach Street',
    'San Francisco',
    'CA',
    '94123',
    '415-555-0102',
    'marina@store.com',
    37.8024, -122.4360,
    'express',
    2000,
    4.3,
    '{"monday": {"open": "07:00", "close": "20:00"}, "tuesday": {"open": "07:00", "close": "20:00"}, "wednesday": {"open": "07:00", "close": "20:00"}, "thursday": {"open": "07:00", "close": "20:00"}, "friday": {"open": "07:00", "close": "21:00"}, "saturday": {"open": "08:00", "close": "21:00"}, "sunday": {"open": "08:00", "close": "20:00"}}',
    'sales_floor',
    15, 20, 80, '2024-03-14 10:00:00', '2024-03-15 14:30:00',
    false, true, NULL,
    12, 84, 360, 80, 320, 960,
    1, 'high', true, 1.3, 'increasing', 0.4,
    CURRENT_TIMESTAMP(),
    true,
    '2023-02-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- Mission Store - Caramel Creme Brulee (In Stock)
    ('NES-ORIG-008', 'Nespresso Original - Caramel Creme Brulee',
    'A sweet and indulgent coffee that combines the rich taste of caramel with the creamy, custard-like notes of creme brulee. Features a smooth, velvety texture and a lingering sweet finish. Perfect for those who enjoy dessert-like coffee flavors.',
    'Coffee Pods',
    3, 'Mission Store',
    '789 Valencia Street',
    'San Francisco',
    'CA',
    '94110',
    '415-555-0103',
    'mission@store.com',
    37.7607, -122.4215,
    'flagship',
    4500,
    4.7,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    25, 20, 90, '2024-03-13 11:00:00', '2024-03-15 14:30:00',
    false, false, NULL,
    10, 70, 300, 70, 280, 840,
    2, 'medium', true, 1.3, 'increasing', 0.4,
    CURRENT_TIMESTAMP(),
    true,
    '2023-03-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- Similar to Caramel Creme Brulee: Sweet dessert flavor
    ('NES-ORIG-009', 'Nespresso Original - Vanilla Custard Pie',
    'A smooth and creamy coffee with rich vanilla custard notes and a hint of buttery pastry. Features a sweet, dessert-like profile with a velvety texture and lingering vanilla finish. Perfect alternative for those who enjoy sweet, dessert-inspired coffee flavors.',
    'Coffee Pods',
    1, 'Downtown Store',
    '123 Main Street',
    'San Francisco',
    'CA',
    '94105',
    '415-555-0101',
    'downtown@store.com',
    37.7749, -122.4194,
    'flagship',
    5000,
    4.5,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    40, 20, 80, '2024-03-10 09:00:00', '2024-03-15 14:30:00',
    false, false, NULL,
    10, 70, 300, 65, 260, 780,
    4, 'low', true, 1.2, 'increasing', 0.3,
    CURRENT_TIMESTAMP(),
    true,
    '2023-01-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- Similar to Caramel Creme Brulee: Caramel flavor
    ('NES-ORIG-010', 'Nespresso Original - Caramel Cookie',
    'A sweet and indulgent coffee that combines rich caramel notes with a hint of buttery cookie flavor. Features a smooth, creamy texture and a sweet, lingering finish. Perfect for those who enjoy caramel-flavored coffee with a dessert-like quality.',
    'Coffee Pods',
    1, 'Downtown Store',
    '123 Main Street',
    'San Francisco',
    'CA',
    '94105',
    '415-555-0101',
    'downtown@store.com',
    37.7749, -122.4194,
    'flagship',
    5000,
    4.5,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    35, 20, 90, '2024-03-12 10:00:00', '2024-03-15 14:30:00',
    false, true, NULL,
    12, 84, 360, 70, 280, 840,
    3, 'high', true, 1.1, 'increasing', 0.35,
    CURRENT_TIMESTAMP(),
    true,
    '2023-01-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- Similar to Caramel Creme Brulee: Sweet flavor profile
    ('NES-ORIG-011', 'Nespresso Original - Hazelnut Muffin',
    'A rich and nutty coffee that combines the warm, toasty flavor of hazelnuts with sweet, bakery-like notes. Features a smooth, creamy texture and a sweet, nutty finish. Perfect for those who enjoy nutty, dessert-inspired coffee flavors.',
    'Coffee Pods',
    1, 'Downtown Store',
    '123 Main Street',
    'San Francisco',
    'CA',
    '94105',
    '415-555-0101',
    'downtown@store.com',
    37.7749, -122.4194,
    'flagship',
    5000,
    4.5,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    45, 25, 100, '2024-03-14 08:00:00', '2024-03-15 14:30:00',
    false, false, NULL,
    8, 56, 240, 60, 240, 720,
    6, 'medium', true, 1.2, 'stable', 0.2,
    CURRENT_TIMESTAMP(),
    true,
    '2023-01-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- Similar to Arpeggio: Same intensity (9), bold flavor profile
    ('NES-ORIG-002', 'Nespresso Original - Ristretto',
    'An intense and powerful coffee with a strong, concentrated flavor. Features deep roasted notes and a rich, full-bodied taste with a lingering finish. Perfect for those who enjoy a bold, traditional espresso experience.',
    'Coffee Pods',
    1, 'Downtown Store',
    '123 Main Street',
    'San Francisco',
    'CA',
    '94105',
    '415-555-0101',
    'downtown@store.com',
    37.7749, -122.4194,
    'flagship',
    5000,
    4.5,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    45, 25, 120, '2024-03-10 09:00:00', '2024-03-15 14:30:00',
    false, false, NULL,
    8, 56, 240, 60, 240, 720,
    6, 'medium', false, 1.0, 'increasing', 0.3,
    CURRENT_TIMESTAMP(),
    true,
    '2023-01-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- Similar to Arpeggio: Popular medium roast alternative
    ('NES-ORIG-003', 'Nespresso Original - Livanto',
    'A balanced and smooth coffee with a round, full-bodied flavor. Features caramelized notes and a sweet, harmonious taste with a lingering finish. Perfect for those who enjoy a well-rounded, medium-intensity coffee.',
    'Coffee Pods',
    1, 'Downtown Store',
    '123 Main Street',
    'San Francisco',
    'CA',
    '94105',
    '415-555-0101',
    'downtown@store.com',
    37.7749, -122.4194,
    'flagship',
    5000,
    4.5,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    35, 20, 100, '2024-03-05 11:00:00', '2024-03-15 14:30:00',
    false, true, NULL,
    7, 49, 210, 50, 210, 630,
    5, 'high', false, 1.0, 'stable', 0.1,
    CURRENT_TIMESTAMP(),
    true,
    '2023-01-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- Similar to Arpeggio: Similar intensity, different system
    ('LAV-POD-004', 'Lavazza A Modo Mio - Classico',
    'A traditional Italian espresso with a rich, full-bodied flavor. Features notes of chocolate and roasted nuts with a smooth, velvety texture. Perfect for those who enjoy a classic, intense espresso experience.',
    'Coffee Pods',
    1, 'Downtown Store',
    '123 Main Street',
    'San Francisco',
    'CA',
    '94105',
    '415-555-0101',
    'downtown@store.com',
    37.7749, -122.4194,
    'flagship',
    5000,
    4.5,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    25, 20, 80, '2024-03-12 10:00:00', '2024-03-15 14:30:00',
    false, true, NULL,
    5, 35, 150, 40, 150, 450,
    5, 'high', false, 1.0, 'increasing', 0.4,
    CURRENT_TIMESTAMP(),
    true,
    '2023-01-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- Different style: Premium Italian espresso
    ('ILLY-POD-005', 'Illy iperespresso - Intenso',
    'A premium Italian espresso with a bold, intense flavor. Features rich, dark chocolate notes and a smooth, velvety texture with a lingering finish. Perfect for those who appreciate a high-quality, traditional Italian espresso.',
    'Coffee Pods',
    1, 'Downtown Store',
    '123 Main Street',
    'San Francisco',
    'CA',
    '94105',
    '415-555-0101',
    'downtown@store.com',
    37.7749, -122.4194,
    'flagship',
    5000,
    4.5,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    60, 15, 75, '2024-03-14 08:00:00', '2024-03-15 14:30:00',
    false, false, NULL,
    4, 28, 120, 30, 120, 360,
    15, 'low', false, 1.0, 'stable', 0.2,
    CURRENT_TIMESTAMP(),
    true,
    '2023-01-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- Different style: Medium roast, different system
    ('TAS-POD-006', 'Tassimo T-Disc - Costa Rica',
    'A medium-roast coffee with bright, citrusy notes and a smooth, balanced flavor. Features hints of caramel and a clean, refreshing finish. Perfect for those who enjoy a lighter, more vibrant coffee experience.',
    'Coffee Pods',
    1, 'Downtown Store',
    '123 Main Street',
    'San Francisco',
    'CA',
    '94105',
    '415-555-0101',
    'downtown@store.com',
    37.7749, -122.4194,
    'flagship',
    5000,
    4.5,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    40, 20, 90, '2024-03-13 09:00:00', '2024-03-15 14:30:00',
    false, false, NULL,
    6, 42, 180, 45, 180, 540,
    7, 'medium', false, 1.0, 'increasing', 0.3,
    CURRENT_TIMESTAMP(),
    true,
    '2023-01-01 00:00:00',
    CURRENT_TIMESTAMP()),

    -- Different style: Specialty drink, not coffee
    ('DOL-POD-007', 'Dolce Gusto - Chococino',
    'A rich and creamy chocolate drink with a hint of coffee. Features smooth, velvety chocolate notes and a sweet, indulgent finish. Perfect for those who enjoy a sweet, dessert-like beverage with a touch of coffee flavor.',
    'Coffee Pods',
    1, 'Downtown Store',
    '123 Main Street',
    'San Francisco',
    'CA',
    '94105',
    '415-555-0101',
    'downtown@store.com',
    37.7749, -122.4194,
    'flagship',
    5000,
    4.5,
    '{"monday": {"open": "08:00", "close": "21:00"}, "tuesday": {"open": "08:00", "close": "21:00"}, "wednesday": {"open": "08:00", "close": "21:00"}, "thursday": {"open": "08:00", "close": "21:00"}, "friday": {"open": "08:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "09:00", "close": "20:00"}}',
    'sales_floor',
    30, 15, 60, '2024-03-11 10:00:00', '2024-03-15 14:30:00',
    false, true, NULL,
    3, 21, 90, 25, 90, 270,
    10, 'medium', true, 1.2, 'stable', 0.1,
    CURRENT_TIMESTAMP(),
    true,
    '2023-01-01 00:00:00',
    CURRENT_TIMESTAMP());

-- Stock alerts table with embedded product and location info
CREATE TABLE IF NOT EXISTS stock_alerts (
    alert_id BIGINT GENERATED ALWAYS AS IDENTITY,
    sku STRING NOT NULL,
    product_name STRING NOT NULL,
    product_category STRING NOT NULL,
    store_name STRING NOT NULL,
    location_type STRING NOT NULL,
    alert_type STRING NOT NULL, -- 'low_stock', 'out_of_stock', 'critical'
    status STRING NOT NULL, -- 'active', 'resolved', 'acknowledged'
    current_quantity INT NOT NULL,
    min_stock_level INT NOT NULL,
    notes STRING
);

-- Insert sample stock alerts
INSERT INTO stock_alerts (
    sku,
    product_name,
    product_category,
    store_name,
    location_type,
    alert_type,
    status,
    current_quantity,
    min_stock_level,
    notes
) VALUES
    -- Critical out of stock alert
    ('NES-ORIG-001', 'Nespresso Original - Arpeggio', 'Coffee Pods', 'Downtown Store', 'sales_floor',
    'out_of_stock', 'active',
    0, 30,
    'High-demand item out of stock. Next restock in 5 days. Consider promoting Ristretto as alternative.'),

    -- New out of stock alert for Caramel Creme Brulee
    ('NES-ORIG-008', 'Nespresso Original - Caramel Creme Brulee', 'Coffee Pods', 'Downtown Store', 'sales_floor',
    'out_of_stock', 'active',
    0, 25,
    'Popular seasonal flavor out of stock. High demand during spring season. Next restock in 3 days. Consider Vanilla Custard Pie or Caramel Cookie as alternatives.'),

    -- Low stock alert for Caramel Cookie
    ('NES-ORIG-010', 'Nespresso Original - Caramel Cookie', 'Coffee Pods', 'Downtown Store', 'sales_floor',
    'low_stock', 'active',
    35, 20,
    'Popular alternative to Caramel Creme Brulee. Stock level approaching minimum.'),

    -- Low stock alert for Livanto
    ('NES-ORIG-003', 'Nespresso Original - Livanto', 'Coffee Pods', 'Downtown Store', 'sales_floor',
    'low_stock', 'active',
    35, 20,
    'Stock level approaching minimum. Consider restocking soon.'),

    -- Low stock alert for Lavazza
    ('LAV-POD-004', 'Lavazza A Modo Mio - Classico', 'Coffee Pods', 'Downtown Store', 'sales_floor',
    'low_stock', 'active',
    25, 20,
    'Stock level close to minimum. High risk of stockout.'),

    -- Low stock alert for Chococino
    ('DOL-POD-007', 'Dolce Gusto - Chococino', 'Coffee Pods', 'Downtown Store', 'sales_floor',
    'low_stock', 'active',
    30, 15,
    'Seasonal item running low. Consider increasing stock for upcoming promotion.'),

    -- Resolved alert (for historical tracking)
    ('NES-ORIG-002', 'Nespresso Original - Ristretto', 'Coffee Pods', 'Downtown Store', 'sales_floor',
    'low_stock', 'resolved',
    45, 25,
    'Stock level restored. Previous low stock situation resolved on 2024-03-10.');

-- Store network table for cross-store inventory checks
CREATE TABLE IF NOT EXISTS store_network (
    store_id BIGINT GENERATED ALWAYS AS IDENTITY,
    store_name STRING NOT NULL,
    address STRING NOT NULL,
    is_active BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL
);

-- Create indexes for better query performance
CREATE INDEX idx_store_inventory_with_location_sku ON store_inventory_with_location(sku);
CREATE INDEX idx_store_inventory_with_location_coordinates ON store_inventory_with_location(latitude, longitude);
CREATE INDEX idx_store_inventory_with_location_store_id ON store_inventory_with_location(store_id);
CREATE INDEX idx_store_inventory_with_location_product_name ON store_inventory_with_location(product_name);
CREATE INDEX idx_store_inventory_with_location_quantity ON store_inventory_with_location(quantity) WHERE quantity > 0;

-- Example queries for finding nearby stores with stock:

-- 1. Find all stores within 5 miles of a given location that have the product in stock
/*
SELECT 
    store_name,
    store_address,
    store_city,
    store_state,
    store_zip_code,
    store_phone,
    store_hours,
    quantity,
    (6371 * acos(cos(radians(37.7749)) * cos(radians(latitude)) * cos(radians(longitude) - radians(-122.4194)) + sin(radians(37.7749)) * sin(radians(latitude)))) AS distance_km
FROM store_inventory_with_location
WHERE 
    sku = 'NES-ORIG-008'
    AND quantity > 0
    AND is_store_active = true
HAVING distance_km <= 8.04672  -- 5 miles in kilometers
ORDER BY distance_km;
*/

-- 2. Find the nearest store with the product in stock
/*
SELECT 
    store_name,
    store_address,
    store_city,
    store_state,
    store_zip_code,
    store_phone,
    store_hours,
    quantity,
    (6371 * acos(cos(radians(37.7749)) * cos(radians(latitude)) * cos(radians(longitude) - radians(-122.4194)) + sin(radians(37.7749)) * sin(radians(latitude)))) AS distance_km
FROM store_inventory_with_location
WHERE 
    sku = 'NES-ORIG-008'
    AND quantity > 0
    AND is_store_active = true
ORDER BY distance_km
LIMIT 1;
*/

-- 3. Find all stores with the product in stock, ordered by distance and quantity
/*
SELECT 
    store_name,
    store_address,
    store_city,
    store_state,
    store_zip_code,
    store_phone,
    store_hours,
    quantity,
    (6371 * acos(cos(radians(37.7749)) * cos(radians(latitude)) * cos(radians(longitude) - radians(-122.4194)) + sin(radians(37.7749)) * sin(radians(latitude)))) AS distance_km
FROM store_inventory_with_location
WHERE 
    sku = 'NES-ORIG-008'
    AND quantity > 0
    AND is_store_active = true
ORDER BY distance_km, quantity DESC;
*/