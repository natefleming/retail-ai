USE IDENTIFIER(:database);

INSERT INTO inventory (
    inventory_id, product_id,
    store_id, store_name, store_address, store_city, store_state, store_zip_code, store_phone, store_type, latitude, longitude,
    store, store_quantity, warehouse, warehouse_quantity, retail_amount, popularity_rating, department, aisle_location, is_closeout,
    min_stock_level, max_stock_level, last_restock_date, last_count_date, is_out_of_stock, is_low_stock, next_restock_date,
    daily_demand_prediction, weekly_demand_prediction, monthly_demand_prediction, last_7_days_sales, last_30_days_sales, last_90_days_sales, days_until_stockout, stockout_risk_level,
    is_seasonal, season_peak_factor, trend_direction, trend_strength,
    last_prediction_update, is_store_active
) VALUES
    -- Nike Air Zoom Pegasus 41 across stores
    (1001, 1, 101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', '303-555-0101', 'flagship', 39.7456, -104.9942,
    'DEN-FLAG', 45, 'WH-WEST', 200, 129.99, 'high', 'Athletic Footwear', 'Aisle 1A', false,
    20, 80, '2025-03-01', '2025-03-10', false, false, '2025-03-25',
    3, 21, 90, 22, 88, 270, 15, 'low',
    false, 1.0, 'stable', 0.7,
    CURRENT_TIMESTAMP(), true),

    (1002, 1, 102, 'SportsPlex - Austin South', '4500 South Lamar Blvd', 'Austin', 'TX', '78745', '512-555-0102', 'standard', 30.2241, -97.7700,
    'AUS-SOUTH', 12, 'WH-SOUTH', 150, 129.99, 'high', 'Athletic Footwear', 'Aisle 1A', false,
    15, 60, '2025-02-15', '2025-03-05', false, true, '2025-03-18',
    2, 14, 60, 16, 58, 180, 6, 'high',
    false, 1.0, 'increasing', 0.8,
    CURRENT_TIMESTAMP(), true),

    (1003, 1, 105, 'SportsPlex - Portland', '1200 NW Couch Street', 'Portland', 'OR', '97209', '503-555-0105', 'standard', 45.5252, -122.6831,
    'PDX-PEARL', 28, 'WH-WEST', 200, 129.99, 'high', 'Athletic Footwear', 'Aisle 1A', false,
    15, 60, '2025-03-05', '2025-03-12', false, false, '2025-03-28',
    3, 21, 90, 19, 82, 255, 9, 'medium',
    false, 1.0, 'stable', 0.6,
    CURRENT_TIMESTAMP(), true),

    -- Adidas Ultraboost 24
    (1004, 2, 101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', '303-555-0101', 'flagship', 39.7456, -104.9942,
    'DEN-FLAG', 22, 'WH-WEST', 100, 189.99, 'high', 'Athletic Footwear', 'Aisle 1B', false,
    10, 40, '2025-02-20', '2025-03-10', false, false, '2025-03-22',
    2, 14, 55, 12, 50, 160, 11, 'medium',
    false, 1.0, 'increasing', 0.7,
    CURRENT_TIMESTAMP(), true),

    (1005, 2, 103, 'SportsPlex - Chicago North', '3200 North Clark Street', 'Chicago', 'IL', '60657', '312-555-0103', 'standard', 41.9403, -87.6567,
    'CHI-NORTH', 8, 'WH-CENTRAL', 80, 189.99, 'high', 'Athletic Footwear', 'Aisle 1B', false,
    10, 40, '2025-02-10', '2025-03-08', false, true, '2025-03-16',
    2, 14, 55, 13, 48, 150, 4, 'critical',
    false, 1.0, 'increasing', 0.8,
    CURRENT_TIMESTAMP(), true),

    -- LeBron XXI
    (1006, 3, 103, 'SportsPlex - Chicago North', '3200 North Clark Street', 'Chicago', 'IL', '60657', '312-555-0103', 'standard', 41.9403, -87.6567,
    'CHI-NORTH', 18, 'WH-CENTRAL', 60, 199.99, 'high', 'Athletic Footwear', 'Aisle 2A', false,
    8, 30, '2025-03-01', '2025-03-10', false, false, '2025-03-25',
    1, 7, 30, 8, 32, 95, 18, 'low',
    false, 1.0, 'stable', 0.5,
    CURRENT_TIMESTAMP(), true),

    -- Under Armour Tech 2.0 Tee across stores
    (1007, 6, 101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', '303-555-0101', 'flagship', 39.7456, -104.9942,
    'DEN-FLAG', 85, 'WH-WEST', 500, 24.99, 'high', 'Athletic Apparel', 'Aisle 5A', false,
    40, 150, '2025-03-01', '2025-03-10', false, false, '2025-04-01',
    5, 35, 150, 38, 145, 440, 17, 'low',
    false, 1.0, 'stable', 0.6,
    CURRENT_TIMESTAMP(), true),

    (1008, 6, 102, 'SportsPlex - Austin South', '4500 South Lamar Blvd', 'Austin', 'TX', '78745', '512-555-0102', 'standard', 30.2241, -97.7700,
    'AUS-SOUTH', 55, 'WH-SOUTH', 400, 24.99, 'high', 'Athletic Apparel', 'Aisle 4A', false,
    30, 120, '2025-02-25', '2025-03-08', false, false, '2025-03-28',
    4, 28, 120, 30, 115, 350, 14, 'low',
    false, 1.0, 'increasing', 0.7,
    CURRENT_TIMESTAMP(), true),

    -- The North Face ThermoBall (seasonal - high demand fall/winter)
    (1009, 8, 101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', '303-555-0101', 'flagship', 39.7456, -104.9942,
    'DEN-FLAG', 35, 'WH-WEST', 80, 229.99, 'high', 'Athletic Apparel', 'Aisle 7A', false,
    15, 60, '2025-02-01', '2025-03-10', false, false, '2025-03-20',
    2, 14, 55, 18, 60, 200, 17, 'low',
    true, 1.8, 'decreasing', 0.6,
    CURRENT_TIMESTAMP(), true),

    (1010, 8, 105, 'SportsPlex - Portland', '1200 NW Couch Street', 'Portland', 'OR', '97209', '503-555-0105', 'standard', 45.5252, -122.6831,
    'PDX-PEARL', 5, 'WH-WEST', 30, 229.99, 'high', 'Athletic Apparel', 'Aisle 6A', false,
    10, 40, '2025-01-15', '2025-03-05', false, true, '2025-03-15',
    2, 14, 55, 15, 52, 175, 2, 'critical',
    true, 2.0, 'stable', 0.8,
    CURRENT_TIMESTAMP(), true),

    -- Wilson Duke Football (seasonal - fall peak)
    (1011, 11, 101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', '303-555-0101', 'flagship', 39.7456, -104.9942,
    'DEN-FLAG', 15, 'WH-WEST', 40, 169.99, 'medium', 'Team Sports', 'Aisle 8A', false,
    5, 25, '2025-01-01', '2025-03-10', false, false, '2025-08-01',
    0, 2, 8, 1, 5, 45, 60, 'low',
    true, 3.0, 'decreasing', 0.9,
    CURRENT_TIMESTAMP(), true),

    -- Bowflex Treadmill (high value, low volume)
    (1012, 14, 101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', '303-555-0101', 'flagship', 39.7456, -104.9942,
    'DEN-FLAG', 3, 'WH-WEST', 8, 2499.99, 'medium', 'Fitness Equipment', 'Floor Display G1', false,
    2, 6, '2025-02-15', '2025-03-10', false, false, '2025-04-01',
    0, 1, 3, 0, 2, 8, 30, 'low',
    false, 1.3, 'stable', 0.4,
    CURRENT_TIMESTAMP(), true),

    -- Coleman Sundome Tent (seasonal - spring/summer peak)
    (1013, 17, 101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', '303-555-0101', 'flagship', 39.7456, -104.9942,
    'DEN-FLAG', 22, 'WH-WEST', 100, 79.99, 'high', 'Outdoor & Camping', 'Aisle 10A', false,
    10, 50, '2025-03-01', '2025-03-10', false, false, '2025-03-28',
    2, 14, 55, 10, 38, 65, 11, 'medium',
    true, 2.5, 'increasing', 0.9,
    CURRENT_TIMESTAMP(), true),

    (1014, 17, 105, 'SportsPlex - Portland', '1200 NW Couch Street', 'Portland', 'OR', '97209', '503-555-0105', 'standard', 45.5252, -122.6831,
    'PDX-PEARL', 30, 'WH-WEST', 100, 79.99, 'high', 'Outdoor & Camping', 'Aisle 8A', false,
    10, 50, '2025-03-05', '2025-03-12', false, false, '2025-03-30',
    3, 21, 80, 18, 55, 90, 10, 'medium',
    true, 2.8, 'increasing', 0.9,
    CURRENT_TIMESTAMP(), true),

    -- YETI Tundra 45 (seasonal - summer)
    (1015, 18, 101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', '303-555-0101', 'flagship', 39.7456, -104.9942,
    'DEN-FLAG', 8, 'WH-WEST', 25, 324.99, 'medium', 'Outdoor & Camping', 'Aisle 10B', false,
    4, 15, '2025-02-15', '2025-03-10', false, false, '2025-04-01',
    1, 4, 15, 3, 10, 20, 8, 'medium',
    true, 2.0, 'increasing', 0.8,
    CURRENT_TIMESTAMP(), true),

    -- Trek Marlin 7 (seasonal - spring/summer)
    (1016, 20, 105, 'SportsPlex - Portland', '1200 NW Couch Street', 'Portland', 'OR', '97209', '503-555-0105', 'standard', 45.5252, -122.6831,
    'PDX-PEARL', 4, 'WH-WEST', 10, 1049.99, 'high', 'Cycling', 'Floor Display I1', false,
    2, 8, '2025-03-01', '2025-03-12', false, false, '2025-04-01',
    0, 2, 6, 1, 4, 8, 14, 'medium',
    true, 2.2, 'increasing', 0.9,
    CURRENT_TIMESTAMP(), true),

    -- Garmin Forerunner 265
    (1017, 23, 101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', '303-555-0101', 'flagship', 39.7456, -104.9942,
    'DEN-FLAG', 10, 'WH-WEST', 30, 449.99, 'high', 'Accessories', 'Display Case J2', false,
    4, 15, '2025-03-01', '2025-03-10', false, false, '2025-03-25',
    1, 5, 18, 4, 16, 50, 10, 'medium',
    false, 1.0, 'stable', 0.6,
    CURRENT_TIMESTAMP(), true),

    -- Nike Club Fleece Hoodie (seasonal - fall/winter)
    (1018, 25, 103, 'SportsPlex - Chicago North', '3200 North Clark Street', 'Chicago', 'IL', '60657', '312-555-0103', 'standard', 41.9403, -87.6567,
    'CHI-NORTH', 40, 'WH-CENTRAL', 200, 64.99, 'high', 'Athletic Apparel', 'Aisle 5B', false,
    20, 80, '2025-02-15', '2025-03-08', false, false, '2025-03-20',
    3, 21, 85, 22, 80, 280, 13, 'low',
    true, 1.8, 'decreasing', 0.7,
    CURRENT_TIMESTAMP(), true),

    -- Nike AF1 at outlet (closeout pricing)
    (1019, 5, 104, 'SportsPlex Outlet - Orlando', '8200 Vineland Ave', 'Orlando', 'FL', '32821', '407-555-0104', 'outlet', 28.3886, -81.4924,
    'ORL-OUTLET', 65, 'WH-SOUTH', 300, 89.99, 'high', 'Athletic Footwear', 'Aisle 2A', true,
    30, 120, '2025-03-01', '2025-03-10', false, false, '2025-04-01',
    5, 35, 140, 32, 130, 400, 13, 'low',
    false, 1.0, 'stable', 0.6,
    CURRENT_TIMESTAMP(), true),

    -- Lululemon Align Pants
    (1020, 30, 101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', '303-555-0101', 'flagship', 39.7456, -104.9942,
    'DEN-FLAG', 18, 'WH-WEST', 60, 98.00, 'high', 'Athletic Apparel', 'Aisle 6C', false,
    10, 40, '2025-03-01', '2025-03-10', false, false, '2025-03-22',
    2, 10, 40, 9, 38, 115, 9, 'medium',
    false, 1.0, 'increasing', 0.8,
    CURRENT_TIMESTAMP(), true),

    -- Callaway Driver (seasonal - spring golf)
    (1021, 28, 101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', '303-555-0101', 'flagship', 39.7456, -104.9942,
    'DEN-FLAG', 5, 'WH-WEST', 12, 599.99, 'medium', 'Golf', 'Floor Display K1', false,
    2, 10, '2025-02-15', '2025-03-10', false, false, '2025-04-01',
    0, 1, 4, 1, 3, 5, 15, 'low',
    true, 2.5, 'increasing', 0.9,
    CURRENT_TIMESTAMP(), true),

    -- Osprey backpack (seasonal - spring/summer)
    (1022, 19, 105, 'SportsPlex - Portland', '1200 NW Couch Street', 'Portland', 'OR', '97209', '503-555-0105', 'standard', 45.5252, -122.6831,
    'PDX-PEARL', 6, 'WH-WEST', 20, 289.99, 'high', 'Outdoor & Camping', 'Aisle 9A', false,
    3, 12, '2025-03-01', '2025-03-12', false, false, '2025-03-28',
    1, 4, 15, 3, 12, 22, 6, 'high',
    true, 2.5, 'increasing', 0.9,
    CURRENT_TIMESTAMP(), true),

    -- Hydro Flask (year-round, high volume)
    (1023, 24, 102, 'SportsPlex - Austin South', '4500 South Lamar Blvd', 'Austin', 'TX', '78745', '512-555-0102', 'standard', 30.2241, -97.7700,
    'AUS-SOUTH', 42, 'WH-SOUTH', 300, 44.95, 'high', 'Accessories', 'Endcap J3', false,
    20, 80, '2025-03-05', '2025-03-10', false, false, '2025-03-25',
    4, 28, 110, 25, 100, 310, 10, 'medium',
    false, 1.2, 'stable', 0.5,
    CURRENT_TIMESTAMP(), true),

    -- Out of stock example: Rogue Dumbbells at Chicago
    (1024, 15, 103, 'SportsPlex - Chicago North', '3200 North Clark Street', 'Chicago', 'IL', '60657', '312-555-0103', 'standard', 41.9403, -87.6567,
    'CHI-NORTH', 0, 'WH-CENTRAL', 3, 899.99, 'high', 'Fitness Equipment', 'Floor Display G2', false,
    1, 5, '2025-01-10', '2025-03-08', true, false, '2025-03-20',
    0, 1, 3, 1, 4, 12, 0, 'critical',
    false, 1.3, 'increasing', 0.8,
    CURRENT_TIMESTAMP(), true);
