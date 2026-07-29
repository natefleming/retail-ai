USE IDENTIFIER(:database);

INSERT INTO dim_stores (
    store_id, store_name, store_address, store_city, store_state, store_zipcode, store_country,
    store_phone, store_email, store_manager_id, store_manager_name, opening_date,
    store_type, store_size_sqft, store_rating, store_hours,
    latitude, longitude, region_id, region_name, district_id, district_name, timezone, is_active,
    has_footwear, has_apparel, has_team_sports, has_fitness, has_outdoor, has_cycling, has_golf, has_hunting_fishing,
    store_details_text, parking_spaces, last_renovation_date, created_at, updated_at
) VALUES
    (101, 'SportsPlex Flagship - Denver', '1500 16th Street Mall', 'Denver', 'CO', '80202', 'USA',
    '303-555-0101', 'denver.flagship@sportsplex.com', 'MGR-101', 'Sarah Mitchell', '2018-04-15',
    'flagship', 65000, 4.7,
    '{"monday": {"open": "09:00", "close": "21:00"}, "tuesday": {"open": "09:00", "close": "21:00"}, "wednesday": {"open": "09:00", "close": "21:00"}, "thursday": {"open": "09:00", "close": "21:00"}, "friday": {"open": "09:00", "close": "22:00"}, "saturday": {"open": "08:00", "close": "22:00"}, "sunday": {"open": "10:00", "close": "20:00"}}',
    39.7456, -104.9942, 'REG-WEST', 'Western Region', 'DIST-MTN', 'Mountain District', 'America/Denver', true,
    true, true, true, true, true, true, true, true,
    'SportsPlex Flagship Denver is our largest store located on the iconic 16th Street Mall in downtown Denver. This 65,000 sq ft flagship features all departments including a full-service bike shop, golf simulator bay, and outdoor gear test area. Offers expert fitting services for running shoes, ski boots, and golf clubs. Located near Union Station with convenient parking garage access. Open 7 days a week with extended weekend hours.',
    250, '2023-06-01', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    (102, 'SportsPlex - Austin South', '4500 South Lamar Blvd', 'Austin', 'TX', '78745', 'USA',
    '512-555-0102', 'austin.south@sportsplex.com', 'MGR-102', 'Marcus Rodriguez', '2020-09-01',
    'standard', 45000, 4.5,
    '{"monday": {"open": "09:00", "close": "21:00"}, "tuesday": {"open": "09:00", "close": "21:00"}, "wednesday": {"open": "09:00", "close": "21:00"}, "thursday": {"open": "09:00", "close": "21:00"}, "friday": {"open": "09:00", "close": "21:00"}, "saturday": {"open": "08:00", "close": "21:00"}, "sunday": {"open": "10:00", "close": "19:00"}}',
    30.2241, -97.7700, 'REG-SOUTH', 'Southern Region', 'DIST-TX', 'Texas District', 'America/Chicago', true,
    true, true, true, true, true, true, false, true,
    'SportsPlex Austin South is a 45,000 sq ft store serving the South Austin community with a strong focus on running, outdoor recreation, and team sports. Features a dedicated running shoe fitting area with gait analysis, extensive camping and hiking section, and youth sports equipment. Popular with UT Austin students and active Austin residents. Located on South Lamar with ample free parking.',
    180, '2022-01-15', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    (103, 'SportsPlex - Chicago North', '3200 North Clark Street', 'Chicago', 'IL', '60657', 'USA',
    '312-555-0103', 'chicago.north@sportsplex.com', 'MGR-103', 'Jennifer Park', '2019-03-20',
    'standard', 40000, 4.4,
    '{"monday": {"open": "09:00", "close": "21:00"}, "tuesday": {"open": "09:00", "close": "21:00"}, "wednesday": {"open": "09:00", "close": "21:00"}, "thursday": {"open": "09:00", "close": "21:00"}, "friday": {"open": "09:00", "close": "21:00"}, "saturday": {"open": "08:00", "close": "21:00"}, "sunday": {"open": "10:00", "close": "19:00"}}',
    41.9403, -87.6567, 'REG-CENTRAL', 'Central Region', 'DIST-IL', 'Illinois District', 'America/Chicago', true,
    true, true, true, true, false, true, false, false,
    'SportsPlex Chicago North is a 40,000 sq ft store in the Lakeview neighborhood, steps from Wrigley Field. Strong in team sports gear for Cubs, Bears, Bulls, and Blackhawks fans. Excellent cycling section serving the Lakefront Trail community. Features a basketball court for shoe testing and a running shoe wall with expert staff. Limited outdoor and camping selection but strong fitness equipment department.',
    120, '2023-09-01', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    (104, 'SportsPlex Outlet - Orlando', '8200 Vineland Ave', 'Orlando', 'FL', '32821', 'USA',
    '407-555-0104', 'orlando.outlet@sportsplex.com', 'MGR-104', 'David Thompson', '2021-11-10',
    'outlet', 35000, 4.2,
    '{"monday": {"open": "10:00", "close": "21:00"}, "tuesday": {"open": "10:00", "close": "21:00"}, "wednesday": {"open": "10:00", "close": "21:00"}, "thursday": {"open": "10:00", "close": "21:00"}, "friday": {"open": "10:00", "close": "22:00"}, "saturday": {"open": "09:00", "close": "22:00"}, "sunday": {"open": "10:00", "close": "20:00"}}',
    28.3886, -81.4924, 'REG-SOUTH', 'Southern Region', 'DIST-FL', 'Florida District', 'America/New_York', true,
    true, true, true, true, true, false, true, false,
    'SportsPlex Outlet Orlando is a 35,000 sq ft outlet store near the International Drive tourist corridor. Offers discounted sporting goods from past seasons and overstock at 20-60% off retail. Strong in athletic footwear, golf equipment, and team sports. Popular with both locals and tourists. Features a clearance section with deep discounts on seasonal merchandise. Located near Premium Outlets with shared parking.',
    200, null, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    (105, 'SportsPlex - Portland', '1200 NW Couch Street', 'Portland', 'OR', '97209', 'USA',
    '503-555-0105', 'portland@sportsplex.com', 'MGR-105', 'Emily Chen', '2022-05-01',
    'standard', 42000, 4.6,
    '{"monday": {"open": "09:00", "close": "21:00"}, "tuesday": {"open": "09:00", "close": "21:00"}, "wednesday": {"open": "09:00", "close": "21:00"}, "thursday": {"open": "09:00", "close": "21:00"}, "friday": {"open": "09:00", "close": "21:00"}, "saturday": {"open": "08:00", "close": "21:00"}, "sunday": {"open": "10:00", "close": "19:00"}}',
    45.5252, -122.6831, 'REG-WEST', 'Western Region', 'DIST-PNW', 'Pacific Northwest District', 'America/Los_Angeles', true,
    true, true, false, true, true, true, false, true,
    'SportsPlex Portland is a 42,000 sq ft store in the Pearl District with a strong outdoor recreation and cycling focus. Features the largest cycling department in the chain with a full-service bike repair shop, a comprehensive hiking and camping section, and an extensive trail running selection. Known for eco-friendly and sustainable product curation. Popular with Portland trail runners, cyclists, and outdoor enthusiasts. Street parking and nearby garage available.',
    80, '2024-02-01', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());
