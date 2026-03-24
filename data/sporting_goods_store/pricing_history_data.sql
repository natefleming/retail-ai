USE IDENTIFIER(:database);

INSERT INTO pricing_history (
    pricing_id, product_id, sku, store_id,
    effective_date, end_date, original_price, new_price, cost, margin_pct,
    price_change_type, price_change_reason, promotion_id, promotion_name, markdown_pct,
    department, sport_category, season, is_active, approved_by,
    created_at, created_by
) VALUES
    -- Current base prices (initial pricing)
    (9001, 1, 'NKE-RUN-001', null,
    '2024-04-01', null, 129.99, 129.99, 65.00, 50.0,
    'initial', 'Launch pricing for Nike Air Zoom Pegasus 41', null, null, 0.0,
    'Athletic Footwear', 'Running', 'Year-Round', true, 'VP-MERCH-01',
    '2024-04-01', 'system'),

    (9002, 2, 'ADI-RUN-001', null,
    '2024-02-01', null, 189.99, 189.99, 85.00, 55.3,
    'initial', 'Launch pricing for Adidas Ultraboost 24', null, null, 0.0,
    'Athletic Footwear', 'Running', 'Year-Round', true, 'VP-MERCH-01',
    '2024-02-01', 'system'),

    -- Spring Running Promotion (10% off select running shoes at Austin)
    (9003, 1, 'NKE-RUN-001', 102,
    '2025-03-01', '2025-03-31', 129.99, 116.99, 65.00, 44.4,
    'promotion', 'Spring Running Event - 10% off', 'PROMO-SPRING-RUN', 'Spring Into Running', 10.0,
    'Athletic Footwear', 'Running', 'Spring', true, 'MERCH-MGR-02',
    '2025-02-25', 'merch_team'),

    -- End of winter markdown on ThermoBall jacket
    (9004, 8, 'TNF-JKT-001', null,
    '2025-02-15', '2025-03-31', 229.99, 183.99, 110.00, 40.2,
    'markdown', 'End of winter markdown - 20% off outerwear', null, null, 20.0,
    'Athletic Apparel', 'Outdoor', 'Winter', true, 'VP-MERCH-01',
    '2025-02-10', 'merch_team'),

    (9005, 8, 'TNF-JKT-001', null,
    '2024-08-01', '2025-02-14', 229.99, 229.99, 110.00, 52.2,
    'initial', 'Fall/Winter 2024 launch pricing', null, null, 0.0,
    'Athletic Apparel', 'Outdoor', 'Fall', false, 'VP-MERCH-01',
    '2024-08-01', 'system'),

    -- Nike AF1 clearance at outlet
    (9006, 5, 'NKE-CLT-001', 104,
    '2025-01-15', null, 114.99, 89.99, 48.00, 46.6,
    'clearance', 'Outlet clearance pricing - older colorway', null, null, 21.7,
    'Athletic Footwear', 'Lifestyle', 'Year-Round', true, 'OUTLET-MGR-04',
    '2025-01-10', 'outlet_team'),

    (9007, 5, 'NKE-CLT-001', null,
    '2020-01-01', null, 114.99, 114.99, 48.00, 58.3,
    'initial', 'Standard retail pricing', null, null, 0.0,
    'Athletic Footwear', 'Lifestyle', 'Year-Round', true, 'VP-MERCH-01',
    '2020-01-01', 'system'),

    -- Competitive price match on Garmin watch
    (9008, 23, 'GRM-WCH-001', null,
    '2025-03-01', '2025-03-15', 449.99, 429.99, 220.00, 48.8,
    'competitive', 'Competitive price match - online retailer undercut', null, null, 4.4,
    'Accessories', 'Running', 'Year-Round', true, 'VP-MERCH-01',
    '2025-02-28', 'pricing_team'),

    (9009, 23, 'GRM-WCH-001', null,
    '2024-03-01', '2025-02-28', 449.99, 449.99, 220.00, 51.1,
    'initial', 'Launch pricing for Forerunner 265', null, null, 0.0,
    'Accessories', 'Running', 'Year-Round', false, 'VP-MERCH-01',
    '2024-03-01', 'system'),

    -- Seasonal price increase on camping gear (peak season)
    (9010, 17, 'COL-TNT-001', null,
    '2025-03-01', null, 79.99, 79.99, 35.00, 56.2,
    'seasonal', 'Standard spring pricing maintained despite supplier cost increase', null, null, 0.0,
    'Outdoor & Camping', 'Camping', 'Spring', true, 'VP-MERCH-01',
    '2025-02-25', 'merch_team'),

    -- YETI promotional pricing for summer launch
    (9011, 18, 'YET-CLR-001', null,
    '2025-04-01', '2025-04-30', 324.99, 299.99, 160.00, 46.7,
    'promotion', 'Summer Kickoff Sale - YETI 8% off', 'PROMO-SUMMER-KICK', 'Summer Kickoff', 7.7,
    'Outdoor & Camping', 'Camping', 'Summer', false, 'VP-MERCH-01',
    '2025-03-15', 'merch_team'),

    -- Lululemon - never discounted (premium brand strategy)
    (9012, 30, 'LLM-LGN-001', null,
    '2023-01-01', null, 98.00, 98.00, 38.00, 61.2,
    'initial', 'Lululemon maintains MSRP - no promotional pricing per brand agreement', null, null, 0.0,
    'Athletic Apparel', 'Yoga', 'Year-Round', true, 'VP-MERCH-01',
    '2023-01-01', 'system'),

    -- Rollback after failed price increase
    (9013, 6, 'UA-TOP-001', null,
    '2025-02-01', null, 27.99, 24.99, 10.00, 60.0,
    'rollback', 'Price rollback after 12% sales decline at $27.99 price point', null, null, 0.0,
    'Athletic Apparel', 'Training', 'Year-Round', true, 'VP-MERCH-01',
    '2025-01-28', 'pricing_team'),

    (9014, 6, 'UA-TOP-001', null,
    '2025-01-01', '2025-01-31', 24.99, 27.99, 10.00, 64.3,
    'initial', 'Price increase test - Q1 2025', null, null, 0.0,
    'Athletic Apparel', 'Training', 'Year-Round', false, 'VP-MERCH-01',
    '2024-12-15', 'pricing_team'),

    -- Golf driver spring launch pricing
    (9015, 28, 'CLB-GLF-001', null,
    '2025-03-01', null, 599.99, 599.99, 280.00, 53.3,
    'initial', 'Spring 2025 golf season launch - Paradym Ai Smoke', null, null, 0.0,
    'Golf', 'Golf', 'Spring', true, 'VP-MERCH-01',
    '2025-02-25', 'system');
