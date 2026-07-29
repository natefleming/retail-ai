USE IDENTIFIER(:database);

INSERT INTO sales_orders (
    order_id, order_line_id, product_id, sku, store_id, store_name, customer_id,
    order_date, order_timestamp, quantity, unit_price, discount_amount, discount_type, promotion_id,
    total_amount, cost_amount, margin_amount, margin_pct,
    payment_method, channel, department, sport_category, season,
    is_return, return_reason, sales_associate_id, created_at
) VALUES
    -- Recent sales at Denver Flagship
    (5001, 10001, 1, 'NKE-RUN-001', 101, 'SportsPlex Flagship - Denver', 'CUST-1001',
    '2025-03-10', '2025-03-10 14:22:00', 1, 129.99, 0.00, null, null,
    129.99, 65.00, 64.99, 50.0,
    'credit', 'in_store', 'Athletic Footwear', 'Running', 'Spring',
    false, null, 'SA-101-05', CURRENT_TIMESTAMP()),

    (5002, 10002, 8, 'TNF-JKT-001', 101, 'SportsPlex Flagship - Denver', 'CUST-1002',
    '2025-03-10', '2025-03-10 15:45:00', 1, 229.99, 46.00, 'markdown', null,
    183.99, 110.00, 73.99, 40.2,
    'credit', 'in_store', 'Athletic Apparel', 'Outdoor', 'Spring',
    false, null, 'SA-101-03', CURRENT_TIMESTAMP()),

    (5003, 10003, 6, 'UA-TOP-001', 101, 'SportsPlex Flagship - Denver', 'CUST-1003',
    '2025-03-09', '2025-03-09 11:15:00', 3, 24.99, 0.00, null, null,
    74.97, 30.00, 44.97, 60.0,
    'debit', 'in_store', 'Athletic Apparel', 'Training', 'Spring',
    false, null, 'SA-101-08', CURRENT_TIMESTAMP()),

    (5004, 10004, 23, 'GRM-WCH-001', 101, 'SportsPlex Flagship - Denver', 'CUST-1004',
    '2025-03-08', '2025-03-08 16:30:00', 1, 449.99, 0.00, null, null,
    449.99, 220.00, 229.99, 51.1,
    'credit', 'in_store', 'Accessories', 'Running', 'Spring',
    false, null, 'SA-101-02', CURRENT_TIMESTAMP()),

    (5005, 10005, 17, 'COL-TNT-001', 101, 'SportsPlex Flagship - Denver', 'CUST-1005',
    '2025-03-08', '2025-03-08 10:00:00', 1, 79.99, 0.00, null, null,
    79.99, 35.00, 44.99, 56.2,
    'cash', 'in_store', 'Outdoor & Camping', 'Camping', 'Spring',
    false, null, 'SA-101-10', CURRENT_TIMESTAMP()),

    -- Austin sales with promotion
    (5006, 10006, 1, 'NKE-RUN-001', 102, 'SportsPlex - Austin South', 'CUST-2001',
    '2025-03-09', '2025-03-09 13:00:00', 1, 129.99, 13.00, 'promotion', 'PROMO-SPRING-RUN',
    116.99, 65.00, 51.99, 44.4,
    'credit', 'in_store', 'Athletic Footwear', 'Running', 'Spring',
    false, null, 'SA-102-03', CURRENT_TIMESTAMP()),

    (5007, 10007, 24, 'HYD-BTL-001', 102, 'SportsPlex - Austin South', 'CUST-2002',
    '2025-03-09', '2025-03-09 15:20:00', 2, 44.95, 0.00, null, null,
    89.90, 36.00, 53.90, 60.0,
    'credit', 'online', 'Accessories', 'General', 'Spring',
    false, null, null, CURRENT_TIMESTAMP()),

    -- Chicago sales
    (5008, 10008, 3, 'NKE-BSK-001', 103, 'SportsPlex - Chicago North', 'CUST-3001',
    '2025-03-10', '2025-03-10 12:10:00', 1, 199.99, 0.00, null, null,
    199.99, 90.00, 109.99, 55.0,
    'credit', 'in_store', 'Athletic Footwear', 'Basketball', 'Spring',
    false, null, 'SA-103-06', CURRENT_TIMESTAMP()),

    (5009, 10009, 25, 'NKE-CLT-002', 103, 'SportsPlex - Chicago North', 'CUST-3002',
    '2025-03-09', '2025-03-09 17:45:00', 1, 64.99, 0.00, null, null,
    64.99, 26.00, 38.99, 60.0,
    'debit', 'in_store', 'Athletic Apparel', 'Lifestyle', 'Spring',
    false, null, 'SA-103-02', CURRENT_TIMESTAMP()),

    -- Orlando outlet sales (clearance)
    (5010, 10010, 5, 'NKE-CLT-001', 104, 'SportsPlex Outlet - Orlando', 'CUST-4001',
    '2025-03-10', '2025-03-10 11:30:00', 1, 89.99, 25.00, 'clearance', null,
    89.99, 48.00, 41.99, 46.7,
    'credit', 'in_store', 'Athletic Footwear', 'Lifestyle', 'Spring',
    false, null, 'SA-104-04', CURRENT_TIMESTAMP()),

    -- Portland sales
    (5011, 10011, 20, 'TRK-BKE-001', 105, 'SportsPlex - Portland', 'CUST-5001',
    '2025-03-08', '2025-03-08 14:00:00', 1, 1049.99, 0.00, null, null,
    1049.99, 520.00, 529.99, 50.5,
    'credit', 'in_store', 'Cycling', 'Cycling', 'Spring',
    false, null, 'SA-105-01', CURRENT_TIMESTAMP()),

    (5012, 10012, 19, 'OSP-BKP-001', 105, 'SportsPlex - Portland', 'CUST-5002',
    '2025-03-09', '2025-03-09 10:30:00', 1, 289.99, 0.00, null, null,
    289.99, 140.00, 149.99, 51.7,
    'credit', 'in_store', 'Outdoor & Camping', 'Hiking', 'Spring',
    false, null, 'SA-105-07', CURRENT_TIMESTAMP()),

    -- Return example
    (5013, 10013, 4, 'NB-TRN-001', 101, 'SportsPlex Flagship - Denver', 'CUST-1006',
    '2025-03-07', '2025-03-07 16:00:00', 1, 139.99, 0.00, null, null,
    -139.99, -62.00, -77.99, 0.0,
    'credit', 'in_store', 'Athletic Footwear', 'Training', 'Spring',
    true, 'Wrong size', 'SA-101-05', CURRENT_TIMESTAMP()),

    -- Online orders
    (5014, 10014, 30, 'LLM-LGN-001', 101, 'SportsPlex Flagship - Denver', 'CUST-1007',
    '2025-03-10', '2025-03-10 09:15:00', 1, 98.00, 0.00, null, null,
    98.00, 38.00, 60.00, 61.2,
    'credit', 'online', 'Athletic Apparel', 'Yoga', 'Spring',
    false, null, null, CURRENT_TIMESTAMP()),

    (5015, 10015, 18, 'YET-CLR-001', 101, 'SportsPlex Flagship - Denver', 'CUST-1008',
    '2025-03-07', '2025-03-07 11:45:00', 1, 324.99, 0.00, null, null,
    324.99, 160.00, 164.99, 50.8,
    'credit', 'in_store', 'Outdoor & Camping', 'Camping', 'Spring',
    false, null, 'SA-101-10', CURRENT_TIMESTAMP());
