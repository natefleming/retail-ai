USE IDENTIFIER(:database);

INSERT INTO purchase_orders (
    po_id, po_number, po_line_id, product_id, sku,
    supplier_id, supplier_name, destination_store_id, destination_warehouse,
    order_date, expected_delivery_date, actual_delivery_date,
    quantity_ordered, quantity_received, unit_cost, total_cost,
    po_status, season, buy_plan_id, department, sport_category,
    lead_time_days, is_reorder, buyer_id, buyer_name,
    approval_status, approved_by, notes, created_at, updated_at
) VALUES
    -- Spring 2025 Buy Plan: Running shoes restock
    (7001, 'PO-2025-SPR-001', 17001, 1, 'NKE-RUN-001',
    'SUP-NIKE-01', 'Nike Distribution Center', 102, 'WH-SOUTH',
    '2025-03-01', '2025-03-15', null,
    120, 0, 65.00, 7800.00,
    'confirmed', 'Spring', 'BP-2025-SPR', 'Athletic Footwear', 'Running',
    14, true, 'BUY-001', 'Lisa Chang',
    'approved', 'VP-MERCH-01', 'Restock for Austin South. High demand expected for spring running season.', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    (7002, 'PO-2025-SPR-002', 17002, 2, 'ADI-RUN-001',
    'SUP-ADI-01', 'Adidas Distribution Americas', 103, 'WH-CENTRAL',
    '2025-02-28', '2025-03-14', null,
    80, 0, 85.00, 6800.00,
    'in_transit', 'Spring', 'BP-2025-SPR', 'Athletic Footwear', 'Running',
    14, true, 'BUY-001', 'Lisa Chang',
    'approved', 'VP-MERCH-01', 'Critical restock for Chicago. Current stock at critical levels.', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    -- Spring/Summer 2025 Buy Plan: Camping seasonal build
    (7003, 'PO-2025-SS-001', 17003, 17, 'COL-TNT-001',
    'SUP-COL-01', 'Coleman / Newell Brands', null, 'WH-WEST',
    '2025-02-15', '2025-02-22', '2025-02-21',
    200, 200, 35.00, 7000.00,
    'received', 'Spring', 'BP-2025-SS', 'Outdoor & Camping', 'Camping',
    7, false, 'BUY-003', 'Ryan Torres',
    'approved', 'VP-MERCH-01', 'Initial spring camping build. Distribute to all stores with outdoor departments.', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    (7004, 'PO-2025-SS-002', 17004, 18, 'YET-CLR-001',
    'SUP-YET-01', 'YETI Holdings', null, 'WH-WEST',
    '2025-02-20', '2025-03-06', '2025-03-05',
    50, 48, 160.00, 7680.00,
    'received', 'Summer', 'BP-2025-SS', 'Outdoor & Camping', 'Camping',
    14, false, 'BUY-003', 'Ryan Torres',
    'approved', 'VP-MERCH-01', 'Summer cooler build. 2 units short on delivery - vendor credit issued.', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    (7005, 'PO-2025-SS-003', 17005, 19, 'OSP-BKP-001',
    'SUP-OSP-01', 'Osprey Packs', null, 'WH-WEST',
    '2025-02-20', '2025-03-06', '2025-03-04',
    40, 40, 140.00, 5600.00,
    'received', 'Spring', 'BP-2025-SS', 'Outdoor & Camping', 'Hiking',
    14, false, 'BUY-003', 'Ryan Torres',
    'approved', 'VP-MERCH-01', 'Spring hiking build. Focus allocation on Portland and Denver.', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    -- Cycling seasonal buy
    (7006, 'PO-2025-SS-004', 17006, 20, 'TRK-BKE-001',
    'SUP-TRK-01', 'Trek Bicycle Corporation', null, 'WH-WEST',
    '2025-01-15', '2025-02-15', '2025-02-18',
    15, 15, 520.00, 7800.00,
    'received', 'Spring', 'BP-2025-SS', 'Cycling', 'Cycling',
    30, false, 'BUY-004', 'Alex Kim',
    'approved', 'VP-MERCH-01', 'Spring cycling build. Allocate primarily to Portland (8) and Denver (5) stores.', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    -- Fall 2025 forward buy: Outerwear
    (7007, 'PO-2025-FW-001', 17007, 8, 'TNF-JKT-001',
    'SUP-TNF-01', 'The North Face / VF Corporation', null, 'WH-WEST',
    '2025-03-10', '2025-07-15', null,
    150, 0, 110.00, 16500.00,
    'confirmed', 'Fall', 'BP-2025-FW', 'Athletic Apparel', 'Outdoor',
    120, false, 'BUY-002', 'Maria Santos',
    'approved', 'VP-MERCH-01', 'Fall/Winter 2025 forward buy. ThermoBall Eco jackets with 20% sustainability premium. Delivery by mid-July for August floor set.', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    (7008, 'PO-2025-FW-002', 17008, 25, 'NKE-CLT-002',
    'SUP-NIKE-01', 'Nike Distribution Center', null, 'WH-CENTRAL',
    '2025-03-10', '2025-07-01', null,
    400, 0, 26.00, 10400.00,
    'submitted', 'Fall', 'BP-2025-FW', 'Athletic Apparel', 'Lifestyle',
    14, false, 'BUY-002', 'Maria Santos',
    'pending', null, 'Fall/Winter hoodie buy. High demand expected. Awaiting VP approval for increased quantity.', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    -- Golf seasonal buy
    (7009, 'PO-2025-SPR-003', 17009, 28, 'CLB-GLF-001',
    'SUP-CLB-01', 'Callaway Golf Company', null, 'WH-WEST',
    '2025-02-01', '2025-02-22', '2025-02-25',
    20, 20, 280.00, 5600.00,
    'received', 'Spring', 'BP-2025-SPR', 'Golf', 'Golf',
    21, false, 'BUY-005', 'James Wilson',
    'approved', 'VP-MERCH-01', 'Spring golf season launch. New Paradym Ai Smoke driver. Allocate to Denver flagship and Orlando.', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()),

    -- Pending/Draft PO
    (7010, 'PO-2025-SPR-004', 17010, 15, 'ROG-DUM-001',
    'SUP-ROG-01', 'Rogue Fitness', 103, 'WH-CENTRAL',
    '2025-03-12', '2025-03-26', null,
    5, 0, 420.00, 2100.00,
    'draft', 'Spring', 'BP-2025-SPR', 'Fitness Equipment', 'Fitness',
    14, true, 'BUY-006', 'Pat Henderson',
    'pending', null, 'Urgent restock for Chicago. Currently out of stock. New Year resolution demand exceeded forecast.', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP());
