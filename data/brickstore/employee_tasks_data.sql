USE IDENTIFIER(:database);

-- Insert sample employee tasks data for the retail store companion app
-- This data supports the current day's tasks shown in the app interface

-- Clear existing data for today (for demo purposes)
DELETE FROM employee_tasks WHERE assigned_date = CURRENT_DATE();

-- Insert today's tasks for various employees across different stores
INSERT INTO employee_tasks (
    task_id, employee_id, store_id, task_title, task_description, task_type, task_category,
    priority_level, task_status, assigned_date, due_date, due_time, estimated_duration_minutes,
    assigned_by, assigned_to, department, location_details, customer_id, customer_name,
    order_id, order_number, product_sku, product_name, quantity_required, notes,
    is_recurring, requires_manager_approval, requires_photo_proof, tags, created_by
) VALUES

-- BOPIS (Buy Online Pick-up In Store) Tasks - High Priority
('TASK-2024-001', 'EMP-001', 1, 'BOPIS Order #B2024-0156', 'Customer order ready for pickup - Designer handbag and accessories', 'BOPIS', 'Customer_Service', 'High', 'Pending', CURRENT_DATE(), CURRENT_DATE(), '10:30:00', 15, 'MGR-001', 'EMP-001', 'Customer Service', 'Pickup Counter', 'CUST-12345', 'Sarah Johnson', 'ORD-B2024-0156', 'B2024-0156', 'BAG-DES-001', 'Designer Leather Handbag', 1, 'VIP customer - handle with care', FALSE, FALSE, TRUE, ARRAY('BOPIS', 'VIP', 'Designer'), 'SYSTEM'),

('TASK-2024-002', 'EMP-002', 1, 'BOPIS Order #B2024-0157', 'Electronics order - iPhone case and wireless charger', 'BOPIS', 'Customer_Service', 'Medium', 'In_Progress', CURRENT_DATE(), CURRENT_DATE(), '11:15:00', 10, 'MGR-001', 'EMP-002', 'Electronics', 'Electronics Counter', 'CUST-12346', 'Michael Chen', 'ORD-B2024-0157', 'B2024-0157', 'CASE-IPH-015', 'iPhone 15 Pro Case', 1, 'Customer called to confirm pickup time', FALSE, FALSE, TRUE, ARRAY('BOPIS', 'Electronics'), 'SYSTEM'),

-- Personal Shopping Services - High Priority
('TASK-2024-003', 'EMP-001', 1, 'Personal Shopping Appointment', 'VIP customer appointment for fall wardrobe consultation', 'Service', 'Customer_Service', 'High', 'Pending', CURRENT_DATE(), CURRENT_DATE(), '14:00:00', 90, 'MGR-001', 'EMP-001', 'Women\'s Fashion', 'Personal Shopping Suite', 'CUST-12347', 'Emma Rodriguez', NULL, NULL, NULL, NULL, NULL, 'Prepare fall collection samples, size 8-10', FALSE, FALSE, FALSE, ARRAY('Personal_Shopping', 'VIP', 'Fashion'), 'MGR-001'),

-- Inventory and Restocking Tasks
('TASK-2024-004', 'EMP-003', 1, 'Restock Designer Section', 'Replenish designer jeans and premium denim section', 'Restock', 'Inventory_Management', 'High', 'Pending', CURRENT_DATE(), CURRENT_DATE(), '12:00:00', 45, 'MGR-001', 'EMP-003', 'Women\'s Fashion', 'Floor 2 - Designer Section', NULL, NULL, NULL, NULL, 'JEAN-DES-001', 'Designer Skinny Jeans', 12, 'Critical stock level - only 2 remaining', FALSE, FALSE, TRUE, ARRAY('Restock', 'Critical', 'Designer'), 'MGR-001'),

('TASK-2024-005', 'EMP-004', 1, 'Inventory Count - Electronics', 'Cycle count for iPhone accessories and cases', 'Inventory', 'Inventory_Management', 'Medium', 'Pending', CURRENT_DATE(), CURRENT_DATE(), '15:30:00', 60, 'MGR-001', 'EMP-004', 'Electronics', 'Electronics Stockroom', NULL, NULL, NULL, NULL, 'CASE-IPH-ALL', 'iPhone Cases (All Models)', NULL, 'Weekly cycle count due', TRUE, FALSE, FALSE, ARRAY('Inventory', 'Cycle_Count', 'Electronics'), 'SYSTEM'),

-- Customer Service Tasks
('TASK-2024-006', 'EMP-002', 1, 'Customer Complaint Resolution', 'Follow up on return request for defective wireless headphones', 'Customer_Service', 'Customer_Service', 'Medium', 'Pending', CURRENT_DATE(), CURRENT_DATE(), '13:00:00', 30, 'MGR-001', 'EMP-002', 'Electronics', 'Customer Service Desk', 'CUST-12348', 'David Park', 'ORD-R2024-0089', 'R2024-0089', 'HEAD-WIR-003', 'Wireless Bluetooth Headphones', 1, 'Customer reports connectivity issues', FALSE, TRUE, FALSE, ARRAY('Customer_Service', 'Return', 'Electronics'), 'MGR-001'),

-- Maintenance and Cleaning Tasks
('TASK-2024-007', 'EMP-005', 1, 'Deep Clean Fitting Rooms', 'Sanitize and deep clean all fitting rooms in women\'s section', 'Cleaning', 'Maintenance', 'Medium', 'Completed', CURRENT_DATE(), CURRENT_DATE(), '09:00:00', 90, 'MGR-001', 'EMP-005', 'Women\'s Fashion', 'Fitting Rooms 1-8', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'Completed morning cleaning routine', TRUE, FALSE, TRUE, ARRAY('Cleaning', 'Fitting_Rooms', 'Daily'), 'SYSTEM'),

('TASK-2024-008', 'EMP-006', 1, 'Restock Checkout Supplies', 'Refill bags, receipt paper, and cleaning supplies at all registers', 'Restock', 'Operations', 'Low', 'Completed', CURRENT_DATE(), CURRENT_DATE(), '08:30:00', 20, 'MGR-001', 'EMP-006', 'Front End', 'All Register Stations', NULL, NULL, NULL, NULL, 'SUP-CHK-001', 'Checkout Supplies Kit', 5, 'Morning setup completed', TRUE, FALSE, FALSE, ARRAY('Restock', 'Checkout', 'Daily'), 'SYSTEM'),

-- Training and Development
('TASK-2024-009', 'EMP-007', 1, 'New Product Training', 'Complete online training module for new fall fashion collection', 'Training', 'Administrative', 'Medium', 'In_Progress', CURRENT_DATE(), CURRENT_DATE(), '16:00:00', 45, 'MGR-001', 'EMP-007', 'Women\'s Fashion', 'Break Room - Training Station', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'Module 3 of 5 completed', FALSE, FALSE, FALSE, ARRAY('Training', 'Product_Knowledge', 'Fashion'), 'HR-001'),

-- Security and Safety Tasks
('TASK-2024-010', 'EMP-008', 1, 'Security System Check', 'Test all security cameras and alarm systems', 'Administrative', 'Operations', 'High', 'Pending', CURRENT_DATE(), CURRENT_DATE(), '17:30:00', 30, 'MGR-001', 'EMP-008', 'Security', 'Security Office', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'Daily security protocol check', TRUE, TRUE, FALSE, ARRAY('Security', 'Daily_Check', 'Safety'), 'SEC-001'),

-- Visual Merchandising Tasks
('TASK-2024-011', 'EMP-009', 1, 'Update Window Display', 'Install new fall collection window display with seasonal decorations', 'Administrative', 'Operations', 'Medium', 'In_Progress', CURRENT_DATE(), CURRENT_DATE(), '11:00:00', 120, 'MGR-001', 'EMP-009', 'Visual Merchandising', 'Front Windows 1-3', NULL, NULL, NULL, NULL, 'DISP-FALL-001', 'Fall Display Package', 1, 'Window 1 completed, working on window 2', FALSE, TRUE, TRUE, ARRAY('Visual_Merchandising', 'Seasonal', 'Windows'), 'VM-001'),

-- Additional BOPIS orders for realistic volume
('TASK-2024-012', 'EMP-003', 1, 'BOPIS Order #B2024-0158', 'Men\'s clothing order - business shirts and ties', 'BOPIS', 'Customer_Service', 'Medium', 'Pending', CURRENT_DATE(), CURRENT_DATE(), '12:45:00', 15, 'MGR-001', 'EMP-003', 'Men\'s Fashion', 'Pickup Counter', 'CUST-12349', 'Robert Wilson', 'ORD-B2024-0158', 'B2024-0158', 'SHIRT-BUS-001', 'Business Dress Shirt', 2, 'Customer requested specific size confirmation', FALSE, FALSE, TRUE, ARRAY('BOPIS', 'Mens_Fashion'), 'SYSTEM'),

('TASK-2024-013', 'EMP-004', 1, 'BOPIS Order #B2024-0159', 'Footwear order - running shoes and athletic socks', 'BOPIS', 'Customer_Service', 'Low', 'Completed', CURRENT_DATE(), CURRENT_DATE(), '09:15:00', 12, 'MGR-001', 'EMP-004', 'Footwear', 'Pickup Counter', 'CUST-12350', 'Lisa Thompson', 'ORD-B2024-0159', 'B2024-0159', 'SHOE-RUN-001', 'Athletic Running Shoes', 1, 'Order picked up successfully', FALSE, FALSE, TRUE, ARRAY('BOPIS', 'Footwear', 'Completed'), 'SYSTEM'),

-- Urgent tasks that need immediate attention
('TASK-2024-014', 'EMP-001', 1, 'Spill Cleanup - Aisle 3', 'Customer spilled coffee in women\'s accessories section', 'Cleaning', 'Maintenance', 'Urgent', 'Completed', CURRENT_DATE(), CURRENT_DATE(), '10:15:00', 10, 'MGR-001', 'EMP-001', 'Women\'s Fashion', 'Aisle 3 - Accessories', NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'Spill cleaned and area secured', FALSE, FALSE, FALSE, ARRAY('Urgent', 'Spill', 'Safety'), 'MGR-001'),

('TASK-2024-015', 'EMP-002', 1, 'Price Check Request', 'Customer inquiry about promotional pricing on electronics bundle', 'Customer_Service', 'Customer_Service', 'Medium', 'Completed', CURRENT_DATE(), CURRENT_DATE(), '10:45:00', 8, 'MGR-001', 'EMP-002', 'Electronics', 'Electronics Floor', 'CUST-12351', 'Jennifer Adams', NULL, NULL, 'BUND-ELEC-001', 'Electronics Bundle Deal', 1, 'Price confirmed and customer assisted', FALSE, FALSE, FALSE, ARRAY('Price_Check', 'Customer_Service', 'Electronics'), 'EMP-002'),

-- Tasks for different stores (to show multi-store capability)
('TASK-2024-016', 'EMP-010', 2, 'BOPIS Order #B2024-0160', 'Jewelry order - engagement ring sizing appointment', 'BOPIS', 'Customer_Service', 'Critical', 'Pending', CURRENT_DATE(), CURRENT_DATE(), '14:30:00', 30, 'MGR-002', 'EMP-010', 'Jewelry', 'Jewelry Counter', 'CUST-12352', 'Mark Stevens', 'ORD-B2024-0160', 'B2024-0160', 'RING-ENG-001', 'Diamond Engagement Ring', 1, 'Special handling required - high value item', FALSE, TRUE, TRUE, ARRAY('BOPIS', 'Jewelry', 'High_Value', 'Critical'), 'SYSTEM'),

('TASK-2024-017', 'EMP-011', 3, 'Inventory Audit', 'Monthly inventory audit for luxury handbags section', 'Inventory', 'Inventory_Management', 'High', 'In_Progress', CURRENT_DATE(), CURRENT_DATE(), '13:30:00', 180, 'MGR-003', 'EMP-011', 'Luxury Goods', 'Luxury Handbags Section', NULL, NULL, NULL, NULL, 'BAG-LUX-ALL', 'Luxury Handbags (All)', NULL, 'Audit 60% complete', FALSE, TRUE, TRUE, ARRAY('Inventory', 'Audit', 'Luxury', 'Monthly'), 'MGR-003');

-- Update some tasks to show realistic completion times and quality scores
UPDATE employee_tasks 
SET 
    started_at = CURRENT_TIMESTAMP() - INTERVAL 2 HOURS,
    completed_at = CURRENT_TIMESTAMP() - INTERVAL 1 HOUR,
    actual_duration_minutes = 85,
    quality_score = 4.8,
    completion_notes = 'Completed ahead of schedule with excellent customer feedback'
WHERE task_id = 'TASK-2024-007';

UPDATE employee_tasks 
SET 
    started_at = CURRENT_TIMESTAMP() - INTERVAL 3 HOURS,
    completed_at = CURRENT_TIMESTAMP() - INTERVAL 2 HOURS 45 MINUTES,
    actual_duration_minutes = 18,
    quality_score = 4.5,
    completion_notes = 'All registers restocked and ready for business'
WHERE task_id = 'TASK-2024-008';

UPDATE employee_tasks 
SET 
    started_at = CURRENT_TIMESTAMP() - INTERVAL 1 HOUR,
    actual_duration_minutes = NULL,
    notes = 'Customer confirmed pickup time, preparing order'
WHERE task_id = 'TASK-2024-002';

UPDATE employee_tasks 
SET 
    started_at = CURRENT_TIMESTAMP() - INTERVAL 30 MINUTES,
    actual_duration_minutes = NULL,
    notes = 'Window 1 complete, working on window 2 design'
WHERE task_id = 'TASK-2024-011';

-- Insert some recurring daily tasks for tomorrow (to show recurring functionality)
INSERT INTO employee_tasks (
    task_id, employee_id, store_id, task_title, task_description, task_type, task_category,
    priority_level, task_status, assigned_date, due_date, due_time, estimated_duration_minutes,
    assigned_by, assigned_to, department, location_details, is_recurring, recurrence_pattern,
    parent_task_id, tags, created_by
) VALUES
('TASK-2024-R001', 'EMP-005', 1, 'Morning Store Opening Checklist', 'Complete opening procedures including lights, music, and safety check', 'Administrative', 'Operations', 'High', 'Pending', CURRENT_DATE() + INTERVAL 1 DAY, CURRENT_DATE() + INTERVAL 1 DAY, '08:00:00', 30, 'SYSTEM', 'EMP-005', 'Operations', 'Entire Store', TRUE, 'Daily', 'TASK-RECURRING-001', ARRAY('Opening', 'Daily', 'Checklist'), 'SYSTEM'),

('TASK-2024-R002', 'EMP-006', 1, 'Evening Store Closing Checklist', 'Complete closing procedures including registers, security, and cleaning', 'Administrative', 'Operations', 'High', 'Pending', CURRENT_DATE() + INTERVAL 1 DAY, CURRENT_DATE() + INTERVAL 1 DAY, '21:00:00', 45, 'SYSTEM', 'EMP-006', 'Operations', 'Entire Store', TRUE, 'Daily', 'TASK-RECURRING-002', ARRAY('Closing', 'Daily', 'Checklist'), 'SYSTEM'); 