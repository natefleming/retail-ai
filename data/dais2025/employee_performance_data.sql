USE IDENTIFIER(:database);

-- Insert sample employee performance data for current month
-- This data supports identifying top employees by department

INSERT INTO employee_performance (
    employee_id, employee_name, store_id, store_name, department, position_title,
    manager_id, hire_date, employment_status, performance_period, period_start_date, period_end_date,
    total_sales_amount, total_transactions, average_transaction_value, sales_target, sales_achievement_percentage,
    upsell_cross_sell_revenue, total_tasks_assigned, total_tasks_completed, task_completion_rate,
    average_task_completion_time_minutes, overdue_tasks, high_priority_tasks_completed,
    average_quality_score, customer_satisfaction_score, customer_complaints, customer_compliments,
    mystery_shopper_score, scheduled_hours, actual_hours_worked, attendance_rate, punctuality_score,
    sick_days_taken, vacation_days_taken, training_hours_completed, certifications_earned,
    training_completion_rate, skill_assessment_score, peer_review_score, mentoring_hours,
    team_projects_participated, leadership_activities, bopis_orders_processed, bopis_processing_time_minutes,
    personal_shopping_sessions, product_knowledge_score, employee_of_month_awards, performance_bonuses_earned,
    recognition_points, peer_nominations, overall_performance_score, performance_ranking_in_department,
    performance_ranking_in_store, performance_trend, goals_set, goals_achieved, development_plan_progress,
    next_promotion_readiness, last_performance_review_date, next_performance_review_date,
    created_at, updated_at, created_by, updated_by
) VALUES

-- Electronics Department - Downtown Market
('EMP-001', 'Sarah Chen', '101', 'Downtown Market', 'Electronics', 'Senior Sales Associate',
'MGR-001', '2022-03-15', 'active', 'monthly', '2024-12-01', '2024-12-31',
45250.00, 185, 244.59, 40000.00, 113.13, 8500.00, 42, 40, 95.24, 18, 2, 15,
4.7, 4.8, 1, 12, 4.6, 160.0, 158.5, 99.06, 4.8, 1, 0, 8.0, 2, 100.0, 4.5,
4.6, 4.0, 3, 2, 28, 12, 5, 4.7, 1, 750.00, 95, 8, 4.75, 1, 2, 'improving',
5, 5, 85.0, 4.2, '2024-06-15', '2025-06-15', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-002', 'Michael Rodriguez', '101', 'Downtown Market', 'Electronics', 'Sales Associate',
'MGR-001', '2023-01-20', 'active', 'monthly', '2024-12-01', '2024-12-31',
38900.00, 165, 235.76, 35000.00, 111.14, 6200.00, 38, 35, 92.11, 22, 3, 12,
4.4, 4.5, 2, 8, 4.3, 160.0, 155.0, 96.88, 4.5, 2, 0, 6.0, 1, 85.0, 4.2,
4.3, 2.0, 2, 1, 22, 15, 3, 4.4, 0, 500.00, 78, 5, 4.35, 2, 4, 'stable',
4, 3, 75.0, 3.8, '2024-07-20', '2025-07-20', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-003', 'Jennifer Park', '101', 'Downtown Market', 'Electronics', 'Sales Associate',
'MGR-001', '2023-08-10', 'active', 'monthly', '2024-12-01', '2024-12-31',
32100.00, 142, 226.06, 35000.00, 91.71, 4800.00, 35, 32, 91.43, 25, 3, 8,
4.1, 4.2, 3, 6, 4.0, 160.0, 152.0, 95.00, 4.2, 3, 0, 4.0, 0, 70.0, 3.9,
4.0, 1.0, 1, 0, 18, 18, 2, 4.1, 0, 300.00, 65, 3, 4.05, 3, 6, 'declining',
4, 2, 50.0, 3.5, '2024-08-10', '2025-08-10', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

-- Footwear Department - Downtown Market
('EMP-004', 'David Kim', '101', 'Downtown Market', 'Footwear', 'Department Lead',
'MGR-001', '2021-11-05', 'active', 'monthly', '2024-12-01', '2024-12-31',
52800.00, 198, 266.67, 45000.00, 117.33, 9800.00, 45, 44, 97.78, 16, 1, 18,
4.8, 4.9, 0, 15, 4.8, 160.0, 160.0, 100.00, 4.9, 0, 0, 10.0, 3, 100.0, 4.8,
4.8, 6.0, 4, 3, 35, 10, 8, 4.9, 2, 1200.00, 120, 12, 4.85, 1, 1, 'improving',
6, 6, 95.0, 4.6, '2024-05-05', '2025-05-05', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-005', 'Lisa Thompson', '101', 'Downtown Market', 'Footwear', 'Senior Sales Associate',
'EMP-004', '2022-07-12', 'active', 'monthly', '2024-12-01', '2024-12-31',
41200.00, 172, 239.53, 38000.00, 108.42, 7100.00, 40, 38, 95.00, 20, 2, 14,
4.5, 4.6, 1, 10, 4.4, 160.0, 157.0, 98.13, 4.6, 1, 0, 7.0, 2, 90.0, 4.4,
4.5, 3.0, 3, 2, 30, 13, 6, 4.6, 1, 800.00, 88, 7, 4.55, 2, 3, 'stable',
5, 4, 80.0, 4.0, '2024-07-12', '2025-07-12', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-006', 'Robert Wilson', '101', 'Downtown Market', 'Footwear', 'Sales Associate',
'EMP-004', '2023-04-03', 'active', 'monthly', '2024-12-01', '2024-12-31',
35600.00, 148, 240.54, 35000.00, 101.71, 5400.00, 36, 33, 91.67, 24, 3, 10,
4.2, 4.3, 2, 7, 4.1, 160.0, 154.0, 96.25, 4.3, 2, 0, 5.0, 1, 75.0, 4.1,
4.2, 2.0, 2, 1, 25, 16, 4, 4.3, 0, 450.00, 72, 4, 4.25, 3, 5, 'stable',
4, 3, 65.0, 3.7, '2024-10-03', '2025-10-03', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

-- Customer Service Department - Downtown Market
('EMP-007', 'Emma Rodriguez', '101', 'Downtown Market', 'Customer Service', 'Customer Service Manager',
'MGR-001', '2020-09-15', 'active', 'monthly', '2024-12-01', '2024-12-31',
15200.00, 85, 178.82, 12000.00, 126.67, 2800.00, 50, 48, 96.00, 15, 2, 20,
4.9, 4.9, 0, 18, 4.9, 160.0, 160.0, 100.00, 4.9, 0, 0, 12.0, 4, 100.0, 4.9,
4.9, 8.0, 5, 4, 45, 8, 12, 4.8, 3, 1500.00, 145, 15, 4.90, 1, 1, 'improving',
6, 6, 100.0, 4.8, '2024-03-15', '2025-03-15', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-008', 'James Anderson', '101', 'Downtown Market', 'Customer Service', 'Senior Customer Service Rep',
'EMP-007', '2022-01-10', 'active', 'monthly', '2024-12-01', '2024-12-31',
8900.00, 62, 143.55, 8000.00, 111.25, 1600.00, 44, 42, 95.45, 18, 2, 16,
4.6, 4.7, 1, 14, 4.5, 160.0, 158.0, 98.75, 4.7, 1, 0, 8.0, 2, 95.0, 4.5,
4.6, 4.0, 3, 2, 38, 9, 10, 4.6, 1, 900.00, 98, 9, 4.65, 2, 2, 'stable',
5, 5, 85.0, 4.2, '2024-07-10', '2025-07-10', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

-- Marina Market Store
('EMP-009', 'Ashley Martinez', '102', 'Marina Market', 'Electronics', 'Department Lead',
'MGR-002', '2021-05-20', 'active', 'monthly', '2024-12-01', '2024-12-31',
48900.00, 192, 254.69, 42000.00, 116.43, 8900.00, 43, 41, 95.35, 17, 2, 16,
4.7, 4.8, 1, 13, 4.6, 160.0, 159.0, 99.38, 4.8, 1, 0, 9.0, 3, 95.0, 4.6,
4.7, 5.0, 4, 3, 32, 11, 7, 4.7, 2, 1000.00, 105, 10, 4.70, 1, 2, 'improving',
5, 5, 90.0, 4.4, '2024-05-20', '2025-05-20', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-010', 'Kevin Chang', '102', 'Marina Market', 'Footwear', 'Senior Sales Associate',
'MGR-002', '2022-09-08', 'active', 'monthly', '2024-12-01', '2024-12-31',
43800.00, 178, 246.07, 40000.00, 109.50, 7500.00, 41, 39, 95.12, 19, 2, 15,
4.6, 4.7, 1, 11, 4.5, 160.0, 157.5, 98.44, 4.7, 1, 0, 7.5, 2, 88.0, 4.5,
4.6, 3.5, 3, 2, 29, 12, 6, 4.6, 1, 850.00, 92, 8, 4.60, 2, 3, 'stable',
5, 4, 82.0, 4.1, '2024-09-08', '2025-09-08', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

-- Mission Market Store
('EMP-011', 'Maria Gonzalez', '103', 'Mission Market', 'Customer Service', 'Customer Service Lead',
'MGR-003', '2021-12-01', 'active', 'monthly', '2024-12-01', '2024-12-31',
12800.00, 78, 164.10, 11000.00, 116.36, 2400.00, 47, 45, 95.74, 16, 2, 18,
4.8, 4.8, 0, 16, 4.7, 160.0, 159.0, 99.38, 4.8, 1, 0, 10.0, 3, 98.0, 4.7,
4.8, 6.0, 4, 3, 42, 9, 11, 4.7, 2, 1100.00, 115, 11, 4.75, 1, 2, 'improving',
6, 6, 92.0, 4.5, '2024-06-01', '2025-06-01', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-012', 'Daniel Lee', '103', 'Mission Market', 'Electronics', 'Sales Associate',
'MGR-003', '2023-02-14', 'active', 'monthly', '2024-12-01', '2024-12-31',
36200.00, 155, 233.55, 34000.00, 106.47, 5800.00, 37, 34, 91.89, 23, 3, 11,
4.3, 4.4, 2, 9, 4.2, 160.0, 153.0, 95.63, 4.4, 2, 0, 6.0, 1, 80.0, 4.2,
4.3, 2.5, 2, 1, 24, 14, 4, 4.3, 0, 600.00, 75, 6, 4.30, 3, 4, 'stable',
4, 3, 70.0, 3.9, '2024-08-14', '2025-08-14', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-013', 'Rachel Kim', '103', 'Mission Market', 'Footwear', 'Sales Associate',
'MGR-003', '2023-06-22', 'active', 'monthly', '2024-12-01', '2024-12-31',
38700.00, 162, 238.89, 36000.00, 107.50, 6400.00, 39, 36, 92.31, 21, 3, 13,
4.4, 4.5, 2, 10, 4.3, 160.0, 155.5, 97.19, 4.5, 2, 0, 6.5, 1, 82.0, 4.3,
4.4, 2.0, 2, 1, 27, 13, 5, 4.4, 0, 650.00, 80, 7, 4.40, 3, 4, 'stable',
4, 3, 72.0, 3.8, '2024-12-22', '2025-12-22', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

-- Additional high performers for comparison
('EMP-014', 'Alex Johnson', '101', 'Downtown Market', 'Electronics', 'Sales Associate',
'MGR-001', '2023-03-01', 'active', 'monthly', '2024-12-01', '2024-12-31',
42100.00, 175, 240.57, 38000.00, 110.79, 7200.00, 40, 38, 95.00, 19, 2, 14,
4.6, 4.7, 1, 11, 4.5, 160.0, 158.0, 98.75, 4.7, 1, 0, 7.0, 2, 90.0, 4.5,
4.6, 3.0, 3, 2, 26, 13, 5, 4.6, 1, 750.00, 85, 8, 4.60, 2, 3, 'improving',
5, 4, 80.0, 4.1, '2024-09-01', '2025-09-01', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-015', 'Sophia Davis', '102', 'Marina Market', 'Customer Service', 'Customer Service Rep',
'MGR-002', '2023-07-15', 'active', 'monthly', '2024-12-01', '2024-12-31',
9800.00, 68, 144.12, 9000.00, 108.89, 1800.00, 42, 40, 95.24, 17, 2, 15,
4.5, 4.6, 1, 12, 4.4, 160.0, 157.0, 98.13, 4.6, 1, 0, 6.0, 1, 85.0, 4.4,
4.5, 2.0, 2, 1, 35, 10, 8, 4.5, 0, 700.00, 82, 7, 4.50, 3, 4, 'stable',
4, 3, 75.0, 3.9, '2024-07-15', '2025-07-15', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

-- Women's Fashion Department - Downtown Market
('EMP-016', 'Isabella Rodriguez', '101', 'Downtown Market', 'Womens Fashion', 'Fashion Department Lead',
'MGR-001', '2020-08-12', 'active', 'monthly', '2024-12-01', '2024-12-31',
58200.00, 215, 270.70, 50000.00, 116.40, 11800.00, 48, 47, 97.92, 14, 1, 20,
4.9, 4.9, 0, 22, 4.8, 160.0, 160.0, 100.00, 4.9, 0, 0, 12.0, 4, 100.0, 4.9,
4.9, 8.0, 5, 4, 48, 8, 18, 4.9, 3, 1800.00, 155, 18, 4.90, 1, 1, 'improving',
6, 6, 98.0, 4.8, '2024-02-12', '2025-02-12', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-017', 'Olivia Chen', '101', 'Downtown Market', 'Womens Fashion', 'Senior Style Consultant',
'EMP-016', '2021-10-05', 'active', 'monthly', '2024-12-01', '2024-12-31',
46800.00, 188, 248.94, 42000.00, 111.43, 9200.00, 44, 42, 95.45, 16, 2, 18,
4.8, 4.8, 0, 19, 4.7, 160.0, 159.0, 99.38, 4.8, 1, 0, 10.0, 3, 95.0, 4.7,
4.8, 6.0, 4, 3, 42, 9, 16, 4.8, 2, 1400.00, 135, 15, 4.80, 1, 2, 'improving',
5, 5, 92.0, 4.6, '2024-04-05', '2025-04-05', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-018', 'Grace Williams', '101', 'Downtown Market', 'Womens Fashion', 'Personal Stylist',
'EMP-016', '2022-04-18', 'active', 'monthly', '2024-12-01', '2024-12-31',
41500.00, 165, 251.52, 38000.00, 109.21, 8100.00, 40, 38, 95.00, 18, 2, 16,
4.7, 4.8, 1, 17, 4.6, 160.0, 158.0, 98.75, 4.8, 1, 0, 8.0, 2, 90.0, 4.6,
4.7, 5.0, 3, 2, 38, 10, 22, 4.7, 1, 1200.00, 125, 14, 4.75, 2, 2, 'stable',
5, 4, 88.0, 4.4, '2024-10-18', '2025-10-18', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-019', 'Zoe Martinez', '101', 'Downtown Market', 'Womens Fashion', 'Sales Associate',
'EMP-016', '2023-01-30', 'active', 'monthly', '2024-12-01', '2024-12-31',
37200.00, 152, 244.74, 35000.00, 106.29, 6800.00, 38, 35, 92.11, 20, 3, 12,
4.5, 4.6, 2, 14, 4.4, 160.0, 156.0, 97.50, 4.6, 2, 0, 6.0, 1, 85.0, 4.4,
4.5, 3.0, 2, 1, 28, 12, 12, 4.5, 0, 800.00, 95, 9, 4.50, 2, 3, 'improving',
4, 3, 78.0, 4.0, '2024-07-30', '2025-07-30', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

-- Women's Fashion Department - Marina Market
('EMP-020', 'Ava Thompson', '102', 'Marina Market', 'Womens Fashion', 'Senior Style Consultant',
'MGR-002', '2021-06-14', 'active', 'monthly', '2024-12-01', '2024-12-31',
44900.00, 178, 252.25, 40000.00, 112.25, 8600.00, 42, 40, 95.24, 17, 2, 17,
4.7, 4.8, 1, 16, 4.6, 160.0, 158.5, 99.06, 4.8, 1, 0, 9.0, 2, 92.0, 4.6,
4.7, 4.5, 3, 2, 36, 10, 15, 4.7, 1, 1100.00, 118, 12, 4.70, 1, 2, 'stable',
5, 4, 85.0, 4.3, '2024-06-14', '2025-06-14', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-021', 'Mia Johnson', '102', 'Marina Market', 'Womens Fashion', 'Personal Stylist',
'MGR-002', '2022-11-22', 'active', 'monthly', '2024-12-01', '2024-12-31',
39800.00, 158, 251.90, 36000.00, 110.56, 7400.00, 39, 37, 94.87, 19, 2, 15,
4.6, 4.7, 1, 15, 4.5, 160.0, 157.0, 98.13, 4.7, 1, 0, 7.5, 2, 88.0, 4.5,
4.6, 4.0, 3, 2, 34, 11, 19, 4.6, 1, 950.00, 108, 11, 4.60, 2, 3, 'improving',
5, 4, 82.0, 4.2, '2024-11-22', '2025-11-22', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

-- Women's Fashion Department - Mission Market
('EMP-022', 'Chloe Garcia', '103', 'Mission Market', 'Womens Fashion', 'Style Consultant',
'MGR-003', '2022-02-28', 'active', 'monthly', '2024-12-01', '2024-12-31',
42100.00, 168, 250.60, 38000.00, 110.79, 7900.00, 41, 39, 95.12, 18, 2, 16,
4.6, 4.7, 1, 16, 4.5, 160.0, 158.0, 98.75, 4.7, 1, 0, 8.0, 2, 90.0, 4.5,
4.6, 4.0, 3, 2, 32, 11, 14, 4.6, 1, 1000.00, 112, 12, 4.65, 2, 2, 'stable',
5, 4, 84.0, 4.2, '2024-08-28', '2025-08-28', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-023', 'Harper Lee', '103', 'Mission Market', 'Womens Fashion', 'Sales Associate',
'MGR-003', '2023-05-16', 'active', 'monthly', '2024-12-01', '2024-12-31',
35900.00, 145, 247.59, 34000.00, 105.59, 6200.00, 37, 34, 91.89, 21, 3, 13,
4.4, 4.5, 2, 12, 4.3, 160.0, 155.0, 96.88, 4.5, 2, 0, 6.0, 1, 80.0, 4.3,
4.4, 2.5, 2, 1, 26, 13, 10, 4.4, 0, 700.00, 88, 8, 4.40, 3, 4, 'improving',
4, 3, 75.0, 3.9, '2024-11-16', '2025-11-16', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'); 