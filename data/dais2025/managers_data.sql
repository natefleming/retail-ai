USE IDENTIFIER(:database);

-- Insert sample manager data
-- These managers correspond to the manager_ids referenced in employee_performance table

INSERT INTO managers (
    manager_id, manager_name, store_id, store_name, department, position_title,
    email_address, phone_number, slack_user_id, teams_user_id,
    reports_to, hire_date, management_start_date, employment_status,
    preferred_communication_method, time_zone, work_schedule, emergency_contact_only,
    max_daily_task_assignments, auto_approve_routine_tasks, requires_approval_for_high_priority,
    created_at, updated_at, created_by, updated_by
) VALUES

-- Store Managers
('MGR-001', 'Robert Chen', '101', 'Downtown Market', 'Store Operations', 'Store Manager',
'robert.chen@brickmart.com', '+1-415-555-0101', 'U01ROBERT', 'robert.chen@brickmart.onmicrosoft.com',
'MGR-DISTRICT', '2019-03-15', '2020-01-01', 'active',
'slack', 'America/Los_Angeles', 'Mon-Fri 7:00-16:00', false,
15, true, true,
CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('MGR-002', 'Lisa Martinez', '102', 'Marina Market', 'Store Operations', 'Store Manager',
'lisa.martinez@brickmart.com', '+1-415-555-0102', 'U02LISA', 'lisa.martinez@brickmart.onmicrosoft.com',
'MGR-DISTRICT', '2018-07-20', '2019-06-01', 'active',
'email', 'America/Los_Angeles', 'Mon-Fri 8:00-17:00', false,
12, true, true,
CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('MGR-003', 'David Kim', '103', 'Mission Market', 'Store Operations', 'Store Manager',
'david.kim@brickmart.com', '+1-415-555-0103', 'U03DAVID', 'david.kim@brickmart.onmicrosoft.com',
'MGR-DISTRICT', '2020-01-10', '2021-03-01', 'active',
'teams', 'America/Los_Angeles', 'Tue-Sat 9:00-18:00', false,
10, false, true,
CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

-- Department Managers (these are also employees who manage others)
('EMP-004', 'David Kim', '101', 'Downtown Market', 'Footwear', 'Department Lead',
'david.kim.footwear@brickmart.com', '+1-415-555-0104', 'U04DAVIDK', 'david.kim.footwear@brickmart.onmicrosoft.com',
'MGR-001', '2021-11-05', '2022-06-01', 'active',
'slack', 'America/Los_Angeles', 'Mon-Fri 10:00-19:00', false,
8, true, false,
CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-007', 'Emma Rodriguez', '101', 'Downtown Market', 'Customer Service', 'Customer Service Manager',
'emma.rodriguez@brickmart.com', '+1-415-555-0107', 'U07EMMA', 'emma.rodriguez@brickmart.onmicrosoft.com',
'MGR-001', '2020-09-15', '2021-12-01', 'active',
'email', 'America/Los_Angeles', 'Mon-Fri 8:00-17:00', false,
6, true, false,
CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-009', 'Ashley Martinez', '102', 'Marina Market', 'Electronics', 'Department Lead',
'ashley.martinez@brickmart.com', '+1-415-555-0109', 'U09ASHLEY', 'ashley.martinez@brickmart.onmicrosoft.com',
'MGR-002', '2021-05-20', '2022-08-01', 'active',
'slack', 'America/Los_Angeles', 'Tue-Sat 11:00-20:00', false,
7, true, false,
CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-011', 'Maria Gonzalez', '103', 'Mission Market', 'Customer Service', 'Customer Service Lead',
'maria.gonzalez@brickmart.com', '+1-415-555-0111', 'U11MARIA', 'maria.gonzalez@brickmart.onmicrosoft.com',
'MGR-003', '2021-12-01', '2023-01-01', 'active',
'teams', 'America/Los_Angeles', 'Wed-Sun 10:00-19:00', false,
5, false, false,
CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

('EMP-016', 'Isabella Rodriguez', '101', 'Downtown Market', 'Womens Fashion', 'Fashion Department Lead',
'isabella.rodriguez@brickmart.com', '+1-415-555-0116', 'U16ISABELLA', 'isabella.rodriguez@brickmart.onmicrosoft.com',
'MGR-001', '2020-08-12', '2022-01-01', 'active',
'slack', 'America/Los_Angeles', 'Mon-Fri 9:00-18:00', false,
8, true, false,
CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'),

-- District Manager
('MGR-DISTRICT', 'Jennifer Walsh', 'DISTRICT', 'San Francisco District', 'District Operations', 'District Manager',
'jennifer.walsh@brickmart.com', '+1-415-555-0001', 'U00JENNIFER', 'jennifer.walsh@brickmart.onmicrosoft.com',
'MGR-REGIONAL', '2017-01-15', '2018-01-01', 'active',
'email', 'America/Los_Angeles', 'Mon-Fri 8:00-17:00', true,
5, false, true,
CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP(), 'system', 'system'); 