USE IDENTIFIER(:database);

-- Managers table for storing manager information and contact details
-- Supports task assignment and communication workflows
CREATE TABLE IF NOT EXISTS managers (
    -- Manager identification
    manager_id STRING COMMENT 'Unique identifier for the manager (matches employee_id for managers)',
    manager_name STRING COMMENT 'Full name of the manager',
    store_id STRING COMMENT 'Store where the manager works',
    store_name STRING COMMENT 'Name of the store location',
    department STRING COMMENT 'Primary department the manager oversees',
    position_title STRING COMMENT 'Manager position title (e.g., Store Manager, Department Manager)',
    
    -- Contact information
    email_address STRING COMMENT 'Manager email address for task assignments and communication',
    phone_number STRING COMMENT 'Manager phone number',
    slack_user_id STRING COMMENT 'Slack user ID for direct messaging',
    teams_user_id STRING COMMENT 'Microsoft Teams user ID',
    
    -- Management details
    reports_to STRING COMMENT 'Manager ID of who this manager reports to',
    hire_date DATE COMMENT 'Date when manager was hired',
    management_start_date DATE COMMENT 'Date when manager started in management role',
    employment_status STRING COMMENT 'Current employment status (active, inactive, terminated)',
    
    -- Availability and preferences
    preferred_communication_method STRING COMMENT 'Preferred method for task assignments (email, slack, teams, phone)',
    time_zone STRING COMMENT 'Manager time zone for scheduling',
    work_schedule STRING COMMENT 'Typical work schedule (e.g., Mon-Fri 9-5, Rotating shifts)',
    emergency_contact_only BOOLEAN COMMENT 'Whether to only contact for emergency tasks',
    
    -- Task assignment preferences
    max_daily_task_assignments INT COMMENT 'Maximum number of task assignments per day',
    auto_approve_routine_tasks BOOLEAN COMMENT 'Whether routine tasks can be auto-assigned',
    requires_approval_for_high_priority BOOLEAN COMMENT 'Whether high priority tasks need explicit approval',
    
    -- Metadata
    created_at TIMESTAMP COMMENT 'Timestamp when record was created',
    updated_at TIMESTAMP COMMENT 'Timestamp when record was last updated',
    created_by STRING COMMENT 'System or user who created the record',
    updated_by STRING COMMENT 'System or user who last updated the record'
)
USING DELTA
COMMENT 'Manager information table for task assignment and communication workflows'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
);

-- Create view for manager lookup with employee information
CREATE OR REPLACE VIEW manager_employee_lookup AS
SELECT 
    m.manager_id,
    m.manager_name,
    m.store_id,
    m.store_name,
    m.department as manager_department,
    m.position_title as manager_title,
    m.email_address,
    m.phone_number,
    m.slack_user_id,
    m.teams_user_id,
    m.preferred_communication_method,
    m.time_zone,
    m.work_schedule,
    m.max_daily_task_assignments,
    m.auto_approve_routine_tasks,
    m.requires_approval_for_high_priority,
    -- Employee information
    e.employee_id,
    e.employee_name,
    e.department as employee_department,
    e.position_title as employee_title,
    e.overall_performance_score,
    e.performance_ranking_in_department
FROM managers m
JOIN employee_performance e ON m.manager_id = e.manager_id
WHERE e.performance_period = 'monthly' 
    AND e.period_start_date = DATE_TRUNC('month', CURRENT_DATE())
    AND e.employment_status = 'active'
    AND m.employment_status = 'active'; 