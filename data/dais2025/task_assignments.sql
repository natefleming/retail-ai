USE IDENTIFIER(:database);

-- Task assignments table for tracking task assignment requests
-- Supports workflow for assigning tasks to top employees through their managers
CREATE TABLE IF NOT EXISTS task_assignments (
    -- Assignment identification
    assignment_id STRING COMMENT 'Unique identifier for the task assignment',
    task_title STRING COMMENT 'Title or brief description of the task',
    task_description STRING COMMENT 'Detailed description of the task to be completed',
    task_type STRING COMMENT 'Type of task (routine, priority, emergency, project, training)',
    priority_level STRING COMMENT 'Priority level (low, medium, high, critical)',
    
    -- Assignment details
    assigned_to_employee_id STRING COMMENT 'Employee ID who the task is assigned to',
    assigned_to_employee_name STRING COMMENT 'Name of the employee assigned the task',
    assigned_by_manager_id STRING COMMENT 'Manager ID who made the assignment',
    assigned_by_manager_name STRING COMMENT 'Name of the manager who made the assignment',
    requested_by STRING COMMENT 'Who originally requested the task (system, user, customer)',
    
    -- Store and department context
    store_id STRING COMMENT 'Store where the task should be completed',
    store_name STRING COMMENT 'Name of the store location',
    department STRING COMMENT 'Department where the task belongs',
    
    -- Timing and scheduling
    created_at TIMESTAMP COMMENT 'When the task assignment was created',
    due_date TIMESTAMP COMMENT 'When the task should be completed',
    estimated_duration_minutes INT COMMENT 'Estimated time to complete the task in minutes',
    scheduled_start_time TIMESTAMP COMMENT 'When the employee should start the task',
    
    -- Status tracking
    assignment_status STRING COMMENT 'Current status (pending, approved, rejected, in_progress, completed, cancelled)',
    manager_approval_status STRING COMMENT 'Manager approval status (pending, approved, rejected)',
    manager_approval_timestamp TIMESTAMP COMMENT 'When manager approved/rejected the assignment',
    manager_approval_notes STRING COMMENT 'Manager notes about the approval/rejection',
    
    -- Employee response
    employee_acceptance_status STRING COMMENT 'Employee acceptance (pending, accepted, declined)',
    employee_acceptance_timestamp TIMESTAMP COMMENT 'When employee accepted/declined',
    employee_notes STRING COMMENT 'Employee notes about the task',
    
    -- Completion tracking
    started_at TIMESTAMP COMMENT 'When the employee started working on the task',
    completed_at TIMESTAMP COMMENT 'When the task was completed',
    actual_duration_minutes INT COMMENT 'Actual time spent on the task',
    completion_quality_score DOUBLE COMMENT 'Quality score for task completion (1.0-5.0)',
    completion_notes STRING COMMENT 'Notes about task completion',
    
    -- Communication tracking
    notification_method STRING COMMENT 'How the assignment was communicated (email, slack, teams, phone)',
    notification_sent_at TIMESTAMP COMMENT 'When the notification was sent',
    notification_delivered BOOLEAN COMMENT 'Whether the notification was successfully delivered',
    follow_up_required BOOLEAN COMMENT 'Whether follow-up is needed',
    follow_up_date TIMESTAMP COMMENT 'When to follow up if needed',
    
    -- Task context and reasoning
    selection_reason STRING COMMENT 'Why this employee was selected for the task',
    performance_score_at_assignment DOUBLE COMMENT 'Employee performance score when assigned',
    department_ranking_at_assignment INT COMMENT 'Employee department ranking when assigned',
    
    -- Metadata
    updated_at TIMESTAMP COMMENT 'Timestamp when record was last updated',
    created_by STRING COMMENT 'System or user who created the assignment',
    updated_by STRING COMMENT 'System or user who last updated the record'
)
USING DELTA
COMMENT 'Task assignment tracking table for managing task assignments to employees through managers'
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
);

-- Create view for active task assignments
CREATE OR REPLACE VIEW active_task_assignments AS
SELECT 
    assignment_id,
    task_title,
    task_description,
    task_type,
    priority_level,
    assigned_to_employee_id,
    assigned_to_employee_name,
    assigned_by_manager_id,
    assigned_by_manager_name,
    store_name,
    department,
    created_at,
    due_date,
    assignment_status,
    manager_approval_status,
    employee_acceptance_status,
    notification_method,
    selection_reason
FROM task_assignments
WHERE assignment_status IN ('pending', 'approved', 'in_progress')
ORDER BY priority_level DESC, created_at ASC;

-- Create view for manager task assignment dashboard
CREATE OR REPLACE VIEW manager_task_dashboard AS
SELECT 
    assigned_by_manager_id,
    assigned_by_manager_name,
    COUNT(*) as total_assignments,
    SUM(CASE WHEN assignment_status = 'pending' THEN 1 ELSE 0 END) as pending_assignments,
    SUM(CASE WHEN assignment_status = 'approved' THEN 1 ELSE 0 END) as approved_assignments,
    SUM(CASE WHEN assignment_status = 'completed' THEN 1 ELSE 0 END) as completed_assignments,
    SUM(CASE WHEN manager_approval_status = 'pending' THEN 1 ELSE 0 END) as pending_approvals,
    AVG(completion_quality_score) as avg_completion_quality,
    AVG(actual_duration_minutes) as avg_completion_time_minutes
FROM task_assignments
WHERE created_at >= DATE_SUB(CURRENT_DATE(), 30)  -- Last 30 days
GROUP BY assigned_by_manager_id, assigned_by_manager_name
ORDER BY total_assignments DESC; 