USE IDENTIFIER(:database);

-- Employee Tasks table for retail store management
-- Supports the store companion app with task tracking, priorities, and assignments
CREATE TABLE IF NOT EXISTS employee_tasks (
    task_id STRING COMMENT 'Unique identifier for each task (UUID or sequential ID)',
    employee_id STRING COMMENT 'Employee ID who is assigned to complete the task',
    store_id STRING COMMENT 'Store location where the task should be performed',
    task_title STRING COMMENT 'Brief descriptive title of the task',
    task_description STRING COMMENT 'Detailed description of what needs to be accomplished',
    task_type STRING COMMENT 'Type of task: BOPIS, Service, Restock, Cleaning, Training, Administrative, Customer_Service, Inventory',
    task_category STRING COMMENT 'Category grouping: Operations, Customer_Service, Inventory_Management, Maintenance, Administrative',
    priority_level STRING COMMENT 'Task priority: Low, Medium, High, Critical, Urgent',
    task_status STRING COMMENT 'Current status: Pending, In_Progress, Completed, Cancelled, On_Hold, Overdue',
    
    -- Scheduling information
    assigned_date DATE COMMENT 'Date when the task was assigned',
    due_date DATE COMMENT 'Target completion date for the task',
    due_time TIMESTAMP COMMENT 'Specific time when task should be completed',
    estimated_duration_minutes INT COMMENT 'Estimated time to complete the task in minutes',
    actual_duration_minutes INT COMMENT 'Actual time taken to complete the task in minutes',
    
    -- Assignment details
    assigned_by STRING COMMENT 'Employee ID of the person who assigned the task',
    assigned_to STRING COMMENT 'Employee ID of the person responsible for completing the task',
    department STRING COMMENT 'Department where the task should be performed (e.g., Electronics, Grocery)',
    location_details STRING COMMENT 'Specific location within store (e.g., Floor 2, Electronics Section, Aisle 5)',
    
    -- Customer/Order related information (for BOPIS, Service tasks)
    customer_id STRING COMMENT 'Customer ID for customer-related tasks',
    customer_name STRING COMMENT 'Customer name for easy identification',
    order_id STRING COMMENT 'Internal order identifier for order-related tasks',
    order_number STRING COMMENT 'Customer-facing order number for reference',
    
    -- Product/Inventory related information (for Restock, Inventory tasks)
    product_sku STRING COMMENT 'Product SKU for inventory and restock tasks',
    product_name STRING COMMENT 'Product name for easy identification',
    quantity_required INT COMMENT 'Quantity of items needed for restock or inventory tasks',
    
    -- Task completion tracking
    started_at TIMESTAMP COMMENT 'Timestamp when employee started working on the task',
    completed_at TIMESTAMP COMMENT 'Timestamp when the task was marked as completed',
    notes STRING COMMENT 'Notes added during task execution for progress tracking',
    completion_notes STRING COMMENT 'Final notes added when task is completed',
    
    -- Recurring task information
    is_recurring BOOLEAN COMMENT 'Whether this task repeats on a schedule (true/false)',
    recurrence_pattern STRING COMMENT 'Recurrence frequency: Daily, Weekly, Monthly, Custom',
    parent_task_id STRING COMMENT 'Reference to parent task if this is a recurring instance',
    
    -- Performance and quality metrics
    quality_score DOUBLE COMMENT 'Quality rating for completed task (1.00 to 5.00 scale)',
    customer_satisfaction_score DOUBLE COMMENT 'Customer satisfaction rating if applicable (1.00 to 5.00 scale)',
    requires_manager_approval BOOLEAN COMMENT 'Whether task completion requires manager approval (true/false)',
    approved_by STRING COMMENT 'Manager employee ID who approved the task completion',
    approved_at TIMESTAMP COMMENT 'Timestamp when manager approved the task completion',
    
    -- System tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Timestamp when the task record was created',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP() COMMENT 'Timestamp when the task record was last modified',
    created_by STRING COMMENT 'Employee ID of who created the task record',
    updated_by STRING COMMENT 'Employee ID of who last updated the task record',
    
    -- Additional metadata (converted to JSON strings for parquet compatibility)
    tags ARRAY<STRING> COMMENT 'Tags for categorization and filtering (e.g., urgent, seasonal, training)',
    attachments STRING COMMENT 'File paths or URLs to related documents as JSON array',
    dependencies STRING COMMENT 'Task IDs that must be completed before this task can start as JSON array',
    
    -- Mobile app specific fields
    requires_photo_proof BOOLEAN COMMENT 'Whether task completion requires photo documentation (true/false)',
    photo_urls STRING COMMENT 'URLs to photos taken during or after task completion as JSON array',
    gps_location STRING COMMENT 'GPS coordinates where the task was completed for verification',
    device_id STRING COMMENT 'Mobile device identifier used to complete the task'
) 
USING DELTA
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true',
    'delta.feature.allowColumnDefaults' = 'supported'
);

-- Create views for common queries used by the store companion app

-- View for today's tasks by employee
CREATE OR REPLACE VIEW employee_daily_tasks AS
SELECT 
    task_id,
    employee_id,
    store_id,
    task_title,
    task_description,
    task_type,
    task_category,
    priority_level,
    task_status,
    due_time,
    estimated_duration_minutes,
    customer_name,
    order_number,
    product_name,
    location_details,
    department,
    notes,
    CASE 
        WHEN due_time IS NOT NULL AND due_time < CURRENT_TIMESTAMP() AND task_status IN ('Pending', 'In_Progress') 
        THEN TRUE 
        ELSE FALSE 
    END AS is_overdue,
    CASE 
        WHEN priority_level IN ('Critical', 'Urgent') THEN 1
        WHEN priority_level = 'High' THEN 2
        WHEN priority_level = 'Medium' THEN 3
        ELSE 4
    END AS priority_sort_order
FROM employee_tasks
WHERE assigned_date = CURRENT_DATE()
ORDER BY priority_sort_order, due_time;

-- View for task performance metrics
CREATE OR REPLACE VIEW task_performance_metrics AS
SELECT 
    assigned_to AS employee_id,
    store_id,
    assigned_date,
    COUNT(*) AS total_tasks,
    COUNT(CASE WHEN task_status = 'Completed' THEN 1 END) AS completed_tasks,
    COUNT(CASE WHEN task_status = 'Pending' THEN 1 END) AS pending_tasks,
    COUNT(CASE WHEN task_status = 'In_Progress' THEN 1 END) AS in_progress_tasks,
    COUNT(CASE WHEN task_status = 'Overdue' THEN 1 END) AS overdue_tasks,
    AVG(CASE WHEN quality_score IS NOT NULL THEN quality_score END) AS avg_quality_score,
    AVG(CASE WHEN customer_satisfaction_score IS NOT NULL THEN customer_satisfaction_score END) AS avg_customer_satisfaction,
    AVG(CASE WHEN actual_duration_minutes IS NOT NULL THEN actual_duration_minutes END) AS avg_completion_time_minutes,
    ROUND(
        COUNT(CASE WHEN task_status = 'Completed' THEN 1 END) * 100.0 / COUNT(*), 
        2
    ) AS completion_percentage
FROM employee_tasks
GROUP BY assigned_to, store_id, assigned_date;

-- View for manager task oversight
CREATE OR REPLACE VIEW manager_task_overview AS
SELECT 
    store_id,
    assigned_date,
    department,
    task_type,
    priority_level,
    COUNT(*) AS task_count,
    COUNT(CASE WHEN task_status = 'Completed' THEN 1 END) AS completed_count,
    COUNT(CASE WHEN task_status IN ('Pending', 'In_Progress') AND due_time < CURRENT_TIMESTAMP() THEN 1 END) AS overdue_count,
    COUNT(CASE WHEN priority_level IN ('Critical', 'Urgent') THEN 1 END) AS high_priority_count,
    AVG(CASE WHEN actual_duration_minutes IS NOT NULL THEN actual_duration_minutes END) AS avg_duration_minutes
FROM employee_tasks
WHERE assigned_date >= DATE_SUB(CURRENT_DATE(), 7)
GROUP BY store_id, assigned_date, department, task_type, priority_level
ORDER BY assigned_date DESC, overdue_count DESC, high_priority_count DESC; 