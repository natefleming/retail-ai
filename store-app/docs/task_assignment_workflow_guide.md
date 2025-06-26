# Task Assignment Workflow Guide

This guide explains how to use the task assignment workflow tools to automatically assign tasks to top performing employees through their managers, with mock notification functionality.

## Overview

The task assignment workflow enables AI agents to:
1. **Identify Top Performers**: Find the best employees in specific departments
2. **Locate Managers**: Identify manager contact information and preferences
3. **Create Task Assignments**: Generate structured task assignments with tracking
4. **Send Notifications**: Mock notification system for manager communication
5. **Track Progress**: Monitor assignment status and completion

## Complete Workflow

### Step 1: Find Top Employee
```python
# Find the best employee in a department
top_employees = find_top_employees_by_department.invoke({
    "department": "Customer Service",
    "limit": 1
})

best_employee = top_employees[0]
# Returns: Emma Rodriguez (EMP-007) - Performance Score: 4.90, Rank #1
```

### Step 2: Identify Manager
```python
# Find the employee's manager
manager_info = find_employee_manager.invoke({
    "employee_id": best_employee["employee_id"]
})

manager = manager_info[0]
# Returns: Robert Chen (MGR-001) - robert.chen@brickmart.com, Slack: U01ROBERT
```

### Step 3: Assign Task
```python
# Create task assignment through manager
assignment_result = assign_task_to_employee.invoke({
    "employee_id": "EMP-007",
    "task_title": "Handle VIP Customer Complaint Resolution",
    "task_description": "Priority customer complaint needs immediate attention...",
    "task_type": "priority",
    "priority_level": "high",
    "due_hours": 2,
    "estimated_duration_minutes": 90
})

# Returns assignment ID, manager notification details, and tracking info
```

## New Tools Available

### 1. `find_employee_manager`

**Purpose**: Find manager information for a specific employee.

**Parameters**:
- `employee_id` (str): Employee ID to find the manager for

**Returns**: Manager information including:
- Manager ID, name, title, contact details
- Communication preferences (email, Slack, Teams, phone)
- Work schedule and availability
- Task assignment preferences and limits
- Employee performance context

**Example**:
```python
manager_info = find_employee_manager.invoke({
    "employee_id": "EMP-007"
})
```

### 2. `assign_task_to_employee`

**Purpose**: Assign a task to an employee through their manager with notification.

**Parameters**:
- `employee_id` (str): Employee ID to assign the task to
- `task_title` (str): Brief title for the task
- `task_description` (str): Detailed description of what needs to be done
- `task_type` (str): Type of task (routine, priority, emergency, project, training)
- `priority_level` (str): Priority level (low, medium, high, critical)
- `due_hours` (int): Hours from now when task should be completed (default: 24)
- `estimated_duration_minutes` (int): Estimated time to complete in minutes (default: 60)

**Returns**: Assignment result including:
- Assignment ID and status
- Manager and employee information
- Task details and timeline
- Notification delivery status
- Human-readable result message

**Example**:
```python
result = assign_task_to_employee.invoke({
    "employee_id": "EMP-007",
    "task_title": "VIP Customer Service",
    "task_description": "Handle premium customer complaint with urgency",
    "task_type": "priority",
    "priority_level": "high",
    "due_hours": 4,
    "estimated_duration_minutes": 120
})
```

### 3. `task_extraction`

**Purpose**: Extract task information from natural language text using LLM.

**Parameters**:
- `input` (str): Natural language text describing a task to be assigned

**Returns**: Dictionary with extracted task information:
- `task_title`: Brief title for the task
- `task_description`: Detailed description
- `task_type`: Type classification
- `priority_level`: Priority assessment
- `due_hours`: Timing requirements
- `estimated_duration_minutes`: Duration estimate

**Example**:
```python
task_info = task_extraction.invoke({
    "input": "We need someone to urgently handle a VIP customer complaint about their recent purchase. This is high priority and needs to be resolved within 2 hours."
})
# Returns structured task information extracted from natural language
```

## Database Schema

### New Tables Created

#### `managers` Table
Stores manager information and contact preferences:
- Manager identification and contact details
- Communication preferences (email, Slack, Teams, phone)
- Work schedules and availability
- Task assignment preferences and limits

#### `task_assignments` Table
Tracks all task assignment requests and their status:
- Assignment identification and details
- Employee and manager information
- Task specifications and timing
- Status tracking (pending, approved, in_progress, completed)
- Communication and notification logs
- Performance context and selection reasoning

### Views Created

#### `manager_employee_lookup`
Joins managers with their employees for easy lookup:
```sql
SELECT manager_info.*, employee_performance.*
FROM managers m
JOIN employee_performance e ON m.manager_id = e.manager_id
```

#### `active_task_assignments`
Shows current active assignments:
```sql
SELECT * FROM task_assignments 
WHERE assignment_status IN ('pending', 'approved', 'in_progress')
```

#### `manager_task_dashboard`
Manager performance dashboard:
```sql
SELECT manager_id, total_assignments, pending_approvals, 
       avg_completion_quality, avg_completion_time
FROM task_assignments
GROUP BY manager_id
```

## Mock Notification System

The system includes a mock notification framework that simulates real-world communication:

### Supported Communication Methods
- **Email**: Sends to manager's email address
- **Slack**: Messages manager's Slack user ID
- **Teams**: Notifies via Microsoft Teams
- **Phone**: Simulates phone notification

### Notification Content
Each notification includes:
- Task title and description
- Priority level and due date
- Employee selection reasoning
- Assignment ID for tracking
- Manager approval request

### Example Notification (Slack)
```
Hi Robert Chen,

New task assignment request for Emma Rodriguez:

📋 Task: Handle VIP Customer Complaint Resolution
🔥 Priority: HIGH
⏰ Due: 2 hours
👤 Employee: Emma Rodriguez (Customer Service Manager)
🏆 Reason: Selected as top performer in Customer Service (rank #1, score: 4.90)

Assignment ID: TASK-A1B2C3D4

Please review and approve this assignment.
```

## Integration Examples

### Scenario 1: Emergency Task Assignment
```python
# 1. Find top performer in relevant department
top_employee = find_top_employees_by_department.invoke({
    "department": "Electronics", 
    "limit": 1
})[0]

# 2. Assign critical task
result = assign_task_to_employee.invoke({
    "employee_id": top_employee["employee_id"],
    "task_title": "Emergency Product Recall Support",
    "task_description": "Assist customers with urgent product recall process",
    "task_type": "emergency",
    "priority_level": "critical",
    "due_hours": 1,
    "estimated_duration_minutes": 180
})
```

### Scenario 2: Personal Shopping Assignment
```python
# 1. Find best personal shopping associate
best_stylist = find_personal_shopping_associates.invoke({
    "expertise_level": "Expert"
})[0]

# 2. Assign VIP shopping session
result = assign_task_to_employee.invoke({
    "employee_id": best_stylist["employee_id"],
    "task_title": "VIP Personal Shopping Session",
    "task_description": "Provide personalized styling service for premium customer",
    "task_type": "routine",
    "priority_level": "medium",
    "due_hours": 24,
    "estimated_duration_minutes": 120
})
```

### Scenario 3: Natural Language Task Creation
```python
# 1. Extract task from natural language
user_request = "We need someone to help with the new product launch demo tomorrow. It's pretty important and will probably take about 2 hours."

task_info = task_extraction.invoke({"input": user_request})

# 2. Find appropriate employee
top_employee = find_top_employees_by_department.invoke({
    "department": "Electronics",
    "limit": 1
})[0]

# 3. Create assignment using extracted information
result = assign_task_to_employee.invoke({
    "employee_id": top_employee["employee_id"],
    **task_info  # Use extracted task information
})
```

## Workflow Benefits

### For Managers
- **Automated Selection**: System identifies best performers automatically
- **Context-Rich Requests**: Full reasoning for employee selection provided
- **Flexible Communication**: Uses manager's preferred communication method
- **Approval Control**: Maintains management oversight and approval process
- **Performance Insights**: Access to employee performance data for decisions

### For Employees
- **Merit-Based Assignment**: Tasks assigned based on performance and skills
- **Clear Expectations**: Detailed task descriptions and timelines
- **Recognition**: Top performers get priority assignments
- **Skill Development**: Assignments matched to employee strengths

### For Organizations
- **Optimal Resource Allocation**: Best employees assigned to critical tasks
- **Performance Tracking**: Full audit trail of assignments and outcomes
- **Efficiency Gains**: Automated identification and assignment process
- **Quality Assurance**: Top performers handle high-priority work
- **Data-Driven Decisions**: Performance metrics guide task distribution

## Error Handling

The system includes comprehensive error handling:

### Employee Not Found
```python
{
    "assignment_id": None,
    "status": "error",
    "message": "Could not find manager for employee EMP-999"
}
```

### Manager Communication Failure
```python
{
    "assignment_id": "TASK-12345",
    "status": "partial_success",
    "notification_sent": False,
    "message": "Assignment created but notification failed"
}
```

### Database Connection Issues
```python
{
    "assignment_id": None,
    "status": "error",
    "message": "Failed to create task assignment record"
}
```

## Configuration Requirements

### Database Setup
Ensure these tables and views exist:
- `employee_performance` (existing)
- `managers` (new)
- `task_assignments` (new)
- `manager_employee_lookup` (view)
- `active_task_assignments` (view)

### Model Configuration
Update `model_config.yaml`:
```yaml
catalog_name: "your_catalog_name"
database_name: "your_database_name"
warehouse_id: "your_warehouse_id"
```

### Manager Data
Populate the `managers` table with:
- Manager contact information
- Communication preferences
- Work schedules and availability
- Task assignment limits and preferences

## Future Enhancements

### Real Communication Integration
- Slack API integration for actual message sending
- Email service integration (SendGrid, AWS SES)
- Microsoft Teams webhook integration
- SMS notifications for urgent tasks

### Advanced Workflow Features
- Automatic task approval for routine assignments
- Escalation workflows for overdue approvals
- Performance-based task routing algorithms
- Integration with calendar systems for scheduling

### Analytics and Reporting
- Task assignment success rates
- Manager response time analytics
- Employee task completion metrics
- Department workload balancing reports

## Security Considerations

- **Access Control**: Ensure proper permissions for manager data access
- **Data Privacy**: Protect employee performance and contact information
- **Audit Logging**: Track all task assignments and approvals
- **Communication Security**: Use secure channels for notifications
- **Role-Based Access**: Limit tool access based on user roles

This workflow provides a complete solution for intelligent task assignment that respects organizational hierarchy while optimizing for performance and efficiency. 