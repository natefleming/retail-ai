#!/usr/bin/env python3
"""
Task Assignment Workflow Example

This script demonstrates the complete workflow for:
1. Finding the top employee in a department
2. Identifying their manager
3. Assigning a task to that employee through their manager
4. Tracking the assignment

This showcases the integration between employee performance tools and task assignment tools.
"""

import json
from datetime import datetime
from retail_ai.tools.employee import (
    create_find_top_employees_by_department_tool,
    create_find_employee_manager_tool,
    create_task_assignment_tool,
    create_task_extraction_tool,
)

def demonstrate_task_assignment_workflow():
    """Demonstrate the complete task assignment workflow."""
    
    # Configuration (these would normally come from model_config)
    catalog_name = "nfleming"  # Replace with your catalog
    database_name = "retail_ai"  # Replace with your database
    warehouse_id = "your_warehouse_id"  # Replace with your warehouse ID
    
    print("🎯 Task Assignment Workflow Demonstration")
    print("=" * 60)
    
    # Create the tools
    find_top_employees = create_find_top_employees_by_department_tool(
        catalog_name, database_name, warehouse_id
    )
    
    find_manager = create_find_employee_manager_tool(
        catalog_name, database_name, warehouse_id
    )
    
    # Note: In real usage, you'd pass an actual LLM instance
    # assign_task = create_task_assignment_tool(
    #     catalog_name, database_name, warehouse_id, llm
    # )
    
    print("\n📋 Scenario: Assign a high-priority customer service task")
    print("-" * 50)
    
    # Step 1: Find top employee in Customer Service
    print("1️⃣ Finding top employee in Customer Service department...")
    try:
        # This would normally execute the tool
        print("   Tool: find_top_employees_by_department")
        print("   Args: {'department': 'Customer Service', 'limit': 1}")
        print("   ✅ Would find: Emma Rodriguez (EMP-007) - Customer Service Manager")
        print("      Performance Score: 4.90, Rank: #1 in department")
        
        # Mock result for demonstration
        top_employee = {
            "employee_id": "EMP-007",
            "employee_name": "Emma Rodriguez",
            "position_title": "Customer Service Manager",
            "department": "Customer Service",
            "store_name": "Downtown Market",
            "overall_performance_score": 4.90,
            "performance_ranking_in_department": 1
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Step 2: Find the employee's manager
    print(f"\n2️⃣ Finding manager for {top_employee['employee_name']}...")
    try:
        print("   Tool: find_employee_manager")
        print(f"   Args: {{'employee_id': '{top_employee['employee_id']}'}}")
        print("   ✅ Would find: Robert Chen (MGR-001) - Store Manager")
        print("      Contact: robert.chen@brickmart.com, Slack: U01ROBERT")
        print("      Preferred Communication: Slack")
        
        # Mock result for demonstration
        manager_info = {
            "manager_id": "MGR-001",
            "manager_name": "Robert Chen",
            "manager_title": "Store Manager",
            "email_address": "robert.chen@brickmart.com",
            "slack_user_id": "U01ROBERT",
            "preferred_communication_method": "slack"
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Step 3: Create task assignment
    print(f"\n3️⃣ Assigning task to {top_employee['employee_name']} via {manager_info['manager_name']}...")
    
    task_details = {
        "task_title": "Handle VIP Customer Complaint Resolution",
        "task_description": "Priority customer complaint from premium member needs immediate attention. Customer reported issues with recent purchase and requires personalized resolution within 2 hours.",
        "task_type": "priority",
        "priority_level": "high",
        "due_hours": 2,
        "estimated_duration_minutes": 90
    }
    
    try:
        print("   Tool: assign_task_to_employee")
        print(f"   Args: {json.dumps(task_details, indent=6)}")
        
        # Mock the assignment process
        assignment_id = "TASK-A1B2C3D4"
        
        print("   ✅ Task assignment created successfully!")
        print(f"      Assignment ID: {assignment_id}")
        print(f"      Status: Pending manager approval")
        print(f"      Notification sent via: {manager_info['preferred_communication_method']}")
        print(f"      Due: {datetime.now().strftime('%Y-%m-%d %H:%M')} (2 hours from now)")
        
        # Mock notification details
        notification_details = {
            "method": "slack",
            "recipient": manager_info["slack_user_id"],
            "message": f"New task assignment request for {top_employee['employee_name']}"
        }
        
        print(f"\n📱 Mock Notification Sent:")
        print(f"   Platform: Slack")
        print(f"   To: {manager_info['manager_name']} ({manager_info['slack_user_id']})")
        print(f"   Message: 'Hi {manager_info['manager_name']}, please assign the following task to {top_employee['employee_name']}:'")
        print(f"   Task: {task_details['task_title']}")
        print(f"   Priority: {task_details['priority_level'].upper()}")
        print(f"   Due: 2 hours")
        print(f"   Reason: Selected as top performer in {top_employee['department']} (rank #{top_employee['performance_ranking_in_department']})")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Step 4: Show workflow summary
    print(f"\n📊 Workflow Summary")
    print("-" * 30)
    print(f"✅ Top Employee Identified: {top_employee['employee_name']}")
    print(f"✅ Manager Located: {manager_info['manager_name']}")
    print(f"✅ Task Assignment Created: {assignment_id}")
    print(f"✅ Notification Sent: {notification_details['method']}")
    print(f"✅ Status: Pending manager approval")
    
    print(f"\n🔄 Next Steps in Real Implementation:")
    print("1. Manager receives notification and reviews task")
    print("2. Manager approves/rejects assignment")
    print("3. If approved, employee receives task notification")
    print("4. Employee accepts/declines task")
    print("5. Task execution and completion tracking")
    print("6. Performance metrics updated")
    
    print(f"\n💡 Benefits of This Workflow:")
    print("• Automated identification of best performers")
    print("• Proper management hierarchy respected")
    print("• Full audit trail of task assignments")
    print("• Performance-based task distribution")
    print("• Multi-channel communication support")
    print("• Real-time status tracking")

def demonstrate_different_scenarios():
    """Show different task assignment scenarios."""
    
    print(f"\n🎭 Additional Scenarios")
    print("=" * 40)
    
    scenarios = [
        {
            "name": "Personal Shopping Assignment",
            "department": "Womens Fashion",
            "task": "VIP Personal Shopping Session",
            "employee": "Isabella Rodriguez",
            "manager": "Robert Chen",
            "priority": "medium",
            "reason": "Expert level personal shopper (22 sessions, 4.8 satisfaction)"
        },
        {
            "name": "Electronics Product Demo",
            "department": "Electronics", 
            "task": "New Product Launch Demo",
            "employee": "Sarah Chen",
            "manager": "Ashley Martinez",
            "priority": "high",
            "reason": "Top sales performer (113% target achievement)"
        },
        {
            "name": "Emergency Inventory Check",
            "department": "Footwear",
            "task": "Urgent Stock Verification",
            "employee": "David Kim",
            "manager": "Robert Chen", 
            "priority": "critical",
            "reason": "Department lead with highest task completion rate (97.8%)"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print(f"   Department: {scenario['department']}")
        print(f"   Best Employee: {scenario['employee']}")
        print(f"   Manager: {scenario['manager']}")
        print(f"   Priority: {scenario['priority'].upper()}")
        print(f"   Selection Reason: {scenario['reason']}")

if __name__ == "__main__":
    demonstrate_task_assignment_workflow()
    demonstrate_different_scenarios()
    
    print(f"\n🚀 Ready to implement task assignment workflow!")
    print("Update the configuration with your actual warehouse details to use in production.") 