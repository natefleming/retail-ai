"""
Tools Package

This package contains all tool creation functions and utilities for the Retail AI system.
Tools are organized by category: product tools, inventory tools, store tools, etc.
"""

from dais2025.tools.customer import (
    create_appointment_preparation_workflow_tool,
    create_customer_preparation_summary_tool,
    create_customer_profile_intelligence_tool,
    create_find_upcoming_customer_appointments_tool,
    create_get_customer_details_tool,
    create_inventory_preselection_tool,
    create_real_time_styling_assistant_tool,
    create_stylist_notification_tool,
)
from dais2025.tools.employee import (
    create_department_extraction_tool,
    create_find_employee_manager_tool,
    create_find_personal_shopping_associates_tool,
    create_find_top_employees_by_department_tool,
    create_task_assignment_tool,
    create_task_extraction_tool,
)
from dais2025.tools.inventory import (
    create_find_inventory_by_sku_tool,
    create_find_store_inventory_by_sku_tool,
    create_place_item_hold_tool,
)
from dais2025.tools.store import (
    create_find_store_by_number_tool,
    create_store_number_extraction_tool,
    find_store_details_by_location_tool,
)

__all__ = [
    # Customer tools
    "create_appointment_preparation_workflow_tool",
    "create_customer_preparation_summary_tool",
    "create_customer_profile_intelligence_tool",
    "create_find_upcoming_customer_appointments_tool",
    "create_get_customer_details_tool",
    "create_inventory_preselection_tool",
    "create_real_time_styling_assistant_tool",
    "create_stylist_notification_tool",
    # Employee tools
    "create_department_extraction_tool",
    "create_find_employee_manager_tool",
    "create_find_personal_shopping_associates_tool",
    "create_find_top_employees_by_department_tool",
    "create_task_assignment_tool",
    "create_task_extraction_tool",
    # Inventory tools
    "create_find_inventory_by_sku_tool",
    "create_find_store_inventory_by_sku_tool",
    "create_place_item_hold_tool",
    # Store tools
    "create_find_store_by_number_tool",
    "create_store_number_extraction_tool",
    "find_store_details_by_location_tool",
]
