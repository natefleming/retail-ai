"""
Employee Tools

This module contains tool creation functions for employee-related operations including
finding top employees by department, identifying best personal shopping associates,
and employee performance analytics.
"""

import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse, StatementState
from databricks_ai_bridge.genie import GenieResponse
from databricks_langchain.genie import Genie
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import tool
from loguru import logger

from dais2025.models import DepartmentInfo, TaskAssignmentInfo
from dao_ai.config import (
    GenieRoomModel,
    LLMModel,
    SchemaModel,
    WarehouseModel,
)


def create_employee_insights_tool(
    genie_room: GenieRoomModel | dict[str, Any],
) -> Callable[[str], GenieResponse]:
    """Create a Genie tool for natural language queries over retail data."""

    logger.debug("create_employee_insights_tool")

    if isinstance(genie_room, dict):
        genie_room = GenieRoomModel(**genie_room)

    space_id: str = genie_room.space_id or os.environ.get("DATABRICKS_GENIE_SPACE_ID")

    genie: Genie = Genie(
        space_id=space_id,
    )

    @tool
    def genie_employee_insights(question: str) -> GenieResponse:
        """
        Get executive-level insights and analytics from retail operations data using Databricks Genie AI.

        This tool provides comprehensive business intelligence for retail decision-making, including:

        **Store Performance & Operations:**
        - Store revenue, profitability, and ROI analysis
        - Performance comparisons across locations and regions
        - Store efficiency metrics and operational KPIs
        - Traffic patterns and conversion rates

        **Financial & Revenue Analytics:**
        - Sales performance trends and forecasting
        - Revenue breakdowns by category, brand, or time period
        - Margin analysis and profitability insights
        - Budget vs actual performance tracking

        **Inventory & Product Intelligence:**
        - Inventory turnover and optimization opportunities
        - Product performance and category analysis
        - Stock-out impact on sales and customer satisfaction
        - Seasonal trends and demand forecasting

        **Customer & Market Insights:**
        - Customer behavior patterns and preferences
        - Market share analysis and competitive positioning
        - Customer acquisition and retention metrics
        - Demographic and geographic performance analysis

        **Strategic Decision Support:**
        - Executive dashboards and KPI summaries
        - Growth opportunities and expansion analysis
        - Risk assessment and performance alerts
        - Operational efficiency recommendations

        **Example Executive Questions:**
        - "What's our quarterly revenue growth compared to last year?"
        - "Which store locations are underperforming and why?"
        - "Show me profitability by product category for this quarter"
        - "What's the ROI on our recent inventory investments?"
        - "Which markets show the highest growth potential?"
        - "How do our conversion rates compare across different store formats?"
        - "What's driving the decline in same-store sales this month?"

        Args:
            question (str): Executive-level business question about retail performance,
                        revenue, operations, or strategic insights

        Returns:
            GenieResponse: Comprehensive analysis with data-driven insights, trends,
                        and actionable recommendations for executive decision-making
        """
        logger.debug(f"executive_insights query: {question}")

        try:
            # Forward the executive question to Genie for comprehensive analysis
            response: GenieResponse = genie.ask_question(question)
            return response

        except Exception as e:
            logger.error(f"Error retrieving executive insights: {e}")
            return "Unable to retrieve executive insights at this time. Please try rephrasing your question or contact support for assistance."

    return genie_employee_insights


def create_find_top_employees_by_department_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    """
    Create a Unity Catalog tool for finding top employees by department.

    This tool enables managers and HR to identify the highest performing employees
    within specific departments based on comprehensive performance metrics including
    sales achievement, task completion, customer satisfaction, and overall performance scores.

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names

    Returns:
        A callable tool function that finds top employees by department
    """
    logger.debug("create_find_top_employees_by_department_tool")

    # Get catalog and database names from config
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def find_top_employees_by_department(department: str, limit: int = 5) -> tuple:
        """
        Find the top performing employees in a specific department.

        This tool retrieves the highest performing employees within a department based on
        overall performance scores, sales achievement, customer satisfaction, and other
        key performance indicators. Useful for identifying star performers, potential
        promotions, recognition programs, and staffing decisions.

        Args:
            department (str): Department name to search within. Examples: "Electronics",
                            "Footwear", "Customer Service", "Womens Fashion"
            limit (int): Maximum number of top employees to return (default: 5)

        Returns:
            tuple: Employee performance data including:
                - employee_id, employee_name, position_title
                - store_name, overall_performance_score
                - sales_achievement_percentage, customer_satisfaction_score
                - task_completion_rate, attendance_rate
                - performance_ranking_in_department
                - personal_shopping_sessions (for fashion departments)
                - recognition_points, employee_of_month_awards
        """
        logger.debug(f"find_top_employees_by_department: {department}, limit: {limit}")

        # Execute query using the pre-built view
        sql_query = f"""
            SELECT 
                department,
                store_name,
                employee_id,
                employee_name,
                position_title,
                overall_performance_score,
                performance_ranking_in_department,
                total_sales_amount,
                sales_achievement_percentage,
                task_completion_rate,
                customer_satisfaction_score,
                attendance_rate,
                personal_shopping_sessions,
                customer_compliments,
                employee_of_month_awards,
                recognition_points,
                performance_trend
            FROM {schema.full_name}.top_employees_by_department
            WHERE LOWER(department) = LOWER('{department}')
            ORDER BY overall_performance_score DESC
            LIMIT {limit}
        """

        # Get workspace client and execute query
        w = WorkspaceClient()

        response: StatementResponse = w.statement_execution.execute_statement(
            warehouse_id=warehouse.warehouse_id,
            statement=sql_query,
            wait_timeout="30s",
        )

        if response.status.state != StatementState.SUCCEEDED:
            logger.error(f"Query failed: {response.status}")
            return ()

        # Convert results to DataFrame and then to tuple
        if response.result and response.result.data_array:
            df = pd.DataFrame(
                response.result.data_array,
                columns=[col.name for col in response.result.manifest.schema.columns],
            )
            logger.debug(f"Found {len(df)} top employees in {department}")
            return tuple(df.to_dict("records"))

        return ()

    return find_top_employees_by_department


def create_find_personal_shopping_associates_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    """
    Create a Unity Catalog tool for finding the best personal shopping associates.

    This tool helps managers identify the most qualified associates for personal shopping
    appointments based on their experience, customer satisfaction scores, product knowledge,
    and availability status.

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names

    Returns:
        A callable tool function that finds best personal shopping associates
    """
    logger.debug("create_find_personal_shopping_associates_tool")

    # Get catalog and database names from config
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def find_personal_shopping_associates(
        store_id: str = None, expertise_level: str = None
    ) -> tuple:
        """
        Find the best associates for personal shopping appointments.

        This tool identifies associates with the highest personal shopping expertise,
        focusing on Women's Fashion department staff with strong customer service skills,
        product knowledge, and availability for personalized shopping experiences.

        Args:
            store_id (str, optional): Specific store ID to search within (e.g., "101", "102", "103").
                                    If not provided, searches across all stores.
            expertise_level (str, optional): Filter by expertise level: "Expert", "Advanced",
                                           "Intermediate", or "Beginner". If not provided,
                                           returns all levels.

        Returns:
            tuple: Personal shopping associate data including:
                - store_id, store_name, employee_id, employee_name, position_title
                - personal_shopping_sessions, customer_satisfaction_score
                - product_knowledge_score, overall_performance_score
                - comprehensive_score, expertise_level, overall_rank
                - availability_status
        """
        logger.debug(
            f"find_personal_shopping_associates: store_id={store_id}, expertise_level={expertise_level}"
        )

        # Build WHERE clause based on parameters
        where_conditions = []
        if store_id:
            where_conditions.append(f"store_id = '{store_id}'")
        if expertise_level:
            where_conditions.append(f"expertise_level = '{expertise_level}'")

        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)

        # Execute query using the pre-built view
        sql_query = f"""
            SELECT 
                store_id,
                store_name,
                employee_id,
                employee_name,
                position_title,
                personal_shopping_sessions,
                customer_satisfaction_score,
                product_knowledge_score,
                overall_performance_score,
                comprehensive_score,
                expertise_level,
                overall_rank
            FROM {schema.full_name}.top_personal_shopping_associates_all_stores
            {where_clause}
            ORDER BY comprehensive_score DESC
            LIMIT 10
        """

        # Get workspace client and execute query
        w = WorkspaceClient()

        response: StatementResponse = w.statement_execution.execute_statement(
            warehouse_id=warehouse.warehouse_id,
            statement=sql_query,
            wait_timeout="30s",
        )

        if response.status.state != StatementState.SUCCEEDED:
            logger.error(f"Query failed: {response.status}")
            return ()

        # Convert results to DataFrame and then to tuple
        if response.result and response.result.data_array:
            df = pd.DataFrame(
                response.result.data_array,
                columns=[col.name for col in response.result.manifest.schema.columns],
            )
            logger.debug(f"Found {len(df)} personal shopping associates")
            return tuple(df.to_dict("records"))

        return ()

    return find_personal_shopping_associates


def create_department_extraction_tool(
    llm_model: LLMModel | dict[str, Any],
) -> Callable[[str], str]:
    """
    Create a tool that leverages an LLM to extract department names from natural language text.

    This tool enables automated extraction of department names from customer queries,
    manager requests, or conversational inputs for employee lookup and performance analysis.

    Args:
        llm: Language model to use for department name extraction from unstructured text

    Returns:
        A callable tool function that extracts department names from input text
    """
    logger.debug("create_department_extraction_tool")

    if isinstance(llm_model, dict):
        llm_model: LLMModel = LLMModel(**llm_model)

    @tool
    def department_extraction(input: str) -> list[str]:
        """
        Extract department names from natural language text using an LLM.

        This tool analyzes unstructured text to identify and extract department names,
        enabling automated department identification from manager queries or employee
        performance discussions.

        Args:
            input: Natural language text that may contain department names

        Returns:
            List of extracted department names found in the input text
        """
        logger.debug(f"department_extraction: {input}")

        llm: LanguageModelLike = llm_model.as_chat_model()

        # Use the LLM with structured output to extract department names
        chain = llm.with_structured_output(DepartmentInfo)
        result = chain.invoke(
            f"Extract any department names from this text. Common departments include: "
            f"Electronics, Footwear, Customer Service, Womens Fashion, Mens Fashion, "
            f"Home & Garden, Sporting Goods, Automotive, Pharmacy, Grocery. "
            f"Text: {input}"
        )

        return result.department_names

    return department_extraction


def create_find_employee_manager_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    """
    Create a Unity Catalog tool for finding an employee's manager.

    This tool helps identify the direct manager of a specific employee,
    useful for task assignments, escalations, and organizational queries.

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names

    Returns:
        A callable tool function that finds an employee's manager
    """
    logger.debug("create_find_employee_manager_tool")

    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def find_employee_manager(employee_id: str) -> tuple:
        """
        Find the direct manager of a specific employee.

        This tool retrieves manager information for a given employee ID,
        including manager contact details and organizational hierarchy.

        Args:
            employee_id (str): The unique identifier of the employee

        Returns:
            tuple: Manager information including:
                - manager_id, manager_name, manager_position
                - manager_email, manager_phone
                - department, store_name
                - reporting_relationship details
        """
        logger.debug(f"find_employee_manager: {employee_id}")

        # Execute query to find employee's manager
        sql_query = f"""
            SELECT 
                e.employee_id,
                e.employee_name,
                e.position_title as employee_position,
                e.department,
                s.store_name,
                m.employee_id as manager_id,
                m.employee_name as manager_name,
                m.position_title as manager_position,
                m.email as manager_email,
                m.phone as manager_phone
            FROM {schema.full_name}.employees e
            LEFT JOIN {schema.full_name}.employees m ON e.manager_id = m.employee_id
            LEFT JOIN {schema.full_name}.stores s ON e.store_id = s.store_id
            WHERE e.employee_id = '{employee_id}'
        """

        # Get workspace client and execute query
        w = WorkspaceClient()

        response: StatementResponse = w.statement_execution.execute_statement(
            warehouse_id=warehouse.warehouse_id,
            statement=sql_query,
            wait_timeout="30s",
        )

        if response.status.state != StatementState.SUCCEEDED:
            logger.error(f"Query failed: {response.status}")
            return ()

        # Convert results to DataFrame and then to tuple
        if response.result and response.result.data_array:
            df = pd.DataFrame(
                response.result.data_array,
                columns=[col.name for col in response.result.manifest.schema.columns],
            )
            logger.debug(f"Found manager information for employee {employee_id}")
            return tuple(df.to_dict("records"))

        return ()

    return find_employee_manager


def create_task_assignment_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    """
    Create a tool for assigning tasks to employees with automatic manager notification.

    This tool creates task assignments, stores them in the database, and automatically
    notifies the employee's manager about the assignment.

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names
        llm: Language model for generating task descriptions and notifications

    Returns:
        A callable tool function that assigns tasks to employees
    """
    logger.debug("create_task_assignment_tool")

    # Get catalog and database names from config
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def assign_task_to_employee(
        employee_id: str,
        task_title: str,
        task_description: str,
        task_type: str = "routine",
        priority_level: str = "medium",
        due_hours: int = 24,
        estimated_duration_minutes: int = 60,
    ) -> dict:
        """
        Assign a task to an employee and notify their manager.

        This tool creates a new task assignment for the specified employee,
        stores it in the task management system, and automatically sends
        a notification to the employee's direct manager.

        Args:
            employee_id (str): The unique identifier of the employee to assign the task to
            task_title (str): Brief title describing the task
            task_description (str): Detailed description of what needs to be done
            task_type (str): Type of task - "routine", "urgent", "training", "special_project"
            priority_level (str): Priority level - "low", "medium", "high", "critical"
            due_hours (int): Number of hours from now when the task is due (default: 24)
            estimated_duration_minutes (int): Estimated time to complete the task in minutes (default: 60)

        Returns:
            dict: Task assignment result including:
                - assignment_id: Unique identifier for the task assignment
                - status: Success/failure status
                - employee_info: Employee details
                - manager_notification: Manager notification status
                - due_date: When the task is due
                - message: Human-readable result message
        """
        logger.debug(f"assign_task_to_employee: {employee_id}, task: {task_title}")

        try:
            # Generate unique assignment ID
            assignment_id = str(uuid.uuid4())

            # Calculate due date
            due_date = datetime.now() + timedelta(hours=due_hours)

            # First, get employee information and their manager
            find_manager = create_find_employee_manager_tool(
                schema=schema, warehouse=warehouse
            )
            employee_manager_info = find_manager.invoke({"employee_id": employee_id})

            if not employee_manager_info:
                return {
                    "assignment_id": assignment_id,
                    "status": "failed",
                    "error": f"Employee {employee_id} not found",
                    "message": f"Could not assign task - employee {employee_id} not found in system",
                }

            employee_info = employee_manager_info[0]

            # Insert task assignment into database
            insert_sql = f"""
                INSERT INTO {schema.full_name}.task_assignments 
                (assignment_id, employee_id, task_title, task_description, task_type, 
                 priority_level, assigned_date, due_date, estimated_duration_minutes, 
                 status, assigned_by)
                VALUES (
                    '{assignment_id}',
                    '{employee_id}',
                    '{task_title.replace("'", "''")}',
                    '{task_description.replace("'", "''")}',
                    '{task_type}',
                    '{priority_level}',
                    CURRENT_TIMESTAMP(),
                    '{due_date.strftime("%Y-%m-%d %H:%M:%S")}',
                    {estimated_duration_minutes},
                    'assigned',
                    'system'
                )
            """

            # Get workspace client and execute insert
            w = WorkspaceClient()

            response: StatementResponse = w.statement_execution.execute_statement(
                warehouse_id=warehouse.warehouse_id,
                statement=insert_sql,
                wait_timeout="30s",
            )

            if response.status.state != StatementState.SUCCEEDED:
                logger.error(f"Task assignment insert failed: {response.status}")
                return {
                    "assignment_id": assignment_id,
                    "status": "failed",
                    "error": "Database insert failed",
                    "message": "Failed to create task assignment in database",
                }

            # Send notification to manager (mock implementation)
            manager_notification = _mock_send_notification(
                employee_info, assignment_id, task_title, task_description
            )

            logger.info(f"Task {assignment_id} assigned to {employee_id}")

            return {
                "assignment_id": assignment_id,
                "status": "success",
                "employee_info": {
                    "employee_id": employee_info.get("employee_id"),
                    "employee_name": employee_info.get("employee_name"),
                    "department": employee_info.get("department"),
                    "store_name": employee_info.get("store_name"),
                },
                "task_details": {
                    "title": task_title,
                    "type": task_type,
                    "priority": priority_level,
                    "due_date": due_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "estimated_duration": f"{estimated_duration_minutes} minutes",
                },
                "manager_notification": manager_notification,
                "message": f"Task '{task_title}' successfully assigned to {employee_info.get('employee_name')} ({employee_id}). Manager {employee_info.get('manager_name', 'N/A')} has been notified.",
            }

        except Exception as e:
            logger.error(f"Error in task assignment: {e}")
            return {
                "assignment_id": assignment_id,
                "status": "failed",
                "error": str(e),
                "message": f"Failed to assign task due to error: {str(e)}",
            }

    return assign_task_to_employee


def _mock_send_notification(
    manager_info: dict, assignment_id: str, task_title: str, task_description: str
) -> dict:
    """
    Mock function to simulate sending notifications to managers.

    In a real implementation, this would integrate with email, Slack, Teams, etc.
    """
    method = manager_info["preferred_communication_method"]
    # manager_name = manager_info["manager_name"]

    # Simulate different notification methods
    if method == "email":
        return {
            "sent": True,
            "method": "email",
            "message": f"Email sent to {manager_info['email_address']}",
        }
    elif method == "slack":
        return {
            "sent": True,
            "method": "slack",
            "message": f"Slack message sent to {manager_info['slack_user_id']}",
        }
    elif method == "teams":
        return {
            "sent": True,
            "method": "teams",
            "message": f"Teams message sent to {manager_info['teams_user_id']}",
        }
    elif method == "phone":
        return {
            "sent": True,
            "method": "phone",
            "message": f"Phone notification sent to {manager_info['phone_number']}",
        }
    else:
        return {
            "sent": False,
            "method": "unknown",
            "message": f"Unknown communication method: {method}",
        }


def create_task_extraction_tool(llm_model: LLMModel | dict[str, Any]) -> Callable:
    """
    Create a tool that leverages an LLM to extract task information from natural language.

    This tool enables automated extraction of task details from user requests,
    making it easier to create structured task assignments.

    Args:
        llm: Language model to use for task information extraction

    Returns:
        A callable tool function that extracts task information from input text
    """
    logger.debug("create_task_extraction_tool")
    if isinstance(llm_model, dict):
        llm_model: LLMModel = LLMModel(**llm_model)

    @tool
    def task_extraction(input: str) -> dict:
        """
        Extract task information from natural language text using an LLM.

        This tool analyzes unstructured text to identify and extract task details
        like title, description, type, priority, and timing requirements.

        Args:
            input: Natural language text describing a task to be assigned

        Returns:
            Dictionary with extracted task information
        """
        logger.debug(f"task_extraction: {input}")

        llm: LanguageModelLike = llm_model.as_chat_model()

        # Use the LLM with structured output to extract task information
        chain = llm.with_structured_output(TaskAssignmentInfo)
        result = chain.invoke(
            f"Extract task assignment information from this text. "
            f"Identify the task title, description, type (routine/priority/emergency/project/training), "
            f"priority level (low/medium/high/critical), and any timing requirements. "
            f"Text: {input}"
        )

        return {
            "task_title": result.task_title,
            "task_description": result.task_description,
            "task_type": result.task_type,
            "priority_level": result.priority_level,
            "due_hours": result.due_hours,
            "estimated_duration_minutes": result.estimated_duration_minutes,
        }

    return task_extraction
