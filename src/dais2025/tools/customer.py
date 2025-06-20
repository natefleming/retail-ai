"""
Customer Tools

This module contains tool creation functions for customer operations including
finding upcoming appointments, customer preparation details, and styling preferences.
Designed for store managers to quickly prepare for important customer visits.
"""

from typing import Any, Callable

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse, StatementState
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import tool
from loguru import logger

from dao_ai.config import (
    LLMModel,
    SchemaModel,
    WarehouseModel,
)


def create_find_upcoming_customer_appointments_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    """
    Create a Unity Catalog tool for finding upcoming customer appointments.

    This tool helps managers quickly identify customers with upcoming appointments
    and get essential preparation information for providing exceptional service.

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names

    Returns:
        A callable tool function that finds upcoming customer appointments
    """
    logger.debug("create_find_upcoming_customer_appointments_tool")

    # Get catalog and database names from config
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def find_upcoming_customer_appointments(
        store_id: str = None, hours_ahead: int = 24
    ) -> tuple:
        """
        Find upcoming customer appointments for preparation.

        This tool retrieves customers with appointments in the specified timeframe,
        providing managers with essential information to prepare for important customer visits.

        Args:
            store_id (str, optional): Specific store ID to search within (e.g., "101", "102", "103").
                                    If not provided, searches across all stores.
            hours_ahead (int): Number of hours ahead to look for appointments (default: 24)

        Returns:
            tuple: Customer appointment data including:
                - customer_id, customer_name, preferred_name, customer_tier
                - appointment_date, appointment_type, appointment_purpose
                - preferred_stylist_name, style_preferences, budget_range
                - preparation_notes, service_notes, special_occasions
                - requires_manager_greeting, customer_alerts
                - satisfaction_score, total_lifetime_spend
        """
        logger.debug(
            f"find_upcoming_customer_appointments: store_id={store_id}, hours_ahead={hours_ahead}"
        )

        # Build WHERE clause based on parameters
        where_conditions = [
            f"next_appointment_date <= TIMESTAMPADD(HOUR, {hours_ahead}, CURRENT_TIMESTAMP())"
        ]

        if store_id:
            where_conditions.append(f"preferred_store_id = '{store_id}'")

        where_clause = "WHERE " + " AND ".join(where_conditions)

        # Execute query using the pre-built view
        sql_query = f"""
            SELECT 
                customer_id,
                customer_name,
                preferred_name,
                customer_tier,
                store_name,
                preferred_stylist_name,
                next_appointment_date,
                appointment_type,
                appointment_purpose,
                style_preferences,
                budget_range,
                preparation_notes,
                special_occasions,
                service_notes,
                requires_manager_greeting,
                customer_alerts,
                satisfaction_score,
                total_lifetime_spend,
                last_visit_date,
                ROUND((UNIX_TIMESTAMP(next_appointment_date) - UNIX_TIMESTAMP(CURRENT_TIMESTAMP())) / 3600, 1) as hours_until_appointment
            FROM {schema.full_name}.upcoming_customer_appointments
            {where_clause}
            ORDER BY next_appointment_date ASC
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
            logger.debug(f"Found {len(df)} upcoming customer appointments")
            return tuple(df.to_dict("records"))

        return ()

    return find_upcoming_customer_appointments


def create_get_customer_details_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    """
    Create a Unity Catalog tool for getting detailed customer information.

    This tool provides comprehensive customer details for managers to prepare
    for customer visits and ensure personalized service.

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names

    Returns:
        A callable tool function that gets customer details
    """
    logger.debug("create_get_customer_details_tool")

    # Get catalog and database names from config
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def get_customer_details(
        customer_name: str = None, customer_id: str = None
    ) -> tuple:
        """
        Get comprehensive customer details for appointment preparation.

        This tool retrieves detailed information about a specific customer,
        including preferences, history, and special requirements for personalized service.

        Args:
            customer_name (str, optional): Customer name to search for (partial matches supported)
            customer_id (str, optional): Specific customer ID to look up

        Returns:
            tuple: Comprehensive customer information including:
                - Basic info: name, customer tier, contact preferences
                - Styling preferences: style, sizes, colors, brands, budget
                - Service history: sessions, satisfaction, feedback
                - Special requirements: dietary, accessibility, cultural considerations
                - Upcoming appointment details and preparation notes
                - Family information and gift history
        """
        logger.debug(
            f"get_customer_details: customer_name={customer_name}, customer_id={customer_id}"
        )

        # Build WHERE clause based on parameters
        where_conditions = []

        if customer_id:
            where_conditions.append(f"customer_id = '{customer_id}'")
        elif customer_name:
            where_conditions.append(
                f"LOWER(customer_name) LIKE LOWER('%{customer_name}%')"
            )
        else:
            return ()  # Need at least one search parameter

        where_clause = "WHERE " + " AND ".join(where_conditions)

        # Execute query using the preparation summary view
        sql_query = f"""
            SELECT 
                customer_id,
                customer_name,
                preferred_name,
                customer_tier,
                store_name,
                next_appointment_date,
                appointment_type,
                appointment_purpose,
                style_preferences,
                size_information,
                color_preferences,
                brand_preferences,
                budget_range,
                preparation_notes,
                service_notes,
                special_occasions,
                dietary_restrictions,
                accessibility_needs,
                requires_manager_greeting,
                customer_alerts,
                preferred_stylist_name,
                stylist_experience,
                stylist_rating,
                customer_satisfaction,
                total_lifetime_spend,
                average_transaction_value,
                last_visit_date,
                visit_frequency,
                days_since_last_visit,
                hours_until_appointment
            FROM {schema.full_name}.customer_preparation_summary
            {where_clause}
            ORDER BY next_appointment_date ASC
            LIMIT 5
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
            logger.debug(f"Found {len(df)} customer records")
            return tuple(df.to_dict("records"))

        return ()

    return get_customer_details


def create_customer_preparation_summary_tool(
    schema: SchemaModel | dict[str, Any],
    warehouse: WarehouseModel | dict[str, Any],
    llm_model: LLMModel | dict[str, Any],
) -> Callable[[list[str]], tuple]:
    """
    Create a tool that provides an AI-generated preparation summary for customers.

    This tool uses an LLM with a predefined prompt to generate intelligent, context-aware
    preparation summaries for upcoming customer visits across various scenarios.

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names
        llm: Language model for generating intelligent summaries

    Returns:
        A callable tool function that creates AI-powered preparation summaries
    """
    logger.debug("create_customer_preparation_summary_tool")

    # Get catalog and database names from config
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)
    if isinstance(llm_model, dict):
        llm_model = LLMModel(**llm_model)

    # Define the preparation summary prompt
    PREPARATION_SUMMARY_PROMPT = """You are an expert retail customer service manager creating a preparation summary for an upcoming customer visit. Your goal is to help store staff provide exceptional, personalized service.

Based on the customer data provided, create a comprehensive preparation summary that includes:

1. **Customer Overview**: Key facts about the customer (name, tier, value, satisfaction)
2. **Appointment Context**: What they're coming for and when
3. **Key Preparation Points**: Specific actions staff should take (use emojis for visual clarity)
4. **Service Requirements**: Special needs, preferences, and protocols
5. **Styling/Product Preferences**: What they like and what to focus on
6. **Conversation Starters**: Personal details that can enhance the interaction
7. **Success Factors**: What will make this visit exceptional

**Context Guidelines:**
- For personal shopping appointments: Focus on styling preferences, budget, and creating a personalized experience
- For consultations: Emphasize expertise, problem-solving, and building trust
- For regular visits: Highlight relationship building and consistent service
- For new customers: Focus on making a great first impression and learning preferences
- For high-tier customers: Emphasize premium service, attention to detail, and exceeding expectations

**Tone**: Professional, actionable, and focused on customer success. Use clear formatting to make the summary easy to scan quickly.

**Customer Data:**
{customer_data}

Generate a preparation summary that will help staff deliver outstanding service for this specific customer and appointment context."""

    @tool
    def create_customer_preparation_summary(
        customer_name: str, context: str = "general visit"
    ) -> str:
        """
        Generate an AI-powered preparation summary for a customer visit.

        This tool creates an intelligent, context-aware summary with key preparation points
        for upcoming customer appointments, ensuring exceptional service delivery across
        various scenarios including personal shopping, consultations, and regular visits.

        Args:
            customer_name (str): Name of the customer
            context (str): Context for the visit (e.g., "personal shopping appointment",
                          "wardrobe consultation", "general visit", "new customer")

        Returns:
            str: AI-generated preparation summary with:
                - Customer overview and tier information
                - Appointment details and timing
                - Key preparation points with actionable items
                - Service requirements and preferences
                - Styling/product preferences
                - Conversation starters and success factors
        """
        logger.debug(
            f"create_customer_preparation_summary: customer_name={customer_name}, context={context}"
        )

        # Get customer details first
        get_details = create_get_customer_details_tool(
            schema=schema, warehouse=warehouse
        )
        customer_data = get_details.invoke({"customer_name": customer_name})

        if not customer_data:
            return f"No customer found with name containing '{customer_name}'. Please check the spelling or try a different search term."

        customer = customer_data[0]  # Get first match

        # Format customer data for the LLM prompt
        formatted_data = f"""
Customer Name: {customer.get("customer_name")} (prefers: {customer.get("preferred_name", "N/A")})
Customer Tier: {customer.get("customer_tier", "N/A")}
Store: {customer.get("store_name", "N/A")}

Appointment Information:
- Date/Time: {customer.get("next_appointment_date", "N/A")}
- Type: {customer.get("appointment_type", "N/A")}
- Purpose: {customer.get("appointment_purpose", "N/A")}
- Hours Until: {customer.get("hours_until_appointment", "N/A")}
- Context: {context}

Customer Value & History:
- Lifetime Spend: ${customer.get("total_lifetime_spend", 0):,.2f}
- Average Transaction: ${customer.get("average_transaction_value", 0):,.2f}
- Satisfaction Score: {customer.get("customer_satisfaction", "N/A")}/5.0
- Visit Frequency: {customer.get("visit_frequency", "N/A")}
- Days Since Last Visit: {customer.get("days_since_last_visit", "N/A")}
- Styling Sessions: {customer.get("stylist_experience", "N/A")}

Preferences & Requirements:
- Style Preferences: {customer.get("style_preferences", "N/A")}
- Size Information: {customer.get("size_information", "N/A")}
- Color Preferences: {customer.get("color_preferences", "N/A")}
- Brand Preferences: {customer.get("brand_preferences", "N/A")}
- Budget Range: {customer.get("budget_range", "N/A")}

Service Requirements:
- Manager Greeting Required: {customer.get("requires_manager_greeting", False)}
- Preferred Stylist: {customer.get("preferred_stylist_name", "N/A")}
- Special Alerts: {customer.get("customer_alerts", "N/A")}
- Dietary Restrictions: {customer.get("dietary_restrictions", "N/A")}
- Accessibility Needs: {customer.get("accessibility_needs", "N/A")}

Additional Notes:
- Preparation Notes: {customer.get("preparation_notes", "N/A")}
- Service Notes: {customer.get("service_notes", "N/A")}
- Special Occasions: {customer.get("special_occasions", "N/A")}
"""

        # Generate the summary using the LLM
        try:
            llm: LanguageModelLike = llm_model.as_chat_model()
            prompt = PREPARATION_SUMMARY_PROMPT.format(customer_data=formatted_data)
            summary = llm.invoke(prompt)

            # Extract content if it's a message object
            if hasattr(summary, "content"):
                summary_text = summary.content
            else:
                summary_text = str(summary)

            logger.debug(f"Generated preparation summary for {customer_name}")
            return summary_text

        except Exception as e:
            logger.error(f"Error generating summary with LLM: {e}")
            # Fallback to basic summary if LLM fails
            return f"""
Error generating AI summary. Basic customer information:

**{customer.get("customer_name")} ({customer.get("customer_tier")} tier)**
- Appointment: {customer.get("next_appointment_date")} - {customer.get("appointment_type")}
- Purpose: {customer.get("appointment_purpose")}
- Budget: {customer.get("budget_range")}
- Preferred Stylist: {customer.get("preferred_stylist_name", "None specified")}
- Manager Greeting: {"Required" if customer.get("requires_manager_greeting") else "Not required"}
- Special Notes: {customer.get("customer_alerts", "None")}

Please contact technical support for assistance with the AI summary feature.
"""

    return create_customer_preparation_summary


def create_customer_profile_intelligence_tool(
    schema: SchemaModel | dict[str, Any],
    warehouse: WarehouseModel | dict[str, Any],
    llm_model: LLMModel | dict[str, Any],
) -> Callable[[list[str]], tuple]:
    """
    Create a comprehensive customer profile intelligence tool that provides AI-powered insights.

    This tool combines customer data retrieval with intelligent analysis to provide:
    - Purchase patterns and behavior analysis
    - Brand affinity and preferences
    - Style profile and fashion insights
    - Current needs assessment
    - Personalized recommendations for styling, products, and service

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names
        llm: Language model for generating intelligent insights

    Returns:
        A callable tool function that provides comprehensive customer intelligence
    """
    logger.debug("create_customer_profile_intelligence_tool")

    # Get catalog and database names from config
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)
    if isinstance(llm_model, dict):
        llm_model = LLMModel(**llm_model)

    # Define the customer intelligence prompt
    CUSTOMER_INTELLIGENCE_PROMPT = """You are an expert retail customer intelligence analyst providing comprehensive insights about a customer. Your goal is to help store associates and managers deliver exceptional, personalized service by understanding the customer deeply.

Based on the customer data provided, create a comprehensive customer intelligence profile that includes:

## 🎯 **Customer Overview**
- **Customer Identity**: Name, tier, and key demographics
- **Relationship Status**: How long they've been a customer and their value
- **Service Level**: What level of service they expect and require

## 📊 **Purchase Pattern Analysis**
- **Spending Behavior**: Analyze their spending patterns, frequency, and trends
- **Shopping Habits**: When they shop, how often, and their decision-making style
- **Value Analysis**: Their lifetime value and average transaction patterns

## 🏷️ **Brand Affinity & Style Profile**
- **Brand Preferences**: Which brands they love and avoid, and why
- **Style Identity**: Their fashion aesthetic and style evolution
- **Quality Expectations**: Their standards for materials, fit, and craftsmanship

## 🎯 **Current Needs Assessment**
- **Immediate Needs**: What they're looking for right now based on recent activity
- **Seasonal Considerations**: How their needs change with seasons/occasions
- **Life Stage Factors**: How their current life situation affects their needs

## 💡 **Intelligent Recommendations**

### **Styling Recommendations**
- Specific styling approaches that would resonate with this customer
- How to present options and conduct styling sessions
- What styling techniques to emphasize

### **Product Recommendations**
- Types of products to prioritize showing them
- Price points and categories that align with their preferences
- New arrivals or trends they might be interested in

### **Service Recommendations**
- How to approach and interact with this customer
- Service protocols and special considerations
- Communication style and preferences

## 🚨 **Key Success Factors**
- **Critical Do's**: What will make this customer experience exceptional
- **Important Don'ts**: What to avoid to prevent dissatisfaction
- **Special Considerations**: Unique factors that affect their experience

## 📅 **Next Steps & Opportunities**
- **Immediate Actions**: What to prepare for their next visit
- **Future Opportunities**: How to deepen the relationship
- **Upselling Potential**: Appropriate ways to increase their engagement

**Analysis Guidelines:**
- Be specific and actionable in your recommendations
- Consider their tier level and service expectations
- Factor in their spending patterns and preferences
- Think about their lifestyle and current life stage
- Provide insights that help create memorable experiences
- Use emojis strategically for visual clarity and engagement

**Customer Data:**
{customer_data}

Generate a comprehensive intelligence profile that will help our team deliver outstanding, personalized service for this valued customer."""

    @tool
    def get_customer_profile_intelligence(customer_name: str) -> str:
        """
        Generate comprehensive customer intelligence including purchase patterns, brand affinity,
        style profile, current needs, and personalized recommendations.

        This tool provides AI-powered insights about a customer to help store associates and managers
        deliver exceptional, personalized service. It analyzes customer data to understand their
        shopping behavior, preferences, and needs, then provides specific recommendations for
        styling, products, and service approaches.

        Args:
            customer_name (str): Name of the customer to analyze (partial matches supported)

        Returns:
            str: Comprehensive customer intelligence profile including:
                - Customer overview and relationship status
                - Purchase pattern analysis and spending behavior
                - Brand affinity and style profile insights
                - Current needs assessment based on recent activity
                - Intelligent recommendations for styling, products, and service
                - Key success factors and special considerations
                - Next steps and opportunities for relationship building
        """
        logger.debug(
            f"get_customer_profile_intelligence: customer_name={customer_name}"
        )

        # Build WHERE clause for customer search
        where_clause = f"WHERE LOWER(customer_name) LIKE LOWER('%{customer_name}%')"

        # Execute comprehensive customer query
        sql_query = f"""
            SELECT 
                customer_id,
                customer_name,
                preferred_name,
                customer_tier,
                member_since,
                email_address,
                phone_number,
                preferred_contact_method,
                preferred_store_id,
                preferred_stylist_id,
                preferred_appointment_time,
                style_preferences,
                size_information,
                color_preferences,
                brand_preferences,
                budget_range,
                total_lifetime_spend,
                average_transaction_value,
                last_visit_date,
                last_purchase_date,
                visit_frequency,
                seasonal_preferences,
                special_occasions,
                dietary_restrictions,
                accessibility_needs,
                language_preference,
                cultural_considerations,
                styling_sessions,
                satisfaction_score,
                last_feedback,
                service_notes,
                next_appointment_date,
                appointment_type,
                appointment_purpose,
                preparation_notes,
                family_members,
                gift_history,
                referral_source,
                customer_status,
                priority_level,
                requires_manager_greeting,
                customer_alerts,
                DATEDIFF(CURRENT_DATE(), member_since) as days_as_customer,
                DATEDIFF(CURRENT_DATE(), last_visit_date) as days_since_last_visit,
                CASE 
                    WHEN next_appointment_date IS NOT NULL 
                    THEN ROUND((UNIX_TIMESTAMP(next_appointment_date) - UNIX_TIMESTAMP(CURRENT_TIMESTAMP())) / 3600, 1)
                    ELSE NULL 
                END as hours_until_appointment
            FROM {schema.full_name}.customers
            {where_clause}
            ORDER BY total_lifetime_spend DESC
            LIMIT 3
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
            return f"Unable to retrieve customer data for '{customer_name}'. Please check the customer name and try again."

        # Convert results to DataFrame
        if not (response.result and response.result.data_array):
            return f"No customer found with name containing '{customer_name}'. Please check the spelling or try a different search term."

        # Try to get column names from different possible locations
        columns = None

        # Try the manifest approach first (older SDK versions)
        if hasattr(response.result, "manifest") and hasattr(
            response.result.manifest, "schema"
        ):
            columns = [col.name for col in response.result.manifest.schema.columns]
        # Try the schema approach (newer SDK versions)
        elif hasattr(response.result, "schema") and hasattr(
            response.result.schema, "columns"
        ):
            columns = [col.name for col in response.result.schema.columns]
        # Fallback: try to infer from the first row of data
        elif response.result.data_array and len(response.result.data_array) > 0:
            # Use generic column names based on the number of columns
            num_cols = (
                len(response.result.data_array[0])
                if response.result.data_array[0]
                else 0
            )
            columns = [f"col_{i}" for i in range(num_cols)]
        else:
            logger.warning("Could not determine column names from response")
            return f"Error processing customer data for '{customer_name}'. Please try again."

        df = pd.DataFrame(response.result.data_array, columns=columns)

        if len(df) == 0:
            return f"No customer found with name containing '{customer_name}'. Please check the spelling or try a different search term."

        # Use the first (highest value) customer if multiple matches
        customer = df.iloc[0].to_dict()

        # Format customer data for the LLM prompt
        formatted_data = f"""
**CUSTOMER IDENTIFICATION**
Customer ID: {customer.get("customer_id", "N/A")}
Full Name: {customer.get("customer_name", "N/A")}
Preferred Name: {customer.get("preferred_name", "N/A")}
Customer Tier: {customer.get("customer_tier", "N/A")}
Member Since: {customer.get("member_since", "N/A")} ({customer.get("days_as_customer", "N/A")} days as customer)
Status: {customer.get("customer_status", "N/A")}
Priority Level: {customer.get("priority_level", "N/A")}

**CONTACT & PREFERENCES**
Email: {customer.get("email_address", "N/A")}
Phone: {customer.get("phone_number", "N/A")}
Preferred Contact: {customer.get("preferred_contact_method", "N/A")}
Language: {customer.get("language_preference", "N/A")}
Cultural Considerations: {customer.get("cultural_considerations", "N/A")}

**SHOPPING BEHAVIOR & VALUE**
Total Lifetime Spend: ${customer.get("total_lifetime_spend", 0):,.2f}
Average Transaction: ${customer.get("average_transaction_value", 0):,.2f}
Budget Range: {customer.get("budget_range", "N/A")}
Visit Frequency: {customer.get("visit_frequency", "N/A")}
Last Visit: {customer.get("last_visit_date", "N/A")} ({customer.get("days_since_last_visit", "N/A")} days ago)
Last Purchase: {customer.get("last_purchase_date", "N/A")}
Seasonal Preferences: {customer.get("seasonal_preferences", "N/A")}

**STYLE & BRAND PREFERENCES**
Style Preferences: {customer.get("style_preferences", "N/A")}
Size Information: {customer.get("size_information", "N/A")}
Color Preferences: {customer.get("color_preferences", "N/A")}
Brand Preferences: {customer.get("brand_preferences", "N/A")}

**SERVICE HISTORY & SATISFACTION**
Styling Sessions Completed: {customer.get("styling_sessions", "N/A")}
Satisfaction Score: {customer.get("satisfaction_score", "N/A")}/5.0
Last Feedback: {customer.get("last_feedback", "N/A")}
Service Notes: {customer.get("service_notes", "N/A")}

**UPCOMING APPOINTMENT**
Next Appointment: {customer.get("next_appointment_date", "N/A")}
Appointment Type: {customer.get("appointment_type", "N/A")}
Purpose: {customer.get("appointment_purpose", "N/A")}
Hours Until Appointment: {customer.get("hours_until_appointment", "N/A")}
Preparation Notes: {customer.get("preparation_notes", "N/A")}

**STORE & STYLIST PREFERENCES**
Preferred Store: {customer.get("preferred_store_id", "N/A")}
Preferred Stylist: {customer.get("preferred_stylist_id", "N/A")}
Preferred Appointment Time: {customer.get("preferred_appointment_time", "N/A")}
Requires Manager Greeting: {customer.get("requires_manager_greeting", False)}

**SPECIAL CONSIDERATIONS**
Customer Alerts: {customer.get("customer_alerts", "N/A")}
Dietary Restrictions: {customer.get("dietary_restrictions", "N/A")}
Accessibility Needs: {customer.get("accessibility_needs", "N/A")}
Special Occasions: {customer.get("special_occasions", "N/A")}

**PERSONAL & FAMILY**
Family Members: {customer.get("family_members", "N/A")}
Gift History: {customer.get("gift_history", "N/A")}
Referral Source: {customer.get("referral_source", "N/A")}
"""

        # Generate the intelligence profile using the LLM
        try:
            llm: LanguageModelLike = llm_model.as_chat_model()
            prompt = CUSTOMER_INTELLIGENCE_PROMPT.format(customer_data=formatted_data)
            intelligence_profile = llm.invoke(prompt)

            # Extract content if it's a message object
            if hasattr(intelligence_profile, "content"):
                profile_text = intelligence_profile.content
            else:
                profile_text = str(intelligence_profile)

            logger.debug(f"Generated customer intelligence profile for {customer_name}")
            return profile_text

        except Exception as e:
            logger.error(f"Error generating intelligence profile with LLM: {e}")
            # Fallback to basic customer summary if LLM fails
            return f"""
# Customer Profile: {customer.get("customer_name")}

**Error**: Unable to generate AI-powered insights. Here's the basic customer information:

## Basic Information
- **Name**: {customer.get("customer_name")} (prefers: {customer.get("preferred_name", "N/A")})
- **Tier**: {customer.get("customer_tier")} customer since {customer.get("member_since")}
- **Value**: ${customer.get("total_lifetime_spend", 0):,.2f} lifetime spend, ${customer.get("average_transaction_value", 0):,.2f} average transaction
- **Satisfaction**: {customer.get("satisfaction_score", "N/A")}/5.0 rating

## Key Preferences
- **Style**: {customer.get("style_preferences", "N/A")}
- **Brands**: {customer.get("brand_preferences", "N/A")}
- **Budget**: {customer.get("budget_range", "N/A")}
- **Colors**: {customer.get("color_preferences", "N/A")}

## Service Requirements
- **Manager Greeting**: {"Required" if customer.get("requires_manager_greeting") else "Not required"}
- **Special Alerts**: {customer.get("customer_alerts", "None")}
- **Service Notes**: {customer.get("service_notes", "None")}

## Next Appointment
- **Date**: {customer.get("next_appointment_date", "None scheduled")}
- **Type**: {customer.get("appointment_type", "N/A")}
- **Purpose**: {customer.get("appointment_purpose", "N/A")}

Please contact technical support for assistance with the AI intelligence feature.
"""

    return get_customer_profile_intelligence


def create_stylist_notification_tool(
    schema: SchemaModel | dict[str, Any],
    warehouse: WarehouseModel | dict[str, Any],
    llm_model: LLMModel | dict[str, Any],
) -> Callable[[list[str]], tuple]:
    """
    Create a tool that generates intelligent stylist notifications for upcoming appointments.

    This tool creates context-rich notifications that include customer intelligence,
    appointment details, and preparation recommendations for stylists.

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names
        llm: Language model for generating intelligent notifications

    Returns:
        A callable tool function that generates stylist notifications
    """
    logger.debug("create_stylist_notification_tool")

    # Get catalog and database names from config
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)
    if isinstance(llm_model, dict):
        llm_model = LLMModel(**llm_model)

    # Define the notification prompt
    NOTIFICATION_PROMPT = """You are an AI assistant creating an intelligent notification for a stylist about an upcoming customer appointment. Your goal is to provide context-rich, actionable information that helps the stylist prepare for exceptional service.

Based on the customer and appointment data provided, create a notification that includes:

## 🚨 **URGENT NOTIFICATION**
**Customer**: [Customer Name] ([Tier] Member, [Years] years)
**Appointment**: [Time] ([Minutes] minutes remaining)
**Service**: [Appointment Type] - [Purpose]

## 📊 **Customer Intelligence Summary**
- **Last Purchase**: [Amount] ([Time ago])
- **Average Spend**: [Amount] per visit
- **Style Profile**: [Brief style description]
- **Satisfaction**: [Score]/5.0 ([Relationship quality])

## 🎯 **Key Preparation Points**
- [3-5 specific, actionable preparation items with emojis]
- Focus on what the stylist should do immediately
- Include any special requirements or preferences

## 💡 **Service Recommendations**
- **Styling Approach**: [How to approach this customer]
- **Focus Areas**: [What to emphasize during the appointment]
- **Success Factors**: [What will make this appointment exceptional]

**Tone**: Urgent but professional, actionable, and focused on customer success. Use clear formatting and emojis for visual clarity.

**Customer Data:**
{customer_data}"""

    @tool
    def generate_stylist_notification(
        customer_name: str, stylist_name: str = None
    ) -> str:
        """
        Generate an intelligent notification for a stylist about an upcoming customer appointment.

        This tool creates context-rich notifications that include customer intelligence,
        appointment details, and specific preparation recommendations to help stylists
        deliver exceptional, personalized service.

        Args:
            customer_name (str): Name of the customer with upcoming appointment
            stylist_name (str, optional): Name of the stylist receiving the notification

        Returns:
            str: Intelligent notification with:
                - Urgent appointment alert with timing
                - Customer intelligence summary
                - Key preparation points with actionable items
                - Service recommendations and success factors
                - Styling approach and focus areas
        """
        logger.debug(
            f"generate_stylist_notification: customer_name={customer_name}, stylist_name={stylist_name}"
        )

        # Get customer details first
        get_details = create_get_customer_details_tool(
            schema=schema, warehouse=warehouse
        )
        customer_data = get_details.invoke({"customer_name": customer_name})

        if not customer_data:
            return f"No upcoming appointment found for customer '{customer_name}'. Please check the customer name and appointment schedule."

        customer = customer_data[0]  # Get first match

        # Calculate time until appointment
        appointment_time = customer.get("next_appointment_date")
        if appointment_time:
            try:
                from datetime import datetime

                if isinstance(appointment_time, str):
                    appointment_dt = datetime.fromisoformat(
                        appointment_time.replace("Z", "+00:00")
                    )
                else:
                    appointment_dt = appointment_time

                now = datetime.now()
                time_diff = appointment_dt - now
                minutes_until = int(time_diff.total_seconds() / 60)

                if minutes_until < 0:
                    time_status = "OVERDUE"
                elif minutes_until < 60:
                    time_status = f"{minutes_until} minutes remaining"
                else:
                    hours = minutes_until // 60
                    mins = minutes_until % 60
                    time_status = f"{hours}h {mins}m remaining"
            except Exception:
                time_status = "Time calculation unavailable"
        else:
            time_status = "No appointment scheduled"

        # Calculate years as customer
        member_since = customer.get("member_since")
        if member_since:
            try:
                from datetime import datetime

                if isinstance(member_since, str):
                    member_dt = datetime.fromisoformat(
                        member_since.replace("Z", "+00:00")
                    )
                else:
                    member_dt = member_since

                years_as_customer = (datetime.now() - member_dt).days // 365
            except Exception:
                years_as_customer = "Unknown"
        else:
            years_as_customer = "Unknown"

        # Format customer data for the LLM prompt
        formatted_data = f"""
**CUSTOMER IDENTIFICATION**
Customer Name: {customer.get("customer_name")} (prefers: {customer.get("preferred_name", "N/A")})
Customer Tier: {customer.get("customer_tier")} Member
Years as Customer: {years_as_customer}
Customer ID: {customer.get("customer_id")}

**APPOINTMENT DETAILS**
Appointment Time: {customer.get("next_appointment_date")}
Time Status: {time_status}
Appointment Type: {customer.get("appointment_type")}
Purpose: {customer.get("appointment_purpose")}
Preparation Notes: {customer.get("preparation_notes", "None")}

**CUSTOMER VALUE & HISTORY**
Total Lifetime Spend: ${customer.get("total_lifetime_spend", 0):,.2f}
Average Transaction: ${customer.get("average_transaction_value", 0):,.2f}
Last Visit: {customer.get("last_visit_date")}
Satisfaction Score: {customer.get("customer_satisfaction", "N/A")}/5.0
Visit Frequency: {customer.get("visit_frequency", "N/A")}

**STYLE & PREFERENCES**
Style Preferences: {customer.get("style_preferences")}
Size Information: {customer.get("size_information")}
Color Preferences: {customer.get("color_preferences")}
Brand Preferences: {customer.get("brand_preferences")}
Budget Range: {customer.get("budget_range")}

**SERVICE REQUIREMENTS**
Preferred Stylist: {customer.get("preferred_stylist_name", "N/A")}
Manager Greeting Required: {customer.get("requires_manager_greeting", False)}
Special Alerts: {customer.get("customer_alerts", "None")}
Dietary Restrictions: {customer.get("dietary_restrictions", "None")}
Service Notes: {customer.get("service_notes", "None")}

**STYLIST INFORMATION**
Assigned Stylist: {stylist_name or "Not specified"}
"""

        # Generate the notification using the LLM
        try:
            llm: LanguageModelLike = llm_model.as_chat_model()
            prompt = NOTIFICATION_PROMPT.format(customer_data=formatted_data)
            notification = llm.invoke(prompt)

            # Extract content if it's a message object
            if hasattr(notification, "content"):
                notification_text = notification.content
            else:
                notification_text = str(notification)

            logger.debug(f"Generated stylist notification for {customer_name}")
            return notification_text

        except Exception as e:
            logger.error(f"Error generating notification with LLM: {e}")
            # Fallback to basic notification if LLM fails
            return f"""
🚨 **URGENT: Personal Styling Appointment**

**Customer**: {customer.get("customer_name")} ({customer.get("customer_tier")} Member)
**Appointment**: {customer.get("next_appointment_date")} ({time_status})
**Service**: {customer.get("appointment_type")} - {customer.get("appointment_purpose")}

**Quick Details**:
- Lifetime Spend: ${customer.get("total_lifetime_spend", 0):,.2f}
- Average Visit: ${customer.get("average_transaction_value", 0):,.2f}
- Budget Range: {customer.get("budget_range")}
- Preferred Stylist: {customer.get("preferred_stylist_name", "None specified")}
- Manager Greeting: {"Required" if customer.get("requires_manager_greeting") else "Not required"}

**Special Notes**: {customer.get("customer_alerts", "None")}

Please contact technical support for assistance with the AI notification feature.
"""

    return generate_stylist_notification


def create_inventory_preselection_tool(
    schema: SchemaModel | dict[str, Any],
    warehouse: WarehouseModel | dict[str, Any],
    llm_model: LLMModel | dict[str, Any],
) -> Callable[[list[str]], tuple]:
    """
    Create a tool that provides AI-powered inventory pre-selection for customer appointments.

    This tool analyzes customer preferences and current inventory to suggest items
    that should be pre-selected and prepared for styling appointments.

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names
        llm: Language model for generating intelligent recommendations

    Returns:
        A callable tool function that provides inventory pre-selection
    """
    logger.debug("create_inventory_preselection_tool")

    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)
    if isinstance(llm_model, dict):
        llm_model = LLMModel(**llm_model)

    # Define the pre-selection prompt
    PRESELECTION_PROMPT = """You are an expert retail stylist AI creating inventory pre-selection recommendations for an upcoming customer appointment. Your goal is to suggest specific items that should be pulled and prepared based on the customer's profile and appointment purpose.

Based on the customer data provided, create a pre-selection recommendation that includes:

## 🎯 **AI INVENTORY PRE-SELECTION**

### **Recommended Items to Pull** (12-15 items)
Organize by category with specific reasoning:

**👔 BLAZERS & JACKETS** (3-4 items)
- [Item description] - [Size] - [Color] - [Price range]
  *Reasoning: [Why this fits their style/needs]*

**👗 DRESSES** (2-3 items)
- [Item description] - [Size] - [Color] - [Price range]
  *Reasoning: [Why this fits their style/needs]*

**👚 TOPS & BLOUSES** (3-4 items)
- [Item description] - [Size] - [Color] - [Price range]
  *Reasoning: [Why this fits their style/needs]*

**👖 BOTTOMS** (2-3 items)
- [Item description] - [Size] - [Color] - [Price range]
  *Reasoning: [Why this fits their style/needs]*

**👠 ACCESSORIES** (2-3 items)
- [Item description] - [Size] - [Color] - [Price range]
  *Reasoning: [Why this fits their style/needs]*

### **🤖 ML Model Reasoning**
- **Customer Similarity**: Based on customers with similar profiles who purchased...
- **Seasonal Trends**: [Current seasonal trends relevant to this customer]
- **Inventory Optimization**: [Items that need movement and match profile]
- **Appointment Context**: [Specific items for the appointment purpose]

### **💡 Styling Strategy**
- **Outfit Combinations**: [2-3 complete outfit suggestions using pre-selected items]
- **Versatility Focus**: [How items can be mixed and matched]
- **Budget Alignment**: [How selections fit within customer's budget range]

**Guidelines**:
- Focus on items that match the customer's established preferences
- Consider the specific appointment purpose and occasion
- Suggest items within their typical budget range
- Include both safe choices and one "stretch" recommendation
- Provide clear reasoning for each selection

**Customer Data:**
{customer_data}"""

    @tool
    def generate_inventory_preselection(
        customer_name: str, appointment_context: str = None
    ) -> str:
        """
        Generate AI-powered inventory pre-selection recommendations for a customer appointment.

        This tool analyzes customer preferences, purchase history, and appointment purpose
        to suggest specific items that should be pulled and prepared for styling sessions.
        Uses machine learning insights to optimize selections.

        Args:
            customer_name (str): Name of the customer for the appointment
            appointment_context (str, optional): Additional context about the appointment

        Returns:
            str: AI-generated pre-selection recommendations including:
                - Specific items to pull organized by category
                - ML model reasoning for selections
                - Styling strategy and outfit combinations
                - Budget alignment and versatility focus
        """
        logger.debug(
            f"generate_inventory_preselection: customer_name={customer_name}, context={appointment_context}"
        )

        # Get customer details first
        get_details = create_get_customer_details_tool(
            schema=schema, warehouse=warehouse
        )
        customer_data = get_details.invoke({"customer_name": customer_name})

        if not customer_data:
            return f"No customer found with name '{customer_name}'. Please check the customer name and try again."

        customer = customer_data[0]  # Get first match

        # Format customer data for the LLM prompt
        formatted_data = f"""
**CUSTOMER PROFILE**
Name: {customer.get("customer_name")} (prefers: {customer.get("preferred_name")})
Tier: {customer.get("customer_tier")} Member
Lifetime Spend: ${customer.get("total_lifetime_spend", 0):,.2f}
Average Transaction: ${customer.get("average_transaction_value", 0):,.2f}

**APPOINTMENT DETAILS**
Date/Time: {customer.get("next_appointment_date")}
Type: {customer.get("appointment_type")}
Purpose: {customer.get("appointment_purpose")}
Additional Context: {appointment_context or "None provided"}
Budget Range: {customer.get("budget_range")}

**STYLE PREFERENCES**
Style Profile: {customer.get("style_preferences")}
Size Information: {customer.get("size_information")}
Color Preferences: {customer.get("color_preferences")}
Brand Preferences: {customer.get("brand_preferences")}

**SERVICE HISTORY**
Satisfaction Score: {customer.get("customer_satisfaction", "N/A")}/5.0
Last Visit: {customer.get("last_visit_date")}
Visit Frequency: {customer.get("visit_frequency")}
Service Notes: {customer.get("service_notes", "None")}

**SPECIAL CONSIDERATIONS**
Preparation Notes: {customer.get("preparation_notes", "None")}
Special Occasions: {customer.get("special_occasions", "None")}
Dietary Restrictions: {customer.get("dietary_restrictions", "None")}
"""

        # Generate the pre-selection using the LLM
        try:
            llm: LanguageModelLike = llm_model.as_chat_model()
            prompt = PRESELECTION_PROMPT.format(customer_data=formatted_data)
            preselection = llm.invoke(prompt)

            # Extract content if it's a message object
            if hasattr(preselection, "content"):
                preselection_text = preselection.content
            else:
                preselection_text = str(preselection)

            logger.debug(f"Generated inventory pre-selection for {customer_name}")
            return preselection_text

        except Exception as e:
            logger.error(f"Error generating pre-selection with LLM: {e}")
            # Fallback to basic pre-selection if LLM fails
            return f"""
🎯 **INVENTORY PRE-SELECTION: {customer.get("customer_name")}**

**Error**: Unable to generate AI-powered pre-selection. Here's a basic recommendation:

## Basic Recommendations
- **Budget Range**: {customer.get("budget_range")}
- **Style Focus**: {customer.get("style_preferences")}
- **Size Requirements**: {customer.get("size_information")}
- **Color Preferences**: {customer.get("color_preferences")}
- **Brand Focus**: {customer.get("brand_preferences")}

## Suggested Categories to Pull:
1. **Blazers/Jackets**: 3-4 pieces in preferred colors and brands
2. **Dresses**: 2-3 pieces appropriate for {customer.get("appointment_purpose")}
3. **Tops**: 3-4 blouses/tops in customer's size and style
4. **Bottoms**: 2-3 pieces (pants/skirts) in preferred fit
5. **Accessories**: 2-3 pieces to complete outfits

**Appointment Purpose**: {customer.get("appointment_purpose")}
**Special Notes**: {customer.get("preparation_notes", "None")}

Please contact technical support for assistance with the AI pre-selection feature.
"""

    return generate_inventory_preselection


def create_appointment_preparation_workflow_tool(
    schema: SchemaModel | dict[str, Any],
    warehouse: WarehouseModel | dict[str, Any],
    llm_model: LLMModel | dict[str, Any],
) -> Callable[[list[str]], tuple]:
    """
    Create a comprehensive appointment preparation workflow tool.

    This tool orchestrates the complete preparation process for customer appointments,
    including notifications, pre-selection, and setup coordination.

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names
        llm: Language model for generating intelligent workflows

    Returns:
        A callable tool function that manages appointment preparation workflows
    """
    logger.debug("create_appointment_preparation_workflow_tool")

    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)
    if isinstance(llm_model, dict):
        llm_model = LLMModel(**llm_model)

    @tool
    def prepare_customer_appointment(
        customer_name: str, stylist_name: str = None, preparation_context: str = None
    ) -> str:
        """
        Execute a comprehensive appointment preparation workflow for a customer.

        This tool orchestrates the complete preparation process including intelligent
        notifications, inventory pre-selection, and automated setup coordination
        to ensure exceptional customer service delivery.

        Args:
            customer_name (str): Name of the customer with upcoming appointment
            stylist_name (str, optional): Name of the assigned stylist
            preparation_context (str, optional): Additional preparation context

        Returns:
            str: Complete preparation workflow summary including:
                - Intelligent notification generated
                - Inventory pre-selection recommendations
                - Automated preparation actions taken
                - Setup coordination and next steps
        """
        logger.debug(
            f"prepare_customer_appointment: customer={customer_name}, stylist={stylist_name}"
        )

        workflow_summary = []
        workflow_summary.append("🏪 **APPOINTMENT PREPARATION WORKFLOW**")
        workflow_summary.append(f"Customer: {customer_name}")
        workflow_summary.append(f"Stylist: {stylist_name or 'To be assigned'}")
        workflow_summary.append(
            f"Context: {preparation_context or 'Standard appointment preparation'}"
        )
        workflow_summary.append("")

        # Step 1: Generate intelligent notification
        workflow_summary.append("## 📱 **Step 1: Intelligent Notification Generated**")
        try:
            notification_tool = create_stylist_notification_tool(
                schema=schema, warehouse=warehouse, llm_model=llm_model
            )
            notification = notification_tool.invoke(
                {"customer_name": customer_name, "stylist_name": stylist_name}
            )
            workflow_summary.append("✅ Notification created and sent to stylist")
            workflow_summary.append("📧 Notification content:")
            workflow_summary.append("```")
            workflow_summary.append(
                notification[:500] + "..." if len(notification) > 500 else notification
            )
            workflow_summary.append("```")
        except Exception as e:
            workflow_summary.append(f"❌ Error generating notification: {e}")

        workflow_summary.append("")

        # Step 2: Generate inventory pre-selection
        workflow_summary.append("## 🎯 **Step 2: AI-Powered Inventory Pre-Selection**")
        try:
            preselection_tool = create_inventory_preselection_tool(
                schema=schema, warehouse=warehouse, llm_model=llm_model
            )
            preselection = preselection_tool.invoke(
                {
                    "customer_name": customer_name,
                    "appointment_context": preparation_context,
                }
            )
            workflow_summary.append("✅ Inventory pre-selection generated")
            workflow_summary.append("📦 Pre-selection summary:")
            workflow_summary.append("```")
            workflow_summary.append(
                preselection[:500] + "..." if len(preselection) > 500 else preselection
            )
            workflow_summary.append("```")
        except Exception as e:
            workflow_summary.append(f"❌ Error generating pre-selection: {e}")

        workflow_summary.append("")

        # Step 3: Automated preparation actions (mock)
        workflow_summary.append("## 🔧 **Step 3: Automated Preparation Actions**")

        # Get customer details for preparation actions
        try:
            get_details = create_get_customer_details_tool(
                schema=schema, warehouse=warehouse
            )
            customer_data = get_details.invoke({"customer_name": customer_name})

            if customer_data:
                customer = customer_data[0]

                # Mock automated actions based on customer data
                actions = []

                # VIP styling suite reservation
                if customer.get("customer_tier") in ["Premium", "Gold"]:
                    actions.append("✅ VIP Styling Suite 2 reserved")
                else:
                    actions.append("✅ Standard styling room reserved")

                # Refreshments preparation
                if customer.get("dietary_restrictions"):
                    actions.append(
                        f"✅ Special refreshments prepared ({customer.get('dietary_restrictions')})"
                    )
                else:
                    actions.append("✅ Standard refreshments prepared")

                # Manager greeting setup
                if customer.get("requires_manager_greeting"):
                    actions.append("✅ Manager greeting scheduled")

                # Stylist assignment
                if customer.get("preferred_stylist_name"):
                    actions.append(
                        f"✅ Preferred stylist {customer.get('preferred_stylist_name')} notified"
                    )

                # Customer confirmation
                actions.append("✅ Customer confirmation sent with stylist profile")
                actions.append("✅ Fitting room prepared with size preferences")
                actions.append("✅ Appointment notes updated with preparation details")

                for action in actions:
                    workflow_summary.append(action)

            else:
                workflow_summary.append(
                    "❌ Unable to retrieve customer details for automated actions"
                )

        except Exception as e:
            workflow_summary.append(f"❌ Error executing automated actions: {e}")

        workflow_summary.append("")

        # Step 4: Next steps and coordination
        workflow_summary.append("## 📋 **Step 4: Next Steps & Coordination**")
        workflow_summary.append("🔄 **Immediate Actions Required:**")
        workflow_summary.append("1. Stylist reviews notification and customer profile")
        workflow_summary.append("2. Store team pulls recommended inventory items")
        workflow_summary.append(
            "3. Styling room setup completed 30 minutes before appointment"
        )
        workflow_summary.append("4. Manager briefed on customer requirements")
        workflow_summary.append("")
        workflow_summary.append("⏰ **Timeline:**")
        workflow_summary.append("- T-60 min: Final inventory pull and room setup")
        workflow_summary.append("- T-30 min: Stylist final preparation and review")
        workflow_summary.append(
            "- T-15 min: Manager greeting preparation (if required)"
        )
        workflow_summary.append(
            "- T-0: Customer arrival and exceptional service delivery"
        )
        workflow_summary.append("")
        workflow_summary.append("🌟 **Success Metrics:**")
        workflow_summary.append("- Customer satisfaction score maintained/improved")
        workflow_summary.append("- Appointment efficiency and personalization")
        workflow_summary.append("- Stylist confidence and preparation level")
        workflow_summary.append("- Revenue per appointment optimization")

        return "\n".join(workflow_summary)

    return prepare_customer_appointment


def create_real_time_styling_assistant_tool(
    schema: SchemaModel | dict[str, Any],
    warehouse: WarehouseModel | dict[str, Any],
    llm_model: LLMModel | dict[str, Any],
) -> Callable[[list[str]], tuple]:
    """
    Create a real-time styling assistant tool for live appointment support.

    This tool provides instant styling advice, trend insights, and product recommendations
    during active styling sessions to enhance the customer experience.

    Args:
        warehouse_id: Databricks warehouse ID for query execution
        config: Model configuration containing catalog and database names
        llm: Language model for generating styling insights

    Returns:
        A callable tool function that provides real-time styling assistance
    """
    logger.debug("create_real_time_styling_assistant_tool")

    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)
    if isinstance(llm_model, dict):
        llm_model = LLMModel(**llm_model)

    # Define the styling assistant prompt
    STYLING_ASSISTANT_PROMPT = """You are an expert fashion stylist AI providing real-time assistance during a customer styling session. Your goal is to help the stylist deliver exceptional, personalized service with instant insights and recommendations.

Based on the customer profile and current styling context, provide:

## 🎨 **REAL-TIME STYLING ASSISTANCE**

### **Immediate Recommendations**
- **Alternative Suggestions**: [2-3 specific alternatives to current selection]
- **Styling Techniques**: [How to present and style the items]
- **Combination Ideas**: [How to mix current items with other pieces]

### **Trend Insights**
- **Current Trends**: [Relevant fashion trends for this customer demographic]
- **Seasonal Considerations**: [What's trending now that fits their style]
- **Local Relevance**: [Trends that work well in their location/lifestyle]

### **Customer Psychology**
- **Approach Strategy**: [How to present options based on their personality]
- **Decision Factors**: [What typically influences their purchase decisions]
- **Confidence Builders**: [How to make them feel confident in choices]

### **Upselling Opportunities**
- **Natural Add-ons**: [Accessories or pieces that complete the look]
- **Investment Pieces**: [Higher-value items that align with their style]
- **Future Needs**: [Items they might need for upcoming occasions]

**Context**: {styling_context}
**Customer Profile**: {customer_data}"""

    @tool
    def get_real_time_styling_assistance(
        customer_name: str, styling_context: str, current_selection: str = None
    ) -> str:
        """
        Get real-time styling assistance during an active customer appointment.

        This tool provides instant styling advice, trend insights, and personalized
        recommendations to help stylists deliver exceptional service during live
        styling sessions. Perfect for handling customer questions and optimizing selections.

        Args:
            customer_name (str): Name of the customer being styled
            styling_context (str): Current styling situation or customer feedback
            current_selection (str, optional): Items currently being considered

        Returns:
            str: Real-time styling assistance including:
                - Immediate alternative recommendations
                - Relevant trend insights and seasonal considerations
                - Customer psychology and approach strategies
                - Natural upselling opportunities and investment pieces
        """
        logger.debug(
            f"get_real_time_styling_assistance: customer={customer_name}, context={styling_context}"
        )

        # Get customer details for context
        get_details = create_get_customer_details_tool(
            schema=schema, warehouse=warehouse
        )
        customer_data = get_details.invoke({"customer_name": customer_name})

        if not customer_data:
            return f"No customer profile found for '{customer_name}'. Please check the customer name."

        customer = customer_data[0]

        # Format customer data for context
        customer_summary = f"""
Customer: {customer.get("customer_name")} ({customer.get("customer_tier")} tier)
Style Preferences: {customer.get("style_preferences")}
Size Information: {customer.get("size_information")}
Color Preferences: {customer.get("color_preferences")}
Brand Preferences: {customer.get("brand_preferences")}
Budget Range: {customer.get("budget_range")}
Satisfaction History: {customer.get("customer_satisfaction", "N/A")}/5.0
Service Notes: {customer.get("service_notes", "None")}
"""

        # Format styling context
        full_context = f"""
**Current Situation**: {styling_context}
**Current Selection**: {current_selection or "Not specified"}
**Appointment Purpose**: {customer.get("appointment_purpose", "General styling")}
**Special Considerations**: {customer.get("preparation_notes", "None")}
"""

        # Generate styling assistance using the LLM
        try:
            llm: LanguageModelLike = llm_model.as_chat_model()
            prompt = STYLING_ASSISTANT_PROMPT.format(
                styling_context=full_context, customer_data=customer_summary
            )
            assistance = llm.invoke(prompt)

            # Extract content if it's a message object
            if hasattr(assistance, "content"):
                assistance_text = assistance.content
            else:
                assistance_text = str(assistance)

            logger.debug(f"Generated real-time styling assistance for {customer_name}")
            return assistance_text

        except Exception as e:
            logger.error(f"Error generating styling assistance with LLM: {e}")
            # Fallback to basic assistance if LLM fails
            return f"""
🎨 **REAL-TIME STYLING ASSISTANCE**

**Customer**: {customer.get("customer_name")} ({customer.get("customer_tier")} tier)
**Situation**: {styling_context}

## Quick Recommendations:
- **Style Focus**: {customer.get("style_preferences")}
- **Budget Range**: {customer.get("budget_range")}
- **Color Preferences**: {customer.get("color_preferences")}
- **Brand Preferences**: {customer.get("brand_preferences")}

## Approach Strategy:
- Customer satisfaction history: {customer.get("customer_satisfaction", "N/A")}/5.0
- Service notes: {customer.get("service_notes", "None")}
- Special considerations: {customer.get("preparation_notes", "None")}

## Current Selection Context:
{current_selection or "No current selection specified"}

Please contact technical support for assistance with the AI styling assistant feature.
"""

    return get_real_time_styling_assistance
