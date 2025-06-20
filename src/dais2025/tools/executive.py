import os
from typing import Any, Callable

from databricks_ai_bridge.genie import GenieResponse
from databricks_langchain.genie import Genie
from langchain_core.tools import tool
from loguru import logger

from dao_ai.config import GenieRoomModel


def create_executive_insights_tool(
    genie_room: GenieRoomModel | dict[str, Any],
) -> Callable[[str], GenieResponse]:
    """Create a Genie tool for natural language queries over retail data."""

    logger.debug("create_executive_insights_tool")

    if isinstance(genie_room, dict):
        genie_room = GenieRoomModel(**genie_room)

    space_id: str = genie_room.space_id or os.environ.get("DATABRICKS_GENIE_SPACE_ID")

    genie: Genie = Genie(
        space_id=space_id,
    )

    @tool
    def genie_executive_insights(question: str) -> GenieResponse:
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

    return genie_executive_insights
