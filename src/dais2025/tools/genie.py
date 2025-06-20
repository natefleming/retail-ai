"""
Genie Tools

This module contains tool creation functions for Databricks Genie AI integration.
Genie provides natural language querying capabilities over configured data sources.
"""

import os
from typing import Any, Callable

from databricks_ai_bridge.genie import GenieResponse
from databricks_langchain.genie import Genie
from langchain_core.tools import tool
from loguru import logger

from dao_ai.config import GenieRoomModel


def create_genie_query_tool(
    genie_room: GenieRoomModel | dict[str, Any],
) -> Callable[[str], GenieResponse]:
    """Create a Genie tool for natural language queries over retail data."""

    logger.debug("create_genie_tool")

    if isinstance(genie_room, dict):
        genie_room = GenieRoomModel(**genie_room)

    space_id: str = genie_room.space_id or os.environ.get("DATABRICKS_GENIE_SPACE_ID")

    genie: Genie = Genie(
        space_id=space_id,
    )

    @tool
    def query_retail_data_with_genie(question: str) -> GenieResponse:
        """
        Query retail data using natural language through Databricks Genie AI.

        This tool can answer complex questions about:
        - Product inventory across all stores
        - Product details, specifications, and pricing
        - Store information, locations, and details
        - Cross-store inventory comparisons
        - Historical sales and trends

        Use this tool when you need to:
        - Answer complex analytical questions about retail data
        - Perform aggregations or calculations across multiple tables
        - Get insights that require joining inventory, products, and store data
        - Answer questions that go beyond simple lookups

        Examples of good questions for Genie:
        - "What are the top 5 best-selling Adidas products across all SF stores?"
        - "Which stores have the highest inventory turnover for sneakers?"
        - "Show me all out-of-stock items and their last restock dates"
        - "Compare inventory levels between downtown and marina locations"

        Args:
            question (str): Natural language question about retail data

        Returns:
            str: Genie's response with data insights and analysis
        """
        logger.debug(f"query_retail_data_with_genie: {question}")

        try:
            # Forward the question to Genie and return its response
            response: GenieResponse = genie.ask_question(question)
            return response

        except Exception as e:
            logger.error(f"Error querying Genie: {e}")
            return "I encountered an error while querying the retail data. Please try rephrasing your question or contact support if the issue persists."

    return query_retail_data_with_genie
