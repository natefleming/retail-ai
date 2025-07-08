from typing import Any, Callable

from langchain_core.tools import tool as create_tool
from loguru import logger

from dao_ai.config import CompositeVariableModel, ToolModel, UnityCatalogFunctionModel
from dao_ai.tools.core import BaseTool


def insert_coffee_order_tool(
    host: CompositeVariableModel | dict[str, Any],
    token: CompositeVariableModel | dict[str, Any],
    tool: ToolModel | dict[str, Any],
) -> Callable[[list[str]], tuple]:
    logger.debug(
        f"Creating insert_coffee_order tool with host: {host}, token: {token}, tool: {tool}"
    )
    if isinstance(host, dict):
        host = CompositeVariableModel(**host)
    if isinstance(token, dict):
        token = CompositeVariableModel(**token)
    if isinstance(tool, dict):
        tool = ToolModel(**tool)

    @create_tool
    def insert_coffee_order(coffee_name: str, size: str, session_id: str) -> str:
        """
        Place a coffee order for a customer. Use this tool when a customer wants to order coffee or other beverages from the menu.

        This tool records the order in the system and returns a confirmation message with order details.
        Call this tool when customers say things like "I'd like to order", "Can I get a", "I want", or similar ordering language.

        Args:
          coffee_name (str): The exact name of the coffee/beverage from the menu (e.g., "Cappuccino", "Latte", "Mocha")
          size (str): The size of the drink - must be "Medium", "Large", or "N/A" for single-size items
          session_id (str): The unique session ID for this customer conversation

        Returns:
          str: Order confirmation message with details and next steps for the customer
        """

        unity_catalog_function: UnityCatalogFunctionModel | dict[str, Any] = (
            tool.function
        )
        if isinstance(unity_catalog_function, dict):
            unity_catalog_function = UnityCatalogFunctionModel(**unity_catalog_function)

        unity_catalog_tool: BaseTool = next(
            iter(unity_catalog_function.as_tools() or []), None
        )
        logger.debug(f"Invoking Unity Catalog tool: {unity_catalog_tool.name}")
        result: str = unity_catalog_tool.invoke(
            {
                "host": host.as_value(),
                "token": token.as_value(),
                "coffee_name": coffee_name,
                "size": size,
                "session_id": session_id,
            }
        )
        logger.debug(f"Order result: {result}")
        return result

    return insert_coffee_order
