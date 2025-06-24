from typing import Any, Callable

from langchain_core.tools import tool as create_tool

from dao_ai.config import CompositeVariableModel, UnityCatalogFunctionModel


def insert_coffee_order_tool(
    host: CompositeVariableModel | dict[str, Any],
    token: CompositeVariableModel | dict[str, Any],
    tool: UnityCatalogFunctionModel | dict[str, Any],
) -> Callable[[list[str]], tuple]:
    if isinstance(host, dict):
        host = CompositeVariableModel(**host)
    if isinstance(token, dict):
        token = CompositeVariableModel(**token)
    if isinstance(tool, dict):
        tool = UnityCatalogFunctionModel(**tool)

    @create_tool
    def insert_coffee_order(coffee_name: str, size: str, thread_id: str) -> str:
        """
        Place a coffee order for a customer. Use this tool when a customer wants to order coffee or other beverages from the menu.

        This tool records the order in the system and returns a confirmation message with order details.
        Call this tool when customers say things like "I'd like to order", "Can I get a", "I want", or similar ordering language.

        Args:
          coffee_name (str): The exact name of the coffee/beverage from the menu (e.g., "Cappuccino", "Latte", "Mocha")
          size (str): The size of the drink - must be "Medium", "Large", or "N/A" for single-size items
          thread_id (str): The unique session ID for this customer conversation

        Returns:
          str: Order confirmation message with details and next steps for the customer
        """

        result: str = tool.as_tool().invoke(
            {
                "host": host,
                "token": token,
                "coffee_name": coffee_name,
                "size": size,
                "session_id": thread_id,
            }
        )
        return result

        # uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)

        # insert_tool = uc_toolkit.tools[0]
        # params_dict = {
        #   "host": host,
        #   "token": token,
        #   "coffee_name": coffee_name,
        #   "size": size,
        #   "session_id": thread_id,
        # }
        # return insert_tool(params_dict)

    return insert_coffee_order
