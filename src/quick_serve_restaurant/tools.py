from typing import Any, Callable

from langchain_core.tools import tool as create_tool
from loguru import logger

from dao_ai.config import CompositeVariableModel, ToolModel, UnityCatalogFunctionModel


def insert_coffee_order_tool(
    tool: ToolModel | dict[str, Any],
    host: CompositeVariableModel | dict[str, Any],
    client_id: CompositeVariableModel | dict[str, Any] | None = None,
    client_secret: CompositeVariableModel | dict[str, Any] | None = None,
) -> Callable[[list[str]], tuple]:
    logger.debug(
        f"Creating insert_coffee_order_tool with tool: {tool}, "
        f"host: {host}, client_id: {client_id}"
    )
    if isinstance(tool, dict):
        tool = ToolModel(**tool)
    if isinstance(host, dict):
        host = CompositeVariableModel(**host)
    if isinstance(client_id, dict):
        client_id = CompositeVariableModel(**client_id)
    if isinstance(client_secret, dict):
        client_secret = CompositeVariableModel(**client_secret)

    if client_id is None or client_secret is None:
        raise ValueError("Both 'client_id' and 'client_secret' must be provided.")

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

        from databricks_langchain import DatabricksFunctionClient

        client = DatabricksFunctionClient()
        logger.debug(
            f"Calling Unity Catalog function directly: {unity_catalog_function.full_name}"
        )

        result = client.execute_function(
            function_name=unity_catalog_function.full_name,
            parameters={
                "host": host.as_value(),
                "client_id": client_id.as_value(),
                "client_secret": client_secret.as_value(),
                "coffee_name": coffee_name,
                "size": size,
                "session_id": session_id,
            },
        )

        # Handle the result from execute_function
        if hasattr(result, "error") and result.error:
            logger.error(f"Unity Catalog function error: {result.error}")
            raise RuntimeError(f"Function execution failed: {result.error}")

        result_value = result.value if hasattr(result, "value") else str(result)
        logger.debug(f"Order result: {result_value}")
        return result_value

    return insert_coffee_order
