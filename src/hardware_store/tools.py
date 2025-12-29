from typing import Any, Callable

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    StatementResponse,
    StatementState,
)
from langchain_core.tools import tool
from loguru import logger

from dao_ai.config import SchemaModel, WarehouseModel


def create_reservation_tool() -> Callable[..., Any]:
    """
    Create a tool for making reservations.

    This factory function generates a tool that can be used to make reservations
    in a system. The tool can be customized with various parameters.

    Returns:
        A callable tool function that performs reservation operations
    """

    @tool
    def make_reservation(
        destination: str,
    ) -> str:
        """
        Make a reservation with the provided details.

        Args:
            reservation_details (dict[str, Any]): Details of the reservation to be made

        Returns:
            str: Confirmation message for the reservation
        """
        logger.debug(f"Making reservation with details: {destination}")
        return "Reservation made successfully!"

    return make_reservation


def create_find_product_by_sku_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def find_product_by_sku(skus: list[str]) -> tuple:
        """
        Find product details by one or more sku values.
        This tool retrieves detailed information about a product based on its SKU.

        Args: skus (list[str]): One or more unique identifiers for retrieve. It may help to use another tool to provide this value. SKU values are between 5-8 alpha numeric characters. SKUs can follow several patterns:
            - 5 digits (e.g., "89042")
            - 7 digits (e.g., "2029546")
            - 5 digits + 'D' for dropship items (e.g., "23238D")
            - 7 digits + 'D' for dropship items (e.g., "3004574D")
            - Alphanumeric codes (e.g., "WHDEOSMC01")
            - Product names (e.g., "Proud Veteran Garden Applique Flag")
            - Price-prefixed codes (e.g., "NK5.99")

        Examples:
            - "89042" (5-digit SKU)
            - "2029546" (7-digit SKU)
            - "23238D" (5-digit dropship SKU)
            - "3004574D" (7-digit dropship SKU)
            - "WHDEOSMC01" (alphanumeric SKU)
            - "NK5.99" (price-prefixed SKU)

        Returns: (tuple): A tuple containing (
            product_id BIGINT
            ,sku STRING
            ,upc STRING
            ,brand_name STRING
            ,product_name STRING
            ,merchandise_class STRING
            ,class_cd STRING
            ,description STRING
        )
        """
        logger.debug(f"find_product_by_sku: skus={skus}")

        w: WorkspaceClient = warehouse.workspace_client

        skus = ",".join([f"'{sku}'" for sku in skus])
        statement: str = f"""
            SELECT * FROM {schema.full_name}.find_product_by_sku(ARRAY({skus}))
        """
        logger.debug(statement)
        statement_response: StatementResponse = w.statement_execution.execute_statement(
            statement=statement, warehouse_id=warehouse.warehouse_id
        )
        while statement_response.status.state in [
            StatementState.PENDING,
            StatementState.RUNNING,
        ]:
            statement_response = w.statement_execution.get_statement(
                statement_response.statement_id
            )

        result_set: tuple = (
            statement_response.result.data_array if statement_response.result else None
        )

        logger.debug(f"find_product_by_sku: result_set={result_set}")

        return result_set

    return find_product_by_sku


def create_find_product_by_upc_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def find_product_by_upc(upcs: list[str]) -> tuple:
        """
        Find product details by one or more upc values.
        This tool retrieves detailed information about a product based on its UPC.

        Args: upcs (list[str]): One or more unique identifiers for retrieve. It may help to use another tool to provide this value. UPC values are between 10-16 alpha numeric characters.

        Returns: (tuple): A tuple containing (
            product_id BIGINT
            ,sku STRING
            ,upc STRING
            ,brand_name STRING
            ,product_name STRING
            ,merchandise_class STRING
            ,class_cd STRING
            ,description STRING
        )
        """
        logger.debug(f"find_product_by_upc: upcs={upcs}")

        w: WorkspaceClient = warehouse.workspace_client

        upcs = ",".join([f"'{upc}'" for upc in upcs])
        statement: str = f"""
            SELECT * FROM {schema.full_name}.find_product_by_upc(ARRAY({upcs}))
        """
        logger.debug(statement)
        statement_response: StatementResponse = w.statement_execution.execute_statement(
            statement=statement, warehouse_id=warehouse.warehouse_id
        )
        while statement_response.status.state in [
            StatementState.PENDING,
            StatementState.RUNNING,
        ]:
            statement_response = w.statement_execution.get_statement(
                statement_response.statement_id
            )

        result_set: tuple = (
            statement_response.result.data_array if statement_response.result else None
        )

        logger.debug(f"find_product_by_upc: result_set={result_set}")

        return result_set

    return find_product_by_upc


def create_find_inventory_by_sku_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def find_inventory_by_sku(skus: list[str]) -> tuple:
        """
        Find product details by one or more sku values.
        This tool retrieves detailed information about a product based on its SKU.

        Args: skus (list[str]): One or more unique identifiers for retrieve. It may help to use another tool to provide this value. SKU values are between 5-8 alpha numeric characters. SKUs can follow several patterns:
            - 5 digits (e.g., "89042")
            - 7 digits (e.g., "2029546")
            - 5 digits + 'D' for dropship items (e.g., "23238D")
            - 7 digits + 'D' for dropship items (e.g., "3004574D")
            - Alphanumeric codes (e.g., "WHDEOSMC01")
            - Product names (e.g., "Proud Veteran Garden Applique Flag")
            - Price-prefixed codes (e.g., "NK5.99")

        Examples:
            - "89042" (5-digit SKU)
            - "2029546" (7-digit SKU)
            - "23238D" (5-digit dropship SKU)
            - "3004574D" (7-digit dropship SKU)
            - "WHDEOSMC01" (alphanumeric SKU)
            - "NK5.99" (price-prefixed SKU)

        Returns: (tuple): A tuple containing (
            inventory_id BIGINT
            ,sku STRING
            ,upc STRING
            ,product_id STRING
            ,store STRING
            ,store_quantity INT
            ,warehouse STRING
            ,warehouse_quantity INT
            ,retail_amount FLOAT
            ,popularity_rating STRING
            ,department STRING
            ,aisle_location STRING
            ,is_closeout BOOLEAN
        )
        """
        logger.debug(f"find_inventory_by_sku: skus={skus}")

        w: WorkspaceClient = warehouse.workspace_client

        skus = ",".join([f"'{sku}'" for sku in skus])
        statement: str = f"""
            SELECT * FROM {schema.full_name}.find_inventory_by_sku(ARRAY({skus}))
        """
        logger.debug(statement)
        statement_response: StatementResponse = w.statement_execution.execute_statement(
            statement=statement, warehouse_id=warehouse.warehouse_id
        )
        while statement_response.status.state in [
            StatementState.PENDING,
            StatementState.RUNNING,
        ]:
            statement_response = w.statement_execution.get_statement(
                statement_response.statement_id
            )

        result_set: tuple = (
            statement_response.result.data_array if statement_response.result else None
        )

        logger.debug(f"find_inventory_by_sku: result_set={result_set}")

        return result_set

    return find_inventory_by_sku


def create_find_inventory_by_upc_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def find_inventory_by_upc(upcs: list[str]) -> tuple:
        """
        Find product details by one or more upc values.
        This tool retrieves detailed information about a product based on its SKU.

        Args: upcs (list[str]): One or more unique identifiers for retrieve. It may help to use another tool to provide this value. UPC values are between 10-16 alpha numeric characters.

        Returns: (tuple): A tuple containing (
            inventory_id BIGINT
            ,sku STRING
            ,upc STRING
            ,product_id STRING
            ,store STRING
            ,store_quantity INT
            ,warehouse STRING
            ,warehouse_quantity INT
            ,retail_amount FLOAT
            ,popularity_rating STRING
            ,department STRING
            ,aisle_location STRING
            ,is_closeout BOOLEAN
        )
        """
        logger.debug(f"find_inventory_by_upc: upcs={upcs}")

        w: WorkspaceClient = warehouse.workspace_client

        upcs = ",".join([f"'{upc}'" for upc in upcs])
        statement: str = f"""
            SELECT * FROM {schema.full_name}.find_inventory_by_upc(ARRAY({upcs}))
        """
        logger.debug(statement)
        statement_response: StatementResponse = w.statement_execution.execute_statement(
            statement=statement, warehouse_id=warehouse.warehouse_id
        )
        while statement_response.status.state in [
            StatementState.PENDING,
            StatementState.RUNNING,
        ]:
            statement_response = w.statement_execution.get_statement(
                statement_response.statement_id
            )

        result_set: tuple = (
            statement_response.result.data_array if statement_response.result else None
        )

        logger.debug(f"find_inventory_by_upc: result_set={result_set}")

        return result_set

    return find_inventory_by_upc


def create_find_store_inventory_by_sku_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[str, list[str]], tuple]:
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def find_store_inventory_by_sku(store: str, skus: list[str]) -> tuple:
        """
        Find product details by one or more sku values.
        This tool retrieves detailed information about a product based on its SKU.

        Args:
            store (str): The store to search for the inventory

            skus (list[str]): One or more unique identifiers for retrieve. It may help to use another tool to provide this value. SKU values are between 5-8 alpha numeric characters. SKUs can follow several patterns:
            - 5 digits (e.g., "89042")
            - 7 digits (e.g., "2029546")
            - 5 digits + 'D' for dropship items (e.g., "23238D")
            - 7 digits + 'D' for dropship items (e.g., "3004574D")
            - Alphanumeric codes (e.g., "WHDEOSMC01")
            - Product names (e.g., "Proud Veteran Garden Applique Flag")
            - Price-prefixed codes (e.g., "NK5.99")

        Examples:
            - "89042" (5-digit SKU)
            - "2029546" (7-digit SKU)
            - "23238D" (5-digit dropship SKU)
            - "3004574D" (7-digit dropship SKU)
            - "WHDEOSMC01" (alphanumeric SKU)
            - "NK5.99" (price-prefixed SKU)

        Returns: (tuple): A tuple containing (
            inventory_id BIGINT
            ,sku STRING
            ,upc STRING
            ,product_id STRING
            ,store STRING
            ,store_quantity INT
            ,warehouse STRING
            ,warehouse_quantity INT
            ,retail_amount FLOAT
            ,popularity_rating STRING
            ,department STRING
            ,aisle_location STRING
            ,is_closeout BOOLEAN
        )
        """
        logger.debug(f"find_store_inventory_by_sku: store={store}, sku={skus}")

        w: WorkspaceClient = warehouse.workspace_client

        skus = ",".join([f"'{sku}'" for sku in skus])
        statement: str = f"""
            SELECT * FROM {schema.full_name}.find_store_inventory_by_sku('{store}', ARRAY({skus}))
        """
        logger.debug(statement)
        statement_response: StatementResponse = w.statement_execution.execute_statement(
            statement=statement, warehouse_id=warehouse.warehouse_id
        )
        while statement_response.status.state in [
            StatementState.PENDING,
            StatementState.RUNNING,
        ]:
            statement_response = w.statement_execution.get_statement(
                statement_response.statement_id
            )

        result_set: tuple = (
            statement_response.result.data_array if statement_response.result else None
        )

        logger.debug(f"find_store_inventory_by_sku: result_set={result_set}")

        return result_set

    return find_store_inventory_by_sku


def create_find_store_inventory_by_upc_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[str, list[str]], tuple]:
    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def find_store_inventory_by_upc(store: str, upcs: list[str]) -> tuple:
        """
        Find product details by one or more sku values.
        This tool retrieves detailed information about a product based on its SKU.

        Args:
            store (str): The store to search for the inventory
            upcs (list[str]): One or more unique identifiers for retrieve. It may help to use another tool to provide this value. UPC values are between 10-16 alpha numeric characters.


        Returns: (tuple): A tuple containing (
            inventory_id BIGINT
            ,sku STRING
            ,upc STRING
            ,product_id STRING
            ,store STRING
            ,store_quantity INT
            ,warehouse STRING
            ,warehouse_quantity INT
            ,retail_amount FLOAT
            ,popularity_rating STRING
            ,department STRING
            ,aisle_location STRING
            ,is_closeout BOOLEAN
        )
        """
        logger.debug(f"find_store_inventory_by_upc: store={store}, upcs={upcs}")

        w: WorkspaceClient = warehouse.workspace_client

        upcs = ",".join([f"'{upc}'" for upc in upcs])
        statement: str = f"""
            SELECT * FROM {schema.full_name}.find_store_inventory_by_upc('{store}', ARRAY({upcs}))
        """
        logger.debug(statement)
        statement_response: StatementResponse = w.statement_execution.execute_statement(
            statement=statement, warehouse_id=warehouse.warehouse_id
        )
        while statement_response.status.state in [
            StatementState.PENDING,
            StatementState.RUNNING,
        ]:
            statement_response = w.statement_execution.get_statement(
                statement_response.statement_id
            )

        result_set: tuple = (
            statement_response.result.data_array if statement_response.result else None
        )

        logger.debug(f"find_store_inventory_by_upc: result_set={result_set}")

        return result_set

    return find_store_inventory_by_upc
