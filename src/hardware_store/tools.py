from typing import Any, Callable, Sequence

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    StatementResponse,
    StatementState,
)
from databricks_langchain import (
    DatabricksVectorSearch,
)
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores.base import VectorStore
from loguru import logger

from dao_ai.config import RetrieverModel, SchemaModel, VectorStoreModel, WarehouseModel


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


def find_product_details_by_description_tool(
    retriever: RetrieverModel | dict[str, Any],
) -> Callable[[str, str], Sequence[Document]]:
    """
    Create a tool for finding product details using vector search with classification filtering.

    This factory function generates a specialized search tool that combines semantic vector search
    with categorical filtering to improve product discovery in retail applications. It enables
    natural language product lookups with classification-based narrowing of results.

    Args:
        retriever: Configuration details for the vector search retriever, including:
            - vector_store: Dictionary with 'endpoint_name' and 'index' for vector search
            - columns: List of columns to retrieve from the vector store
            - search_parameters: Additional parameters for customizing the search behavior
    Returns:
        A callable tool function that performs vector search for product details
        based on natural language descriptions and classification filters
    """
    if isinstance(retriever, dict):
        retriever = RetrieverModel(**retriever)

    logger.debug("find_product_details_by_description_tool")

    @tool
    @mlflow.trace(span_type="RETRIEVER", name="vector_search")
    def find_product_details_by_description(content: str) -> Sequence[Document]:
        """
        Find products matching a description.

        This tool performs semantic search over product data to find items that match
        the given description text

        Args:
          content (str): Natural language description of the product(s) to find

        Returns:
          Sequence[Document]: A list of matching product documents with relevant metadata
        """
        logger.debug(f"find_product_details_by_description: content={content}")

        vector_store: VectorStoreModel = retriever.vector_store

        # Initialize the Vector Search client with endpoint and index configuration
        vector_search: VectorStore = DatabricksVectorSearch(
            endpoint=vector_store.endpoint.name,
            index_name=vector_store.index.full_name,
            columns=retriever.columns,
            client_args={},
            workspace_client=vector_store.workspace_client,
        )

        search_params: dict[str, Any] = retriever.search_parameters.model_dump()
        if "num_results" in search_params:
            search_params["k"] = search_params.pop("num_results")

        documents: Sequence[Document] = vector_search.similarity_search(
            query=content, **search_params
        )

        logger.debug(f"found {len(documents)} documents")
        return documents

    return find_product_details_by_description


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
