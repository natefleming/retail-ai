"""
Store Tools

This module contains tool creation functions for store-related operations including
store lookup by number/ID, store location search, and store number extraction.
"""

from typing import Any, Callable, Sequence

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse, StatementState
from databricks_langchain import DatabricksVectorSearch
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import tool
from langchain_core.vectorstores.base import VectorStore
from loguru import logger

from dais2025.models import StoreInfo
from dao_ai.config import (
    RetrieverModel,
    SchemaModel,
    WarehouseModel,
)


def find_store_details_by_location_tool(
    retriever: RetrieverModel | dict[str, Any],
) -> Callable[[str, str], Sequence[Document]]:
    """
    Create a tool for finding store details using vector search with location-based filtering.

    This factory function generates a specialized search tool that combines semantic vector search
    to find stores based on location descriptions, store names, or geographic references. It enables
    natural language store lookups for retail applications.

    Args:
        endpoint_name: Name of the Databricks Vector Search endpoint to query
        index_name: Name of the specific vector index containing store information
        columns: List of column names to include in the search results
        k: Maximum number of results to return (default: 10)

    Returns:
        A callable tool function that performs semantic store search
    """
    logger.debug("find_store_details_by_location_tool")

    if isinstance(retriever, dict):
        retriever = RetrieverModel(**retriever)

    @tool
    @mlflow.trace(span_type="RETRIEVER", name="store_vector_search")
    def find_store_details_by_location(content: str) -> Sequence[Document]:
        """
        Find store details using semantic vector search based on location descriptions.

        This tool performs semantic search across the store directory to find locations that match
        the provided description, including store names, addresses, cities, or geographic references.
        It's particularly useful for natural language store discovery and location services.

        Args:
            content: Natural language description of the store location to search for

        Returns:
            Sequence of Document objects containing matching store information
        """
        logger.debug(f"find_store_details_by_location: {content}")

        # Initialize the vector search client
        vector_search: VectorStore = DatabricksVectorSearch(
            endpoint=retriever.vector_store.endpoint.name,
            index_name=retriever.vector_store.index.full_name,
            columns=retriever.columns,
            client_args={},
        )

        search_params: dict[str, Any] = retriever.search_parameters.model_dump()
        if "num_results" in search_params:
            search_params["k"] = search_params.pop("num_results")

        documents: Sequence[Document] = vector_search.similarity_search(
            query=content, **search_params
        )

        logger.debug(f"found {len(documents)} documents")
        return documents

    return find_store_details_by_location


def create_find_store_by_number_tool(
    schema: SchemaModel | dict[str, Any], warehouse: WarehouseModel | dict[str, Any]
) -> Callable[[list[str]], tuple]:
    """Create a Unity Catalog tool for finding stores by store number/ID."""

    if isinstance(schema, dict):
        schema = SchemaModel(**schema)
    if isinstance(warehouse, dict):
        warehouse = WarehouseModel(**warehouse)

    @tool
    def find_store_by_number(store_numbers: list[str]) -> tuple:
        """
        Find store details by one or more store numbers or IDs.
        This tool retrieves detailed information about stores based on their store numbers.

        Args:
            store_numbers (list[str]): One or more store identifiers to retrieve.
                                     Store numbers are typically 3-4 digit numeric values.
                                     Examples: ["101", "205", "1001"]

        Returns:
            (tuple): A tuple containing store information with fields like:
                store_id STRING
                ,store_name STRING
                ,store_address STRING
                ,store_city STRING
                ,store_state STRING
                ,store_zipcode STRING
                ,store_country STRING
                ,store_phone STRING
                ,store_email STRING
                ,store_manager_id STRING
                ,opening_date DATE
                ,store_area_sqft DOUBLE
                ,is_open_24_hours BOOLEAN
                ,latitude DOUBLE
                ,longitude DOUBLE
                ,region_id STRING
        """
        logger.debug(f"find_store_by_number: {store_numbers}")

        # Convert list to SQL array format
        store_numbers_str = ", ".join([f"'{store_num}'" for store_num in store_numbers])

        # Execute the Unity Catalog function
        sql_query = f"""
            SELECT * FROM {schema.full_name}.find_store_by_number(ARRAY({store_numbers_str}))
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
            logger.debug(f"Found {len(df)} stores")
            return tuple(df.to_dict("records"))

        return ()

    return find_store_by_number


def create_store_number_extraction_tool(llm: LanguageModelLike) -> Callable[[str], str]:
    """
    Create a tool that leverages an LLM to extract store numbers from natural language text.

    In GenAI applications, this tool enables automated extraction of store numbers from
    customer queries, support tickets, or conversational inputs without requiring
    explicit structured input. This facilitates store lookups, location services, and
    store-specific assistance in conversational AI systems.

    Args:
        llm: Language model to use for store number extraction from unstructured text

    Returns:
        A callable tool function that extracts a list of store numbers from input text
    """
    logger.debug("create_store_number_extraction_tool")

    @tool
    def store_number_extraction(input: str) -> list[str]:
        """
        Extract store numbers from natural language text using an LLM.

        This tool analyzes unstructured text to identify and extract store number codes,
        enabling automated store identification from customer conversations or queries.

        Args:
            input: Natural language text that may contain store numbers

        Returns:
            List of extracted store numbers found in the input text
        """
        logger.debug(f"store_number_extraction: {input}")

        # Use the LLM with structured output to extract store numbers
        chain = llm.with_structured_output(StoreInfo)
        result = chain.invoke(
            f"Extract any store numbers from this text. Store numbers are typically 3-4 digit numeric values: {input}"
        )

        return result.store_numbers

    return store_number_extraction
