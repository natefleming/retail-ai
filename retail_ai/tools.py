import os
from io import StringIO
from typing import Any, Callable, Literal, Sequence

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    StatementResponse,
    StatementState,
)
from databricks_ai_bridge.genie import GenieResponse
from databricks_langchain import (
    DatabricksFunctionClient,
    DatabricksVectorSearch,
    UCFunctionToolkit,
)
from databricks_langchain.genie import Genie
from databricks_langchain.vector_search_retriever_tool import VectorSearchRetrieverTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores.base import VectorStore
from loguru import logger
from pydantic import BaseModel, Field
from unitycatalog.ai.core.base import FunctionExecutionResult

from retail_ai.catalog import full_name
from retail_ai.config import (
    BaseFunctionModel,
    FactoryFunctionModel,
    FunctionType,
    GenieRoomModel,
    McpFunctionModel,
    PythonFunctionModel,
    RetrieverModel,
    SchemaModel,
    ToolModel,
    UnityCatalogFunctionModel,
    WarehouseModel,
)
from retail_ai.utils import load_function


def find_allowable_classifications(schema: SchemaModel) -> Sequence[str]:
    """
    Retrieve the list of allowable product classifications from a Unity Catalog function.

    This function queries a predefined UDF in the Databricks Unity Catalog to get a list
    of valid product classifications that can be used for categorizing products and
    filtering search results.

    Args:
        w: Databricks WorkspaceClient instance for API access
        catalog_name: Name of the Unity Catalog containing the function
        schema_name: Name of the database/schema containing the function

    Returns:
        A sequence of strings representing valid product classifications

    Raises:
        Exception: If the Unity Catalog function execution fails
    """

    logger.debug(
        f"catalog_name={schema.catalog_name}, schema_name={schema.schema_name}"
    )

    w: WorkspaceClient = WorkspaceClient()

    client: DatabricksFunctionClient = DatabricksFunctionClient(client=w)

    # Execute the Unity Catalog function to retrieve classifications
    result: FunctionExecutionResult = client.execute_function(
        function_name=f"{schema.full_name}.find_allowable_product_classifications",
        parameters={},
    )

    # Handle any execution errors
    if result.error:
        raise Exception(result.error)

    # Parse the CSV result into a pandas DataFrame
    pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))
    # Extract the classification column as a list
    classifications: Sequence = pdf["classification"].tolist()

    logger.debug(f"classifications={classifications}")

    return classifications


def create_product_classification_tool(
    llm: LanguageModelLike,
    allowable_classifications: Sequence[str],
    k: int = 1,
) -> Callable[[str], list[str]]:
    """
    Create a tool that uses an LLM to classify product descriptions into predefined categories.

    This factory function generates a tool that leverages a language model to classify
    product descriptions into one or more of the allowable classifications. The number of
    classifications returned is determined by the 'k' parameter.

    Args:
        llm: Language model to use for classification
        allowable_classifications: List of valid classification categories
        k: Number of classifications to return (default: 1)

    Returns:
        A callable tool function that takes a product description and returns a list of classifications
    """
    logger.debug(
        f"create_product_classification_tool: allowable_classifications={allowable_classifications}"
    )

    # Define a Pydantic model to enforce valid classifications through type checking
    class Classifier(BaseModel):
        classifications: list[Literal[tuple(allowable_classifications)]] = Field(
            ...,
            description=f"The classifications of the product. Return {k} classifications from: {allowable_classifications}",
        )

    @tool
    def product_classification(input: str) -> list[str]:
        """
        This tool lets you extract classifications from a product description or prompt.
        This classification can be used to apply a filter during vector search lookup.

        Args:
            input (str): The input prompt to ask to classify the product

        Returns:
            list[str]: A list of {k} classifications for the product
        """
        logger.debug(f"product_classification: input={input}")
        # Configure the LLM to output in the structured Classifier format
        llm_with_tools: LanguageModelLike = llm.with_structured_output(Classifier)
        # Invoke the LLM to classify the input text
        classifications: list[str] = llm_with_tools.invoke(input=input).classifications

        # Ensure we return exactly k classifications
        if len(classifications) > k:
            classifications = classifications[:k]

        logger.debug(f"product_classification: classifications={classifications}")
        return classifications

    return product_classification


def find_product_details_by_description_tool(
    retriever: dict[str, Any],
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

        # Initialize the Vector Search client with endpoint and index configuration
        vector_search: VectorStore = DatabricksVectorSearch(
            endpoint=retriever.vector_store.endpoint_name,
            index_name=retriever.vector_store.index.full_name,
            columns=retriever.columns,
            client_args={},
        )

        documents: Sequence[Document] = vector_search.similarity_search(
            query=content, **retriever.search_parameters
        )

        logger.debug(f"found {len(documents)} documents")
        return documents

    return find_product_details_by_description


tool_registry: dict[str, BaseTool] = {}


def create_tools(tool_models: Sequence[ToolModel]) -> Sequence[BaseTool]:
    """
    Create a list of tools based on the provided configuration.

    This factory function generates a list of tools based on the specified configurations.
    Each tool is created according to its type and parameters defined in the configuration.

    Args:
        tool_configs: A sequence of dictionaries containing tool configurations

    Returns:
        A sequence of BaseTool objects created from the provided configurations
    """

    logger.debug("create_tools")

    tools: list[BaseTool] = []

    for tool_config in tool_models:
        name: str = tool_config.name
        tool: BaseTool = tool_registry.get(name)
        if tool is None:
            logger.debug(f"Creating tool: {name}...")
            function: BaseFunctionModel = tool_config.function
            function_type: FunctionType = function.type

            match function_type:
                case FunctionType.UNITY_CATALOG:
                    tool = create_uc_tool(function)
                case FunctionType.FACTORY:
                    tool = create_factory_tool(function)
                case FunctionType.PYTHON:
                    tool = create_python_tool(function)
                case FunctionType.MCP:
                    tool = create_mcp_tool(function)
                case _:
                    raise ValueError(f"Unknown tool type: {function_type}")

            tool_registry[name] = tool
        else:
            logger.debug(f"Tool {name} already exists, reusing it.")

        tools.append(tool)

    return tools


def create_mcp_tool(
    function: McpFunctionModel,
) -> Callable[..., Any]:
    """
    Create a tool for invoking a Databricks MCP function.

    This factory function wraps a Databricks MCP function as a callable tool that can be
    invoked by agents during reasoning.

    Args:
        function: McpFunctionModel instance containing the function details

    Returns:
        A callable tool function that wraps the specified MCP function
    """
    logger.debug(f"create_mcp_tool: {function.full_name}")

    raise NotImplementedError("MCP tools are not yet implemented.")


def create_factory_tool(
    function: FactoryFunctionModel,
) -> Callable[..., Any]:
    """
    Create a factory tool from a FactoryFunctionModel.
    This factory function dynamically loads a Python function and returns it as a callable tool.
    Args:
        function: FactoryFunctionModel instance containing the function details
    Returns:
        A callable tool function that wraps the specified factory function
    """
    logger.debug(f"create_factory_tool: {function.full_name}")

    factory: Callable[..., Any] = load_function(function_name=function.full_name)
    tool: Callable[..., Any] = factory(**function.args)
    return tool


def create_python_tool(
    function: PythonFunctionModel,
) -> Callable[..., Any]:
    """
    Create a Python tool from a Python function model.
    This factory function wraps a Python function as a callable tool that can be
    invoked by agents during reasoning.
    Args:
        function: PythonFunctionModel instance containing the function details
    Returns:
        A callable tool function that wraps the specified Python function
    """
    logger.debug(f"create_python_tool: {function.full_name}")

    # Load the Python function dynamically
    tool: Callable[..., Any] = load_function(function_name=function.full_name)
    return tool


def create_uc_tool(function: UnityCatalogFunctionModel) -> Sequence[BaseTool]:
    """
    Create LangChain tools from Unity Catalog functions.

    This factory function wraps Unity Catalog functions as LangChain tools,
    making them available for use by agents. Each UC function becomes a callable
    tool that can be invoked by the agent during reasoning.

    Args:
        function: UnityCatalogFunctionModel instance containing the function details

    Returns:
        A sequence of BaseTool objects that wrap the specified UC functions
    """

    logger.debug(f"create_uc_tool: {function.full_name}")

    # set_uc_function_client(DatabricksFunctionClient(WorkspaceClient()))

    client: DatabricksFunctionClient = DatabricksFunctionClient()

    toolkit: UCFunctionToolkit = UCFunctionToolkit(
        function_names=[function.full_name], client=client
    )

    tool = next(iter(toolkit.tools or []), None)
    return tool


def create_vector_search_tool(
    retriever: dict[str, Any],
) -> BaseTool:
    """
    Create a Vector Search tool for retrieving documents from a Databricks Vector Search index.

    This function creates a tool that enables semantic search over product information,
    documentation, or other content. It also registers the retriever schema with MLflow
    for proper integration with the model serving infrastructure.

    Args:
        retriever: Configuration details for the vector search retriever, including:
            - name: Name of the tool
            - description: Description of the tool's purpose
            - primary_key: Primary key column for the vector store
            - text_column: Text column used for vector search
            - doc_uri: URI for documentation or additional context
            - vector_store: Dictionary with 'endpoint_name' and 'index' for vector search
            - columns: List of columns to retrieve from the vector store
            - search_parameters: Additional parameters for customizing the search behavior

    Returns:
        A BaseTool instance that can perform vector search operations
    """

    vector_store: dict[str, Any] = retriever.get("vector_store", {})

    index_name: str = full_name(**vector_store.get("index"))
    columns: Sequence[str] = retriever.get("columns", [])
    search_parameters: dict[str, Any] = retriever.get("search_parameters", {})
    primary_key: str = vector_store.get("primary_key")
    doc_uri: str = vector_store.get("doc_uri")
    text_column: str = vector_store.get("embedding_source_column")

    vector_search_tool: BaseTool = VectorSearchRetrieverTool(
        name="",
        description="",
        index_name=index_name,
        columns=columns,
        **search_parameters,
    )

    # Register the retriever schema with MLflow for model serving integration
    mlflow.models.set_retriever_schema(
        name="retriever",
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        other_columns=columns,
    )

    return vector_search_tool


def create_genie_tool(
    genie_room: GenieRoomModel | dict[str, Any],
) -> Callable[[str], GenieResponse]:
    """
    Create a tool for interacting with Databricks Genie for natural language queries to databases.

    This factory function generates a tool that leverages Databricks Genie to translate natural
    language questions into SQL queries and execute them against retail databases. This enables
    answering questions about inventory, sales, and other structured retail data.

    Args:
        space_id: Databricks workspace ID where Genie is configured. If None, tries to
                get it from DATABRICKS_GENIE_SPACE_ID environment variable.

    Returns:
        A callable tool function that processes natural language queries through Genie
    """
    logger.debug("create_genie_tool")

    if isinstance(genie_room, dict):
        genie_room = GenieRoomModel(**genie_room)

    space_id: str = genie_room.space_id or os.environ.get("DATABRICKS_GENIE_SPACE_ID")

    genie: Genie = Genie(
        space_id=space_id,
    )

    @tool
    def genie_tool(question: str) -> GenieResponse:
        """
        This tool lets you have a conversation and chat with tabular data about <topic>. You should ask
        questions about the data and the tool will try to answer them.
        Please ask simple clear questions that can be answer by sql queries. If you need to do statistics or other forms of testing defer to using another tool.
        Try to ask for aggregations on the data and ask very simple questions.
        Prefer to call this tool multiple times rather than asking a complex question.

        Args:
            question (str): The question to ask to ask Genie

        Returns:
            response (GenieResponse): An object containing the Genie response
        """
        # Forward the question to Genie and return its response
        response: GenieResponse = genie.ask_question(question)
        return response

    return genie_tool


def search_tool() -> BaseTool:
    logger.debug("search_tool")
    return DuckDuckGoSearchRun(output_format="list")


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

        w: WorkspaceClient = WorkspaceClient()

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

        w: WorkspaceClient = WorkspaceClient()

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

        w: WorkspaceClient = WorkspaceClient()

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

        w: WorkspaceClient = WorkspaceClient()

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

        w: WorkspaceClient = WorkspaceClient()

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

        w: WorkspaceClient = WorkspaceClient()

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
