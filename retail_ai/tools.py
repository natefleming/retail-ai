import os
from io import StringIO
from typing import Callable, Literal, Optional, Sequence

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.genie import GenieResponse
from databricks_langchain import (DatabricksFunctionClient,
                                  DatabricksVectorSearch, UCFunctionToolkit)
from databricks_langchain.genie import Genie
from databricks_langchain.vector_search_retriever_tool import \
    VectorSearchRetrieverTool
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores.base import VectorStore
from loguru import logger
from pydantic import BaseModel, Field
from unitycatalog.ai.core.base import FunctionExecutionResult


def find_allowable_classifications(w: WorkspaceClient, catalog_name: str, database_name: str) -> Sequence[str]:
    """
    Retrieve the list of allowable product classifications from a Unity Catalog function.
    
    This function queries a predefined UDF in the Databricks Unity Catalog to get a list
    of valid product classifications that can be used for categorizing products and
    filtering search results.
    
    Args:
        w: Databricks WorkspaceClient instance for API access
        catalog_name: Name of the Unity Catalog containing the function
        database_name: Name of the database/schema containing the function
        
    Returns:
        A sequence of strings representing valid product classifications
        
    Raises:
        Exception: If the Unity Catalog function execution fails
    """
    client: DatabricksFunctionClient = DatabricksFunctionClient(client=w)
    
    # Execute the Unity Catalog function to retrieve classifications
    result: FunctionExecutionResult = client.execute_function(
        function_name=f"{catalog_name}.{database_name}.find_allowable_product_classifications",
        parameters={}
    )

    # Handle any execution errors
    if result.error:
        raise Exception(result.error)

    # Parse the CSV result into a pandas DataFrame
    pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))
    # Extract the classification column as a list
    classifications: Sequence = pdf['classification'].tolist()
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
    logger.debug(f"create_product_classification_tool: allowable_classifications={allowable_classifications}")

    # Define a Pydantic model to enforce valid classifications through type checking
    class Classifier(BaseModel):
        classifications: list[Literal[tuple(allowable_classifications)]] = (
            Field(
                ...,
                description=f"The classifications of the product. Return {k} classifications from: {allowable_classifications}"
            )
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


def create_sku_extraction_tool(llm: LanguageModelLike) -> Callable[[str], str]:
  """
  Create a tool that leverages an LLM to extract SKU identifiers from natural language text.
  
  In GenAI applications, this tool enables automated extraction of product SKUs from
  customer queries, support tickets, reviews, or conversational inputs without requiring
  explicit structured input. This facilitates product lookups, inventory checks, and
  personalized recommendations in conversational AI systems.
  
  Args:
    llm: Language model to use for SKU extraction from unstructured text
    
  Returns:
    A callable tool function that extracts a list of SKUs from input text
  """
  logger.debug("create_sku_extraction_tool: initializing")

  # Define a Pydantic model to enforce structured output from the LLM
  class SkuIdentifier(BaseModel):
    sku: list[str] = (
      Field(
        ...,
        description="The SKU of the product. Typically 8-12 characters", 
        default_factory=list
      )
    )

  @tool
  def sku_extraction(input: str) -> list[str]:
    """
    Extract product SKUs from natural language text for product identification.
    
    This tool parses unstructured text to identify and extract product SKUs,
    enabling downstream tools to perform inventory checks, price lookups,
    or detailed product information retrieval. It handles various text formats
    including customer queries, product descriptions, and conversation history.
    
    Args:
      input (str): Natural language text that may contain product SKU references
             (e.g., "I'm looking for information about SKU ABC123")

    Returns:
      list[str]: A list of extracted SKU identifiers (empty list if none found)
    """
    # Configure the LLM to output in the structured SkuIdentifier format
    llm_with_tools: LanguageModelLike = llm.with_structured_output(SkuIdentifier)
    # Invoke the LLM to extract SKUs from the input text
    skus: list[str] = llm_with_tools.invoke(input=input).sku

    logger.debug(f"sku_extraction: extracted skus={skus}")
    return skus
    
  return sku_extraction


def create_find_product_details_by_description(endpoint_name: str, index_name: str, columns: Sequence[str], filter_column: str, k: int = 10) -> Callable[[str, str], Sequence[Document]]:
  """
  Create a tool for finding product details using vector search with classification filtering.
  
  This factory function generates a specialized search tool that combines semantic vector search 
  with categorical filtering to improve product discovery in retail applications. It enables 
  natural language product lookups with classification-based narrowing of results.
  
  Args:
    endpoint_name: Name of the Databricks Vector Search endpoint to query
    index_name: Name of the specific vector index containing product information
    columns: List of columns to retrieve from the product database
    filter_column: Database column name that contains product classification values
    k: Maximum number of search results to return (default: 10)
    
  Returns:
    A callable tool function that performs filtered vector search using both 
    product descriptions and classification categories
  """
  @tool
  def find_product_details_by_description(content: str, product_classifications: list[str]) -> Sequence[Document]:
    """
    Find products matching a description, filtered by product classifications.
    
    This tool performs semantic search over product data to find items that match 
    the given description text, while limiting results to products belonging to the 
    specified classification categories. It enables more targeted product lookups
    compared to pure text-based search.

    Args:
      content (str): Natural language description of the product(s) to find
      product_classifications (list[str]): List of product classifications to filter by,
                        typically obtained from the `product_classification` tool
    
    Returns:
      Sequence[Document]: A list of matching product documents with relevant metadata
    """
    # Initialize the Vector Search client with endpoint and index configuration
    vector_search: VectorStore = DatabricksVectorSearch(
      endpoint=endpoint_name,
      index_name=index_name,
      columns=columns,
      client_args={},
    )

    # Execute vector similarity search with classification-based filtering
    # to narrow results to specific product categories
    documents: Sequence[Document] = vector_search.similarity_search(
      query=content, k=k, filter={filter_column: product_classifications}
    )

    return documents
  
  return find_product_details_by_description
            
            
def create_uc_tools(function_names: str | Sequence[str]) -> Sequence[BaseTool]:
    """
    Create LangChain tools from Unity Catalog functions.
    
    This factory function wraps Unity Catalog functions as LangChain tools,
    making them available for use by agents. Each UC function becomes a callable
    tool that can be invoked by the agent during reasoning.
    
    Args:
        function_names: Single function name or list of function names in format 
                      "catalog.schema.function"
        
    Returns:
        A sequence of BaseTool objects that wrap the specified UC functions
    """
    if isinstance(function_names, str):
        function_names = [function_names]
    toolkit: UCFunctionToolkit = UCFunctionToolkit(function_names=function_names)
    return toolkit.tools


def create_vector_search_tool(
    name: str,
    description: str,
    index_name: str,
    primary_key: str = "id",
    text_column: str = "content",
    doc_uri: str = "doc_uri",
    columns: Sequence[str] = None,
    search_parameters: dict[str, str] = {},
) -> BaseTool:
    """
    Create a Vector Search tool for retrieving documents from a Databricks Vector Search index.
    
    This function creates a tool that enables semantic search over product information,
    documentation, or other content. It also registers the retriever schema with MLflow
    for proper integration with the model serving infrastructure.
    
    Args:
        name: Name of the tool (used for tool selection)
        description: Description of the tool's purpose (used for tool selection)
        index_name: Name of the Vector Search index to query
        primary_key: Field name of the document's primary identifier
        text_column: Field name that contains the main text content
        doc_uri: Field name for storing document URI/location
        columns: Specific columns to retrieve from the vector store (None retrieves all)
        search_parameters: Additional parameters to customize vector search behavior
        
    Returns:
        A BaseTool instance that can perform vector search operations
    """
    vector_search_tool: BaseTool = (
        VectorSearchRetrieverTool(
            name=name,
            description=description,
            index_name=index_name,
            columns=columns,
            **search_parameters,
        )
    )

    # Register the retriever schema with MLflow for model serving integration
    mlflow.models.set_retriever_schema(
        name=name,
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        other_columns=columns
    )

    return vector_search_tool


def create_genie_tool(space_id: Optional[str] = None) -> Callable[[str], GenieResponse]:
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
    # Try to get space_id from environment variable if not provided
    space_id = space_id or os.environ.get("DATABRICKS_GENIE_SPACE_ID")
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

