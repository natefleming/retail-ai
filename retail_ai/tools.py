import os
from typing import (
  Callable, 
  Optional, 
  Sequence, 
  Literal
)

from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.genie import GenieResponse
from databricks_langchain import UCFunctionToolkit
from databricks_langchain.genie import Genie
from databricks_langchain.vector_search_retriever_tool import (
    VectorSearchRetrieverTool,
)
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_core.vectorstores.base import VectorStore
from databricks_langchain import DatabricksFunctionClient, UCFunctionToolkit
from langchain_core.language_models import LanguageModelLike
from langchain_core.documents import Document

from langchain_core.tools import BaseTool, tool

from pydantic import BaseModel, Field
import pandas as pd
from io import StringIO
import mlflow

from loguru import logger

from unitycatalog.ai.core.base import FunctionExecutionResult


def find_allowable_classifications(w: WorkspaceClient, catalog_name: str, database_name: str) -> Sequence[str]:
  client: DatabricksFunctionClient = DatabricksFunctionClient(client=w)
  
  result: FunctionExecutionResult = client.execute_function(
      function_name=f"{catalog_name}.{database_name}.find_allowable_product_classifications",
      parameters={}
  )

  if result.error:
    raise Exception(result.error)

  pdf: pd.DataFrame = pd.read_csv(StringIO(result.value))
  classifications: Sequence = pdf['classification'].tolist()
  return classifications


def create_product_classification_tool(llm: LanguageModelLike, allowable_classifications: Sequence[str]) -> Callable[[str], str]:

  logger.debug(f"create_product_classification_tool: allowable_classifications={allowable_classifications}")

  class Classifier(BaseModel):

    classification: Literal[tuple(allowable_classifications)] = (
        Field(
            ...,
            description=f"The classification of the product. Must be one of: {allowable_classifications}"
          )
    )

  @tool
  def product_classification(input: str) -> str:
    """
    This tool lets you extract a classification from a product description or prompt. 
    This classification can be used to apply a filter during vector search lookup

    Args:
        input (str): The input prompt to ask to classify the product

    Returns:
        str: The classification of the product
    """  

    llm_with_tools: LanguageModelLike = llm.with_structured_output(Classifier)
    classification: str = llm_with_tools.invoke(input=input).classification

    logger.debug(f"product_classification: classification={classification}")
    return classification
    
  return product_classification


def create_find_product_details_by_description(endpoint_name: str, index_name: str, columns: Sequence[str], filter_column: str, k: int = 10) -> Callable[[str, str], Sequence[Document]]:

  @tool
  def find_product_details_by_description(content: str, product_classification: str) -> Sequence[Document]:
    """
    This tool lets you find product details by description. It will return a list of documents that match the description.

    Args:
        content (str): The content to search for
        product_classification (str): The classification of the product. This can be returned from the `product_classification` tool
    """
    vector_search: VectorStore = DatabricksVectorSearch(
        endpoint=endpoint_name,
        index_name=index_name,
        columns=columns,
        client_args={},
    )

    documents: Sequence[Document] = vector_search.similarity_search(
        query=content, k=k, filter={filter_column: [product_classification]}
    )

    return documents
  
  return find_product_details_by_description
            
            


def create_uc_tools(function_names: str | Sequence[str]) -> Sequence[BaseTool]:
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
  vector_search_tool: BaseTool = (
    VectorSearchRetrieverTool(
      name=name,
      description=description,
      index_name=index_name,
      columns=columns,
      **search_parameters,
    )
  )

  mlflow.models.set_retriever_schema(
      name=name,
      primary_key=primary_key,
      text_column=text_column,
      doc_uri=doc_uri,
      other_columns=columns
  )

  return vector_search_tool


def create_genie_tool(space_id: Optional[str] = None) -> Callable[[str], GenieResponse]:

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

        response: GenieResponse = genie.ask_question(question)
        return response

    return genie_tool

