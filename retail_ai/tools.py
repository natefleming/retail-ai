import os
from typing import Callable, Optional, Sequence

from databricks_ai_bridge.genie import GenieResponse
from databricks_langchain import UCFunctionToolkit
from databricks_langchain.genie import Genie
from databricks_langchain.vector_search_retriever_tool import \
    VectorSearchRetrieverTool
from langchain_core.tools import BaseTool, tool


def create_uc_tools(function_names: str | Sequence[str]) -> Sequence[BaseTool]:
  if isinstance(function_names, str):
    function_names = [function_names]
  toolkit: UCFunctionToolkit = UCFunctionToolkit(function_names=function_names)
  return toolkit.tools


def create_vector_search_tool(
  name: str,
  description: str,
  index_name: str,
  columns: Sequence[str] = None,
) -> BaseTool:
  vector_search_tool: BaseTool = (
    VectorSearchRetrieverTool(
      name=name,
      description=description,
      index_name=index_name,
      columns=columns,
    )
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

