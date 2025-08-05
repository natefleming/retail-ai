from typing import Sequence

from databricks_langchain import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
)
from langchain_core.runnables.base import RunnableLike
from loguru import logger

from dao_ai.config import (
    UnityCatalogFunctionModel,
)
from dao_ai.tools.human_in_the_loop import as_human_in_the_loop


def create_uc_tools(
    function: UnityCatalogFunctionModel | str,
) -> Sequence[RunnableLike]:
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

    logger.debug(f"create_uc_tools: {function}")

    if isinstance(function, UnityCatalogFunctionModel):
        function = function.full_name

    client: DatabricksFunctionClient = DatabricksFunctionClient()

    toolkit: UCFunctionToolkit = UCFunctionToolkit(
        function_names=[function], client=client
    )

    tools = toolkit.tools or []

    logger.debug(f"Retrieved tools: {tools}")

    tools = [as_human_in_the_loop(tool=tool, function=function) for tool in tools]

    return tools
