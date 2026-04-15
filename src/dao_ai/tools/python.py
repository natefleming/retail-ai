from typing import Any, Callable, Sequence

from langchain_core.runnables.base import RunnableLike
from langchain_core.tools import BaseToolkit
from loguru import logger

from dao_ai.config import (
    FactoryFunctionModel,
    PythonFunctionModel,
)
from dao_ai.utils import load_function


def create_factory_tool(
    function: FactoryFunctionModel,
) -> RunnableLike | Sequence[RunnableLike]:
    """
    Create tool(s) from a FactoryFunctionModel.

    The factory function may return a single tool, a list of tools, or a
    :class:`BaseToolkit` whose ``get_tools()`` is expanded.

    Args:
        function: FactoryFunctionModel instance containing the function details

    Returns:
        A single tool or a sequence of tools produced by the factory.
    """
    logger.trace("Creating factory tool", function=function.full_name)

    factory: Callable[..., Any] = load_function(function_name=function.full_name)
    result: Any = factory(**function.args)

    if isinstance(result, BaseToolkit):
        return result.get_tools()
    return result


def create_python_tool(
    function: PythonFunctionModel | str,
) -> RunnableLike:
    """
    Create a Python tool from a Python function model.
    This factory function wraps a Python function as a callable tool that can be
    invoked by agents during reasoning.
    Args:
        function: PythonFunctionModel instance containing the function details
    Returns:
        A callable tool function that wraps the specified Python function
    """
    function_name = (
        function.full_name if isinstance(function, PythonFunctionModel) else function
    )
    logger.trace("Creating Python tool", function=function_name)

    if isinstance(function, PythonFunctionModel):
        function = function.full_name

    # Load the Python function dynamically
    tool: RunnableLike = load_function(function_name=function)
    # HITL is now handled at middleware level via HumanInTheLoopMiddleware
    return tool
