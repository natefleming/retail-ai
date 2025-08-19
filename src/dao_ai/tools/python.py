from typing import Any, Callable

from langchain_core.runnables.base import RunnableLike
from loguru import logger

from dao_ai.config import (
    FactoryFunctionModel,
    PythonFunctionModel,
)
from dao_ai.tools.human_in_the_loop import as_human_in_the_loop
from dao_ai.utils import load_function


def create_factory_tool(
    function: FactoryFunctionModel,
) -> RunnableLike:
    """
    Create a factory tool from a FactoryFunctionModel.
    This factory function dynamically loads a Python function and returns it as a callable tool.
    Args:
        function: FactoryFunctionModel instance containing the function details
    Returns:
        A callable tool function that wraps the specified factory function
    """
    logger.debug(f"create_factory_tool: {function}")

    factory: Callable[..., Any] = load_function(function_name=function.full_name)
    tool: Callable[..., Any] = factory(**function.args)
    tool = as_human_in_the_loop(
        tool=tool,
        function=function,
    )
    return tool


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
    logger.debug(f"create_python_tool: {function}")

    if isinstance(function, PythonFunctionModel):
        function = function.full_name

    # Load the Python function dynamically
    tool: Callable[..., Any] = load_function(function_name=function)

    tool = as_human_in_the_loop(
        tool=tool,
        function=function,
    )
    return tool
