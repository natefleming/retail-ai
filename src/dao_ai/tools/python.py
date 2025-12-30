from typing import Any, Callable

from langchain_core.runnables.base import RunnableLike
from loguru import logger

from dao_ai.config import (
    FactoryFunctionModel,
    PythonFunctionModel,
)
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
    logger.trace("Creating factory tool", function=function.full_name)

    factory: Callable[..., Any] = load_function(function_name=function.full_name)
    tool: RunnableLike = factory(**function.args)
    # HITL is now handled at middleware level via HumanInTheLoopMiddleware
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
