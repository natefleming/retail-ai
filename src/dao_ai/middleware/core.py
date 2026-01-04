"""
Core middleware utilities for DAO AI.

This module provides the factory function for creating middleware instances
from fully qualified function names.
"""

from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from loguru import logger

from dao_ai.state import AgentState, Context
from dao_ai.utils import load_function


def create_factory_middleware(
    function_name: str,
    args: dict[str, Any] | None = None,
) -> list[AgentMiddleware[AgentState, Context]]:
    """
    Create middleware from a factory function.


    This factory function dynamically loads a Python function and calls it
    with the provided arguments to create a middleware instance.

    The factory function should return a middleware object compatible with
    LangChain's create_agent middleware parameter (AgentMiddleware or any
    callable/object that implements the middleware interface).

    Args:
        function_name: Fully qualified name of the factory function
                       (e.g., 'my_module.create_custom_middleware')
        args: Arguments to pass to the factory function

    Returns:
        List containing the AgentMiddleware instance(s) returned by the factory function.
        If the factory returns a single middleware, it is wrapped in a list.
        If the factory returns a list, it is returned as-is.

    Raises:
        ImportError: If the function cannot be loaded

    Example:
        # Factory function in my_module.py:
        def create_custom_middleware(threshold: float = 0.5) -> AgentMiddleware[AgentState, Context]:
            return MyCustomMiddleware(threshold=threshold)

        # Usage:
        middleware = create_factory_middleware(
            function_name="my_module.create_custom_middleware",
            args={"threshold": 0.8}
        )
    """
    if args is None:
        args = {}

    logger.trace("Creating factory middleware", function_name=function_name, args=args)

    factory: Callable[
        ...,
        AgentMiddleware[AgentState, Context]
        | list[AgentMiddleware[AgentState, Context]],
    ] = load_function(function_name=function_name)
    result = factory(**args)

    # Normalize to list
    if isinstance(result, list):
        middlewares = result
    else:
        middlewares = [result]

    logger.trace(
        "Created middleware from factory",
        middleware_count=len(middlewares),
        middleware_types=[type(m).__name__ for m in middlewares],
    )
    return middlewares
