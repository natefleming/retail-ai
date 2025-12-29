"""
Core middleware utilities for DAO AI.

This module provides the factory function for creating middleware instances
from fully qualified function names.
"""

from typing import Any, Callable

from loguru import logger

from dao_ai.utils import load_function


def create_factory_middleware(
    function_name: str,
    args: dict[str, Any] | None = None,
) -> Any:
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
        A middleware instance returned by the factory function

    Raises:
        ImportError: If the function cannot be loaded

    Example:
        # Factory function in my_module.py:
        def create_custom_middleware(threshold: float = 0.5) -> AgentMiddleware:
            return MyCustomMiddleware(threshold=threshold)

        # Usage:
        middleware = create_factory_middleware(
            function_name="my_module.create_custom_middleware",
            args={"threshold": 0.8}
        )
    """
    if args is None:
        args = {}

    logger.debug(f"Creating factory middleware: {function_name} with args: {args}")

    factory: Callable[..., Any] = load_function(function_name=function_name)
    middleware: Any = factory(**args)

    logger.debug(f"Created middleware from factory: {type(middleware).__name__}")
    return middleware
