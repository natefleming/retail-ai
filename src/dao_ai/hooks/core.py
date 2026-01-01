"""
Hook utilities for DAO AI.

This module provides the create_hooks function for resolving FunctionHook
references to callable functions. Individual validation hooks have been
migrated to middleware - see dao_ai.middleware.message_validation.
"""

from typing import Any, Callable, Sequence

from loguru import logger

from dao_ai.config import AppConfig, FunctionHook, PythonFunctionModel


def create_hooks(
    function_hooks: FunctionHook | list[FunctionHook] | None,
) -> Sequence[Callable[..., Any]]:
    """
    Resolve FunctionHook references to callable functions.

    Args:
        function_hooks: A single FunctionHook or list of FunctionHooks to resolve

    Returns:
        Sequence of callable functions
    """
    logger.trace("Creating hooks", function_hooks=function_hooks)
    hooks: list[Callable[..., Any]] = []
    if not function_hooks:
        return []
    if not isinstance(function_hooks, (list, tuple, set)):
        function_hooks = [function_hooks]
    for function_hook in function_hooks:
        if isinstance(function_hook, str):
            function_hook = PythonFunctionModel(name=function_hook)
        hooks.extend(function_hook.as_tools())
    logger.trace("Created hooks", hooks_count=len(hooks))
    return hooks


def null_hook(state: dict[str, Any], config: Any) -> dict[str, Any]:
    """A no-op hook that returns an empty dict."""
    logger.trace("Executing null hook")
    return {}


def null_initialization_hook(config: AppConfig) -> None:
    """A no-op initialization hook."""
    logger.trace("Executing null initialization hook")


def null_shutdown_hook(config: AppConfig) -> None:
    """A no-op shutdown hook."""
    logger.trace("Executing null shutdown hook")
