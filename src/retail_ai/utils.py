import importlib
import importlib.metadata
import re
from importlib.metadata import version
from typing import Any, Callable, Sequence

from loguru import logger


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def get_installed_packages() -> dict[str, str]:
    """Get all installed packages with versions"""

    packages: Sequence[str] = [
        f"databricks-agents=={version('databricks-agents')}",
        f"databricks-langchain=={version('databricks-langchain')}",
        f"databricks-sdk=={version('databricks-sdk')}",
        f"duckduckgo-search=={version('duckduckgo-search')}",
        f"langchain=={version('langchain')}",
        f"langchain-mcp-adapters=={version('langchain-mcp-adapters')}",
        f"langgraph=={version('langgraph')}",
        f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
        f"langgraph-reflection=={version('langgraph-reflection')}",
        f"langgraph-supervisor=={version('langgraph-supervisor')}",
        f"langgraph-swarm=={version('langgraph-swarm')}",
        f"langmem=={version('langmem')}",
        f"loguru=={version('loguru')}",
        f"mlflow=={version('mlflow')}",
        f"openevals=={version('openevals')}",
        f"psycopg[binary,pool]=={version('psycopg')}",
        f"pydantic=={version('pydantic')}",
        f"unitycatalog-ai[databricks]=={version('unitycatalog-ai')}",
        f"unitycatalog-langchain[databricks]=={version('unitycatalog-langchain')}",
    ]
    return packages


def load_function(function_name: str) -> Callable[..., Any]:
    """
    Dynamically import and return a callable function using its fully qualified name.

    This utility function allows dynamic loading of functions from their string
    representation, enabling configuration-driven function resolution at runtime.
    It's particularly useful for loading different components based on configuration
    without hardcoding import statements.

    Args:
        fqn: Fully qualified name of the function to import, in the format
             "module.submodule.function_name"

    Returns:
        The imported callable function

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the function doesn't exist in the module
        TypeError: If the resolved object is not callable

    Example:
        >>> func = callable_from_fqn("retail_ai.models.get_latest_model_version")
        >>> version = func("my_model")
    """
    logger.debug(f"Loading function: {function_name}")
    try:
        # Split the FQN into module path and function name
        module_path, func_name = function_name.rsplit(".", 1)

        # Dynamically import the module
        module = importlib.import_module(module_path)

        # Get the function from the module
        func = getattr(module, func_name)

        # Verify that the resolved object is callable
        if not callable(func):
            raise TypeError(f"Function {func_name} is not callable.")

        return func
    except (ImportError, AttributeError, TypeError) as e:
        # Provide a detailed error message that includes the original exception
        raise ImportError(f"Failed to import {function_name}: {e}")
