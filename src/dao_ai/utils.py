import importlib
import importlib.metadata
import os
import re
import site
from importlib.metadata import version
from typing import Any, Callable, Sequence

from loguru import logger

import dao_ai


def is_lib_provided(lib_name: str, pip_requirements: Sequence[str]) -> bool:
    return any(
        re.search(rf"\b{re.escape(lib_name)}\b", requirement)
        for requirement in pip_requirements
    )


def is_installed():
    current_file = os.path.abspath(dao_ai.__file__)
    site_packages = [os.path.abspath(path) for path in site.getsitepackages()]
    if site.getusersitepackages():
        site_packages.append(os.path.abspath(site.getusersitepackages()))

    found: bool = any(current_file.startswith(pkg_path) for pkg_path in site_packages)
    logger.debug(
        f"Checking if dao_ai is installed: {found} (current file: {current_file}"
    )
    return found


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def get_installed_packages() -> dict[str, str]:
    """Get all installed packages with versions"""

    packages: Sequence[str] = [
        f"databricks-agents=={version('databricks-agents')}",
        f"databricks-langchain=={version('databricks-langchain')}",
        f"databricks-sdk[openai]=={version('databricks-sdk')}",
        f"duckduckgo-search=={version('duckduckgo-search')}",
        f"langchain=={version('langchain')}",
        f"langchain-mcp-adapters=={version('langchain-mcp-adapters')}",
        f"langchain-openai=={version('langchain-openai')}",
        f"langgraph=={version('langgraph')}",
        f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
        f"langgraph-prebuilt=={version('langgraph-prebuilt')}",
        f"langgraph-supervisor=={version('langgraph-supervisor')}",
        f"langgraph-swarm=={version('langgraph-swarm')}",
        f"langmem=={version('langmem')}",
        f"loguru=={version('loguru')}",
        f"mlflow=={version('mlflow')}",
        f"openevals=={version('openevals')}",
        f"openpyxl=={version('openpyxl')}",
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
        >>> func = callable_from_fqn("dao_ai.models.get_latest_model_version")
        >>> version = func("my_model")
    """
    logger.debug(f"Loading function: {function_name}")

    import sys

    # Light diagnostics: path and meta importers (helpful in Databricks Serverless)
    try:
        meta_finders = [
            getattr(f, "__class__", type(f)).__name__ for f in sys.meta_path
        ]
        logger.debug(f"sys.path ({len(sys.path)}): {sys.path}")
        logger.debug(f"sys.meta_path ({len(sys.meta_path)}): {meta_finders}")
    except Exception:
        # Do not fail if diagnostics can't be collected
        pass

    try:
        # Split the FQN into module path and function name
        module_path, func_name = function_name.rsplit(".", 1)

        # Invalidate caches to ensure new finders/paths are considered (important on first import)
        importlib.invalidate_caches()

        # Progressive import: import parents first to initialize namespace packages/loaders
        # e.g., for "a.b.c.d", import "a", then "a.b", then "a.b.c", then full module
        parts = module_path.split(".")
        parents = [".".join(parts[:i]) for i in range(1, len(parts))]
        for parent in parents:
            if parent and parent not in sys.modules:
                try:
                    importlib.import_module(parent)
                    logger.debug(f"Imported parent package: {parent}")
                except ImportError as pe:
                    # Parent import may fail for some structures; continue to try full module
                    logger.debug(f"Parent import failed for {parent}: {pe}")

        # Try to dynamically import the target module
        try:
            module = importlib.import_module(module_path)
        except ImportError as first_error:
            # Retry once after another cache invalidation; this often helps after a kernel just started
            importlib.invalidate_caches()
            try:
                module = importlib.import_module(module_path)
            except ImportError as second_error:
                # Provide richer context, including whether top-level package is present
                top_level = parts[0] if parts else module_path
                in_sys_modules = top_level in sys.modules
                msg = (
                    f"Failed to import {module_path}. "
                    f"top_level_present={in_sys_modules}, top_level={top_level}. "
                    f"first_error={first_error}, second_error={second_error}"
                )
                raise ImportError(msg)

        # Get the function from the module
        func = getattr(module, func_name)

        # Verify that the resolved object is callable
        if not callable(func):
            raise TypeError(f"Function {func_name} is not callable.")

        return func
    except (ImportError, AttributeError, TypeError) as e:
        # Provide a detailed error message that includes the original exception
        raise ImportError(f"Failed to import {function_name}: {e}")
