import importlib
import importlib.metadata
import os
import re
import site
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
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


def get_default_databricks_host() -> str | None:
    """Get the default Databricks workspace host.

    Attempts to get the host from:
    1. DATABRICKS_HOST environment variable
    2. WorkspaceClient ambient authentication (e.g., from ~/.databrickscfg)

    Returns:
        The Databricks workspace host URL, or None if not available.
    """
    # Try environment variable first
    host: str | None = os.environ.get("DATABRICKS_HOST")
    if host:
        return host

    # Fall back to WorkspaceClient
    try:
        from databricks.sdk import WorkspaceClient

        w: WorkspaceClient = WorkspaceClient()
        return w.config.host
    except Exception:
        logger.debug("Could not get default Databricks host from WorkspaceClient")
        return None


def dao_ai_version() -> str:
    """
    Get the dao-ai package version, with fallback for source installations.

    Tries to get the version from installed package metadata first. If the package
    is not installed (e.g., running from source), falls back to reading from
    pyproject.toml. Returns "dev" if neither method works.

    Returns:
        str: The version string, or "dev" if version cannot be determined
    """
    try:
        # Try to get version from installed package metadata
        return version("dao-ai")
    except PackageNotFoundError:
        # Package not installed, try reading from pyproject.toml
        logger.debug(
            "dao-ai package not installed, attempting to read version from pyproject.toml"
        )
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # Fallback for Python < 3.11
            except ImportError:
                logger.warning(
                    "Cannot determine dao-ai version: package not installed and tomllib/tomli not available"
                )
                return "dev"

        try:
            # Find pyproject.toml relative to this file
            project_root = Path(__file__).parents[2]
            pyproject_path = project_root / "pyproject.toml"

            if not pyproject_path.exists():
                logger.warning(
                    f"Cannot determine dao-ai version: pyproject.toml not found at {pyproject_path}"
                )
                return "dev"

            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)
                pkg_version = pyproject_data.get("project", {}).get("version", "dev")
                logger.debug(
                    f"Read version {pkg_version} from pyproject.toml at {pyproject_path}"
                )
                return pkg_version
        except Exception as e:
            logger.warning(f"Cannot determine dao-ai version from pyproject.toml: {e}")
            return "dev"


def get_installed_packages() -> dict[str, str]:
    """Get all installed packages with versions"""

    packages: Sequence[str] = [
        f"databricks-agents=={version('databricks-agents')}",
        f"databricks-langchain=={version('databricks-langchain')}",
        f"databricks-mcp=={version('databricks-mcp')}",
        f"databricks-sdk[openai]=={version('databricks-sdk')}",
        f"ddgs=={version('ddgs')}",
        f"flashrank=={version('flashrank')}",
        f"langchain=={version('langchain')}",
        f"langchain-mcp-adapters=={version('langchain-mcp-adapters')}",
        f"langchain-openai=={version('langchain-openai')}",
        f"langchain-tavily=={version('langchain-tavily')}",
        f"langgraph=={version('langgraph')}",
        f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
        f"langgraph-prebuilt=={version('langgraph-prebuilt')}",
        f"langgraph-supervisor=={version('langgraph-supervisor')}",
        f"langgraph-swarm=={version('langgraph-swarm')}",
        f"langmem=={version('langmem')}",
        f"loguru=={version('loguru')}",
        f"mcp=={version('mcp')}",
        f"mlflow=={version('mlflow')}",
        f"nest-asyncio=={version('nest-asyncio')}",
        f"openevals=={version('openevals')}",
        f"openpyxl=={version('openpyxl')}",
        f"psycopg[binary,pool]=={version('psycopg')}",
        f"pydantic=={version('pydantic')}",
        f"pyyaml=={version('pyyaml')}",
        f"tomli=={version('tomli')}",
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
        The imported callable function or langchain tool

    Raises:
        ImportError: If the module cannot be imported
        AttributeError: If the function doesn't exist in the module
        TypeError: If the resolved object is not callable or invocable

    Example:
        >>> func = callable_from_fqn("dao_ai.models.get_latest_model_version")
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

        # Verify that the resolved object is callable or is a langchain tool
        # In langchain 1.x, StructuredTool objects are not directly callable
        # but have an invoke() method
        is_callable = callable(func)
        is_langchain_tool = hasattr(func, "invoke") and hasattr(func, "name")

        if not is_callable and not is_langchain_tool:
            raise TypeError(f"Function {func_name} is not callable or invocable.")

        return func
    except (ImportError, AttributeError, TypeError) as e:
        # Provide a detailed error message that includes the original exception
        raise ImportError(f"Failed to import {function_name}: {e}")


def is_in_model_serving() -> bool:
    """Check if running in Databricks Model Serving environment.

    Detects Model Serving by checking for environment variables that are
    typically set in that environment.
    """
    # Primary check - explicit Databricks Model Serving env var
    if os.environ.get("IS_IN_DB_MODEL_SERVING_ENV", "false").lower() == "true":
        return True

    # Secondary check - Model Serving sets these environment variables
    if os.environ.get("DATABRICKS_MODEL_SERVING_ENV"):
        return True

    # Check for cluster type indicator
    cluster_type = os.environ.get("DATABRICKS_CLUSTER_TYPE", "")
    if "model-serving" in cluster_type.lower():
        return True

    # Check for model serving specific paths
    if os.path.exists("/opt/conda/envs/mlflow-env"):
        return True

    return False
