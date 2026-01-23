import importlib
import importlib.metadata
import json
import os
import re
import site
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel

import dao_ai

T = TypeVar("T", bound=BaseModel)


def is_lib_provided(lib_name: str, pip_requirements: Sequence[str]) -> bool:
    return any(
        re.search(rf"\b{re.escape(lib_name)}\b", requirement)
        for requirement in pip_requirements
    )


def is_installed() -> bool:
    current_file = os.path.abspath(dao_ai.__file__)
    site_packages = [os.path.abspath(path) for path in site.getsitepackages()]
    if site.getusersitepackages():
        site_packages.append(os.path.abspath(site.getusersitepackages()))

    found: bool = any(current_file.startswith(pkg_path) for pkg_path in site_packages)
    logger.trace(
        "Checking if dao_ai is installed", is_installed=found, current_file=current_file
    )
    return found


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def normalize_host(host: str | None) -> str | None:
    """Ensure host URL has https:// scheme.

    The DATABRICKS_HOST environment variable should always include the https://
    scheme, but some environments (e.g., Databricks Apps infrastructure) may
    provide the host without it. This function normalizes the host to ensure
    it has the proper scheme.

    Args:
        host: The host URL, with or without scheme

    Returns:
        The host URL with https:// scheme, or None if host is None/empty
    """
    if not host:
        return None
    host = host.strip()
    if not host:
        return None
    if not host.startswith("http://") and not host.startswith("https://"):
        return f"https://{host}"
    return host


def get_default_databricks_host() -> str | None:
    """Get the default Databricks workspace host.

    Attempts to get the host from:
    1. DATABRICKS_HOST environment variable
    2. WorkspaceClient ambient authentication (e.g., from ~/.databrickscfg)

    Returns:
        The Databricks workspace host URL (with https:// scheme), or None if not available.
    """
    # Try environment variable first
    host: str | None = os.environ.get("DATABRICKS_HOST")
    if host:
        return normalize_host(host)

    # Fall back to WorkspaceClient
    try:
        from databricks.sdk import WorkspaceClient

        w: WorkspaceClient = WorkspaceClient()
        return normalize_host(w.config.host)
    except Exception:
        logger.trace("Could not get default Databricks host from WorkspaceClient")
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
        logger.trace(
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
                    "Cannot determine dao-ai version: pyproject.toml not found",
                    path=str(pyproject_path),
                )
                return "dev"

            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)
                pkg_version = pyproject_data.get("project", {}).get("version", "dev")
                logger.trace(
                    "Read version from pyproject.toml",
                    version=pkg_version,
                    path=str(pyproject_path),
                )
                return pkg_version
        except Exception as e:
            logger.warning(
                "Cannot determine dao-ai version from pyproject.toml", error=str(e)
            )
            return "dev"


def get_installed_packages() -> dict[str, str]:
    """Get all installed packages with versions"""

    packages: Sequence[str] = [
        f"databricks-agents=={version('databricks-agents')}",
        f"databricks-langchain[memory]=={version('databricks-langchain')}",
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
    logger.trace("Loading function", function_name=function_name)

    try:
        # Split the FQN into module path and function name
        module_path, func_name = function_name.rsplit(".", 1)

        # Dynamically import the module
        module = importlib.import_module(module_path)

        # Get the function from the module
        func: Any = getattr(module, func_name)

        # Verify that the resolved object is callable or is a LangChain tool
        # In langchain 1.x, StructuredTool objects are not directly callable
        # but have an invoke() method
        is_callable: bool = callable(func)
        is_langchain_tool: bool = isinstance(func, BaseTool)

        if not is_callable and not is_langchain_tool:
            raise TypeError(f"Function {func_name} is not callable or invocable.")

        return func
    except (ImportError, AttributeError, TypeError) as e:
        # Provide a detailed error message that includes the original exception
        raise ImportError(f"Failed to import {function_name}: {e}")


def type_from_fqn(type_name: str) -> type:
    """
    Load a type from a fully qualified name (FQN).

    Dynamically imports and returns a type (class) from a module using its
    fully qualified name. Useful for loading Pydantic models, dataclasses,
    or any Python type specified as a string in configuration files.

    Args:
        type_name: Fully qualified type name in format "module.path.ClassName"

    Returns:
        The imported type/class

    Raises:
        ValueError: If the FQN format is invalid
        ImportError: If the module cannot be imported
        AttributeError: If the type doesn't exist in the module
        TypeError: If the resolved object is not a type

    Example:
        >>> ProductModel = type_from_fqn("my_models.ProductInfo")
        >>> instance = ProductModel(name="Widget", price=9.99)
    """
    logger.trace("Loading type", type_name=type_name)

    try:
        # Split the FQN into module path and class name
        parts = type_name.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid type name '{type_name}'. "
                "Expected format: 'module.path.ClassName'"
            )

        module_path, class_name = parts

        # Dynamically import the module
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Could not import module '{module_path}' for type '{type_name}': {e}"
            ) from e

        # Get the class from the module
        if not hasattr(module, class_name):
            raise AttributeError(
                f"Module '{module_path}' does not have attribute '{class_name}'"
            )

        resolved_type = getattr(module, class_name)

        # Verify it's actually a type
        if not isinstance(resolved_type, type):
            raise TypeError(
                f"'{type_name}' resolved to {resolved_type}, which is not a type"
            )

        return resolved_type

    except (ValueError, ImportError, AttributeError, TypeError) as e:
        # Provide a detailed error message that includes the original exception
        raise type(e)(f"Failed to load type '{type_name}': {e}") from e


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


def get_databricks_response_format(model_class: type[BaseModel]) -> dict[str, Any]:
    """Create a Databricks-compatible response_format for structured output.

    Databricks requires the json_schema response format to have a 'name' field.
    This function creates the properly formatted response_format dictionary
    from a Pydantic model.

    Args:
        model_class: A Pydantic model class to use as the output schema

    Returns:
        A dictionary suitable for use with llm.bind(response_format=...)

    Example:
        >>> response_format = get_databricks_response_format(MyModel)
        >>> bound_llm = llm.bind(response_format=response_format)
        >>> result = bound_llm.invoke(prompt)
    """
    schema = model_class.model_json_schema()

    # Remove $defs from the schema - Databricks doesn't support complex refs
    # We need to inline any referenced definitions
    if "$defs" in schema:
        schema = _inline_schema_defs(schema)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_class.__name__,
            "schema": schema,
            "strict": True,
        },
    }


def _inline_schema_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline $defs references in a JSON schema.

    Databricks doesn't support $ref and complex nested definitions,
    so we need to inline them.

    Args:
        schema: The original JSON schema with $defs

    Returns:
        A schema with all references inlined
    """
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def resolve_refs(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "$ref" in obj:
                # Extract the definition name from #/$defs/DefinitionName
                ref_path = obj["$ref"]
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path[len("#/$defs/") :]
                    if def_name in defs:
                        # Return a copy of the definition with refs resolved
                        return resolve_refs(defs[def_name].copy())
                return obj
            return {k: resolve_refs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item) for item in obj]
        return obj

    return resolve_refs(schema)


def _repair_json(content: str) -> str | None:
    """Attempt to repair malformed JSON from LLM output.

    Handles common issues:
    - Extra text before/after JSON object
    - Truncated JSON (unclosed brackets/braces)
    - Trailing commas

    Args:
        content: The potentially malformed JSON string

    Returns:
        Repaired JSON string if successful, None otherwise
    """
    # 1. Extract JSON object if wrapped in extra text
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    content = content[start : end + 1]

    # 2. Try parsing as-is first
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        pass

    # 3. Fix trailing commas before closing brackets
    content = re.sub(r",\s*}", "}", content)
    content = re.sub(r",\s*]", "]", content)

    # 4. Try to close unclosed brackets/braces
    open_braces = content.count("{") - content.count("}")
    open_brackets = content.count("[") - content.count("]")

    if open_braces > 0 or open_brackets > 0:
        # Remove trailing comma if present
        content = content.rstrip().rstrip(",")
        content += "]" * open_brackets + "}" * open_braces

    # 5. Final validation
    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        return None


def invoke_with_structured_output(
    llm: BaseChatModel,
    prompt: str,
    model_class: type[T],
) -> T | None:
    """Invoke an LLM with Databricks-compatible structured output.

    Uses response_format with json_schema type and proper 'name' field
    as required by Databricks Foundation Model APIs.

    Args:
        llm: The language model to invoke
        prompt: The prompt to send to the model
        model_class: The Pydantic model class for the expected output

    Returns:
        An instance of model_class, or None if parsing fails
    """
    response_format = get_databricks_response_format(model_class)
    bound_llm = llm.bind(response_format=response_format)

    response = bound_llm.invoke(prompt)

    content = response.content
    if not isinstance(content, str):
        return None

    try:
        # Try parsing the JSON directly
        result_dict = json.loads(content)
        return model_class.model_validate(result_dict)
    except json.JSONDecodeError as e:
        # Attempt JSON repair
        repaired = _repair_json(content)
        if repaired:
            try:
                result_dict = json.loads(repaired)
                logger.debug("JSON repair successful", model_class=model_class.__name__)
                return model_class.model_validate(result_dict)
            except (json.JSONDecodeError, Exception):
                pass
        logger.warning(
            "Failed to parse structured output",
            error=str(e),
            model_class=model_class.__name__,
        )
        return None
    except Exception as e:
        logger.warning(
            "Failed to parse structured output",
            error=str(e),
            model_class=model_class.__name__,
        )
        return None
