import importlib
import importlib.metadata
import json
import os
import re
import site
import time
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TypeVar

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


def is_published() -> bool:
    """Check if dao-ai was installed from PyPI (not from a local file or editable install).

    Returns True only if the package was installed from a package index.
    Returns False if installed from a local wheel, editable install, or source path.
    Used by create_agent() to decide whether to pin to a PyPI version or use code_paths.
    """
    if not is_installed():
        return False
    try:
        from importlib.metadata import distribution

        dist = distribution("dao-ai")
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            import json

            data = json.loads(direct_url)
            url = data.get("url", "")
            # file:// URLs indicate local installs (editable or wheel)
            if url.startswith("file://"):
                logger.trace("dao-ai installed from local file, not PyPI", url=url)
                return False
    except Exception:
        pass
    logger.trace("dao-ai appears to be installed from PyPI")
    return True


def resolve_use_local_source(development: bool | None) -> bool:
    """Decide whether a deploy should ship local dao-ai source/wheel vs PyPI.

    Single source of truth for the ``--development`` tri-state shared by the
    CLI handlers, the deploy notebook, and the Databricks provider so they all
    agree:

    - ``development=True``  → force local source/wheel (test unreleased code).
    - ``development=False`` → force the published PyPI package.
    - ``development=None``  → auto: local when dao-ai is a local/editable install
      (``not is_published()``), PyPI otherwise. Preserves the historical default
      while letting the ``--development/--no-development`` flag override it.
    """
    if development is not None:
        return development
    return not is_published()


def _wheel_from_direct_url() -> Path | None:
    """Return the wheel file dao-ai was installed from, if any.

    When a user runs ``pip install /path/to/dao_ai-X.Y.Z.whl`` (including
    workspace paths like ``/Workspace/.../wheels/dao_ai-...whl`` on
    Databricks Serverless), pip records the source path in
    ``direct_url.json``. Surfacing that path lets ``deploy_apps_agent``
    bundle the same wheel into the deployed app without searching for a
    local ``dist/`` directory that doesn't exist on Serverless.
    """
    try:
        from importlib.metadata import distribution

        dist = distribution("dao-ai")
        direct_url = dist.read_text("direct_url.json")
        if not direct_url:
            return None
        data = json.loads(direct_url)
        url = data.get("url", "")
        if not url.startswith("file://"):
            return None
        path = Path(url.removeprefix("file://"))
        if path.is_file() and path.suffix == ".whl":
            logger.debug("Resolved dao-ai wheel from direct_url.json", path=str(path))
            return path
    except Exception as e:
        logger.debug(f"Could not resolve wheel from direct_url.json: {e}")
    return None


@contextmanager
def dev_local_version(pyproject_path: Path) -> Iterator[None]:
    """Temporarily stamp a unique PEP 440 local version segment on the build.

    A development wheel is built at the *same* base version as the published
    package (e.g. ``0.1.115``). An Apps container that already has that version
    installed treats the bundled ``./dist/<wheel>`` as "already satisfied" and
    silently keeps the stale published code, so local source edits never take
    effect on redeploy. ``--force-reinstall`` can't fix this — it is not a valid
    line in an Apps ``requirements.txt`` (the installer parses each line as a
    requirement).

    Stamping a unique local version (``0.1.115+dev<epoch>``) makes pip treat the
    dev wheel as strictly newer than the published base version, so it always
    reinstalls — while remaining a legal requirement (a version, not a flag) and
    never masquerading as a real release. The original ``pyproject.toml`` is
    restored on exit so the working tree is left unchanged.

    Wrap every ``uv build --wheel`` call that produces a dev wheel with this
    (generate-agent, deploy, generate-mcp) so ``--development`` behaves
    uniformly across all deploy paths.
    """
    original = pyproject_path.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', original, flags=re.MULTILINE)
    if not match:
        # No static version line (e.g. dynamic version) — nothing to stamp.
        yield
        return
    base = match.group(1)
    # Skip if a local segment is already present (idempotent / user-managed).
    local = base if "+" in base else f"{base}+dev{int(time.time())}"
    stamped = original.replace(match.group(0), f'version = "{local}"', 1)
    try:
        pyproject_path.write_text(stamped)
        logger.info("Stamped dev-build local version", version=local)
        yield
    finally:
        pyproject_path.write_text(original)


def find_dev_wheel() -> Path | None:
    """Find an existing dao-ai wheel in known locations.

    Returns the most recently modified wheel, or None if dao-ai was installed
    from PyPI or no wheel is found. Does NOT build a wheel -- callers decide
    what to do when None is returned.

    Search order:
        1. The wheel recorded in ``direct_url.json`` (when dao-ai was
           installed from a local/workspace ``.whl`` path).
        2. Project dist/ (local dev: relative to this source file)
        3. Bundle artifact paths (job cluster: ../dist/, ../../artifacts/.internal/)
        4. CWD dist/ (fallback)
    """
    if is_published():
        return None

    direct: Path | None = _wheel_from_direct_url()
    if direct:
        return direct

    search_dirs: list[Path] = [
        Path(__file__).parents[2] / "dist",
        Path("../dist").resolve(),
        Path("../../artifacts/.internal").resolve(),
        Path("dist").resolve(),
    ]

    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        wheels = sorted(
            search_dir.glob("dao_ai-*.whl"),
            key=lambda p: p.stat().st_mtime,
        )
        if wheels:
            logger.debug("Found dev wheel", path=str(wheels[-1]))
            return wheels[-1]

    logger.debug("No dev wheel found in any search path")
    return None


def is_source_layout(package_dir: Path) -> bool:
    """Check if a package directory is part of a source tree (not site-packages).

    Used to guard against iterating site-packages when adding code_paths
    for MLflow model logging.
    """
    site_packages_dirs = [os.path.abspath(p) for p in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        site_packages_dirs.append(os.path.abspath(user_site))
    abs_dir = os.path.abspath(package_dir)
    return not any(abs_dir.startswith(sp) for sp in site_packages_dirs)


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
        import tomllib  # stdlib on Python 3.11+ (our floor)

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


def get_installed_packages(extras: set[str] | None = None) -> list[str]:
    """Get pinned pip requirement strings for packages used by dao-ai.

    Used by the dev-mode Model Serving path: a bare ``code/<wheel>`` code-path
    cannot carry a ``[extras]`` suffix, so the packages backing each active
    extra must be pinned explicitly here or they will be absent in the serving
    container and the model will crash at load.

    Args:
        extras: The set of extras the config exercises (already expanded — no
            ``"all"`` sentinel). Optional-extra packages are pinned only when
            their extra is present. ``None`` pins core packages only.

    Returns:
        A list of ``package==version`` requirement strings.
    """
    extras = extras or set()

    packages: list[str] = [
        f"databricks-agents=={version('databricks-agents')}",
        f"databricks-langchain[memory]=={version('databricks-langchain')}",
        f"databricks-mcp=={version('databricks-mcp')}",
        f"databricks-sdk[openai]=={version('databricks-sdk')}",
        f"langchain=={version('langchain')}",
        f"langchain-mcp-adapters=={version('langchain-mcp-adapters')}",
        f"langchain-openai=={version('langchain-openai')}",
        f"langgraph=={version('langgraph')}",
        f"langgraph-checkpoint-postgres=={version('langgraph-checkpoint-postgres')}",
        f"langgraph-prebuilt=={version('langgraph-prebuilt')}",
        f"loguru=={version('loguru')}",
        f"mcp=={version('mcp')}",
        f"mlflow=={version('mlflow')}",
        f"nest-asyncio=={version('nest-asyncio')}",
        f"psycopg[binary,pool]=={version('psycopg')}",
        f"pydantic=={version('pydantic')}",
        f"pyyaml=={version('pyyaml')}",
        f"unitycatalog-ai[databricks]=={version('unitycatalog-ai')}",
        f"unitycatalog-langchain[databricks]=={version('unitycatalog-langchain')}",
    ]

    # Optional-extra packages: pin only when the config uses the extra. The
    # extra's package must already be importable in this (dev) environment;
    # if it isn't, skip the pin rather than crash — the missing-extra error
    # will surface at runtime with an actionable message.
    optional_pins: dict[str, str] = {
        "a2a": "a2a-sdk",
        "rerank": "flashrank",
        "deepagents": "deepagents",
        "memory": "langmem",
        "search": "ddgs",
        "excel": "openpyxl",
    }
    for extra, dist_name in optional_pins.items():
        if extra not in extras:
            continue
        try:
            packages.append(f"{dist_name}=={version(dist_name)}")
        except PackageNotFoundError:
            logger.warning(
                "Config uses an extra whose package is not installed; "
                "skipping its version pin. The deployed model will fail to "
                "import this feature unless the package is available.",
                extra=extra,
                package=dist_name,
            )

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


def is_in_notebook() -> bool:
    """Check if running inside an interactive Databricks notebook.

    Used by the extras resolver: in a notebook the convenient default is to
    install every optional feature (``dao-ai[all]``) rather than the precise,
    size-minimal set used by CLI/bundle deploys. Model Serving is explicitly
    excluded — it runs inside a Databricks runtime but is not interactive and
    is size-sensitive.

    Returns:
        True if an interactive IPython/Databricks notebook kernel is detected.
    """
    # Model Serving is a Databricks runtime but not an interactive notebook.
    if is_in_model_serving():
        return False

    # A live ``dbutils`` in the caller's builtins is the strongest signal of a
    # Databricks notebook kernel.
    try:
        import builtins

        if getattr(builtins, "dbutils", None) is not None:
            return True
    except Exception:
        pass

    # Fall back to an interactive IPython kernel (ZMQ shell = notebook, not a
    # plain terminal REPL).
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is not None and type(ipython).__name__ == "ZMQInteractiveShell":
            return True
    except Exception:
        pass

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
