"""
Bundle generation module for creating Databricks Asset Bundle files from dao-ai config.

Generates a complete, deployable bundle directory containing:
- databricks.yaml: Bundle definition with app config, resources, scopes
- dao_ai.yaml: Copy of the dao-ai agent config
- pyproject.toml: Python project with dao-ai dependency
- .gitignore, .python-version: Scaffolding files

Usage:
    from dao_ai.apps.bundle import write_bundle
    from dao_ai.config import AppConfig

    config = AppConfig.from_file("my_config.yaml")
    write_bundle(config, Path("./my-bundle"), force=False)
"""

import shutil
import subprocess
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from dao_ai.apps.resources import (
    _extract_env_vars_from_config,
    generate_app_resources,
    generate_user_api_scopes,
)
from dao_ai.config import AppConfig

_BUNDLE_RESOURCE_CONVERTERS: dict[str, str] = {
    "serving-endpoint": "serving_endpoint",
    "sql-warehouse": "sql_warehouse",
    "genie-space": "genie_space",
    "database": "database",
    "secret": "secret",
    "app": "app",
    "table": "uc_securable",
    "volume": "uc_securable",
    "function": "uc_securable",
    "connection": "uc_securable",
    "vector-search-index": "uc_securable",
}

_DEDUP_KEY_EXTRACTORS: dict[str, Any] = {
    "serving_endpoint": lambda r: r["serving_endpoint"]["name"],
    "sql_warehouse": lambda r: r["sql_warehouse"]["id"],
    "genie_space": lambda r: r["genie_space"]["space_id"],
    "database": lambda r: (
        r["database"]["instance_name"],
        r["database"]["database_name"],
    ),
    "secret": lambda r: (r["secret"]["scope"], r["secret"]["key"]),
    "app": lambda r: r["app"]["name"],
    "uc_securable": lambda r: r["uc_securable"]["securable_full_name"],
}

_PLATFORM_PROVIDED_ENV_VARS: set[str] = {"DATABRICKS_HOST"}

_BUNDLE_PERMISSION_MAP: dict[str, str] = {
    "CAN_EXECUTE": "EXECUTE",
    "CAN_READ": "READ_VOLUME",
    "CAN_SELECT": "SELECT",
    "USE_CONNECTION": "USE_CONNECTION",
}

_GITIGNORE_CONTENT = """\
.venv/
.databricks/
dist/
*.egg-info/
__pycache__/
*.pyc
.vscode/
bundle_config_schema.json
dao_ai_schema.json
"""

_GITIGNORE_DEV_CONTENT = """\
.venv/
.databricks/
*.egg-info/
__pycache__/
*.pyc
.vscode/
bundle_config_schema.json
dao_ai_schema.json
"""

_PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "0.1.0"
description = "DAO AI Agent: {name}"
requires-python = ">=3.11"
dependencies = [
    "dao-ai>={dao_ai_version}",
]

[tool.hatch.build.targets.wheel]
packages = ["src/{package_name}"]
sources = ["src"]
"""

_PYPROJECT_DEV_TEMPLATE = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "0.1.0"
description = "DAO AI Agent: {name} (development build)"
requires-python = ">=3.11"
dependencies = [
    "dao-ai",
]

[tool.uv.sources]
dao-ai = {{ path = "dist/{wheel_filename}" }}

[tool.hatch.build.targets.wheel]
packages = ["src/{package_name}"]
sources = ["src"]
"""


def _get_dao_ai_version() -> str:
    """Return the currently installed dao-ai version for pinning in generated bundles."""
    try:
        return pkg_version("dao-ai")
    except Exception:
        return "0.1.0"


def _convert_single_resource(resource: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single flat app.yaml resource dict to bundle nested format."""
    resource_type: str = resource.get("type", "")
    bundle_key: str | None = _BUNDLE_RESOURCE_CONVERTERS.get(resource_type)
    if bundle_key is None:
        logger.warning(f"Unsupported resource type for bundle: {resource_type}")
        return None

    result: dict[str, Any] = {"name": resource["name"]}
    permission: str = resource.get("permissions", [{}])[0].get("level", "")

    if resource_type == "serving-endpoint":
        result["serving_endpoint"] = {
            "name": resource["serving_endpoint_name"],
            "permission": permission,
        }
    elif resource_type == "sql-warehouse":
        result["sql_warehouse"] = {
            "id": resource["sql_warehouse_id"],
            "permission": permission,
        }
    elif resource_type == "genie-space":
        result["genie_space"] = {
            "name": resource.get("name", ""),
            "space_id": resource["genie_space_id"],
            "permission": permission,
        }
    elif resource_type == "database":
        result["database"] = {
            "instance_name": resource["database_instance_name"],
            "database_name": resource.get(
                "database_name", resource["database_instance_name"]
            ),
            "permission": permission,
        }
    elif resource_type == "secret":
        result["secret"] = {
            "scope": resource["scope"],
            "key": resource["key"],
            "permission": permission,
        }
    elif resource_type == "app":
        # Grants the deployed app's service principal access to another
        # Databricks App (e.g. an MCP server hosted as its own App).
        result["app"] = {
            "name": resource["app_name"],
            "permission": permission,
        }
    elif resource_type in (
        "table",
        "volume",
        "function",
        "connection",
        "vector-search-index",
    ):
        full_name: str = (
            resource.get("table_name")
            or resource.get("volume_name")
            or resource.get("function_name")
            or resource.get("connection_name")
            or resource.get("vector_search_index_name", "")
        )
        # Vector search indexes are UC tables (TABLE_ONLINE_VECTOR_INDEX_*)
        # and work as TABLE securables for maximum workspace compatibility.
        securable_type_map: dict[str, str] = {
            "table": "TABLE",
            "volume": "VOLUME",
            "function": "FUNCTION",
            "connection": "CONNECTION",
            "vector-search-index": "TABLE",
        }
        securable_type: str = securable_type_map[resource_type]
        bundle_permission: str = _BUNDLE_PERMISSION_MAP.get(permission, permission)
        result["uc_securable"] = {
            "securable_full_name": full_name,
            "securable_type": securable_type,
            "permission": bundle_permission,
        }

    return result


def _convert_to_bundle_resources(
    app_resources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert flat app.yaml resource dicts to bundle nested format with deduplication.

    Deduplicates by the underlying Databricks resource identity, keeping only
    the first occurrence when multiple config keys reference the same resource.
    """
    seen: set[Any] = set()
    result: list[dict[str, Any]] = []

    for resource in app_resources:
        converted = _convert_single_resource(resource)
        if converted is None:
            continue

        bundle_key: str | None = None
        for key in _BUNDLE_RESOURCE_CONVERTERS.values():
            if key in converted:
                bundle_key = key
                break

        if bundle_key is None:
            continue

        extractor = _DEDUP_KEY_EXTRACTORS.get(bundle_key)
        if extractor:
            dedup_key = (bundle_key, extractor(converted))
            if dedup_key in seen:
                logger.debug(
                    f"Skipping duplicate resource: {converted['name']} ({dedup_key})"
                )
                continue
            seen.add(dedup_key)

        result.append(converted)

    logger.info(
        f"Converted {len(result)} bundle resources (from {len(app_resources)} app resources)"
    )
    return result


def generate_databricks_yaml(
    config: AppConfig,
    development: bool = False,
    config_filename: str = "dao_ai.yaml",
) -> str:
    """Generate a complete databricks.yaml bundle definition from an AppConfig.

    Reuses generate_app_resources(), _extract_env_vars_from_config(), and
    generate_user_api_scopes() from resources.py -- only the format translation
    to the bundle schema is new.

    When development=True, omits the artifacts section so the pre-built
    dao-ai wheel is uploaded as a regular source file (not intercepted as
    an artifact).

    Note on deployment_target:
        The emitted bundle is always Databricks-Apps-shaped
        (`resources.apps.<name>` with its `resources` list and optional
        `user_api_scopes`). This bundle works regardless of
        `app.deployment_target`:

        - `apps`           → the App IS the deployment target.
        - `model_serving`  → the App process registers the MLflow model
                             and creates the serving endpoint at runtime
                             (via `dao_ai.apps.server`). No separate
                             bundle is needed; users who only want the
                             serving endpoint typically use
                             `dao-ai deploy-agent` instead of
                             `generate-bundle` + `databricks bundle deploy`.

        `generate-bundle` therefore intentionally ignores
        `app.deployment_target`; the enum selects the runtime code path,
        not the bundle layout.
    """
    app_name: str = config.app.name.lower().replace("_", "-")

    enable_chat_proxy: bool = (
        config.app.enable_chat_proxy
        if config.app.enable_chat_proxy is not None
        else True
    )

    env_vars: list[dict[str, str]] = [
        {"name": "MLFLOW_TRACKING_URI", "value": "databricks"},
        {"name": "MLFLOW_REGISTRY_URI", "value": "databricks-uc"},
        {"name": "MLFLOW_EXPERIMENT_ID", "value_from": "experiment"},
        {"name": "DAO_AI_CONFIG_PATH", "value": config_filename},
    ]

    if enable_chat_proxy:
        env_vars.extend(
            [
                {"name": "API_PROXY", "value": "http://localhost:8000/invocations"},
                {"name": "CHAT_APP_PORT", "value": "3000"},
                {"name": "CHAT_PROXY_TIMEOUT_SECONDS", "value": "300"},
            ]
        )

    config_env_vars = _extract_env_vars_from_config(config)
    config_env_vars = [
        e for e in config_env_vars if e["name"] not in _PLATFORM_PROVIDED_ENV_VARS
    ]
    base_env_names: set[str] = {e["name"] for e in env_vars}
    for config_env in config_env_vars:
        if config_env["name"] not in base_env_names:
            if "valueFrom" in config_env:
                config_env["value_from"] = config_env.pop("valueFrom")
            env_vars.append(config_env)

    app_resources = generate_app_resources(config)
    bundle_resources = _convert_to_bundle_resources(app_resources)

    experiment_key: str = f"{app_name}-experiment"
    experiment_app_resource: dict[str, Any] = {
        "name": "experiment",
        "experiment": {
            "experiment_id": f"${{resources.experiments.{experiment_key}.id}}",
            "permission": "CAN_EDIT",
        },
    }
    bundle_resources.insert(0, experiment_app_resource)

    user_api_scopes = generate_user_api_scopes(config)

    app_command: list[str] = (
        ["python", "-m", "dao_ai.apps.start_app"]
        if enable_chat_proxy
        else ["python", "-m", "dao_ai.apps.server"]
    )

    app_def: dict[str, Any] = {
        "name": app_name,
        "description": config.app.description or f"DAO AI Agent: {app_name}",
        "source_code_path": "${workspace.file_path}",
        "config": {
            "command": app_command,
            "env": env_vars,
        },
        "resources": bundle_resources,
    }

    if user_api_scopes:
        app_def["user_api_scopes"] = user_api_scopes

    bundle: dict[str, Any] = {
        "bundle": {
            "name": app_name,
            "engine": "direct",
        },
        "resources": {
            "experiments": {
                experiment_key: {
                    "name": f"/Users/${{workspace.current_user.userName}}/{app_name}",
                },
            },
            "apps": {
                app_name: app_def,
            },
        },
        "targets": {
            "dev": {
                "default": True,
                "mode": "development",
            },
        },
    }

    # Only include artifacts for non-dev bundles. In dev mode the pre-built
    # dao-ai wheel lives in dist/ and must be uploaded as a regular source
    # file, not intercepted by the artifact system.
    if not development:
        bundle["artifacts"] = {
            "default": {
                "type": "whl",
                "build": "uv build",
                "path": ".",
            },
        }

    return yaml.dump(bundle, default_flow_style=False, sort_keys=False)


def _write_file(path: Path, content: str, force: bool) -> bool:
    """Write content to a file, respecting the force flag. Returns True if written."""
    if path.exists() and not force:
        print(
            f"  WARNING: Skipping {path.name} (already exists, use --force to overwrite)"
        )
        return False
    path.write_text(content)
    logger.info(f"Wrote {path.name}")
    return True


def write_bundle(
    config: AppConfig,
    output_dir: Path,
    force: bool = False,
    development: bool = False,
) -> None:
    """Write a complete, deployable Databricks Apps bundle directory.

    Generates databricks.yaml, copies the dao-ai config, and creates
    scaffolding files (pyproject.toml, .gitignore, .python-version).

    When development=True, copies the local dao-ai source into the bundle
    and generates a pyproject.toml that builds from local source instead of
    pulling dao-ai from PyPI.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    app_name: str = config.app.name.lower().replace("_", "-")
    written: list[str] = []
    skipped: list[str] = []

    def _track(path: Path, content: str) -> None:
        if _write_file(path, content, force):
            written.append(path.name)
        else:
            skipped.append(path.name)

    source_config: str | None = getattr(config, "_source_config_path", None)
    config_filename: str = Path(source_config).name if source_config else "dao_ai.yaml"

    _track(
        output_dir / "databricks.yaml",
        generate_databricks_yaml(
            config, development=development, config_filename=config_filename
        ),
    )

    if source_config:
        dest = output_dir / config_filename
        if dest.exists() and not force:
            print(
                f"  WARNING: Skipping {config_filename} (already exists, use --force to overwrite)"
            )
            skipped.append(config_filename)
        else:
            shutil.copy2(source_config, dest)
            logger.info(f"Copied config as {config_filename}")
            written.append(config_filename)
    else:
        logger.warning("No source config path found -- skipping config copy")

    package_name = app_name.replace("-", "_")

    if development:
        from dao_ai.utils import find_dev_wheel

        # In development mode, always rebuild the wheel from local source
        # when source is available. Reusing an existing pre-built wheel is
        # a silent footgun: the deploy succeeds but runs stale code against
        # the user's fresh edits. We only fall back to an existing wheel
        # when source isn't present — e.g. when ``dao-ai`` is running from
        # an installed package and there is no tree to rebuild from.
        #
        # Implementation note: we clear ``dist/dao_ai-*.whl`` before
        # rebuilding so the globbed "latest" result is unambiguous, and the
        # caller downstream won't accidentally pick up an orphan wheel from
        # a previous build.
        project_root: Path = Path(__file__).parents[3]
        source_dir: Path = project_root / "src" / "dao_ai"

        wheel_path: Path | None
        if source_dir.is_dir():
            logger.info(
                "Rebuilding dao-ai wheel from local source (development mode)",
                project_root=str(project_root),
            )
            # Clear existing wheels so the build result is unambiguous.
            for stale in (project_root / "dist").glob("dao_ai-*.whl"):
                stale.unlink()

            result = subprocess.run(
                ["uv", "build", "--wheel"],
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Wheel build failed: {result.stderr}")

            wheels = sorted(
                (project_root / "dist").glob("dao_ai-*.whl"),
                key=lambda p: p.stat().st_mtime,
            )
            if not wheels:
                raise RuntimeError(
                    f"No wheel found in {project_root / 'dist'} after build"
                )
            wheel_path = wheels[-1]
        else:
            wheel_path = find_dev_wheel()
            if not wheel_path:
                raise RuntimeError(
                    "No dao-ai source or pre-built wheel found; cannot "
                    "generate a development bundle."
                )
            logger.info("Using existing dev wheel", wheel=wheel_path.name)

        # Copy wheel into bundle's dist/ directory
        dist_dir = output_dir / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        dest_wheel = dist_dir / wheel_path.name
        shutil.copy2(wheel_path, dest_wheel)
        logger.info("Copied dao-ai wheel for development build", wheel=wheel_path.name)
        written.append(f"dist/{wheel_path.name}")

        # Write dev pyproject.toml referencing local wheel
        _track(
            output_dir / "pyproject.toml",
            _PYPROJECT_DEV_TEMPLATE.format(
                name=app_name,
                package_name=package_name,
                wheel_filename=wheel_path.name,
            ),
        )

        # Create stub package for user's custom code additions
        stub_dir = output_dir / "src" / package_name
        stub_init = stub_dir / "__init__.py"
        if not stub_init.exists() or force:
            stub_dir.mkdir(parents=True, exist_ok=True)
            stub_init.write_text("")
            logger.info(f"Created stub package src/{package_name}/")
            written.append(f"src/{package_name}/__init__.py")
    else:
        _track(
            output_dir / "pyproject.toml",
            _PYPROJECT_TEMPLATE.format(
                name=app_name,
                package_name=package_name,
                dao_ai_version=_get_dao_ai_version(),
            ),
        )

        # Create stub package so the wheel builds and users can add custom code
        stub_dir = output_dir / "src" / package_name
        stub_init = stub_dir / "__init__.py"
        if not stub_init.exists() or force:
            stub_dir.mkdir(parents=True, exist_ok=True)
            stub_init.write_text("")
            logger.info(f"Created stub package src/{package_name}/")
            written.append(f"src/{package_name}/__init__.py")

    # Generate uv.lock so the app runtime uses uv to install from pyproject.toml.
    # When force=True, delete any existing lock first. Otherwise `uv lock` is
    # a no-op when pyproject.toml hasn't changed, even if the local wheel it
    # points at has been rebuilt with a new hash — leading to hash-mismatch
    # failures on deploy.
    lock_path: Path = output_dir / "uv.lock"
    if force and lock_path.exists():
        lock_path.unlink()
        logger.info("Removed existing uv.lock for regeneration (force)")

    logger.info("Resolving dependencies with uv lock...")
    result = subprocess.run(
        ["uv", "lock"],
        cwd=output_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        written.append("uv.lock")
        logger.info("Generated uv.lock")
    else:
        logger.error(f"Failed to generate uv.lock: {result.stderr.strip()}")
        print(
            f"  ERROR: uv lock failed. The app will not install dependencies correctly.\n"
            f"         {result.stderr.strip()}"
        )

    _track(
        output_dir / ".gitignore",
        _GITIGNORE_DEV_CONTENT if development else _GITIGNORE_CONTENT,
    )
    _track(output_dir / ".python-version", "3.11\n")

    print(f"\nBundle generated in {output_dir}/\n")
    for name in written:
        print(f"  {name:<20s} (created)")
    for name in skipped:
        print(f"  {name:<20s} (skipped, already exists)")

    if skipped:
        print("\n  Re-run with --force to overwrite existing files.")

    print("\nNext steps:")
    print(f"  cd {output_dir}")
    print("  uv sync")
    print("  databricks bundle deploy --target dev")
    print(f"  databricks bundle run {app_name} --target dev")
    print()
