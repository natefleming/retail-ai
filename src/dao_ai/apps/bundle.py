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
    "volume": "uc_securable",
    "function": "uc_securable",
}

_DEDUP_KEY_EXTRACTORS: dict[str, Any] = {
    "serving_endpoint": lambda r: r["serving_endpoint"]["name"],
    "sql_warehouse": lambda r: r["sql_warehouse"]["id"],
    "genie_space": lambda r: r["genie_space"]["space_id"],
    "database": lambda r: (r["database"]["instance_name"], r["database"]["database_name"]),
    "secret": lambda r: (r["secret"]["scope"], r["secret"]["key"]),
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

_PYPROJECT_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
description = "DAO AI Agent: {name}"
requires-python = ">=3.11"
dependencies = [
    "dao-ai>={dao_ai_version}",
]
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
            "database_name": resource.get("database_name", resource["database_instance_name"]),
            "permission": permission,
        }
    elif resource_type == "secret":
        result["secret"] = {
            "scope": resource["scope"],
            "key": resource["key"],
            "permission": permission,
        }
    elif resource_type in ("volume", "function"):
        full_name: str = resource.get("volume_name") or resource.get("function_name", "")
        securable_type: str = "VOLUME" if resource_type == "volume" else "FUNCTION"
        bundle_permission: str = _BUNDLE_PERMISSION_MAP.get(permission, permission)
        result["uc_securable"] = {
            "securable_full_name": full_name,
            "securable_type": securable_type,
            "permission": bundle_permission,
        }

    return result


def _convert_to_bundle_resources(app_resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                logger.debug(f"Skipping duplicate resource: {converted['name']} ({dedup_key})")
                continue
            seen.add(dedup_key)

        result.append(converted)

    logger.info(f"Converted {len(result)} bundle resources (from {len(app_resources)} app resources)")
    return result


def generate_databricks_yaml(config: AppConfig) -> str:
    """Generate a complete databricks.yaml bundle definition from an AppConfig.

    Reuses generate_app_resources(), _extract_env_vars_from_config(), and
    generate_user_api_scopes() from resources.py -- only the format translation
    to the bundle schema is new.
    """
    app_name: str = config.app.name.lower().replace("_", "-")

    env_vars: list[dict[str, str]] = [
        {"name": "MLFLOW_TRACKING_URI", "value": "databricks"},
        {"name": "MLFLOW_REGISTRY_URI", "value": "databricks-uc"},
        {"name": "MLFLOW_EXPERIMENT_ID", "value_from": "experiment"},
        {"name": "DAO_AI_CONFIG_PATH", "value": "dao_ai.yaml"},
    ]

    config_env_vars = _extract_env_vars_from_config(config)
    config_env_vars = [e for e in config_env_vars if e["name"] not in _PLATFORM_PROVIDED_ENV_VARS]
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

    app_def: dict[str, Any] = {
        "name": app_name,
        "description": config.app.description or f"DAO AI Agent: {app_name}",
        "source_code_path": "${workspace.file_path}",
        "config": {
            "command": ["python", "-m", "dao_ai.apps.server"],
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
        "artifacts": {
            "default": {
                "type": "whl",
                "build": "uv build",
                "path": ".",
            },
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

    return yaml.dump(bundle, default_flow_style=False, sort_keys=False)


def _write_file(path: Path, content: str, force: bool) -> bool:
    """Write content to a file, respecting the force flag. Returns True if written."""
    if path.exists() and not force:
        print(f"  WARNING: Skipping {path.name} (already exists, use --force to overwrite)")
        return False
    path.write_text(content)
    logger.info(f"Wrote {path.name}")
    return True


def write_bundle(config: AppConfig, output_dir: Path, force: bool = False) -> None:
    """Write a complete, deployable Databricks Apps bundle directory.

    Generates databricks.yaml, copies the dao-ai config, and creates
    scaffolding files (pyproject.toml, .gitignore, .python-version).
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

    _track(output_dir / "databricks.yaml", generate_databricks_yaml(config))

    source_config: str | None = getattr(config, "_source_config_path", None)
    if source_config:
        dest = output_dir / "dao_ai.yaml"
        if dest.exists() and not force:
            print("  WARNING: Skipping dao_ai.yaml (already exists, use --force to overwrite)")
            skipped.append("dao_ai.yaml")
        else:
            shutil.copy2(source_config, dest)
            logger.info("Copied config as dao_ai.yaml")
            written.append("dao_ai.yaml")
    else:
        logger.warning("No source config path found -- skipping dao_ai.yaml copy")

    _track(
        output_dir / "pyproject.toml",
        _PYPROJECT_TEMPLATE.format(name=app_name, dao_ai_version=_get_dao_ai_version()),
    )
    _track(output_dir / ".gitignore", _GITIGNORE_CONTENT)
    _track(output_dir / ".python-version", "3.11\n")

    print(f"\nBundle generated in {output_dir}/\n")
    for name in written:
        print(f"  {name:<20s} (created)")
    for name in skipped:
        print(f"  {name:<20s} (skipped, already exists)")

    if skipped:
        print(f"\n  Re-run with --force to overwrite existing files.")

    print(f"\nNext steps:")
    print(f"  cd {output_dir}")
    print(f"  uv sync")
    print(f"  databricks bundle deploy --target dev")
    print(f"  databricks bundle run {app_name} --target dev")
    print()
