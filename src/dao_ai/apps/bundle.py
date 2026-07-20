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
    write_bundle(config, Path("./my-bundle"), overwrite=False)
"""

import io
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml
from loguru import logger
from ruamel.yaml import YAML

from dao_ai.apps.resources import (
    _extract_env_vars_from_config,
    generate_app_resources,
    generate_user_api_scopes,
)
from dao_ai.config import AppConfig, value_of

_BUNDLE_RESOURCE_CONVERTERS: dict[str, str] = {
    "serving-endpoint": "serving_endpoint",
    "sql-warehouse": "sql_warehouse",
    "genie-space": "genie_space",
    "secret": "secret",
    "app": "app",
    "postgres": "postgres",
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
    "secret": lambda r: (r["secret"]["scope"], r["secret"]["key"]),
    "app": lambda r: r["app"]["name"],
    "postgres": lambda r: (r["postgres"]["database"], r["postgres"].get("branch")),
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
# Deps are installed from requirements.txt at deploy time (which points
# at the local wheel under dist/). Pyproject is metadata + hatch build
# target for any user code under src/{package_name}/.
dependencies = []

[tool.hatch.build.targets.wheel]
packages = ["src/{package_name}"]
sources = ["src"]
"""


def _make_requirements_txt(
    *,
    development: bool,
    wheel_filename: Optional[str] = None,
) -> str:
    """Build the requirements.txt content for an app bundle.

    The published pin is left *unbounded* (``dao-ai``) rather than
    floor-pinned to the locally-installed version. ``_get_dao_ai_version()``
    reflects whatever is checked out in the developer's tree, which may
    be an unreleased pre-publish build. Baking that floor into
    requirements.txt would cause Apps to fail with ``Could not find a
    version that satisfies …``. Users who need reproducibility can
    tighten the pin by hand.

    Development mode: reference the bundled wheel via a relative path
    (``./dist/<wheel>``). Pip installs the wheel and resolves transitive
    deps from public PyPI from the wheel's declared dependency metadata.
    """
    if development:
        if not wheel_filename:
            raise ValueError(
                "_make_requirements_txt: wheel_filename is required in development mode."
            )
        return f"./dist/{wheel_filename}\n"
    return "dao-ai\n"


def _get_dao_ai_version() -> str:
    """Return the currently installed dao-ai version for pinning in generated bundles."""
    try:
        return pkg_version("dao-ai")
    except Exception:
        return "0.1.0"


@contextmanager
def _dev_local_version(pyproject_path: Path) -> Iterator[None]:
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
    stamped = original.replace(
        match.group(0), f'version = "{local}"', 1
    )
    try:
        pyproject_path.write_text(stamped)
        logger.info("Stamped dev-build local version", version=local)
        yield
    finally:
        pyproject_path.write_text(original)


def _strip_parameters_block(rendered_yaml: str) -> str:
    """Drop the top-level ``parameters:`` block from a rendered config while
    preserving anchor names, comments, key order, and merge keys.

    PyYAML's ``safe_load`` -> ``safe_dump`` round-trip discards original
    anchor names (rewriting ``&hardware_store_schema`` as ``&id001`` etc.),
    drops comments, and may reorder keys. ruamel.yaml's round-trip mode
    keeps all of that intact, so the deployed config stays readable.

    Args:
        rendered_yaml: YAML text after ``${param.NAME}`` substitution.

    Returns:
        YAML text with the top-level ``parameters:`` key removed.
        If parsing fails or the structure isn't a mapping, the input is
        returned unchanged so we never silently corrupt the user's config.
    """
    rt = YAML(typ="rt")
    rt.preserve_quotes = True
    # Avoid line-wrapping long anchored strings on dump.
    rt.width = 4096

    try:
        data = rt.load(rendered_yaml)
    except Exception as exc:
        logger.warning(f"ruamel parse failed; emitting rendered YAML unchanged: {exc}")
        return rendered_yaml

    # Only mappings have a top-level `parameters:` key. A document whose root
    # is a list, scalar, or None is returned untouched.
    from collections.abc import MutableMapping

    if not isinstance(data, MutableMapping):
        return rendered_yaml

    data.pop("parameters", None)

    buf = io.StringIO()
    rt.dump(data, buf)
    return buf.getvalue()


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
    elif resource_type == "postgres":
        # Lakebase autoscaling project. The platform grants the app SP
        # CAN_CONNECT_AND_CREATE on the project once this resource binds.
        postgres_block: dict[str, Any] = {
            "database": resource["database"],
            "permission": permission,
        }
        branch = resource.get("branch")
        if branch:
            postgres_block["branch"] = branch
        result["postgres"] = postgres_block
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


def _build_app_block(
    config: AppConfig,
    config_filename: str,
    *,
    app_command: list[str] | None = None,
    include_chat_ui: bool = True,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Build the App + experiment dicts shared by `databricks.yaml` and
    `resources/app.yml`.

    Args:
        config: The dao-ai config to derive resources from.
        config_filename: Basename of the config file dropped alongside the
            bundle — surfaced to the runtime via ``DAO_AI_CONFIG_PATH``.
        app_command: Optional override for the App's runtime command. When
            unset, defaults to ``python -m dao_ai.apps.{start_app,server}``
            based on ``config.app.enable_chat_proxy``. Alternate hosts
            (e.g. ``dao-ai generate-mcp``) pass their own command here so
            everything else — env vars, resources, experiment binding —
            can be reused verbatim.
        include_chat_ui: When False, skip the chat-proxy UI env vars
            (``dao_ai.apps.chat_ui.chat_ui_env_vars``). Alternate hosts
            without a bundled chat UI (e.g. the MCP server) opt out here.

    Returns:
        (app_name, experiments_block, apps_block)
    """
    app_name: str = config.app.name.lower().replace("_", "-")

    enable_chat_proxy: bool = (
        config.app.enable_chat_proxy
        if config.app.enable_chat_proxy is not None
        else True
    )

    # The experiment is ALWAYS bound as an App resource named "experiment"
    # (see below). ``MLFLOW_EXPERIMENT_ID`` is uniformly sourced from that
    # binding via ``value_from: experiment`` — DABs resolves it to the
    # bundle-declared experiment's id (auto-derived default case) or to
    # the literal id we set on the resource (``experiment`` configured
    # case). When ``config.app.experiment`` is set we call ``.create(w)``
    # here to populate ``id`` from ``name`` if needed (creating the
    # experiment via ``DatabricksProvider.create_experiment`` when
    # permitted), then use the resolved id verbatim on the app resource.
    _external_experiment_id: str | None = None
    if config.app.experiment is not None:
        config.app.experiment.create()
        _external_experiment_id = config.app.experiment.resolved_id
    env_vars: list[dict[str, str]] = [
        {"name": "MLFLOW_TRACKING_URI", "value": "databricks"},
        {"name": "MLFLOW_REGISTRY_URI", "value": "databricks-uc"},
        {"name": "MLFLOW_EXPERIMENT_ID", "value_from": "experiment"},
        {"name": "DAO_AI_CONFIG_PATH", "value": config_filename},
    ]

    # When trace_location is configured, expose the warehouse id so handlers.py
    # can route MLflow trace export through it. Must be the bare warehouse id
    # (a `value_from: trace_warehouse` would inject the HTTP path, which the
    # MLflow tracing exporter rejects).
    if config.app and config.app.trace_location:
        env_vars.append(
            {
                "name": "MLFLOW_TRACING_SQL_WAREHOUSE_ID",
                "value": value_of(config.app.trace_location.warehouse_id),
            }
        )

    if enable_chat_proxy and include_chat_ui:
        from dao_ai.apps.chat_ui import chat_ui_env_vars

        env_vars.extend(chat_ui_env_vars())

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

    # When trace_location is configured, attach the SQL warehouse + the 3
    # OTEL trace tables as app resources so the platform grants the App SP
    # CAN_USE on the warehouse and SELECT on the OTEL tables (which auto-
    # grants USE CATALOG + USE SCHEMA). generate_app_resources above already
    # emits the flat-format resources for everything declared in
    # config.resources.{warehouses,tables,...}, but the trace-location-
    # synthesized warehouse + OTEL tables come from app.trace_location and
    # _extract_raw_trace_location_resources returns them already in the
    # nested bundle format, so we append them directly here.
    if config.app and config.app.trace_location:
        from dao_ai.apps.resources import _extract_raw_trace_location_resources

        existing_wh_ids: set[str] = set()
        if config.resources:
            for _wh in config.resources.warehouses.values():
                try:
                    existing_wh_ids.add(value_of(_wh.warehouse_id))
                except Exception:
                    pass
        bundle_resources.extend(
            _extract_raw_trace_location_resources(
                config.app.trace_location,
                existing_warehouse_ids=existing_wh_ids,
            )
        )

    # Always bind the experiment as an App resource so the runtime SP can
    # read/write traces to it via auto-auth. Two variants:
    #   * external ``experiment.id`` set → bind by literal id (admin
    #     provisioned the experiment; nothing to declare in
    #     ``experiments_block``). Requested permission downgrades to
    #     CAN_READ when ``manage_permissions=false``, since the deployer
    #     may lack GRANT rights on an admin-owned experiment.
    #   * otherwise (``experiment.name`` or auto-derived) → declare in
    #     ``experiments_block`` and bind via ``${resources.experiments.<key>.id}``
    #     so DABs materializes + grants CAN_EDIT to the App SP.
    experiment_key: str = f"{app_name}-experiment"
    if _external_experiment_id:
        _experiment_permission: str = (
            "CAN_EDIT" if config.app.manage_permissions else "CAN_READ"
        )
        experiment_app_resource: dict[str, Any] = {
            "name": "experiment",
            "experiment": {
                "experiment_id": _external_experiment_id,
                "permission": _experiment_permission,
            },
        }
    else:
        experiment_app_resource = {
            "name": "experiment",
            "experiment": {
                "experiment_id": f"${{resources.experiments.{experiment_key}.id}}",
                "permission": "CAN_EDIT",
            },
        }
    bundle_resources.insert(0, experiment_app_resource)

    user_api_scopes = generate_user_api_scopes(config)

    # Bare `python -m` — no `uv run` wrapper. Apps' native uv support runs
    # `uv sync --locked --no-dev` at BUILD phase and puts .venv/bin on PATH,
    # so the runtime `python` is already venv-python with dao-ai installed.
    if app_command is None:
        app_command = (
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

    if config.app.space:
        app_def["space"] = config.app.space

    # experiments_block sourcing:
    #   - config.app.experiment set → we already resolved the id via
    #     ``ensure_resolved`` above (creating from name if needed). The
    #     experiment lives outside the bundle's lifecycle, so DABs
    #     doesn't declare it — the app resource binding (below) points
    #     at the resolved id literally.
    #   - config.app.experiment omitted → declare
    #     ``/Users/${workspace.current_user.userName}/<app_name>`` in the
    #     bundle so DABs creates + owns it.
    if _external_experiment_id:
        experiments_block: dict[str, Any] = {}
    else:
        experiments_block = {
            experiment_key: {
                "name": f"/Users/${{workspace.current_user.userName}}/{app_name}",
            },
        }
    apps_block: dict[str, Any] = {
        app_name: app_def,
    }
    return app_name, experiments_block, apps_block


def generate_databricks_yaml(
    config: AppConfig,
    development: bool = False,
    config_filename: str = "dao_ai.yaml",
    *,
    app_command: list[str] | None = None,
    include_chat_ui: bool = True,
    include_artifacts: bool = True,
) -> str:
    """Generate the trimmed root `databricks.yaml` for a dao-ai bundle.

    Emits only the bundle-level concerns: ``bundle:`` (with ``include:
    [resources/*.yml]``), ``targets:``, and either ``artifacts:`` or
    ``sync:`` depending on ``development``. The App + experiment block
    lives in ``resources/app.yml`` (see :func:`generate_resources_app_yaml`)
    so users can drop sibling ``resources/*.yml`` files (Workflow Jobs,
    Pipelines, etc.) into the bundle without having to edit the
    regen-owned ``databricks.yaml``.

    When development=True, omits the artifacts section so the pre-built
    dao-ai wheel is uploaded as a regular source file (not intercepted as
    an artifact).

    Note on deployment_target:
        The emitted bundle is always Databricks-Apps-shaped
        (``resources.apps.<name>`` with its ``resources`` list and optional
        ``user_api_scopes``). This bundle works regardless of
        ``app.deployment_target``:

        - ``apps``           → the App IS the deployment target.
        - ``model_serving``  → the App process registers the MLflow model
                               and creates the serving endpoint at runtime
                               (via ``dao_ai.apps.server``). No separate
                               bundle is needed; users who only want the
                               serving endpoint typically use
                               ``dao-ai deploy-agent`` instead of
                               ``generate-bundle`` + ``databricks bundle deploy``.

        ``generate-bundle`` therefore intentionally ignores
        ``app.deployment_target``; the enum selects the runtime code path,
        not the bundle layout.
    """
    app_name, _experiments_block, _apps_block = _build_app_block(
        config,
        config_filename,
        app_command=app_command,
        include_chat_ui=include_chat_ui,
    )

    bundle: dict[str, Any] = {
        "bundle": {
            "name": app_name,
            "engine": "direct",
        },
        "include": ["resources/*.yml"],
        "targets": {
            "dev": {
                "default": True,
                "mode": "development",
            },
        },
    }

    if development:
        # In dev mode the pre-built wheel lives in dist/ and must be
        # uploaded as a regular source file. The bundle CLI excludes
        # .whl files by default, so we add an explicit sync include.
        bundle["sync"] = {
            "include": ["dist/*.whl"],
        }
    elif include_artifacts:
        bundle["artifacts"] = {
            "default": {
                "type": "whl",
                "build": "uv build",
                "path": ".",
            },
        }

    return yaml.dump(bundle, default_flow_style=False, sort_keys=False)


def generate_resources_app_yaml(
    config: AppConfig,
    config_filename: str = "dao_ai.yaml",
    *,
    app_command: list[str] | None = None,
    include_chat_ui: bool = True,
) -> str:
    """Generate ``resources/app.yml`` — the App + experiment block.

    This file is owned by ``generate-bundle``; sibling ``resources/*.yml``
    files (e.g. ``resources/jobs.yml``, ``resources/pipelines.yml``) are
    written by users and are never touched by the generator.

    See :func:`_build_app_block` for ``app_command`` / ``include_chat_ui``
    semantics — alternate hosts (e.g. ``dao-ai generate-mcp``) forward
    both here to reuse the App resource shape verbatim.
    """
    _app_name, experiments_block, apps_block = _build_app_block(
        config,
        config_filename,
        app_command=app_command,
        include_chat_ui=include_chat_ui,
    )

    resources_doc: dict[str, Any] = {
        "resources": {
            "experiments": experiments_block,
            "apps": apps_block,
        },
    }
    return yaml.dump(resources_doc, default_flow_style=False, sort_keys=False)


def _write_file(path: Path, content: str, overwrite: bool) -> bool:
    """Write content to a file, respecting overwrite. Returns True if written."""
    if path.exists() and not overwrite:
        print(
            f"  WARNING: Skipping {path.name} (already exists; use --overwrite)"
        )
        return False
    path.write_text(content)
    logger.info(f"Wrote {path.name}")
    return True


def write_bundle(
    config: AppConfig,
    output_dir: Path,
    overwrite: bool = False,
    development: bool = False,
) -> None:
    """Write a complete, deployable Databricks Apps bundle directory.

    Generates databricks.yaml, copies the dao-ai config, and creates
    scaffolding files (pyproject.toml, .gitignore, .python-version).

    When development=True, copies the local dao-ai source into the bundle
    and generates a pyproject.toml that builds from local source instead of
    pulling dao-ai from PyPI.

    Refuses to write into the dao-ai source repo root — the generator emits
    a ``pyproject.toml`` with ``name = "<app_name>"`` and would silently
    clobber the dao-ai project descriptor if the deployer accidentally ran
    ``generate-bundle`` from inside a dao-ai checkout (the default
    ``--output-dir`` is the CWD).
    """
    resolved_output = output_dir.resolve()
    if (resolved_output / "src" / "dao_ai" / "config.py").exists():
        raise ValueError(
            f"Refusing to write bundle into the dao-ai source repo "
            f"({resolved_output}). ``generate-bundle`` would clobber the "
            "dao-ai project's ``pyproject.toml``. Pass ``-o <path>`` to "
            "target a fresh directory, e.g. "
            "``dao-ai generate-bundle -c <config> -o ./bundle``."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    app_name: str = config.app.name.lower().replace("_", "-")
    written: list[str] = []
    skipped: list[str] = []

    # Warn loudly when the bundle is being built without `trace_location`:
    # Databricks Apps containers cannot reach the artifact-storage host the
    # default MLflow control-plane trace exporter PUTs spans to, so spans
    # silently fail to persist on Apps. Configuring `app.trace_location`
    # routes export through a SQL warehouse → UC OTEL tables (reachable
    # from Apps). Model Serving deploys aren't affected. The warning is
    # informational — generate-bundle still emits a working bundle either way.
    if config.app is None or config.app.trace_location is None:
        _trace_location_warning = (
            "app.trace_location is NOT set. MLflow trace SPANS will NOT "
            "persist when this bundle runs on Databricks Apps — control-plane "
            "trace export targets a storage host that Apps containers cannot "
            "reach, so spans are silently dropped. To capture traces, set "
            "`app.trace_location` in your config (see "
            "config/examples/01_getting_started/ai_gateway.yaml for the YAML "
            "shape). Local notebook/CLI runs and Model Serving deploys are "
            "not affected by this."
        )
        logger.warning(_trace_location_warning)
        print(f"\n  ⚠  {_trace_location_warning}\n")

    def _track(path: Path, content: str) -> None:
        if _write_file(path, content, overwrite):
            written.append(path.name)
        else:
            skipped.append(path.name)

    source_config: str | None = getattr(config, "_source_config_path", None)
    config_filename: str = Path(source_config).name if source_config else "dao_ai.yaml"

    # The chat UI (e2e-chatbot-app-next) is cloned and built at runtime
    # by start_app.py on the Apps container, matching the official
    # Databricks agent template pattern.  No pre-build needed here.

    _track(
        output_dir / "databricks.yaml",
        generate_databricks_yaml(
            config, development=development, config_filename=config_filename
        ),
    )

    # The App + experiment block lives in resources/app.yml so users can drop
    # sibling resources/*.yml files (jobs, pipelines, etc.) into the bundle
    # without conflicting with the regen-owned databricks.yaml.
    resources_dir = output_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    _track(
        resources_dir / "app.yml",
        generate_resources_app_yaml(config, config_filename=config_filename),
    )

    if source_config:
        dest = output_dir / config_filename
        if dest.exists() and not overwrite:
            print(
                f"  WARNING: Skipping {config_filename} (exists; use --overwrite)"
            )
            skipped.append(config_filename)
        else:
            # Prefer the rendered YAML (with ${param.NAME} already substituted
            # and the parameters: declaration block stripped) so the deployed
            # app does not need the original CLI --var arguments. Fall back to
            # a plain copy if the config wasn't loaded via from_file.
            rendered: str | None = getattr(config, "_rendered_yaml", None)
            if rendered is not None:
                dest.write_text(_strip_parameters_block(rendered))
                logger.info(
                    f"Wrote rendered config as {config_filename} (parameters baked in)"
                )
            else:
                shutil.copy2(source_config, dest)
                logger.info(f"Copied config as {config_filename}")
            written.append(config_filename)
    else:
        logger.warning("No source config path found -- skipping config copy")

    # Copy local skill directories into the bundle so they're uploaded with
    # the app source and reachable by deepagents' SkillsMiddleware at
    # runtime. Volume-backed skills are NOT copied (they live on UC volumes
    # and are read directly via the ``/Volumes/...`` path).
    from dao_ai.skills import _project_root, collect_local_skill_dirs

    project_root: Path = _project_root()
    for skill_dir_str in collect_local_skill_dirs(config):
        src_dir: Path = Path(skill_dir_str)
        if not src_dir.exists():
            continue
        # Preserve the relative layout under the project root (e.g.
        # skills/<vertical>/<skill>) so that ``Path.cwd() / spec.path`` resolves
        # at runtime when the bundle root is the app's CWD.
        try:
            rel: Path = src_dir.relative_to(project_root)
        except ValueError:
            # Skill dir is not under the project root — fall back to copying
            # under skills/<basename>. The user can still reference the skill
            # via the rendered absolute path in the deployed config.
            rel = Path("skills") / src_dir.name
        dest = output_dir / rel
        if dest.exists() and not overwrite:
            logger.info(
                "Skipping skill directory copy (exists; use --overwrite)",
                skill=str(rel),
            )
            skipped.append(str(rel))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src_dir, dest)
        logger.info("Copied skill directory into bundle", skill=str(rel))
        written.append(str(rel))

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

            # Stamp a unique local version so the dev wheel always out-ranks the
            # same-base-version published package in the Apps container (pip
            # would otherwise skip a same-version reinstall — see
            # ``_dev_local_version``).
            with _dev_local_version(project_root / "pyproject.toml"):
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

        # Write dev pyproject.toml (metadata + hatch build target).
        # Deps are installed from requirements.txt at deploy time.
        _track(
            output_dir / "pyproject.toml",
            _PYPROJECT_DEV_TEMPLATE.format(
                name=app_name,
                package_name=package_name,
            ),
        )

        # Write requirements.txt pointing at the bundled wheel. Apps'
        # build phase picks this up directly and runs ``pip install -r
        # requirements.txt`` — no uv.lock needed, no pypi-proxy URLs to
        # rewrite, no ambient-env coupling.
        _track(
            output_dir / "requirements.txt",
            _make_requirements_txt(development=True, wheel_filename=wheel_path.name),
        )

        # Create stub package for user's custom code additions
        stub_dir = output_dir / "src" / package_name
        stub_init = stub_dir / "__init__.py"
        if not stub_init.exists() or overwrite:
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

        # Write requirements.txt with version-ranged dao-ai pin. Apps'
        # build phase runs ``pip install -r requirements.txt`` from
        # public PyPI — same path used by deploy_apps_agent's published
        # branch (kept in sync intentionally).
        _track(
            output_dir / "requirements.txt",
            _make_requirements_txt(development=False),
        )

        # Create stub package so the wheel builds and users can add custom code
        stub_dir = output_dir / "src" / package_name
        stub_init = stub_dir / "__init__.py"
        if not stub_init.exists() or overwrite:
            stub_dir.mkdir(parents=True, exist_ok=True)
            stub_init.write_text("")
            logger.info(f"Created stub package src/{package_name}/")
            written.append(f"src/{package_name}/__init__.py")

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
        print("\n  Re-run with --overwrite to overwrite existing files.")

    print("\nNext steps:")
    print(f"  cd {output_dir}")
    print("  databricks bundle deploy --target dev")
    if config.app and config.app.trace_location:
        # Idempotent — safe on every deploy — but load-bearing on
        # re-deploys and after trace_location changes. See
        # `dao-ai link-trace-destination --help` for the full story.
        print(
            f"  dao-ai link-trace-destination -c <your-config.yaml>"
            f"   # links UC trace destination for {app_name}"
        )
    print(f"  databricks bundle run {app_name} --target dev")
    print()
    print("  # Apps' build phase installs deps directly from requirements.txt;")
    print("  # no uv sync or URL rewrite required.")
    print()
