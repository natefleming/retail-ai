"""
Bundle generation module for creating Databricks Asset Bundle files from dao-ai config.

Generates a complete, deployable bundle directory containing:
- databricks.yaml: Bundle definition with app config, resources, scopes
- dao_ai.yaml: Copy of the dao-ai agent config
- pyproject.toml + uv.lock: dao-ai dependency + portable pinned closure
  (Apps' build phase runs `uv sync --locked --no-dev`)
- .gitignore, .python-version: Scaffolding files

Usage:
    from dao_ai.apps.bundle import write_bundle
    from dao_ai.config import AppConfig

    config = AppConfig.from_file("my_config.yaml")
    write_bundle(config, Path("./my-bundle"), overwrite=False)
"""

import io
import shutil
import subprocess
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml
from loguru import logger
from ruamel.yaml import YAML

from dao_ai._locking import generate_bundle_lock
from dao_ai.apps.resources import (
    _extract_env_vars_from_config,
    generate_app_resources,
    generate_user_api_scopes,
)
from dao_ai.code_paths import _SRC_DIRNAME, code_path_sync_globs
from dao_ai.config import AppConfig, value_of


def dump_bundle_yaml(doc: dict[str, Any]) -> str:
    """Serialize a Databricks Asset Bundle document to YAML.

    Single serialization convention shared by every dao-ai DAB generator
    (``generate-agent``, ``generate-mcp``, and ``dao-ai generate-workflow``): block
    style, insertion order preserved (``sort_keys=False``) so the emitted
    ``databricks.yaml`` reads top-down like the bundle spec.
    """
    return yaml.dump(doc, default_flow_style=False, sort_keys=False)


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
    "dao-ai{extras}=={dao_ai_version}",{extra_deps}
]

[tool.hatch.build.targets.wheel]
# ``packages = ["src"]`` auto-discovers every top-level package under ``src/``
# (the ``src/`` convention), so ``src/foo/bar.py`` installs as ``foo.bar``.
packages = ["src"]
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
# The dao-ai requirement is redirected to the bundled local wheel via
# ``[tool.uv.sources]`` below, so ``uv lock`` records the local build and
# Apps' ``uv sync`` installs THIS code (not PyPI). Everything else resolves
# from public PyPI through dao-ai's own dependency metadata.
dependencies = [
    "dao-ai{extras}",{extra_deps}
]

[tool.hatch.build.targets.wheel]
# ``packages = ["src"]`` auto-discovers every top-level package under ``src/``
# (the ``src/`` convention), so ``src/foo/bar.py`` installs as ``foo.bar``.
packages = ["src"]
sources = ["src"]

[tool.uv.sources]
dao-ai = {{ path = "dist/{wheel_filename}" }}
"""


def _get_dao_ai_version() -> str:
    """Return the installed dao-ai version for pinning in generated bundles."""
    try:
        return pkg_version("dao-ai")
    except Exception:
        return "0.1.0"


def _format_extra_deps(pip_requirements: Optional[Sequence[str]]) -> str:
    """Render user ``config.app.pip_requirements`` as extra lines inside the
    generated pyproject ``dependencies`` array (so ``uv lock`` captures them
    alongside dao-ai). Returns "" when there are none, otherwise a leading
    newline + one indented, quoted requirement per line (no trailing comma —
    the template's dao-ai line already carries one before this block)."""
    reqs = list(pip_requirements or [])
    if not reqs:
        return ""
    return "\n" + "\n".join(f'    "{r}",' for r in reqs)


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


def _retain_only_parameters(rendered_yaml: str, keep: set[str]) -> str:
    """Keep only ``keep`` names under the top-level ``parameters:`` block; drop
    the rest (and the whole block if nothing remains). Preserves anchors,
    comments, and key order via the same ruamel round-trip as
    :func:`_strip_parameters_block`.

    Used by the workflow staging path when a Genie room's ``space_id`` is bound
    to a deferred ``${var.X}`` reference: the deferred params' declarations must
    survive into the staged config so ``06_deploy_agent`` can probe them against
    the provisioning task's taskValues (``AppConfig.from_file`` loops
    ``for name in declarations``). All other declarations are dropped, matching
    :func:`_strip_parameters_block`.
    """
    if not keep:
        return _strip_parameters_block(rendered_yaml)

    rt = YAML(typ="rt")
    rt.preserve_quotes = True
    rt.width = 4096

    try:
        data = rt.load(rendered_yaml)
    except Exception as exc:
        logger.warning(f"ruamel parse failed; emitting rendered YAML unchanged: {exc}")
        return rendered_yaml

    from collections.abc import MutableMapping

    if not isinstance(data, MutableMapping):
        return rendered_yaml

    params = data.get("parameters")
    if isinstance(params, MutableMapping):
        for name in [k for k in params if k not in keep]:
            del params[name]
        if not params:
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
        f"Converted {len(result)} bundle resources "
        f"(from {len(app_resources)} app resources)"
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
    app_name: str = config.app.app_resource_name

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

    # Forward the uvicorn worker count to the backend server (start_app reads
    # DAO_AI_APP_WORKERS and passes it as --workers). Raises the concurrent-
    # request ceiling on multi-core Apps compute.
    if config.app and config.app.workers:
        env_vars.append(
            {"name": "DAO_AI_APP_WORKERS", "value": str(config.app.workers)}
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

    # Coerce workload_size → Apps compute_size (None for Small/Medium leaves the
    # platform default MEDIUM; Large/XLarge set the tier). Raw string so XLARGE
    # passes through regardless of the installed SDK ComputeSize enum.
    apps_compute_size = config.app.apps_compute_size()
    if apps_compute_size:
        app_def["compute_size"] = apps_compute_size

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

    Note on serving mode:
        The emitted bundle is always Databricks-Apps-shaped
        (``resources.apps.<name>`` with its ``resources`` list and optional
        ``user_api_scopes``). This bundle works regardless of serving mode:

        - ``apps``           → the App IS the deployment target.
        - ``model_serving``  → the App process registers the MLflow model
                               and creates the serving endpoint at runtime
                               (via ``dao_ai.apps.server``). No separate
                               bundle is needed; users who only want the
                               serving endpoint typically use
                               ``dao-ai deploy-agent`` instead of
                               ``generate-agent`` + ``databricks bundle deploy``.

        ``generate-agent`` therefore intentionally ignores the serving mode;
        the ``--mode`` CLI flag selects the runtime code path,
        not the bundle layout.
    """
    app_name, _experiments_block, _apps_block = _build_app_block(
        config,
        config_filename,
        app_command=app_command,
        include_chat_ui=include_chat_ui,
    )

    # Explicit sync.include for the App's own source. Databricks bundle sync
    # honors the surrounding repo's .gitignore, so when the bundle is staged
    # under a gitignored dir (dao-ai's default `.dao-ai/bundle/agent/<app>`) a
    # bare `.`-sync uploads nothing and the App fails at run with "no files
    # found". Listing the source files here force-includes them regardless of
    # git-ignore. Harmless when the bundle dir is not ignored.
    sync_include: list[str] = [
        "src/**",
        "resources/**",
        "*.yaml",
        "*.yml",
        "*.toml",
        "uv.lock",
        ".python-version",
    ]
    # Custom code (app.code_paths) is copied into the bundle at config-relative
    # paths (e.g. ``tools/**``); force-include those top-level dirs so they sync
    # from a gitignored staging dir. Entries at the bundle root that aren't dirs
    # are already covered by the globs above.
    for glob in code_path_sync_globs(config):
        if glob not in sync_include:
            sync_include.append(glob)

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
        sync_include.append("dist/*.whl")
    elif include_artifacts:
        bundle["artifacts"] = {
            "default": {
                "type": "whl",
                "build": "uv build",
                "path": ".",
            },
        }

    bundle["sync"] = {"include": sync_include}

    return dump_bundle_yaml(bundle)


def generate_resources_app_yaml(
    config: AppConfig,
    config_filename: str = "dao_ai.yaml",
    *,
    app_command: list[str] | None = None,
    include_chat_ui: bool = True,
) -> str:
    """Generate ``resources/app.yml`` — the App + experiment block.

    This file is owned by ``generate-agent``; sibling ``resources/*.yml``
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
    return dump_bundle_yaml(resources_doc)


def _write_file(path: Path, content: str, overwrite: bool) -> bool:
    """Write content to a file, respecting overwrite. Returns True if written."""
    if path.exists() and not overwrite:
        print(f"  WARNING: Skipping {path.name} (already exists; use --overwrite)")
        return False
    path.write_text(content)
    logger.info(f"Wrote {path.name}")
    return True


def write_bundle(
    config: AppConfig,
    staging_dir: Path,
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
    ``generate-agent`` from inside a dao-ai checkout (the default
    ``--staging-dir`` is the CWD).

    The staging dir is ephemeral build output: dao-ai-generated files
    (databricks.yaml, resources/app.yml, pyproject.toml, uv.lock, scaffold) are
    (re)written every build, while user-owned content (the rendered config,
    code_paths, src/<pkg>, skills, ``resources/`` overlays) is copied once and
    never overwritten unless ``overwrite``. To add your own bundle resources
    without editing a generated file, declare them via ``app.resource_paths`` or
    drop them in a colocated ``resources/`` dir (merged by DABs' ``include:
    [resources/*.yml]``).
    """
    resolved_output = staging_dir.resolve()
    if (resolved_output / "src" / "dao_ai" / "config.py").exists():
        raise ValueError(
            f"Refusing to write bundle into the dao-ai source repo "
            f"({resolved_output}). ``generate-agent`` would clobber the "
            "dao-ai project's ``pyproject.toml``. Pass ``-s <path>`` to "
            "target a fresh directory, e.g. "
            "``dao-ai generate-agent -c <config> -s ./bundle``."
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    app_name: str = config.app.app_resource_name
    written: list[str] = []
    skipped: list[str] = []
    # User code (src/ + code_paths + source config) that already existed at the
    # dest and was left untouched — never overwritten, even with --overwrite.
    preserved: list[str] = []

    def _track(path: Path, content: str) -> None:
        if _write_file(path, content, overwrite):
            written.append(path.name)
        else:
            skipped.append(path.name)

    source_config: str | None = config._source_config_path
    config_filename: str = Path(source_config).name if source_config else "dao_ai.yaml"

    # The chat UI (e2e-chatbot-app-next) is cloned and built at runtime
    # by start_app.py on the Apps container, matching the official
    # Databricks agent template pattern.  No pre-build needed here.

    _track(
        staging_dir / "databricks.yaml",
        generate_databricks_yaml(
            config, development=development, config_filename=config_filename
        ),
    )

    # The App + experiment block lives in resources/app.yml so users can drop
    # sibling resources/*.yml files (jobs, pipelines, etc.) into the bundle
    # without conflicting with the regen-owned databricks.yaml.
    resources_dir = staging_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    _track(
        resources_dir / "app.yml",
        generate_resources_app_yaml(config, config_filename=config_filename),
    )

    if source_config:
        dest = staging_dir / config_filename
        # Never write over the user's ORIGINAL config (would strip their
        # parameters: block irreversibly). Belt-and-suspenders behind the
        # overlap hard-error in the CLI.
        if dest.resolve() == Path(source_config).resolve():
            preserved.append(config_filename)
        elif dest.exists() and not overwrite:
            print(f"  WARNING: Skipping {config_filename} (exists; use --overwrite)")
            skipped.append(config_filename)
        else:
            # Prefer the rendered YAML (with ${param.NAME} already substituted
            # and the parameters: declaration block stripped) so the deployed
            # app does not need the original CLI --var arguments. Fall back to
            # a plain copy if the config wasn't loaded via from_file.
            rendered: str | None = config._rendered_yaml
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
    from dao_ai.skills import _skill_base_dir, collect_local_skill_dirs

    # Anchor on the config's own directory — the same anchor
    # ``collect_local_skill_dirs`` resolved these paths against — rather than
    # ``_project_root()``, which walks up from the CWD. For a config outside the
    # CWD tree (a git checkout in the cache, most obviously) the CWD-based root
    # can never contain the skill dirs, so ``relative_to`` below would always
    # raise and flatten ``skills/<vertical>/<skill>`` to ``skills/<skill>`` while
    # the rendered config still named the nested path — leaving the skill
    # unreachable at runtime. ``_skill_base_dir`` falls back to
    # ``_project_root()`` when the config has no source path.
    project_root: Path = _skill_base_dir(config)
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
        dest = staging_dir / rel
        # In-place (output overlaps the skill source): never rmtree the user's
        # own skills dir. Belt-and-suspenders behind the overlap hard-error.
        if dest.resolve() == src_dir.resolve():
            preserved.append(str(rel))
            continue
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

    # Copy the config's custom code (app.code_paths) into the bundle next to the
    # config so it is importable at runtime: the bundle root is the app CWD and
    # ``add_code_paths_to_sys_path`` inserts each entry's parent onto sys.path.
    # This is the uniform declaration shared with Model Serving and the direct
    # Apps/pipeline deploys; the manual ``src/<package>`` wheel route still works
    # for users who prefer to hand-package their code.
    from dao_ai.code_paths import (
        discover_src_packages,
        iter_code_path_stagings,
        iter_resource_path_stagings,
        walk_code_path_files,
    )

    for src, code_dest in iter_code_path_stagings(config):
        for file_src, file_dest in walk_code_path_files(src, code_dest):
            out = staging_dir / file_dest
            # User code is sacred: never overwrite it (even with --overwrite),
            # and never copy a file onto itself when output overlaps the source.
            if file_src.resolve() == out.resolve():
                preserved.append(file_dest)
                continue
            if out.exists():
                preserved.append(file_dest)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_src, out)
            written.append(file_dest)

    # Convention: copy any colocated ``src/<pkg>`` packages into the bundle's
    # ``src/`` so hatch (``packages=["src"]``) builds them into the app wheel —
    # ``src/foo/bar.py`` installs as ``foo.bar``. No config declaration needed.
    for pkg_dir in discover_src_packages(config):
        for file_src, file_dest in walk_code_path_files(
            pkg_dir, f"{_SRC_DIRNAME}/{pkg_dir.name}"
        ):
            out = staging_dir / file_dest
            if file_src.resolve() == out.resolve():
                preserved.append(file_dest)
                continue
            if out.exists():
                preserved.append(file_dest)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_src, out)
            written.append(file_dest)

    # Copy the config's DAB resource overlays (app.resource_paths + the colocated
    # resources/ convention) into the bundle's resources/ directory, where the
    # generated databricks.yaml's ``include: [resources/*.yml]`` merges them at
    # deploy — so users add their own Jobs/Pipelines/etc. without editing any
    # generated file. Copied from the config dir once; an existing staged copy is
    # refreshed only under --overwrite (matching the field's documented contract),
    # and never copied onto itself.
    for res_src, res_dest in iter_resource_path_stagings(config):
        out = staging_dir / res_dest
        if res_src.resolve() == out.resolve():
            preserved.append(res_dest)
            continue
        if out.exists() and not overwrite:
            preserved.append(res_dest)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(res_src, out)
        written.append(res_dest)

    package_name = app_name.replace("-", "_")

    # Optional-feature extras this config exercises, threaded into the dao-ai
    # dependency so `uv sync --locked` installs them (e.g. "[a2a,rerank]").
    from dao_ai._extras import format_extras_suffix, resolve_required_extras_or_all

    extras_suffix: str = format_extras_suffix(
        resolve_required_extras_or_all(config, target="apps")
    )

    # User-declared extra pip packages (config.app.pip_requirements) — folded
    # into the generated pyproject dependencies so `uv lock` captures them for
    # the deployer's custom code (parity with Model Serving / pipeline).
    # config.app.code_paths files are copied into the bundle above (next to the
    # config) so they import at runtime via ``add_code_paths_to_sys_path``; the
    # manual ``src/<package>`` hatch-wheel route still works for hand-packaged code.
    user_pip_requirements: list[str] = (
        list(config.app.pip_requirements) if config.app else []
    )
    extra_deps: str = _format_extra_deps(user_pip_requirements)

    if development:
        from dao_ai.utils import dev_local_version, find_dev_wheel

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
            # ``dev_local_version``).
            with dev_local_version(project_root / "pyproject.toml"):
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
        dist_dir = staging_dir / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        dest_wheel = dist_dir / wheel_path.name
        shutil.copy2(wheel_path, dest_wheel)
        logger.info("Copied dao-ai wheel for development build", wheel=wheel_path.name)
        written.append(f"dist/{wheel_path.name}")

        # Write dev pyproject.toml (metadata + hatch build target). The dao-ai
        # requirement is redirected to the bundled local wheel via
        # ``[tool.uv.sources]`` so the generated uv.lock installs THIS code.
        _track(
            staging_dir / "pyproject.toml",
            _PYPROJECT_DEV_TEMPLATE.format(
                name=app_name,
                package_name=package_name,
                wheel_filename=wheel_path.name,
                extras=extras_suffix,
                extra_deps=extra_deps,
            ),
        )

        # Create stub package for user's custom code additions. Must exist
        # before locking so uv can build the local project. Only scaffold when
        # ABSENT — never overwrite an existing __init__.py (the user may have a
        # real src/<app_name> package, copied by the src/ loop above), even with
        # --overwrite; clobbering it with an empty file would destroy their code.
        stub_dir = staging_dir / "src" / package_name
        stub_init = stub_dir / "__init__.py"
        if not stub_init.exists():
            stub_dir.mkdir(parents=True, exist_ok=True)
            stub_init.write_text("")
            logger.info(f"Created stub package src/{package_name}/")
            written.append(f"src/{package_name}/__init__.py")

        # Generate the portable uv.lock. Apps' build phase runs
        # ``uv sync --locked --no-dev`` from pyproject.toml + uv.lock (no
        # requirements.txt). The lock references the bundled wheel via the
        # ``[tool.uv.sources]`` path and pins the full public-PyPI closure.
        generate_bundle_lock(staging_dir)
        written.append("uv.lock")
    else:
        _track(
            staging_dir / "pyproject.toml",
            _PYPROJECT_TEMPLATE.format(
                name=app_name,
                package_name=package_name,
                dao_ai_version=_get_dao_ai_version(),
                extras=extras_suffix,
                extra_deps=extra_deps,
            ),
        )

        # Create stub package so the wheel builds and users can add custom code.
        # Only scaffold when ABSENT — never overwrite an existing __init__.py
        # (the user may have a real src/<app_name> package, copied above), even
        # with --overwrite; clobbering it with an empty file destroys their code.
        stub_dir = staging_dir / "src" / package_name
        stub_init = stub_dir / "__init__.py"
        if not stub_init.exists():
            stub_dir.mkdir(parents=True, exist_ok=True)
            stub_init.write_text("")
            logger.info(f"Created stub package src/{package_name}/")
            written.append(f"src/{package_name}/__init__.py")

        # Generate the portable uv.lock. Apps' build phase runs
        # ``uv sync --locked --no-dev`` from pyproject.toml + uv.lock (no
        # requirements.txt). The sole ``dao-ai>={ver}`` dep resolves the full
        # public-PyPI closure at lock time; the lock pins it.
        generate_bundle_lock(staging_dir)
        written.append("uv.lock")

    _track(
        staging_dir / ".gitignore",
        _GITIGNORE_DEV_CONTENT if development else _GITIGNORE_CONTENT,
    )
    _track(staging_dir / ".python-version", "3.11\n")

    print(f"\nBundle generated in {staging_dir}/\n")
    for name in written:
        print(f"  {name:<20s} (created)")
    for name in skipped:
        print(f"  {name:<20s} (skipped, already exists)")
    for name in preserved:
        print(f"  {name:<20s} (preserved — your code, not overwritten)")

    if skipped:
        print(
            "\n  Re-run with --overwrite to overwrite generated scaffold "
            "(your src/, code_paths, and config are always preserved)."
        )

    print("\nNext steps:")
    print(f"  cd {staging_dir}")
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
    print("  # Apps' build phase installs deps via `uv sync --locked` from")
    print("  # pyproject.toml + uv.lock (portable, public-CDN URLs).")
    print()
