"""Staging-bundle generation for the ``dao-ai generate-workflow`` Lakeflow job.

Where :mod:`dao_ai.apps.bundle` emits a Databricks *App* bundle, this module
emits the multi-task *Job* (``deploy_job``) bundle that ``dao-ai generate-workflow``
submits. The whole point is to run from an installed ``dao-ai`` wheel with **no
source checkout**: the ``databricks.yaml`` is built programmatically here
(matching ``generate_databricks_yaml`` in :mod:`dao_ai.apps.bundle`), and the
step notebooks ship as package data under ``dao_ai/pipeline/`` and are
materialized into a staging directory.

Config-referenced ``ddl``/``data`` assets are colocated with the config and
resolved against the config's own directory (see ``AppConfig.from_file`` /
``DatasetModel.resolve_asset_path``), so they stage next to the staged config
under ``config/`` and resolve identically when the notebook reloads the staged
config::

    <staging_dir>/
      databricks.yaml          # built programmatically (dict -> yaml.dump)
      notebooks/NN_*.py        # packaged step notebooks
      config/<name>.yaml       # the resolved dao-ai config
      config/functions/...     # copied config-relative UC-function ddl files
      config/data/...          # copied config-relative dataset ddl/data files
      dist/dao_ai-*.whl        # development mode only: the current build's wheel

dao-ai (with its own transitive deps) installs via the serverless job
environment's ``dao_ai_dep`` dependency (see ``generate_pipeline_databricks_yaml``),
mirroring the Apps deploy paths: the bundled ``dist/`` wheel in development mode,
or the ``dao-ai`` PyPI package otherwise. There is no pinned ``requirements.txt``.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

from loguru import logger

from dao_ai.apps.bundle import (
    _retain_only_parameters,
    _strip_parameters_block,
    _write_file,
    dump_bundle_yaml,
)
from dao_ai.config import AppConfig
from dao_ai.config_vars import ParameterDeclarationModel, substitute_params
from dao_ai.utils import dao_ai_version, normalize_name

# Package location of the step notebooks shipped in the wheel.
_NOTEBOOKS_PKG = "dao_ai.pipeline.notebooks"

# The clouds each get their own bundle target so one app has a distinct
# deployment identity per cloud (``<app>-<cloud>``).
_CLOUDS = ("azure", "aws", "gcp")

# The provisioning job's task DAG. Static — it derives nothing from the config
# (only the bundle name does), so it is expressed once here as data. Each entry
# is (task_key, notebook, depends_on, extra_base_parameters).
_PIPELINE_TASKS: tuple[tuple[str, str, tuple[str, ...], dict[str, str]], ...] = (
    # Creates the service principal and stores its credentials — no grants (see
    # grant-service-principal below). Runs first because provision-lakebase needs
    # the SP to exist: DatabaseModel.create() makes a Postgres role whose subject
    # is the SP's client_id. Idempotent, so it is safe on every pipeline run.
    ("provision-service-principal", "00_provision_service_principal.py", (), {}),
    ("ingest-and-transform", "01_ingest_and_transform.py", (), {}),
    (
        "provision-vector-search",
        "02_provision_vector_search.py",
        ("ingest-and-transform",),
        {},
    ),
    (
        "provision-lakebase",
        "03_provision_lakebase.py",
        ("provision-service-principal",),
        {},
    ),
    (
        "unity-catalog-tools",
        "04_unity_catalog_tools.py",
        ("provision-vector-search", "provision-lakebase"),
        {},
    ),
    ("provision-genie", "05_provision_genie.py", ("unity-catalog-tools",), {}),
    # Authorizes the service principal AFTER every resource it needs exists —
    # tables (01/02), the Lakebase project (03), UC functions (04), the Genie
    # space (05) — and BEFORE the agent goes live, because a deployed agent needs
    # its permissions at startup. Granting used to happen inside
    # provision-service-principal, at the front of the DAG, where every target was
    # still absent.
    (
        "grant-service-principal",
        "06_grant_service_principal.py",
        (
            "provision-service-principal",
            "provision-lakebase",
            "unity-catalog-tools",
            "provision-genie",
        ),
        {},
    ),
    (
        "deploy-agents",
        "07_deploy_agent.py",
        ("grant-service-principal",),
        {
            "mode": "${var.mode}",
            "as_mcp": "${var.as_mcp}",
            "development": "${var.development}",
        },
    ),
    (
        "generate-evaluation-data",
        "08_generate_evaluation_data.py",
        ("provision-vector-search",),
        {},
    ),
    (
        "run-evaluation",
        "09_run_evaluation.py",
        ("deploy-agents", "generate-evaluation-data"),
        {"mode": "${var.mode}"},
    ),
)


# The single-task DAG for the thin ``dao-ai agent --mode model_serving`` Job
# bundle: it skips all provisioning (01–05, 07–08) and runs ONLY the deploy-agent
# notebook, which logs+registers the MLflow model and deploys the serving
# endpoint. Same notebook the pipeline's ``deploy-agents`` task runs; here it is
# the only task, so it has no ``depends_on``. ``mode``/``development`` ride the
# same bundle variables the pipeline uses, so the shared Job deploy driver
# (``run_databricks_command``) forwards them unchanged.
_MODEL_SERVING_AGENT_TASKS: tuple[
    tuple[str, str, tuple[str, ...], dict[str, str]], ...
] = (
    (
        "deploy-agent",
        "07_deploy_agent.py",
        (),
        {
            "mode": "${var.mode}",
            "as_mcp": "${var.as_mcp}",
            "development": "${var.development}",
        },
    ),
)


def _build_job_bundle_yaml(
    config: AppConfig,
    development: bool,
    *,
    tasks_spec: tuple[tuple[str, str, tuple[str, ...], dict[str, str]], ...],
    default_mode: str,
    extras_target: str,
) -> str:
    """Build a Lakeflow **Job** bundle ``databricks.yaml`` (dict -> YAML).

    Shared by the multi-task provisioning pipeline
    (:func:`generate_pipeline_databricks_yaml`) and the thin single-task
    model_serving agent bundle
    (:func:`generate_model_serving_agent_databricks_yaml`). The variables,
    serverless environment, per-cloud targets, and sync globs are identical
    across both; only the task DAG, the ``mode`` variable default, and the
    extras-resolution target differ.

    Programmatic, matching :func:`dao_ai.apps.bundle.generate_databricks_yaml`
    (both serialize via :func:`dao_ai.apps.bundle.dump_bundle_yaml`). There is no
    ``artifacts:`` block: the dao-ai wheel is not built at ``bundle deploy`` time.
    In development mode a pre-built wheel is staged into ``dist/`` (added to
    ``sync.include`` so the CLI uploads it as a source file); otherwise the
    notebooks install ``dao-ai`` from PyPI.
    """
    if config.app is None:
        raise ValueError("Config must have an 'app' section to build the bundle.")
    app_name = normalize_name(config.app.name)

    sync_include = [
        "config/**",
        "notebooks/*.py",
    ]
    # DAB resource overlays (app.resource_paths + the colocated resources/
    # convention) stage under resources/ and are merged via the bundle's
    # ``include: [resources/*.yml]`` (added below) — the same seam the agent/mcp
    # bundles expose, so a config's overlays behave identically no matter which
    # noun deploys it.
    if config.app is not None and _has_resource_overlays(config):
        sync_include.append("resources/**")
    # Config-referenced assets stage under ``config/`` (covered above). A config
    # that references a sibling use case's shared assets via ``../`` stages them
    # outside ``config/`` at the bundle root; add explicit globs so those upload
    # too (the staging dir is gitignored, so nothing syncs implicitly).
    # ``app.code_paths`` files also stage under ``config/`` (see
    # ``_stage_code_paths``), so ``config/**`` already covers them — no extra
    # code_paths globs are needed here.
    for glob in _asset_sync_globs(config):
        if glob not in sync_include:
            sync_include.append(glob)
    # Development mode: the ``dao_ai_dep`` var is a BARE local wheel path (no
    # ``[extras]`` — the bundle globs local-path deps, and ``[a2a]`` would be a
    # glob char class). So the optional-feature extras the config needs can't
    # ride on the wheel; pin their backing packages as separate, glob-safe PyPI
    # deps instead — same approach as the Model Serving dev path. Published mode
    # keeps extras on the ``dao-ai[extras]==ver`` spec (a PyPI spec is glob-safe).
    extra_dep_pins: list[str] = []
    if development:
        # The bundle CLI excludes .whl by default; include the staged wheel so
        # the serverless environment can install it (../dist/<wheel> via the
        # ``dao_ai_dep`` variable).
        sync_include.append("dist/*.whl")

        from dao_ai._extras import expand_all, resolve_required_extras_or_all
        from dao_ai.utils import get_installed_packages

        required_extras = expand_all(
            resolve_required_extras_or_all(config, target=extras_target)
        )
        extra_dep_pins = get_installed_packages(required_extras)

    tasks: list[dict[str, Any]] = []
    for task_key, notebook, depends_on, extra_params in tasks_spec:
        task: dict[str, Any] = {"task_key": task_key}
        if depends_on:
            task["depends_on"] = [{"task_key": d} for d in depends_on]
        task["notebook_task"] = {
            "notebook_path": f"./notebooks/{notebook}",
            "base_parameters": {"config-path": "${var.config_path}", **extra_params},
        }
        task["environment_key"] = "dao-ai-env"
        tasks.append(task)

    bundle: dict[str, Any] = {
        "bundle": {"name": app_name},
        "sync": {"include": sync_include},
        "variables": {
            "config_path": {
                "description": "Path to the configuration file for the job.",
            },
            "cloud": {
                "description": "Cloud provider (azure, aws, gcp) - set per target.",
            },
            "mode": {
                "description": "Agent serving platform (model_serving, apps).",
                "default": default_mode,
            },
            "as_mcp": {
                "description": (
                    "Serve the agent over MCP instead of the chat UI "
                    "(true/false; requires mode=apps). Deploys as mcp-<app>."
                ),
                "default": "false",
            },
            "development": {
                "description": (
                    "Source selection (auto, true, false) - true ships the "
                    "local dao-ai wheel, false pins PyPI, auto detects."
                ),
                "default": "auto",
            },
            "dao_ai_dep": {
                "description": (
                    "dao-ai dependency for the serverless environment - the "
                    "bundled './dist/<wheel>' in development mode, or the "
                    "version-pinned PyPI spec otherwise, each carrying the "
                    "optional-feature extras the config uses (e.g. "
                    "'dao-ai[a2a,rerank]==X.Y.Z'). The CLI overrides this per "
                    "deploy; the default pins the generating version so a raw "
                    "``databricks bundle deploy`` stays reproducible."
                ),
                "default": f"dao-ai=={dao_ai_version()}",
            },
        },
        "resources": {
            "jobs": {
                "deploy_job": {
                    "name": f"{app_name}-job",
                    "tags": {"app_name": app_name},
                    "environments": [
                        {
                            "environment_key": "dao-ai-env",
                            "spec": {
                                "environment_version": "5",
                                # dao-ai via the var (published: dao-ai[extras]==ver;
                                # development: bare local wheel), then the dev-mode
                                # extra-feature package pins (glob-safe PyPI specs,
                                # empty in published mode), then any user-declared
                                # extra pip packages. Parity with Model Serving / Apps.
                                "dependencies": [
                                    "${var.dao_ai_dep}",
                                    *extra_dep_pins,
                                    *(
                                        list(config.app.pip_requirements)
                                        if config.app
                                        else []
                                    ),
                                ],
                            },
                        }
                    ],
                    "tasks": tasks,
                }
            }
        },
        "targets": {
            f"{app_name}-{cloud}": {
                "mode": "development",
                "variables": {"cloud": cloud},
            }
            for cloud in _CLOUDS
        },
    }

    # Merge user DAB resource overlays (app.resource_paths + the colocated
    # resources/ convention) the same way the agent/mcp bundles do — so a config's
    # overlays deploy identically regardless of noun. Only emitted when the config
    # actually has overlays, so a plain workflow bundle stays free of a dangling
    # (empty) include.
    if config.app is not None and _has_resource_overlays(config):
        bundle["include"] = ["resources/*.yml"]

    return dump_bundle_yaml(bundle)


def generate_pipeline_databricks_yaml(config: AppConfig, development: bool) -> str:
    """Build the multi-task ``deploy_job`` bundle ``databricks.yaml`` (dict -> YAML).

    Emits the Lakeflow **Job** — the 8-task provisioning DAG — with a per-cloud
    target (``<app>-<cloud>``) and the five bundle variables the step notebooks
    read. See :func:`_build_job_bundle_yaml` for the shared bundle shape.
    """
    return _build_job_bundle_yaml(
        config,
        development,
        tasks_spec=_PIPELINE_TASKS,
        default_mode="apps",
        extras_target="pipeline",
    )


def generate_model_serving_agent_databricks_yaml(
    config: AppConfig, development: bool
) -> str:
    """Build the thin single-task model_serving agent ``databricks.yaml``.

    Same Job bundle shape as the provisioning pipeline
    (:func:`_build_job_bundle_yaml`), but the DAG is a single ``deploy-agent``
    task that runs ``07_deploy_agent.py`` to register the MLflow model and deploy
    the serving endpoint — no ingest/vector-search/lakebase/genie/eval tasks. The
    ``mode`` variable defaults to ``model_serving`` and extras resolve for the
    Model Serving target (a leaner serving image than the full pipeline).
    """
    return _build_job_bundle_yaml(
        config,
        development,
        tasks_spec=_MODEL_SERVING_AGENT_TASKS,
        default_mode="model_serving",
        extras_target="model_serving",
    )


def _materialize_notebooks(
    staging_dir: Path, overwrite: bool, *, only: set[str] | None = None
) -> list[str]:
    """Copy the packaged step notebooks into ``<staging_dir>/notebooks/``.

    Only the wired ``NN_*.py`` step notebooks are materialized; the package
    ``__init__.py`` marker is skipped. When ``only`` is given, restrict to those
    filenames (the thin model_serving agent bundle stages just
    ``07_deploy_agent.py``); otherwise materialize every step notebook.
    """
    notebooks_dir = staging_dir / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for entry in files(_NOTEBOOKS_PKG).iterdir():
        if entry.name == "__init__.py" or not entry.name.endswith(".py"):
            continue
        if only is not None and entry.name not in only:
            continue
        if _write_file(
            notebooks_dir / entry.name,
            entry.read_text(encoding="utf-8"),
            overwrite,
        ):
            written.append(f"notebooks/{entry.name}")
    return written


def _referenced_asset_paths(config: AppConfig) -> list[str]:
    """Collect the relative filesystem paths a config's provisioning steps read.

    Walks ``config.datasets`` (``ddl``/``data``) and
    ``config.unity_catalog_functions`` (``ddl``) and returns the values that are
    *string* relative filesystem paths — i.e. NOT ``VolumeModel`` /
    ``VolumePathModel`` references (those live on UC volumes and need no
    staging) and NOT absolute paths.

    Paths are config-relative (``functions/find_x.sql``): assets are colocated
    with the config. Returned verbatim so the caller can resolve them against
    the config's own directory and stage them next to the staged config.
    """
    candidates: list[str | None] = []
    for dataset in config.datasets or []:
        candidates.append(dataset.ddl if isinstance(dataset.ddl, str) else None)
        candidates.append(dataset.data if isinstance(dataset.data, str) else None)
    for fn in config.unity_catalog_functions or []:
        candidates.append(fn.ddl if isinstance(fn.ddl, str) else None)

    paths: list[str] = []
    for value in candidates:
        if value and not Path(value).is_absolute() and value not in paths:
            paths.append(value)
    return paths


def _asset_sync_globs(config: AppConfig) -> list[str]:
    """Bundle-root-relative sync globs covering the staged asset locations.

    Assets stage next to the staged config (under ``config/<rel>``), so most
    are already covered by ``config/**``. Config-relative paths that climb out
    of ``config/`` with ``../`` (a config referencing a sibling use case's
    shared assets) land elsewhere under the bundle root; the staging dir is
    gitignored, so those top-level dirs need explicit sync coverage or the
    files never upload. Returns the extra ``<top-dir>/**`` globs needed.
    """
    import posixpath

    globs: list[str] = []
    for rel in _referenced_asset_paths(config):
        # Normalize ``config/../x/y`` → ``x/y`` to find the bundle-root top dir.
        norm_str = posixpath.normpath(posixpath.join("config", rel))
        top = norm_str.split("/", 1)[0]
        glob = f"{top}/**"
        if top not in {"config", ""} and glob not in globs:
            globs.append(glob)
    return globs


def _stage_assets(
    config: AppConfig,
    staging_dir: Path,
    overwrite: bool,
) -> tuple[list[str], list[str]]:
    """Copy config-referenced data/functions files into the staging bundle.

    Config asset paths are relative to the config file's own directory — e.g.
    ``data: data/products.snappy.parquet`` in
    ``.../hardware_store/hardware_store.yaml`` means
    ``.../hardware_store/data/products.snappy.parquet``. The config stages to
    ``<staging_dir>/config/<name>.yaml``, so we copy each asset to the same
    path relative to that staged config (``<staging_dir>/config/<rel>``); the
    provisioning notebook reloads the staged config and its
    :meth:`DatasetModel.resolve_asset_path` resolves the same relative path
    against the staged config's directory, finding the copy.

    Returns ``(copied, missing)`` where ``missing`` lists referenced paths not
    found at stage time (reported to the user, never silently dropped — a
    missing seed file means a provisioning step will fail at run time).
    """
    source_config: str | None = config._source_config_path
    if source_config is None:
        # No source path (e.g. programmatically built config) — nothing to
        # resolve relative paths against, so nothing to stage.
        return [], []

    src_anchor = Path(source_config).resolve().parent
    dest_anchor = (staging_dir / "config").resolve()
    copied: list[str] = []
    missing: list[str] = []

    def _staged_label(dest: Path) -> str:
        """Bundle-relative label for the summary (e.g. ``config/foo/bar.sql``)."""
        try:
            return str(dest.relative_to(staging_dir))
        except ValueError:
            return str(dest)

    for rel in _referenced_asset_paths(config):
        src = (src_anchor / rel).resolve()
        dest = (dest_anchor / rel).resolve()
        if not src.exists():
            missing.append(rel)
            continue
        if src == dest:
            # Staging in place (staging_dir is the source tree) — nothing to copy.
            continue
        if dest.exists() and not overwrite:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(_staged_label(dest))

    return copied, missing


def _stage_code_paths(
    config: AppConfig,
    staging_dir: Path,
) -> tuple[list[str], list[str]]:
    """Copy ``config.app.code_paths`` files into the staged bundle under ``config/``.

    Custom code stages next to the staged config (``config/<dest>``) so that when
    ``07_deploy_agent.py`` reloads the staged config, ``add_code_paths_to_sys_path``
    inserts the staged parent and ``create_agent``'s ``collect_code_paths`` resolves
    against the staged config directory.

    User code is sacred: an existing file at the dest is preserved (never
    overwritten, even in a user-managed ``-o`` dir), and a file is never copied
    onto itself. Returns ``(copied, preserved)`` bundle-relative labels.
    """
    from dao_ai.code_paths import iter_code_path_stagings, walk_code_path_files

    dest_anchor = (staging_dir / "config").resolve()
    copied: list[str] = []
    preserved: list[str] = []
    for src, dest in iter_code_path_stagings(config):
        for file_src, file_dest in walk_code_path_files(src, dest):
            file_out = (dest_anchor / file_dest).resolve()
            label = _bundle_label(file_out, staging_dir)
            if file_src == file_out or file_out.exists():
                preserved.append(label)
                continue
            file_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_src, file_out)
            copied.append(label)
    return copied, preserved


def _stage_src_packages(
    config: AppConfig,
    staging_dir: Path,
) -> tuple[list[str], list[str]]:
    """Copy colocated ``src/<pkg>`` packages into the staged bundle under ``config/src``.

    The ``src/`` convention: packages stage under ``config/src/<pkg>`` (next to
    the staged config), so when ``07_deploy_agent.py`` reloads the staged config
    its ``src/`` anchor is ``config/src`` — ``collect_serving_code_paths`` passes
    each ``config/src/<pkg>`` to ``log_model`` (MLflow -> ``code/<pkg>``) and
    ``prepend_src_to_sys_path`` puts ``config/src`` on ``sys.path``, both yielding
    ``<pkg>.mod``.

    User code is sacred (see :func:`_stage_code_paths`). Returns
    ``(copied, preserved)`` bundle-relative labels.
    """
    from dao_ai.code_paths import (
        _SRC_DIRNAME,
        discover_src_packages,
        walk_code_path_files,
    )

    dest_anchor = (staging_dir / "config").resolve()
    copied: list[str] = []
    preserved: list[str] = []
    for pkg_dir in discover_src_packages(config):
        for file_src, file_dest in walk_code_path_files(
            pkg_dir, f"{_SRC_DIRNAME}/{pkg_dir.name}"
        ):
            file_out = (dest_anchor / file_dest).resolve()
            label = _bundle_label(file_out, staging_dir)
            if file_src == file_out or file_out.exists():
                preserved.append(label)
                continue
            file_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_src, file_out)
            copied.append(label)
    return copied, preserved


def _has_resource_overlays(config: AppConfig) -> bool:
    """True if the config contributes any DAB resource overlay.

    Either an explicit ``app.resource_paths`` entry or an implicit ``*.yml`` under
    the colocated ``resources/`` directory. Used to decide whether the workflow
    bundle emits the ``include: [resources/*.yml]`` seam + ``resources/**`` sync
    glob (skipped for a plain bundle so no dangling empty include is written).
    """
    from dao_ai.code_paths import discover_resource_overlays

    app = config.app
    if app is None:
        return False
    return bool(app.resource_paths) or bool(discover_resource_overlays(config))


def _stage_resource_overlays(
    config: AppConfig,
    staging_dir: Path,
    overwrite: bool,
) -> tuple[list[str], list[str]]:
    """Copy the config's DAB resource overlays into ``resources/``.

    Parity with the agent/mcp bundles: each overlay (from ``app.resource_paths`` or
    the colocated ``resources/`` convention) lands flat at ``resources/<basename>``
    (bundle root), where the generated ``databricks.yaml``'s
    ``include: [resources/*.yml]`` merges it at deploy — so a config's overlays
    behave identically whichever noun deploys it. Copied once; an existing staged
    copy is refreshed only under ``overwrite`` (matching the field's documented
    contract), and never copied onto itself. Returns ``(copied, preserved)``
    bundle-relative labels.
    """
    from dao_ai.code_paths import iter_resource_path_stagings

    copied: list[str] = []
    preserved: list[str] = []
    for res_src, res_dest in iter_resource_path_stagings(config):
        file_out = (staging_dir / res_dest).resolve()
        label = _bundle_label(file_out, staging_dir)
        if res_src.resolve() == file_out:
            preserved.append(label)
            continue
        if file_out.exists() and not overwrite:
            preserved.append(label)
            continue
        file_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(res_src, file_out)
        copied.append(label)
    return copied, preserved


def _bundle_label(file_out: Path, staging_dir: Path) -> str:
    """Bundle-relative label for the staging summary (best-effort)."""
    try:
        return str(file_out.relative_to(staging_dir))
    except ValueError:
        return str(file_out)


def _deferred_provided_params(config: AppConfig) -> set[str]:
    """Parameter names to LEAVE unresolved (``${var.X}``) in the staged config.

    A parameter declared ``provided: true`` gets its value dynamically at run
    time — e.g. ``05_provision_genie`` creates a Genie space and forwards its id
    via taskValues keyed by the param name, which ``07_deploy_agent`` then injects
    via ``AppConfig.from_file(task_values=...)``. For that to work the staged
    config must keep the ``${var.X}`` reference (and its declaration) rather than
    bake it.

    A param is deferred only when the operator did NOT supply it on the CLI
    (``--param``/``--var``): an operator-supplied value means "use this" — it bakes
    to a literal and no run-time fill-in happens. Keying off the declaration flag
    (not genie-room ``space_id`` inference) keeps this general: any ``provided``
    param is deferred, whatever consumes it.
    """
    declarations = config._declarations or {}
    if not declarations:
        return set()
    supplied: set[str] = config._operator_supplied_params or set()
    return {
        name
        for name, decl in declarations.items()
        if decl.provided and name not in supplied
    }


def _staged_config_text(
    config: AppConfig, *, defer_provided: bool = True
) -> str | None:
    """The dao-ai config text to stage for a workflow bundle.

    Default: the fully-substituted rendered YAML with the ``parameters:`` block
    stripped (deployed job needs no ``--var``). When the config has
    ``provided: true`` params to defer (see :func:`_deferred_provided_params`),
    re-render the pre-substitution source leaving those refs in place and retain
    ONLY their declarations, so ``05_provision_genie`` can provision and
    ``07_deploy_agent`` can inject the forwarded id. Returns None if the config
    was not loaded via ``AppConfig.from_file`` (no rendered text available).

    ``defer_provided=False`` bakes every param to a literal (no ``${var.X}``
    left) — used by the thin model_serving agent bundle, which has no
    provisioning task to fill deferred values (the CLI asserts all ``provided``
    params are satisfied before staging), mirroring the Apps bundle.
    """
    rendered: str | None = config._rendered_yaml
    if rendered is None:
        return None

    defer: set[str] = _deferred_provided_params(config) if defer_provided else set()
    if not defer:
        return _strip_parameters_block(rendered)

    # Re-render from the pre-substitution source with the deferred names left in
    # place, then keep only the deferred declarations. Falls back to the plain
    # rendered text if the config lacks the stashed pre-substitution inputs.
    source_text: str | None = config._workspace_resolved_yaml
    if source_text is None:
        return _strip_parameters_block(rendered)
    declarations: dict[str, ParameterDeclarationModel] = config._declarations or {}
    deferred_render: str = substitute_params(
        source_text,
        declarations=declarations,
        cli_vars=config._substitution_vars or None,
        defer=defer,
        source=config._source_config_path or "<config>",
    )
    return _retain_only_parameters(deferred_render, keep=defer)


def write_pipeline_bundle(
    config: AppConfig,
    staging_dir: Path,
    overwrite: bool = False,
    development: bool = False,
) -> None:
    """Stage a deployable multi-task ``deploy_job`` bundle for ``dao-ai workflow``.

    Thin wrapper over :func:`_write_job_bundle` staging the full 8-task
    provisioning DAG. See that function for the staging contract.
    """
    _write_job_bundle(
        config,
        staging_dir,
        overwrite=overwrite,
        development=development,
        yaml_generator=generate_pipeline_databricks_yaml,
        notebook_only=None,
        defer_provided=True,
        label="Workflow",
    )


def write_model_serving_agent_bundle(
    config: AppConfig,
    staging_dir: Path,
    overwrite: bool = False,
    development: bool = False,
) -> None:
    """Stage the thin single-task model_serving agent ``deploy_job`` bundle.

    The ``dao-ai agent --mode model_serving`` DAB analogue of the Apps/MCP
    bundles: a Lakeflow Job whose one ``deploy-agent`` task runs
    ``07_deploy_agent.py`` to register the MLflow model and deploy the serving
    endpoint. No provisioning tasks and no upstream ``provided``-param filler, so
    the config is baked fully (``defer_provided=False``, like the Apps bundle);
    the CLI asserts all ``provided`` params are satisfied before staging.
    Thin wrapper over :func:`_write_job_bundle`.
    """
    _write_job_bundle(
        config,
        staging_dir,
        overwrite=overwrite,
        development=development,
        yaml_generator=generate_model_serving_agent_databricks_yaml,
        notebook_only={"07_deploy_agent.py"},
        defer_provided=False,
        label="Model serving agent",
    )


def _write_job_bundle(
    config: AppConfig,
    staging_dir: Path,
    *,
    overwrite: bool,
    development: bool,
    yaml_generator: Callable[[AppConfig, bool], str],
    notebook_only: set[str] | None,
    defer_provided: bool,
    label: str,
) -> None:
    """Stage a deployable Job (``deploy_job``) bundle into ``staging_dir``.

    Shared body for :func:`write_pipeline_bundle` (full provisioning DAG) and
    :func:`write_model_serving_agent_bundle` (single deploy-agent task). Both
    materialize the same asset kinds — ``databricks.yaml``, step notebooks, the
    resolved config, dev wheel, and config-relative data/functions/code — so
    ``databricks bundle deploy/run`` can be invoked from there with no source
    checkout. dao-ai installs via the serverless env's ``dao_ai_dep`` dependency.

    Config-referenced ``ddl``/``data`` assets are colocated with the config and
    resolved against its own directory (via ``AppConfig.from_file``), so they
    stage next to the staged config under ``config/`` and resolve identically
    when the notebook reloads the staged config.

    Args:
        config: The loaded dao-ai config (via ``AppConfig.from_file`` — the
            source path it records is what asset paths resolve against).
        staging_dir: Directory to stage the bundle into.
        overwrite: Overwrite existing staged files.
        development: When True, stage the current build's dao-ai wheel under
            ``dist/`` so the notebooks install *this* code rather than the
            published PyPI package. When no local wheel/source is available this
            raises rather than silently falling back to PyPI.
        yaml_generator: Builds the ``databricks.yaml`` text from the config +
            development flag (the pipeline vs model_serving Job shape).
        notebook_only: When set, materialize only these step notebooks;
            otherwise every wired ``NN_*.py`` notebook.
        defer_provided: Passed to :func:`_staged_config_text` — True keeps
            deferred ``provided`` params as ``${var.X}`` (pipeline fills them at
            run time); False bakes every param (model_serving has no filler).
        label: Human label for the staging summary printout.

    The staging dir is ephemeral build output: dao-ai-generated files
    (databricks.yaml, step notebooks) are (re)written every build, while
    user-owned content (the staged config, referenced data/functions assets,
    code_paths, src/<pkg>, dist wheel) is copied once and preserved on re-stage.
    """
    if config.app is None:
        raise ValueError("Config must have an 'app' section to stage a job bundle.")

    staging_dir = staging_dir.resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    # User code (src/ + code_paths + source config) left untouched.
    preserved_user_code: list[str] = []

    # Packaged/derived artifacts (databricks.yaml, requirements.txt, notebooks)
    # are ALWAYS regenerated — they are 100% derived from the installed wheel and
    # the current config, never hand-edited in the staging dir. Regenerating
    # unconditionally (ignoring `overwrite`) is what keeps a re-staged dir
    # consistent: the bundle definition must match the config it deploys, or
    # `bundle deploy` fails on a target that isn't in a stale databricks.yaml.
    # Only copied-in *content* (the config, data/functions assets) honors
    # `overwrite`.

    # 1. databricks.yaml — built programmatically (dict -> YAML). Development
    #    mode adds the `dist/*.whl` sync include so the staged wheel uploads.
    _write_file(
        staging_dir / "databricks.yaml",
        yaml_generator(config, development),
        overwrite=True,
    )
    written.append("databricks.yaml")

    # 2. Step notebooks. (No requirements.txt: the serverless environment
    #    installs dao-ai — which pulls its own transitive deps — via the
    #    ``dao_ai_dep`` dependency, mirroring the Apps deploy paths.)
    notebook_rels = _materialize_notebooks(
        staging_dir, overwrite=True, only=notebook_only
    )
    written.extend(notebook_rels)

    # 4. The resolved config, under config/<name>.yaml — a sibling of notebooks/,
    #    so the `../config/<name>.yaml` the job passes as `config-path` resolves
    #    from a notebook's working directory. Exactly one config is staged; the
    #    notebooks have no discovery fallback and read `config-path` only. Prefer
    #    the rendered YAML (${param.NAME} substituted, parameters: block
    #    stripped) so the deployed job needs no --var arguments.
    source_config: str | None = config._source_config_path
    config_filename = Path(source_config).name if source_config else "dao_ai.yaml"
    config_dir = staging_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dest = config_dir / config_filename
    config_rel = f"config/{config_filename}"
    if source_config and config_dest.resolve() == Path(source_config).resolve():
        # Never write over the user's ORIGINAL config in place (would strip
        # their parameters: block). Belt-and-suspenders behind the CLI guard.
        preserved_user_code.append(config_rel)
    elif config_dest.exists() and not overwrite:
        logger.info(f"Skipping {config_filename} (exists; use --overwrite)")
    else:
        staged_text: str | None = _staged_config_text(
            config, defer_provided=defer_provided
        )
        if staged_text is not None:
            config_dest.write_text(staged_text)
        elif source_config:
            shutil.copy2(source_config, config_dest)
        else:
            raise ValueError(
                "Config has no source path or rendered YAML; cannot stage it. "
                "Load the config via AppConfig.from_file."
            )
        written.append(config_rel)

    # 5. Development wheel: build the current source into a uniquely-versioned
    #    (``+dev<epoch>``) wheel and stage it under dist/, so the serverless
    #    environment installs THIS code. The CLI points ``dao_ai_dep`` at the
    #    staged wheel (``./dist/<name>``); in published mode ``dao_ai_dep``
    #    stays ``dao-ai`` and no wheel is staged.
    if development:
        import subprocess

        from dao_ai.utils import dev_local_version, find_dev_wheel

        project_root: Path = Path(__file__).parents[3]
        source_dir: Path = project_root / "src" / "dao_ai"
        if source_dir.is_dir():
            # Clear existing wheels so the freshly-built one is unambiguous.
            for stale in (project_root / "dist").glob("dao_ai-*.whl"):
                stale.unlink()
            # Stamp a unique local version so the dev wheel out-ranks the
            # same-base published version (see ``dev_local_version``).
            with dev_local_version(project_root / "pyproject.toml"):
                result = subprocess.run(
                    ["uv", "build", "--wheel"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                )
            if result.returncode != 0:
                raise RuntimeError(f"Wheel build failed: {result.stderr}")
            built = sorted(
                (project_root / "dist").glob("dao_ai-*.whl"),
                key=lambda p: p.stat().st_mtime,
            )
            wheel_path: Path | None = built[-1] if built else None
        else:
            # No source tree (running from an installed package) — reuse an
            # existing wheel.
            wheel_path = find_dev_wheel()
        if not wheel_path:
            raise RuntimeError(
                f"No local dao-ai wheel found for a --development {label.lower()} "
                "bundle. Build one first with `uv build --wheel`."
            )
        dist_dir = staging_dir / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        dest_wheel = dist_dir / wheel_path.name
        for stale in dist_dir.glob("dao_ai-*.whl"):
            if stale != dest_wheel:
                stale.unlink()
        shutil.copy2(wheel_path, dest_wheel)
        written.append(f"dist/{wheel_path.name}")
        logger.info(
            f"Staged dao-ai wheel for development {label.lower()} bundle",
            wheel=wheel_path.name,
        )

    # 6. Config-referenced data/functions asset files.
    copied, missing = _stage_assets(config, staging_dir, overwrite)
    written.extend(copied)

    # 6b. Custom code (app.code_paths + colocated src/) staged next to the config
    # so the deploy notebook's create_agent/deploy_agent find it. User code is
    # sacred — existing files are preserved, never overwritten.
    cp_copied, cp_preserved = _stage_code_paths(config, staging_dir)
    src_copied, src_preserved = _stage_src_packages(config, staging_dir)
    written.extend(cp_copied)
    written.extend(src_copied)
    preserved_user_code.extend(cp_preserved)
    preserved_user_code.extend(src_preserved)

    # Local skill directories, staged under ``config/skills/...`` for the same
    # reason as code_paths above: ``07_deploy_agent.py`` reloads the *staged*
    # config, so the skills have to sit beside that copy for its relative sources
    # to resolve and for ``collect_skills_code_paths`` to find content to ship to
    # Model Serving. Previously nothing staged them and a DAB deploy produced an
    # agent with no skill content at all.
    from dao_ai.skills import (
        assert_skill_assets_resolvable,
        stage_instruction_files,
        stage_skill_dirs,
    )

    assert_skill_assets_resolvable(config, target="Workflow bundle")
    sk_copied, sk_skipped, sk_preserved = stage_skill_dirs(
        config, staging_dir, overwrite=overwrite, prefix="config"
    )
    written.extend(sk_copied)
    preserved_user_code.extend(sk_skipped)
    preserved_user_code.extend(sk_preserved)

    # ``instruction_files`` under the same ``config/`` prefix: the notebook reloads
    # the staged config, so that is the directory its relative paths anchor on.
    in_copied, in_skipped, in_preserved = stage_instruction_files(
        config, staging_dir, overwrite=overwrite, prefix="config"
    )
    written.extend(in_copied)
    preserved_user_code.extend(in_skipped)
    preserved_user_code.extend(in_preserved)

    # 6c. DAB resource overlays (app.resource_paths + resources/ convention) into
    # resources/, merged by the bundle's include: [resources/*.yml] — parity with
    # the agent/mcp bundles.
    res_copied, res_preserved = _stage_resource_overlays(config, staging_dir, overwrite)
    written.extend(res_copied)
    preserved_user_code.extend(res_preserved)

    print(f"\n{label} bundle staged in {staging_dir}/\n")
    for name in sorted(written):
        print(f"  {name}")
    for name in sorted(preserved_user_code):
        print(f"  {name}  (preserved — your code, not overwritten)")
    if missing:
        print(
            "\n  WARNING: the config references asset files that were not found "
            "at stage time (their provisioning steps will fail at run time):"
        )
        for name in missing:
            print(f"    {name}")
        print(
            "  These are resolved relative to the staged notebooks/ dir. Run "
            "from a tree that contains them, point them at UC volume paths, or "
            "use --development from a full checkout."
        )
