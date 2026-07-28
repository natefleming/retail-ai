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
from importlib.resources import files
from pathlib import Path
from typing import Any

from loguru import logger

from dao_ai.apps.bundle import (
    _retain_only_parameters,
    _sha256_file,
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
    ("ingest-and-transform", "01_ingest_and_transform.py", (), {}),
    (
        "provision-vector-search",
        "02_provision_vector_search.py",
        ("ingest-and-transform",),
        {},
    ),
    ("provision-lakebase", "03_provision_lakebase.py", (), {}),
    (
        "unity-catalog-tools",
        "04_unity_catalog_tools.py",
        ("provision-vector-search", "provision-lakebase"),
        {},
    ),
    ("provision-genie", "05_provision_genie.py", ("unity-catalog-tools",), {}),
    (
        "deploy-agents",
        "06_deploy_agent.py",
        ("provision-genie",),
        {"mode": "${var.mode}", "development": "${var.development}"},
    ),
    (
        "generate-evaluation-data",
        "07_generate_evaluation_data.py",
        ("provision-vector-search",),
        {},
    ),
    (
        "run-evaluation",
        "08_run_evaluation.py",
        ("deploy-agents", "generate-evaluation-data"),
        {},
    ),
)


def generate_pipeline_databricks_yaml(config: AppConfig, development: bool) -> str:
    """Build the ``deploy_job`` bundle ``databricks.yaml`` (dict -> YAML).

    Programmatic, matching :func:`dao_ai.apps.bundle.generate_databricks_yaml`
    (both serialize via :func:`dao_ai.apps.bundle.dump_bundle_yaml`). Unlike the
    App generators this emits a Lakeflow **Job** — the 8-task provisioning DAG —
    with a per-cloud target (``<app>-<cloud>``) and the four bundle variables the
    step notebooks read.

    There is no ``artifacts:`` block: the dao-ai wheel is not built at
    ``bundle deploy`` time. In development mode a pre-built wheel is staged into
    ``dist/`` (added to ``sync.include`` so the CLI uploads it as a source file);
    otherwise the notebooks install ``dao-ai`` from PyPI.
    """
    if config.app is None:
        raise ValueError("Config must have an 'app' section to build the bundle.")
    app_name = normalize_name(config.app.name)

    sync_include = [
        "config/**",
        "notebooks/*.py",
    ]
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
            resolve_required_extras_or_all(config, target="pipeline")
        )
        extra_dep_pins = get_installed_packages(required_extras)

    tasks: list[dict[str, Any]] = []
    for task_key, notebook, depends_on, extra_params in _PIPELINE_TASKS:
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
                "description": "Agent serving mode (model_serving, apps, mcp).",
                "default": "apps",
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

    return dump_bundle_yaml(bundle)


def _materialize_notebooks(staging_dir: Path, overwrite: bool) -> list[str]:
    """Copy the packaged step notebooks into ``<staging_dir>/notebooks/``.

    Only the wired ``NN_*.py`` step notebooks are materialized; the package
    ``__init__.py`` marker is skipped.
    """
    notebooks_dir = staging_dir / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for entry in files(_NOTEBOOKS_PKG).iterdir():
        if entry.name == "__init__.py" or not entry.name.endswith(".py"):
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
    ``06_deploy_agent.py`` reloads the staged config, ``add_code_paths_to_sys_path``
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
    the staged config), so when ``06_deploy_agent.py`` reloads the staged config
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
    via taskValues keyed by the param name, which ``06_deploy_agent`` then injects
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


def _staged_config_text(config: AppConfig) -> str | None:
    """The dao-ai config text to stage for a workflow bundle.

    Default: the fully-substituted rendered YAML with the ``parameters:`` block
    stripped (deployed job needs no ``--var``). When the config has
    ``provided: true`` params to defer (see :func:`_deferred_provided_params`),
    re-render the pre-substitution source leaving those refs in place and retain
    ONLY their declarations, so ``05_provision_genie`` can provision and
    ``06_deploy_agent`` can inject the forwarded id. Returns None if the config
    was not loaded via ``AppConfig.from_file`` (no rendered text available).
    """
    rendered: str | None = config._rendered_yaml
    if rendered is None:
        return None

    defer: set[str] = _deferred_provided_params(config)
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
) -> dict[str, str]:
    """Stage a deployable ``deploy_job`` bundle for ``dao-ai generate-workflow``.

    Materializes the packaged pipeline assets (``databricks.yaml`` template,
    step notebooks) plus the resolved dao-ai config into ``staging_dir``, so
    ``databricks bundle deploy/run`` can be invoked from there with no source
    checkout. dao-ai installs via the serverless env's ``dao_ai_dep`` dependency.

    Config-referenced ``ddl``/``data`` assets are colocated with the config and
    resolved against its own directory (via ``AppConfig.from_file``), so they
    stage next to the staged config under ``config/`` and resolve identically
    when the provisioning notebook reloads the staged config.

    Args:
        config: The loaded dao-ai config (via ``AppConfig.from_file`` — the
            source path it records is what asset paths resolve against).
        staging_dir: Directory to stage the bundle into.
        overwrite: Overwrite existing staged files.
        development: When True, stage the current build's dao-ai wheel under
            ``dist/`` so the step notebooks install *this* code rather than the
            published PyPI package. When no local wheel/source is available this
            raises rather than silently falling back to PyPI.

    Returns the staging registry: ``{relative_posix_path: sha256}`` for the files
    dao-ai *generated* (databricks.yaml, step notebooks). The staged config,
    referenced assets, code_paths, src/<pkg>, and dist wheel are excluded so
    hand-edits to them never trip edit-detection. The caller stamps this into
    ``.dao-ai-manifest.yaml``.
    """
    if config.app is None:
        raise ValueError(
            "Config must have an 'app' section to stage a pipeline bundle."
        )

    staging_dir = staging_dir.resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    # User code (src/ + code_paths + source config) left untouched.
    preserved_user_code: list[str] = []
    # Content hashes of the files dao-ai generated (databricks.yaml + notebooks),
    # keyed by staging-dir-relative POSIX path. The staged config and assets are
    # user-editable, so they're excluded.
    registry: dict[str, str] = {}

    def _register(rel: str) -> None:
        path = staging_dir / rel
        if path.exists():
            registry[Path(rel).as_posix()] = _sha256_file(path)

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
        generate_pipeline_databricks_yaml(config, development=development),
        overwrite=True,
    )
    written.append("databricks.yaml")
    _register("databricks.yaml")

    # 2. Step notebooks. (No requirements.txt: the serverless environment
    #    installs dao-ai — which pulls its own transitive deps — via the
    #    ``dao_ai_dep`` dependency, mirroring the Apps deploy paths.)
    notebook_rels = _materialize_notebooks(staging_dir, overwrite=True)
    written.extend(notebook_rels)
    for rel in notebook_rels:
        _register(rel)

    # 4. The resolved config, under config/<name>.yaml so the notebook's
    #    `../config` discovery and an explicit config-path both resolve. Prefer
    #    the rendered YAML (${param.NAME} substituted, parameters: block
    #    stripped) so the deployed job needs no --var arguments.
    source_config: str | None = config._source_config_path
    config_filename = Path(source_config).name if source_config else "dao_ai.yaml"
    config_dir = staging_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dest = config_dir / config_filename
    if source_config and config_dest.resolve() == Path(source_config).resolve():
        # Never write over the user's ORIGINAL config in place (would strip
        # their parameters: block). Belt-and-suspenders behind the CLI guard.
        preserved_user_code.append(f"config/{config_filename}")
    elif config_dest.exists() and not overwrite:
        logger.info(f"Skipping {config_filename} (exists; use --overwrite)")
    else:
        staged_text: str | None = _staged_config_text(config)
        if staged_text is not None:
            config_dest.write_text(staged_text)
        elif source_config:
            shutil.copy2(source_config, config_dest)
        else:
            raise ValueError(
                "Config has no source path or rendered YAML; cannot stage it. "
                "Load the config via AppConfig.from_file."
            )
        written.append(f"config/{config_filename}")

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
                "No local dao-ai wheel found for a --development pipeline "
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
            "Staged dao-ai wheel for development pipeline",
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

    print(f"\nPipeline bundle staged in {staging_dir}/\n")
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

    return registry
