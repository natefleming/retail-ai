"""Staging-bundle generation for the ``dao-ai generate-workflow`` Lakeflow job.

Where :mod:`dao_ai.apps.bundle` emits a Databricks *App* bundle, this module
emits the multi-task *Job* (``deploy_job``) bundle that ``dao-ai generate-workflow``
submits. The whole point is to run from an installed ``dao-ai`` wheel with **no
source checkout**: the ``databricks.yaml`` is built programmatically here
(matching ``generate_databricks_yaml`` in :mod:`dao_ai.apps.bundle`), and the
step notebooks ship as package data under ``dao_ai/pipeline/`` and are
materialized into a staging directory.

The staging layout mirrors the historic repo layout so the relative paths baked
into notebooks (``../config``, ``../dist``) and into configs (``../data/...``,
``../functions/...``) resolve unchanged — no path rewriting::

    <output_dir>/
      databricks.yaml          # built programmatically (dict -> yaml.dump)
      notebooks/NN_*.py        # packaged step notebooks
      config/<name>.yaml       # the resolved dao-ai config
      dist/dao_ai-*.whl        # development mode only: the current build's wheel
      data/<vertical>/...      # copied referenced dataset ddl/data files
      functions/<vertical>/... # copied referenced UC-function ddl files

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
    _strip_parameters_block,
    _write_file,
    dump_bundle_yaml,
)
from dao_ai.config import AppConfig
from dao_ai.utils import normalize_name

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
    ("provision-vector-search", "02_provision_vector_search.py",
     ("ingest-and-transform",), {}),
    ("provision-lakebase", "03_provision_lakebase.py", (), {}),
    ("unity-catalog-tools", "04_unity_catalog_tools.py",
     ("provision-vector-search", "provision-lakebase"), {}),
    ("provision-genie", "05_provision_genie.py", ("unity-catalog-tools",), {}),
    ("deploy-agents", "06_deploy_agent.py", ("provision-genie",),
     {"deployment-target": "${var.deployment_target}",
      "development": "${var.development}"}),
    ("generate-evaluation-data", "07_generate_evaluation_data.py",
     ("provision-vector-search",), {}),
    ("run-evaluation", "08_run_evaluation.py",
     ("deploy-agents", "generate-evaluation-data"), {}),
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
        "data/**",
        "functions/**",
        "functions/**/*.sql",
        "notebooks/*.py",
    ]
    if development:
        # The bundle CLI excludes .whl by default; include the staged wheel so
        # the serverless environment can install it (../dist/<wheel> via the
        # ``dao_ai_dep`` variable).
        sync_include.append("dist/*.whl")

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
            "deployment_target": {
                "description": "Agent deployment target (model_serving, apps, both).",
                "default": "model_serving",
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
                    "'dao-ai' PyPI spec otherwise. Installed by the job's "
                    "serverless environment before each task's notebook runs."
                ),
                "default": "dao-ai",
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
                                "dependencies": [
                                    "${var.dao_ai_dep}"
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


def _materialize_notebooks(output_dir: Path, overwrite: bool) -> list[str]:
    """Copy the packaged step notebooks into ``<output_dir>/notebooks/``.

    Only the wired ``NN_*.py`` step notebooks are materialized; the package
    ``__init__.py`` marker is skipped.
    """
    notebooks_dir = output_dir / "notebooks"
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

    Notebooks run from the staged ``notebooks/`` dir, so config paths are
    written relative to it (``../data/...``). Returned paths are kept verbatim
    so the caller can resolve + stage them preserving their layout.
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


def _stage_assets(
    config: AppConfig,
    output_dir: Path,
    source_root: Path,
    overwrite: bool,
) -> tuple[list[str], list[str]]:
    """Copy config-referenced data/functions files into the staging bundle.

    Config asset paths are relative to the notebook CWD, which historically is
    ``<root>/notebooks`` — e.g. ``../data/hardware_store/products.snappy.parquet``
    means ``<root>/data/hardware_store/products.snappy.parquet``. We resolve the
    *source* against ``source_root/notebooks`` (the user's tree) and copy each
    file to the same relative location under the staged bundle, so it resolves
    identically when the notebook runs in-workspace (CWD == staged
    ``notebooks/``).

    Returns ``(copied, missing)`` where ``missing`` lists referenced paths not
    found at stage time (reported to the user, never silently dropped — a
    missing seed file means a provisioning step will fail at run time).
    """
    src_anchor = (source_root / "notebooks").resolve()
    dest_anchor = (output_dir / "notebooks").resolve()
    copied: list[str] = []
    missing: list[str] = []

    def _staged_label(dest: Path) -> str:
        """Bundle-relative label for the summary (e.g. ``data/foo/bar.sql``)."""
        try:
            return str(dest.relative_to(output_dir))
        except ValueError:
            return str(dest)

    for rel in _referenced_asset_paths(config):
        src = (src_anchor / rel).resolve()
        dest = (dest_anchor / rel).resolve()
        if not src.exists():
            missing.append(rel)
            continue
        if src == dest:
            # Staging in place (output_dir is the source tree) — nothing to copy.
            continue
        if dest.exists() and not overwrite:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(_staged_label(dest))

    return copied, missing


def write_pipeline_bundle(
    config: AppConfig,
    output_dir: Path,
    overwrite: bool = False,
    development: bool = False,
    source_root: Path | None = None,
) -> None:
    """Stage a deployable ``deploy_job`` bundle for ``dao-ai generate-workflow``.

    Materializes the packaged pipeline assets (``databricks.yaml`` template,
    step notebooks) plus the resolved dao-ai config into ``output_dir``, so
    ``databricks bundle deploy/run`` can be invoked from there with no source
    checkout. dao-ai installs via the serverless env's ``dao_ai_dep`` dependency.

    Args:
        config: The loaded dao-ai config (via ``AppConfig.from_file``).
        output_dir: Directory to stage the bundle into.
        overwrite: Overwrite existing staged files.
        development: When True, stage the current build's dao-ai wheel under
            ``dist/`` so the step notebooks install *this* code rather than the
            published PyPI package. When no local wheel/source is available this
            raises rather than silently falling back to PyPI.
        source_root: Tree the config's relative asset paths (``../data/...``,
            ``../functions/...``) are resolved against when staging. Defaults to
            the current working directory — the historic invocation dir where a
            ``notebooks/`` sibling of ``data/``/``functions/`` lives.
    """
    if config.app is None:
        raise ValueError(
            "Config must have an 'app' section to stage a pipeline bundle."
        )

    output_dir = output_dir.resolve()
    source_root = (source_root or Path.cwd()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

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
        output_dir / "databricks.yaml",
        generate_pipeline_databricks_yaml(config, development=development),
        overwrite=True,
    )
    written.append("databricks.yaml")

    # 2. Step notebooks. (No requirements.txt: the serverless environment
    #    installs dao-ai — which pulls its own transitive deps — via the
    #    ``dao_ai_dep`` dependency, mirroring the Apps deploy paths.)
    written.extend(_materialize_notebooks(output_dir, overwrite=True))

    # 4. The resolved config, under config/<name>.yaml so the notebook's
    #    `../config` discovery and an explicit config-path both resolve. Prefer
    #    the rendered YAML (${param.NAME} substituted, parameters: block
    #    stripped) so the deployed job needs no --var arguments.
    source_config: str | None = getattr(config, "_source_config_path", None)
    config_filename = Path(source_config).name if source_config else "dao_ai.yaml"
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dest = config_dir / config_filename
    if config_dest.exists() and not overwrite:
        logger.info(f"Skipping {config_filename} (exists; use --overwrite)")
    else:
        rendered: str | None = getattr(config, "_rendered_yaml", None)
        if rendered is not None:
            config_dest.write_text(_strip_parameters_block(rendered))
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
        dist_dir = output_dir / "dist"
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
    copied, missing = _stage_assets(config, output_dir, source_root, overwrite)
    written.extend(copied)

    print(f"\nPipeline bundle staged in {output_dir}/\n")
    for name in sorted(written):
        print(f"  {name}")
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
