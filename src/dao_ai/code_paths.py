"""Helpers for shipping a config's custom code (``app.code_paths``) to every
deployment target.

``config.app.code_paths`` lists extra Python files/directories a config needs at
runtime — e.g. a ``type: python`` tool whose ``function.name`` is
``my_module.my_func``, or a custom agent module. These are declared once and must
reach all deployment targets:

* **Model Serving (MLflow)** — :func:`collect_code_paths` feeds
  ``mlflow.pyfunc.log_model(code_paths=...)``, which copies each entry into
  ``<model_dir>/code/<basename>/`` and prepends ``<model_dir>/code`` to
  ``sys.path`` at load time.

* **Databricks Apps** — the deployer uploads each entry next to the config in the
  app source directory (via :func:`iter_code_path_stagings`). At runtime the app
  CWD is that directory and ``AppConfig.add_code_paths_to_sys_path`` (a config
  validator) inserts each entry's parent onto ``sys.path``, so the module imports.

* **workflow / agent bundles (dao-ai <noun> generate)** — the bundle
  generators stage each entry next to the staged config using the same
  :func:`iter_code_path_stagings` plan, so the job/app finds them.

Path contract: relative ``code_paths`` resolve against the **config file's own
directory** (identical to ``ddl``/``data`` assets and skills), with a legacy
fallback to the process CWD so pre-existing Model Serving configs keep working.
Absolute paths pass through unchanged.
"""

from __future__ import annotations

import posixpath
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from dao_ai.skills import _skill_base_dir

if TYPE_CHECKING:
    from dao_ai.config import AppConfig

# Bundle-relative dest for entries that cannot be placed next to the config
# (absolute paths, or relative paths that climb out with ``../``). Mirrors the
# ``skills/<basename>`` fallback used for skill bundling.
_CODE_FALLBACK_PREFIX = "code"

# Convention: a ``src/`` directory colocated with the config auto-ships every
# top-level package under it as custom code (no ``code_paths`` declaration
# needed). The import FQN drops the ``src`` prefix — ``src/foo/bar.py`` imports
# as ``foo.bar`` — in every target (see the module docstring / the deploy code).
_SRC_DIRNAME = "src"

# Convention: a ``resources/`` directory colocated with the config auto-ships
# every ``*.yml``/``*.yaml`` file under it as a DAB overlay (no ``resource_paths``
# declaration needed), mirroring the ``src/`` convention. Merged into the bundle
# via the generated ``databricks.yaml``'s ``include: [resources/*.yml]``.
_RESOURCES_DIRNAME = "resources"

# Directory entries whose contents dao-ai never needs to ship.
_SKIP_DIR_NAMES = {"__pycache__"}


def _code_paths_base_dir(config: "AppConfig") -> Path:
    """Directory that relative ``code_paths`` resolve against at bundle time.

    Custom code is colocated with the config (like ``ddl``/``data`` assets and
    ``skills/``), so the anchor is the config file's own directory. Reuses
    :func:`dao_ai.skills._skill_base_dir`, which falls back to the project root
    for a programmatically built config with no source path.
    """
    return _skill_base_dir(config)


def resolve_code_path(entry: str, config: "AppConfig") -> Path | None:
    """Resolve one ``code_paths`` entry to an absolute path, or ``None`` if missing.

    Absolute entries pass through verbatim. Relative entries resolve against the
    config directory (:func:`_code_paths_base_dir`); if that does not exist, a
    legacy fallback tries the process CWD (preserving pre-existing Model Serving
    deploys that ran with CWD-relative paths). Returns ``None`` when neither
    anchor locates the entry.
    """
    path = Path(entry)
    if path.is_absolute():
        return path if path.exists() else None

    config_relative = (_code_paths_base_dir(config) / path).resolve()
    if config_relative.exists():
        return config_relative

    cwd_relative = (Path.cwd() / path).resolve()
    if cwd_relative.exists():
        logger.debug(
            "Resolved code_path against CWD (legacy); prefer config-relative paths",
            entry=entry,
            resolved=str(cwd_relative),
        )
        return cwd_relative

    return None


def collect_code_paths(config: "AppConfig") -> list[str]:
    """Resolved absolute ``code_paths`` for ``mlflow.pyfunc.log_model(code_paths=...)``.

    Sibling of :func:`dao_ai.skills.collect_skills_code_paths`. Order-preserving
    and deduplicated. Raises ``FileNotFoundError`` for an entry that resolves
    nowhere, preserving the fail-loud contract the Model Serving deploy has always
    enforced.
    """
    app = config.app
    if app is None or not app.code_paths:
        return []

    resolved: list[str] = []
    seen: set[str] = set()
    for entry in app.code_paths:
        path = resolve_code_path(entry, config)
        if path is None:
            raise FileNotFoundError(f"Code path does not exist: {entry}")
        as_str = path.as_posix()
        if as_str not in seen:
            seen.add(as_str)
            resolved.append(as_str)
    return resolved


def prepend_code_paths_to_sys_path(config: "AppConfig") -> None:
    """Put resolved ``code_paths`` parents on ``sys.path`` for in-process import.

    ``AppConfig.add_code_paths_to_sys_path`` runs at config construction, before
    ``from_file`` stamps ``_source_config_path``, so it can only anchor relative
    entries at the process CWD. At deploy time (``mlflow.pyfunc.log_model``
    validation loads the model in-process) the CWD is the developer's project
    root, not the config directory, so a config-relative custom module would not
    import. Resolving against the config directory here and inserting each
    entry's parent makes the validation load succeed. Idempotent.

    Best-effort: unlike :func:`collect_code_paths` (which fails loud so a deploy
    never ships a missing module), this skips entries that don't resolve — it is
    called from ``AppConfig.from_file`` for every consumer (validate, schema,
    graph display), where a not-yet-present path must not block loading.
    """
    import sys

    app = config.app
    if app is None or not app.code_paths:
        return

    for entry in app.code_paths:
        resolved = resolve_code_path(entry, config)
        if resolved is None:
            continue
        parent = str(resolved.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
            logger.debug("Prepended resolved code_path parent to sys.path", path=parent)


def _src_dir(config: "AppConfig") -> Path:
    """The ``src/`` directory colocated with the config (may not exist)."""
    return _code_paths_base_dir(config) / _SRC_DIRNAME


def discover_src_packages(config: "AppConfig") -> list[Path]:
    """Top-level package directories under the config's colocated ``src/``.

    The ``src/`` convention auto-ships custom code with NO ``code_paths``
    declaration: any top-level directory under ``<config_dir>/src`` is a
    package (namespace packages allowed — ``__init__.py`` is not required).
    Loose files directly under ``src/`` are ignored (``src/`` holds packages,
    not modules); ``__pycache__`` and ``*.egg-info`` are skipped.

    Callers ship these so ``src/foo/bar.py`` imports as ``foo.bar`` (never
    ``src.foo``): Model Serving passes each dir to ``log_model(code_paths=...)``
    (MLflow copies it to ``code/foo/``); the wheel path lets hatch
    (``packages=["src"]``) discover it; local validation puts ``src/`` itself on
    ``sys.path`` (see :func:`prepend_src_to_sys_path`). Sorted, deduped; ``[]``
    when ``src/`` is absent or empty.
    """
    src_dir = _src_dir(config)
    if not src_dir.is_dir():
        return []

    packages: list[Path] = []
    for child in sorted(src_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in _SKIP_DIR_NAMES or child.name.endswith(".egg-info"):
            continue
        packages.append(child.resolve())
    return packages


def prepend_src_to_sys_path(config: "AppConfig") -> None:
    """Put ``<config_dir>/src`` on ``sys.path`` so ``src/`` packages import.

    Best-effort, idempotent. Makes ``import foo`` resolve during in-process
    ``log_model`` validation (and any config load), matching the Model Serving
    ``code/foo/`` layout and the Apps wheel — all three yield ``foo.bar``.
    No-op when ``src/`` is absent.
    """
    import sys

    src_dir = _src_dir(config)
    if not src_dir.is_dir():
        return
    src_path = str(src_dir.resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        logger.debug("Prepended src/ to sys.path", path=src_path)


def collect_serving_code_paths(config: "AppConfig") -> list[str]:
    """Model Serving ``code_paths``: explicit ``code_paths`` + ``src/`` packages.

    The deduped (by resolved posix path) union of :func:`collect_code_paths`
    (explicit, out-of-tree, fail-loud on missing) and :func:`discover_src_packages`
    (the colocated ``src/`` convention). Passing each ``src/`` package dir makes
    MLflow copy it to ``code/<pkg>`` → import ``<pkg>.mod``. Deduping prevents a
    double-ship when a ``code_paths`` entry points into ``src/``.
    """
    resolved: list[str] = list(collect_code_paths(config))
    seen: set[str] = set(resolved)
    for pkg in discover_src_packages(config):
        as_str = pkg.as_posix()
        if as_str not in seen:
            seen.add(as_str)
            resolved.append(as_str)
    return resolved


def iter_code_path_stagings(config: "AppConfig") -> list[tuple[Path, str]]:
    """Plan how each ``code_paths`` entry is staged/uploaded next to the config.

    Returns ``(source_abs, bundle_relative_dest)`` pairs shared by every bundle
    generator and the Apps direct-deploy upload. ``bundle_relative_dest``
    preserves the config-relative layout so the file lands next to the
    staged/uploaded ``dao_ai.yaml`` (entry ``tools/x.py`` → dest ``tools/x.py``),
    which is exactly where ``add_code_paths_to_sys_path`` looks at runtime.

    Entries that are absolute or climb out of the config directory with ``../``
    cannot sit next to the config; they fall back to ``code/<basename>`` and a
    warning is logged (their runtime resolution then depends on the deployed
    config carrying a matching relative path). Directories are returned as a
    single pair; callers walk them (see :func:`walk_code_path_files`).
    """
    app = config.app
    if app is None or not app.code_paths:
        return []

    base_dir = _code_paths_base_dir(config)
    stagings: list[tuple[Path, str]] = []
    for entry in app.code_paths:
        source = resolve_code_path(entry, config)
        if source is None:
            logger.warning(
                "code_paths entry not found; skipping staging",
                entry=entry,
            )
            continue

        dest = _bundle_relative_dest(entry, source, base_dir)
        stagings.append((source, dest))
    return stagings


def _bundle_relative_dest(entry: str, source: Path, base_dir: Path) -> str:
    """Config-relative dest for a staged entry, or a ``code/<name>`` fallback.

    A relative entry that stays within the config directory keeps its declared
    layout. An absolute entry, or a relative one that climbs out with ``../``,
    cannot sit next to the config, so it lands under ``code/<basename>``.
    """
    raw = Path(entry)
    if not raw.is_absolute():
        normalized = posixpath.normpath(raw.as_posix())
        if not normalized.startswith("../") and normalized != "..":
            return normalized

    name = source.name
    logger.warning(
        "code_paths entry is absolute or climbs outside the config directory; "
        "staging under 'code/' — prefer paths colocated with the config",
        entry=entry,
        dest=f"{_CODE_FALLBACK_PREFIX}/{name}",
    )
    return f"{_CODE_FALLBACK_PREFIX}/{name}"


def walk_code_path_files(source: Path, dest: str) -> list[tuple[Path, str]]:
    """Expand one staging pair into ``(file_abs, file_relative_dest)`` per file.

    A file staging yields itself. A directory staging yields every contained
    file (recursively), skipping ``__pycache__`` and compiled ``.pyc``, with each
    file's dest computed relative to ``dest``.
    """
    if source.is_file():
        return [(source, dest)]

    files: list[tuple[Path, str]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIR_NAMES for part in path.relative_to(source).parts):
            continue
        if path.suffix == ".pyc":
            continue
        rel = path.relative_to(source).as_posix()
        files.append((path, posixpath.join(dest, rel)))
    return files


# Generated resource files the writers own; a user overlay may not reuse these
# basenames (they'd collide in the flat ``resources/`` dir). Keep in sync with the
# writers that emit them: ``resources/app.yml`` is written by
# ``dao_ai.apps.bundle.write_bundle`` / ``dao_ai.mcp.generate.write_mcp_bundle``
# via :func:`dao_ai.apps.bundle.generate_resources_app_yaml`. Add a name here if a
# generator starts emitting another ``resources/<name>.yml``.
_RESERVED_RESOURCE_NAMES: set[str] = {"app.yml"}

# DAB resource-overlay file extensions the ``resources/`` convention auto-ships.
_RESOURCE_SUFFIXES: tuple[str, ...] = (".yml", ".yaml")


def _resources_dir(config: "AppConfig") -> Path:
    """The ``resources/`` directory colocated with the config (may not exist)."""
    return _code_paths_base_dir(config) / _RESOURCES_DIRNAME


def discover_resource_overlays(config: "AppConfig") -> list[Path]:
    """DAB overlay files under the config's colocated ``resources/`` directory.

    The ``resources/`` convention auto-ships DAB overlays with NO
    ``resource_paths`` declaration (mirroring the ``src/`` convention for code):
    any top-level ``*.yml``/``*.yaml`` file directly under ``<config_dir>/resources``
    is a bundle-resource overlay. Subdirectories, non-YAML files, and the reserved
    generated name (``app.yml``) are skipped. Sorted, deduped; ``[]`` when
    ``resources/`` is absent or empty.
    """
    resources_dir = _resources_dir(config)
    if not resources_dir.is_dir():
        return []

    overlays: list[Path] = []
    for child in sorted(resources_dir.iterdir()):
        if not child.is_file():
            continue
        if child.suffix not in _RESOURCE_SUFFIXES:
            continue
        if child.name in _RESERVED_RESOURCE_NAMES:
            continue
        overlays.append(child.resolve())
    return overlays


def iter_resource_path_stagings(config: "AppConfig") -> list[tuple[Path, str]]:
    """Plan how DAB resource overlays are staged into the bundle's ``resources/``.

    Two sources, both landing flat at ``resources/<basename>`` — DABs merges the
    generated ``databricks.yaml``'s ``include: [resources/*.yml]`` over that flat
    directory, so an overlay only needs to land there (a declared subpath is not
    preserved):

    * **explicit** ``app.resource_paths`` entries (relative to the config dir, same
      anchor as ``code_paths``; absolute paths pass through);
    * **implicit** ``*.yml``/``*.yaml`` files under the colocated ``resources/``
      directory (:func:`discover_resource_overlays`) — the ``resources/`` convention,
      mirroring ``src/`` for code.

    Fails loud (never silently drops a resource the user meant to ship):

    * an explicit entry that resolves nowhere raises ``FileNotFoundError`` —
      matching the :func:`collect_code_paths` contract for custom code;
    * two overlays whose basenames collide (both flatten to the same
      ``resources/<name>``), or one whose basename is a generated file the writer
      owns (e.g. ``app.yml``), raise ``ValueError`` rather than letting one file
      silently clobber or shadow the other.
    """
    app = config.app
    if app is None:
        return []

    # Explicit entries first (deterministic), then the implicit resources/ dir.
    # discover_resource_overlays already excludes reserved names; an explicit
    # entry that overlaps an implicit one dedups by resolved path below.
    explicit: list[tuple[str, Path]] = []
    for entry in app.resource_paths:
        source = resolve_code_path(entry, config)
        if source is None:
            raise FileNotFoundError(
                f"resource_paths entry does not exist: {entry}"
            )
        explicit.append((entry, source))
    implicit: list[tuple[str, Path]] = [
        (str(p), p) for p in discover_resource_overlays(config)
    ]

    stagings: list[tuple[Path, str]] = []
    seen_names: dict[str, str] = {}  # basename -> originating label
    seen_paths: set[str] = set()  # resolved source paths, to dedup explicit∩implicit
    for label, source in [*explicit, *implicit]:
        resolved = source.resolve()
        if resolved.as_posix() in seen_paths:
            continue
        name = source.name
        if name in _RESERVED_RESOURCE_NAMES:
            raise ValueError(
                f"resource_paths entry '{label}' has reserved basename "
                f"'{name}' — it would collide with the generated "
                f"resources/{name}. Rename the overlay file."
            )
        if name in seen_names:
            raise ValueError(
                f"resource overlays '{seen_names[name]}' and '{label}' both map "
                f"to resources/{name}; overlay basenames must be unique. Rename one."
            )
        seen_names[name] = label
        seen_paths.add(resolved.as_posix())
        stagings.append((source, posixpath.join("resources", name)))
    return stagings


def code_path_sync_globs(config: "AppConfig") -> list[str]:
    """Bundle-root-relative ``<top>/**`` globs for staged code_paths.

    Mirrors :func:`dao_ai.pipeline.bundle._asset_sync_globs`. The pipeline bundle
    stages code under ``config/<rel>`` (already covered by ``config/**``), but an
    entry that lands under the ``code/`` fallback needs its own glob so
    ``databricks bundle sync`` uploads it from a gitignored staging dir. Returns
    the extra globs (deduped); entries under ``config/`` contribute nothing.
    """
    globs: list[str] = []
    for _source, dest in iter_code_path_stagings(config):
        top = dest.split("/", 1)[0]
        glob = f"{top}/**"
        if top not in {"", "config"} and glob not in globs:
            globs.append(glob)
    return globs
