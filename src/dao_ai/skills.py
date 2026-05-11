"""Helpers for shipping deepagents skills with the model artifact.

Skills are directories of Markdown content (``SKILL.md``, optionally ``AGENTS.md``)
that teach a deep_agent how to do a task. They can live locally (relative to the
project root) or in a Unity Catalog volume.

Two deployment targets need to find skill files at runtime:

* **Databricks Apps** — ``write_bundle()`` copies skill directories into the
  bundle output directory under ``skills/<vertical>/<skill>/``. At runtime the
  bundle root is the app's CWD, so the relative path declared in YAML resolves
  directly via ``Path.cwd() / "skills/<vertical>/<skill>"``.

* **Model Serving (MLflow)** — ``mlflow.pyfunc.log_model`` copies entries from
  ``code_paths`` into ``<model_dir>/code/<basename>/`` and prepends
  ``<model_dir>/code`` to ``sys.path``. To preserve the full
  ``skills/<vertical>/<skill>/`` layout, this module ships the *project-root*
  ``skills/`` directory (not individual leaf dirs). At inference time the layout
  appears at ``<model_dir>/code/skills/<vertical>/<skill>/`` and the relative
  YAML path resolves against the entry that the MLflow runtime adds to
  ``sys.path``.

The runtime resolver tries multiple anchors in order — env var, CWD, then
``sys.path`` entries — and uses the first one where the relative path exists.
This keeps the same YAML config valid in dev, in Apps, and in Model Serving.

Volume-backed skills (``/Volumes/...``) are not bundled at all; the underlying
``VolumeModel`` is wired into the deployment resource block for permissions and
the path passes through to ``create_deep_agent`` verbatim.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from dao_ai.config import AppConfig, DeepAgentModel, SkillModel


# Convention: dao-ai expects user skills under ``<project_root>/skills/`` —
# sibling of ``functions/`` and per-vertical subdirectory layout. This is the
# directory shipped via mlflow ``code_paths`` and copied into the Apps bundle.
SKILLS_DIRNAME = "skills"


def _project_root() -> Path:
    """Best-effort project root for resolving relative skill paths at *bundle* time.

    Honors ``DAO_AI_PROJECT_ROOT`` if set (used by tests and CI). Otherwise
    walks upward from the current working directory looking for a ``pyproject.toml``
    or ``databricks.yaml`` marker. Falls back to CWD.

    Used at bundle/log_model time, NOT at inference time. Use
    :func:`_runtime_anchors` to resolve paths at inference time.
    """
    env_root: str | None = os.environ.get("DAO_AI_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    cwd: Path = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").exists() or (
            candidate / "databricks.yaml"
        ).exists():
            return candidate
    return cwd


def _runtime_anchors() -> list[Path]:
    """Yield candidate anchor directories for resolving relative skill paths at runtime.

    Order, first-match-wins:

    1. ``DAO_AI_PROJECT_ROOT`` env var, if set
    2. ``Path.cwd()`` — covers Databricks Apps (bundle root is CWD) and dev
    3. Each existing ``sys.path`` entry — covers Model Serving (mlflow prepends
       ``<model_dir>/code`` so ``<model_dir>/code/skills/...`` resolves against it)

    Anchors are deduplicated while preserving order.
    """
    anchors: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        try:
            resolved = p.resolve()
        except OSError:
            return
        key = str(resolved)
        if key in seen:
            return
        if not resolved.is_dir():
            return
        seen.add(key)
        anchors.append(resolved)

    env_root: str | None = os.environ.get("DAO_AI_PROJECT_ROOT")
    if env_root:
        _add(Path(env_root))

    _add(Path.cwd())

    for entry in sys.path:
        if not entry:
            continue
        try:
            _add(Path(entry))
        except (TypeError, ValueError):
            continue

    return anchors


def _resolve_runtime_path(rel_path: str) -> Path | None:
    """Find the first runtime anchor under which ``rel_path`` exists."""
    for anchor in _runtime_anchors():
        candidate = (anchor / rel_path).resolve()
        if candidate.exists():
            return candidate
    return None


SKILLS_MIDDLEWARE_FACTORY = "dao_ai.middleware.skills.create_skills_middleware"


def _iter_agent_skill_sources(config: "AppConfig") -> list[str]:
    """Yield every skill ``sources`` path declared on any agent's middleware.

    After the AppConfig ``_translate_agent_skills_to_middleware`` validator
    runs, ``AgentModel.skills`` has been emptied and the corresponding
    SkillsMiddleware factory entries live in ``agent.middleware``. This
    helper walks those entries and pulls out the ``sources`` list for each
    so the bundle generator and code_paths collector can find them.
    """
    if config.app is None or not config.app.agents:
        return []
    sources: list[str] = []
    for agent in config.app.agents:
        for mw in agent.middleware:
            if mw.name != SKILLS_MIDDLEWARE_FACTORY:
                continue
            raw_sources = mw.args.get("sources") or []
            for s in raw_sources:
                if isinstance(s, str):
                    sources.append(s)
    return sources


def _iter_deep_agent_skills(config: "AppConfig") -> list["SkillModel | str"]:
    """Yield every skill spec referenced anywhere in the config's deep_agent block.

    Includes the top-level ``orchestration.deep_agent.skills`` and any
    sub-agent ``skills``. Returns the raw spec entries (``SkillModel`` or string)
    in declaration order. Empty when no deep_agent is configured.

    NOTE: Agent-level skills (``AgentModel.skills``) are translated to
    middleware at AppConfig load time, so they're picked up via
    :func:`_iter_agent_skill_sources` separately rather than here.
    """
    from dao_ai.config import SubAgentModel

    if config.app is None or config.app.orchestration is None:
        return []

    deep_agent: "DeepAgentModel | None" = config.app.orchestration.deep_agent
    if deep_agent is None:
        return []

    collected: list["SkillModel | str"] = list(deep_agent.skills or [])
    for sub in deep_agent.subagents or []:
        if isinstance(sub, SubAgentModel):
            collected.extend(sub.skills or [])
    return collected


def _resolve_local_skill_dir(
    spec: "SkillModel | str", config: "AppConfig"
) -> Path | None:
    """Resolve a single skill spec to an absolute local directory, if local.

    Returns ``None`` for volume-backed skills (those handle their own permissions
    and are passed through verbatim at runtime). Used at *bundle* time so the
    project-root anchor is correct.
    """
    if isinstance(spec, str):
        # Inline path string. Look it up in resources.skills first; otherwise
        # treat as a local relative path.
        named: SkillModel | None = (
            config.resources.skills.get(spec) if config.resources else None
        )
        if named is not None:
            spec = named
        else:
            # Raw /Volumes/... strings are volume-backed, not local.
            if spec.startswith("/Volumes/"):
                return None
            return (_project_root() / spec).resolve()

    if spec.is_volume_backed:
        return None

    # Local SkillModel: path field is a plain string.
    assert isinstance(spec.path, str)
    return (_project_root() / spec.path).resolve()


def collect_skills_code_paths(config: "AppConfig") -> list[str]:
    """Return paths for ``mlflow.pyfunc.log_model(code_paths=...)``.

    Returns the *project-root* ``skills/`` directory when any local skill is
    referenced, so mlflow preserves the full ``skills/<vertical>/<skill>/``
    layout under ``<model_dir>/code/skills/...``. Returning leaf skill dirs
    instead would flatten to ``<model_dir>/code/<skill>/``, breaking the
    relative-path lookup at inference.

    No-op (returns ``[]``) when the config does not use ``deep_agent``, every
    skill is volume-backed, or the project has no ``skills/`` directory.

    Volume-backed skills are excluded — they're read directly from
    ``/Volumes/...`` at runtime and have their permissions wired through the
    underlying ``VolumeModel.as_resources()``.
    """
    has_local: bool = False
    for spec in _iter_deep_agent_skills(config):
        local_dir: Path | None = _resolve_local_skill_dir(spec, config)
        if local_dir is None:
            continue
        if not local_dir.exists():
            logger.warning(
                "Skill directory does not exist; skipping bundling. "
                "Either fix the path or remove the skill from the config.",
                path=str(local_dir),
            )
            continue
        has_local = True

    # Agent-level skills (translated into middleware) — any non-volume source
    # counts as "we have at least one local skill" for the mlflow code_paths
    # inclusion decision.
    for source in _iter_agent_skill_sources(config):
        if not source.startswith("/Volumes/"):
            has_local = True
            break

    if not has_local:
        return []

    skills_root: Path = (_project_root() / SKILLS_DIRNAME).resolve()
    if not skills_root.is_dir():
        logger.warning(
            "Local skills referenced but project ``skills/`` directory missing — "
            "Model Serving will not have skill content at inference time.",
            expected=str(skills_root),
        )
        return []

    logger.info(
        "Including project skills/ directory in mlflow code_paths",
        path=str(skills_root),
    )
    return [str(skills_root)]


def collect_local_skill_dirs(config: "AppConfig") -> list[str]:
    """Return absolute paths of every local skill directory referenced by the config.

    Used by the Apps bundle generator (``apps/bundle.py::write_bundle``) to copy
    skill directories into the bundle output dir. Distinct from
    :func:`collect_skills_code_paths` — that function returns the *parent*
    ``skills/`` dir for mlflow, while this one returns each leaf dir for
    surgical bundle copies.

    Walks two sources:

    * ``deep_agent.skills`` and ``subagents[].skills`` (the orchestration block).
    * ``app.agents[].middleware`` entries that point at the SkillsMiddleware
      factory — these are the residue of ``AgentModel.skills`` after the
      AppConfig load-time translator runs.

    Volume-backed skills are excluded.
    """
    seen: dict[str, None] = {}

    # 1. orchestration-level skills
    for spec in _iter_deep_agent_skills(config):
        local_dir: Path | None = _resolve_local_skill_dir(spec, config)
        if local_dir is None:
            continue
        if not local_dir.exists():
            continue
        seen.setdefault(str(local_dir), None)

    # 2. agent-level skills (translated into middleware ``sources``)
    for source in _iter_agent_skill_sources(config):
        if source.startswith("/Volumes/"):
            continue
        candidate = Path(source).resolve()
        # If as_middleware() resolved to an absolute parent dir (e.g.
        # /project/skills), the parent IS the source. The bundle copy
        # wants every leaf skill subdir under it — but `collect_local_skill_dirs`
        # historically returns leaf dirs. To keep that contract we yield
        # each subdir of the source that contains a SKILL.md.
        if candidate.is_dir():
            for child in candidate.iterdir():
                if child.is_dir() and (child / "SKILL.md").exists():
                    seen.setdefault(str(child.resolve()), None)
    return list(seen.keys())


def resolve_skill_runtime_paths(
    skills: list["SkillModel | str"], config: "AppConfig"
) -> list[str]:
    """Resolve a list of skill specs to *parent* directories for ``create_deep_agent``.

    deepagents' ``SkillsMiddleware`` treats each entry in ``skills=[...]`` as a
    "source directory" — it lists the source's children and looks for a
    ``SKILL.md`` inside each subdirectory. So if the user declares a SkillModel
    that points at the leaf skill folder (e.g. ``skills/sg/research`` containing
    ``SKILL.md``), we pass the **parent** (``skills/sg``) so the middleware can
    discover ``research/SKILL.md`` (and any sibling skills).

    Behavior:

    * Volume-backed skills (``/Volumes/...``) → take the parent of the
      ``VolumePathModel.full_name`` (e.g. ``/Volumes/cat/schema/vol/research``
      becomes ``/Volumes/cat/schema/vol``).
    * Local skills → resolve the leaf against the runtime anchors (env var, CWD,
      ``sys.path``) and return the parent.
    * Multiple SkillModel entries that share a parent dedupe to one entry —
      deepagents discovers all skills under the source on its own.
    * String entries are first looked up in ``resources.skills``; otherwise
      treated as inline paths.

    Returns parent path strings in declaration order, deduplicated. Missing
    local directories are logged and skipped.
    """
    from dao_ai.config import SkillModel

    seen: dict[str, None] = {}
    for spec in skills or []:
        if isinstance(spec, str):
            named: SkillModel | None = (
                config.resources.skills.get(spec) if config.resources else None
            )
            spec = named if named is not None else SkillModel(name=spec, path=spec)

        if spec.is_volume_backed:
            # Pass the parent of the volume path so deepagents discovers
            # the skill subdirs inside the volume root.
            full = spec.runtime_path.rstrip("/")
            parent = full.rsplit("/", 1)[0] or "/"
            seen.setdefault(parent, None)
            continue

        # Local SkillModel: resolve the leaf, then return its parent.
        assert isinstance(spec.path, str)
        runtime_dir: Path | None = _resolve_runtime_path(spec.path)
        if runtime_dir is None:
            logger.warning(
                "Skill directory not found under any runtime anchor; skipping",
                name=spec.name,
                rel_path=spec.path,
                anchors_tried=[str(a) for a in _runtime_anchors()],
            )
            continue
        seen.setdefault(str(runtime_dir.parent), None)

    return list(seen.keys())
