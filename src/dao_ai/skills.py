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

**The contract: relative in the config, resolved at graph build.** A skill path
declared in YAML stays relative everywhere it is stored — in ``SkillModel.path``,
in the ``sources`` of the middleware that :meth:`SkillModel.as_middleware`
emits, and therefore in the config that ``create_agent`` serializes into the
model artifact. Nothing ever bakes the *loading* machine's absolute path into a
config, because the machine that loads a config is routinely not the machine
that runs the agent: a provisioning job resolves ``skills/x`` to a driver
directory that does not exist in the serving container, and the agent then comes
up silently skill-less.

Resolution happens once, at the moment the middleware is instantiated, against
an ordered list of anchors — the config's own directory (pushed for the duration
of the build by :func:`skill_anchors`), then ``DAO_AI_PROJECT_ROOT``, then CWD,
then each ``sys.path`` entry. Those last two are what make the same relative
path valid in Apps (bundle root is CWD) and in Model Serving (mlflow prepends
``<model_dir>/code``). Absolute sources are passed through untouched, so a
hand-written ``/Volumes/...`` or ``/abs/skills`` keeps working.

Volume-backed skills (``/Volumes/...``) are not bundled at all; the underlying
``VolumeModel`` is wired into the deployment resource block for permissions and
the path passes through to ``create_deep_agent`` verbatim.

``deep_agent.instruction_files`` (deepagents' ``memory=``, ``AGENTS.md``-style
files spliced into the system prompt) travels the same road and is handled here
for that reason: same anchors, same staging sites, same deploy-time gate. It
differs in exactly one way that matters — an entry names a **file**, not a
directory of skills — so it gets its own resolver
(:func:`resolve_instruction_file_path`) rather than reusing the skill one.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from dao_ai.config import AppConfig, DeepAgentModel, SkillModel


# Convention: dao-ai expects user skills colocated with the config under
# ``<config-dir>/skills/`` — sibling of the config's ``functions/``/``data/``.
# This is the directory shipped via mlflow ``code_paths`` and copied into the
# Apps bundle.
SKILLS_DIRNAME = "skills"


def _skill_base_dir(config: "AppConfig") -> Path:
    """Directory that relative skill paths resolve against at *bundle* time.

    Skills are colocated with the config (``skills/<skill>`` next to the YAML),
    so the anchor is the config file's own directory when the config was loaded
    via ``AppConfig.from_file``. Falls back to :func:`_project_root` when the
    config has no source path (e.g. a programmatically built config).

    ``local_config_path``, not ``source_config_path``: the latter holds the git
    locator when loaded via ``AppConfig.from_git``, and ``Path(locator).parent``
    is a directory that does not exist — so a git-sourced project's ``skills/``,
    ``code_paths``, and ``src/`` silently resolved to nothing. The former is the
    config's real path inside the checkout, against which they resolve exactly as
    a local project's do.

    Used at bundle/log_model time, NOT at inference time. Use
    :func:`_runtime_anchors` to resolve paths at inference time.
    """
    source_config: str | None = config.local_config_path
    if source_config is not None:
        return Path(source_config).resolve().parent
    return _project_root()


def _project_root() -> Path:
    """Best-effort project root for resolving relative skill paths at *bundle* time.

    Honors ``DAO_AI_PROJECT_ROOT`` if set (used by tests and CI). Otherwise
    walks upward from the current working directory looking for a ``pyproject.toml``
    or ``databricks.yaml`` marker. Falls back to CWD.

    Fallback anchor for :func:`_skill_base_dir` when a config has no source
    path. Use :func:`_runtime_anchors` to resolve paths at inference time.
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


# Anchors pushed for the duration of a graph build. The config's own directory
# is the one anchor that cannot be recovered from the ambient process — CWD and
# ``sys.path`` describe where the *process* started, not where the config came
# from — and the factory that instantiates the middleware never sees the config.
_SKILL_ANCHORS: ContextVar[tuple[Path, ...]] = ContextVar(
    "dao_ai_skill_anchors", default=()
)


@contextmanager
def skill_anchors(*dirs: Path) -> Iterator[None]:
    """Prepend ``dirs`` to the runtime anchor list for the duration of the block.

    Set once per graph build by
    :func:`dao_ai.orchestration.core.create_orchestration_graph`, so that every
    middleware instantiated underneath it can resolve a relative skill source
    against the config's own directory without the config having to be threaded
    through half a dozen factory signatures.

    Scoped rather than global, and read rather than written by the resolvers, so
    a build never leaves the process in a state that changes how the *next* build
    resolves. Contextvars propagate into asyncio tasks but **not** into
    ``ThreadPoolExecutor`` workers; that is fine here because middleware is built
    eagerly and synchronously inside the ``with`` block, well before any node
    runs.
    """
    token = _SKILL_ANCHORS.set((*dirs, *_SKILL_ANCHORS.get()))
    try:
        yield
    finally:
        _SKILL_ANCHORS.reset(token)


def config_skill_anchor(config: "AppConfig") -> tuple[Path, ...]:
    """The config's own directory, as an anchor tuple — empty when it has none.

    Deliberately empty unless ``local_config_path`` is set, rather than falling
    back to :func:`_skill_base_dir`: that helper falls back in turn to
    :func:`_project_root`, which walks up from the CWD. Anchoring a config that
    came from a URL on a local directory is exactly the boundary crossing that
    ``AppConfig._reject_relative_assets_for_remote_config`` exists to prevent —
    a remote config must not be able to name local files.
    """
    if config.local_config_path is None:
        return ()
    return (_skill_base_dir(config),)


def _runtime_anchors(extra_anchors: Sequence[Path] = ()) -> list[Path]:
    """Yield candidate anchor directories for resolving relative skill paths at runtime.

    Order, first-match-wins:

    1. ``extra_anchors``, then any anchors pushed by :func:`skill_anchors` —
       in practice the config's own directory
    2. ``DAO_AI_PROJECT_ROOT`` env var, if set
    3. ``Path.cwd()`` — covers Databricks Apps (bundle root is CWD) and dev
    4. Each existing ``sys.path`` entry — covers Model Serving (mlflow prepends
       ``<model_dir>/code`` so ``<model_dir>/code/skills/...`` resolves against it)

    The config directory leads because it is the most specific: it is where the
    author wrote the path down, and it is correct even when the process was
    started from somewhere else entirely.

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

    for anchor in (*extra_anchors, *_SKILL_ANCHORS.get()):
        _add(anchor)

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


def _resolve_runtime_path(
    rel_path: str, extra_anchors: Sequence[Path] = ()
) -> Path | None:
    """Find the first runtime anchor under which ``rel_path`` exists."""
    for anchor in _runtime_anchors(extra_anchors):
        candidate = (anchor / rel_path).resolve()
        if candidate.exists():
            return candidate
    return None


def _holds_skills(directory: Path) -> bool:
    """True if ``directory`` is a SkillsMiddleware source dir — i.e. holds skills.

    The middleware's convention is that a source is a *parent*: it lists the
    source's children and reads ``SKILL.md`` from each. So the question that
    decides whether a candidate is the right one is whether any child has a
    ``SKILL.md``, not whether the candidate itself does.
    """
    try:
        return any((child / "SKILL.md").is_file() for child in directory.iterdir())
    except OSError:
        return False


def resolve_skill_source_dir(
    source: str, extra_anchors: Sequence[Path] = ()
) -> Path | None:
    """Resolve one SkillsMiddleware ``sources`` entry to a real directory.

    Returns ``None`` when the caller should use ``source`` verbatim — either it
    is already absolute, or no anchor matched. Absolute is decided by
    :func:`os.path.isabs` and not by a leading-slash test on the string: under a
    filesystem backend rooted somewhere other than ``/``, a source like
    ``/skills/x`` is not "relative to the root" and cannot be repaired by
    stripping the slash, so it has to pass through unchanged.

    Among the anchors, prefers the first whose candidate actually holds skills
    over the first that merely exists. Without that, an empty or unrelated
    ``skills/`` under an earlier anchor — a bare CWD in a monorepo is enough —
    would shadow the real one and the agent would come up with no skills and no
    error.
    """
    if os.path.isabs(source):
        return None

    # ``as_middleware`` emits the *parent* of the skill leaf, so a leaf declared
    # with no directory component ("product-lookup") yields "." — meaning "the
    # anchor itself is the source dir".
    relative: str = source.strip()
    if relative in ("", "."):
        relative = ""

    fallback: Path | None = None
    for anchor in _runtime_anchors(extra_anchors):
        candidate: Path = (anchor / relative).resolve() if relative else anchor
        if not candidate.is_dir():
            continue
        if _holds_skills(candidate):
            return candidate
        if fallback is None:
            fallback = candidate
    return fallback


def resolve_instruction_file_path(
    source: str, extra_anchors: Sequence[Path] = ()
) -> Path | None:
    """Resolve one ``instruction_files`` entry to a real file.

    The file-shaped sibling of :func:`resolve_skill_source_dir`, sharing its
    anchor list and its "``None`` means use the source verbatim" contract.
    Separate rather than parameterized because the predicate is the whole
    function: a skills source is a directory whose *children* hold ``SKILL.md``,
    while an instruction file is a single file, and a resolver that accepted
    either would happily return a directory to deepagents'
    ``MemoryMiddleware``, which then reports ``file_not_found`` and drops it
    without a word.

    Absolute entries (including ``/Volumes/...``) return ``None`` and pass
    through, keyed on :func:`os.path.isabs` for the same reason as skills.
    """
    if os.path.isabs(source):
        return None

    relative: str = source.strip()
    if not relative:
        return None

    for anchor in _runtime_anchors(extra_anchors):
        candidate: Path = (anchor / relative).resolve()
        if candidate.is_file():
            return candidate
    return None


SKILLS_MIDDLEWARE_FACTORY = "dao_ai.middleware.skills.create_skills_middleware"


def _iter_agent_skill_sources(config: "AppConfig") -> list[str]:
    """Yield every *filesystem-backed* skill ``sources`` path in the config.

    After the AppConfig ``_translate_agent_skills_to_middleware`` validator
    runs, ``AgentModel.skills`` has been emptied and the corresponding
    SkillsMiddleware factory entries live in ``agent.middleware``. This
    helper walks those entries and pulls out the ``sources`` list for each
    so the bundle generator and code_paths collector can find them.

    Only entries whose ``backend_type`` is ``"filesystem"`` are yielded, and the
    default is ``"state"`` — matching :func:`create_skills_middleware`, which
    resolves sources against the filesystem for that backend alone and passes them
    through verbatim for ``state``, ``store``, and ``volume``. Under any of those
    three a "source" is a key into graph state, a LangGraph Store namespace, or a
    volume path, and nothing about it is expected on the deploying machine's disk.
    Yielding them anyway made every consumer here disagree with the runtime: the
    deploy gate refused a shipped example (``examples/12_middleware`` declares
    ``sources: [/skills/base/, /skills/user/]`` with the default ``state``
    backend) and blocked ``agent create``, ``agent build``, the workflow bundler,
    and the Apps deploy for a config that serves correctly.

    Delegates the traversal to :func:`dao_ai._extras._iter_middleware`, which is
    the same walker the extras check uses, so a middleware holder can only be
    forgotten in one place instead of two. Walking just ``config.app.agents`` —
    as this did — missed a skills middleware declared on a subagent, an
    orchestration pattern, or the top-level registry, and a missed holder here
    means the skill's files are never staged into the bundle.
    """
    from dao_ai._extras import _iter_middleware

    sources: list[str] = []
    for mw in _iter_middleware(config):
        if mw.name != SKILLS_MIDDLEWARE_FACTORY:
            continue
        if mw.args.get("backend_type", "state") != "filesystem":
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


def _is_databricks_fuse_path(entry: str) -> bool:
    """Whether ``entry`` names a Databricks FUSE mount instead of a local path.

    ``/Volumes``, ``/Workspace``, and ``/dbfs`` are mounted inside Databricks
    compute and absent from the laptop running the CLI, so a bundle-time existence
    check on one says nothing about whether it will be there at runtime — and
    failing the deploy over it refuses a config that would have served correctly.
    Reuses :data:`dao_ai.git_source._FUSE_ROOTS` rather than restating the
    prefixes, so the two modules cannot drift on what counts as runtime-only.

    Matching is per path segment, so ``/Volumesnotreally/x`` is still treated as a
    local path and still checked.
    """
    from dao_ai.git_source import _is_fuse_root

    return _is_fuse_root(Path(entry))


def _declared_skill_path(spec: "SkillModel | str", config: "AppConfig") -> str | None:
    """The path a skill spec declares, as written — ``None`` when volume-backed.

    Resolves a bare-string entry through ``resources.skills`` first, because a
    string is two different things depending on the config: a key into that
    registry, or an inline path. Callers that need to reason about the *declared*
    path rather than a resolved local directory (the remote-config guard, which
    must decide whether the path is relative) would otherwise have to repeat that
    lookup and would get it wrong for one of the two forms.
    """
    from dao_ai.config import SkillModel

    if isinstance(spec, str):
        named: SkillModel | None = (
            config.resources.skills.get(spec) if config.resources else None
        )
        if named is None:
            return spec
        spec = named

    if not isinstance(spec, SkillModel) or spec.is_volume_backed:
        return None
    assert isinstance(spec.path, str)
    return spec.path


def _resolve_local_skill_dir(
    spec: "SkillModel | str", config: "AppConfig"
) -> Path | None:
    """Resolve a single skill spec to an absolute local directory, if local.

    Returns ``None`` for volume-backed skills (those handle their own permissions
    and are passed through verbatim at runtime) and for any other Databricks FUSE
    mount, which is equally unavailable on the deploying machine. Used at *bundle*
    time so the project-root anchor is correct.
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
            # Raw /Volumes/... and friends are read over FUSE at runtime, not local.
            if _is_databricks_fuse_path(spec):
                return None
            return (_skill_base_dir(config) / spec).resolve()

    if spec.is_volume_backed:
        return None

    # Local SkillModel: path field is a plain string.
    assert isinstance(spec.path, str)
    if _is_databricks_fuse_path(spec.path):
        return None
    return (_skill_base_dir(config) / spec.path).resolve()


def _iter_instruction_files(config: "AppConfig") -> list[str]:
    """Every ``instruction_files`` entry declared in the config, in order.

    ``DeepAgentModel`` is the only holder: deepagents exposes ``memory=`` on the
    top-level agent only, and dao-ai has no sub-agent equivalent field. If one is
    ever added, this is the single place that has to learn about it — every
    collector, stager, and deploy gate below reads the config through here.
    """
    if config.app is None or config.app.orchestration is None:
        return []
    deep_agent: "DeepAgentModel | None" = config.app.orchestration.deep_agent
    if deep_agent is None:
        return []
    return [entry for entry in (deep_agent.instruction_files or []) if entry]


def _resolve_local_instruction_file(entry: str, config: "AppConfig") -> Path | None:
    """Resolve one ``instruction_files`` entry to an absolute local file, if local.

    Returns ``None`` for entries on a Databricks FUSE mount, which are read in
    place at runtime by the container's own identity and have nothing to stage.
    Every other entry — relative *or* absolute — comes back as a local path so
    callers can check it. Exempting all absolutes instead would let
    ``instruction_files: [/Users/me/proj/instructions/AGENTS.md]`` past the deploy
    gate untested, stage nothing, and render ``(No memory loaded)`` in the
    container: the silent failure this module exists to close. Exempting only
    ``/Volumes`` goes too far the other way and fails the deploy for a perfectly
    good ``/Workspace/Shared/AGENTS.md``, which exists on the compute that will
    read it and nowhere near the machine running the deploy.

    A non-FUSE absolute entry still cannot reproduce its declared path inside a
    serving container, so the collectors below warn and skip it; the point of
    resolving it here is that a *typo* fails the deploy instead of the deploy
    succeeding with a mute agent. Bundle-time anchor, same as
    :func:`_resolve_local_skill_dir`.
    """
    if _is_databricks_fuse_path(entry):
        return None
    return (_skill_base_dir(config) / entry).resolve()


def collect_instruction_file_code_paths(config: "AppConfig") -> list[str]:
    """Return paths for ``mlflow.pyfunc.log_model(code_paths=...)``.

    ``log_model`` flattens: a *file* entry lands at ``<model_dir>/code/<basename>``,
    losing any directory component. So an entry like ``instructions/AGENTS.md``
    cannot be shipped as itself — it would arrive as ``code/AGENTS.md`` and the
    relative path in the deployed config would resolve nowhere. Ship the entry's
    **top-level directory** instead (``<config_dir>/instructions``), exactly as
    :func:`collect_skills_code_paths` ships the whole ``skills/`` tree, so the
    declared layout survives under ``<model_dir>/code/``. An entry with no
    directory component (``AGENTS.md``) ships as the file itself, which flattens
    to the same place it started.

    Entries that are absolute, or climb out of the config directory, are skipped
    with a warning: nothing this function could ship would make the deployed
    config's path resolve.
    """
    base_dir: Path = _skill_base_dir(config)
    collected: list[str] = []
    for entry in _iter_instruction_files(config):
        local: Path | None = _resolve_local_instruction_file(entry, config)
        if local is None:
            continue
        if not local.is_file():
            logger.warning(
                "Instruction file does not exist; skipping. Either fix the path "
                "or remove it from deep_agent.instruction_files.",
                entry=entry,
                path=str(local),
            )
            continue
        try:
            rel: Path = local.relative_to(base_dir)
        except ValueError:
            logger.warning(
                "Instruction file lives outside the config directory, so Model "
                "Serving cannot reproduce its declared path — the agent will "
                "start without it. Move it next to the config.",
                entry=entry,
                path=str(local),
            )
            continue
        # ``rel.parts[0]`` is the directory mlflow must copy wholesale to keep the
        # declared layout; for a bare filename it is the file itself.
        ship: Path = base_dir / rel.parts[0]
        collected.append(str(ship))

    deduped: list[str] = list(dict.fromkeys(collected))
    if deduped:
        logger.info(
            "Including instruction file paths in mlflow code_paths",
            paths=deduped,
        )
    return deduped


def collect_skills_code_paths(config: "AppConfig") -> list[str]:
    """Return paths for ``mlflow.pyfunc.log_model(code_paths=...)``.

    Returns the config's colocated ``skills/`` directory when any local skill
    is referenced, so mlflow preserves the full ``skills/<vertical>/<skill>/``
    layout under ``<model_dir>/code/skills/...``. Returning leaf skill dirs
    instead would flatten to ``<model_dir>/code/<skill>/``, breaking the
    relative-path lookup at inference.

    No-op (returns ``[]``) when the config does not use ``deep_agent``, every
    skill is volume-backed, or the config has no colocated ``skills/`` directory.

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
        if not _is_databricks_fuse_path(source):
            has_local = True
            break

    if not has_local:
        return []

    skills_root: Path = (_skill_base_dir(config) / SKILLS_DIRNAME).resolve()
    if not skills_root.is_dir():
        logger.warning(
            "Local skills referenced but colocated ``skills/`` directory missing — "
            "Model Serving will not have skill content at inference time.",
            expected=str(skills_root),
        )
        return []

    logger.info(
        "Including colocated skills/ directory in mlflow code_paths",
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
        if _is_databricks_fuse_path(source):
            continue
        # ``as_middleware`` emits relative sources, so this is the normal branch:
        # anchor on the config's own directory, which is where the author wrote
        # the path relative to. A hand-written absolute source is used as given.
        src_path = Path(source)
        candidate = (
            src_path.resolve()
            if src_path.is_absolute()
            else (_skill_base_dir(config) / src_path).resolve()
        )
        # A source is the *parent* of the skill leaves (deepagents lists a
        # source's children and reads SKILL.md from each), while this function
        # contracts to return leaf dirs — so descend one level.
        if candidate.is_dir():
            for child in candidate.iterdir():
                if child.is_dir() and (child / "SKILL.md").exists():
                    seen.setdefault(str(child.resolve()), None)
    return list(seen.keys())


def unresolvable_skills(config: "AppConfig") -> list[str]:
    """Name every declared local skill whose directory is missing at bundle time.

    Returns one human-readable line per problem, empty when everything resolves.
    The counterpart to the warn-and-continue behavior in
    ``create_skills_middleware``: at serve time a missing skill degrades one
    agent and raising would take the endpoint down, but at deploy time there is a
    human watching and the right move is to stop before shipping an artifact
    whose skills can never load.
    """
    from dao_ai.config import SkillModel

    problems: list[str] = []
    base_dir: Path = _skill_base_dir(config)

    for spec in _iter_deep_agent_skills(config):
        local_dir: Path | None = _resolve_local_skill_dir(spec, config)
        if local_dir is None or local_dir.is_dir():
            continue
        name: str = spec.name if isinstance(spec, SkillModel) else str(spec)
        problems.append(f"skill {name!r}: no directory at {local_dir}")

    for source in _iter_agent_skill_sources(config):
        if _is_databricks_fuse_path(source):
            continue
        src_path = Path(source)
        candidate: Path = (
            src_path.resolve()
            if src_path.is_absolute()
            else (base_dir / src_path).resolve()
        )
        if candidate.is_dir():
            continue
        problems.append(f"skill source {source!r}: no directory at {candidate}")

    return problems


def unresolvable_instruction_files(config: "AppConfig") -> list[str]:
    """Name every declared instruction file that is missing at bundle time.

    Same contract and rationale as :func:`unresolvable_skills`, and the failure it
    guards against is even quieter: deepagents' ``MemoryMiddleware`` treats a
    ``file_not_found`` source as a no-op and renders ``(No memory loaded)`` into
    the prompt, so an agent deployed with a bad path behaves like one that was
    never given instructions at all.
    """
    problems: list[str] = []
    for entry in _iter_instruction_files(config):
        local: Path | None = _resolve_local_instruction_file(entry, config)
        if local is None or local.is_file():
            continue
        problems.append(f"instruction file {entry!r}: no file at {local}")
    return problems


def assert_skill_assets_resolvable(config: "AppConfig", *, target: str) -> None:
    """Raise if any declared skill or instruction file is missing.

    Called before every deploy — the single gate for both asset families, so a
    new deploy path only has to remember one call. ``target`` names the deploy
    path for the error message (e.g. ``"Model Serving"``, ``"Apps bundle"``) so
    the operator knows which command stopped.
    """
    problems: list[str] = [
        *unresolvable_skills(config),
        *unresolvable_instruction_files(config),
    ]
    if not problems:
        return
    detail: str = "\n".join(f"  - {p}" for p in problems)
    raise ValueError(
        f"{target}: {len(problems)} declared skill asset(s) cannot be found, so "
        f"the deployed agent would silently run without them:\n{detail}\n"
        "Skill and instruction-file paths are resolved relative to the config "
        "file's own directory. Fix the path, or remove the entry from the config."
    )


def iter_skill_stagings(config: "AppConfig") -> list[tuple[Path, str]]:
    """Plan how each local skill directory is staged/uploaded next to the config.

    Returns ``(source_abs, bundle_relative_dest)`` pairs, the skills counterpart
    of :func:`dao_ai.code_paths.iter_code_path_stagings` and shared by the same
    set of callers: all three bundle generators and the Apps direct-deploy
    upload.

    The relative dest (``skills/<vertical>/<skill>``) is load-bearing, not
    cosmetic: it is what lets the same relative source in the deployed config
    resolve at runtime against the bundle root. Flattening to ``skills/<skill>``
    would break every nested path. A directory outside the config's own tree
    cannot keep its declared layout, so it falls back to ``skills/<basename>``.
    """
    base_dir: Path = _skill_base_dir(config)

    stagings: list[tuple[Path, str]] = []
    for skill_dir_str in collect_local_skill_dirs(config):
        src_dir: Path = Path(skill_dir_str)
        if not src_dir.exists():
            continue
        try:
            rel: Path = src_dir.relative_to(base_dir)
        except ValueError:
            # Outside the config's tree — keep the leaf discoverable rather than
            # dropping it. The staged config still names the original path.
            rel = Path(SKILLS_DIRNAME) / src_dir.name
        stagings.append((src_dir, rel.as_posix()))
    return stagings


def stage_skill_dirs(
    config: "AppConfig",
    staging_dir: Path,
    *,
    overwrite: bool = False,
    prefix: str = "",
) -> tuple[list[str], list[str], list[str]]:
    """Copy every local skill directory into ``staging_dir``, layout preserved.

    Shared by all three bundlers. Every deploy target that reloads a *staged*
    config needs the skill content staged beside it, and for a long time only the
    Apps bundler did it — an MCP or DAB deploy shipped an agent whose config named
    skills that were nowhere in the bundle, so it came up with no skills and no
    error, exactly like the Model Serving pointer bug but for content instead of
    pointers.

    The relative layout comes from :func:`iter_skill_stagings`, shared with the
    Apps direct-deploy upload.

    Args:
        config: the config whose skills to stage.
        staging_dir: bundle root to copy into.
        overwrite: replace an existing staged copy. Off by default so a user's
            own files are never clobbered.
        prefix: subdirectory of ``staging_dir`` to stage under (e.g. ``config``
            for the DAB pipeline bundle, whose notebooks reload the staged config
            from ``config/``, making that the directory skills must sit beside).

    Returns:
        ``(written, skipped, preserved)`` — relative path strings, for the
        caller's bundle report.
    """
    written: list[str] = []
    skipped: list[str] = []
    preserved: list[str] = []

    for src_dir, rel_str in iter_skill_stagings(config):
        rel: Path = Path(rel_str)
        if prefix:
            rel = Path(prefix) / rel
        dest: Path = staging_dir / rel
        # In-place staging (output overlaps the source): never rmtree a user's
        # own skills dir.
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

    return written, skipped, preserved


def iter_instruction_file_stagings(config: "AppConfig") -> list[tuple[Path, str]]:
    """Plan how each local instruction file is staged/uploaded next to the config.

    Returns ``(source_abs, bundle_relative_dest)`` pairs in the same shape as
    :func:`iter_skill_stagings` and :func:`dao_ai.code_paths.iter_code_path_stagings`,
    so every caller stages all three asset families through one loop.

    Entries already covered by a skill staging are dropped. That is the common
    case, not an edge one: the natural place to put an ``AGENTS.md`` is inside the
    skill it belongs to (as ``examples/13_orchestration`` does), and the skill
    staging copies its directory wholesale. Without this the bundle report would
    claim to skip a file it had in fact just written.

    Entries outside the config's tree are dropped with a warning rather than
    relocated: unlike a skill directory, whose staged copy is found by anchored
    resolution, an instruction file is looked up by its declared path — a
    relocated copy would not be found, so the fallback would be a silent lie.
    """
    base_dir: Path = _skill_base_dir(config)
    skill_dests: list[str] = [dest for _src, dest in iter_skill_stagings(config)]

    stagings: list[tuple[Path, str]] = []
    for entry in _iter_instruction_files(config):
        src: Path | None = _resolve_local_instruction_file(entry, config)
        if src is None or not src.is_file():
            continue
        try:
            rel: Path = src.relative_to(base_dir)
        except ValueError:
            logger.warning(
                "Instruction file lives outside the config directory; not staged. "
                "Move it next to the config so the deployed agent can find it.",
                entry=entry,
                path=str(src),
            )
            continue
        dest: str = rel.as_posix()
        if any(dest == d or dest.startswith(f"{d}/") for d in skill_dests):
            continue
        stagings.append((src, dest))
    return stagings


def stage_instruction_files(
    config: "AppConfig",
    staging_dir: Path,
    *,
    overwrite: bool = False,
    prefix: str = "",
) -> tuple[list[str], list[str], list[str]]:
    """Copy every local instruction file into ``staging_dir``, layout preserved.

    The instruction-file counterpart of :func:`stage_skill_dirs`, with the same
    arguments, the same ``(written, skipped, preserved)`` return, and the same
    reason for existing: a target that reloads a *staged* config needs the content
    staged beside it.
    """
    written: list[str] = []
    skipped: list[str] = []
    preserved: list[str] = []

    for src, rel_str in iter_instruction_file_stagings(config):
        rel: Path = Path(rel_str)
        if prefix:
            rel = Path(prefix) / rel
        dest: Path = staging_dir / rel
        if dest.resolve() == src.resolve():
            preserved.append(str(rel))
            continue
        if dest.exists() and not overwrite:
            logger.info(
                "Skipping instruction file copy (exists; use --overwrite)",
                instruction_file=str(rel),
            )
            skipped.append(str(rel))
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        logger.info("Copied instruction file into bundle", instruction_file=str(rel))
        written.append(str(rel))

    return written, skipped, preserved


def resolve_instruction_file_runtime_paths(
    instruction_files: list[str], config: "AppConfig"
) -> list[str]:
    """Resolve ``instruction_files`` to real paths for ``create_deep_agent(memory=)``.

    The instruction-file twin of :func:`resolve_skill_runtime_paths`, and it
    exists for the same reason: the raw YAML string is relative to the *config*,
    while deepagents resolves a relative source against the backend's ``cwd`` —
    ``Path.cwd()`` for a ``FilesystemBackend`` with no ``root_dir``. Those agree
    only when the process happens to have been started from the config directory,
    which a notebook, a ``from_git`` checkout, and ``dao-ai run -c <path>`` all
    fail to do.

    Absolute entries pass through verbatim. A relative entry that no anchor
    locates is dropped with a warning naming the anchors tried — deepagents would
    otherwise drop it silently and render ``(No memory loaded)``.
    """
    anchor: tuple[Path, ...] = config_skill_anchor(config)
    resolved: list[str] = []
    for entry in instruction_files or []:
        if os.path.isabs(entry):
            resolved.append(entry)
            continue
        found: Path | None = resolve_instruction_file_path(entry, anchor)
        if found is None:
            logger.warning(
                "Instruction file not found under any runtime anchor; the agent "
                "will start WITHOUT it",
                instruction_file=entry,
                anchors_tried=[str(a) for a in _runtime_anchors(anchor)],
            )
            continue
        resolved.append(str(found))

    return list(dict.fromkeys(resolved))


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

        # Local SkillModel: resolve the leaf, then return its parent. The config's
        # own directory is passed explicitly rather than relied upon through the
        # ambient anchor stack, so this stays correct when called directly (the
        # deep_agent graph builder is not the only caller).
        assert isinstance(spec.path, str)
        anchor: tuple[Path, ...] = config_skill_anchor(config)
        runtime_dir: Path | None = _resolve_runtime_path(spec.path, anchor)
        if runtime_dir is None:
            logger.warning(
                "Skill directory not found under any runtime anchor; skipping",
                name=spec.name,
                rel_path=spec.path,
                anchors_tried=[str(a) for a in _runtime_anchors(anchor)],
            )
            continue
        logger.info(
            "Resolved skill directory",
            name=spec.name,
            rel_path=spec.path,
            resolved=str(runtime_dir),
        )
        seen.setdefault(str(runtime_dir.parent), None)

    return list(seen.keys())
