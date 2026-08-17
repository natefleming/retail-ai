"""
Skills middleware for DAO AI agents.

This module provides a factory function for creating SkillsMiddleware instances
from the Deep Agents library. SkillsMiddleware discovers and exposes reusable
agent skills from ``SKILL.md`` files, following the
`Agent Skills specification <https://agentskills.io/specification>`_.

Skills use progressive disclosure: agents see a brief listing of available
skills (name + description) in their system prompt, then read the full
``SKILL.md`` instructions on demand when a skill is needed.

Example:
    from dao_ai.middleware import create_skills_middleware

    middleware = create_skills_middleware(
        sources=["/skills/user/", "/skills/project/"],
    )

YAML Config:
    middleware:
      - name: dao_ai.middleware.skills.create_skills_middleware
        args:
          sources:
            - "/skills/user/"
            - "/skills/project/"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from dao_ai.middleware._backends import resolve_backend

if TYPE_CHECKING:
    from deepagents.middleware.skills import SkillsMiddleware

    from dao_ai.config import VolumePathModel

__all__ = [
    "create_skills_middleware",
]


def create_skills_middleware(
    sources: list[str],
    backend_type: str = "state",
    root_dir: str | None = None,
    volume_path: str | VolumePathModel | None = None,
) -> SkillsMiddleware:
    """
    Create a SkillsMiddleware for discovering and exposing agent skills.

    This factory function creates a SkillsMiddleware from the Deep Agents
    library that discovers ``SKILL.md`` files from configured sources and
    injects a skill listing into the system prompt with progressive
    disclosure.

    Each skill directory should contain a ``SKILL.md`` file with YAML
    frontmatter:

    .. code-block:: markdown

        ---
        name: web-research
        description: Structured approach to web research
        ---

        # Web Research Skill
        ## When to Use
        - User asks you to research a topic
        ...

    Skills are loaded in source order, with later sources overriding
    earlier ones when skills have the same name (last one wins). This
    enables layering: base -> user -> project skills.

    Relative sources are resolved here, at instantiation, rather than being
    stored resolved in the config: this is the first point in the lifecycle that
    runs on the machine which will actually serve the agent. A source is tried
    against the config's own directory (pushed by the graph builder), then
    ``DAO_AI_PROJECT_ROOT``, CWD, and each ``sys.path`` entry — which is what
    makes one relative path correct in dev, in Apps (bundle root is the CWD) and
    in Model Serving (mlflow prepends ``<model_dir>/code``). Absolute sources are
    used verbatim. Only the ``filesystem`` backend resolves; ``state``, ``store``
    and ``volume`` address their own namespaces.

    An unresolvable source warns and is passed through rather than raising. A
    missing skill degrades the agent; raising here would take the whole endpoint
    down instead, turning a degradation into an outage. Deploy- and bundle-time
    paths fail loudly instead, where a human is watching.

    Args:
        sources: List of paths to skill directories. Paths use POSIX
            conventions (forward slashes). Relative paths are resolved as
            described above; absolute paths are relative to the backend root.
            Later sources have higher priority.
        backend_type: Backend for file storage. One of ``"state"``
            (ephemeral, default), ``"filesystem"`` (real disk),
            ``"store"`` (persistent), or ``"volume"`` (Databricks
            Unity Catalog Volume).
        root_dir: Root directory for filesystem backend. Required when
            ``backend_type="filesystem"``.
        volume_path: Volume path for Databricks Volume backend.
            Required when ``backend_type="volume"``.

    Returns:
        A configured SkillsMiddleware instance.

    Raises:
        ValueError: If sources is empty.

    Example:
        from dao_ai.middleware import create_skills_middleware

        # Load from multiple skill sources (later overrides earlier)
        middleware = create_skills_middleware(
            sources=[
                "/skills/base/",
                "/skills/user/",
                "/skills/project/",
            ],
            backend_type="filesystem",
            root_dir="/",
        )
    """
    if not sources:
        raise ValueError("At least one source path is required for SkillsMiddleware.")

    from dao_ai._extras import require_extra

    require_extra("deepagents", feature="Skills middleware")
    from deepagents.middleware.skills import SkillsMiddleware

    backend = resolve_backend(
        backend_type=backend_type,
        root_dir=root_dir,
        volume_path=volume_path,
    )

    resolved_sources: list[str] = (
        _resolve_filesystem_sources(sources, root_dir)
        if backend_type == "filesystem"
        else list(sources)
    )

    logger.debug(
        "Creating SkillsMiddleware",
        backend_type=backend_type,
        source_count=len(resolved_sources),
        sources=resolved_sources,
    )

    middleware = SkillsMiddleware(
        backend=backend,
        sources=resolved_sources,
    )

    logger.info(
        "SkillsMiddleware created",
        backend_type=backend_type,
        source_count=len(resolved_sources),
        # The resolved value, not the declared one: when a deployed agent turns
        # out to have no skills this log line is the whole diagnosis, and its
        # absence is why the Model Serving path stayed broken silently.
        sources=resolved_sources,
    )
    return middleware


def _resolve_filesystem_sources(
    sources: list[str], root_dir: str | None = None
) -> list[str]:
    """Anchor each relative source on a real directory; pass absolutes through.

    Returns a new list — ``sources`` is the live ``MiddlewareModel.args`` value,
    and mutating it would write a machine-specific path back into the config that
    ``create_agent`` serializes into the model artifact. That is the very bug this
    resolution exists to fix, so building the graph must leave the config
    byte-identical.

    ``root_dir`` leads the anchor list when the caller declared a real one. The
    backend it configures is what resolves a relative source when this function
    passes one through, so ignoring it here meant a hand-written
    ``root_dir: /workspace/skills_root`` lost to whatever same-named directory
    happened to sit under the CWD — a *worse* answer than the pass-through it
    replaced, and the one case where resolving is a regression rather than a fix.

    ``"/"`` is excluded because it is not a declared root: it is the placeholder
    :meth:`SkillModel.as_middleware` emits for a config-relative source, since
    ``FilesystemBackend`` requires some root. Honoring it would anchor every
    source at the filesystem root, where a source of ``"."`` resolves to ``/``
    itself and a leaf named like a top-level directory quietly wins.
    """
    from dao_ai.skills import _runtime_anchors, resolve_skill_source_dir

    extra_anchors: tuple[Path, ...] = (
        (Path(root_dir),) if root_dir is not None and root_dir != "/" else ()
    )

    resolved: list[str] = []
    for source in sources:
        target = resolve_skill_source_dir(source, extra_anchors)
        if target is not None:
            resolved.append(str(target))
            continue
        resolved.append(source)
        if not os.path.isabs(source):
            logger.warning(
                "Skill source directory not found under any runtime anchor — "
                "the agent will run WITHOUT the skills under it",
                source=source,
                anchors_tried=[str(a) for a in _runtime_anchors(extra_anchors)],
                backend_type="filesystem",
            )
    return resolved
