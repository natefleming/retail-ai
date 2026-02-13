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

from typing import TYPE_CHECKING

from deepagents.middleware.skills import SkillsMiddleware
from loguru import logger

from dao_ai.middleware._backends import resolve_backend

if TYPE_CHECKING:
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

    Args:
        sources: List of paths to skill directories. Paths use POSIX
            conventions (forward slashes) and are relative to the backend
            root. Later sources have higher priority.
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

    backend = resolve_backend(
        backend_type=backend_type,
        root_dir=root_dir,
        volume_path=volume_path,
    )

    logger.debug(
        "Creating SkillsMiddleware",
        backend_type=backend_type,
        source_count=len(sources),
        sources=sources,
    )

    middleware = SkillsMiddleware(
        backend=backend,
        sources=sources,
    )

    logger.info(
        "SkillsMiddleware created",
        backend_type=backend_type,
        source_count=len(sources),
    )
    return middleware
