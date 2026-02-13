"""
AGENTS.md memory middleware for DAO AI agents.

This module provides a factory function for creating MemoryMiddleware instances
from the Deep Agents library. MemoryMiddleware loads agent context from
``AGENTS.md`` files at startup and injects it into the system prompt, following
the `agents.md specification <https://agents.md/>`_.

This is a different memory model from DAO AI's existing database-backed memory
(``dao_ai.memory``). While ``dao_ai.memory`` persists agent state in
PostgreSQL/Lakebase, the AGENTS.md approach stores context as markdown files
that agents can read and update via filesystem tools.

Example:
    from dao_ai.middleware import create_agents_memory_middleware

    middleware = create_agents_memory_middleware(
        sources=["~/.deepagents/AGENTS.md", "./.deepagents/AGENTS.md"],
    )

YAML Config:
    middleware:
      - name: dao_ai.middleware.memory_agents.create_agents_memory_middleware
        args:
          sources:
            - "~/.deepagents/AGENTS.md"
            - "./.deepagents/AGENTS.md"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.middleware.memory import MemoryMiddleware
from loguru import logger

from dao_ai.middleware._backends import resolve_backend

if TYPE_CHECKING:
    from dao_ai.config import VolumePathModel

__all__ = [
    "create_agents_memory_middleware",
]


def create_agents_memory_middleware(
    sources: list[str],
    backend_type: str = "state",
    root_dir: str | None = None,
    volume_path: str | VolumePathModel | None = None,
) -> MemoryMiddleware:
    """
    Create a MemoryMiddleware for loading AGENTS.md context files.

    This factory function creates a MemoryMiddleware from the Deep Agents
    library that loads agent memory from ``AGENTS.md`` files at startup and
    injects the content into the system prompt.

    Memory sources are loaded in order, with all content combined. The
    middleware also encourages the agent to update memory files via
    ``edit_file`` when it learns useful information from interactions.

    Common memory file contents include:
        - Project overview and architecture
        - Build/test commands
        - Code style guidelines
        - User preferences

    Args:
        sources: List of paths to AGENTS.md files to load. Paths use POSIX
            conventions (forward slashes) and are relative to the backend
            root. Sources are loaded in order with content concatenated.
        backend_type: Backend for file storage. One of ``"state"``
            (ephemeral, default), ``"filesystem"`` (real disk),
            ``"store"`` (persistent), or ``"volume"`` (Databricks
            Unity Catalog Volume).
        root_dir: Root directory for filesystem backend. Required when
            ``backend_type="filesystem"``.
        volume_path: Volume path for Databricks Volume backend.
            Required when ``backend_type="volume"``.

    Returns:
        A configured MemoryMiddleware instance.

    Raises:
        ValueError: If sources is empty.

    Example:
        from dao_ai.middleware import create_agents_memory_middleware

        # Load from multiple sources
        middleware = create_agents_memory_middleware(
            sources=[
                "~/.deepagents/AGENTS.md",
                "./.deepagents/AGENTS.md",
            ],
            backend_type="filesystem",
            root_dir="/",
        )
    """
    if not sources:
        raise ValueError("At least one source path is required for MemoryMiddleware.")

    backend = resolve_backend(
        backend_type=backend_type,
        root_dir=root_dir,
        volume_path=volume_path,
    )

    logger.debug(
        "Creating MemoryMiddleware",
        backend_type=backend_type,
        source_count=len(sources),
        sources=sources,
    )

    middleware = MemoryMiddleware(
        backend=backend,
        sources=sources,
    )

    logger.info(
        "MemoryMiddleware created",
        backend_type=backend_type,
        source_count=len(sources),
    )
    return middleware
