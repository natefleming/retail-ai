"""
Filesystem middleware for DAO AI agents.

This module provides a factory function for creating FilesystemMiddleware instances
from the Deep Agents library. FilesystemMiddleware equips agents with file operation
tools: ``ls``, ``read_file``, ``write_file``, ``edit_file``, ``glob``, ``grep``,
and optionally ``execute`` (for shell commands when using a sandbox backend).

The middleware also auto-evicts large tool results to the filesystem when they
exceed a configurable token threshold, preventing context window saturation.

Example:
    from dao_ai.middleware import create_filesystem_middleware

    middleware = create_filesystem_middleware()

    # With filesystem backend for real file access:
    middleware = create_filesystem_middleware(
        backend_type="filesystem",
        root_dir="/workspace",
    )

    # With Databricks Volume backend:
    middleware = create_filesystem_middleware(
        backend_type="volume",
        volume_path="/Volumes/catalog/schema/volume",
    )

YAML Config:
    middleware:
      - name: dao_ai.middleware.filesystem.create_filesystem_middleware
        args:
          backend_type: state
          tool_token_limit_before_evict: 15000
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.middleware.filesystem import FilesystemMiddleware
from loguru import logger

from dao_ai.config import PromptModel
from dao_ai.middleware._backends import resolve_backend
from dao_ai.middleware._prompt_utils import resolve_prompt

if TYPE_CHECKING:
    from dao_ai.config import VolumePathModel

__all__ = [
    "create_filesystem_middleware",
]


def create_filesystem_middleware(
    backend_type: str = "state",
    root_dir: str | None = None,
    volume_path: str | VolumePathModel | None = None,
    tool_token_limit_before_evict: int | None = 20000,
    system_prompt: str | PromptModel | None = None,
    custom_tool_descriptions: dict[str, str] | None = None,
) -> FilesystemMiddleware:
    """
    Create a FilesystemMiddleware for agent file operations.

    This factory function creates a FilesystemMiddleware from the Deep
    Agents library that provides agents with filesystem tools (``ls``,
    ``read_file``, ``write_file``, ``edit_file``, ``glob``, ``grep``).
    When the backend supports it (``SandboxBackendProtocol``), an
    ``execute`` tool for shell commands is also added.

    The middleware auto-evicts large tool results to the filesystem when
    they exceed ``tool_token_limit_before_evict`` tokens, replacing them
    with a truncated preview and file reference.

    Args:
        backend_type: Backend for file storage. One of ``"state"``
            (ephemeral, default), ``"filesystem"`` (real disk),
            ``"store"`` (persistent via LangGraph Store), or
            ``"volume"`` (Databricks Unity Catalog Volume).
        root_dir: Root directory for filesystem backend. Required when
            ``backend_type="filesystem"``.
        volume_path: Volume path for Databricks Volume backend. Can be
            a string (e.g. ``"/Volumes/catalog/schema/volume"``) or a
            ``VolumePathModel``. Required when
            ``backend_type="volume"``.
        tool_token_limit_before_evict: Token limit before evicting a
            tool result to the filesystem. Set to ``None`` to disable
            eviction. Defaults to 20000.
        system_prompt: Custom system prompt override for filesystem
            tool guidance. Accepts a plain string or a ``PromptModel``
            from the prompt registry.
        custom_tool_descriptions: Optional dict mapping tool names to
            custom descriptions.

    Returns:
        A configured FilesystemMiddleware instance.

    Example:
        from dao_ai.middleware import create_filesystem_middleware

        # Ephemeral state backend (default)
        middleware = create_filesystem_middleware()

        # Real filesystem with custom eviction threshold
        middleware = create_filesystem_middleware(
            backend_type="filesystem",
            root_dir="/workspace",
            tool_token_limit_before_evict=10000,
        )

        # Databricks Volume backend
        middleware = create_filesystem_middleware(
            backend_type="volume",
            volume_path="/Volumes/catalog/schema/volume",
        )
    """
    backend = resolve_backend(
        backend_type=backend_type,
        root_dir=root_dir,
        volume_path=volume_path,
    )

    logger.debug(
        "Creating FilesystemMiddleware",
        backend_type=backend_type,
        tool_token_limit_before_evict=tool_token_limit_before_evict,
        custom_system_prompt=system_prompt is not None,
    )

    resolved_system_prompt: str | None = (
        resolve_prompt(system_prompt) if system_prompt is not None else None
    )

    middleware = FilesystemMiddleware(
        backend=backend,
        system_prompt=resolved_system_prompt,
        custom_tool_descriptions=custom_tool_descriptions,
        tool_token_limit_before_evict=tool_token_limit_before_evict,
    )

    logger.info(
        "FilesystemMiddleware created",
        backend_type=backend_type,
    )
    return middleware
