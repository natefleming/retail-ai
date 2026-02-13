"""
Shared backend resolution utility for Deep Agents middleware.

This module provides a helper function to resolve backend types from simple
string identifiers, used by all Deep Agents middleware factory functions
in DAO AI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends import StateBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.store import StoreBackend
from loguru import logger

if TYPE_CHECKING:
    from dao_ai.config import VolumePathModel

__all__ = [
    "resolve_backend",
]


def resolve_backend(
    backend_type: str = "state",
    root_dir: str | None = None,
    volume_path: str | VolumePathModel | None = None,
) -> BackendProtocol | type[BackendProtocol]:
    """
    Resolve a backend type string to a Deep Agents backend instance or
    factory.

    This utility maps simple string identifiers to the corresponding
    deepagents backend classes, making it easy to configure backends via
    YAML.

    Args:
        backend_type: The type of backend to create. One of:
            - ``"state"`` (default): Ephemeral storage in LangGraph state.
              Returns the ``StateBackend`` class (used as a factory).
            - ``"filesystem"``: Real filesystem backend. Requires
              ``root_dir``.
            - ``"store"``: Persistent storage via LangGraph Store.
            - ``"volume"``: Databricks Unity Catalog Volume backend.
              Requires ``volume_path``.
        root_dir: Root directory for filesystem backend. Required when
            ``backend_type="filesystem"``, ignored otherwise.
        volume_path: Volume path for the Databricks Volume backend.
            Can be a string (e.g. ``"/Volumes/catalog/schema/volume"``)
            or a ``VolumePathModel`` instance. Required when
            ``backend_type="volume"``, ignored otherwise.

    Returns:
        A backend instance or factory callable compatible with deepagents
        middleware.

    Raises:
        ValueError: If backend_type is not recognized, or if required
            parameters are missing for the chosen backend.

    Example:
        from dao_ai.middleware._backends import resolve_backend

        # Ephemeral state backend (default)
        backend = resolve_backend("state")

        # Filesystem backend
        backend = resolve_backend("filesystem", root_dir="/workspace")

        # Databricks Volume backend
        backend = resolve_backend(
            "volume",
            volume_path="/Volumes/catalog/schema/volume",
        )
    """
    if backend_type == "state":
        logger.debug("Resolving backend", backend_type=backend_type)
        return StateBackend

    if backend_type == "filesystem":
        if root_dir is None:
            raise ValueError(
                "root_dir is required for filesystem backend. "
                "Specify the root directory for file operations."
            )
        logger.debug(
            "Resolving backend",
            backend_type=backend_type,
            root_dir=root_dir,
        )
        return FilesystemBackend(root_dir=root_dir)

    if backend_type == "store":
        logger.debug("Resolving backend", backend_type=backend_type)
        return StoreBackend

    if backend_type == "volume":
        from dao_ai.middleware.backends.volume import (
            DatabricksVolumeBackend,
        )

        if volume_path is None:
            raise ValueError(
                "volume_path is required for volume backend. "
                "Provide a string path "
                "(e.g. '/Volumes/catalog/schema/volume') "
                "or a VolumePathModel instance."
            )
        logger.debug(
            "Resolving backend",
            backend_type=backend_type,
            volume_path=str(volume_path),
        )
        return DatabricksVolumeBackend(volume_path=volume_path)

    raise ValueError(
        f"Unknown backend_type: {backend_type!r}. "
        f"Must be one of: 'state', 'filesystem', 'store', 'volume'."
    )
