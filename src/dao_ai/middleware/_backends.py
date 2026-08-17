"""
Shared backend resolution utility for Deep Agents middleware.

This module provides a helper function to resolve backend types from simple
string identifiers, used by all Deep Agents middleware factory functions
in DAO AI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

    from dao_ai.config import VolumePathModel

__all__ = [
    "resolve_backend",
]


def resolve_backend(
    backend_type: str = "state",
    root_dir: str | None = None,
    volume_path: str | VolumePathModel | None = None,
    virtual_mode: bool = False,
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
        virtual_mode: Whether the filesystem backend treats incoming paths as
            *virtual* paths anchored at ``root_dir``. Only meaningful for
            ``backend_type="filesystem"``; ignored otherwise. Always passed on
            to deepagents explicitly rather than left to its default, because
            that default flipped from ``False`` to ``True`` in 0.7.0 and
            ``pyproject.toml`` allows ``deepagents>=0.5.7`` — leaving it
            implicit means the behaviour changes under the caller's feet on
            upgrade. Defaults to ``False`` (real host paths) because that is
            what dao-ai's own resolvers produce; see
            :func:`dao_ai.middleware.filesystem.create_filesystem_middleware`
            for the agent-facing case, which defaults the other way.

            ``True`` additionally blocks ``..``/``~`` traversal and rejects
            paths resolving outside ``root_dir``. That is path-based
            confinement, not sandboxing or process isolation — deepagents is
            explicit that ``virtual_mode=False`` "provides no security even
            with ``root_dir`` set".

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
    from dao_ai._extras import require_extra

    require_extra("deepagents", feature="Deep Agents backends")
    from deepagents.backends import StateBackend
    from deepagents.backends.filesystem import FilesystemBackend
    from deepagents.backends.store import StoreBackend

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
            virtual_mode=virtual_mode,
        )
        # ``virtual_mode`` is always stated, never left to deepagents' default:
        # that default flipped to True in 0.7.0, which reinterprets an absolute
        # path as a *virtual* one anchored at ``root_dir`` (leading slash
        # stripped, remainder joined underneath) and rejects anything landing
        # outside it. Since ``deepagents>=0.5.7`` is allowed, an implicit default
        # would change behaviour on upgrade. The *value* is the caller's call:
        # dao-ai's skill and instruction-file resolvers hand this backend real
        # host paths (a config dir, an mlflow ``code`` dir, a ``/Volumes`` mount)
        # and need False, while agent-facing file tools want the confinement
        # True brings. See the ``virtual_mode`` arg docs above.
        return FilesystemBackend(root_dir=root_dir, virtual_mode=virtual_mode)

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
