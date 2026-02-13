"""Databricks Volume backend for Deep Agents middleware.

This module provides a ``DatabricksVolumeBackend`` that implements the
deepagents ``BackendProtocol``, enabling agents to read and write files
in Databricks Unity Catalog Volumes.

All file operations are mapped to the Databricks SDK ``WorkspaceClient.files``
API. Paths within the agent are relative to the volume root (e.g.
``/data/file.txt`` maps to ``/Volumes/catalog/schema/volume/data/file.txt``).

Example:
    from dao_ai.middleware.backends.volume import DatabricksVolumeBackend

    backend = DatabricksVolumeBackend(
        volume_path="/Volumes/my_catalog/my_schema/agent_workspace",
    )

    # Or using VolumePathModel from config:
    from dao_ai.config import VolumePathModel, VolumeModel, SchemaModel

    volume_path = VolumePathModel(
        volume=VolumeModel(
            name="agent_workspace",
            schema=SchemaModel(
                catalog_name="my_catalog",
                schema_name="my_schema",
            ),
        ),
    )
    backend = DatabricksVolumeBackend(volume_path=volume_path)

YAML Config:
    middleware:
      - name: dao_ai.middleware.filesystem.create_filesystem_middleware
        args:
          backend_type: volume
          volume_path: /Volumes/my_catalog/my_schema/agent_workspace
"""

from __future__ import annotations

import io
from datetime import datetime
from fnmatch import fnmatch
from typing import TYPE_CHECKING

from databricks.sdk import WorkspaceClient
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from deepagents.backends.utils import (
    check_empty_content,
    format_content_with_line_numbers,
    perform_string_replacement,
)
from loguru import logger

if TYPE_CHECKING:
    from dao_ai.config import VolumePathModel

__all__ = [
    "DatabricksVolumeBackend",
]


class DatabricksVolumeBackend(BackendProtocol):
    """Backend that reads and writes files in Databricks Unity Catalog Volumes.

    This backend maps the deepagents filesystem abstraction to Databricks
    Volumes via the ``WorkspaceClient.files`` API. Agent-visible paths are
    relative to the configured volume root.

    Args:
        volume_path: The volume root path. Can be:
            - A string like ``"/Volumes/catalog/schema/volume"``
            - A ``VolumePathModel`` instance (resolved via ``full_name``)
        w: Optional ``WorkspaceClient`` instance. If not provided, one is
            created using environment-based authentication.

    Example:
        >>> backend = DatabricksVolumeBackend(
        ...     volume_path="/Volumes/my_catalog/my_schema/workspace",
        ... )
        >>> info = backend.ls_info("/")
    """

    def __init__(
        self,
        volume_path: str | VolumePathModel,
        w: WorkspaceClient | None = None,
    ) -> None:
        from dao_ai.config import VolumePathModel

        if isinstance(volume_path, VolumePathModel):
            self._root = volume_path.full_name.rstrip("/")
        else:
            self._root = volume_path.rstrip("/")

        if not self._root.startswith("/Volumes/"):
            raise ValueError(
                f"volume_path must start with '/Volumes/': got {self._root!r}"
            )

        self._w = w or WorkspaceClient()
        logger.debug(
            "DatabricksVolumeBackend initialized",
            volume_root=self._root,
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _abs_path(self, path: str) -> str:
        """Map an agent-relative path to the absolute volume path."""
        path = path.strip()
        if not path or path == "/":
            return self._root
        clean = path.lstrip("/")
        return f"{self._root}/{clean}"

    def _rel_path(self, abs_path: str) -> str:
        """Map an absolute volume path back to an agent-relative path."""
        if abs_path.startswith(self._root):
            suffix = abs_path[len(self._root) :]
            return suffix if suffix else "/"
        return abs_path

    # ------------------------------------------------------------------
    # BackendProtocol implementation
    # ------------------------------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files and directories at the given path.

        Args:
            path: Agent-relative directory path (e.g. ``"/"``, ``"/data"``).

        Returns:
            List of ``FileInfo`` dicts with ``path``, ``is_dir``, and
            ``size`` fields.
        """
        abs_dir = self._abs_path(path)
        try:
            entries = list(self._w.files.list_directory_contents(abs_dir))
        except Exception as exc:
            logger.warning(
                "ls_info failed",
                path=abs_dir,
                error=str(exc),
            )
            return []

        results: list[FileInfo] = []
        for entry in entries:
            is_dir = bool(entry.is_directory)
            entry_name = entry.name or ""
            # Build agent-relative path
            if path == "/" or path == "":
                rel = f"/{entry_name}"
            else:
                normalized = path.rstrip("/")
                rel = f"{normalized}/{entry_name}"
            if is_dir:
                rel = rel.rstrip("/") + "/"

            info: FileInfo = {
                "path": rel,
                "is_dir": is_dir,
            }
            if entry.file_size is not None:
                info["size"] = int(entry.file_size)
            if entry.last_modified is not None:
                if isinstance(entry.last_modified, datetime):
                    info["modified_at"] = entry.last_modified.isoformat()
                else:
                    info["modified_at"] = str(entry.last_modified)
            results.append(info)

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Agent-relative file path.
            offset: Line offset (0-indexed).
            limit: Maximum number of lines to return.

        Returns:
            Formatted content with line numbers, or an error string.
        """
        abs_path = self._abs_path(file_path)
        try:
            resp = self._w.files.download(abs_path)
            raw_bytes = resp.contents.read()
            content = raw_bytes.decode("utf-8")
        except Exception as exc:
            return f"Error: File '{file_path}' not found ({exc})"

        empty_msg = check_empty_content(content)
        if empty_msg:
            return empty_msg

        lines = content.splitlines()
        start_idx = offset
        end_idx = min(start_idx + limit, len(lines))

        if start_idx >= len(lines):
            return (
                f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"
            )

        selected = lines[start_idx:end_idx]
        return format_content_with_line_numbers(selected, start_line=start_idx + 1)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file. Returns error if file already exists.

        Args:
            file_path: Agent-relative path for the new file.
            content: Text content to write.

        Returns:
            ``WriteResult`` with ``files_update=None`` (external backend).
        """
        abs_path = self._abs_path(file_path)

        # Check if file already exists (create-only semantics)
        try:
            self._w.files.get_metadata(abs_path)
            return WriteResult(
                error=(
                    f"Cannot write to {file_path} because it already "
                    f"exists. Read and then make an edit, or write to "
                    f"a new path."
                )
            )
        except Exception:
            # File does not exist -- proceed to write
            pass

        try:
            # Ensure parent directory exists
            parent = abs_path.rsplit("/", 1)[0]
            if parent and parent != self._root:
                try:
                    self._w.files.create_directory(parent)
                except Exception:
                    pass  # Directory may already exist

            data = content.encode("utf-8")
            self._w.files.upload(
                abs_path,
                contents=io.BytesIO(data),
                overwrite=False,
            )
            return WriteResult(path=file_path, files_update=None)
        except Exception as exc:
            return WriteResult(error=f"Error writing file '{file_path}': {exc}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Agent-relative file path.
            old_string: Text to find and replace.
            new_string: Replacement text.
            replace_all: If True, replace all occurrences.

        Returns:
            ``EditResult`` with ``files_update=None`` (external backend).
        """
        abs_path = self._abs_path(file_path)

        # Download current content
        try:
            resp = self._w.files.download(abs_path)
            raw_bytes = resp.contents.read()
            content = raw_bytes.decode("utf-8")
        except Exception:
            return EditResult(error=f"Error: File '{file_path}' not found")

        # Perform replacement
        result = perform_string_replacement(
            content, old_string, new_string, replace_all
        )
        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result

        # Upload modified content
        try:
            data = new_content.encode("utf-8")
            self._w.files.upload(
                abs_path,
                contents=io.BytesIO(data),
                overwrite=True,
            )
            return EditResult(
                path=file_path,
                files_update=None,
                occurrences=int(occurrences),
            )
        except Exception as exc:
            return EditResult(error=f"Error editing file '{file_path}': {exc}")

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Search for a literal text pattern in files.

        Downloads files from the volume and scans content line by line.

        Args:
            pattern: Literal string to search for.
            path: Optional directory path to search in.
            glob: Optional glob pattern to filter filenames.

        Returns:
            List of ``GrepMatch`` dicts, or error string.
        """
        search_path = path or "/"
        all_files = self._list_recursive(search_path)

        if glob:
            all_files = [
                f for f in all_files if fnmatch(f["path"].rsplit("/", 1)[-1], glob)
            ]

        matches: list[GrepMatch] = []
        for file_info in all_files:
            if file_info.get("is_dir"):
                continue
            file_rel = file_info["path"]
            abs_fp = self._abs_path(file_rel)
            try:
                resp = self._w.files.download(abs_fp)
                raw = resp.contents.read()
                text = raw.decode("utf-8")
            except Exception:
                continue

            for line_num, line in enumerate(text.splitlines(), 1):
                if pattern in line:
                    matches.append(
                        {
                            "path": file_rel,
                            "line": line_num,
                            "text": line,
                        }
                    )
        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g. ``"*.py"``, ``"**/*.txt"``).
            path: Base directory to search from.

        Returns:
            List of matching ``FileInfo`` dicts.
        """
        all_files = self._list_recursive(path)

        results: list[FileInfo] = []
        for file_info in all_files:
            if file_info.get("is_dir"):
                continue
            rel = file_info["path"]
            # Build relative-to-search-path for matching
            if path == "/" or path == "":
                match_path = rel.lstrip("/")
            else:
                normalized = path.rstrip("/")
                if rel.startswith(normalized + "/"):
                    match_path = rel[len(normalized) + 1 :]
                else:
                    match_path = rel.lstrip("/")

            if fnmatch(match_path, pattern):
                results.append(file_info)

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the volume.

        Args:
            files: List of ``(path, content_bytes)`` tuples.

        Returns:
            List of ``FileUploadResponse`` objects.
        """
        responses: list[FileUploadResponse] = []
        for rel_path, content_bytes in files:
            abs_path = self._abs_path(rel_path)
            try:
                parent = abs_path.rsplit("/", 1)[0]
                if parent and parent != self._root:
                    try:
                        self._w.files.create_directory(parent)
                    except Exception:
                        pass
                self._w.files.upload(
                    abs_path,
                    contents=io.BytesIO(content_bytes),
                    overwrite=True,
                )
                responses.append(FileUploadResponse(path=rel_path, error=None))
            except PermissionError:
                responses.append(
                    FileUploadResponse(path=rel_path, error="permission_denied")
                )
            except Exception:
                responses.append(
                    FileUploadResponse(path=rel_path, error="invalid_path")
                )
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the volume.

        Args:
            paths: List of agent-relative file paths.

        Returns:
            List of ``FileDownloadResponse`` objects.
        """
        responses: list[FileDownloadResponse] = []
        for rel_path in paths:
            abs_path = self._abs_path(rel_path)
            try:
                resp = self._w.files.download(abs_path)
                content = resp.contents.read()
                responses.append(
                    FileDownloadResponse(path=rel_path, content=content, error=None)
                )
            except Exception:
                responses.append(
                    FileDownloadResponse(
                        path=rel_path,
                        content=None,
                        error="file_not_found",
                    )
                )
        return responses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _list_recursive(self, path: str) -> list[FileInfo]:
        """Recursively list all files under the given path.

        Args:
            path: Agent-relative directory path.

        Returns:
            Flat list of ``FileInfo`` dicts for all files (no directories).
        """
        results: list[FileInfo] = []
        stack = [path]

        while stack:
            current = stack.pop()
            entries = self.ls_info(current)
            for entry in entries:
                if entry.get("is_dir"):
                    # Strip trailing slash for recursion
                    dir_path = entry["path"].rstrip("/")
                    stack.append(dir_path)
                else:
                    results.append(entry)

        return results
