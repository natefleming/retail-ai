"""
Tests for DatabricksVolumeBackend.

Tests use a mocked WorkspaceClient to verify backend behavior without
requiring a live Databricks workspace.

Run with:
    pytest tests/dao_ai/test_volume_backend.py -v -m unit
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from dao_ai.middleware.backends.volume import DatabricksVolumeBackend

# ------------------------------------------------------------------
# Fixtures and helpers
# ------------------------------------------------------------------


@dataclass
class _DirectoryEntry:
    """Minimal mock of databricks.sdk.service.files.DirectoryEntry."""

    name: str
    is_directory: bool = False
    file_size: int | None = None
    last_modified: str | None = None


@dataclass
class _DownloadResponse:
    """Minimal mock of databricks.sdk.service.files.DownloadResponse."""

    contents: io.BytesIO


def _make_download(text: str) -> _DownloadResponse:
    """Create a mock download response from text content."""
    return _DownloadResponse(contents=io.BytesIO(text.encode("utf-8")))


@pytest.fixture()
def mock_ws() -> MagicMock:
    """Create a mock WorkspaceClient with a files attribute."""
    ws = MagicMock()
    ws.files = MagicMock()
    return ws


@pytest.fixture()
def backend(mock_ws: MagicMock) -> DatabricksVolumeBackend:
    """Create a DatabricksVolumeBackend with a mocked client."""
    return DatabricksVolumeBackend(
        volume_path="/Volumes/cat/schema/vol",
        w=mock_ws,
    )


# ------------------------------------------------------------------
# Constructor tests
# ------------------------------------------------------------------


@pytest.mark.unit
class TestConstructor:
    """Tests for DatabricksVolumeBackend initialization."""

    def test_string_volume_path(self, mock_ws: MagicMock) -> None:
        """Test creating backend with a string volume path."""
        b = DatabricksVolumeBackend(
            volume_path="/Volumes/cat/schema/vol",
            w=mock_ws,
        )
        assert b._root == "/Volumes/cat/schema/vol"

    def test_string_trailing_slash_stripped(self, mock_ws: MagicMock) -> None:
        """Test trailing slash is stripped from volume path."""
        b = DatabricksVolumeBackend(
            volume_path="/Volumes/cat/schema/vol/",
            w=mock_ws,
        )
        assert b._root == "/Volumes/cat/schema/vol"

    def test_volume_path_model(self, mock_ws: MagicMock) -> None:
        """Test creating backend with a VolumePathModel."""
        from dao_ai.config import SchemaModel, VolumeModel, VolumePathModel

        vpm = VolumePathModel(
            volume=VolumeModel(
                name="vol",
                schema=SchemaModel(
                    catalog_name="cat",
                    schema_name="schema",
                ),
            ),
        )
        b = DatabricksVolumeBackend(volume_path=vpm, w=mock_ws)
        assert b._root == "/Volumes/cat/schema/vol"

    def test_invalid_path_raises(self, mock_ws: MagicMock) -> None:
        """Test that non-/Volumes/ path raises ValueError."""
        with pytest.raises(ValueError, match="must start with"):
            DatabricksVolumeBackend(
                volume_path="/dbfs/some/path",
                w=mock_ws,
            )


# ------------------------------------------------------------------
# ls_info tests
# ------------------------------------------------------------------


@pytest.mark.unit
class TestLsInfo:
    """Tests for ls_info."""

    def test_ls_root(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test listing root directory."""
        mock_ws.files.list_directory_contents.return_value = [
            _DirectoryEntry(name="data", is_directory=True, file_size=0),
            _DirectoryEntry(name="readme.md", is_directory=False, file_size=42),
        ]
        result = backend.ls_info("/")
        assert len(result) == 2
        # Sorted by path
        assert result[0]["path"] == "/data/"
        assert result[0]["is_dir"] is True
        assert result[1]["path"] == "/readme.md"
        assert result[1]["is_dir"] is False
        assert result[1]["size"] == 42

    def test_ls_subdirectory(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test listing a subdirectory."""
        mock_ws.files.list_directory_contents.return_value = [
            _DirectoryEntry(name="file.txt", is_directory=False, file_size=10),
        ]
        result = backend.ls_info("/data")
        assert len(result) == 1
        assert result[0]["path"] == "/data/file.txt"

    def test_ls_error_returns_empty(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test that errors return empty list."""
        mock_ws.files.list_directory_contents.side_effect = Exception("Not found")
        result = backend.ls_info("/nonexistent")
        assert result == []


# ------------------------------------------------------------------
# read tests
# ------------------------------------------------------------------


@pytest.mark.unit
class TestRead:
    """Tests for read."""

    def test_read_file(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test reading a file returns numbered content."""
        mock_ws.files.download.return_value = _make_download("line1\nline2\nline3")
        result = backend.read("/data/file.txt")
        assert "1\tline1" in result
        assert "2\tline2" in result
        assert "3\tline3" in result

    def test_read_with_offset(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test reading with offset skips lines."""
        mock_ws.files.download.return_value = _make_download("line1\nline2\nline3")
        result = backend.read("/file.txt", offset=1, limit=1)
        assert "line1" not in result
        assert "2\tline2" in result
        assert "line3" not in result

    def test_read_offset_exceeds_length(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test that offset beyond file length returns error."""
        mock_ws.files.download.return_value = _make_download("line1\nline2")
        result = backend.read("/file.txt", offset=100)
        assert "Error" in result
        assert "exceeds file length" in result

    def test_read_not_found(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test reading missing file returns error string."""
        mock_ws.files.download.side_effect = Exception("Not found")
        result = backend.read("/missing.txt")
        assert "Error" in result
        assert "not found" in result


# ------------------------------------------------------------------
# write tests
# ------------------------------------------------------------------


@pytest.mark.unit
class TestWrite:
    """Tests for write."""

    def test_write_new_file(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test writing a new file succeeds."""
        mock_ws.files.get_metadata.side_effect = Exception("Not found")
        result = backend.write("/new.txt", "hello world")
        assert result.error is None
        assert result.path == "/new.txt"
        assert result.files_update is None
        mock_ws.files.upload.assert_called_once()

    def test_write_existing_file_errors(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test that writing to existing file returns error."""
        mock_ws.files.get_metadata.return_value = MagicMock()
        result = backend.write("/existing.txt", "content")
        assert result.error is not None
        assert "already exists" in result.error
        mock_ws.files.upload.assert_not_called()

    def test_write_creates_parent_directory(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test that write creates parent directories."""
        mock_ws.files.get_metadata.side_effect = Exception("Not found")
        backend.write("/deep/nested/file.txt", "content")
        mock_ws.files.create_directory.assert_called_once_with(
            "/Volumes/cat/schema/vol/deep/nested"
        )


# ------------------------------------------------------------------
# edit tests
# ------------------------------------------------------------------


@pytest.mark.unit
class TestEdit:
    """Tests for edit."""

    def test_edit_replaces_string(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test editing replaces a string and re-uploads."""
        mock_ws.files.download.return_value = _make_download("Hello World")
        result = backend.edit("/file.txt", "World", "Universe")
        assert result.error is None
        assert result.path == "/file.txt"
        assert result.occurrences == 1
        assert result.files_update is None
        # Verify re-upload with new content
        call_args = mock_ws.files.upload.call_args
        uploaded = call_args[1]["contents"].read()
        assert uploaded == b"Hello Universe"

    def test_edit_not_found(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test editing missing file returns error."""
        mock_ws.files.download.side_effect = Exception("Not found")
        result = backend.edit("/missing.txt", "old", "new")
        assert result.error is not None
        assert "not found" in result.error

    def test_edit_string_not_found(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test editing with non-matching string returns error."""
        mock_ws.files.download.return_value = _make_download("Hello World")
        result = backend.edit("/file.txt", "Nonexistent", "Replacement")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_edit_multiple_without_replace_all(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test that multiple matches without replace_all errors."""
        mock_ws.files.download.return_value = _make_download("foo bar foo")
        result = backend.edit("/file.txt", "foo", "baz")
        assert result.error is not None
        assert "2 times" in result.error

    def test_edit_replace_all(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test replace_all replaces all occurrences."""
        mock_ws.files.download.return_value = _make_download("foo bar foo")
        result = backend.edit("/file.txt", "foo", "baz", replace_all=True)
        assert result.error is None
        assert result.occurrences == 2
        call_args = mock_ws.files.upload.call_args
        uploaded = call_args[1]["contents"].read()
        assert uploaded == b"baz bar baz"


# ------------------------------------------------------------------
# grep_raw tests
# ------------------------------------------------------------------


@pytest.mark.unit
class TestGrepRaw:
    """Tests for grep_raw."""

    def test_grep_finds_matches(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test grep finds matching lines."""
        mock_ws.files.list_directory_contents.return_value = [
            _DirectoryEntry(name="file.py", is_directory=False, file_size=100),
        ]
        mock_ws.files.download.return_value = _make_download(
            "import os\nimport sys\nprint('hello')"
        )
        matches = backend.grep_raw("import")
        assert isinstance(matches, list)
        assert len(matches) == 2
        assert matches[0]["text"] == "import os"
        assert matches[0]["line"] == 1
        assert matches[1]["text"] == "import sys"
        assert matches[1]["line"] == 2

    def test_grep_no_matches(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test grep with no matches returns empty list."""
        mock_ws.files.list_directory_contents.return_value = [
            _DirectoryEntry(name="file.txt", is_directory=False, file_size=10),
        ]
        mock_ws.files.download.return_value = _make_download("hello world")
        matches = backend.grep_raw("nonexistent")
        assert matches == []

    def test_grep_with_glob_filter(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test grep with glob filters filenames."""
        mock_ws.files.list_directory_contents.return_value = [
            _DirectoryEntry(name="file.py", is_directory=False, file_size=10),
            _DirectoryEntry(name="file.txt", is_directory=False, file_size=10),
        ]
        # Only download for .py files should happen
        mock_ws.files.download.return_value = _make_download("import os")
        matches = backend.grep_raw("import", glob="*.py")
        assert isinstance(matches, list)
        # Only .py file should match
        for m in matches:
            assert m["path"].endswith(".py")


# ------------------------------------------------------------------
# glob_info tests
# ------------------------------------------------------------------


@pytest.mark.unit
class TestGlobInfo:
    """Tests for glob_info."""

    def test_glob_matches_pattern(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test glob finds matching files."""
        mock_ws.files.list_directory_contents.return_value = [
            _DirectoryEntry(name="app.py", is_directory=False, file_size=100),
            _DirectoryEntry(name="readme.md", is_directory=False, file_size=50),
            _DirectoryEntry(name="test.py", is_directory=False, file_size=80),
        ]
        results = backend.glob_info("*.py")
        assert len(results) == 2
        paths = [r["path"] for r in results]
        assert "/app.py" in paths
        assert "/test.py" in paths

    def test_glob_no_matches(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test glob with no matches returns empty list."""
        mock_ws.files.list_directory_contents.return_value = [
            _DirectoryEntry(name="file.txt", is_directory=False, file_size=10),
        ]
        results = backend.glob_info("*.py")
        assert results == []


# ------------------------------------------------------------------
# upload_files / download_files tests
# ------------------------------------------------------------------


@pytest.mark.unit
class TestFileTransfer:
    """Tests for upload_files and download_files."""

    def test_upload_files(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test uploading multiple files."""
        responses = backend.upload_files([("/a.txt", b"aaa"), ("/b.txt", b"bbb")])
        assert len(responses) == 2
        assert all(r.error is None for r in responses)

    def test_download_files(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test downloading multiple files."""
        mock_ws.files.download.return_value = _make_download("content")
        responses = backend.download_files(["/a.txt"])
        assert len(responses) == 1
        assert responses[0].error is None
        assert responses[0].content == b"content"

    def test_download_missing_file(
        self,
        backend: DatabricksVolumeBackend,
        mock_ws: MagicMock,
    ) -> None:
        """Test downloading missing file returns error."""
        mock_ws.files.download.side_effect = Exception("Not found")
        responses = backend.download_files(["/missing.txt"])
        assert len(responses) == 1
        assert responses[0].error == "file_not_found"
        assert responses[0].content is None


# ------------------------------------------------------------------
# resolve_backend integration test
# ------------------------------------------------------------------


@pytest.mark.unit
class TestResolveBackend:
    """Tests for resolve_backend with volume type."""

    def test_resolve_volume_backend(self) -> None:
        """Test that resolve_backend creates a DatabricksVolumeBackend."""
        from dao_ai.middleware.backends.volume import (
            DatabricksVolumeBackend as VolumeBackendCls,
        )

        with patch("dao_ai.middleware.backends.volume.WorkspaceClient") as mock_ws_cls:
            mock_ws_cls.return_value = MagicMock()
            from dao_ai.middleware._backends import resolve_backend

            result = resolve_backend(
                "volume",
                volume_path="/Volumes/cat/schema/vol",
            )
            assert isinstance(result, VolumeBackendCls)

    def test_resolve_volume_missing_path_raises(self) -> None:
        """Test that volume backend without path raises ValueError."""
        from dao_ai.middleware._backends import resolve_backend

        with pytest.raises(ValueError, match="volume_path is required"):
            resolve_backend("volume")

    def test_resolve_unknown_mentions_volume(self) -> None:
        """Test that unknown backend error message lists volume."""
        from dao_ai.middleware._backends import resolve_backend

        with pytest.raises(ValueError, match="volume"):
            resolve_backend("unknown")
