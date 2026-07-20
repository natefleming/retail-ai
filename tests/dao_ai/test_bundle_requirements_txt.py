"""Tests for the unified requirements.txt emission across bundle paths.

After the refactor that drops ``uv.lock`` everywhere in favor of a
``requirements.txt`` that Apps' build phase recognizes directly, all
four file-emit paths must produce a ``requirements.txt``:

- ``dao-ai generate-bundle``   (published mode)
- ``dao-ai generate-bundle --development``
- ``dao-ai deploy --target apps`` (programmatic, published)
- ``dao-ai deploy --target apps`` (programmatic, dev)

Tests in this file cover the two ``generate-bundle`` paths via the
``write_bundle`` API. The two ``deploy_apps_agent`` paths are covered in
``test_databricks.py``.
"""

from __future__ import annotations

import pytest

from dao_ai.apps.bundle import (
    _PYPROJECT_DEV_TEMPLATE,
    _dev_local_version,
    _make_requirements_txt,
)

# ---------------------------------------------------------------------------
# _make_requirements_txt — content helpers
# ---------------------------------------------------------------------------


class TestMakeRequirementsTxt:
    def test_published_installs_dao_ai_unbounded(self) -> None:
        """The published pin is intentionally unbounded — the locally-installed
        version may be an unreleased pre-publish build, so floor-pinning to
        it would cause Apps to fail with ``Could not find a version``."""
        content: str = _make_requirements_txt(development=False)
        assert content.strip() == "dao-ai", (
            f"Published requirements.txt must install unbounded dao-ai; got: {content!r}"
        )

    def test_published_does_not_reference_local_wheel(self) -> None:
        content: str = _make_requirements_txt(development=False)
        assert "./dist/" not in content
        assert ".whl" not in content

    def test_dev_references_local_wheel(self) -> None:
        content: str = _make_requirements_txt(
            development=True,
            wheel_filename="dao_ai-0.1.96-py3-none-any.whl",
        )
        assert content.strip() == "./dist/dao_ai-0.1.96-py3-none-any.whl"

    def test_dev_requires_wheel_filename(self) -> None:
        with pytest.raises(ValueError, match="wheel_filename is required"):
            _make_requirements_txt(development=True)


# ---------------------------------------------------------------------------
# _PYPROJECT_DEV_TEMPLATE — no longer references the wheel
# ---------------------------------------------------------------------------


class TestPyprojectDevTemplate:
    def test_dev_template_renders_with_just_name_and_package_name(self) -> None:
        rendered: str = _PYPROJECT_DEV_TEMPLATE.format(
            name="foo-bar", package_name="foo_bar"
        )
        # No wheel reference — that lives in requirements.txt now.
        assert "tool.uv.sources" not in rendered
        assert "wheel_filename" not in rendered
        # Deps empty; requirements.txt handles installs.
        assert "dependencies = []" in rendered


class TestDevLocalVersion:
    """The dev build stamps a unique PEP 440 local version, then restores."""

    def _write_pyproject(self, tmp_path, version: str) -> "object":
        p = tmp_path / "pyproject.toml"
        p.write_text(
            f'[project]\nname = "dao-ai"\nversion = "{version}"\n'
            'description = "x"\n'
        )
        return p

    def test_stamps_local_segment_and_restores(self, tmp_path) -> None:
        p = self._write_pyproject(tmp_path, "0.1.115")
        original = p.read_text()
        seen = {}
        with _dev_local_version(p):
            seen["during"] = p.read_text()
        # Restored exactly on exit.
        assert p.read_text() == original
        # Inside the context, a +dev local segment was present.
        import re

        m = re.search(r'version = "([^"]+)"', seen["during"])
        assert m is not None
        assert m.group(1).startswith("0.1.115+dev")

    def test_restores_even_on_exception(self, tmp_path) -> None:
        p = self._write_pyproject(tmp_path, "0.1.115")
        original = p.read_text()
        with pytest.raises(RuntimeError):
            with _dev_local_version(p):
                raise RuntimeError("build failed")
        assert p.read_text() == original

    def test_noop_when_version_already_local(self, tmp_path) -> None:
        """An existing local segment is left as-is (idempotent)."""
        p = self._write_pyproject(tmp_path, "0.1.115+dev999")
        with _dev_local_version(p):
            import re

            m = re.search(r'version = "([^"]+)"', p.read_text())
            assert m.group(1) == "0.1.115+dev999"

    def test_noop_when_no_static_version(self, tmp_path) -> None:
        """A dynamic-version pyproject (no ``version =`` line) is untouched."""
        p = tmp_path / "pyproject.toml"
        p.write_text('[project]\nname = "dao-ai"\ndynamic = ["version"]\n')
        original = p.read_text()
        with _dev_local_version(p):
            assert p.read_text() == original
        assert p.read_text() == original


# ---------------------------------------------------------------------------
# write_bundle file emission — covered end-to-end by live `dao-ai
# generate-bundle` validation on fevm. The unit tests above lock in the
# building-block helpers (_make_requirements_txt, _PYPROJECT_DEV_TEMPLATE,
# _dev_local_version) that those code paths use.
# ---------------------------------------------------------------------------
