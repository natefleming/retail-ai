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


# ---------------------------------------------------------------------------
# write_bundle file emission — covered end-to-end by live `dao-ai
# generate-bundle` validation on fevm. The unit tests above lock in the
# building-block helpers (_make_requirements_txt, _PYPROJECT_DEV_TEMPLATE)
# that those code paths use.
# ---------------------------------------------------------------------------
