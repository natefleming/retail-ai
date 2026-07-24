"""Tests for the uv.lock-based dependency emission across Apps bundle paths.

Apps' build phase runs ``uv sync --locked --no-dev`` when a bundle ships
``pyproject.toml`` + ``uv.lock`` (and no ``requirements.txt``). These tests
lock in the building blocks:

- ``_PYPROJECT_DEV_TEMPLATE`` redirects dao-ai to the bundled wheel via
  ``[tool.uv.sources]`` so the generated lock installs THIS code.
- ``dao_ai._locking.generate_bundle_lock`` rewrites the internal mirror host to
  the public CDN and refuses a lock that still references the mirror.

Full ``write_bundle`` / ``deploy_apps_agent`` file emission is covered live on
fevm (real ``uv lock`` needs network + a real wheel).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dao_ai.apps.bundle import (
    _PYPROJECT_DEV_TEMPLATE,
    _PYPROJECT_TEMPLATE,
    _format_extra_deps,
)
from dao_ai.utils import dev_local_version

# ---------------------------------------------------------------------------
# pyproject templates — dev redirects dao-ai to the bundled wheel
# ---------------------------------------------------------------------------


class TestPyprojectTemplates:
    def test_dev_template_redirects_dao_ai_to_local_wheel(self) -> None:
        rendered: str = _PYPROJECT_DEV_TEMPLATE.format(
            name="foo-bar",
            package_name="foo_bar",
            wheel_filename="dao_ai-0.2.0-py3-none-any.whl",
            extras="[a2a]",
            extra_deps="",
        )
        assert "[tool.uv.sources]" in rendered
        assert 'dao-ai = { path = "dist/dao_ai-0.2.0-py3-none-any.whl" }' in rendered
        # The requirement itself is declared (with extras) so uv locks it.
        assert '"dao-ai[a2a]"' in rendered

    def test_published_template_pins_exact_version_with_extras(self) -> None:
        rendered: str = _PYPROJECT_TEMPLATE.format(
            name="foo-bar",
            package_name="foo_bar",
            dao_ai_version="0.2.0",
            extras="[a2a,rerank]",
            extra_deps="",
        )
        # Exact pin (==) for reproducible redeploys, extras threaded through.
        assert "dao-ai[a2a,rerank]==0.2.0" in rendered
        # No local wheel source in published mode.
        assert "tool.uv.sources" not in rendered

    def test_published_template_threads_user_pip_requirements(self) -> None:
        # config.app.pip_requirements flow through the {extra_deps} slot so the
        # generated pyproject's dependency array captures the deployer's own
        # packages alongside dao-ai.
        rendered: str = _PYPROJECT_TEMPLATE.format(
            name="foo-bar",
            package_name="foo_bar",
            dao_ai_version="0.2.0",
            extras="",
            extra_deps=_format_extra_deps(["httpx>=0.27"]),
        )
        assert '"httpx>=0.27",' in rendered


# ---------------------------------------------------------------------------
# generate_bundle_lock — host-swap + poison guard (fake `uv` via monkeypatch)
# ---------------------------------------------------------------------------


class TestGenerateBundleLock:
    def _fake_uv_lock(self, bundle_dir: Path, lock_body: str):
        """Return a subprocess.run replacement that writes ``lock_body`` as the
        lock and reports success, standing in for a real ``uv lock``."""

        def _run(cmd, cwd=None, capture_output=None, text=None, check=None):
            (Path(cwd) / "uv.lock").write_text(lock_body)

            class _R:
                returncode = 0
                stderr = ""

            return _R()

        return _run

    def test_rewrites_mirror_host_to_public_cdn(self, tmp_path, monkeypatch) -> None:
        from dao_ai import _locking

        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        poisoned = (
            'wheels = [{ url = '
            '"https://pypi-proxy.dev.databricks.com/packages/aa/bb/x.whl" }]\n'
        )
        monkeypatch.setattr(
            _locking.subprocess,
            "run",
            self._fake_uv_lock(tmp_path, poisoned),
        )
        _locking.generate_bundle_lock(tmp_path)
        result = (tmp_path / "uv.lock").read_text()
        assert "pypi-proxy.dev.databricks.com" not in result
        assert "files.pythonhosted.org" in result

    def test_raises_if_mirror_survives(self, tmp_path, monkeypatch) -> None:
        """A mirror reference that the host-swap can't reach (e.g. a bare host
        without the trailing slash form) must fail loudly, never ship."""
        from dao_ai import _locking

        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        # ``source = { registry = "https://pypi-proxy.dev.databricks.com/simple/" }``
        # is swapped, but an index reference without the https:// prefix isn't —
        # simulate a residue to prove the guard trips.
        residue = 'index = "pypi-proxy.dev.databricks.com"\n'
        monkeypatch.setattr(
            _locking.subprocess,
            "run",
            self._fake_uv_lock(tmp_path, residue),
        )
        with pytest.raises(RuntimeError, match="internal mirror"):
            _locking.generate_bundle_lock(tmp_path)

    def test_raises_actionable_message_on_unsatisfiable_dao_ai(
        self, tmp_path, monkeypatch
    ) -> None:
        """Published pin not yet on the index -> actionable release-time message."""
        from dao_ai import _locking

        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")

        def _run(cmd, cwd=None, capture_output=None, text=None, check=None):
            class _R:
                returncode = 1
                stderr = (
                    "No solution found when resolving dependencies:\n"
                    "Because only dao-ai<=0.1.111 is available and your project "
                    "depends on dao-ai>=0.2.0 ..."
                )

            return _R()

        monkeypatch.setattr(_locking.subprocess, "run", _run)
        with pytest.raises(RuntimeError, match="not\n?.*yet published|--development"):
            _locking.generate_bundle_lock(tmp_path)


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
        with dev_local_version(p):
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
            with dev_local_version(p):
                raise RuntimeError("build failed")
        assert p.read_text() == original

    def test_noop_when_version_already_local(self, tmp_path) -> None:
        """An existing local segment is left as-is (idempotent)."""
        p = self._write_pyproject(tmp_path, "0.1.115+dev999")
        with dev_local_version(p):
            import re

            m = re.search(r'version = "([^"]+)"', p.read_text())
            assert m.group(1) == "0.1.115+dev999"

    def test_noop_when_no_static_version(self, tmp_path) -> None:
        """A dynamic-version pyproject (no ``version =`` line) is untouched."""
        p = tmp_path / "pyproject.toml"
        p.write_text('[project]\nname = "dao-ai"\ndynamic = ["version"]\n')
        original = p.read_text()
        with dev_local_version(p):
            assert p.read_text() == original
        assert p.read_text() == original
