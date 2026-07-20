"""The generate-mcp dev build must stamp a unique PEP 440 local version.

Mirrors the deploy-path coverage in test_databricks / test_bundle_requirements:
every ``uv build --wheel`` site that produces a dev wheel wraps the build in
``dao_ai.utils.dev_local_version`` so an Apps container reinstalls the local
wheel instead of silently keeping the same-base-version published package.
"""

from __future__ import annotations

import pathlib
import subprocess
from unittest.mock import patch


def test_generate_mcp_dev_build_stamps_local_version(tmp_path) -> None:
    """_bundle_local_wheel wraps uv build in dev_local_version, so the project's
    pyproject.toml carries a +dev local segment during the build and is restored
    after."""
    from dao_ai.mcp import generate

    project_root = pathlib.Path(generate.__file__).parents[3]
    pyproject = project_root / "pyproject.toml"
    original = pyproject.read_text()

    captured: dict[str, str] = {}

    def fake_run(cmd, cwd, **kw):
        # Snapshot the version line as seen by the build subprocess.
        captured["during"] = (pathlib.Path(cwd) / "pyproject.toml").read_text()
        # Drop a fake wheel so the post-build glob succeeds.
        dist = pathlib.Path(cwd) / "dist"
        dist.mkdir(parents=True, exist_ok=True)
        (dist / "dao_ai-0.1.115+devtest-py3-none-any.whl").write_bytes(b"x")

        class _R:
            returncode = 0
            stderr = ""

        return _R()

    written: list[str] = []
    try:
        with patch.object(subprocess, "run", fake_run):
            generate._bundle_local_wheel(tmp_path, written=written)
        assert "+dev" in captured["during"], "build did not see a +dev version"
    finally:
        # Guarantee the real tree is restored even if the assert fails.
        pyproject.write_text(original)

    # dev_local_version restores the original on exit.
    assert pyproject.read_text() == original
