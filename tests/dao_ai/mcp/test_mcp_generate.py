"""Unit tests for dao_ai.mcp.generate.write_mcp_bundle."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _fixtures import mcp_config  # noqa: E402


@pytest.mark.unit
def test_write_mcp_bundle_emits_expected_files(tmp_path: Path) -> None:
    from dao_ai.mcp.config import load_app_config
    from dao_ai.mcp.generate import write_mcp_bundle

    with mcp_config(tmp_path) as path:
        config = load_app_config(path)
    out = tmp_path / "out"
    write_mcp_bundle(config, out, force=True)

    expected = {
        "databricks.yml",
        "app.yaml",
        "pyproject.toml",
        "README.md",
    }
    produced = {p.name for p in out.iterdir() if p.is_file()}
    assert expected.issubset(produced)
    assert "requirements.txt" not in produced, (
        "MCP bundle must not ship requirements.txt — its presence would "
        "force Apps onto the legacy pip-install path and skip native uv."
    )
    assert "uv.lock" not in produced, (
        "MCP bundle must not ship uv.lock — lock is user-owned and "
        "produced by `uv sync` after generate-mcp."
    )

    pyproject = (out / "pyproject.toml").read_text()
    assert "dao-ai[mcp]" in pyproject
    assert 'dao-ai-mcp-server = "dao_ai.mcp.server:main"' in pyproject

    databricks = (out / "databricks.yml").read_text()
    assert "engine: direct" in databricks
    assert "apps:" in databricks

    app_yaml = (out / "app.yaml").read_text()
    # Bare console-script command — no `uv run` wrapper. Apps' native uv
    # BUILD installs the console script into .venv/bin/ for the runtime.
    assert "dao-ai-mcp-server" in app_yaml
    assert "uv" not in app_yaml, (
        f"app.yaml must not reference `uv` in the runtime command; got:\n{app_yaml}"
    )


@pytest.mark.unit
def test_write_mcp_bundle_rejects_config_without_recognized_tools(
    tmp_path: Path,
) -> None:
    """A config whose tools all use unrecognized factories should not generate."""
    from dao_ai.mcp.config import load_app_config
    from dao_ai.mcp.generate import write_mcp_bundle

    yaml_text = """
parameters:
  catalog:
    default: test
tools:
  current_time:
    name: current_time
    function:
      type: python
      name: dao_ai.tools.current_time_tool
"""
    src = tmp_path / "src.yaml"
    src.write_text(yaml_text)

    config = load_app_config(str(src))
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="No MCP-registerable tools"):
        write_mcp_bundle(config, out, force=True)
