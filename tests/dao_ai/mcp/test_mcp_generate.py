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
        config = load_app_config(path, initialize=False)
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
def test_write_mcp_bundle_readme_names_the_agent_tool(tmp_path: Path) -> None:
    """README must advertise the single tool derived from `app.name`."""
    from dao_ai.mcp.config import load_app_config
    from dao_ai.mcp.generate import write_mcp_bundle

    with mcp_config(tmp_path) as path:
        config = load_app_config(path, initialize=False)
    out = tmp_path / "out"
    write_mcp_bundle(config, out, force=True)

    readme = (out / "README.md").read_text()
    # `app.name: mcp-dao-ai-test` in the fixture → slugified tool name.
    assert "mcp_dao_ai_test" in readme
    assert "Test MCP agent server" in readme


@pytest.mark.unit
def test_write_mcp_bundle_requires_app_name(tmp_path: Path) -> None:
    """A config without `app.name` should error early."""
    from dao_ai.mcp.generate import write_mcp_bundle

    class _StubConfig:
        app = None
        parameters: dict = {}

    with pytest.raises(ValueError, match="config.app.name"):
        write_mcp_bundle(_StubConfig(), tmp_path / "out", force=True)


@pytest.mark.unit
def test_write_mcp_bundle_wires_mlflow_experiment(tmp_path: Path) -> None:
    """Phase 2 Change 1: experiment provisioning must mirror generate-bundle.

    Asserts that the emitted DAB declares an experiment in the top-level
    ``experiments:`` block, binds it as an App resource, and injects
    ``MLFLOW_EXPERIMENT_ID: value_from: experiment`` into the app.yaml env
    block. When ``app.trace_location`` is set, also asserts the trace
    warehouse binding + ``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` env var.
    """
    from dao_ai.mcp.config import load_app_config
    from dao_ai.mcp.generate import write_mcp_bundle

    with mcp_config(tmp_path) as path:
        config = load_app_config(path, initialize=False)
    out = tmp_path / "out"
    write_mcp_bundle(config, out, force=True)

    databricks = (out / "databricks.yml").read_text()
    app_yaml = (out / "app.yaml").read_text()

    # Experiment resource declared at top-level + bound in the App.
    assert "experiments:" in databricks, (
        f"expected top-level experiments: block; got:\n{databricks}"
    )
    assert "mcp-dao-ai-test-experiment" in databricks
    assert "name: experiment" in databricks
    assert "${resources.experiments.mcp-dao-ai-test-experiment.id}" in databricks

    # env var wired via valueFrom (camelCase — Apps runtime consumes this
    # file directly; snake_case only works in DABs resources blocks).
    assert "MLFLOW_EXPERIMENT_ID" in app_yaml
    assert "valueFrom: experiment" in app_yaml, (
        f"expected valueFrom: experiment for MLFLOW_EXPERIMENT_ID; got:\n{app_yaml}"
    )

    # trace_location is set in the fixture — expect warehouse env var and
    # the warehouse bound as an App resource. The fixture's
    # `resources.warehouses.default` already binds `wh-test`, so
    # `_extract_raw_trace_location_resources` de-dupes and doesn't emit a
    # second `trace_warehouse` — we just verify the id is present as a
    # sql_warehouse resource.
    assert "MLFLOW_TRACING_SQL_WAREHOUSE_ID" in app_yaml
    assert "wh-test" in app_yaml
    assert "sql_warehouse" in databricks and "id: wh-test" in databricks
