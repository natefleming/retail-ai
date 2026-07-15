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

    # Layout mirrors generate-bundle: databricks.yaml + resources/app.yml
    # (no standalone app.yaml — the App's runtime command/env is embedded
    # in the DAB resource via generate_resources_app_yaml).
    expected = {
        "databricks.yaml",
        "pyproject.toml",
        "requirements.txt",
        ".gitignore",
        ".python-version",
        "README.md",
    }
    top_level_files = {p.name for p in out.iterdir() if p.is_file()}
    assert expected.issubset(top_level_files)
    assert (out / "resources" / "app.yml").is_file(), (
        "MCP bundle must emit resources/app.yml (the shared bundle "
        "layout) so downstream `bundle run` picks up the App resource."
    )
    assert not (out / "app.yaml").exists(), (
        "MCP bundle must NOT ship a standalone app.yaml — command/env are "
        "embedded in the DAB `apps.<key>.config` block via the shared "
        "generate_resources_app_yaml helper."
    )
    assert "uv.lock" not in top_level_files, (
        "MCP bundle must not ship uv.lock — Apps' build phase installs "
        "directly from requirements.txt; a shipped lock would force the "
        "legacy uv-sync path and re-introduce pypi-proxy URL pain."
    )

    pyproject = (out / "pyproject.toml").read_text()
    assert "dao-ai[mcp]" in pyproject
    assert "[project.scripts]" not in pyproject, (
        "Generated pyproject must not declare the dao-ai-mcp-server console "
        "script — the DAB invokes the module via `python -m` and doesn't "
        "need the script wired into .venv/bin/."
    )

    requirements = (out / "requirements.txt").read_text()
    # The version pin must be unbounded: the locally-installed dao-ai
    # (`_get_dao_ai_version()`) may be an unreleased pre-publish build,
    # so floor-pinning would cause Apps to fail with ``Could not find a
    # version that satisfies …`` at build time.
    assert requirements.strip() == "dao-ai[mcp]", (
        f"requirements.txt must install unbounded dao-ai[mcp]; got:\n{requirements}"
    )

    databricks = (out / "databricks.yaml").read_text()
    assert "engine: direct" in databricks
    assert "include:" in databricks and "resources/*.yml" in databricks, (
        "databricks.yaml must include the resources/*.yml split so "
        "resources/app.yml is picked up on deploy."
    )
    # MCP should NOT emit an artifacts: block — there's no user wheel to
    # build; deps install from requirements.txt.
    assert "artifacts:" not in databricks, (
        "MCP bundle must not declare an artifacts: block; there's no "
        "local wheel to build. Got:\n" + databricks
    )

    app_yml = (out / "resources" / "app.yml").read_text()
    # PATH-independent invocation: `python -m dao_ai.mcp.server` resolves
    # via the venv Python regardless of whether .venv/bin/ is on PATH in
    # the Apps runtime container. Mirrors dao_ai.apps.bundle._build_app_block.
    assert "python" in app_yml and "dao_ai.mcp.server" in app_yml
    assert "dao-ai-mcp-server" not in app_yml, (
        f"resources/app.yml must not depend on the bare console script; got:\n{app_yml}"
    )
    # DABs-embedded env uses snake_case `value_from`, not `valueFrom`.
    assert "value_from: experiment" in app_yml, (
        "MLFLOW_EXPERIMENT_ID must be sourced via DABs `value_from: "
        "experiment` so the platform substitutes the experiment id at "
        "deploy time (mirrors generate-bundle). Got:\n" + app_yml
    )
    # Config env var is unified with bundle: DAO_AI_CONFIG_PATH (not
    # DAO_AI_MCP_CONFIG_PATH). mcp/server.py reads the same name.
    assert "DAO_AI_CONFIG_PATH" in app_yml, (
        "MCP server reads DAO_AI_CONFIG_PATH (unified with bundle); the "
        "generated resources/app.yml must set it. Got:\n" + app_yml
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
    """Experiment provisioning must mirror generate-bundle exactly.

    After the shared-helper refactor, the App + experiment block lives in
    ``resources/app.yml`` (snake_case ``value_from``), not a standalone
    ``app.yaml``. Asserts the experiment is declared as a top-level
    resource, bound to the App, and wired via ``MLFLOW_EXPERIMENT_ID:
    value_from: experiment``. Trace-location plumbing asserts unchanged.
    """
    from dao_ai.mcp.config import load_app_config
    from dao_ai.mcp.generate import write_mcp_bundle

    with mcp_config(tmp_path) as path:
        config = load_app_config(path, initialize=False)
    out = tmp_path / "out"
    write_mcp_bundle(config, out, force=True)

    app_yml = (out / "resources" / "app.yml").read_text()

    # Experiment resource declared at top-level + bound in the App.
    assert "experiments:" in app_yml, (
        f"expected top-level experiments: block; got:\n{app_yml}"
    )
    assert "mcp-dao-ai-test-experiment" in app_yml
    assert "name: experiment" in app_yml
    assert "${resources.experiments.mcp-dao-ai-test-experiment.id}" in app_yml

    # env var wired via `value_from` (snake_case — DABs translates to the
    # camelCase form when it renders the platform-side app.yaml).
    assert "MLFLOW_EXPERIMENT_ID" in app_yml
    assert "value_from: experiment" in app_yml, (
        f"expected value_from: experiment for MLFLOW_EXPERIMENT_ID; got:\n{app_yml}"
    )

    # trace_location is set in the fixture — expect warehouse env var and
    # the warehouse bound as an App resource. The fixture's
    # `resources.warehouses.default` already binds `wh-test`, so
    # `_extract_raw_trace_location_resources` de-dupes and doesn't emit a
    # second `trace_warehouse` — we just verify the id is present as a
    # sql_warehouse resource.
    assert "MLFLOW_TRACING_SQL_WAREHOUSE_ID" in app_yml
    assert "wh-test" in app_yml
    assert "sql_warehouse" in app_yml and "id: wh-test" in app_yml
