"""Unit tests for dao_ai.mcp.config."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _fixtures import mcp_config  # noqa: E402


@pytest.mark.unit
def test_load_app_config_substitutes_vars(tmp_path: Path) -> None:
    from dao_ai.mcp.config import load_app_config

    with mcp_config(tmp_path) as path:
        config = load_app_config(path, initialize=False)

    assert set(config.tools.keys()) == {
        "product_vector_search",
        "ask_retail",
        "ask_inventory",
    }
    retail = config.resources.genie_rooms["retail"]
    assert str(retail.space_id) == "01f00000000000000000000000000001"

    db = config.resources.databases["lakebase"]
    assert db.project == "test-lakebase"
    assert db.branch == "production"


@pytest.mark.unit
def test_server_name_reads_app_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dao_ai.mcp.config import load_app_config, server_name_for

    monkeypatch.delenv("DAO_AI_MCP_SERVER_NAME", raising=False)
    with mcp_config(tmp_path) as path:
        config = load_app_config(path, initialize=False)
    # The fixture declares `app.name: mcp-dao-ai-test`.
    assert server_name_for(config) == "mcp-dao-ai-test"


@pytest.mark.unit
def test_server_name_honors_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dao_ai.mcp.config import load_app_config, server_name_for

    monkeypatch.setenv("DAO_AI_MCP_SERVER_NAME", "custom-server")
    with mcp_config(tmp_path) as path:
        config = load_app_config(path, initialize=False)
    assert server_name_for(config) == "custom-server"


@pytest.mark.unit
def test_log_level_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dao_ai.mcp.config import load_app_config, log_level_for

    monkeypatch.delenv("DAO_AI_MCP_LOG_LEVEL", raising=False)
    with mcp_config(tmp_path) as path:
        config = load_app_config(path, initialize=False)
    # Fixture sets `app.log_level: WARNING`.
    assert log_level_for(config) == "WARNING"
