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
        config = load_app_config(path)

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
def test_server_name_falls_back_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dao_ai.mcp.config import (
        DEFAULT_SERVER_NAME,
        load_app_config,
        server_name_for,
    )

    monkeypatch.delenv("DAO_AI_MCP_SERVER_NAME", raising=False)
    with mcp_config(tmp_path) as path:
        config = load_app_config(path)
    # No `app:` block in the sample config + no env override => package default.
    assert server_name_for(config) == DEFAULT_SERVER_NAME


@pytest.mark.unit
def test_server_name_honors_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dao_ai.mcp.config import load_app_config, server_name_for

    monkeypatch.setenv("DAO_AI_MCP_SERVER_NAME", "custom-server")
    with mcp_config(tmp_path) as path:
        config = load_app_config(path)
    assert server_name_for(config) == "custom-server"


@pytest.mark.unit
def test_log_level_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dao_ai.mcp.config import DEFAULT_LOG_LEVEL, load_app_config, log_level_for

    monkeypatch.delenv("DAO_AI_MCP_LOG_LEVEL", raising=False)
    with mcp_config(tmp_path) as path:
        config = load_app_config(path)
    assert log_level_for(config) == DEFAULT_LOG_LEVEL


@pytest.mark.unit
def test_app_block_with_mcp_only_drives_server_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``app: { name, mcp_only: true }`` controls both server_name and log_level
    without needing agents or registered_model — proves the AppModel escape
    hatch works and that the MCP server reads ``app.name``."""
    from dao_ai.mcp.config import load_app_config, log_level_for, server_name_for

    yaml_text = """
parameters:
  warehouse_id: { description: warehouse }

app:
  name: mcp-merchandising-analytics
  description: Test MCP server.
  log_level: WARNING
  mcp_only: true

resources:
  warehouses:
    wh: &wh
      warehouse_id: ${var.warehouse_id}
  genie_rooms:
    room: &room
      space_id: 01f00000000000000000000000000001

tools:
  ask_merch:
    name: ask_merch
    function:
      type: factory
      name: dao_ai.tools.create_genie_toolkit
      args:
        name: ask_merch
        description: test
        genie_room: *room
        lru_cache_parameters:
          warehouse: *wh
          capacity: 5
          time_to_live_seconds: 60
"""
    path = tmp_path / "with_app.yaml"
    path.write_text(yaml_text)
    monkeypatch.setenv("WAREHOUSE_ID", "wh-test")
    monkeypatch.delenv("DAO_AI_MCP_SERVER_NAME", raising=False)
    monkeypatch.delenv("DAO_AI_MCP_LOG_LEVEL", raising=False)

    config = load_app_config(str(path))
    assert server_name_for(config) == "mcp-merchandising-analytics"
    assert log_level_for(config) == "WARNING"
    assert config.app is not None
    assert config.app.mcp_only is True
