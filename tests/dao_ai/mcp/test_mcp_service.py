"""Unit tests for dao_ai.mcp.service.register_tools_from_config."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _fixtures import mcp_config  # noqa: E402


@pytest.mark.unit
def test_register_tools_from_config_advertises_expected_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All recognized tools become MCP tools; Genie adds a feedback companion."""
    from mcp.server.fastmcp import FastMCP

    from dao_ai.mcp.adapters import genie as genie_module
    from dao_ai.mcp.adapters import vector_search as vs_module
    from dao_ai.mcp.config import load_app_config
    from dao_ai.mcp.service import register_tools_from_config

    with mcp_config(tmp_path) as path:
        config = load_app_config(path)

    # Stub the VS factory + Genie cache chain so we don't hit Databricks.
    fake_lc_tool = MagicMock(name="fake_langchain_tool")
    fake_lc_tool.name = "product_vector_search"
    fake_lc_tool.description = "Search the product catalog."
    monkeypatch.setattr(
        vs_module, "create_vector_search_tool", lambda **kw: fake_lc_tool
    )

    stub_service = MagicMock()
    stub_service.initialize.return_value = stub_service
    monkeypatch.setattr(
        genie_module, "PostgresContextAwareGenieService", lambda **kw: stub_service
    )
    monkeypatch.setattr(genie_module, "LRUCacheService", lambda **kw: stub_service)
    monkeypatch.setattr(genie_module, "GenieService", lambda **kw: stub_service)
    monkeypatch.setattr(genie_module, "Genie", lambda **kw: MagicMock())

    mcp = FastMCP("dao-ai-mcp-test", stateless_http=True, json_response=True)
    registered = register_tools_from_config(mcp, config, workspace_client=MagicMock())

    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}

    # Two genie toolkits → two query tools + two feedback companions.
    # One VS factory → one tool. Total: 5 MCP tools.
    assert names == {
        "product_vector_search",
        "ask_retail",
        "ask_retail_feedback",
        "ask_inventory",
        "ask_inventory_feedback",
    }
    assert registered == {"product_vector_search", "ask_retail", "ask_inventory"}


@pytest.mark.unit
def test_register_tools_fails_when_no_adapter_recognized(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config with no recognized factories raises a clear error."""
    from mcp.server.fastmcp import FastMCP

    from dao_ai.mcp.config import load_app_config
    from dao_ai.mcp.service import register_tools_from_config

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
    path = tmp_path / "no_mcp_tools.yaml"
    path.write_text(yaml_text)

    config = load_app_config(str(path))
    mcp = FastMCP("test", stateless_http=True, json_response=True)

    with pytest.raises(ValueError, match="no MCP-registerable tools"):
        register_tools_from_config(mcp, config, workspace_client=MagicMock())
