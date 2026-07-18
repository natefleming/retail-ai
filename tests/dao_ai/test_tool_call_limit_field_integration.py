"""
Integration tests for the `call_limit` tool field against live Databricks endpoints.

These tests load tool_call_limit_field.yaml end-to-end, create a ResponsesAgent,
and validate that a tool's `call_limit` auto-registers a ToolCallLimitMiddleware
on every agent that uses the tool — without any per-agent middleware wiring.

Run with:
    pytest tests/dao_ai/test_tool_call_limit_field_integration.py -v -m integration -s
"""

import sys
from pathlib import Path

import pytest
from conftest import has_databricks_env
from langchain.agents.middleware import ToolCallLimitMiddleware

from dao_ai.config import AppConfig
from dao_ai.middleware.tool_call_limit import (
    create_tool_call_limit_middlewares_from_tool_models,
)
from dao_ai.models import ResponsesAgent

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def call_limit_config_path() -> Path:
    """Path to the tool_call_limit_field.yaml example configuration."""
    return (
        Path(__file__).parents[2]
        / "config"
        / "examples"
        / "12_middleware"
        / "tool_call_limit_field.yaml"
    )


@pytest.fixture
def app_config(call_limit_config_path: Path) -> AppConfig:
    """AppConfig loaded from the call_limit example."""
    return AppConfig.from_file(call_limit_config_path)


# =============================================================================
# Test Cases
# =============================================================================


@pytest.mark.integration
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_call_limit_config_loads_and_registers_middleware(
    app_config: AppConfig,
) -> None:
    """
    Config loads, and each agent using a call_limited tool derives a
    ToolCallLimitMiddleware from the tool's call_limit — even though no agent
    declares any tool-call-limit middleware explicitly.
    """
    assert app_config is not None
    assert app_config.app is not None
    assert len(app_config.agents) >= 2

    for agent in app_config.agents.values():
        # No explicit tool-call-limit middleware on the agent.
        limit_mws = create_tool_call_limit_middlewares_from_tool_models(agent.tools)
        limited = {mw.tool_name for mw in limit_mws}
        print(f"Agent '{agent.name}' auto-limited tools: {limited}", file=sys.stderr)

        # Every agent in this example uses genie_tool, which is call-limited.
        assert any(isinstance(m, ToolCallLimitMiddleware) for m in limit_mws), (
            f"Agent '{agent.name}' should inherit a tool call limit"
        )
        # Genie tool object-form limit: run_limit=2, thread_limit=8, error.
        genie_mw = next(
            (mw for mw in limit_mws if mw.tool_name == "retail_genie_tool"), None
        )
        assert genie_mw is not None
        assert genie_mw.run_limit == 2
        assert genie_mw.thread_limit == 8
        assert genie_mw.exit_behavior == "error"


@pytest.mark.integration
@pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
def test_call_limit_agent_builds_with_middleware(app_config: AppConfig) -> None:
    """
    The ResponsesAgent builds successfully with call-limit middleware attached,
    confirming the wiring path (nodes._create_middleware_list) runs end-to-end.
    """
    responses_agent: ResponsesAgent = app_config.as_responses_agent()
    assert responses_agent is not None
    assert hasattr(responses_agent, "predict")
    print(
        f"Successfully created ResponsesAgent: {type(responses_agent)}",
        file=sys.stderr,
    )
