from typing import Sequence
from unittest.mock import patch

import pytest

from dao_ai.config import (
    AppConfig,
    FunctionType,
    PythonFunctionModel,
    SearchToolModel,
    ToolModel,
)
from dao_ai.tools import create_tools, resolve_tool_names
from dao_ai.tools.core import tool_registry

excluded_tools: Sequence[str] = [
    "vector_search",
    "genie",
    "weather",  # MCP tool that requires external server
]


@pytest.mark.unit
def test_create_tools(config: AppConfig) -> None:
    """Test that tools can be created from configuration."""
    tool_models: list[ToolModel] = config.find_tools(
        lambda tool: (
            not any(excluded in tool.name for excluded in excluded_tools)
            and tool.function.type != FunctionType.UNITY_CATALOG
        )
    )

    tools = create_tools(tool_models)

    assert tools is not None


@pytest.mark.unit
def test_create_tools_empty_list() -> None:
    """Test that create_tools handles empty list."""
    tools = create_tools([])
    assert tools is not None
    assert len(tools) == 0


@pytest.mark.unit
def test_obo_mcp_tool_discovery_failure_is_skipped_not_fatal() -> None:
    """An OBO MCP tool whose discovery (tools/list) fails at build time is
    skipped with a warning, so the rest of the agent still loads."""
    from dao_ai.config import McpFunctionModel

    tool_registry.clear()
    fn = McpFunctionModel(service="system.ai.atlassian", on_behalf_of_user=True)
    tm = ToolModel(name="atlassian_mcp", function=fn)
    with patch(
        "dao_ai.tools.core.create_hooks",
        side_effect=RuntimeError("Failed to list MCP tools ... please login first"),
    ):
        tools = create_tools([tm])
    assert tools == []  # skipped, no exception
    tool_registry.clear()


@pytest.mark.unit
def test_obo_mcp_tool_non_auth_failure_still_raises() -> None:
    """An OBO MCP tool that fails for a NON-auth reason (bug, network fault) must
    still raise — only auth/discovery (login/credential/403) failures are skipped."""
    from dao_ai.config import McpFunctionModel

    tool_registry.clear()
    fn = McpFunctionModel(service="system.ai.microsoft_365", on_behalf_of_user=True)
    tm = ToolModel(name="ms365_mcp", function=fn)
    with patch(
        "dao_ai.tools.core.create_hooks",
        side_effect=RuntimeError("connection refused while building tool"),
    ):
        with pytest.raises(RuntimeError):
            create_tools([tm])
    tool_registry.clear()


@pytest.mark.unit
def test_obo_mcp_number_containing_403_is_not_auth_and_raises() -> None:
    """A non-auth failure whose message merely CONTAINS '403'/'401' as part of an
    unrelated number (e.g. a timeout '4030ms') must still raise — the auth-status
    match is word-boundary'd, so it is not misclassified as an auth-discovery skip."""
    from dao_ai.config import McpFunctionModel

    tool_registry.clear()
    fn = McpFunctionModel(service="system.ai.atlassian", on_behalf_of_user=True)
    tm = ToolModel(name="atlassian_mcp", function=fn)
    with patch(
        "dao_ai.tools.core.create_hooks",
        side_effect=RuntimeError("Read timeout after 4030ms building tool"),
    ):
        with pytest.raises(RuntimeError):
            create_tools([tm])
    tool_registry.clear()


@pytest.mark.unit
def test_non_obo_mcp_tool_discovery_failure_still_raises() -> None:
    """A non-OBO MCP tool that fails discovery is a genuine misconfiguration and
    must still raise (no silent skip)."""
    from dao_ai.config import McpFunctionModel

    tool_registry.clear()
    fn = McpFunctionModel(url="https://example.com/mcp", on_behalf_of_user=False)
    tm = ToolModel(name="broken_mcp", function=fn)
    with patch(
        "dao_ai.tools.core.create_hooks",
        side_effect=RuntimeError("connection refused"),
    ):
        with pytest.raises(RuntimeError):
            create_tools([tm])
    tool_registry.clear()


class TestResolveToolNames:
    """Tests for the shared resolve_tool_names helper (registry reuse)."""

    @pytest.fixture(autouse=True)
    def _clear_registry(self):
        """Isolate each test from registry state left by other tests."""
        tool_registry.clear()
        yield
        tool_registry.clear()

    @pytest.mark.unit
    def test_registry_miss_falls_back_to_as_tools(self) -> None:
        """With an empty registry, names come from function.as_tools()."""
        tm = ToolModel(name="search", function=SearchToolModel())
        assert resolve_tool_names(tm) == ["duckduckgo_search"]

    @pytest.mark.unit
    def test_registry_hit_reuses_without_calling_as_tools(self) -> None:
        """After create_tools populates the registry, no second as_tools()."""
        tm = ToolModel(name="search", function=SearchToolModel())
        create_tools([tm])  # populates tool_registry["search"]

        with patch.object(
            type(tm.function), "as_tools", wraps=tm.function.as_tools
        ) as spy:
            names = resolve_tool_names(tm)

        assert names == ["duckduckgo_search"]
        assert spy.call_count == 0, "should reuse registry, not rebuild tools"

    @pytest.mark.unit
    def test_string_function_falls_back_to_tool_model_name(self) -> None:
        """A bare string function reference resolves to the ToolModel name."""
        tm = ToolModel(name="some_reference", function="some_reference")
        assert resolve_tool_names(tm) == ["some_reference"]

    @pytest.mark.unit
    def test_as_tools_exception_falls_back_to_tool_model_name(self) -> None:
        """If as_tools() raises, resolution falls back to the ToolModel name."""
        tm = ToolModel(
            name="boom",
            function=PythonFunctionModel(name="dao_ai.tools.current_time_tool"),
        )
        with patch.object(
            type(tm.function), "as_tools", side_effect=RuntimeError("no server")
        ):
            assert resolve_tool_names(tm) == ["boom"]

    @pytest.mark.unit
    def test_python_tool_resolves_runtime_name(self) -> None:
        """A python function resolves to the underlying tool's runtime name."""
        tm = ToolModel(
            name="clock",
            function=PythonFunctionModel(name="dao_ai.tools.current_time_tool"),
        )
        assert resolve_tool_names(tm) == ["current_time_tool"]
