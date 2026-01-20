"""
Integration tests for MCP tool filtering and list_mcp_tools.

Tests the end-to-end filtering behavior with mock MCP servers.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from mcp.types import Tool

from dao_ai.config import DatabricksAppModel, McpFunctionModel
from dao_ai.tools.mcp import (
    MCPToolInfo,
    _build_connection_config,
    create_mcp_tools,
    list_mcp_tools,
)


@pytest.fixture
def mock_mcp_tools():
    """Mock tools returned from an MCP server."""
    return [
        Tool(name="query_sales", description="Query sales data", inputSchema={}),
        Tool(
            name="query_inventory", description="Query inventory data", inputSchema={}
        ),
        Tool(name="list_tables", description="List all tables", inputSchema={}),
        Tool(name="list_schemas", description="List all schemas", inputSchema={}),
        Tool(name="get_table_schema", description="Get table schema", inputSchema={}),
        Tool(name="drop_table", description="Drop a table", inputSchema={}),
        Tool(name="delete_record", description="Delete a record", inputSchema={}),
        Tool(name="execute_ddl", description="Execute DDL statement", inputSchema={}),
        Tool(name="admin_user", description="Admin tool", inputSchema={}),
        Tool(
            name="query_sensitive", description="Query sensitive data", inputSchema={}
        ),
    ]


def _setup_mock_client(mock_client_class, mock_tools):
    """Helper to setup mock MCP client."""
    mock_session = AsyncMock()
    mock_session.list_tools.return_value = MagicMock(tools=mock_tools)
    mock_session.call_tool = AsyncMock(return_value=MagicMock(content=[]))

    mock_client = MagicMock()
    mock_client.session.return_value.__aenter__.return_value = mock_session
    mock_client_class.return_value = mock_client

    return mock_client


class TestMCPFilteringIntegration:
    """Integration tests for MCP tool filtering."""

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_no_filters_loads_all_tools(self, mock_client_class, mock_mcp_tools):
        """Test that all tools are loaded when no filters are specified."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
        )

        tools = create_mcp_tools(function)

        # Should load all 10 tools
        assert len(tools) == 10
        tool_names = [t.name for t in tools]
        assert "query_sales" in tool_names
        assert "drop_table" in tool_names
        assert "execute_ddl" in tool_names

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_include_tools_exact_names(self, mock_client_class, mock_mcp_tools):
        """Test include_tools with exact tool names."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            include_tools=["query_sales", "list_tables"],
        )

        tools = create_mcp_tools(function)

        # Should only load 2 specified tools
        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "query_sales" in tool_names
        assert "list_tables" in tool_names
        assert "drop_table" not in tool_names

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_include_tools_with_patterns(self, mock_client_class, mock_mcp_tools):
        """Test include_tools with glob patterns."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            include_tools=["query_*", "list_*"],
        )

        tools = create_mcp_tools(function)

        # Should load query_* and list_* tools (5 total: 3 query, 2 list)
        assert len(tools) == 5
        tool_names = [t.name for t in tools]
        assert "query_sales" in tool_names
        assert "query_inventory" in tool_names
        assert "query_sensitive" in tool_names  # Also matches query_*
        assert "list_tables" in tool_names
        assert "list_schemas" in tool_names
        assert "drop_table" not in tool_names

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_exclude_tools_exact_names(self, mock_client_class, mock_mcp_tools):
        """Test exclude_tools with exact tool names."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            exclude_tools=["drop_table", "delete_record", "execute_ddl"],
        )

        tools = create_mcp_tools(function)

        # Should load all except the 3 excluded (7 total)
        assert len(tools) == 7
        tool_names = [t.name for t in tools]
        assert "query_sales" in tool_names
        assert "list_tables" in tool_names
        assert "drop_table" not in tool_names
        assert "delete_record" not in tool_names
        assert "execute_ddl" not in tool_names

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_exclude_tools_with_patterns(self, mock_client_class, mock_mcp_tools):
        """Test exclude_tools with glob patterns."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            exclude_tools=["drop_*", "delete_*", "execute_*", "admin_*"],
        )

        tools = create_mcp_tools(function)

        # Should exclude 4 tools matching patterns (6 remaining)
        assert len(tools) == 6
        tool_names = [t.name for t in tools]
        assert "query_sales" in tool_names
        assert "list_tables" in tool_names
        assert "drop_table" not in tool_names
        assert "delete_record" not in tool_names
        assert "execute_ddl" not in tool_names
        assert "admin_user" not in tool_names

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_hybrid_include_and_exclude(self, mock_client_class, mock_mcp_tools):
        """Test combining include_tools and exclude_tools."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            include_tools=["query_*", "list_*"],
            exclude_tools=["*_sensitive"],  # Exclude sensitive queries
        )

        tools = create_mcp_tools(function)

        # Should load query_* and list_* except query_sensitive (4 total: 2 query + 2 list)
        assert len(tools) == 4
        tool_names = [t.name for t in tools]
        assert "query_sales" in tool_names
        assert "query_inventory" in tool_names
        assert "list_tables" in tool_names
        assert "list_schemas" in tool_names
        assert "query_sensitive" not in tool_names  # Excluded by pattern

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_exclude_overrides_include(self, mock_client_class, mock_mcp_tools):
        """Test that exclude_tools takes precedence over include_tools."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            include_tools=["query_*"],  # Include all query tools
            exclude_tools=["*_sensitive"],  # But exclude sensitive ones
        )

        tools = create_mcp_tools(function)

        # query_sensitive matches both patterns, exclude wins
        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "query_sales" in tool_names
        assert "query_inventory" in tool_names
        assert "query_sensitive" not in tool_names

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_real_world_sql_safe_filtering(self, mock_client_class, mock_mcp_tools):
        """Test real-world scenario: SQL server with read-only filtering."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            include_tools=["query_*", "list_*", "get_*"],
            exclude_tools=["*_sensitive"],
        )

        tools = create_mcp_tools(function)

        # Should have: query_sales, query_inventory, list_tables,
        # list_schemas, get_table_schema (5 total)
        assert len(tools) == 5
        tool_names = [t.name for t in tools]

        # Read operations included
        assert "query_sales" in tool_names
        assert "query_inventory" in tool_names
        assert "list_tables" in tool_names
        assert "get_table_schema" in tool_names

        # Write operations excluded
        assert "drop_table" not in tool_names
        assert "delete_record" not in tool_names
        assert "execute_ddl" not in tool_names

        # Sensitive data excluded
        assert "query_sensitive" not in tool_names

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_real_world_block_dangerous_operations(
        self, mock_client_class, mock_mcp_tools
    ):
        """Test real-world scenario: Block dangerous operations only."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            # No include list - allow everything except dangerous ops
            exclude_tools=["drop_*", "delete_*", "execute_*", "admin_*"],
        )

        tools = create_mcp_tools(function)

        # Should exclude drop, delete, execute, admin (4 excluded, 6 remaining)
        assert len(tools) == 6
        tool_names = [t.name for t in tools]

        # Safe operations included
        assert "query_sales" in tool_names
        assert "list_tables" in tool_names

        # Dangerous operations excluded
        assert "drop_table" not in tool_names
        assert "delete_record" not in tool_names
        assert "execute_ddl" not in tool_names
        assert "admin_user" not in tool_names

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_no_matching_tools_returns_empty(self, mock_client_class, mock_mcp_tools):
        """Test that filtering to zero tools returns empty list."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            include_tools=["nonexistent_*"],  # No tools match
        )

        tools = create_mcp_tools(function)

        # Should return empty list
        assert len(tools) == 0


class TestListMCPTools:
    """Tests for the list_mcp_tools function."""

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_returns_mcp_tool_info_list(self, mock_client_class, mock_mcp_tools):
        """Test that list_mcp_tools returns a list of MCPToolInfo."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
        )

        tool_infos = list_mcp_tools(function)

        # Should return MCPToolInfo instances
        assert len(tool_infos) == 10
        assert all(isinstance(t, MCPToolInfo) for t in tool_infos)

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_tool_info_contains_expected_fields(
        self, mock_client_class, mock_mcp_tools
    ):
        """Test that MCPToolInfo contains name, description, and input_schema."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
        )

        tool_infos = list_mcp_tools(function)

        # Find query_sales
        query_sales = next(t for t in tool_infos if t.name == "query_sales")

        assert query_sales.name == "query_sales"
        assert query_sales.description == "Query sales data"
        assert query_sales.input_schema == {}

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_apply_filters_true_respects_filters(
        self, mock_client_class, mock_mcp_tools
    ):
        """Test that filters are applied when apply_filters=True (default)."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            include_tools=["query_*"],
        )

        tool_infos = list_mcp_tools(function, apply_filters=True)

        # Should only return query_* tools
        assert len(tool_infos) == 3  # query_sales, query_inventory, query_sensitive
        assert all(t.name.startswith("query_") for t in tool_infos)

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_apply_filters_false_ignores_filters(
        self, mock_client_class, mock_mcp_tools
    ):
        """Test that filters are ignored when apply_filters=False."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            include_tools=["query_*"],  # Would normally filter to 3 tools
            exclude_tools=["*"],  # Would normally exclude all
        )

        tool_infos = list_mcp_tools(function, apply_filters=False)

        # Should return ALL tools regardless of filters
        assert len(tool_infos) == 10
        tool_names = [t.name for t in tool_infos]
        assert "query_sales" in tool_names
        assert "drop_table" in tool_names

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_to_dict_for_serialization(self, mock_client_class, mock_mcp_tools):
        """Test that MCPToolInfo can be serialized to dict."""
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
        )

        tool_infos = list_mcp_tools(function)
        query_sales = next(t for t in tool_infos if t.name == "query_sales")

        # Convert to dict for JSON serialization
        tool_dict = query_sales.to_dict()

        assert tool_dict == {
            "name": "query_sales",
            "description": "Query sales data",
            "input_schema": {},
        }

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_ui_use_case_discover_all_then_user_selects(
        self, mock_client_class, mock_mcp_tools
    ):
        """
        Test UI use case: discover all tools, user selects some.

        Simulates the flow where:
        1. UI calls list_mcp_tools with apply_filters=False to show all options
        2. User selects specific tools
        3. Final config uses those selections as include_tools
        """
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        # Step 1: Discover all tools (for UI display)
        discovery_function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
        )
        all_tools = list_mcp_tools(discovery_function, apply_filters=False)

        # UI presents all 10 tools to user
        assert len(all_tools) == 10
        available_names = [t.name for t in all_tools]

        # Step 2: User selects some tools (simulated)
        user_selected = ["query_sales", "list_tables", "get_table_schema"]
        assert all(name in available_names for name in user_selected)

        # Step 3: Create final config with user selection
        final_function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            include_tools=user_selected,
        )
        filtered_tools = list_mcp_tools(final_function)

        # Should only include user-selected tools
        assert len(filtered_tools) == 3
        assert set(t.name for t in filtered_tools) == set(user_selected)

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    def test_ui_use_case_show_what_would_be_included(
        self, mock_client_class, mock_mcp_tools
    ):
        """
        Test UI use case: preview which tools match include/exclude patterns.

        Simulates the flow where UI shows users which tools will be loaded
        based on their current filter configuration.
        """
        _setup_mock_client(mock_client_class, mock_mcp_tools)

        # User configures patterns
        function = McpFunctionModel(
            type="mcp",
            url="http://test.com/mcp",
            include_tools=["query_*", "list_*"],
            exclude_tools=["*_sensitive"],
        )

        # Get all tools (for comparison)
        all_tools = list_mcp_tools(function, apply_filters=False)
        all_names = {t.name for t in all_tools}

        # Get filtered tools (what will actually be loaded)
        filtered_tools = list_mcp_tools(function, apply_filters=True)
        filtered_names = {t.name for t in filtered_tools}

        # Calculate excluded tools for UI display
        excluded_names = all_names - filtered_names

        # Verify correct tools are included/excluded
        assert "query_sales" in filtered_names
        assert "query_inventory" in filtered_names
        assert "list_tables" in filtered_names
        assert "list_schemas" in filtered_names

        assert "query_sensitive" in excluded_names  # Explicitly excluded
        assert "drop_table" in excluded_names  # Not in include list
        assert "admin_user" in excluded_names  # Not in include list


class TestMCPToolInfo:
    """Tests for the MCPToolInfo dataclass."""

    def test_dataclass_fields(self):
        """Test that MCPToolInfo has expected fields."""
        info = MCPToolInfo(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
        )

        assert info.name == "test_tool"
        assert info.description == "A test tool"
        assert info.input_schema == {"type": "object", "properties": {}}

    def test_to_dict_with_none_description(self):
        """Test to_dict handles None description."""
        info = MCPToolInfo(
            name="test_tool",
            description=None,
            input_schema={},
        )

        result = info.to_dict()

        assert result == {
            "name": "test_tool",
            "description": None,
            "input_schema": {},
        }

    def test_to_dict_is_json_serializable(self):
        """Test that to_dict output is JSON serializable."""
        import json

        info = MCPToolInfo(
            name="test_tool",
            description="A test",
            input_schema={"type": "object"},
        )

        # Should not raise
        json_str = json.dumps(info.to_dict())
        assert "test_tool" in json_str


class TestBuildConnectionConfigWithApp:
    """Tests for _build_connection_config with DatabricksAppModel source."""

    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_app_source_uses_oauth_provider(self, mock_provider_class):
        """Test that app source uses DatabricksOAuthClientProvider with function's workspace_client."""
        from unittest.mock import PropertyMock

        # Create mock app with URL
        mock_app_instance = Mock()
        mock_app_instance.url = "https://my-mcp-app.cloud.databricks.com"

        # Create mock workspace client
        mock_ws = Mock()
        mock_ws.apps.get.return_value = mock_app_instance

        # Create DatabricksAppModel
        app_model = DatabricksAppModel(name="my-mcp-app")

        # Create McpFunctionModel with app source
        function = McpFunctionModel(app=app_model)

        # Mock the workspace_client property on McpFunctionModel
        # (since app has no auth configured, function's workspace_client is used)
        with patch.object(
            type(function),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=mock_ws,
        ):
            # Build connection config
            config = _build_connection_config(function)

            # Verify structure - app URLs get /mcp suffix
            assert config["url"] == "https://my-mcp-app.cloud.databricks.com/mcp"
            assert config["transport"] == "http"
            assert "auth" in config

            # Verify OAuth provider was called with function's workspace client
            mock_provider_class.assert_called_once_with(mock_ws)

    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_app_source_config_structure(self, mock_provider_class):
        """Test that app source returns correct config structure."""
        from unittest.mock import PropertyMock

        mock_auth = Mock()
        mock_provider_class.return_value = mock_auth

        # Create mock app
        mock_app_instance = Mock()
        mock_app_instance.url = "https://test-app.azuredatabricks.net"

        mock_ws = Mock()
        mock_ws.apps.get.return_value = mock_app_instance

        app_model = DatabricksAppModel(name="test-app")
        function = McpFunctionModel(app=app_model)

        # IMPORTANT: McpFunctionModel.url resolution tries function.workspace_client first
        # (it may have tool-level auth). Patch that, not just app_model.workspace_client.
        with patch.object(
            type(function),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=mock_ws,
        ):
            config = _build_connection_config(function)

        # Verify complete config structure - app URLs get /mcp suffix
        assert config == {
            "url": "https://test-app.azuredatabricks.net/mcp",
            "transport": "http",
            "auth": mock_auth,
        }

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_create_mcp_tools_with_app_source(
        self, mock_provider_class, mock_client_class, mock_mcp_tools
    ):
        """Test creating MCP tools with DatabricksAppModel as source."""
        from unittest.mock import PropertyMock

        # Setup mock client
        mock_session = AsyncMock()
        mock_session.list_tools.return_value = MagicMock(tools=mock_mcp_tools)
        mock_session.call_tool = AsyncMock(return_value=MagicMock(content=[]))

        mock_client = MagicMock()
        mock_client.session.return_value.__aenter__.return_value = mock_session
        mock_client_class.return_value = mock_client

        # Create mock app
        mock_app_instance = Mock()
        mock_app_instance.url = "https://my-mcp-app.cloud.databricks.com"

        mock_ws = Mock()
        mock_ws.apps.get.return_value = mock_app_instance

        app_model = DatabricksAppModel(name="my-mcp-app")

        # Mock the workspace_client property
        with patch.object(
            type(app_model),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=mock_ws,
        ):
            # Create MCP function with app source
            function = McpFunctionModel(app=app_model)

            # Create tools
            tools = create_mcp_tools(function)

            # Verify tools were created
            assert len(tools) == 10
            tool_names = [t.name for t in tools]
            assert "query_sales" in tool_names

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_list_mcp_tools_with_app_source(
        self, mock_provider_class, mock_client_class, mock_mcp_tools
    ):
        """Test listing MCP tools with DatabricksAppModel as source."""
        from unittest.mock import PropertyMock

        # Setup mock client
        mock_session = AsyncMock()
        mock_session.list_tools.return_value = MagicMock(tools=mock_mcp_tools)

        mock_client = MagicMock()
        mock_client.session.return_value.__aenter__.return_value = mock_session
        mock_client_class.return_value = mock_client

        # Create mock app
        mock_app_instance = Mock()
        mock_app_instance.url = "https://my-mcp-app.cloud.databricks.com"

        mock_ws = Mock()
        mock_ws.apps.get.return_value = mock_app_instance

        app_model = DatabricksAppModel(name="my-mcp-app")

        # Mock the workspace_client property
        with patch.object(
            type(app_model),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=mock_ws,
        ):
            # Create MCP function with app source
            function = McpFunctionModel(app=app_model)

            # List tools
            tool_infos = list_mcp_tools(function)

            # Verify tool infos
            assert len(tool_infos) == 10
            assert all(isinstance(t, MCPToolInfo) for t in tool_infos)

    @patch("dao_ai.tools.mcp.MultiServerMCPClient")
    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_app_source_with_filtering(
        self, mock_provider_class, mock_client_class, mock_mcp_tools
    ):
        """Test app source with include/exclude tool filters."""
        from unittest.mock import PropertyMock

        # Setup mock client
        mock_session = AsyncMock()
        mock_session.list_tools.return_value = MagicMock(tools=mock_mcp_tools)
        mock_session.call_tool = AsyncMock(return_value=MagicMock(content=[]))

        mock_client = MagicMock()
        mock_client.session.return_value.__aenter__.return_value = mock_session
        mock_client_class.return_value = mock_client

        # Create mock app
        mock_app_instance = Mock()
        mock_app_instance.url = "https://my-mcp-app.cloud.databricks.com"

        mock_ws = Mock()
        mock_ws.apps.get.return_value = mock_app_instance

        app_model = DatabricksAppModel(name="my-mcp-app")

        # Mock the workspace_client property
        with patch.object(
            type(app_model),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=mock_ws,
        ):
            # Create MCP function with app source and filters
            function = McpFunctionModel(
                app=app_model,
                include_tools=["query_*"],
                exclude_tools=["*_sensitive"],
            )

            # Create tools
            tools = create_mcp_tools(function)

            # Should only have query_* tools except query_sensitive
            assert len(tools) == 2
            tool_names = [t.name for t in tools]
            assert "query_sales" in tool_names
            assert "query_inventory" in tool_names
            assert "query_sensitive" not in tool_names


class TestBuildConnectionConfigUnifiedAuth:
    """Tests for unified authentication through IsDatabricksResource.workspace_client."""

    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_genie_room_uses_own_workspace_client(self, mock_provider_class):
        """Test that genie_room source uses its own workspace_client for auth when auth is configured."""
        from unittest.mock import PropertyMock

        from dao_ai.config import GenieRoomModel

        mock_auth = Mock()
        mock_provider_class.return_value = mock_auth

        # Create mock workspace client for genie room
        mock_ws = Mock()

        # Genie room must have auth configured to take priority
        genie_room = GenieRoomModel(
            name="test-genie", space_id="space_123", on_behalf_of_user=True
        )

        # Mock the workspace_client property
        with patch.object(
            type(genie_room),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=mock_ws,
        ):
            function = McpFunctionModel(
                genie_room=genie_room,
                workspace_host="https://workspace.databricks.com",
            )

            config = _build_connection_config(function)

            # Verify OAuth provider was called with genie_room's workspace client
            mock_provider_class.assert_called_once_with(mock_ws)
            assert config["transport"] == "http"
            assert config["auth"] == mock_auth

    @patch("dao_ai.providers.databricks.VectorSearchClient")
    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_vector_search_uses_own_workspace_client(
        self, mock_provider_class, mock_vsc_class
    ):
        """Test that vector_search source uses its own workspace_client for auth when auth is configured."""
        from unittest.mock import PropertyMock

        from dao_ai.config import IndexModel, SchemaModel, TableModel, VectorStoreModel

        mock_auth = Mock()
        mock_provider_class.return_value = mock_auth

        # Create mock workspace client for vector search
        mock_ws = Mock()

        schema = SchemaModel(catalog_name="catalog", schema_name="schema")
        table = TableModel(schema=schema, name="table")
        index = IndexModel(schema=schema, name="index")

        # Vector search must have auth configured to take priority
        vector_search = VectorStoreModel(
            source_table=table,
            embedding_source_column="text",
            index=index,
            primary_key="id",
            on_behalf_of_user=True,
        )

        # Mock the workspace_client property
        with patch.object(
            type(vector_search),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=mock_ws,
        ):
            function = McpFunctionModel(
                vector_search=vector_search,
                workspace_host="https://workspace.databricks.com",
            )

            config = _build_connection_config(function)

            # Verify OAuth provider was called with vector_search's workspace client
            mock_provider_class.assert_called_once_with(mock_ws)
            assert config["transport"] == "http"
            assert config["auth"] == mock_auth

    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_url_source_uses_mcpfunction_workspace_client(self, mock_provider_class):
        """Test that direct URL source uses McpFunctionModel's own workspace_client."""
        from unittest.mock import PropertyMock

        mock_auth = Mock()
        mock_provider_class.return_value = mock_auth

        # Create mock workspace client for McpFunctionModel itself
        mock_ws = Mock()

        function = McpFunctionModel(url="https://example.com/mcp")

        # Mock the workspace_client property
        with patch.object(
            type(function),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=mock_ws,
        ):
            config = _build_connection_config(function)

            # Verify OAuth provider was called with McpFunctionModel's workspace client
            mock_provider_class.assert_called_once_with(mock_ws)
            assert config["transport"] == "http"
            assert config["auth"] == mock_auth

    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_sql_source_uses_mcpfunction_workspace_client(self, mock_provider_class):
        """Test that sql=True source uses McpFunctionModel's own workspace_client."""
        from unittest.mock import PropertyMock

        mock_auth = Mock()
        mock_provider_class.return_value = mock_auth

        # Create mock workspace client for McpFunctionModel itself
        mock_ws = Mock()

        function = McpFunctionModel(
            sql=True,
            workspace_host="https://workspace.databricks.com",
        )

        # Mock the workspace_client property
        with patch.object(
            type(function),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=mock_ws,
        ):
            config = _build_connection_config(function)

            # Verify OAuth provider was called with McpFunctionModel's workspace client
            mock_provider_class.assert_called_once_with(mock_ws)
            assert config["transport"] == "http"

    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_functions_source_uses_mcpfunction_workspace_client(
        self, mock_provider_class
    ):
        """Test that functions source uses McpFunctionModel's workspace_client (SchemaModel has no auth)."""
        from unittest.mock import PropertyMock

        from dao_ai.config import SchemaModel

        mock_auth = Mock()
        mock_provider_class.return_value = mock_auth

        # Create mock workspace client for McpFunctionModel itself
        mock_ws = Mock()

        schema = SchemaModel(catalog_name="catalog", schema_name="schema")
        function = McpFunctionModel(
            functions=schema,
            workspace_host="https://workspace.databricks.com",
        )

        # Mock the workspace_client property
        with patch.object(
            type(function),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=mock_ws,
        ):
            config = _build_connection_config(function)

            # Verify OAuth provider was called with McpFunctionModel's workspace client
            # (SchemaModel doesn't inherit from IsDatabricksResource)
            mock_provider_class.assert_called_once_with(mock_ws)
            assert config["transport"] == "http"

    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_connection_has_priority_over_mcpfunction_auth(self, mock_provider_class):
        """Test that connection's workspace_client takes priority over McpFunctionModel's when connection has auth."""
        from unittest.mock import PropertyMock

        mock_auth = Mock()
        mock_provider_class.return_value = mock_auth

        from dao_ai.config import ConnectionModel

        # Create separate workspace clients for connection and function
        connection_ws = Mock(name="connection_ws")
        function_ws = Mock(name="function_ws")

        # Connection must have auth configured to take priority
        connection = ConnectionModel(name="test-connection", on_behalf_of_user=True)
        function = McpFunctionModel(connection=connection)

        # Mock both workspace_client properties
        with patch.object(
            type(connection),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=connection_ws,
        ):
            with patch.object(
                type(function),
                "workspace_client",
                new_callable=PropertyMock,
                return_value=function_ws,
            ):
                _build_connection_config(function)

                # Verify connection's workspace client was used, not function's
                mock_provider_class.assert_called_once_with(connection_ws)

    @patch("databricks_mcp.DatabricksOAuthClientProvider")
    def test_app_has_priority_over_mcpfunction_auth(self, mock_provider_class):
        """Test that app's workspace_client takes priority over McpFunctionModel's when app has auth."""
        from unittest.mock import PropertyMock

        mock_auth = Mock()
        mock_provider_class.return_value = mock_auth

        # Create separate workspace clients for app and function
        app_ws = Mock(name="app_ws")
        function_ws = Mock(name="function_ws")

        # Create mock app
        mock_app_instance = Mock()
        mock_app_instance.url = "https://my-app.databricks.com"
        app_ws.apps.get.return_value = mock_app_instance

        # App must have auth configured to take priority
        app = DatabricksAppModel(name="my-app", on_behalf_of_user=True)
        function = McpFunctionModel(app=app)

        # Mock both workspace_client properties
        with patch.object(
            type(app),
            "workspace_client",
            new_callable=PropertyMock,
            return_value=app_ws,
        ):
            with patch.object(
                type(function),
                "workspace_client",
                new_callable=PropertyMock,
                return_value=function_ws,
            ):
                _build_connection_config(function)

                # Verify app's workspace client was used, not function's
                mock_provider_class.assert_called_once_with(app_ws)
