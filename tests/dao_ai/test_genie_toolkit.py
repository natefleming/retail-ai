"""Unit tests for GenieToolkit, create_genie_toolkit, and related features.

Tests cover:
- GenieToolkit structure and tool bundling
- create_genie_toolkit producing query + feedback tools
- create_genie_tool producing uncached query tool only
- Empty result auto-invalidation (invalidate_on_empty_result)
- Invalidate cascade through the service stack
- Session state message_id
"""

import json
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from langchain_core.tools import BaseTool

from dao_ai.config import (
    GenieLRUCacheParametersModel,
    WarehouseModel,
)
from dao_ai.genie.cache import CacheResult, LRUCacheService
from dao_ai.genie.cache.base import GenieServiceBase, SQLCacheEntry
from dao_ai.genie.core import GenieResponse
from dao_ai.tools.genie import (
    GenieToolkit,
    _response_to_json_with_cache,
    create_genie_tool,
    create_genie_toolkit,
)

# ============================================================================
# GenieToolkit class tests
# ============================================================================


class TestGenieToolkit:
    """Tests for the GenieToolkit container class."""

    def test_toolkit_returns_tools_via_get_tools(self) -> None:
        mock_tool_1 = Mock(spec=BaseTool)
        mock_tool_2 = Mock(spec=BaseTool)
        toolkit = GenieToolkit(tools=[mock_tool_1, mock_tool_2])

        tools = toolkit.get_tools()

        assert len(tools) == 2
        assert tools[0] is mock_tool_1
        assert tools[1] is mock_tool_2

    def test_toolkit_empty_by_default(self) -> None:
        toolkit = GenieToolkit()

        assert toolkit.get_tools() == []


# ============================================================================
# _response_to_json_with_cache tests
# ============================================================================


class TestResponseToJsonWithCache:
    """Tests for _response_to_json_with_cache including cache_hit field."""

    def test_includes_cache_hit_true(self) -> None:
        response = GenieResponse(
            result="test data",
            query="SELECT 1",
            description="Test",
            conversation_id="conv-1",
        )

        json_str = _response_to_json_with_cache(response, cache_hit=True)
        data = json.loads(json_str)

        assert data["cache_hit"] is True
        assert data["result"] == "test data"

    def test_includes_cache_hit_false(self) -> None:
        response = GenieResponse(
            result="test data",
            query="SELECT 1",
            description="Test",
            conversation_id="conv-1",
        )

        json_str = _response_to_json_with_cache(response, cache_hit=False)
        data = json.loads(json_str)

        assert data["cache_hit"] is False

    def test_handles_dataframe_result(self) -> None:
        response = GenieResponse(
            result=pd.DataFrame({"col": [1, 2]}),
            query="SELECT col FROM t",
            conversation_id="conv-1",
        )

        json_str = _response_to_json_with_cache(response, cache_hit=True)
        data = json.loads(json_str)

        assert "col" in data["result"]
        assert data["cache_hit"] is True


# ============================================================================
# Empty result auto-invalidation tests (LRU)
# ============================================================================


class TestLRUEmptyResultInvalidation:
    """Tests for invalidate_on_empty_result in LRU cache."""

    @pytest.fixture
    def lru_params_with_invalidation(self) -> Mock:
        params = Mock(spec=GenieLRUCacheParametersModel)
        params.capacity = 100
        params.time_to_live_seconds = 86400
        params.warehouse = Mock(spec=WarehouseModel)
        params.invalidate_on_empty_result = True
        return params

    @pytest.fixture
    def lru_params_without_invalidation(self) -> Mock:
        params = Mock(spec=GenieLRUCacheParametersModel)
        params.capacity = 100
        params.time_to_live_seconds = 86400
        params.warehouse = Mock(spec=WarehouseModel)
        params.invalidate_on_empty_result = False
        return params

    def test_empty_result_invalidates_when_enabled(
        self, lru_params_with_invalidation: Mock
    ) -> None:
        """Empty DataFrame should trigger invalidation and Genie fallback."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        fresh_response = GenieResponse(
            result=pd.DataFrame({"a": [1]}),
            query="SELECT a FROM fresh",
            description="Fresh result",
            conversation_id="conv-1",
        )
        mock_impl.ask_question.return_value = CacheResult(
            response=fresh_response, cache_hit=False, served_by=None
        )

        service = LRUCacheService(
            impl=mock_impl, parameters=lru_params_with_invalidation
        )

        stale_response = GenieResponse(
            result=pd.DataFrame({"a": [1]}),
            query="SELECT a FROM stale WHERE date = CURRENT_DATE()",
            description="Stale",
            conversation_id="conv-old",
        )
        key = service._normalize_key("test question", None)
        service._put(key, stale_response)

        with patch.object(service, "_execute_sql") as mock_exec:
            mock_exec.return_value = pd.DataFrame()

            result = service.ask_question("test question")

        assert result.cache_hit is False
        mock_impl.ask_question.assert_called_once()

    def test_empty_result_kept_when_disabled(
        self, lru_params_without_invalidation: Mock
    ) -> None:
        """Empty DataFrame should be returned as cache hit when invalidation is off."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = LRUCacheService(
            impl=mock_impl, parameters=lru_params_without_invalidation
        )

        stale_response = GenieResponse(
            result=pd.DataFrame({"a": [1]}),
            query="SELECT a FROM t WHERE 1=0",
            description="Always empty",
            conversation_id="conv-1",
        )
        key = service._normalize_key("test question", None)
        service._put(key, stale_response)

        with patch.object(service, "_execute_sql") as mock_exec:
            mock_exec.return_value = pd.DataFrame()

            result = service.ask_question("test question")

        assert result.cache_hit is True
        mock_impl.ask_question.assert_not_called()

    def test_nonempty_result_not_invalidated(
        self, lru_params_with_invalidation: Mock
    ) -> None:
        """Non-empty DataFrame should be treated as a normal cache hit."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = LRUCacheService(
            impl=mock_impl, parameters=lru_params_with_invalidation
        )

        cached_response = GenieResponse(
            result=pd.DataFrame({"a": [1]}),
            query="SELECT a FROM t",
            description="Valid",
            conversation_id="conv-1",
        )
        key = service._normalize_key("test question", None)
        service._put(key, cached_response)

        with patch.object(service, "_execute_sql") as mock_exec:
            mock_exec.return_value = pd.DataFrame({"a": [42]})

            result = service.ask_question("test question")

        assert result.cache_hit is True
        mock_impl.ask_question.assert_not_called()


# ============================================================================
# Invalidate cascade tests
# ============================================================================


class TestInvalidateCascade:
    """Tests for invalidate() cascading through the service stack."""

    def test_base_service_invalidate_is_noop(self) -> None:
        """GenieServiceBase.invalidate returns False (no-op)."""
        base = Mock(spec=GenieServiceBase)
        base.invalidate = GenieServiceBase.invalidate.__get__(base)

        result = base.invalidate("test question")

        assert result is False

    def test_lru_invalidate_cascades_to_impl(self) -> None:
        """LRU invalidate should also call impl.invalidate."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"
        mock_impl.invalidate.return_value = True

        params = Mock(spec=GenieLRUCacheParametersModel)
        params.capacity = 100
        params.time_to_live_seconds = 86400
        params.warehouse = Mock()

        service = LRUCacheService(impl=mock_impl, parameters=params)

        key = service._normalize_key("test question", None)
        service._cache[key] = SQLCacheEntry(
            query="SELECT 1",
            description="Test",
            conversation_id="conv-1",
            created_at=datetime.now(),
        )

        result = service.invalidate("test question")

        assert result is True
        assert key not in service._cache
        mock_impl.invalidate.assert_called_once_with("test question", None)

    def test_lru_invalidate_returns_true_if_impl_removed(self) -> None:
        """LRU invalidate returns True even when LRU has no match but impl does."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"
        mock_impl.invalidate.return_value = True

        params = Mock(spec=GenieLRUCacheParametersModel)
        params.capacity = 100
        params.time_to_live_seconds = 86400
        params.warehouse = Mock()

        service = LRUCacheService(impl=mock_impl, parameters=params)

        result = service.invalidate("nonexistent question")

        assert result is True

    def test_lru_invalidate_returns_false_when_nothing_removed(self) -> None:
        """LRU invalidate returns False when neither layer has the entry."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"
        mock_impl.invalidate.return_value = False

        params = Mock(spec=GenieLRUCacheParametersModel)
        params.capacity = 100
        params.time_to_live_seconds = 86400
        params.warehouse = Mock()

        service = LRUCacheService(impl=mock_impl, parameters=params)

        result = service.invalidate("nonexistent question")

        assert result is False


# ============================================================================
# create_genie_toolkit factory tests
# ============================================================================


class TestCreateGenieToolkit:
    """Tests for the create_genie_toolkit factory function."""

    @pytest.fixture
    def mock_genie_room(self) -> dict:
        return {
            "space_id": "test-space-id",
            "name": "Test Room",
            "on_behalf_of_user": False,
        }

    @pytest.fixture
    def mock_lru_params(self) -> GenieLRUCacheParametersModel:
        return GenieLRUCacheParametersModel(
            warehouse=WarehouseModel(warehouse_id="test-warehouse"),
            capacity=10,
            time_to_live_seconds=60,
        )

    def test_toolkit_has_two_tools_when_cache_configured(
        self, mock_genie_room: dict, mock_lru_params: GenieLRUCacheParametersModel
    ) -> None:
        """Toolkit should contain query + feedback tools when cache is configured."""
        toolkit = create_genie_toolkit(
            genie_room=mock_genie_room,
            name="query_test",
            lru_cache_parameters=mock_lru_params,
        )

        assert isinstance(toolkit, GenieToolkit)
        tools = toolkit.get_tools()
        assert len(tools) == 2

        tool_names = [t.name for t in tools]
        assert "query_test" in tool_names
        assert "query_test_feedback" in tool_names

    def test_feedback_tool_description_references_query_tool(
        self, mock_genie_room: dict, mock_lru_params: GenieLRUCacheParametersModel
    ) -> None:
        """Feedback tool description should reference the query tool name."""
        toolkit = create_genie_toolkit(
            genie_room=mock_genie_room,
            name="my_genie_query",
            lru_cache_parameters=mock_lru_params,
        )

        tools = toolkit.get_tools()
        feedback_tool = [t for t in tools if "feedback" in t.name][0]

        assert "my_genie_query" in feedback_tool.description

    def test_default_tool_name(
        self, mock_genie_room: dict, mock_lru_params: GenieLRUCacheParametersModel
    ) -> None:
        """Default tool name should be 'genie_tool'."""
        toolkit = create_genie_toolkit(
            genie_room=mock_genie_room,
            lru_cache_parameters=mock_lru_params,
        )

        tools = toolkit.get_tools()
        tool_names = [t.name for t in tools]
        assert "genie_tool" in tool_names
        assert "genie_tool_feedback" in tool_names


# ============================================================================
# create_genie_tool factory tests (simple/uncached)
# ============================================================================


class TestCreateGenieTool:
    """Tests for the simplified create_genie_tool factory."""

    @pytest.fixture
    def mock_genie_room(self) -> dict:
        return {
            "space_id": "test-space-id",
            "name": "Test Room",
            "on_behalf_of_user": False,
        }

    def test_returns_single_tool(self, mock_genie_room: dict) -> None:
        """create_genie_tool should return a single tool."""
        tool = create_genie_tool(genie_room=mock_genie_room, name="simple_query")

        assert tool.name == "simple_query"

    def test_no_feedback_tool(self, mock_genie_room: dict) -> None:
        """create_genie_tool should NOT produce a toolkit."""
        tool = create_genie_tool(genie_room=mock_genie_room)

        assert not isinstance(tool, GenieToolkit)


# ============================================================================
# Session state message_id tests
# ============================================================================


class TestSessionStateMessageId:
    """Tests for message_id in GenieSpaceState."""

    def test_genie_space_state_has_message_id(self) -> None:
        from dao_ai.state import GenieSpaceState

        state = GenieSpaceState(
            conversation_id="conv-1",
            message_id="msg-123",
        )

        assert state.message_id == "msg-123"

    def test_genie_space_state_message_id_defaults_to_none(self) -> None:
        from dao_ai.state import GenieSpaceState

        state = GenieSpaceState(conversation_id="conv-1")

        assert state.message_id is None

    def test_update_space_stores_message_id(self) -> None:
        from dao_ai.state import GenieState

        genie_state = GenieState()

        genie_state.update_space(
            space_id="space-1",
            conversation_id="conv-1",
            message_id="msg-456",
        )

        assert genie_state.spaces["space-1"].message_id == "msg-456"

    def test_update_space_message_id_defaults_to_none(self) -> None:
        from dao_ai.state import GenieState

        genie_state = GenieState()

        genie_state.update_space(
            space_id="space-1",
            conversation_id="conv-1",
        )

        assert genie_state.spaces["space-1"].message_id is None


# ============================================================================
# DatabaseModel connect_timeout tests
# ============================================================================


class TestDatabaseModelConnectTimeout:
    """Tests for connect_timeout on DatabaseModel."""

    def test_autoscaling_defaults_to_30(self) -> None:
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(project="my-project")
        assert db.connect_timeout == 30

    def test_provisioned_defaults_to_10(self) -> None:
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(instance_name="my-instance")
        assert db.connect_timeout == 10

    def test_standard_postgres_defaults_to_10(self) -> None:
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(
            name="my-pg",
            host="localhost",
            user="test",
            password="test",
        )
        assert db.connect_timeout == 10

    def test_explicit_override(self) -> None:
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(project="my-project", connect_timeout=60)
        assert db.connect_timeout == 60

    def test_connect_timeout_in_connection_params(self) -> None:
        """connect_timeout should appear in the connection_params dict."""
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(
            name="test-db",
            host="test-host.example.com",
            user="testuser",
            password="testpass",
            connect_timeout=15,
        )

        params = db.connection_params
        assert params["connect_timeout"] == 15
        assert params["host"] == "test-host.example.com"
