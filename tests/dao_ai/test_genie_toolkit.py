"""Unit tests for GenieToolkit, create_genie_toolkit, and related features.

Tests cover:
- GenieToolkit structure and tool bundling
- create_genie_toolkit producing query + feedback tools
- create_genie_tool producing uncached query tool only
- Empty result auto-invalidation (invalidate_on_empty_result)
- Invalidate cascade through the service stack
- Session state message_id
- _CacheHitTracker (Circuit Breaker pattern)
- Auto-invalidation on consecutive cache hits
- Enhanced feedback tool descriptions and cache-hit hints
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
    _CacheHitTracker,
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

        with patch("dao_ai.genie.cache.lru.execute_sql_via_warehouse") as mock_exec:
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

        with patch("dao_ai.genie.cache.lru.execute_sql_via_warehouse") as mock_exec:
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

        with patch("dao_ai.genie.cache.lru.execute_sql_via_warehouse") as mock_exec:
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

    def test_instance_name_alias_defaults_to_30(self) -> None:
        """`instance_name` is now a deprecated alias for `project`; both
        resolve to autoscaling Lakebase, so connect_timeout defaults to 30."""
        import warnings

        from dao_ai.config import DatabaseModel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            db = DatabaseModel(instance_name="my-instance")
        assert db.connect_timeout == 30
        assert db.is_lakebase is True
        assert db.project == "my-instance"

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


# ============================================================================
# _CacheHitTracker tests (Circuit Breaker pattern)
# ============================================================================


class TestCacheHitTracker:
    """Tests for the _CacheHitTracker used in auto-invalidation."""

    def test_record_returns_one_on_first_cache_hit(self) -> None:
        tracker = _CacheHitTracker()

        count = tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")

        assert count == 1

    def test_record_returns_zero_on_cache_miss(self) -> None:
        tracker = _CacheHitTracker()

        count = tracker.record("space-1", cache_hit=False, sql_query="SELECT 1")

        assert count == 0

    def test_record_increments_on_consecutive_identical_hits(self) -> None:
        tracker = _CacheHitTracker()

        c1 = tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")
        c2 = tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")
        c3 = tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")

        assert c1 == 1
        assert c2 == 2
        assert c3 == 3

    def test_record_resets_on_cache_miss(self) -> None:
        tracker = _CacheHitTracker()

        tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")
        tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")
        count = tracker.record("space-1", cache_hit=False, sql_query="SELECT 1")

        assert count == 0

    def test_record_resets_on_different_sql(self) -> None:
        tracker = _CacheHitTracker()

        tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")
        tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")
        count = tracker.record("space-1", cache_hit=True, sql_query="SELECT 2")

        assert count == 1

    def test_caps_history_at_max(self) -> None:
        tracker = _CacheHitTracker(max_history=5)

        for _ in range(10):
            tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")

        assert len(tracker._history["space-1"]) == 5

    def test_handles_none_sql(self) -> None:
        """Cache hit with no SQL query hashes to None, treated as non-matching."""
        tracker = _CacheHitTracker()

        count = tracker.record("space-1", cache_hit=True, sql_query=None)

        # None hash is treated the same as a cache miss (no SQL to compare)
        assert count == 0

    def test_tracks_independently_per_space(self) -> None:
        tracker = _CacheHitTracker()

        tracker.record("space-A", cache_hit=True, sql_query="SELECT 1")
        tracker.record("space-A", cache_hit=True, sql_query="SELECT 1")
        count_a = tracker.record("space-A", cache_hit=True, sql_query="SELECT 1")

        count_b = tracker.record("space-B", cache_hit=True, sql_query="SELECT 1")

        assert count_a == 3
        assert count_b == 1

    def test_empty_history_returns_zero(self) -> None:
        tracker = _CacheHitTracker()

        assert tracker._consecutive_count([]) == 0

    def test_consecutive_count_with_none_at_end(self) -> None:
        """Cache miss (None) at end of history returns 0."""
        assert _CacheHitTracker._consecutive_count(["abc", "abc", None]) == 0

    def test_consecutive_count_mixed(self) -> None:
        assert _CacheHitTracker._consecutive_count(["abc", "def", "def"]) == 2


# ============================================================================
# Auto-invalidation (Circuit Breaker) tests
# ============================================================================


class TestAutoInvalidation:
    """Tests for auto-invalidation triggered by consecutive cache hits."""

    def test_auto_invalidation_triggers_at_threshold(self) -> None:
        """After max_consecutive_cache_hits, response shows auto_invalidated."""
        response = GenieResponse(
            result="cached data",
            query="SELECT cached FROM t",
            description="Cached",
            conversation_id="conv-1",
        )
        fresh_response = GenieResponse(
            result="fresh data",
            query="SELECT fresh FROM t",
            description="Fresh",
            conversation_id="conv-1",
        )

        _cached_result = CacheResult(
            response=response, cache_hit=True, served_by="LRUCacheService"
        )
        _fresh_result = CacheResult(
            response=fresh_response, cache_hit=False, served_by=None
        )

        json_str = _response_to_json_with_cache(
            response,
            cache_hit=True,
            consecutive_cache_hits=3,
            feedback_tool_name="genie_tool_feedback",
            auto_invalidated=False,
        )
        data = json.loads(json_str)

        assert data["cache_hit"] is True
        assert data["consecutive_cache_hits"] == 3
        assert "_hint" in data
        assert "auto_invalidated" not in data

    def test_auto_invalidation_disabled_when_none(self) -> None:
        """max_consecutive_cache_hits=None should never set auto_invalidated."""
        response = GenieResponse(
            result="data",
            query="SELECT 1",
            description="Test",
            conversation_id="conv-1",
        )

        json_str = _response_to_json_with_cache(
            response,
            cache_hit=True,
            consecutive_cache_hits=10,
            feedback_tool_name="genie_tool_feedback",
            auto_invalidated=False,
        )
        data = json.loads(json_str)

        assert "auto_invalidated" not in data

    def test_auto_invalidation_response_json(self) -> None:
        """When auto_invalidated=True, response includes reason."""
        response = GenieResponse(
            result="fresh data",
            query="SELECT fresh FROM t",
            description="Fresh result",
            conversation_id="conv-1",
        )

        json_str = _response_to_json_with_cache(
            response,
            cache_hit=False,
            consecutive_cache_hits=0,
            feedback_tool_name="test_feedback",
            auto_invalidated=True,
        )
        data = json.loads(json_str)

        assert data["auto_invalidated"] is True
        assert "Circuit breaker" in data["auto_invalidation_reason"]
        assert data["cache_hit"] is False

    def test_auto_invalidation_resets_after_fresh_result(self) -> None:
        """Tracker count resets to 0 after recording a cache miss."""
        tracker = _CacheHitTracker()

        tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")
        tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")
        tracker.record("space-1", cache_hit=True, sql_query="SELECT 1")

        count = tracker.record("space-1", cache_hit=False, sql_query="SELECT fresh")

        assert count == 0

        next_hit = tracker.record("space-1", cache_hit=True, sql_query="SELECT new")
        assert next_hit == 1

    def test_tracker_threshold_detection(self) -> None:
        """Verify the threshold detection logic used by auto-invalidation."""
        tracker = _CacheHitTracker()
        threshold = 3

        for i in range(threshold):
            count = tracker.record("space-1", cache_hit=True, sql_query="SELECT bad")

        assert count >= threshold

    def test_auto_invalidation_cascades_invalidate(self) -> None:
        """The feedback tool calls invalidate() which cascades through layers."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"
        mock_impl.invalidate.return_value = True

        params = Mock(spec=GenieLRUCacheParametersModel)
        params.capacity = 100
        params.time_to_live_seconds = 86400
        params.warehouse = Mock()

        service = LRUCacheService(impl=mock_impl, parameters=params)

        key = service._normalize_key("test question", "conv-1")
        service._cache[key] = SQLCacheEntry(
            query="SELECT 1",
            description="Test",
            conversation_id="conv-1",
            created_at=datetime.now(),
        )

        result = service.invalidate("test question", "conv-1")

        assert result is True
        assert key not in service._cache
        mock_impl.invalidate.assert_called_once_with("test question", "conv-1")


# ============================================================================
# Enhanced descriptions and cache-hit hints tests
# ============================================================================


class TestEnhancedDescriptions:
    """Tests for enhanced feedback tool descriptions and cache-hit response hints."""

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

    def test_feedback_description_contains_must(
        self, mock_genie_room: dict, mock_lru_params: GenieLRUCacheParametersModel
    ) -> None:
        toolkit = create_genie_toolkit(
            genie_room=mock_genie_room,
            name="test_query",
            lru_cache_parameters=mock_lru_params,
        )

        tools = toolkit.get_tools()
        feedback_tool = [t for t in tools if "feedback" in t.name][0]

        assert "MUST" in feedback_tool.description

    def test_feedback_description_contains_trigger_conditions(
        self, mock_genie_room: dict, mock_lru_params: GenieLRUCacheParametersModel
    ) -> None:
        toolkit = create_genie_toolkit(
            genie_room=mock_genie_room,
            name="test_query",
            lru_cache_parameters=mock_lru_params,
        )

        tools = toolkit.get_tools()
        feedback_tool = [t for t in tools if "feedback" in t.name][0]

        assert "wrong" in feedback_tool.description.lower()
        assert "empty" in feedback_tool.description.lower()
        assert "cache_hit=true" in feedback_tool.description

    def test_feedback_description_references_query_tool(
        self, mock_genie_room: dict, mock_lru_params: GenieLRUCacheParametersModel
    ) -> None:
        toolkit = create_genie_toolkit(
            genie_room=mock_genie_room,
            name="my_custom_query",
            lru_cache_parameters=mock_lru_params,
        )

        tools = toolkit.get_tools()
        feedback_tool = [t for t in tools if "feedback" in t.name][0]

        assert "my_custom_query" in feedback_tool.description

    def test_cache_hit_response_includes_hint(self) -> None:
        response = GenieResponse(
            result="test data",
            query="SELECT 1",
            description="Test",
            conversation_id="conv-1",
        )

        json_str = _response_to_json_with_cache(
            response,
            cache_hit=True,
            feedback_tool_name="genie_tool_feedback",
        )
        data = json.loads(json_str)

        assert "_hint" in data
        assert "cache" in data["_hint"].lower()
        assert "genie_tool_feedback" in data["_hint"]

    def test_cache_miss_response_excludes_hint(self) -> None:
        response = GenieResponse(
            result="test data",
            query="SELECT 1",
            description="Test",
            conversation_id="conv-1",
        )

        json_str = _response_to_json_with_cache(
            response,
            cache_hit=False,
            feedback_tool_name="genie_tool_feedback",
        )
        data = json.loads(json_str)

        assert "_hint" not in data

    def test_response_includes_consecutive_cache_hits(self) -> None:
        response = GenieResponse(
            result="test data",
            query="SELECT 1",
            description="Test",
            conversation_id="conv-1",
        )

        json_str = _response_to_json_with_cache(
            response,
            cache_hit=True,
            consecutive_cache_hits=3,
            feedback_tool_name="genie_tool_feedback",
        )
        data = json.loads(json_str)

        assert data["consecutive_cache_hits"] == 3

    def test_response_excludes_consecutive_cache_hits_when_one(self) -> None:
        response = GenieResponse(
            result="test data",
            query="SELECT 1",
            description="Test",
            conversation_id="conv-1",
        )

        json_str = _response_to_json_with_cache(
            response,
            cache_hit=True,
            consecutive_cache_hits=1,
            feedback_tool_name="genie_tool_feedback",
        )
        data = json.loads(json_str)

        assert "consecutive_cache_hits" not in data

    def test_create_genie_toolkit_accepts_max_consecutive_cache_hits(
        self, mock_genie_room: dict, mock_lru_params: GenieLRUCacheParametersModel
    ) -> None:
        """Verify the parameter is accepted without error."""
        toolkit = create_genie_toolkit(
            genie_room=mock_genie_room,
            name="query_test",
            lru_cache_parameters=mock_lru_params,
            max_consecutive_cache_hits=3,
        )

        assert isinstance(toolkit, GenieToolkit)
        assert len(toolkit.get_tools()) == 2
