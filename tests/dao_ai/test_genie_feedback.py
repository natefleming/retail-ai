"""Unit and integration tests for Genie feedback functionality."""

from datetime import datetime
from unittest.mock import Mock

import pytest
from databricks.sdk import WorkspaceClient

from dao_ai.config import (
    GenieInMemorySemanticCacheParametersModel,
    GenieLRUCacheParametersModel,
    WarehouseModel,
)
from dao_ai.genie import (
    DatabricksGenie,
    DatabricksGenieResponse,
    Genie,
    GenieFeedbackRating,
    GenieResponse,
    GenieService,
)
from dao_ai.genie.cache import (
    CacheResult,
    InMemoryContextAwareGenieService,
    LRUCacheService,
    SQLCacheEntry,
)
from dao_ai.genie.cache.base import get_latest_message_id, get_message_content
from dao_ai.genie.cache.context_aware.in_memory import InMemoryCacheEntry

# ============================================================================
# Unit Tests for Helper Functions
# ============================================================================


class TestGetLatestMessageId:
    """Unit tests for get_latest_message_id function."""

    def test_returns_message_id_from_latest_message(self) -> None:
        """Test that the most recent message_id is returned."""
        mock_client = Mock(spec=WorkspaceClient)

        # Create mock messages
        mock_messages = [
            Mock(message_id="msg-001", id="id-001"),
            Mock(message_id="msg-002", id="id-002"),
            Mock(message_id="msg-003", id="id-003"),  # Most recent
        ]
        mock_response = Mock(messages=mock_messages)
        mock_client.genie.list_conversation_messages.return_value = mock_response

        result = get_latest_message_id(
            workspace_client=mock_client,
            space_id="test-space",
            conversation_id="test-conv",
        )

        assert result == "msg-003"  # Should return the last message's ID
        mock_client.genie.list_conversation_messages.assert_called_once_with(
            space_id="test-space",
            conversation_id="test-conv",
        )

    def test_falls_back_to_id_when_message_id_missing(self) -> None:
        """Test fallback to legacy 'id' field when message_id is None."""
        mock_client = Mock(spec=WorkspaceClient)

        mock_messages = [
            Mock(message_id=None, id="legacy-id-001"),
        ]
        mock_response = Mock(messages=mock_messages)
        mock_client.genie.list_conversation_messages.return_value = mock_response

        result = get_latest_message_id(
            workspace_client=mock_client,
            space_id="test-space",
            conversation_id="test-conv",
        )

        assert result == "legacy-id-001"

    def test_returns_none_when_no_messages(self) -> None:
        """Test that None is returned when there are no messages."""
        mock_client = Mock(spec=WorkspaceClient)

        mock_response = Mock(messages=[])
        mock_client.genie.list_conversation_messages.return_value = mock_response

        result = get_latest_message_id(
            workspace_client=mock_client,
            space_id="test-space",
            conversation_id="test-conv",
        )

        assert result is None

    def test_returns_none_on_api_error(self) -> None:
        """Test that None is returned when API call fails."""
        mock_client = Mock(spec=WorkspaceClient)
        mock_client.genie.list_conversation_messages.side_effect = Exception(
            "API Error"
        )

        result = get_latest_message_id(
            workspace_client=mock_client,
            space_id="test-space",
            conversation_id="test-conv",
        )

        assert result is None


class TestGetMessageContent:
    """Unit tests for get_message_content function."""

    def test_returns_message_content(self) -> None:
        """Test that message content is returned."""
        mock_client = Mock(spec=WorkspaceClient)

        mock_message = Mock(content="What is the total sales?")
        mock_client.genie.get_message.return_value = mock_message

        result = get_message_content(
            workspace_client=mock_client,
            space_id="test-space",
            conversation_id="test-conv",
            message_id="test-msg",
        )

        assert result == "What is the total sales?"
        mock_client.genie.get_message.assert_called_once_with(
            space_id="test-space",
            conversation_id="test-conv",
            message_id="test-msg",
        )

    def test_returns_none_on_api_error(self) -> None:
        """Test that None is returned when API call fails."""
        mock_client = Mock(spec=WorkspaceClient)
        mock_client.genie.get_message.side_effect = Exception("API Error")

        result = get_message_content(
            workspace_client=mock_client,
            space_id="test-space",
            conversation_id="test-conv",
            message_id="test-msg",
        )

        assert result is None


# ============================================================================
# Unit Tests for Extended Genie/GenieResponse Classes
# ============================================================================


class TestExtendedGenieResponse:
    """Unit tests for extended GenieResponse with message_id."""

    def test_genie_response_has_message_id_field(self) -> None:
        """Test that GenieResponse has message_id field."""
        response = GenieResponse(
            result="test result",
            query="SELECT 1",
            description="Test query",
            conversation_id="conv-123",
            message_id="msg-456",
        )

        assert response.message_id == "msg-456"
        assert response.conversation_id == "conv-123"
        assert response.query == "SELECT 1"

    def test_genie_response_message_id_defaults_to_none(self) -> None:
        """Test that message_id defaults to None."""
        response = GenieResponse(
            result="test result",
        )

        assert response.message_id is None

    def test_genie_response_extends_databricks_response(self) -> None:
        """Test that GenieResponse extends DatabricksGenieResponse."""
        assert issubclass(GenieResponse, DatabricksGenieResponse)

    def test_genie_extends_databricks_genie(self) -> None:
        """Test that Genie extends DatabricksGenie."""
        assert issubclass(Genie, DatabricksGenie)


class TestCacheResultMessageId:
    """Unit tests for CacheResult with message_id and cache_entry_id."""

    def test_cache_result_has_message_id_field(self) -> None:
        """Test that CacheResult has message_id field."""
        response = GenieResponse(result="test")
        result = CacheResult(
            response=response,
            cache_hit=False,
            served_by=None,
            message_id="msg-123",
        )

        assert result.message_id == "msg-123"

    def test_cache_result_message_id_defaults_to_none(self) -> None:
        """Test that message_id defaults to None."""
        response = GenieResponse(result="test")
        result = CacheResult(
            response=response,
            cache_hit=False,
        )

        assert result.message_id is None

    def test_cache_result_propagates_message_id_on_cache_miss(self) -> None:
        """Test that message_id is propagated on cache miss."""
        response = GenieResponse(
            result="test",
            message_id="msg-original",
        )
        result = CacheResult(
            response=response,
            cache_hit=False,
            served_by=None,
            message_id="msg-original",
        )

        assert result.cache_hit is False
        assert result.message_id == "msg-original"

    def test_cache_result_has_message_id_on_cache_hit(self) -> None:
        """Test that cache hits can have message_id from stored entry."""
        response = GenieResponse(result="test")
        result = CacheResult(
            response=response,
            cache_hit=True,
            served_by="PostgresCache",
            message_id="msg-original",
            cache_entry_id=42,
        )

        assert result.cache_hit is True
        assert result.message_id == "msg-original"
        assert result.cache_entry_id == 42

    def test_cache_result_has_cache_entry_id_field(self) -> None:
        """Test that CacheResult has cache_entry_id field."""
        response = GenieResponse(result="test")
        result = CacheResult(
            response=response,
            cache_hit=True,
            served_by="PostgresCache",
            message_id="msg-123",
            cache_entry_id=123,
        )

        assert result.cache_entry_id == 123

    def test_cache_result_cache_entry_id_defaults_to_none(self) -> None:
        """Test that cache_entry_id defaults to None."""
        response = GenieResponse(result="test")
        result = CacheResult(
            response=response,
            cache_hit=True,
            served_by="LRUCache",
        )

        assert result.cache_entry_id is None

    def test_lru_cache_hit_has_no_cache_entry_id(self) -> None:
        """Test that LRU cache hits have no cache_entry_id (in-memory only)."""
        response = GenieResponse(result="test")
        result = CacheResult(
            response=response,
            cache_hit=True,
            served_by="LRUCache",
            message_id="msg-123",
            cache_entry_id=None,  # LRU is in-memory, no DB row ID
        )

        assert result.cache_hit is True
        assert result.message_id == "msg-123"
        assert result.cache_entry_id is None


# ============================================================================
# Unit Tests for SQLCacheEntry and InMemoryCacheEntry with new fields
# ============================================================================


class TestSQLCacheEntryFields:
    """Unit tests for SQLCacheEntry with message_id and cache_entry_id."""

    def test_sql_cache_entry_has_message_id(self) -> None:
        """Test that SQLCacheEntry has message_id field."""
        entry = SQLCacheEntry(
            query="SELECT 1",
            description="Test",
            conversation_id="conv-123",
            created_at=datetime.now(),
            message_id="msg-456",
        )

        assert entry.message_id == "msg-456"

    def test_sql_cache_entry_message_id_defaults_to_none(self) -> None:
        """Test that message_id defaults to None."""
        entry = SQLCacheEntry(
            query="SELECT 1",
            description="Test",
            conversation_id="conv-123",
            created_at=datetime.now(),
        )

        assert entry.message_id is None

    def test_sql_cache_entry_has_cache_entry_id(self) -> None:
        """Test that SQLCacheEntry has cache_entry_id field."""
        entry = SQLCacheEntry(
            query="SELECT 1",
            description="Test",
            conversation_id="conv-123",
            created_at=datetime.now(),
            cache_entry_id=42,
        )

        assert entry.cache_entry_id == 42

    def test_sql_cache_entry_cache_entry_id_defaults_to_none(self) -> None:
        """Test that cache_entry_id defaults to None."""
        entry = SQLCacheEntry(
            query="SELECT 1",
            description="Test",
            conversation_id="conv-123",
            created_at=datetime.now(),
        )

        assert entry.cache_entry_id is None


class TestInMemoryCacheEntryFields:
    """Unit tests for InMemoryCacheEntry with message_id."""

    def test_in_memory_cache_entry_has_message_id(self) -> None:
        """Test that InMemoryCacheEntry has message_id field."""
        entry = InMemoryCacheEntry(
            genie_space_id="test-space",
            question="What is inventory?",
            conversation_context="",
            question_embedding=[0.1, 0.2, 0.3],
            context_embedding=[0.0, 0.0, 0.0],
            sql_query="SELECT * FROM inventory",
            description="Inventory query",
            conversation_id="conv-123",
            created_at=datetime.now(),
            last_accessed_at=datetime.now(),
            message_id="msg-456",
        )

        assert entry.message_id == "msg-456"

    def test_in_memory_cache_entry_message_id_defaults_to_none(self) -> None:
        """Test that message_id defaults to None."""
        entry = InMemoryCacheEntry(
            genie_space_id="test-space",
            question="What is inventory?",
            conversation_context="",
            question_embedding=[0.1, 0.2, 0.3],
            context_embedding=[0.0, 0.0, 0.0],
            sql_query="SELECT * FROM inventory",
            description="Inventory query",
            conversation_id="conv-123",
            created_at=datetime.now(),
            last_accessed_at=datetime.now(),
        )

        assert entry.message_id is None


# ============================================================================
# Unit Tests for GenieService.send_feedback
# ============================================================================


class TestGenieServiceSendFeedback:
    """Unit tests for GenieService.send_feedback method."""

    @pytest.fixture
    def mock_genie(self) -> Mock:
        """Create a mock Genie instance."""
        mock = Mock(spec=Genie)
        mock.space_id = "test-space"
        return mock

    @pytest.fixture
    def mock_workspace_client(self) -> Mock:
        """Create a mock WorkspaceClient."""
        return Mock(spec=WorkspaceClient)

    def test_sends_feedback_with_provided_message_id(
        self, mock_genie: Mock, mock_workspace_client: Mock
    ) -> None:
        """Test that feedback is sent when message_id is provided."""
        service = GenieService(genie=mock_genie, workspace_client=mock_workspace_client)

        service.send_feedback(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.POSITIVE,
            message_id="test-msg",
        )

        mock_workspace_client.genie.send_message_feedback.assert_called_once_with(
            space_id="test-space",
            conversation_id="test-conv",
            message_id="test-msg",
            rating=GenieFeedbackRating.POSITIVE,
        )

    def test_looks_up_message_id_when_not_provided(
        self, mock_genie: Mock, mock_workspace_client: Mock
    ) -> None:
        """Test that message_id is looked up when not provided."""
        # Mock the message lookup
        mock_messages = [Mock(message_id="found-msg-id", id="id-001")]
        mock_response = Mock(messages=mock_messages)
        mock_workspace_client.genie.list_conversation_messages.return_value = (
            mock_response
        )

        service = GenieService(genie=mock_genie, workspace_client=mock_workspace_client)

        service.send_feedback(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.NEGATIVE,
            message_id=None,
        )

        # Should have looked up the message_id
        mock_workspace_client.genie.list_conversation_messages.assert_called_once()

        # Should have sent feedback with the looked-up message_id
        mock_workspace_client.genie.send_message_feedback.assert_called_once_with(
            space_id="test-space",
            conversation_id="test-conv",
            message_id="found-msg-id",
            rating=GenieFeedbackRating.NEGATIVE,
        )

    def test_skips_feedback_when_message_id_not_found(
        self, mock_genie: Mock, mock_workspace_client: Mock
    ) -> None:
        """Test that feedback is skipped when message_id cannot be found."""
        # Mock empty messages
        mock_response = Mock(messages=[])
        mock_workspace_client.genie.list_conversation_messages.return_value = (
            mock_response
        )

        service = GenieService(genie=mock_genie, workspace_client=mock_workspace_client)

        service.send_feedback(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.NEGATIVE,
            message_id=None,
        )

        # Should NOT have sent feedback
        mock_workspace_client.genie.send_message_feedback.assert_not_called()

    def test_handles_api_error_gracefully(
        self, mock_genie: Mock, mock_workspace_client: Mock
    ) -> None:
        """Test that API errors are handled gracefully."""
        mock_workspace_client.genie.send_message_feedback.side_effect = Exception(
            "API Error"
        )

        service = GenieService(genie=mock_genie, workspace_client=mock_workspace_client)

        # Should not raise exception
        service.send_feedback(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.POSITIVE,
            message_id="test-msg",
        )


# ============================================================================
# Unit Tests for LRUCacheService.send_feedback
# ============================================================================


class TestLRUCacheServiceSendFeedback:
    """Unit tests for LRUCacheService.send_feedback method."""

    @pytest.fixture
    def mock_impl(self) -> Mock:
        """Create a mock underlying service."""
        mock = Mock()
        mock.space_id = "test-space"
        return mock

    @pytest.fixture
    def mock_parameters(self) -> Mock:
        """Create mock cache parameters."""
        params = Mock(spec=GenieLRUCacheParametersModel)
        params.capacity = 100
        params.time_to_live_seconds = 3600
        params.warehouse = Mock(spec=WarehouseModel)
        return params

    def test_forwards_feedback_when_not_cache_hit(
        self, mock_impl: Mock, mock_parameters: Mock
    ) -> None:
        """Test that feedback is forwarded to underlying service when not a cache hit."""
        service = LRUCacheService(impl=mock_impl, parameters=mock_parameters)

        service.send_feedback(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.POSITIVE,
            message_id="test-msg",
            was_cache_hit=False,
        )

        mock_impl.send_feedback.assert_called_once_with(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.POSITIVE,
            message_id="test-msg",
            was_cache_hit=False,
        )

    def test_skips_forward_when_cache_hit(
        self, mock_impl: Mock, mock_parameters: Mock
    ) -> None:
        """Test that feedback is NOT forwarded when response was from cache."""
        service = LRUCacheService(impl=mock_impl, parameters=mock_parameters)

        service.send_feedback(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.POSITIVE,
            message_id="test-msg",
            was_cache_hit=True,
        )

        # Should NOT forward to underlying service
        mock_impl.send_feedback.assert_not_called()

    def test_invalidates_cache_on_negative_feedback(
        self, mock_impl: Mock, mock_parameters: Mock
    ) -> None:
        """Test that cache entries are invalidated on negative feedback."""
        service = LRUCacheService(impl=mock_impl, parameters=mock_parameters)

        # Add cache entry with matching conversation_id
        service._cache["key1"] = SQLCacheEntry(
            query="SELECT 1",
            description="Test query",
            conversation_id="test-conv",
            created_at=datetime.now(),
        )
        service._cache["key2"] = SQLCacheEntry(
            query="SELECT 2",
            description="Other query",
            conversation_id="other-conv",
            created_at=datetime.now(),
        )

        service.send_feedback(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.NEGATIVE,
            message_id="test-msg",
            was_cache_hit=True,
        )

        # Only matching entry should be removed
        assert "key1" not in service._cache
        assert "key2" in service._cache

    def test_positive_feedback_does_not_invalidate_cache(
        self, mock_impl: Mock, mock_parameters: Mock
    ) -> None:
        """Test that positive feedback does NOT invalidate cache."""
        service = LRUCacheService(impl=mock_impl, parameters=mock_parameters)

        service._cache["key1"] = SQLCacheEntry(
            query="SELECT 1",
            description="Test query",
            conversation_id="test-conv",
            created_at=datetime.now(),
        )

        service.send_feedback(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.POSITIVE,
            message_id="test-msg",
            was_cache_hit=True,
        )

        # Entry should still be in cache
        assert "key1" in service._cache


# ============================================================================
# Unit Tests for InMemoryContextAwareGenieService Feedback
# ============================================================================


class TestInMemoryContextAwareSendFeedback:
    """Unit tests for InMemoryContextAwareGenieService feedback methods."""

    @pytest.fixture
    def mock_workspace_client(self) -> Mock:
        """Create a mock WorkspaceClient."""
        return Mock(spec=WorkspaceClient)

    @pytest.fixture
    def mock_parameters(self) -> Mock:
        """Create mock cache parameters."""
        params = Mock(spec=GenieInMemorySemanticCacheParametersModel)
        params.context_window_size = 3
        params.max_context_tokens = 2000
        params.embedding_model = "databricks-gte-large-en"
        params.embedding_dims = None
        params.time_to_live_seconds = 86400
        params.similarity_threshold = 0.85
        params.context_similarity_threshold = 0.80
        params.question_weight = 0.6
        params.context_weight = 0.4
        params.capacity = None
        params.warehouse = Mock(spec=WarehouseModel)
        return params

    def test_invalidate_by_question_removes_matching_entry(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that _invalidate_by_question removes matching entries."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = InMemoryContextAwareGenieService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Add cache entries
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="test-space",
                question="What is inventory?",
                conversation_context="",
                question_embedding=[0.1, 0.2, 0.3],
                context_embedding=[0.0, 0.0, 0.0],
                sql_query="SELECT * FROM inventory",
                description="Inventory query",
                conversation_id="conv-1",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="test-space",
                question="What are sales?",
                conversation_context="",
                question_embedding=[0.4, 0.5, 0.6],
                context_embedding=[0.0, 0.0, 0.0],
                sql_query="SELECT * FROM sales",
                description="Sales query",
                conversation_id="conv-2",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )

        result = service._invalidate_by_question("What is inventory?")

        assert result is True
        assert len(service._cache) == 1
        assert service._cache[0].question == "What are sales?"

    def test_invalidate_by_question_returns_false_when_not_found(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that _invalidate_by_question returns False when no match found."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = InMemoryContextAwareGenieService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        result = service._invalidate_by_question("Unknown question")

        assert result is False

    def test_invalidate_by_question_only_matches_same_space(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that _invalidate_by_question only matches entries from same space."""
        mock_impl = Mock()
        mock_impl.space_id = "space-A"

        service = InMemoryContextAwareGenieService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Add entry from different space
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="space-B",  # Different space
                question="What is inventory?",
                conversation_context="",
                question_embedding=[0.1, 0.2, 0.3],
                context_embedding=[0.0, 0.0, 0.0],
                sql_query="SELECT * FROM inventory",
                description="Inventory query",
                conversation_id="conv-1",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )

        result = service._invalidate_by_question("What is inventory?")

        # Should not match because space_id is different
        assert result is False
        assert len(service._cache) == 1

    def test_send_feedback_invalidates_cache_on_negative_rating(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that send_feedback invalidates cache on negative feedback."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = InMemoryContextAwareGenieService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Add cache entry
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="test-space",
                question="What is inventory?",
                conversation_context="",
                question_embedding=[0.1, 0.2, 0.3],
                context_embedding=[0.0, 0.0, 0.0],
                sql_query="SELECT * FROM inventory",
                description="Inventory query",
                conversation_id="test-conv",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )

        # Mock message lookup
        mock_messages = [Mock(message_id="test-msg", id="id-001")]
        mock_response = Mock(messages=mock_messages)
        mock_workspace_client.genie.list_conversation_messages.return_value = (
            mock_response
        )

        mock_message = Mock(content="What is inventory?")
        mock_workspace_client.genie.get_message.return_value = mock_message

        service.send_feedback(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.NEGATIVE,
            message_id=None,
            was_cache_hit=True,
        )

        # Cache should be invalidated
        assert len(service._cache) == 0

        # Should NOT forward to underlying service since was_cache_hit=True
        mock_impl.send_feedback.assert_not_called()

    def test_send_feedback_forwards_when_not_cache_hit(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that send_feedback forwards to underlying service when not cache hit."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = InMemoryContextAwareGenieService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        service.send_feedback(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.POSITIVE,
            message_id="test-msg",
            was_cache_hit=False,
        )

        # Should forward to underlying service
        mock_impl.send_feedback.assert_called_once_with(
            conversation_id="test-conv",
            rating=GenieFeedbackRating.POSITIVE,
            message_id="test-msg",
            was_cache_hit=False,
        )


# ============================================================================
# Integration Test Marker (for future integration tests)
# ============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skip(reason="Integration test - requires real Databricks environment")
def test_send_feedback_integration() -> None:
    """
    Integration test for sending feedback to Genie.

    This test requires:
    - Valid DATABRICKS_HOST and DATABRICKS_TOKEN environment variables
    - Access to a Genie space with existing conversations

    This test will make real API calls to Databricks.
    """
    # This is a placeholder for future integration tests
    # when real Databricks environment is available
    pass
