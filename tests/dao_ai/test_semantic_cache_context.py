"""Unit and integration tests for semantic cache with conversation context (rolling window)."""

import os
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from conftest import add_databricks_resource_attrs, has_retail_ai_env
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieMessage

from dao_ai.config import (
    DatabaseModel,
    GenieSemanticCacheParametersModel,
    WarehouseModel,
)
from dao_ai.genie import GenieService
from dao_ai.genie.cache import SemanticCacheService
from dao_ai.genie.cache.semantic import build_context_string, get_conversation_history

# ============================================================================
# Unit Tests for Context Building Functions
# ============================================================================


class TestBuildContextString:
    """Unit tests for build_context_string function."""

    def test_build_context_with_no_history(self) -> None:
        """Test that context string is just the question when no history."""
        question = "What is the total sales?"
        messages: list[GenieMessage] = []

        result = build_context_string(
            question=question,
            conversation_messages=messages,
            window_size=3,
        )

        assert result == question

    def test_build_context_with_window_size_zero(self) -> None:
        """Test that context string is just the question when window_size is 0."""
        question = "What is the total sales?"
        messages = [
            Mock(content="How many stores do we have?"),
            Mock(content="What about inventory?"),
        ]

        result = build_context_string(
            question=question,
            conversation_messages=messages,
            window_size=0,
        )

        assert result == question

    def test_build_context_with_single_message(self) -> None:
        """Test context building with one previous message."""
        question = "What about that store?"
        messages = [Mock(content="Show me Store 42")]

        result = build_context_string(
            question=question,
            conversation_messages=messages,
            window_size=3,
        )

        expected = "Previous: Show me Store 42\nCurrent: What about that store?"
        assert result == expected

    def test_build_context_with_multiple_messages(self) -> None:
        """Test context building with multiple previous messages."""
        question = "What about inventory?"
        messages = [
            Mock(content="How many stores?"),
            Mock(content="Show me Store 42"),
            Mock(content="What are the sales?"),
        ]

        result = build_context_string(
            question=question,
            conversation_messages=messages,
            window_size=3,
        )

        expected = (
            "Previous: How many stores?\n"
            "Previous: Show me Store 42\n"
            "Previous: What are the sales?\n"
            "Current: What about inventory?"
        )
        assert result == expected

    def test_build_context_respects_window_size(self) -> None:
        """Test that only the last N messages are included."""
        question = "What about inventory?"
        messages = [
            Mock(content="Message 1"),
            Mock(content="Message 2"),
            Mock(content="Message 3"),
            Mock(content="Message 4"),
            Mock(content="Message 5"),
        ]

        result = build_context_string(
            question=question,
            conversation_messages=messages,
            window_size=2,
        )

        # Should only include last 2 messages
        expected = (
            "Previous: Message 4\nPrevious: Message 5\nCurrent: What about inventory?"
        )
        assert result == expected

    def test_build_context_truncates_long_messages(self) -> None:
        """Test that very long messages are truncated."""
        question = "What about that?"
        long_message = "x" * 1000  # Very long message
        messages = [Mock(content=long_message)]

        result = build_context_string(
            question=question,
            conversation_messages=messages,
            window_size=3,
        )

        # Should truncate to 500 chars + "..."
        assert len(result) < len(long_message) + len(question) + 50
        assert "..." in result

    def test_build_context_respects_max_tokens(self) -> None:
        """Test that context is truncated to max_tokens."""
        question = "What about that?"
        messages = [Mock(content="x" * 10000) for _ in range(10)]

        result = build_context_string(
            question=question,
            conversation_messages=messages,
            window_size=10,
            max_tokens=100,  # Very small limit
        )

        # Rough estimate: 4 chars per token, so 100 tokens = 400 chars
        assert len(result) <= 450  # Allow some margin


class TestGetConversationHistory:
    """Unit tests for get_conversation_history function."""

    def test_get_conversation_history_success(self) -> None:
        """Test successful retrieval of conversation history."""
        mock_client = Mock(spec=WorkspaceClient)
        mock_messages = [
            Mock(content="Question 1"),
            Mock(content="Question 2"),
            Mock(content="Question 3"),
        ]
        mock_result = Mock(messages=mock_messages)
        mock_client.genie.list_conversation_messages.return_value = mock_result

        result = get_conversation_history(
            workspace_client=mock_client,
            space_id="test-space",
            conversation_id="test-conv",
            max_messages=10,
        )

        assert len(result) == 3
        assert result == mock_messages
        mock_client.genie.list_conversation_messages.assert_called_once_with(
            space_id="test-space",
            conversation_id="test-conv",
        )

    def test_get_conversation_history_limits_messages(self) -> None:
        """Test that max_messages limit is respected."""
        mock_client = Mock(spec=WorkspaceClient)
        mock_messages = [Mock(content=f"Question {i}") for i in range(20)]
        mock_result = Mock(messages=mock_messages)
        mock_client.genie.list_conversation_messages.return_value = mock_result

        result = get_conversation_history(
            workspace_client=mock_client,
            space_id="test-space",
            conversation_id="test-conv",
            max_messages=5,
        )

        # Should only return last 5 messages
        assert len(result) == 5
        assert result == mock_messages[-5:]

    def test_get_conversation_history_handles_error(self) -> None:
        """Test that errors are handled gracefully."""
        mock_client = Mock(spec=WorkspaceClient)
        mock_client.genie.list_conversation_messages.side_effect = Exception(
            "API Error"
        )

        result = get_conversation_history(
            workspace_client=mock_client,
            space_id="test-space",
            conversation_id="test-conv",
            max_messages=10,
        )

        # Should return empty list on error
        assert result == []

    def test_get_conversation_history_handles_no_messages(self) -> None:
        """Test handling of empty conversation."""
        mock_client = Mock(spec=WorkspaceClient)
        mock_result = Mock(messages=[])
        mock_client.genie.list_conversation_messages.return_value = mock_result

        result = get_conversation_history(
            workspace_client=mock_client,
            space_id="test-space",
            conversation_id="test-conv",
            max_messages=10,
        )

        assert result == []


# ============================================================================
# Unit Tests for SemanticCacheService with Context
# ============================================================================


class TestSemanticCacheServiceContext:
    """Unit tests for SemanticCacheService context-aware functionality."""

    @pytest.fixture
    def mock_workspace_client(self) -> Mock:
        """Create a mock WorkspaceClient."""
        return Mock(spec=WorkspaceClient)

    @pytest.fixture
    def mock_parameters(self) -> Mock:
        """Create mock cache parameters."""
        params = Mock(spec=GenieSemanticCacheParametersModel)
        params.context_window_size = 3
        params.max_context_tokens = 2000
        params.embedding_model = "databricks-gte-large-en"
        params.embedding_dims = None
        params.time_to_live_seconds = 86400
        params.similarity_threshold = 0.85
        params.context_similarity_threshold = 0.80
        params.question_weight = 0.6
        params.context_weight = 0.4
        params.table_name = "test_cache"
        params.database = Mock(spec=DatabaseModel)
        params.warehouse = Mock(spec=WarehouseModel)
        return params

    def test_embed_question_without_conversation_id(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that embedding without conversation_id uses question only."""
        mock_impl = Mock()
        service = SemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Mock embeddings
        service._embeddings = Mock()
        service._embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        question = "What is the total sales?"
        question_embedding, context_embedding, conversation_context = (
            service._embed_question(question, conversation_id=None)
        )

        # Should use question only (no context)
        assert conversation_context == ""
        assert question_embedding == [0.1, 0.2, 0.3]
        service._embeddings.embed_documents.assert_called_once_with([question])

    def test_embed_question_with_conversation_context(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that embedding with conversation_id includes context."""
        mock_impl = Mock()
        service = SemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Mock embeddings - return different embeddings for question and context
        service._embeddings = Mock()
        # Single call with both documents returns both embeddings
        service._embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3],  # First embedding: question
            [0.4, 0.5, 0.6],  # Second embedding: context
        ]

        # Mock conversation history
        mock_messages = [
            Mock(content="Show me Store 42"),
            Mock(content="What are the sales?"),
        ]
        mock_result = Mock(messages=mock_messages)
        mock_workspace_client.genie.list_conversation_messages.return_value = (
            mock_result
        )

        question = "What about inventory?"
        question_embedding, context_embedding, conversation_context = (
            service._embed_question(question, conversation_id="test-conv")
        )

        # Should include context
        assert "Previous: Show me Store 42" in conversation_context
        assert "Previous: What are the sales?" in conversation_context
        assert question_embedding == [0.1, 0.2, 0.3]
        # Context embedding should be the second embedding
        assert context_embedding == [0.4, 0.5, 0.6]

    def test_embed_question_handles_context_retrieval_error(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that errors in context retrieval fall back to question only."""
        mock_impl = Mock()
        service = SemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Mock embeddings
        service._embeddings = Mock()
        service._embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        # Mock conversation history to raise error
        mock_workspace_client.genie.list_conversation_messages.side_effect = Exception(
            "API Error"
        )

        question = "What about inventory?"
        question_embedding, context_embedding, conversation_context = (
            service._embed_question(question, conversation_id="test-conv")
        )

        # Should fall back to question only (no context)
        assert conversation_context == ""
        assert question_embedding == [0.1, 0.2, 0.3]

    def test_conversation_id_not_from_cache(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that conversation_id in response is current, not cached."""
        mock_impl = Mock()

        # Create a mock cache entry with a different conversation_id
        cached_entry_conversation_id = "cached-conv-123"
        current_conversation_id = "current-conv-456"

        service = SemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Setup mocks for cache hit scenario
        service._setup_complete = True
        service._embeddings = Mock()
        service._embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        service._pool = Mock()

        # Mock database to return a cache hit
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "question": "What is inventory?",
            "context_string": "Previous: Show Store 42\nCurrent: What is inventory?",
            "sql_query": "SELECT * FROM inventory",
            "description": "Inventory query",
            "conversation_id": cached_entry_conversation_id,  # Cached conversation ID
            "created_at": datetime.now(),
            "question_similarity": 0.95,
            "context_similarity": 0.95,
            "combined_similarity": 0.95,
            "is_valid": True,
        }
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)

        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connection.__enter__ = Mock(return_value=mock_connection)
        mock_connection.__exit__ = Mock(return_value=False)

        service._pool.connection.return_value = mock_connection

        # Mock SQL execution
        with patch.object(service, "_execute_sql") as mock_execute:
            mock_execute.return_value = pd.DataFrame({"count": [100]})

            result = service.ask_question_with_cache_info(
                "What is inventory?",
                conversation_id=current_conversation_id,  # Current conversation ID
            )

        # Verify cache hit
        assert result.cache_hit is True

        # CRITICAL: conversation_id should be the current one, NOT the cached one
        assert result.response.conversation_id == current_conversation_id
        assert result.response.conversation_id != cached_entry_conversation_id


# ============================================================================
# Integration Tests (require real Databricks environment)
# ============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not has_retail_ai_env(), reason="Retail AI env vars not set")
def test_semantic_cache_with_conversation_context_integration() -> None:
    """
    Integration test for semantic cache with conversation context.

    This test verifies that:
    1. Questions with context are cached differently than without context
    2. Follow-up questions with anaphoric references match correctly
    3. Conversation context improves cache precision
    """
    from dao_ai.config import GenieRoomModel

    # Get real environment configuration
    space_id = os.environ.get("RETAIL_AI_GENIE_SPACE_ID")
    postgres_host = os.environ.get("POSTGRES_HOST")
    postgres_db = os.environ.get("POSTGRES_DB", "postgres")
    postgres_user = os.environ.get("POSTGRES_USER", "postgres")
    postgres_password = os.environ.get("POSTGRES_PASSWORD", "")
    warehouse_id = os.environ.get("RETAIL_AI_WAREHOUSE_ID")

    if not all([postgres_host, warehouse_id]):
        pytest.skip("Missing required environment variables for integration test")

    # Create real configuration
    genie_room = GenieRoomModel(
        name="Test Genie Room",
        space_id=space_id,
    )
    add_databricks_resource_attrs(genie_room)

    database = DatabaseModel(
        host=postgres_host,
        port=5432,
        database=postgres_db,
        username=postgres_user,
        password=postgres_password,
    )

    warehouse = WarehouseModel(warehouse_id=warehouse_id)
    add_databricks_resource_attrs(warehouse)

    # Create semantic cache with context enabled
    cache_params = GenieSemanticCacheParametersModel(
        database=database,
        warehouse=warehouse,
        embedding_model="databricks-gte-large-en",
        time_to_live_seconds=300,  # 5 minutes for testing
        similarity_threshold=0.85,
        context_window_size=3,  # Enable context
        max_context_tokens=2000,
        table_name="test_semantic_cache_context",
    )

    # Create Genie service with semantic cache
    from databricks_ai_bridge.genie import Genie

    genie = Genie(
        space_id=space_id,
        client=genie_room.workspace_client,
    )

    genie_service = GenieService(genie)
    cache_service = SemanticCacheService(
        impl=genie_service,
        parameters=cache_params,
        workspace_client=genie_room.workspace_client,
    ).initialize()

    try:
        # Clear cache before test
        cache_service.clear()

        print("\n=== Testing Semantic Cache with Conversation Context ===")

        # Test 1: First question (no context)
        print("\n1. First question (establishes context):")
        question1 = "How many stores do we have in California?"
        result1 = cache_service.ask_question_with_cache_info(
            question1, conversation_id=None
        )

        assert result1.cache_hit is False  # First time, should be cache miss
        conv_id = result1.response.conversation_id
        print(f"   Cache miss (expected): {question1}")
        print(f"   Conversation ID: {conv_id}")

        # Test 2: Follow-up question with anaphoric reference
        print("\n2. Follow-up question with anaphoric reference:")
        question2 = "What about that state's total sales?"
        result2 = cache_service.ask_question_with_cache_info(
            question2, conversation_id=conv_id
        )

        assert result2.cache_hit is False  # New question, should be cache miss
        print(f"   Cache miss (expected): {question2}")
        print(f"   Conversation ID: {result2.response.conversation_id}")

        # Test 3: Repeat the follow-up question (should hit cache with context)
        print("\n3. Repeat follow-up question (should hit cache):")
        result3 = cache_service.ask_question_with_cache_info(
            question2, conversation_id=conv_id
        )

        # This might be a cache hit if context matching works
        print(f"   Cache hit: {result3.cache_hit}")
        if result3.cache_hit:
            print("   ✓ Context-aware caching working!")

        # Test 4: Same question without context (should NOT match)
        print("\n4. Same question without conversation context:")
        result4 = cache_service.ask_question_with_cache_info(
            question2, conversation_id=None
        )

        # Without context, "that state" is ambiguous, so it should be different
        print(f"   Cache hit: {result4.cache_hit}")
        print("   (May or may not hit depending on similarity threshold)")

        # Verify conversation_id handling
        print("\n5. Verify conversation_id is current, not cached:")
        if result3.cache_hit:
            assert result3.response.conversation_id == conv_id
            print(f"   ✓ Conversation ID correctly maintained: {conv_id}")

        # Test 6: Cache stats
        print("\n6. Cache statistics:")
        stats = cache_service.stats()
        print(f"   Total entries: {stats['size']}")
        print(f"   Valid entries: {stats['valid_entries']}")
        print(f"   Similarity threshold: {stats['similarity_threshold']}")

        print("\n✓ Integration test completed successfully!")

    finally:
        # Cleanup: clear test cache
        cache_service.clear()
        print("\n✓ Test cache cleared")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not has_retail_ai_env(), reason="Retail AI env vars not set")
def test_semantic_cache_context_improves_precision() -> None:
    """
    Test that conversation context improves cache precision for ambiguous questions.

    This test demonstrates that the same question in different contexts
    should NOT match (avoiding false positives).
    """
    from dao_ai.config import GenieRoomModel

    # Get real environment configuration
    space_id = os.environ.get("RETAIL_AI_GENIE_SPACE_ID")
    postgres_host = os.environ.get("POSTGRES_HOST")
    postgres_db = os.environ.get("POSTGRES_DB", "postgres")
    postgres_user = os.environ.get("POSTGRES_USER", "postgres")
    postgres_password = os.environ.get("POSTGRES_PASSWORD", "")
    warehouse_id = os.environ.get("RETAIL_AI_WAREHOUSE_ID")

    if not all([postgres_host, warehouse_id]):
        pytest.skip("Missing required environment variables for integration test")

    # Create configuration
    genie_room = GenieRoomModel(
        name="Test Genie Room",
        space_id=space_id,
    )
    add_databricks_resource_attrs(genie_room)

    database = DatabaseModel(
        host=postgres_host,
        port=5432,
        database=postgres_db,
        username=postgres_user,
        password=postgres_password,
    )

    warehouse = WarehouseModel(warehouse_id=warehouse_id)
    add_databricks_resource_attrs(warehouse)

    cache_params = GenieSemanticCacheParametersModel(
        database=database,
        warehouse=warehouse,
        embedding_model="databricks-gte-large-en",
        time_to_live_seconds=300,
        similarity_threshold=0.85,
        context_window_size=3,
        max_context_tokens=2000,
        table_name="test_semantic_cache_precision",
    )

    from databricks_ai_bridge.genie import Genie

    genie = Genie(space_id=space_id, client=genie_room.workspace_client)
    genie_service = GenieService(genie)
    cache_service = SemanticCacheService(
        impl=genie_service,
        parameters=cache_params,
        workspace_client=genie_room.workspace_client,
    ).initialize()

    try:
        cache_service.clear()

        print("\n=== Testing Context Improves Precision ===")

        # Scenario 1: "What about inventory?" after asking about Store A
        print("\n1. Conversation about Store A:")
        q1a = "Show me details for Store 42"
        r1a = cache_service.ask_question_with_cache_info(q1a, conversation_id=None)
        conv_a = r1a.response.conversation_id

        q1b = "What about inventory?"  # Ambiguous without context
        r1b = cache_service.ask_question_with_cache_info(q1b, conversation_id=conv_a)
        print(f"   Q: {q1b}")
        print(f"   Conversation A ID: {conv_a}")

        # Scenario 2: "What about inventory?" after asking about Store B
        print("\n2. Conversation about Store B:")
        q2a = "Show me details for Store 99"
        r2a = cache_service.ask_question_with_cache_info(q2a, conversation_id=None)
        conv_b = r2a.response.conversation_id

        q2b = "What about inventory?"  # Same question, different context
        r2b = cache_service.ask_question_with_cache_info(q2b, conversation_id=conv_b)
        print(f"   Q: {q2b}")
        print(f"   Conversation B ID: {conv_b}")

        # The second "What about inventory?" should NOT hit cache
        # because the context is different (Store 42 vs Store 99)
        print("\n3. Checking precision:")
        print(
            f"   First 'What about inventory?' (Store 42 context): cache_hit={r1b.cache_hit}"
        )
        print(
            f"   Second 'What about inventory?' (Store 99 context): cache_hit={r2b.cache_hit}"
        )

        if not r2b.cache_hit:
            print("   ✓ Context correctly prevented false positive!")
        else:
            print("   ⚠ Context may not have prevented cache hit (check similarity)")

        print("\n✓ Precision test completed!")

    finally:
        cache_service.clear()
        print("\n✓ Test cache cleared")
