"""Unit tests for cache SQL execution fallback logic."""

from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from databricks_ai_bridge.genie import GenieResponse

from dao_ai.config import (
    GenieInMemorySemanticCacheParametersModel,
    GenieLRUCacheParametersModel,
    GenieSemanticCacheParametersModel,
    WarehouseModel,
)
from dao_ai.genie.cache import (
    InMemoryContextAwareGenieService,
    LRUCacheService,
    PostgresContextAwareGenieService,
)
from dao_ai.genie.cache.base import CacheResult


class TestLRUCacheFallback:
    """Test fallback logic for LRU cache when cached SQL execution fails."""

    @pytest.fixture
    def mock_parameters(self) -> Mock:
        """Create mock LRU cache parameters."""
        params = Mock(spec=GenieLRUCacheParametersModel)
        params.capacity = 100
        params.time_to_live_seconds = 86400
        params.warehouse = Mock(spec=WarehouseModel)
        return params

    def test_fallback_on_sql_execution_failure(self, mock_parameters: Mock) -> None:
        """Test that cache falls back to Genie when cached SQL execution fails."""
        # Create mock implementation
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Create fresh response for fallback
        fresh_response = GenieResponse(
            result=pd.DataFrame({"count": [200]}),
            query="SELECT COUNT(*) as count FROM new_table",
            description="Fresh query after schema change",
            conversation_id="conv-123",
        )
        mock_impl.ask_question.return_value = CacheResult(
            response=fresh_response,
            cache_hit=False,
            served_by=None,
        )

        # Create cache service
        service = LRUCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
        )

        # Pre-populate cache with a valid entry
        stale_response = GenieResponse(
            result=pd.DataFrame({"count": [100]}),
            query="SELECT COUNT(*) as count FROM old_table",
            description="Stale query",
            conversation_id="conv-old",
        )
        key = service._normalize_key("How many records?", None)
        service._put(key, stale_response)

        # Verify cache has entry
        assert service.size == 1

        # Mock _execute_sql to return error (simulating table doesn't exist)
        with patch.object(service, "_execute_sql") as mock_execute:
            mock_execute.return_value = (
                "SQL execution failed: Table 'old_table' not found"
            )

            # Ask the same question - should hit cache but fail, then fallback
            result = service.ask_question_with_cache_info("How many records?", None)

        # Verify fallback occurred
        assert result.cache_hit is False  # Should be False due to fallback
        assert result.served_by is None  # Should be None for fallback
        assert result.response == fresh_response

        # Verify Genie was called as fallback
        mock_impl.ask_question.assert_called_once_with("How many records?", None)

        # Verify cache was updated with fresh SQL
        cached_entry = service._get(key)
        assert cached_entry is not None
        assert cached_entry.query == "SELECT COUNT(*) as count FROM new_table"


class TestPostgresContextAwareCacheFallback:
    """Test fallback logic for PostgresContextAwareGenieService when cached SQL execution fails."""

    @pytest.fixture
    def mock_parameters(self) -> Mock:
        """Create mock semantic cache parameters."""
        params = Mock(spec=GenieSemanticCacheParametersModel)
        params.time_to_live_seconds = 86400
        params.similarity_threshold = 0.85
        params.context_similarity_threshold = 0.80
        params.question_weight = 0.6
        params.context_weight = 0.4
        params.context_window_size = 3
        params.max_context_tokens = 2000
        params.embedding_model = "databricks-gte-large-en"
        params.embedding_dims = 3
        params.table_name = "test_semantic_cache"
        params.warehouse = Mock(spec=WarehouseModel)
        params.database = Mock()
        return params

    def test_fallback_on_sql_execution_failure(self, mock_parameters: Mock) -> None:
        """Test that context-aware cache falls back to Genie when cached SQL execution fails."""
        # Create mock implementation
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Create fresh response for fallback
        fresh_response = GenieResponse(
            result=pd.DataFrame({"count": [300]}),
            query="SELECT COUNT(*) as count FROM updated_table",
            description="Fresh query after permission change",
            conversation_id="conv-456",
        )
        mock_impl.ask_question.return_value = CacheResult(
            response=fresh_response,
            cache_hit=False,
            served_by=None,
        )

        # Create cache service
        service = PostgresContextAwareGenieService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=None,
        )

        # Mock setup components
        service._setup_complete = True
        service._embedding_dims = 3
        service._embeddings = Mock()
        service._embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]

        # Mock the connection pool with proper context manager support
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_pool = Mock()
        mock_pool.connection.return_value = mock_conn
        service._pool = mock_pool

        # Mock _find_similar to return a cached entry
        from dao_ai.genie.cache.base import SQLCacheEntry

        stale_entry = SQLCacheEntry(
            query="SELECT COUNT(*) as count FROM restricted_table",
            description="Stale query with permission issue",
            conversation_id="conv-old",
            created_at=datetime.now(),
        )

        with patch.object(service, "_find_similar") as mock_find:
            mock_find.return_value = (stale_entry, 0.95)  # High similarity

            # Mock _execute_sql to return error (simulating permission denied)
            with patch.object(service, "_execute_sql") as mock_execute:
                mock_execute.return_value = "SQL execution failed: Permission denied on table 'restricted_table'"

                # Mock _store_entry to avoid database operations
                with patch.object(service, "_store_entry"):
                    # Ask question - should hit cache but fail, then fallback
                    result = service.ask_question_with_cache_info(
                        "How many items?", None
                    )

        # Verify fallback occurred
        assert result.cache_hit is False  # Should be False due to fallback
        assert result.served_by is None  # Should be None for fallback
        assert result.response == fresh_response

        # Verify Genie was called as fallback
        mock_impl.ask_question.assert_called_once_with("How many items?", None)


class TestInMemoryContextAwareCacheFallback:
    """Test fallback logic for InMemoryContextAwareGenieService when cached SQL execution fails."""

    @pytest.fixture
    def mock_parameters(self) -> Mock:
        """Create mock in-memory semantic cache parameters."""
        params = Mock(spec=GenieInMemorySemanticCacheParametersModel)
        params.time_to_live_seconds = 86400
        params.similarity_threshold = 0.85
        params.context_similarity_threshold = 0.80
        params.question_weight = 0.6
        params.context_weight = 0.4
        params.context_window_size = 3
        params.max_context_tokens = 2000
        params.embedding_model = "databricks-gte-large-en"
        params.embedding_dims = 3
        params.capacity = 100
        params.warehouse = Mock(spec=WarehouseModel)
        return params

    def test_fallback_on_sql_execution_failure(self, mock_parameters: Mock) -> None:
        """Test that in-memory context-aware cache falls back to Genie when cached SQL execution fails."""
        # Create mock implementation
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Create fresh response for fallback
        fresh_response = GenieResponse(
            result=pd.DataFrame({"total": [500]}),
            query="SELECT SUM(amount) as total FROM current_view",
            description="Fresh query after view redefinition",
            conversation_id="conv-789",
        )
        mock_impl.ask_question.return_value = CacheResult(
            response=fresh_response,
            cache_hit=False,
            served_by=None,
        )

        # Create cache service
        service = InMemoryContextAwareGenieService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=None,
        )

        # Mock setup components
        service._setup_complete = True
        service._embedding_dims = 3
        service._embeddings = Mock()
        service._embeddings.embed_documents.return_value = [[0.4, 0.5, 0.6]]

        # Mock _find_similar to return a cached entry
        from dao_ai.genie.cache.base import SQLCacheEntry

        stale_entry = SQLCacheEntry(
            query="SELECT SUM(amount) as total FROM old_view",
            description="Stale query with dropped view",
            conversation_id="conv-old",
            created_at=datetime.now(),
        )

        with patch.object(service, "_find_similar") as mock_find:
            mock_find.return_value = (stale_entry, 0.92)  # High similarity

            # Mock _execute_sql to return error (simulating view doesn't exist)
            with patch.object(service, "_execute_sql") as mock_execute:
                mock_execute.return_value = (
                    "SQL execution failed: View 'old_view' does not exist"
                )

                # Mock _store_entry to avoid actual storage
                with patch.object(service, "_store_entry"):
                    # Ask question - should hit cache but fail, then fallback
                    result = service.ask_question_with_cache_info(
                        "What's the total?", None
                    )

        # Verify fallback occurred
        assert result.cache_hit is False  # Should be False due to fallback
        assert result.served_by is None  # Should be None for fallback
        assert result.response == fresh_response

        # Verify Genie was called as fallback
        mock_impl.ask_question.assert_called_once_with("What's the total?", None)

    def test_successful_execution_returns_cache_hit(
        self, mock_parameters: Mock
    ) -> None:
        """Test that successful SQL execution returns cache hit (no fallback)."""
        # Create mock implementation
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Create cache service
        service = InMemoryContextAwareGenieService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=None,
        )

        # Mock setup components
        service._setup_complete = True
        service._embedding_dims = 3
        service._embeddings = Mock()
        service._embeddings.embed_documents.return_value = [[0.7, 0.8, 0.9]]

        # Mock _find_similar to return a cached entry
        from dao_ai.genie.cache.base import SQLCacheEntry

        valid_entry = SQLCacheEntry(
            query="SELECT COUNT(*) FROM products",
            description="Valid query",
            conversation_id="conv-123",
            created_at=datetime.now(),
        )

        with patch.object(service, "_find_similar") as mock_find:
            mock_find.return_value = (valid_entry, 0.95)

            # Mock _execute_sql to return successful result (DataFrame)
            with patch.object(service, "_execute_sql") as mock_execute:
                mock_execute.return_value = pd.DataFrame({"count": [100]})

                # Ask question - should hit cache and succeed
                result = service.ask_question_with_cache_info(
                    "How many products?", None
                )

        # Verify cache hit (no fallback)
        assert result.cache_hit is True
        assert result.served_by == service.name
        assert isinstance(result.response.result, pd.DataFrame)

        # Verify Genie was NOT called (no fallback needed)
        mock_impl.ask_question.assert_not_called()
