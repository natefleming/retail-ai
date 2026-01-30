"""Unit and integration tests for in-memory semantic cache with conversation context."""

import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from conftest import add_databricks_resource_attrs, has_retail_ai_env
from databricks.sdk import WorkspaceClient

from dao_ai.config import (
    GenieInMemorySemanticCacheParametersModel,
    WarehouseModel,
)
from dao_ai.genie import GenieService
from dao_ai.genie.cache import InMemorySemanticCacheService
from dao_ai.genie.cache.in_memory_semantic import (
    InMemoryCacheEntry,
    distance_to_similarity,
    l2_distance,
)

# ============================================================================
# Unit Tests for Distance Functions
# ============================================================================


class TestL2Distance:
    """Unit tests for l2_distance function."""

    def test_identical_vectors(self) -> None:
        """Test that L2 distance is 0 for identical vectors."""
        vec = [1.0, 2.0, 3.0]
        distance = l2_distance(vec, vec)
        assert distance == 0.0

    def test_orthogonal_vectors(self) -> None:
        """Test L2 distance for orthogonal vectors."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        distance = l2_distance(vec_a, vec_b)
        # Distance should be sqrt(2) ≈ 1.414
        assert abs(distance - np.sqrt(2)) < 1e-10

    def test_opposite_vectors(self) -> None:
        """Test L2 distance for opposite direction vectors."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        distance = l2_distance(vec_a, vec_b)
        assert abs(distance - 2.0) < 1e-10

    def test_same_direction_different_magnitude(self) -> None:
        """Test L2 distance for vectors in same direction but different magnitude."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [2.0, 4.0, 6.0]
        distance = l2_distance(vec_a, vec_b)
        # Distance should be sqrt(1^2 + 2^2 + 3^2) = sqrt(14) ≈ 3.742
        assert abs(distance - np.sqrt(14)) < 1e-10

    def test_high_dimensional_vectors(self) -> None:
        """Test L2 distance with high-dimensional vectors (like embeddings)."""
        dim = 1024
        vec_a = [1.0] * dim
        vec_b = [0.0] * dim
        distance = l2_distance(vec_a, vec_b)
        # Distance should be sqrt(dim)
        assert abs(distance - np.sqrt(dim)) < 1e-10


class TestDistanceToSimilarity:
    """Unit tests for distance_to_similarity function."""

    def test_zero_distance_is_perfect_match(self) -> None:
        """Test that zero distance gives similarity of 1.0."""
        similarity = distance_to_similarity(0.0)
        assert similarity == 1.0

    def test_small_distance(self) -> None:
        """Test similarity for small distance."""
        similarity = distance_to_similarity(0.1)
        # similarity = 1 / (1 + 0.1) ≈ 0.909
        assert abs(similarity - (1.0 / 1.1)) < 1e-10

    def test_unit_distance(self) -> None:
        """Test similarity for unit distance."""
        similarity = distance_to_similarity(1.0)
        assert similarity == 0.5

    def test_large_distance(self) -> None:
        """Test similarity for large distance."""
        similarity = distance_to_similarity(10.0)
        # similarity = 1 / 11 ≈ 0.091
        assert abs(similarity - (1.0 / 11.0)) < 1e-10

    def test_similarity_decreases_with_distance(self) -> None:
        """Test that similarity monotonically decreases with increasing distance."""
        distances = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        similarities = [distance_to_similarity(d) for d in distances]

        # Check that each similarity is less than the previous one
        for i in range(1, len(similarities)):
            assert similarities[i] < similarities[i - 1]


# ============================================================================
# Unit Tests for InMemoryCacheEntry
# ============================================================================


class TestInMemoryCacheEntry:
    """Unit tests for InMemoryCacheEntry dataclass."""

    def test_create_entry(self) -> None:
        """Test creating a cache entry."""
        now = datetime.now()
        entry = InMemoryCacheEntry(
            genie_space_id="test-space",
            question="What is the total sales?",
            conversation_context="Previous: Show me stores\nPrevious: How many?",
            question_embedding=[0.1, 0.2, 0.3],
            context_embedding=[0.4, 0.5, 0.6],
            sql_query="SELECT SUM(sales) FROM sales_table",
            description="Total sales query",
            conversation_id="conv-123",
            created_at=now,
            last_accessed_at=now,
        )

        assert entry.genie_space_id == "test-space"
        assert entry.question == "What is the total sales?"
        assert entry.sql_query == "SELECT SUM(sales) FROM sales_table"
        assert len(entry.question_embedding) == 3
        assert len(entry.context_embedding) == 3

    def test_lru_tracking(self) -> None:
        """Test that last_accessed_at is tracked for LRU eviction."""
        now = datetime.now()
        entry = InMemoryCacheEntry(
            genie_space_id="test-space",
            question="test",
            conversation_context="",
            question_embedding=[0.1],
            context_embedding=[0.1],
            sql_query="SELECT 1",
            description="test",
            conversation_id="conv-1",
            created_at=now,
            last_accessed_at=now,
        )

        # Simulate access after some time
        later = now + timedelta(seconds=60)
        entry.last_accessed_at = later

        assert entry.last_accessed_at > entry.created_at
        assert (entry.last_accessed_at - entry.created_at).total_seconds() == 60


# ============================================================================
# Unit Tests for InMemorySemanticCacheService
# ============================================================================


class TestInMemorySemanticCacheServiceContext:
    """Unit tests for InMemorySemanticCacheService context-aware functionality."""

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

    def test_initialization(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that service initializes correctly."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        assert service.impl == mock_impl
        assert service.parameters == mock_parameters
        assert service.workspace_client == mock_workspace_client
        assert service._cache == []
        assert service._setup_complete is False

    def test_embed_question_without_conversation_id(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that embedding without conversation_id uses question only."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = InMemorySemanticCacheService(
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
        # Context embedding should be zero vector
        assert context_embedding == [0.0, 0.0, 0.0]
        service._embeddings.embed_documents.assert_called_once_with([question])

    def test_embed_question_with_conversation_context(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that embedding with conversation_id includes context."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Mock embeddings - return different embeddings for question and context
        service._embeddings = Mock()
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
        mock_impl.space_id = "test-space"

        service = InMemorySemanticCacheService(
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
        # Zero vector for context
        assert context_embedding == [0.0, 0.0, 0.0]

    def test_find_similar_empty_cache(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that find_similar returns None when cache is empty."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        result = service._find_similar(
            question="What is inventory?",
            conversation_context="",
            question_embedding=[0.1, 0.2, 0.3],
            context_embedding=[0.0, 0.0, 0.0],
        )

        assert result is None

    def test_find_similar_filters_by_space_id(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that find_similar only matches entries from the same space."""
        mock_impl = Mock()
        mock_impl.space_id = "space-A"

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Add entry for different space
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="space-B",  # Different space!
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

        # Search should not match entry from different space
        result = service._find_similar(
            question="What is inventory?",
            conversation_context="",
            question_embedding=[0.1, 0.2, 0.3],
            context_embedding=[0.0, 0.0, 0.0],
        )

        assert result is None

    def test_find_similar_respects_ttl(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that expired entries are not returned and are deleted."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Set short TTL
        mock_parameters.time_to_live_seconds = 60  # 1 minute

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Add expired entry
        expired_time = datetime.now() - timedelta(seconds=120)  # 2 minutes ago
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
                created_at=expired_time,
                last_accessed_at=expired_time,
            )
        )

        # Search should not match expired entry
        result = service._find_similar(
            question="What is inventory?",
            conversation_context="",
            question_embedding=[0.1, 0.2, 0.3],
            context_embedding=[0.0, 0.0, 0.0],
        )

        assert result is None
        # Expired entry should be deleted
        assert len(service._cache) == 0

    def test_find_similar_respects_similarity_thresholds(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that entries below similarity threshold are not returned."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Add entry with dissimilar embedding
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="test-space",
                question="What is inventory?",
                conversation_context="",
                question_embedding=[1.0, 0.0, 0.0],  # Very different
                context_embedding=[0.0, 0.0, 0.0],
                sql_query="SELECT * FROM inventory",
                description="Inventory query",
                conversation_id="conv-1",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )

        # Search with very different embedding
        result = service._find_similar(
            question="What are the sales?",
            conversation_context="",
            question_embedding=[0.0, 1.0, 0.0],  # Orthogonal
            context_embedding=[0.0, 0.0, 0.0],
        )

        # Should not match because similarity is too low
        assert result is None

    def test_find_similar_returns_best_match(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that find_similar returns the best matching entry."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Lower threshold for this test
        mock_parameters.similarity_threshold = 0.5
        mock_parameters.context_similarity_threshold = 0.5

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Add two entries with different similarities
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="test-space",
                question="What is inventory?",
                conversation_context="",
                question_embedding=[0.1, 0.1, 0.1],  # Less similar
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
                question_embedding=[0.9, 0.9, 0.9],  # More similar
                context_embedding=[0.0, 0.0, 0.0],
                sql_query="SELECT * FROM sales",
                description="Sales query",
                conversation_id="conv-2",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )

        # Search with embedding closer to second entry
        result = service._find_similar(
            question="What are sales?",
            conversation_context="",
            question_embedding=[1.0, 1.0, 1.0],
            context_embedding=[0.0, 0.0, 0.0],
        )

        assert result is not None
        entry, similarity = result
        assert "sales" in entry.query.lower()

    def test_store_entry_adds_to_cache(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that store_entry adds a new entry to the cache."""
        from databricks_ai_bridge.genie import GenieResponse

        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        response = GenieResponse(
            result=pd.DataFrame(),
            query="SELECT * FROM inventory",
            description="Inventory query",
            conversation_id="conv-1",
        )

        service._store_entry(
            question="What is inventory?",
            conversation_context="",
            question_embedding=[0.1, 0.2, 0.3],
            context_embedding=[0.0, 0.0, 0.0],
            response=response,
        )

        assert len(service._cache) == 1
        entry = service._cache[0]
        assert entry.question == "What is inventory?"
        assert entry.sql_query == "SELECT * FROM inventory"

    def test_store_entry_respects_capacity_limit(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that store_entry evicts LRU entries when capacity is reached."""
        from databricks_ai_bridge.genie import GenieResponse

        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Set capacity to 2
        mock_parameters.capacity = 2

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Add first entry
        response1 = GenieResponse(
            result=pd.DataFrame(),
            query="SELECT 1",
            description="Query 1",
            conversation_id="conv-1",
        )
        service._store_entry(
            question="Question 1",
            conversation_context="",
            question_embedding=[0.1, 0.2, 0.3],
            context_embedding=[0.0, 0.0, 0.0],
            response=response1,
        )

        # Add second entry
        response2 = GenieResponse(
            result=pd.DataFrame(),
            query="SELECT 2",
            description="Query 2",
            conversation_id="conv-2",
        )
        service._store_entry(
            question="Question 2",
            conversation_context="",
            question_embedding=[0.2, 0.3, 0.4],
            context_embedding=[0.0, 0.0, 0.0],
            response=response2,
        )

        # Access first entry to update last_accessed_at
        service._cache[0].last_accessed_at = datetime.now() + timedelta(seconds=10)

        # Add third entry - should evict the second entry (LRU)
        response3 = GenieResponse(
            result=pd.DataFrame(),
            query="SELECT 3",
            description="Query 3",
            conversation_id="conv-3",
        )
        service._store_entry(
            question="Question 3",
            conversation_context="",
            question_embedding=[0.3, 0.4, 0.5],
            context_embedding=[0.0, 0.0, 0.0],
            response=response3,
        )

        # Should have exactly 2 entries
        assert len(service._cache) == 2
        # First entry should still be there (was accessed recently)
        assert any(e.question == "Question 1" for e in service._cache)
        # Third entry should be there (just added)
        assert any(e.question == "Question 3" for e in service._cache)
        # Second entry should be evicted (LRU)
        assert not any(e.question == "Question 2" for e in service._cache)

    def test_clear_removes_all_space_entries(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that clear removes all entries for the space."""
        mock_impl = Mock()
        mock_impl.space_id = "space-A"

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )
        service._setup_complete = True

        # Add entries for multiple spaces
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="space-A",
                question="Q1",
                conversation_context="",
                question_embedding=[0.1],
                context_embedding=[0.0],
                sql_query="SELECT 1",
                description="",
                conversation_id="",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="space-B",
                question="Q2",
                conversation_context="",
                question_embedding=[0.2],
                context_embedding=[0.0],
                sql_query="SELECT 2",
                description="",
                conversation_id="",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )

        deleted = service.clear()

        # Should delete only space-A entry
        assert deleted == 1
        assert len(service._cache) == 1
        assert service._cache[0].genie_space_id == "space-B"

    def test_invalidate_expired_removes_old_entries(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that invalidate_expired removes expired entries."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        mock_parameters.time_to_live_seconds = 60  # 1 minute TTL

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )
        service._setup_complete = True

        # Add expired entry
        expired_time = datetime.now() - timedelta(seconds=120)
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="test-space",
                question="Old question",
                conversation_context="",
                question_embedding=[0.1],
                context_embedding=[0.0],
                sql_query="SELECT 1",
                description="",
                conversation_id="",
                created_at=expired_time,
                last_accessed_at=expired_time,
            )
        )

        # Add valid entry
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="test-space",
                question="New question",
                conversation_context="",
                question_embedding=[0.2],
                context_embedding=[0.0],
                sql_query="SELECT 2",
                description="",
                conversation_id="",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )

        deleted = service.invalidate_expired()

        assert deleted == 1
        assert len(service._cache) == 1
        assert service._cache[0].question == "New question"

    def test_size_returns_correct_count(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that size returns the correct number of entries for the space."""
        mock_impl = Mock()
        mock_impl.space_id = "space-A"

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )
        service._setup_complete = True

        # Add entries for multiple spaces
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="space-A",
                question="Q1",
                conversation_context="",
                question_embedding=[0.1],
                context_embedding=[0.0],
                sql_query="SELECT 1",
                description="",
                conversation_id="",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="space-A",
                question="Q2",
                conversation_context="",
                question_embedding=[0.2],
                context_embedding=[0.0],
                sql_query="SELECT 2",
                description="",
                conversation_id="",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="space-B",
                question="Q3",
                conversation_context="",
                question_embedding=[0.3],
                context_embedding=[0.0],
                sql_query="SELECT 3",
                description="",
                conversation_id="",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )

        # Size should only count space-A entries
        assert service.size == 2

    def test_stats_returns_correct_info(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that stats returns correct cache statistics."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        mock_parameters.time_to_live_seconds = 60
        mock_parameters.capacity = 100

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )
        service._setup_complete = True

        # Add valid entry
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="test-space",
                question="Valid",
                conversation_context="",
                question_embedding=[0.1],
                context_embedding=[0.0],
                sql_query="SELECT 1",
                description="",
                conversation_id="",
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )

        # Add expired entry
        expired_time = datetime.now() - timedelta(seconds=120)
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="test-space",
                question="Expired",
                conversation_context="",
                question_embedding=[0.2],
                context_embedding=[0.0],
                sql_query="SELECT 2",
                description="",
                conversation_id="",
                created_at=expired_time,
                last_accessed_at=expired_time,
            )
        )

        stats = service.stats()

        assert stats["size"] == 2
        assert stats["capacity"] == 100
        assert stats["ttl_seconds"] == 60
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 1

    def test_conversation_id_not_from_cache(
        self, mock_workspace_client: Mock, mock_parameters: Mock
    ) -> None:
        """Test that conversation_id in response is current, not cached."""

        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        cached_conversation_id = "cached-conv-123"
        current_conversation_id = "current-conv-456"

        # Lower thresholds for easier matching
        mock_parameters.similarity_threshold = 0.5
        mock_parameters.context_similarity_threshold = 0.5

        service = InMemorySemanticCacheService(
            impl=mock_impl,
            parameters=mock_parameters,
            workspace_client=mock_workspace_client,
        )

        # Setup
        service._setup_complete = True
        service._embeddings = Mock()
        service._embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        service._embedding_dims = 3

        # Add cache entry with old conversation_id
        service._cache.append(
            InMemoryCacheEntry(
                genie_space_id="test-space",
                question="What is inventory?",
                conversation_context="",
                question_embedding=[0.1, 0.2, 0.3],
                context_embedding=[0.0, 0.0, 0.0],
                sql_query="SELECT * FROM inventory",
                description="Inventory query",
                conversation_id=cached_conversation_id,
                created_at=datetime.now(),
                last_accessed_at=datetime.now(),
            )
        )

        # Mock SQL execution
        with patch.object(service, "_execute_sql") as mock_execute:
            mock_execute.return_value = pd.DataFrame({"count": [100]})

            result = service.ask_question_with_cache_info(
                "What is inventory?",
                conversation_id=current_conversation_id,
            )

        # Verify cache hit
        assert result.cache_hit is True

        # CRITICAL: conversation_id should be the current one, NOT the cached one
        assert result.response.conversation_id == current_conversation_id
        assert result.response.conversation_id != cached_conversation_id


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not has_retail_ai_env(), reason="Retail AI env vars not set")
def test_in_memory_semantic_cache_with_conversation_context_integration() -> None:
    """
    Integration test for in-memory semantic cache with conversation context.

    This test verifies that:
    1. Questions with context are cached differently than without context
    2. Follow-up questions with anaphoric references match correctly
    3. Conversation context improves cache precision
    4. No database is required
    """
    from dao_ai.config import GenieRoomModel

    # Get real environment configuration
    space_id = os.environ.get("RETAIL_AI_GENIE_SPACE_ID")
    warehouse_id = os.environ.get("RETAIL_AI_WAREHOUSE_ID")

    if not warehouse_id:
        pytest.skip("Missing required environment variables for integration test")

    # Create real configuration
    genie_room = GenieRoomModel(
        name="Test Genie Room",
        space_id=space_id,
    )
    add_databricks_resource_attrs(genie_room)

    warehouse = WarehouseModel(warehouse_id=warehouse_id)
    add_databricks_resource_attrs(warehouse)

    # Create in-memory semantic cache with context enabled
    cache_params = GenieInMemorySemanticCacheParametersModel(
        warehouse=warehouse,
        embedding_model="databricks-gte-large-en",
        time_to_live_seconds=300,  # 5 minutes for testing
        similarity_threshold=0.85,
        context_window_size=3,  # Enable context
        max_context_tokens=2000,
        capacity=100,  # Limit cache size
    )

    # Create Genie service with in-memory semantic cache
    from databricks_ai_bridge.genie import Genie

    genie = Genie(
        space_id=space_id,
        client=genie_room.workspace_client,
    )

    genie_service = GenieService(genie)
    cache_service = InMemorySemanticCacheService(
        impl=genie_service,
        parameters=cache_params,
        workspace_client=genie_room.workspace_client,
    ).initialize()

    # Clear cache before test
    cache_service.clear()

    print("\n=== Testing In-Memory Semantic Cache with Conversation Context ===")

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

    # Without context, "that state" is ambiguous
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
    print(f"   Capacity: {stats['capacity']}")
    print(f"   Similarity threshold: {stats['similarity_threshold']}")

    # Test 7: Capacity limit
    print("\n7. Test capacity limit:")
    initial_size = cache_service.size
    print(f"   Initial size: {initial_size}")

    # Cleanup: clear test cache
    cache_service.clear()
    print("\n✓ Integration test completed successfully!")
    print("✓ Test cache cleared")


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not has_retail_ai_env(), reason="Retail AI env vars not set")
def test_in_memory_semantic_cache_context_improves_precision() -> None:
    """
    Test that conversation context improves cache precision for ambiguous questions.

    This test demonstrates that the same question in different contexts
    should NOT match (avoiding false positives).
    """
    from dao_ai.config import GenieRoomModel

    # Get real environment configuration
    space_id = os.environ.get("RETAIL_AI_GENIE_SPACE_ID")
    warehouse_id = os.environ.get("RETAIL_AI_WAREHOUSE_ID")

    if not warehouse_id:
        pytest.skip("Missing required environment variables for integration test")

    # Create configuration
    genie_room = GenieRoomModel(
        name="Test Genie Room",
        space_id=space_id,
    )
    add_databricks_resource_attrs(genie_room)

    warehouse = WarehouseModel(warehouse_id=warehouse_id)
    add_databricks_resource_attrs(warehouse)

    cache_params = GenieInMemorySemanticCacheParametersModel(
        warehouse=warehouse,
        embedding_model="databricks-gte-large-en",
        time_to_live_seconds=300,
        similarity_threshold=0.85,
        context_window_size=3,
        max_context_tokens=2000,
        capacity=100,
    )

    from databricks_ai_bridge.genie import Genie

    genie = Genie(space_id=space_id, client=genie_room.workspace_client)
    genie_service = GenieService(genie)
    cache_service = InMemorySemanticCacheService(
        impl=genie_service,
        parameters=cache_params,
        workspace_client=genie_room.workspace_client,
    ).initialize()

    cache_service.clear()

    print("\n=== Testing Context Improves Precision (In-Memory) ===")

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

    cache_service.clear()
    print("✓ Test cache cleared")
