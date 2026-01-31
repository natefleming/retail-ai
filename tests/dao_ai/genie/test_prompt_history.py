"""Unit and integration tests for prompt history tracking in semantic cache."""

from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from dao_ai.config import (
    DatabaseModel,
    GenieSemanticCacheParametersModel,
    WarehouseModel,
)
from dao_ai.genie.cache import SemanticCacheService
from dao_ai.genie.cache.base import CacheResult
from databricks_ai_bridge.genie import GenieResponse


class TestPromptHistoryStorage:
    """Test prompt history storage functionality."""

    @pytest.fixture
    def mock_database(self) -> DatabaseModel:
        """Create mock database model."""
        # Create a real DatabaseModel instance for testing (Postgres)
        return DatabaseModel(
            name="test_db",
            host="localhost",
            port=5432,
            user="test",
            password="test",
        )

    @pytest.fixture
    def mock_warehouse(self) -> WarehouseModel:
        """Create mock warehouse model."""
        from databricks.sdk import WorkspaceClient
        
        # Create a real WarehouseModel with mocked workspace_client
        with patch('databricks.sdk.WorkspaceClient'):
            return WarehouseModel(
                warehouse_id="test_warehouse",
            )

    @pytest.fixture
    def cache_parameters(self, mock_database: DatabaseModel, mock_warehouse: WarehouseModel) -> GenieSemanticCacheParametersModel:
        """Create cache parameters with prompt history enabled."""
        return GenieSemanticCacheParametersModel(
            database=mock_database,
            warehouse=mock_warehouse,
            embedding_model="databricks-gte-large-en",
            time_to_live_seconds=86400,
            similarity_threshold=0.85,
            context_similarity_threshold=0.80,
            prompt_history_table="test_prompt_history",
            context_window_size=2,
        )

    def test_prompt_history_table_creation(self, cache_parameters: GenieSemanticCacheParametersModel) -> None:
        """Test that prompt history table is created with correct schema."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Mock the connection pool
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_pool = Mock()
        mock_pool.connection.return_value = mock_conn

        with patch('dao_ai.memory.postgres.PostgresPoolManager.get_pool', return_value=mock_pool):
            service = SemanticCacheService(
                impl=mock_impl,
                parameters=cache_parameters,
                workspace_client=None,
            )
            service.initialize()

        # Verify prompt history table creation SQL was called
        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        
        # Check that prompt history table creation was attempted
        prompt_table_creates = [sql for sql in execute_calls if 'test_prompt_history' in sql and 'CREATE TABLE' in sql]
        assert len(prompt_table_creates) > 0, "Prompt history table should be created"
        
        # Verify schema includes required columns
        prompt_table_sql = prompt_table_creates[0]
        assert 'genie_space_id' in prompt_table_sql
        assert 'conversation_id' in prompt_table_sql
        assert 'prompt' in prompt_table_sql
        assert 'cache_hit' in prompt_table_sql
        assert 'created_at' in prompt_table_sql

    def test_store_user_prompt(self, cache_parameters: GenieSemanticCacheParametersModel) -> None:
        """Test storing user prompts in history."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Mock database operations
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_pool = Mock()
        mock_pool.connection.return_value = mock_conn

        service = SemanticCacheService(
            impl=mock_impl,
            parameters=cache_parameters,
            workspace_client=None,
        )
        service._pool = mock_pool
        service._setup_complete = True

        # Store a prompt
        service._store_user_prompt(
            prompt="What are total sales?",
            conversation_id="conv-123",
            cache_hit=False,
        )

        # Verify INSERT was called with correct parameters
        assert mock_cursor.execute.called
        call_args = mock_cursor.execute.call_args[0]
        sql = call_args[0]
        params = call_args[1]
        
        assert 'INSERT INTO test_prompt_history' in sql
        assert params[0] == "test-space"  # genie_space_id
        assert params[1] == "conv-123"    # conversation_id
        assert params[2] == "What are total sales?"  # prompt
        assert params[3] is False  # cache_hit

    def test_get_local_prompt_history(self, cache_parameters: GenieSemanticCacheParametersModel) -> None:
        """Test retrieving prompt history with correct LIMIT."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Mock database operations
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {'prompt': 'Filter by Q1'},
            {'prompt': 'What are total sales?'},
        ]
        
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_pool = Mock()
        mock_pool.connection.return_value = mock_conn

        service = SemanticCacheService(
            impl=mock_impl,
            parameters=cache_parameters,
            workspace_client=None,
        )
        service._pool = mock_pool
        service._setup_complete = True

        # Retrieve history
        prompts = service._get_local_prompt_history(
            conversation_id="conv-123",
            max_prompts=2,
        )

        # Verify SELECT was called with LIMIT
        assert mock_cursor.execute.called
        call_args = mock_cursor.execute.call_args[0]
        sql = call_args[0]
        params = call_args[1]
        
        assert 'SELECT prompt' in sql
        assert 'FROM test_prompt_history' in sql
        assert 'LIMIT' in sql
        assert params[2] == 2  # LIMIT value
        
        # Verify prompts are returned in chronological order
        assert prompts == ['What are total sales?', 'Filter by Q1']

    def test_update_prompt_cache_hit(self, cache_parameters: GenieSemanticCacheParametersModel) -> None:
        """Test updating cache_hit flag for a prompt."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Mock database operations
        mock_cursor = Mock()
        mock_cursor.rowcount = 1
        
        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_pool = Mock()
        mock_pool.connection.return_value = mock_conn

        service = SemanticCacheService(
            impl=mock_impl,
            parameters=cache_parameters,
            workspace_client=None,
        )
        service._pool = mock_pool
        service._setup_complete = True

        # Update cache_hit flag
        service._update_prompt_cache_hit(
            conversation_id="conv-123",
            prompt="Show by region",
            cache_hit=True,
        )

        # Verify UPDATE was called
        assert mock_cursor.execute.called
        call_args = mock_cursor.execute.call_args[0]
        sql = call_args[0]
        params = call_args[1]
        
        assert 'UPDATE test_prompt_history' in sql
        assert 'SET cache_hit' in sql
        assert 'WHERE genie_space_id' in sql
        assert params[0] is True  # new cache_hit value


class TestPromptHistoryContextBuilding:
    """Test context building from prompt history."""

    @pytest.fixture
    def cache_service_with_history(self) -> SemanticCacheService:
        """Create a cache service with mocked prompt history."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Create real model instances
        database = DatabaseModel(
            name="test_db",
            host="localhost",
            port=5432,
        )
        with patch('databricks.sdk.WorkspaceClient'):
            warehouse = WarehouseModel(warehouse_id="test_warehouse")
        
        parameters = GenieSemanticCacheParametersModel(
            database=database,
            warehouse=warehouse,
            context_window_size=2,
        )

        service = SemanticCacheService(
            impl=mock_impl,
            parameters=parameters,
            workspace_client=None,
        )
        
        # Mock the setup and embeddings
        service._setup_complete = True
        service._embedding_dims = 3
        service._embeddings = Mock()
        service._embeddings.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        return service

    def test_context_excludes_current_prompt(self, cache_service_with_history: SemanticCacheService) -> None:
        """Test that context embedding does not include the current prompt."""
        # Mock prompt history retrieval (returns PREVIOUS prompts only)
        with patch.object(cache_service_with_history, '_get_local_prompt_history', return_value=['What are sales?', 'Filter by Q1']):
            question_emb, context_emb, context_str = cache_service_with_history._embed_question(
                question="Show by region",
                conversation_id="conv-123",
            )

        # Verify embeddings were generated
        assert question_emb == [0.1, 0.2, 0.3]
        assert context_emb == [0.4, 0.5, 0.6]
        
        # Verify context string contains PREVIOUS prompts only
        assert "What are sales?" in context_str
        assert "Filter by Q1" in context_str
        assert "Show by region" not in context_str  # Current prompt should NOT be in context
        
        # Verify embed_documents was called with separate question and context
        embed_calls = cache_service_with_history._embeddings.embed_documents.call_args[0][0]
        assert len(embed_calls) == 2
        assert embed_calls[0] == "Show by region"  # Question
        assert "Show by region" not in embed_calls[1]  # Context should not include current

    def test_sliding_window(self, cache_service_with_history: SemanticCacheService) -> None:
        """Test that context window slides correctly (LIMIT behavior)."""
        # Simulate 4 prompts, window_size=2, should return only last 2
        with patch.object(cache_service_with_history, '_get_local_prompt_history') as mock_get:
            # Mock returns only 2 prompts (LIMIT 2 in SQL)
            mock_get.return_value = ['Prompt 3', 'Prompt 4']
            
            _, _, context_str = cache_service_with_history._embed_question(
                question="Prompt 5",
                conversation_id="conv-123",
            )
            
            # Verify only window_size prompts are retrieved
            mock_get.assert_called_once()
            assert mock_get.call_args[1]['max_prompts'] == 2
            
            # Verify context only includes last 2 prompts
            assert "Prompt 3" in context_str
            assert "Prompt 4" in context_str


class TestPromptHistoryIntegration:
    """Integration tests using real Lakebase instance."""

    @pytest.fixture
    def lakebase_database(self) -> DatabaseModel:
        """Create DatabaseModel for retail-consumer-goods Lakebase instance."""
        # For Lakebase, only instance_name is needed
        # workspace_client is created automatically from environment
        return DatabaseModel(
            name="retail-consumer-goods",
            instance_name="retail-consumer-goods",
        )

    @pytest.fixture
    def lakebase_warehouse(self) -> WarehouseModel:
        """Create WarehouseModel for testing."""
        # Use retail-consumer-goods warehouse from config
        # workspace_client is created automatically from environment
        return WarehouseModel(
            warehouse_id="148ccb90800933a1",  # From genie_semantic_cache.yaml
        )

    @pytest.mark.integration
    def test_prompt_history_full_flow_lakebase(
        self,
        lakebase_database: DatabaseModel,
        lakebase_warehouse: WarehouseModel,
    ) -> None:
        """Test full prompt history flow with real Lakebase instance."""
        import os
        from databricks_ai_bridge.genie import Genie
        from dao_ai.genie.core import GenieService
        
        # Use provided Genie space ID for testing
        genie_space_id = "01f0c482e842191587af6a40ad4044d8"
        
        # Use unique table names for this test run to avoid permission issues
        import uuid
        test_suffix = uuid.uuid4().hex[:8]
        
        # Create cache service (prompt history is always enabled)
        parameters = GenieSemanticCacheParametersModel(
            database=lakebase_database,
            warehouse=lakebase_warehouse,
            table_name=f"test_semantic_cache_{test_suffix}",  # Unique cache table
            prompt_history_table=f"test_prompt_history_{test_suffix}",  # Unique history table
            context_window_size=2,
            embedding_model="databricks-gte-large-en",
        )
        
        genie = Genie(space_id=genie_space_id)
        genie_service = GenieService(genie)
        
        cache_service = SemanticCacheService(
            impl=genie_service,
            parameters=parameters,
            workspace_client=lakebase_database.workspace_client,
        ).initialize()
        
        try:
            print(f"\n=== Starting Integration Test ===")
            
            # Prompt 1: First prompt (no context) - Let Genie create conversation
            print("\nPrompt 1: What items do we sell?")
            result1 = cache_service.ask_question(
                "What items do we sell?",
                conversation_id=None,  # Let Genie create new conversation
            )
            test_conv_id = result1.response.conversation_id
            print(f"Result 1: cache_hit={result1.cache_hit}, conversation_id={test_conv_id}")
            
            # Prompt 2: Second prompt (context includes prompt 1) - Use same conversation
            print("\nPrompt 2: Filter by main courses")
            result2 = cache_service.ask_question(
                "Filter by main courses",
                conversation_id=test_conv_id,
            )
            print(f"Result 2: cache_hit={result2.cache_hit}")
            
            # Prompt 3: Different prompt (context includes prompts 1 and 2)
            print("\nPrompt 3: Show by category")
            result3 = cache_service.ask_question(
                "Show by category",
                conversation_id=test_conv_id,
            )
            print(f"Result 3: cache_hit={result3.cache_hit}")
            
            # Verify prompts were stored
            print("\n=== Verifying Prompt History ===")
            prompts = cache_service._get_local_prompt_history(
                conversation_id=test_conv_id,
                max_prompts=10,
            )
            print(f"Stored prompts count: {len(prompts)}")
            for i, p in enumerate(prompts):
                print(f"  {i+1}. {p}")
            assert len(prompts) == 3, f"Expected 3 prompts, got {len(prompts)}"
            assert prompts[0] == "What items do we sell?"
            assert prompts[1] == "Filter by main courses"
            assert prompts[2] == "Show by category"
            print("✅ All prompts stored correctly in order!")
            
            # Test public utility methods
            print("\n=== Testing Utility Methods ===")
            history = cache_service.get_prompt_history(test_conv_id)
            assert len(history) == 3
            print(f"get_prompt_history returned {len(history)} prompts with metadata")
            
            # Test export
            exported_text = cache_service.export_prompt_history(test_conv_id, "text")
            print(f"\nExported history (text):\n{exported_text}")
            
            # Test stats
            stats = cache_service.stats()
            print(f"\n=== Cache Statistics ===")
            print(f"Cache size: {stats['size']}")
            if 'prompt_history' in stats:
                print(f"Total prompts: {stats['prompt_history']['total_prompts']}")
                print(f"Cache hit prompts: {stats['prompt_history']['cache_hit_prompts']}")
                print(f"Cache miss prompts: {stats['prompt_history']['cache_miss_prompts']}")
                print(f"Cache hit rate: {stats['prompt_history']['cache_hit_rate']:.2%}")
            
            print("\n✅ Integration Test PASSED!")
            
        finally:
            # Cleanup: clear test data
            print(f"\n=== Cleaning Up Test Data ===")
            deleted_prompts = cache_service.clear_prompt_history(test_conv_id)
            print(f"Deleted {deleted_prompts} test prompts")
            cache_service.clear()
            print("✅ Cleanup complete")


def test_configuration_validation() -> None:
    """Test that prompt history configuration is validated correctly."""
    # Create real model instances
    database = DatabaseModel(
        name="test_db",
        host="localhost",
    )
    with patch('databricks.sdk.WorkspaceClient'):
        warehouse = WarehouseModel(warehouse_id="test_warehouse")
    
    # Test default values (prompt history is always enabled)
    params = GenieSemanticCacheParametersModel(
        database=database,
        warehouse=warehouse,
    )
    assert params.prompt_history_table == "genie_prompt_history"
    assert params.context_window_size == 3
    
    # Test custom values
    params_custom = GenieSemanticCacheParametersModel(
        database=database,
        warehouse=warehouse,
        prompt_history_table="custom_table",
        context_window_size=5,
    )
    assert params_custom.prompt_history_table == "custom_table"
    assert params_custom.context_window_size == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
