"""Unit and integration tests for prompt history tracking in semantic cache."""

from unittest.mock import Mock, patch

import pytest

from dao_ai.config import (
    DatabaseModel,
    GenieSemanticCacheParametersModel,
    WarehouseModel,
)
from dao_ai.genie.cache import SemanticCacheService


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

        # Create a real WarehouseModel with mocked workspace_client
        with patch("databricks.sdk.WorkspaceClient"):
            return WarehouseModel(
                warehouse_id="test_warehouse",
            )

    @pytest.fixture
    def cache_parameters(
        self, mock_database: DatabaseModel, mock_warehouse: WarehouseModel
    ) -> GenieSemanticCacheParametersModel:
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

    def test_prompt_history_table_creation(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
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

        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=mock_pool,
        ):
            service = SemanticCacheService(
                impl=mock_impl,
                parameters=cache_parameters,
                workspace_client=None,
            )
            service.initialize()

        # Verify prompt history table creation SQL was called
        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]

        # Check that prompt history table creation was attempted
        prompt_table_creates = [
            sql
            for sql in execute_calls
            if "test_prompt_history" in sql and "CREATE TABLE" in sql
        ]
        assert len(prompt_table_creates) > 0, "Prompt history table should be created"

        # Verify schema includes required columns
        prompt_table_sql = prompt_table_creates[0]
        assert "genie_space_id" in prompt_table_sql
        assert "conversation_id" in prompt_table_sql
        assert "prompt" in prompt_table_sql
        assert "cache_hit" in prompt_table_sql
        assert "created_at" in prompt_table_sql

    def test_store_user_prompt(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
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
        # Note: There may be additional calls (like _enforce_prompt_history_limit)
        assert mock_cursor.execute.called

        # Find the INSERT call among all execute calls
        insert_call = None
        for call in mock_cursor.execute.call_args_list:
            sql = call[0][0]
            if "INSERT INTO test_prompt_history" in sql:
                insert_call = call
                break

        assert insert_call is not None, (
            "INSERT INTO test_prompt_history not found in execute calls"
        )
        sql = insert_call[0][0]
        params = insert_call[0][1]

        assert "INSERT INTO test_prompt_history" in sql
        assert params[0] == "test-space"  # genie_space_id
        assert params[1] == "conv-123"  # conversation_id
        assert params[2] == "What are total sales?"  # prompt
        assert params[3] is False  # cache_hit

    def test_get_local_prompt_history(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
        """Test retrieving prompt history with correct LIMIT."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Mock database operations
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {"prompt": "Filter by Q1"},
            {"prompt": "What are total sales?"},
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

        assert "SELECT prompt" in sql
        assert "FROM test_prompt_history" in sql
        assert "LIMIT" in sql
        assert params[2] == 2  # LIMIT value

        # Verify prompts are returned in chronological order
        assert prompts == ["What are total sales?", "Filter by Q1"]

    def test_update_prompt_cache_hit(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
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

        assert "UPDATE test_prompt_history" in sql
        assert "SET cache_hit" in sql
        assert "WHERE genie_space_id" in sql
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
        with patch("databricks.sdk.WorkspaceClient"):
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
        service._embeddings.embed_documents = Mock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        return service

    def test_context_excludes_current_prompt(
        self, cache_service_with_history: SemanticCacheService
    ) -> None:
        """Test that context embedding does not include the current prompt."""
        # Mock prompt history retrieval (returns PREVIOUS prompts only)
        with patch.object(
            cache_service_with_history,
            "_get_local_prompt_history",
            return_value=["What are sales?", "Filter by Q1"],
        ):
            question_emb, context_emb, context_str = (
                cache_service_with_history._embed_question(
                    question="Show by region",
                    conversation_id="conv-123",
                )
            )

        # Verify embeddings were generated
        assert question_emb == [0.1, 0.2, 0.3]
        assert context_emb == [0.4, 0.5, 0.6]

        # Verify context string contains PREVIOUS prompts only
        assert "What are sales?" in context_str
        assert "Filter by Q1" in context_str
        assert (
            "Show by region" not in context_str
        )  # Current prompt should NOT be in context

        # Verify embed_documents was called with separate question and context
        embed_calls = cache_service_with_history._embeddings.embed_documents.call_args[
            0
        ][0]
        assert len(embed_calls) == 2
        assert embed_calls[0] == "Show by region"  # Question
        assert (
            "Show by region" not in embed_calls[1]
        )  # Context should not include current

    def test_sliding_window(
        self, cache_service_with_history: SemanticCacheService
    ) -> None:
        """Test that context window slides correctly (LIMIT behavior)."""
        # Simulate 4 prompts, window_size=2, should return only last 2
        with patch.object(
            cache_service_with_history, "_get_local_prompt_history"
        ) as mock_get:
            # Mock returns only 2 prompts (LIMIT 2 in SQL)
            mock_get.return_value = ["Prompt 3", "Prompt 4"]

            _, _, context_str = cache_service_with_history._embed_question(
                question="Prompt 5",
                conversation_id="conv-123",
            )

            # Verify only window_size prompts are retrieved
            mock_get.assert_called_once()
            assert mock_get.call_args[1]["max_prompts"] == 2

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
            print("\n=== Starting Integration Test ===")

            # Prompt 1: First prompt (no context) - Let Genie create conversation
            print("\nPrompt 1: What items do we sell?")
            result1 = cache_service.ask_question(
                "What items do we sell?",
                conversation_id=None,  # Let Genie create new conversation
            )
            test_conv_id = result1.response.conversation_id
            print(
                f"Result 1: cache_hit={result1.cache_hit}, conversation_id={test_conv_id}"
            )

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
                print(f"  {i + 1}. {p}")
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
            print("\n=== Cache Statistics ===")
            print(f"Cache size: {stats['size']}")
            if "prompt_history" in stats:
                print(f"Total prompts: {stats['prompt_history']['total_prompts']}")
                print(
                    f"Cache hit prompts: {stats['prompt_history']['cache_hit_prompts']}"
                )
                print(
                    f"Cache miss prompts: {stats['prompt_history']['cache_miss_prompts']}"
                )
                print(
                    f"Cache hit rate: {stats['prompt_history']['cache_hit_rate']:.2%}"
                )

            print("\n✅ Integration Test PASSED!")

        finally:
            # Cleanup: clear test data
            print("\n=== Cleaning Up Test Data ===")
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
    with patch("databricks.sdk.WorkspaceClient"):
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


class TestFromSpace:
    """Tests for from_space() method that imports from Genie space."""

    @pytest.fixture
    def mock_database(self) -> DatabaseModel:
        """Create mock database model."""
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
        with patch("databricks.sdk.WorkspaceClient"):
            return WarehouseModel(warehouse_id="test_warehouse")

    @pytest.fixture
    def cache_parameters(
        self, mock_database: DatabaseModel, mock_warehouse: WarehouseModel
    ) -> GenieSemanticCacheParametersModel:
        """Create cache parameters."""
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

    def test_from_space_requires_workspace_client(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
        """Test that from_space raises error if workspace_client is None."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        service = SemanticCacheService(
            impl=mock_impl,
            parameters=cache_parameters,
            workspace_client=None,  # No workspace client
        )

        with pytest.raises(ValueError) as exc_info:
            service.from_space()

        assert "workspace_client is required" in str(exc_info.value)

    def test_from_space_uses_self_space_id_when_none(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
        """Test that from_space uses self.space_id when space_id is None."""
        mock_impl = Mock()
        mock_impl.space_id = "default-space-id"

        mock_workspace_client = Mock()
        # Return empty conversations list
        mock_workspace_client.genie.list_conversations.return_value = Mock(
            conversations=[], next_page_token=None
        )

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
            workspace_client=mock_workspace_client,
        )
        service._pool = mock_pool
        service._setup_complete = True

        result = service.from_space()  # No space_id provided

        # Should use self.space_id
        mock_workspace_client.genie.list_conversations.assert_called_once()
        call_kwargs = mock_workspace_client.genie.list_conversations.call_args[1]
        assert call_kwargs["space_id"] == "default-space-id"
        assert result is service  # Returns self

    def test_from_space_returns_self(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
        """Test that from_space returns self for method chaining."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        mock_workspace_client = Mock()
        mock_workspace_client.genie.list_conversations.return_value = Mock(
            conversations=[], next_page_token=None
        )

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
            workspace_client=mock_workspace_client,
        )
        service._pool = mock_pool
        service._setup_complete = True

        result = service.from_space()

        assert result is service

    def test_from_space_fetches_all_conversations(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
        """Test that from_space paginates through all conversations."""
        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        # Mock conversations with pagination
        mock_conv1 = Mock()
        mock_conv1.conversation_id = "conv-1"
        mock_conv2 = Mock()
        mock_conv2.conversation_id = "conv-2"

        mock_workspace_client = Mock()
        mock_workspace_client.genie.list_conversations.side_effect = [
            Mock(conversations=[mock_conv1], next_page_token="page2"),
            Mock(conversations=[mock_conv2], next_page_token=None),
        ]
        mock_workspace_client.genie.list_conversation_messages.return_value = Mock(
            messages=[]
        )

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
            workspace_client=mock_workspace_client,
        )
        service._pool = mock_pool
        service._setup_complete = True

        service.from_space()

        # Should have called list_conversations twice (pagination)
        assert mock_workspace_client.genie.list_conversations.call_count == 2

    def test_from_space_stores_prompts_in_history(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
        """Test that from_space stores prompts in history table."""
        from datetime import datetime, timezone

        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        mock_conv = Mock()
        mock_conv.conversation_id = "conv-1"

        mock_message = Mock()
        mock_message.content = "What are total sales?"
        mock_message.created_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        mock_message.attachments = None

        mock_workspace_client = Mock()
        mock_workspace_client.genie.list_conversations.return_value = Mock(
            conversations=[mock_conv], next_page_token=None
        )
        mock_workspace_client.genie.list_conversation_messages.return_value = Mock(
            messages=[mock_message]
        )

        mock_cursor = Mock()
        mock_cursor.rowcount = 1  # Indicate successful insert

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
            workspace_client=mock_workspace_client,
        )
        service._pool = mock_pool
        service._setup_complete = True

        service.from_space()

        # Verify INSERT with ON CONFLICT was called
        insert_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "INSERT INTO test_prompt_history" in str(call)
            and "ON CONFLICT" in str(call[0][0])
        ]
        assert len(insert_calls) > 0, "Should have called INSERT with ON CONFLICT"

    def test_from_space_stores_sql_in_cache(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
        """Test that from_space stores messages with SQL attachments in cache."""
        from datetime import datetime, timezone

        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        mock_conv = Mock()
        mock_conv.conversation_id = "conv-1"

        # Message with SQL attachment
        mock_query = Mock()
        mock_query.query = "SELECT * FROM sales"
        mock_query.description = "Get all sales"

        mock_attachment = Mock()
        mock_attachment.query = mock_query

        mock_message = Mock()
        mock_message.content = "Show me all sales"
        mock_message.created_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        mock_message.attachments = [mock_attachment]

        mock_workspace_client = Mock()
        mock_workspace_client.genie.list_conversations.return_value = Mock(
            conversations=[mock_conv], next_page_token=None
        )
        mock_workspace_client.genie.list_conversation_messages.return_value = Mock(
            messages=[mock_message]
        )

        mock_cursor = Mock()
        mock_cursor.rowcount = 1

        mock_conn = Mock()
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_pool = Mock()
        mock_pool.connection.return_value = mock_conn

        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 768

        service = SemanticCacheService(
            impl=mock_impl,
            parameters=cache_parameters,
            workspace_client=mock_workspace_client,
        )
        service._pool = mock_pool
        service._setup_complete = True
        service._embeddings = mock_embeddings

        service.from_space()

        # Verify embeddings were generated
        assert mock_embeddings.embed_query.called

        # Verify cache INSERT with ON CONFLICT was called
        insert_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "INSERT INTO" in str(call[0][0])
            and "sql_query" in str(call[0][0])
            and "ON CONFLICT" in str(call[0][0])
        ]
        assert len(insert_calls) > 0, "Should have called cache INSERT with ON CONFLICT"

    def test_from_space_skips_duplicates(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
        """Test that from_space skips duplicate entries (rowcount=0)."""
        from datetime import datetime, timezone

        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        mock_conv = Mock()
        mock_conv.conversation_id = "conv-1"

        mock_message = Mock()
        mock_message.content = "Duplicate prompt"
        mock_message.created_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        mock_message.attachments = None

        mock_workspace_client = Mock()
        mock_workspace_client.genie.list_conversations.return_value = Mock(
            conversations=[mock_conv], next_page_token=None
        )
        mock_workspace_client.genie.list_conversation_messages.return_value = Mock(
            messages=[mock_message]
        )

        mock_cursor = Mock()
        mock_cursor.rowcount = 0  # Indicate duplicate (no rows inserted)

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
            workspace_client=mock_workspace_client,
        )
        service._pool = mock_pool
        service._setup_complete = True

        # Should complete without error even with duplicates
        result = service.from_space()
        assert result is service

    def test_from_space_filters_by_datetime_range(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
        """Test that from_space respects from_datetime and to_datetime filters."""
        from datetime import datetime, timedelta, timezone

        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        mock_conv = Mock()
        mock_conv.conversation_id = "conv-1"

        now = datetime.now(timezone.utc)
        old_message = Mock()
        old_message.content = "Old message"
        old_message.created_timestamp = int((now - timedelta(days=10)).timestamp() * 1000)
        old_message.attachments = None

        recent_message = Mock()
        recent_message.content = "Recent message"
        recent_message.created_timestamp = int((now - timedelta(days=1)).timestamp() * 1000)
        recent_message.attachments = None

        mock_workspace_client = Mock()
        mock_workspace_client.genie.list_conversations.return_value = Mock(
            conversations=[mock_conv], next_page_token=None
        )
        mock_workspace_client.genie.list_conversation_messages.return_value = Mock(
            messages=[old_message, recent_message]
        )

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
            workspace_client=mock_workspace_client,
        )
        service._pool = mock_pool
        service._setup_complete = True

        # Filter to only include messages from last 5 days
        service.from_space(from_datetime=now - timedelta(days=5))

        # Should only have one INSERT (recent message, old message filtered out)
        insert_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "INSERT INTO test_prompt_history" in str(call[0][0])
        ]
        assert len(insert_calls) == 1

    def test_from_space_limits_max_messages(
        self, cache_parameters: GenieSemanticCacheParametersModel
    ) -> None:
        """Test that from_space respects max_messages limit."""
        from datetime import datetime, timedelta, timezone

        mock_impl = Mock()
        mock_impl.space_id = "test-space"

        mock_conv = Mock()
        mock_conv.conversation_id = "conv-1"

        now = datetime.now(timezone.utc)
        messages = []
        for i in range(5):
            msg = Mock()
            msg.content = f"Message {i}"
            msg.created_timestamp = int((now - timedelta(hours=i)).timestamp() * 1000)
            msg.attachments = None
            messages.append(msg)

        mock_workspace_client = Mock()
        mock_workspace_client.genie.list_conversations.return_value = Mock(
            conversations=[mock_conv], next_page_token=None
        )
        mock_workspace_client.genie.list_conversation_messages.return_value = Mock(
            messages=messages
        )

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
            workspace_client=mock_workspace_client,
        )
        service._pool = mock_pool
        service._setup_complete = True

        # Limit to 2 messages
        service.from_space(max_messages=2)

        # Should only have 2 INSERTs (limited by max_messages)
        insert_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "INSERT INTO test_prompt_history" in str(call[0][0])
        ]
        assert len(insert_calls) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
