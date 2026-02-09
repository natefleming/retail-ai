"""
Diagnostic script to validate semantic search in PostgresContextAwareGenieService.

Tests directly against the retail-consumer-goods Lakebase instance to verify:
1. Embeddings are generated correctly
2. Entries can be stored and retrieved
3. Similar questions produce cache hits
4. The zero-context vs real-context scenario

Run with:
    python -m pytest tests/dao_ai/genie/test_semantic_search_diagnostic.py -v -s
"""

import os
from datetime import datetime
from unittest.mock import Mock

import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from dao_ai.config import (  # noqa: E402
    DatabaseModel,
    GenieContextAwareCacheParametersModel,
    WarehouseModel,
)
from dao_ai.genie.cache import PostgresContextAwareGenieService  # noqa: E402
from dao_ai.genie.core import GenieResponse  # noqa: E402

# Skip all tests if required env vars not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("DATABRICKS_WAREHOUSE_ID"),
        reason="DATABRICKS_WAREHOUSE_ID not set",
    ),
]


@pytest.fixture
def database_model() -> DatabaseModel:
    """Lakebase database model."""
    return DatabaseModel(instance_name="retail-consumer-goods")


@pytest.fixture
def warehouse_model() -> WarehouseModel:
    """Warehouse model from env."""
    wh_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    assert wh_id, "DATABRICKS_WAREHOUSE_ID must be set"
    return WarehouseModel(warehouse_id=wh_id)


@pytest.fixture
def cache_service(
    database_model: DatabaseModel, warehouse_model: WarehouseModel
) -> PostgresContextAwareGenieService:
    """Create a cache service with unique test tables."""
    test_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    params = GenieContextAwareCacheParametersModel(
        database=database_model,
        warehouse=warehouse_model,
        embedding_model="databricks-gte-large-en",
        table_name=f"test_semantic_diag_{test_id}",
        prompt_history_table=f"test_semantic_diag_prompts_{test_id}",
        time_to_live_seconds=3600,
        similarity_threshold=0.85,
        context_similarity_threshold=0.80,
        context_window_size=2,
    )

    mock_impl = Mock()
    mock_impl.space_id = "test-diagnostic-space"

    service = PostgresContextAwareGenieService(
        impl=mock_impl,
        parameters=params,
        workspace_client=None,
    )
    service.initialize()

    yield service

    # Cleanup
    try:
        service.drop_tables()
    except Exception as e:
        print(f"Cleanup warning: {e}")


class TestSemanticSearchDiagnostic:
    """Diagnostic tests for semantic search functionality."""

    def test_embeddings_generated(
        self, cache_service: PostgresContextAwareGenieService
    ) -> None:
        """Test that embeddings are generated and have correct dimensions."""
        question = "What are the ingredients in a vanilla latte?"
        q_emb, ctx_emb, ctx_str = cache_service._embed_question(question)

        print(f"\n  Embedding dims: {len(q_emb)}")
        print(f"  Context string: {ctx_str!r}")
        print(f"  Context embedding is zero: {all(x == 0.0 for x in ctx_emb)}")
        print(f"  Question embedding sample: {q_emb[:5]}")

        assert len(q_emb) > 0, "Question embedding should not be empty"
        assert len(ctx_emb) == len(q_emb), "Embedding dims should match"
        assert ctx_str == "", "No context expected for first question"
        assert all(x == 0.0 for x in ctx_emb), "Context should be zero vector"

    def test_store_and_find_exact_match(
        self, cache_service: PostgresContextAwareGenieService
    ) -> None:
        """Test storing an entry and finding it with the exact same question."""
        question = "What are the ingredients in a vanilla latte?"

        # Generate embeddings (no context)
        q_emb, ctx_emb, ctx_str = cache_service._embed_question(question)

        # Store an entry
        response = GenieResponse(
            result="test result",
            query="SELECT * FROM items WHERE name LIKE '%vanilla latte%'",
            description="Vanilla latte ingredients",
            conversation_id="test-conv-1",
        )
        cache_service._store_entry(
            question=question,
            conversation_context=ctx_str,
            question_embedding=q_emb,
            context_embedding=ctx_emb,
            response=response,
            message_id="test-msg-1",
        )

        # Search with the exact same question and embeddings
        result = cache_service._find_similar(
            question=question,
            conversation_context=ctx_str,
            question_embedding=q_emb,
            context_embedding=ctx_emb,
            conversation_id="test-conv-1",
        )

        print(f"\n  Find result: {result is not None}")
        if result is not None:
            entry, similarity = result
            print(f"  Combined similarity: {similarity:.4f}")
            print(f"  Cached question: {entry.query[:80]}")
        else:
            print("  ERROR: Exact same question not found!")

        assert result is not None, "Exact same question should be found"
        entry, similarity = result
        assert similarity > 0.95, (
            f"Exact match should have very high similarity, got {similarity:.4f}"
        )

    def test_find_similar_question(
        self, cache_service: PostgresContextAwareGenieService
    ) -> None:
        """Test finding a semantically similar (but not identical) question."""
        original = "What are the ingredients in a vanilla latte?"
        similar = "What ingredients does a vanilla latte have?"

        # Store with original question
        q_emb, ctx_emb, ctx_str = cache_service._embed_question(original)
        response = GenieResponse(
            result="test result",
            query="SELECT * FROM items WHERE name LIKE '%vanilla latte%'",
            description="Vanilla latte ingredients",
            conversation_id="test-conv-1",
        )
        cache_service._store_entry(
            question=original,
            conversation_context=ctx_str,
            question_embedding=q_emb,
            context_embedding=ctx_emb,
            response=response,
        )

        # Search with similar question
        q_emb2, ctx_emb2, ctx_str2 = cache_service._embed_question(similar)
        result = cache_service._find_similar(
            question=similar,
            conversation_context=ctx_str2,
            question_embedding=q_emb2,
            context_embedding=ctx_emb2,
        )

        print(f"\n  Original: {original}")
        print(f"  Similar:  {similar}")
        print(f"  Found: {result is not None}")
        if result is not None:
            entry, similarity = result
            print(f"  Combined similarity: {similarity:.4f}")

        assert result is not None, "Similar question should be found"

    def test_dissimilar_question_not_matched(
        self, cache_service: PostgresContextAwareGenieService
    ) -> None:
        """Test that a dissimilar question does NOT match."""
        original = "What are the ingredients in a vanilla latte?"
        dissimilar = "How many stores are in California?"

        # Store with original question
        q_emb, ctx_emb, ctx_str = cache_service._embed_question(original)
        response = GenieResponse(
            result="test result",
            query="SELECT * FROM items WHERE name LIKE '%vanilla latte%'",
            description="Vanilla latte ingredients",
            conversation_id="test-conv-1",
        )
        cache_service._store_entry(
            question=original,
            conversation_context=ctx_str,
            question_embedding=q_emb,
            context_embedding=ctx_emb,
            response=response,
        )

        # Search with dissimilar question
        q_emb2, ctx_emb2, ctx_str2 = cache_service._embed_question(dissimilar)
        result = cache_service._find_similar(
            question=dissimilar,
            conversation_context=ctx_str2,
            question_embedding=q_emb2,
            context_embedding=ctx_emb2,
        )

        print(f"\n  Original:   {original}")
        print(f"  Dissimilar: {dissimilar}")
        print(f"  Found: {result is not None}")
        if result is not None:
            _, similarity = result
            print(f"  Combined similarity: {similarity:.4f} (should be low)")

        assert result is None, "Dissimilar question should NOT match"

    def test_zero_context_vs_real_context_mismatch(
        self, cache_service: PostgresContextAwareGenieService
    ) -> None:
        """
        Reproduce the exact bug: entry stored with zero context vector,
        then looked up with real context vector → should this hit or miss?

        This is the scenario from the user's logs:
          question_sim=0.9729 | context_sim=0.0402 → MISS
        """
        question = "What are the ingredients in a vanilla latte?"

        # Step 1: Store with NO context (zero vector) - simulates first message
        q_emb, zero_ctx_emb, empty_ctx = cache_service._embed_question(question)
        assert empty_ctx == "", "Should have empty context"
        assert all(x == 0.0 for x in zero_ctx_emb), "Should be zero vector"

        response = GenieResponse(
            result="test result",
            query="SELECT * FROM items WHERE name LIKE '%vanilla latte%'",
            description="Vanilla latte ingredients",
            conversation_id="test-conv-1",
        )
        cache_service._store_entry(
            question=question,
            conversation_context=empty_ctx,
            question_embedding=q_emb,
            context_embedding=zero_ctx_emb,
            response=response,
        )

        # Step 2: Search with REAL context (simulates second message with prompt history)
        conversation_context = "Previous: What are the ingredients in a vanilla latte?"
        q_emb2, real_ctx_emb, ctx_str = cache_service._generate_dual_embeddings(
            question, conversation_context
        )

        print(f"\n  Question: {question}")
        print("  Stored context: (empty/zero vector)")
        print(f"  Query context: {conversation_context}")
        print(
            f"  Real context embedding is zero: {all(x == 0.0 for x in real_ctx_emb)}"
        )

        result = cache_service._find_similar(
            question=question,
            conversation_context=conversation_context,
            question_embedding=q_emb2,
            context_embedding=real_ctx_emb,
        )

        print(f"  Found: {result is not None}")
        if result is not None:
            entry, similarity = result
            print(f"  Combined similarity: {similarity:.4f}")
            print("  RESULT: Cache HIT (context mismatch did NOT block)")
        else:
            print(
                "  RESULT: Cache MISS (zero-context vs real-context mismatch blocks hit)"
            )
            print(
                "  THIS IS THE BUG - high question similarity blocked by low context similarity"
            )

        # With cosine similarity + zero-context skip, this should now be a HIT
        assert result is not None, (
            "Entry stored with no context should match when question similarity is high, "
            "even if query has real context (zero-context bypass)"
        )
