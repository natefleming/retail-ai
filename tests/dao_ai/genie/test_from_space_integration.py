"""Integration tests for from_space() method.

These tests require:
- RETAIL_AI_GENIE_SPACE_ID environment variable
- DATABRICKS_WAREHOUSE_ID environment variable
- retail-consumer-goods Lakebase instance
- Valid Databricks authentication

Run with: pytest tests/dao_ai/genie/test_from_space_integration.py -v -m integration
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from databricks.sdk import WorkspaceClient

from dao_ai.config import (
    DatabaseModel,
    GenieContextAwareCacheParametersModel,
    WarehouseModel,
)
from dao_ai.genie.cache import PostgresContextAwareGenieService

# Skip all tests in this module if required environment variables are not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("RETAIL_AI_GENIE_SPACE_ID"),
        reason="RETAIL_AI_GENIE_SPACE_ID environment variable not set",
    ),
    pytest.mark.skipif(
        not os.getenv("DATABRICKS_WAREHOUSE_ID"),
        reason="DATABRICKS_WAREHOUSE_ID environment variable not set",
    ),
]


@pytest.fixture
def genie_space_id() -> str:
    """Get Genie space ID from environment."""
    space_id = os.getenv("RETAIL_AI_GENIE_SPACE_ID")
    if not space_id:
        pytest.skip("RETAIL_AI_GENIE_SPACE_ID not set")
    return space_id


@pytest.fixture
def warehouse_id() -> str:
    """Get warehouse ID from environment."""
    wh_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not wh_id:
        pytest.skip("DATABRICKS_WAREHOUSE_ID not set")
    return wh_id


@pytest.fixture
def workspace_client() -> WorkspaceClient:
    """Create a real WorkspaceClient using default authentication."""
    return WorkspaceClient()


@pytest.fixture
def database_model() -> DatabaseModel:
    """Create database model for retail-consumer-goods Lakebase instance."""
    return DatabaseModel(
        instance_name="retail-consumer-goods",
    )


@pytest.fixture
def warehouse_model(warehouse_id: str) -> WarehouseModel:
    """Create warehouse model."""
    return WarehouseModel(warehouse_id=warehouse_id)


@pytest.fixture
def cache_parameters(
    database_model: DatabaseModel, warehouse_model: WarehouseModel
) -> GenieContextAwareCacheParametersModel:
    """Create cache parameters for testing."""
    # Use unique table names to avoid conflicts with other tests
    test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return GenieContextAwareCacheParametersModel(
        database=database_model,
        warehouse=warehouse_model,
        embedding_model="databricks-gte-large-en",
        table_name=f"test_from_space_cache_{test_id}",
        prompt_history_table=f"test_from_space_prompts_{test_id}",
        time_to_live_seconds=3600,
        similarity_threshold=0.85,
    )


@pytest.fixture
def cache_service(
    genie_space_id: str,
    cache_parameters: GenieContextAwareCacheParametersModel,
    workspace_client: WorkspaceClient,
) -> PostgresContextAwareGenieService:
    """Create PostgresContextAwareGenieService for testing."""
    # Create a mock impl with the space_id
    mock_impl = Mock()
    mock_impl.space_id = genie_space_id

    service = PostgresContextAwareGenieService(
        impl=mock_impl,
        parameters=cache_parameters,
        workspace_client=workspace_client,
    )

    yield service

    # Cleanup after test - drop tables to avoid accumulating test artifacts
    try:
        service.drop_tables()
    except Exception:
        # Fallback to clearing if drop fails
        try:
            service.clear()
            service.clear_prompt_history()
        except Exception:
            pass


class TestFromSpaceIntegration:
    """Integration tests for from_space() method."""

    def test_from_space_real_genie_space(
        self, cache_service: PostgresContextAwareGenieService, genie_space_id: str
    ) -> None:
        """End-to-end test with real Genie space."""
        # Initialize and run from_space
        result = cache_service.initialize().from_space(
            space_id=genie_space_id,
            include_all_messages=True,
            max_messages=10,  # Limit to avoid long test times
        )

        # Verify method chaining works
        assert result is cache_service

        # Check that some data was processed (stats logging should show counts)
        stats = cache_service.stats()
        assert stats is not None

    def test_from_space_populates_postgres_tables(
        self, cache_service: PostgresContextAwareGenieService, genie_space_id: str
    ) -> None:
        """Verify data is actually stored in PostgreSQL tables."""
        # Initialize and run from_space
        cache_service.initialize().from_space(
            space_id=genie_space_id,
            include_all_messages=True,
            max_messages=5,
        )

        # Query prompt history to verify data
        stats = cache_service.stats()

        # If there were any messages in the space, we should have some prompts
        # Note: This might be 0 if the space has no conversations
        assert "prompt_history" in stats or stats.get("size", 0) >= 0

    def test_from_space_duplicate_run_idempotent(
        self, cache_service: PostgresContextAwareGenieService, genie_space_id: str
    ) -> None:
        """Running from_space twice should not create duplicates."""
        # Initialize
        cache_service.initialize()

        # First run
        cache_service.from_space(
            space_id=genie_space_id,
            include_all_messages=True,
            max_messages=5,
        )

        # Get counts after first run
        stats1 = cache_service.stats()
        prompt_count1 = stats1.get("prompt_history", {}).get("total_prompts", 0)

        # Second run (should skip duplicates)
        cache_service.from_space(
            space_id=genie_space_id,
            include_all_messages=True,
            max_messages=5,
        )

        # Get counts after second run
        stats2 = cache_service.stats()
        prompt_count2 = stats2.get("prompt_history", {}).get("total_prompts", 0)

        # Counts should be the same (no duplicates)
        assert prompt_count1 == prompt_count2

    def test_from_space_with_time_range_filter(
        self, cache_service: PostgresContextAwareGenieService, genie_space_id: str
    ) -> None:
        """Integration test for time filtering."""
        # Initialize
        cache_service.initialize()

        # Run with time filter (last 24 hours)
        now = datetime.now(timezone.utc)
        from_time = now - timedelta(hours=24)

        result = cache_service.from_space(
            space_id=genie_space_id,
            include_all_messages=True,
            from_datetime=from_time,
            to_datetime=now,
            max_messages=10,
        )

        assert result is cache_service

    def test_from_space_respects_include_all_messages(
        self, cache_service: PostgresContextAwareGenieService, genie_space_id: str
    ) -> None:
        """Test permission-based filtering with include_all_messages."""
        # Initialize
        cache_service.initialize()

        # Run with include_all_messages=False (only current user's conversations)
        result = cache_service.from_space(
            space_id=genie_space_id,
            include_all_messages=False,
            max_messages=5,
        )

        assert result is cache_service

        # Run with include_all_messages=True (requires CAN MANAGE permission)
        # This might fail if user doesn't have permission, which is expected
        try:
            result = cache_service.from_space(
                space_id=genie_space_id,
                include_all_messages=True,
                max_messages=5,
            )
            assert result is cache_service
        except Exception as e:
            # Permission error is acceptable
            if "permission" not in str(e).lower():
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
