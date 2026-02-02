"""
Genie service implementations and caching layers.

This package provides core Genie functionality that can be used across
different contexts (tools, direct integration, etc.).

Main exports:
- Genie: Extended Genie class that captures message_id in responses
- GenieResponse: Extended response class with message_id field
- GenieService: Service implementation wrapping Genie
- GenieServiceBase: Abstract base class for service implementations
- GenieFeedbackRating: Enum for feedback ratings (POSITIVE, NEGATIVE, NONE)

Original databricks_ai_bridge classes (aliased):
- DatabricksGenie: Original Genie from databricks_ai_bridge
- DatabricksGenieResponse: Original GenieResponse from databricks_ai_bridge

Cache implementations are available in the cache subpackage:
- dao_ai.genie.cache.lru: LRU (Least Recently Used) cache
- dao_ai.genie.cache.context_aware.postgres: PostgreSQL context-aware cache
- dao_ai.genie.cache.context_aware.in_memory: In-memory context-aware cache

Example usage:
    from dao_ai.genie import Genie, GenieService, GenieFeedbackRating

    # Create Genie with message_id support
    genie = Genie(space_id="my-space")
    response = genie.ask_question("What are total sales?")
    print(response.message_id)  # Now available!

    # Use with GenieService
    service = GenieService(genie)
    result = service.ask_question("What are total sales?")

    # Send feedback using captured message_id
    service.send_feedback(
        conversation_id=result.response.conversation_id,
        rating=GenieFeedbackRating.POSITIVE,
        message_id=result.message_id,  # Available from CacheResult
        was_cache_hit=result.cache_hit,
    )
"""

from databricks.sdk.service.dashboards import GenieFeedbackRating

from dao_ai.genie.cache import (
    CacheResult,
    ContextAwareGenieService,
    GenieServiceBase,
    InMemoryContextAwareGenieService,
    LRUCacheService,
    PostgresContextAwareGenieService,
    SQLCacheEntry,
)
from dao_ai.genie.cache.base import get_latest_message_id, get_message_content
from dao_ai.genie.core import (
    DatabricksGenie,
    DatabricksGenieResponse,
    Genie,
    GenieResponse,
    GenieService,
)

__all__ = [
    # Extended Genie classes (primary - use these)
    "Genie",
    "GenieResponse",
    # Original databricks_ai_bridge classes (aliased)
    "DatabricksGenie",
    "DatabricksGenieResponse",
    # Service classes
    "GenieService",
    "GenieServiceBase",
    # Feedback
    "GenieFeedbackRating",
    # Helper functions
    "get_latest_message_id",
    "get_message_content",
    # Cache types (from cache subpackage)
    "CacheResult",
    "ContextAwareGenieService",
    "InMemoryContextAwareGenieService",
    "LRUCacheService",
    "PostgresContextAwareGenieService",
    "SQLCacheEntry",
]
