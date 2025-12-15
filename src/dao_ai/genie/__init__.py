"""
Genie service implementations and caching layers.

This package provides core Genie functionality that can be used across
different contexts (tools, direct integration, etc.).

Main exports:
- GenieService: Core service implementation wrapping Databricks Genie SDK
- GenieServiceBase: Abstract base class for service implementations

Cache implementations are available in the cache subpackage:
- dao_ai.genie.cache.lru: LRU (Least Recently Used) cache
- dao_ai.genie.cache.semantic: Semantic similarity cache using pg_vector

Example usage:
    from dao_ai.genie import GenieService
    from dao_ai.genie.cache import LRUCacheService, SemanticCacheService
"""

import mlflow
from databricks_ai_bridge.genie import Genie, GenieResponse
from mlflow.entities import SpanType

from dao_ai.genie.cache import (
    CacheResult,
    GenieServiceBase,
    LRUCacheService,
    SemanticCacheService,
    SQLCacheEntry,
)


class GenieService(GenieServiceBase):
    """Concrete implementation of GenieServiceBase using the Genie SDK."""

    genie: Genie

    def __init__(self, genie: Genie) -> None:
        self.genie = genie

    @mlflow.trace(name="genie_ask_question", span_type=SpanType.TOOL)
    def ask_question(
        self, question: str, conversation_id: str | None = None
    ) -> GenieResponse:
        response: GenieResponse = self.genie.ask_question(
            question, conversation_id=conversation_id
        )
        # Log response metadata to trace
        mlflow.update_current_trace(
            tags={
                "genie_space_id": self.genie.space_id,
                "conversation_id": response.conversation_id or "",
                "has_sql": "true" if response.query else "false",
            }
        )
        return response


__all__ = [
    # Service classes
    "GenieService",
    "GenieServiceBase",
    # Cache types (from cache subpackage)
    "CacheResult",
    "LRUCacheService",
    "SemanticCacheService",
    "SQLCacheEntry",
]
