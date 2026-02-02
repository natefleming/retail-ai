"""
Context-aware cache implementations for Genie SQL queries.

This package provides context-aware caching layers that use semantic similarity
matching with dual embeddings (question + conversation context) for high-precision
cache lookups.

Available implementations:
- InMemoryContextAwareGenieService: In-memory cache with L2 distance matching
- PostgresContextAwareGenieService: PostgreSQL pg_vector-based persistent cache

Base classes:
- ContextAwareGenieService: Abstract base for all context-aware cache implementations
- PersistentContextAwareGenieCacheService: Abstract base for database-backed implementations
"""

from dao_ai.genie.cache.context_aware.base import ContextAwareGenieService
from dao_ai.genie.cache.context_aware.in_memory import InMemoryContextAwareGenieService
from dao_ai.genie.cache.context_aware.persistent import (
    PersistentContextAwareGenieCacheService,
)
from dao_ai.genie.cache.context_aware.postgres import PostgresContextAwareGenieService

__all__ = [
    # Base classes
    "ContextAwareGenieService",
    "PersistentContextAwareGenieCacheService",
    # Implementations
    "InMemoryContextAwareGenieService",
    "PostgresContextAwareGenieService",
]
