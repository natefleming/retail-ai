"""
Genie cache implementations.

This package provides caching layers for Genie SQL queries that can be
chained together using the decorator pattern.

Available cache implementations:
- LRUCacheService: In-memory LRU cache with O(1) exact match lookup
- SemanticCacheService: PostgreSQL pg_vector-based semantic similarity cache

Example usage:
    from dao_ai.genie.cache import LRUCacheService, SemanticCacheService

    # Chain caches: LRU (checked first) -> Semantic (checked second) -> Genie
    genie_service = SemanticCacheService(
        impl=GenieService(genie),
        parameters=semantic_params,
        genie_space_id=space_id,
    )
    genie_service = LRUCacheService(
        impl=genie_service,
        parameters=lru_params,
    )
"""

from dao_ai.genie.cache.base import (
    CacheResult,
    GenieServiceBase,
    SQLCacheEntry,
    execute_sql_via_warehouse,
)
from dao_ai.genie.cache.lru import LRUCacheService
from dao_ai.genie.cache.semantic import SemanticCacheService

__all__ = [
    # Base types
    "CacheResult",
    "GenieServiceBase",
    "SQLCacheEntry",
    "execute_sql_via_warehouse",
    # Cache implementations
    "LRUCacheService",
    "SemanticCacheService",
]
