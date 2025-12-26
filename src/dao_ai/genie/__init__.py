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

from dao_ai.genie.cache import (
    CacheResult,
    GenieServiceBase,
    LRUCacheService,
    SemanticCacheService,
    SQLCacheEntry,
)
from dao_ai.genie.core import GenieService

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
