"""
Genie cache implementations.

This package provides caching layers for Genie SQL queries that can be
chained together using the decorator pattern.

Available cache implementations:
- LRUCacheService: In-memory LRU cache with O(1) exact match lookup
- PostgresContextAwareGenieService: PostgreSQL pg_vector-based context-aware cache
- InMemoryContextAwareGenieService: In-memory context-aware cache

Example usage:
    from dao_ai.genie.cache import LRUCacheService, PostgresContextAwareGenieService

    # Chain caches: LRU (checked first) -> Context-aware (checked second) -> Genie
    genie_service = PostgresContextAwareGenieService(
        impl=GenieService(genie),
        parameters=context_aware_params,
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
)
from dao_ai.genie.cache.context_aware import (
    ContextAwareGenieService,
    InMemoryContextAwareGenieService,
    PersistentContextAwareGenieCacheService,
    PostgresContextAwareGenieService,
)
from dao_ai.genie.cache.core import execute_sql_via_warehouse
from dao_ai.genie.cache.lru import LRUCacheService
from dao_ai.genie.cache.optimization import (
    ContextAwareCacheEvalDataset,
    ContextAwareCacheEvalEntry,
    ThresholdOptimizationResult,
    clear_judge_cache,
    generate_eval_dataset_from_cache,
    optimize_context_aware_cache_thresholds,
    semantic_match_judge,
)

__all__ = [
    # Base types
    "CacheResult",
    "GenieServiceBase",
    "SQLCacheEntry",
    "execute_sql_via_warehouse",
    # Context-aware base classes
    "ContextAwareGenieService",
    "PersistentContextAwareGenieCacheService",
    # Cache implementations
    "InMemoryContextAwareGenieService",
    "LRUCacheService",
    "PostgresContextAwareGenieService",
    # Optimization
    "ContextAwareCacheEvalDataset",
    "ContextAwareCacheEvalEntry",
    "ThresholdOptimizationResult",
    "clear_judge_cache",
    "generate_eval_dataset_from_cache",
    "optimize_context_aware_cache_thresholds",
    "semantic_match_judge",
]
