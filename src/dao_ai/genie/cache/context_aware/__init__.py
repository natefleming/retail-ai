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

Optimization:
- optimize_context_aware_cache_thresholds: Tune cache thresholds using Bayesian optimization
- generate_eval_dataset_from_cache: Generate evaluation datasets from cache entries
"""

from dao_ai.genie.cache.context_aware.base import ContextAwareGenieService
from dao_ai.genie.cache.context_aware.in_memory import InMemoryContextAwareGenieService
from dao_ai.genie.cache.context_aware.optimization import (
    ContextAwareCacheEvalDataset,
    ContextAwareCacheEvalEntry,
    ThresholdOptimizationResult,
    clear_judge_cache,
    generate_eval_dataset_from_cache,
    optimize_context_aware_cache_thresholds,
    semantic_match_judge,
)
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
    # Optimization
    "ContextAwareCacheEvalDataset",
    "ContextAwareCacheEvalEntry",
    "ThresholdOptimizationResult",
    "clear_judge_cache",
    "generate_eval_dataset_from_cache",
    "optimize_context_aware_cache_thresholds",
    "semantic_match_judge",
]
