"""
Base classes and types for Genie cache implementations.

This module provides the foundational types used across different cache
implementations (LRU, Semantic, etc.). It contains only abstract base classes
and data structures - no concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from databricks_ai_bridge.genie import GenieResponse

if TYPE_CHECKING:
    from dao_ai.genie.cache.base import CacheResult


class GenieServiceBase(ABC):
    """Abstract base class for Genie service implementations."""

    @abstractmethod
    def ask_question(
        self, question: str, conversation_id: str | None = None
    ) -> "CacheResult":
        """
        Ask a question to Genie and return the response with cache metadata.

        All implementations return CacheResult to provide consistent cache information,
        even when caching is disabled (cache_hit=False, served_by=None).
        """
        pass

    @property
    @abstractmethod
    def space_id(self) -> str:
        """The space ID for the Genie service."""
        pass


@dataclass
class SQLCacheEntry:
    """
    A cache entry storing the SQL query metadata for re-execution.

    Instead of caching the full result, we cache the SQL query so that
    on cache hit we can re-execute it to get fresh data.
    """

    query: str
    description: str
    conversation_id: str
    created_at: datetime


@dataclass
class CacheResult:
    """
    Result of a cache-aware query with metadata about cache behavior.

    Attributes:
        response: The GenieResponse (fresh data, possibly from cached SQL)
        cache_hit: Whether the SQL query came from cache
        served_by: Name of the layer that served the cached SQL (None if from origin)
    """

    response: GenieResponse
    cache_hit: bool
    served_by: str | None = None
