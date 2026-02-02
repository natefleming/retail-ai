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

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieFeedbackRating
from databricks_ai_bridge.genie import GenieResponse
from loguru import logger

if TYPE_CHECKING:
    from dao_ai.genie.cache.base import CacheResult


def get_latest_message_id(
    workspace_client: WorkspaceClient,
    space_id: str,
    conversation_id: str,
) -> str | None:
    """
    Look up the most recent message ID for a conversation.

    This is used when sending feedback without a specific message_id.

    Args:
        workspace_client: The Databricks workspace client
        space_id: The Genie space ID
        conversation_id: The conversation ID to look up

    Returns:
        The message_id of the most recent message, or None if not found
    """
    try:
        response = workspace_client.genie.list_conversation_messages(
            space_id=space_id,
            conversation_id=conversation_id,
        )
        if response.messages:
            # Messages are returned in order; get the last one (most recent)
            messages = list(response.messages)
            if messages:
                # Use message_id if available, fall back to id (legacy)
                latest = messages[-1]
                return latest.message_id if latest.message_id else latest.id
        return None
    except Exception as e:
        logger.warning(
            "Failed to look up latest message_id",
            space_id=space_id,
            conversation_id=conversation_id,
            error=str(e),
        )
        return None


def get_message_content(
    workspace_client: WorkspaceClient,
    space_id: str,
    conversation_id: str,
    message_id: str,
) -> str | None:
    """
    Get the content (question text) of a specific message.

    This is used to find matching cache entries when invalidating on negative feedback.

    Args:
        workspace_client: The Databricks workspace client
        space_id: The Genie space ID
        conversation_id: The conversation ID
        message_id: The message ID to look up

    Returns:
        The message content (question text), or None if not found
    """
    try:
        message = workspace_client.genie.get_message(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
        )
        return message.content if message else None
    except Exception as e:
        logger.warning(
            "Failed to get message content",
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
            error=str(e),
        )
        return None


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

    @property
    @abstractmethod
    def workspace_client(self) -> WorkspaceClient | None:
        """Optional WorkspaceClient for API operations like from_space() and feedback."""
        pass

    @abstractmethod
    def send_feedback(
        self,
        conversation_id: str,
        rating: GenieFeedbackRating,
        message_id: str | None = None,
        was_cache_hit: bool = False,
    ) -> None:
        """
        Send feedback for a Genie message.

        This method sends user feedback (positive/negative/none) to the Genie service
        and handles cache invalidation for negative feedback.

        Args:
            conversation_id: The conversation containing the message
            rating: The feedback rating (POSITIVE, NEGATIVE, or NONE)
            message_id: Optional message ID. If None, looks up the most recent message
                in the conversation.
            was_cache_hit: Whether the response being rated was served from cache.
                If True and rating is NEGATIVE, the cache entry is invalidated but
                no feedback is sent to the Genie API (since no Genie message exists
                for cached responses).

        Note:
            For cached responses (was_cache_hit=True), the message_id from the
            original Genie response is stored in the cache entry. This enables
            sending feedback to Genie even for cache hits.

            The cache_entry_id in CacheResult can be used to trace which cache
            entry was used for a particular prompt in genie_prompt_history.
        """
        pass


@dataclass
class SQLCacheEntry:
    """
    A cache entry storing the SQL query metadata for re-execution.

    Instead of caching the full result, we cache the SQL query so that
    on cache hit we can re-execute it to get fresh data.

    Attributes:
        query: The SQL query to execute
        description: Description of the query
        conversation_id: The conversation ID where this query originated
        created_at: When the entry was created
        message_id: The original Genie message ID (for feedback on cache hits)
        cache_entry_id: The database row ID (for persistent caches)
    """

    query: str
    description: str
    conversation_id: str
    created_at: datetime
    message_id: str | None = None
    cache_entry_id: int | None = None


@dataclass
class CacheResult:
    """
    Result of a cache-aware query with metadata about cache behavior.

    Attributes:
        response: The GenieResponse (fresh data, possibly from cached SQL)
        cache_hit: Whether the SQL query came from cache
        served_by: Name of the layer that served the cached SQL (None if from origin)
        message_id: The Genie message ID (for sending feedback). Available from
            the extended Genie class on cache miss, or from stored cache entry on hit.
        cache_entry_id: The database row ID of the cache entry that served this hit.
            Only populated on cache hits from persistent caches (PostgreSQL).
            Can be used to trace back to genie_prompt_history entries.
    """

    response: GenieResponse
    cache_hit: bool
    served_by: str | None = None
    message_id: str | None = None
    cache_entry_id: int | None = None
