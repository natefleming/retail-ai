"""
Abstract base class for context-aware Genie cache implementations.

This module provides the foundational abstract base class for all context-aware
cache implementations. It extracts common code for:
- Dual embedding generation (question + conversation context)
- Ask question flow with error handling and graceful fallback
- SQL execution with retry logic
- Common properties and initialization patterns

Subclasses must implement storage-specific methods:
- _find_similar(): Find semantically similar cached entry
- _store_entry(): Store new cache entry
- _setup(): Initialize resources (embeddings, storage)
- invalidate_expired(): Remove expired entries
- clear(): Clear all entries for space
- stats(): Return cache statistics
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import timedelta
from typing import Any, Self, TypeVar

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import (
    GenieFeedbackRating,
    GenieListConversationMessagesResponse,
    GenieMessage,
)
from databricks_ai_bridge.genie import GenieResponse
from loguru import logger

from dao_ai.config import LLMModel, WarehouseModel
from dao_ai.genie.cache.base import (
    CacheResult,
    GenieServiceBase,
    SQLCacheEntry,
)
from dao_ai.genie.cache.core import execute_sql_via_warehouse

# Type variable for subclass return types
T = TypeVar("T", bound="ContextAwareGenieService")


def get_conversation_history(
    workspace_client: WorkspaceClient,
    space_id: str,
    conversation_id: str,
    max_messages: int = 10,
) -> list[GenieMessage]:
    """
    Retrieve conversation history from Genie.

    Args:
        workspace_client: The Databricks workspace client
        space_id: The Genie space ID
        conversation_id: The conversation ID to retrieve
        max_messages: Maximum number of messages to retrieve

    Returns:
        List of GenieMessage objects representing the conversation history
    """
    try:
        # Use the Genie API to retrieve conversation messages
        response: GenieListConversationMessagesResponse = (
            workspace_client.genie.list_conversation_messages(
                space_id=space_id,
                conversation_id=conversation_id,
            )
        )

        # Return the most recent messages up to max_messages
        if response.messages is not None:
            all_messages: list[GenieMessage] = list(response.messages)
            return (
                all_messages[-max_messages:]
                if len(all_messages) > max_messages
                else all_messages
            )
        return []
    except Exception as e:
        logger.warning(
            "Failed to retrieve conversation history",
            conversation_id=conversation_id,
            error=str(e),
        )
        return []


def build_context_string(
    question: str,
    conversation_messages: list[GenieMessage],
    window_size: int,
    max_tokens: int = 2000,
) -> str:
    """
    Build a context-aware question string using rolling window.

    This function creates a concatenated string that includes recent conversation
    turns to provide context for semantic similarity matching.

    Args:
        question: The current question
        conversation_messages: List of previous conversation messages
        window_size: Number of previous turns to include
        max_tokens: Maximum estimated tokens (rough approximation: 4 chars = 1 token)

    Returns:
        Context-aware question string formatted for embedding
    """
    if window_size <= 0 or not conversation_messages:
        return question

    # Take the last window_size messages (most recent)
    recent_messages = (
        conversation_messages[-window_size:]
        if len(conversation_messages) > window_size
        else conversation_messages
    )

    # Build context parts
    context_parts: list[str] = []

    for msg in recent_messages:
        # Only include messages with content from the history
        if msg.content:
            # Limit message length to prevent token overflow
            content: str = msg.content
            if len(content) > 500:  # Truncate very long messages
                content = content[:500] + "..."
            context_parts.append(f"Previous: {content}")

    # Add current question
    context_parts.append(f"Current: {question}")

    # Join with newlines
    context_string = "\n".join(context_parts)

    # Rough token limit check (4 chars ≈ 1 token)
    estimated_tokens = len(context_string) / 4
    if estimated_tokens > max_tokens:
        # Truncate to fit max_tokens
        target_chars = max_tokens * 4
        original_length = len(context_string)
        context_string = context_string[:target_chars] + "..."
        logger.trace(
            "Truncated context string",
            original_chars=original_length,
            target_chars=target_chars,
            max_tokens=max_tokens,
        )

    return context_string


class ContextAwareGenieService(GenieServiceBase):
    """
    Abstract base class for context-aware Genie cache implementations.

    This class provides shared implementation for:
    - Dual embedding generation (question + conversation context)
    - Main ask_question flow with error handling
    - SQL execution with warehouse
    - Common properties (time_to_live, similarity_threshold, etc.)

    Subclasses must implement storage-specific methods for finding similar
    entries, storing new entries, and managing cache lifecycle.

    Error Handling:
        All cache operations are wrapped in try/except to ensure graceful
        degradation. If any cache operation fails, the request is delegated
        to the underlying service without caching.

    Thread Safety:
        Subclasses are responsible for thread safety of storage operations.
        This base class does not provide synchronization primitives.
    """

    # Common attributes - subclasses should define these
    impl: GenieServiceBase
    _workspace_client: WorkspaceClient | None
    name: str
    _embeddings: Any  # DatabricksEmbeddings
    _embedding_dims: int | None
    _setup_complete: bool

    # Abstract methods that subclasses must implement
    @abstractmethod
    def _setup(self) -> None:
        """
        Initialize resources required by the cache implementation.

        This method is called lazily before first use. Implementations should:
        - Initialize embedding model
        - Set up storage (database connection, in-memory structures, etc.)
        - Create necessary tables/indexes if applicable

        This method should be idempotent (safe to call multiple times).
        """
        pass

    @abstractmethod
    def _find_similar(
        self,
        question: str,
        conversation_context: str,
        question_embedding: list[float],
        context_embedding: list[float],
        conversation_id: str | None = None,
    ) -> tuple[SQLCacheEntry, float] | None:
        """
        Find a semantically similar cached entry using dual embedding matching.

        Args:
            question: The original question (for logging)
            conversation_context: The conversation context string
            question_embedding: The embedding vector of just the question
            context_embedding: The embedding vector of the conversation context
            conversation_id: Optional conversation ID (for logging)

        Returns:
            Tuple of (SQLCacheEntry, combined_similarity_score) if found, None otherwise
        """
        pass

    @abstractmethod
    def _store_entry(
        self,
        question: str,
        conversation_context: str,
        question_embedding: list[float],
        context_embedding: list[float],
        response: GenieResponse,
        message_id: str | None = None,
    ) -> None:
        """
        Store a new cache entry with dual embeddings.

        Args:
            question: The user's question
            conversation_context: Previous conversation context string
            question_embedding: Embedding of the question
            context_embedding: Embedding of the conversation context
            response: The GenieResponse containing query, description, etc.
            message_id: The Genie message ID from the original API response.
                Stored with the cache entry to enable feedback on cache hits.
        """
        pass

    def invalidate_expired(self) -> int | dict[str, int]:
        """
        Template method for removing expired entries from the cache.

        This method implements the TTL check and delegates to
        _delete_expired_entries() for the actual deletion.

        Returns:
            Number of entries deleted, or dict with counts by category
        """
        self._setup()
        ttl_seconds = self.time_to_live_seconds

        if ttl_seconds is None or ttl_seconds < 0:
            return self._get_empty_expiration_result()

        return self._delete_expired_entries(ttl_seconds)

    @abstractmethod
    def _delete_expired_entries(self, ttl_seconds: int) -> int | dict[str, int]:
        """
        Delete expired entries from storage.

        Args:
            ttl_seconds: TTL in seconds for determining expiration

        Returns:
            Number of entries deleted, or dict with counts by category
        """
        pass

    def _get_empty_expiration_result(self) -> int | dict[str, int]:
        """
        Return the empty result for invalidate_expired when TTL is disabled.

        Override this in subclasses that return dict to return appropriate empty dict.

        Returns:
            0 by default, or empty dict for subclasses that return dict
        """
        return 0

    def clear(self) -> int:
        """
        Template method for clearing all entries from the cache.

        This method calls _setup() and delegates to _delete_all_entries().

        Returns:
            Number of entries deleted
        """
        self._setup()
        return self._delete_all_entries()

    @abstractmethod
    def _delete_all_entries(self) -> int:
        """
        Delete all entries for this Genie space from storage.

        Returns:
            Number of entries deleted
        """
        pass

    def stats(self) -> dict[str, Any]:
        """
        Template method for returning cache statistics.

        This method uses the Template Method pattern to consolidate the common
        stats calculation algorithm. Subclasses provide counting implementations
        via abstract methods and can add additional stats via hook methods.

        Returns:
            Dict with cache statistics (size, ttl, thresholds, etc.)
        """
        self._setup()
        ttl_seconds = self.time_to_live_seconds
        ttl = self.time_to_live

        # Calculate base stats using abstract counting methods
        if ttl_seconds is None or ttl_seconds < 0:
            total = self._count_all_entries()
            base_stats: dict[str, Any] = {
                "size": total,
                "ttl_seconds": None,
                "similarity_threshold": self.similarity_threshold,
                "expired_entries": 0,
                "valid_entries": total,
            }
        else:
            total, expired = self._count_entries_with_ttl(ttl_seconds)
            base_stats = {
                "size": total,
                "ttl_seconds": ttl.total_seconds() if ttl else None,
                "similarity_threshold": self.similarity_threshold,
                "expired_entries": expired,
                "valid_entries": total - expired,
            }

        # Add any additional stats from subclasses
        additional_stats = self._get_additional_stats()
        base_stats.update(additional_stats)

        return base_stats

    @abstractmethod
    def _count_all_entries(self) -> int:
        """
        Count all cache entries for this Genie space.

        Returns:
            Total number of cache entries
        """
        pass

    @abstractmethod
    def _count_entries_with_ttl(self, ttl_seconds: int) -> tuple[int, int]:
        """
        Count total and expired entries for this Genie space.

        Args:
            ttl_seconds: TTL in seconds for determining expiration

        Returns:
            Tuple of (total_entries, expired_entries)
        """
        pass

    def _get_additional_stats(self) -> dict[str, Any]:
        """
        Hook method to add additional stats from subclasses.

        Override this method to add subclass-specific statistics like
        capacity (in-memory) or prompt history stats (postgres).

        Returns:
            Dict with additional stats to merge into base stats
        """
        return {}

    # Properties that subclasses should implement or inherit
    @property
    @abstractmethod
    def warehouse(self) -> WarehouseModel:
        """The warehouse used for executing cached SQL queries."""
        pass

    @property
    @abstractmethod
    def time_to_live(self) -> timedelta | None:
        """Time-to-live for cache entries. None means never expires."""
        pass

    @property
    @abstractmethod
    def similarity_threshold(self) -> float:
        """Minimum similarity for cache hit (using L2 distance converted to similarity)."""
        pass

    @property
    def embedding_dims(self) -> int:
        """Dimension size for embeddings (auto-detected if not configured)."""
        if self._embedding_dims is None:
            raise RuntimeError(
                "Embedding dimensions not yet initialized. Call _setup() first."
            )
        return self._embedding_dims

    @property
    def space_id(self) -> str:
        """The Genie space ID from the underlying service."""
        return self.impl.space_id

    @property
    def workspace_client(self) -> WorkspaceClient | None:
        """Get workspace client, delegating to impl if not set."""
        if self._workspace_client is not None:
            return self._workspace_client
        return self.impl.workspace_client

    @property
    def time_to_live_seconds(self) -> int | None:
        """TTL in seconds (None or negative = never expires)."""
        ttl = self.time_to_live
        if ttl is None:
            return None
        return int(ttl.total_seconds())

    # Abstract method for embedding - subclasses must implement
    @abstractmethod
    def _embed_question(
        self, question: str, conversation_id: str | None = None
    ) -> tuple[list[float], list[float], str]:
        """
        Generate dual embeddings for a question with conversation context.

        Args:
            question: The question to embed
            conversation_id: Optional conversation ID for retrieving context

        Returns:
            Tuple of (question_embedding, context_embedding, conversation_context_string)
        """
        pass

    # Shared implementation methods
    def initialize(self) -> Self:
        """
        Eagerly initialize the cache service.

        Call this during tool creation to:
        - Validate configuration early (fail fast)
        - Initialize resources before any requests
        - Avoid first-request latency from lazy initialization

        Returns:
            self for method chaining
        """
        self._setup()
        return self

    def _initialize_embeddings(
        self,
        embedding_model: str | LLMModel,
        embedding_dims: int | None = None,
    ) -> None:
        """
        Initialize the embeddings model and detect dimensions.

        This helper method handles embedding model initialization for subclasses.

        Args:
            embedding_model: The embedding model name or LLMModel instance
            embedding_dims: Optional pre-configured embedding dimensions
        """
        # Convert embedding_model to LLMModel if it's a string
        model: LLMModel = (
            LLMModel(name=embedding_model)
            if isinstance(embedding_model, str)
            else embedding_model
        )
        self._embeddings = model.as_embeddings_model()

        # Auto-detect embedding dimensions if not provided
        if embedding_dims is None:
            sample_embedding: list[float] = self._embeddings.embed_query("test")
            self._embedding_dims = len(sample_embedding)
            logger.debug(
                "Auto-detected embedding dimensions",
                layer=self.name,
                dims=self._embedding_dims,
            )
        else:
            self._embedding_dims = embedding_dims

    def _embed_question_with_genie_history(
        self,
        question: str,
        conversation_id: str | None,
        context_window_size: int,
        max_context_tokens: int,
    ) -> tuple[list[float], list[float], str]:
        """
        Generate dual embeddings using Genie API for conversation history.

        This method retrieves conversation history from the Genie API and
        generates dual embeddings for semantic matching.

        Args:
            question: The question to embed
            conversation_id: Optional conversation ID for retrieving context
            context_window_size: Number of previous messages to include
            max_context_tokens: Maximum tokens for context string

        Returns:
            Tuple of (question_embedding, context_embedding, conversation_context_string)
        """
        conversation_context = ""

        # If conversation context is enabled and available
        if (
            self.workspace_client is not None
            and conversation_id is not None
            and context_window_size > 0
        ):
            try:
                # Retrieve conversation history from Genie API
                conversation_messages = get_conversation_history(
                    workspace_client=self.workspace_client,
                    space_id=self.space_id,
                    conversation_id=conversation_id,
                    max_messages=context_window_size * 2,  # Get extra for safety
                )

                # Build context string
                if conversation_messages:
                    recent_messages = (
                        conversation_messages[-context_window_size:]
                        if len(conversation_messages) > context_window_size
                        else conversation_messages
                    )

                    context_parts: list[str] = []
                    for msg in recent_messages:
                        if msg.content:
                            content: str = msg.content
                            if len(content) > 500:
                                content = content[:500] + "..."
                            context_parts.append(f"Previous: {content}")

                    conversation_context = "\n".join(context_parts)

                    # Truncate if too long
                    estimated_tokens = len(conversation_context) / 4
                    if estimated_tokens > max_context_tokens:
                        target_chars = max_context_tokens * 4
                        conversation_context = (
                            conversation_context[:target_chars] + "..."
                        )

                logger.trace(
                    "Using conversation context from Genie API",
                    layer=self.name,
                    messages_count=len(conversation_messages),
                    window_size=context_window_size,
                )
            except Exception as e:
                logger.warning(
                    "Failed to build conversation context, using question only",
                    layer=self.name,
                    error=str(e),
                )
                conversation_context = ""

        return self._generate_dual_embeddings(question, conversation_context)

    def _generate_dual_embeddings(
        self, question: str, conversation_context: str
    ) -> tuple[list[float], list[float], str]:
        """
        Generate dual embeddings for question and conversation context.

        Args:
            question: The question to embed
            conversation_context: The conversation context string

        Returns:
            Tuple of (question_embedding, context_embedding, conversation_context)
        """
        if conversation_context:
            # Embed both question and context
            embeddings: list[list[float]] = self._embeddings.embed_documents(
                [question, conversation_context]
            )
            question_embedding = embeddings[0]
            context_embedding = embeddings[1]
        else:
            # Only embed question, use zero vector for context
            embeddings = self._embeddings.embed_documents([question])
            question_embedding = embeddings[0]
            context_embedding = [0.0] * len(question_embedding)  # Zero vector

        return question_embedding, context_embedding, conversation_context

    @mlflow.trace(name="execute_cached_sql")
    def _execute_sql(self, sql: str) -> pd.DataFrame | str:
        """
        Execute SQL using the warehouse and return results.

        Args:
            sql: The SQL query to execute

        Returns:
            DataFrame with results, or error message string if execution failed
        """
        return execute_sql_via_warehouse(
            warehouse=self.warehouse,
            sql=sql,
            layer_name=self.name,
        )

    def _build_cache_hit_response(
        self,
        cached: SQLCacheEntry,
        result: pd.DataFrame,
        conversation_id: str | None,
    ) -> CacheResult:
        """
        Build a CacheResult for a cache hit.

        Args:
            cached: The cached SQL entry
            result: The fresh DataFrame from SQL execution
            conversation_id: The current conversation ID

        Returns:
            CacheResult with cache_hit=True, including message_id and cache_entry_id
            from the original cached entry for traceability and feedback support.
        """
        # IMPORTANT: Use the current conversation_id (from the request), not the cached one
        # This ensures the conversation continues properly
        response = GenieResponse(
            result=result,
            query=cached.query,
            description=cached.description,
            conversation_id=conversation_id
            if conversation_id
            else cached.conversation_id,
        )
        # Cache hit - include message_id from original response for feedback support
        # and cache_entry_id for traceability to genie_prompt_history
        return CacheResult(
            response=response,
            cache_hit=True,
            served_by=self.name,
            message_id=cached.message_id,
            cache_entry_id=cached.cache_entry_id,
        )

    def ask_question(
        self, question: str, conversation_id: str | None = None
    ) -> CacheResult:
        """
        Ask a question, using semantic cache if a similar query exists.

        On cache hit, re-executes the cached SQL to get fresh data.
        Returns CacheResult with cache metadata.

        This method wraps ask_question_with_cache_info with error handling
        to ensure graceful degradation on cache failures.

        Args:
            question: The question to ask
            conversation_id: Optional conversation ID for context

        Returns:
            CacheResult with fresh response and cache metadata
        """
        try:
            return self.ask_question_with_cache_info(question, conversation_id)
        except Exception as e:
            logger.warning(
                "Context-aware cache operation failed, delegating to underlying service",
                layer=self.name,
                error=str(e),
                exc_info=True,
            )
            # Graceful degradation: fall back to underlying service
            return self.impl.ask_question(question, conversation_id)

    def ask_question_with_cache_info(
        self,
        question: str,
        conversation_id: str | None = None,
    ) -> CacheResult:
        """
        Template method for asking a question with cache lookup.

        This method implements the cache lookup algorithm using the Template Method
        pattern. Subclasses can customize behavior by overriding hook methods:
        - _before_cache_lookup(): Called before cache search (e.g., store prompt)
        - _after_cache_hit(): Called after a cache hit (e.g., update prompt flags)
        - _after_cache_miss(): Called after a cache miss (e.g., store prompt)

        Args:
            question: The question to ask
            conversation_id: Optional conversation ID for context and continuation

        Returns:
            CacheResult with fresh response and cache metadata
        """
        self._setup()

        # Step 1: Generate dual embeddings
        question_embedding, context_embedding, conversation_context = (
            self._embed_question(question, conversation_id)
        )

        # Step 2: Hook for pre-lookup actions (e.g., store prompt in history)
        self._before_cache_lookup(question, conversation_id)

        # Step 3: Search for similar cached entry
        cache_result = self._find_similar(
            question,
            conversation_context,
            question_embedding,
            context_embedding,
            conversation_id,
        )

        # Step 4: Handle cache hit or miss
        if cache_result is not None:
            cached, combined_similarity = cache_result

            result = self._handle_cache_hit(
                question,
                conversation_id,
                cached,
                combined_similarity,
                conversation_context,
                question_embedding,
                context_embedding,
            )

            # Hook for post-cache-hit actions (e.g., update prompt cache_hit flag)
            self._after_cache_hit(question, conversation_id, result)

            return result

        # Handle cache miss
        result = self._handle_cache_miss(
            question,
            conversation_id,
            conversation_context,
            question_embedding,
            context_embedding,
        )

        # Hook for post-cache-miss actions (e.g., store prompt if not done earlier)
        self._after_cache_miss(question, conversation_id, result)

        return result

    def _before_cache_lookup(self, question: str, conversation_id: str | None) -> None:
        """
        Hook method called before cache lookup.

        Override this method to perform actions before searching the cache,
        such as storing the prompt in history.

        Args:
            question: The question being asked
            conversation_id: Optional conversation ID
        """
        pass

    def _after_cache_hit(
        self,
        question: str,
        conversation_id: str | None,
        result: CacheResult,
    ) -> None:
        """
        Hook method called after a cache hit.

        Override this method to perform actions after a successful cache hit,
        such as updating prompt history flags.

        Args:
            question: The question that was asked
            conversation_id: Optional conversation ID
            result: The cache result
        """
        pass

    def _after_cache_miss(
        self,
        question: str,
        conversation_id: str | None,
        result: CacheResult,
    ) -> None:
        """
        Hook method called after a cache miss.

        Override this method to perform actions after a cache miss,
        such as storing prompt history if not done earlier.

        Args:
            question: The question that was asked
            conversation_id: Optional conversation ID
            result: The cache result
        """
        pass

    def _handle_cache_hit(
        self,
        question: str,
        conversation_id: str | None,
        cached: SQLCacheEntry,
        combined_similarity: float,
        conversation_context: str,
        question_embedding: list[float],
        context_embedding: list[float],
    ) -> CacheResult:
        """
        Handle a cache hit - execute cached SQL and return response.

        This method handles the common cache hit logic including SQL execution,
        stale cache fallback, and response building.

        Args:
            question: The original question
            conversation_id: The conversation ID
            cached: The cached SQL entry
            combined_similarity: The similarity score
            conversation_context: The conversation context string
            question_embedding: The question embedding
            context_embedding: The context embedding

        Returns:
            CacheResult with the response
        """
        logger.debug(
            "Cache hit",
            layer=self.name,
            combined_similarity=f"{combined_similarity:.3f}",
            question=question[:50],
            conversation_id=conversation_id,
        )

        # Re-execute the cached SQL to get fresh data
        result: pd.DataFrame | str = self._execute_sql(cached.query)

        # Check if SQL execution failed (returns error string instead of DataFrame)
        if isinstance(result, str):
            logger.warning(
                "Cached SQL execution failed, falling back to Genie",
                layer=self.name,
                question=question[:80],
                conversation_id=conversation_id,
                cached_sql=cached.query[:80],
                error=result[:200],
                space_id=self.space_id,
            )

            # Subclass should handle stale entry cleanup
            self._on_stale_cache_entry(question)

            # Fall back to Genie to get fresh SQL
            logger.info(
                "Delegating to Genie for fresh SQL",
                layer=self.name,
                question=question[:80],
                conversation_id=conversation_id,
                space_id=self.space_id,
                delegating_to=type(self.impl).__name__,
            )
            fallback_result: CacheResult = self.impl.ask_question(
                question, conversation_id
            )

            # Store the fresh SQL in cache
            if fallback_result.response.query:
                self._store_entry(
                    question,
                    conversation_context,
                    question_embedding,
                    context_embedding,
                    fallback_result.response,
                    message_id=fallback_result.message_id,
                )
                logger.info(
                    "Stored fresh SQL from fallback",
                    layer=self.name,
                    fresh_sql=fallback_result.response.query[:80],
                    space_id=self.space_id,
                    message_id=fallback_result.message_id,
                )
            else:
                logger.warning(
                    "Fallback response has no SQL query to cache",
                    layer=self.name,
                    question=question[:80],
                    space_id=self.space_id,
                )

            # Return as cache miss (fallback scenario)
            # Propagate message_id from fallback result
            return CacheResult(
                response=fallback_result.response,
                cache_hit=False,
                served_by=None,
                message_id=fallback_result.message_id,
            )

        # Build and return cache hit response
        return self._build_cache_hit_response(cached, result, conversation_id)

    def _on_stale_cache_entry(self, question: str) -> None:
        """
        Called when a stale cache entry is detected (SQL execution failed).

        Subclasses can override this to clean up the stale entry from storage.

        Args:
            question: The question that had a stale cache entry
        """
        # Default implementation does nothing - subclasses should override
        pass

    def _handle_cache_miss(
        self,
        question: str,
        conversation_id: str | None,
        conversation_context: str,
        question_embedding: list[float],
        context_embedding: list[float],
    ) -> CacheResult:
        """
        Handle a cache miss - delegate to underlying service and store result.

        Args:
            question: The original question
            conversation_id: The conversation ID
            conversation_context: The conversation context string
            question_embedding: The question embedding
            context_embedding: The context embedding

        Returns:
            CacheResult from the underlying service
        """
        logger.info(
            "Cache MISS",
            layer=self.name,
            question=question[:80],
            conversation_id=conversation_id,
            space_id=self.space_id,
            similarity_threshold=self.similarity_threshold,
            delegating_to=type(self.impl).__name__,
        )

        result: CacheResult = self.impl.ask_question(question, conversation_id)

        # Store in cache if we got a SQL query
        if result.response.query:
            logger.debug(
                "Storing new cache entry",
                layer=self.name,
                question=question[:50],
                conversation_id=conversation_id,
                space=self.space_id,
                message_id=result.message_id,
            )
            self._store_entry(
                question,
                conversation_context,
                question_embedding,
                context_embedding,
                result.response,
                message_id=result.message_id,
            )
        else:
            logger.warning(
                "Not caching: response has no SQL query",
                layer=self.name,
                question=question[:50],
            )

        # Propagate message_id from underlying service result
        return CacheResult(
            response=result.response,
            cache_hit=False,
            served_by=None,
            message_id=result.message_id,
        )

    @abstractmethod
    def _invalidate_by_question(self, question: str) -> bool:
        """
        Invalidate cache entries matching a specific question.

        This method is called when negative feedback is received to remove
        the corresponding cache entry.

        Args:
            question: The question text to match and invalidate

        Returns:
            True if an entry was found and invalidated, False otherwise
        """
        pass

    @mlflow.trace(name="genie_context_aware_send_feedback")
    def send_feedback(
        self,
        conversation_id: str,
        rating: GenieFeedbackRating,
        message_id: str | None = None,
        was_cache_hit: bool = False,
    ) -> None:
        """
        Send feedback for a Genie message with cache invalidation.

        For context-aware caches, this method:
        1. If was_cache_hit is False: forwards feedback to the underlying service
        2. If rating is NEGATIVE: invalidates any matching cache entries

        Args:
            conversation_id: The conversation containing the message
            rating: The feedback rating (POSITIVE, NEGATIVE, or NONE)
            message_id: Optional message ID. If None, looks up the most recent message.
            was_cache_hit: Whether the response being rated was served from cache.

        Note:
            For cached responses (was_cache_hit=True), only cache invalidation is
            performed. No feedback is sent to the Genie API because cached responses
            don't have a corresponding Genie message.

            Future Enhancement: To enable full Genie feedback for cached responses,
            the cache would need to store the original message_id. This would require:
            1. Adding message_id column to cache tables
            2. Adding message_id field to SQLCacheEntry dataclass
            3. Capturing message_id from the original Genie API response
               (databricks_ai_bridge.genie.GenieResponse doesn't expose this)
            4. Using WorkspaceClient directly instead of databricks_ai_bridge
        """
        invalidated = False

        # Handle cache invalidation on negative feedback
        if rating == GenieFeedbackRating.NEGATIVE:
            # Need to look up the message content to find matching cache entries
            if self.workspace_client is not None:
                from dao_ai.genie.cache.base import (
                    get_latest_message_id,
                    get_message_content,
                )

                # Get message_id if not provided
                target_message_id = message_id
                if target_message_id is None:
                    target_message_id = get_latest_message_id(
                        workspace_client=self.workspace_client,
                        space_id=self.space_id,
                        conversation_id=conversation_id,
                    )

                # Get the message content (question) to find matching cache entries
                if target_message_id:
                    question = get_message_content(
                        workspace_client=self.workspace_client,
                        space_id=self.space_id,
                        conversation_id=conversation_id,
                        message_id=target_message_id,
                    )

                    if question:
                        invalidated = self._invalidate_by_question(question)
                        if invalidated:
                            logger.info(
                                "Invalidated cache entry due to negative feedback",
                                layer=self.name,
                                question=question[:80],
                                conversation_id=conversation_id,
                                message_id=target_message_id,
                            )
                        else:
                            logger.debug(
                                "No cache entry found to invalidate for negative feedback",
                                layer=self.name,
                                question=question[:80],
                                conversation_id=conversation_id,
                            )
                    else:
                        logger.warning(
                            "Could not retrieve message content for cache invalidation",
                            layer=self.name,
                            conversation_id=conversation_id,
                            message_id=target_message_id,
                        )
                else:
                    logger.warning(
                        "Could not find message_id for cache invalidation",
                        layer=self.name,
                        conversation_id=conversation_id,
                    )
            else:
                logger.warning(
                    "No workspace_client available for cache invalidation",
                    layer=self.name,
                    conversation_id=conversation_id,
                )

        # Forward feedback to underlying service if not a cache hit
        # For cache hits, there's no Genie message to provide feedback on
        if was_cache_hit:
            logger.info(
                "Skipping Genie API feedback - response was served from cache",
                layer=self.name,
                conversation_id=conversation_id,
                rating=rating.value if rating else None,
                cache_invalidated=invalidated,
            )
            return

        # Forward to underlying service
        logger.debug(
            "Forwarding feedback to underlying service",
            layer=self.name,
            conversation_id=conversation_id,
            rating=rating.value if rating else None,
            delegating_to=type(self.impl).__name__,
        )
        self.impl.send_feedback(
            conversation_id=conversation_id,
            rating=rating,
            message_id=message_id,
            was_cache_hit=False,  # Already handled, so pass False
        )
