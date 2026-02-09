"""
Abstract base class for persistent (database-backed) context-aware Genie cache implementations.

This module provides the foundational abstract base class for database-backed
cache implementations. It adds:
- Connection pooling management
- Transaction handling with retry logic
- Prompt history storage and retrieval
- Database error handling with exponential backoff

Subclasses must implement database-specific methods:
- _create_table_if_not_exists(): Create database schema
- _get_pool(): Get database connection pool
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, TypeVar

from loguru import logger
from psycopg import sql

from dao_ai.config import DatabaseModel
from dao_ai.genie.cache.context_aware.base import (
    ContextAwareGenieService,
    get_conversation_history,
)

# Type variable for return types
T = TypeVar("T")

# Type alias for database row (dict due to row_factory=dict_row)
DbRow = dict[str, Any]


class PersistentContextAwareGenieCacheService(ContextAwareGenieService):
    """
    Abstract base class for database-backed context-aware Genie cache implementations.

    This class extends ContextAwareGenieService with database-specific functionality:
    - Connection pool management
    - Prompt history tracking for conversation context
    - Retry logic for transient database failures
    - Schema creation and management

    Subclasses must implement:
    - _get_pool(): Return the database connection pool
    - _create_table_if_not_exists(): Create required database tables
    - Database-specific _find_similar() and _store_entry() implementations

    Thread Safety:
        Uses connection pooling for thread-safe database access.
        All database operations use connection context managers.
    """

    # Additional attributes for persistent implementations
    _pool: Any  # ConnectionPool

    @property
    @abstractmethod
    def database(self) -> DatabaseModel:
        """The database used for storing cache entries."""
        pass

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Name of the cache table."""
        pass

    @property
    @abstractmethod
    def prompt_history_table(self) -> str:
        """Name of the prompt history table."""
        pass

    @property
    @abstractmethod
    def context_window_size(self) -> int:
        """Number of previous prompts to include in context."""
        pass

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Maximum tokens for context string."""
        pass

    @property
    @abstractmethod
    def context_similarity_threshold(self) -> float:
        """Minimum similarity for context matching."""
        pass

    @property
    @abstractmethod
    def question_weight(self) -> float:
        """Weight for question similarity in combined score."""
        pass

    @property
    @abstractmethod
    def context_weight(self) -> float:
        """Weight for context similarity in combined score."""
        pass

    @property
    @abstractmethod
    def max_prompt_history_length(self) -> int:
        """Maximum number of prompts to keep per conversation."""
        pass

    @property
    @abstractmethod
    def time_to_live_seconds(self) -> int | None:
        """TTL in seconds (None or negative = never expires)."""
        pass

    @abstractmethod
    def _create_table_if_not_exists(self) -> None:
        """
        Create the cache and prompt history tables if they don't exist.

        This method should handle:
        - Creating the cache table with vector columns
        - Creating indexes for efficient similarity search
        - Creating the prompt history table
        - Handling schema migrations if needed
        """
        pass

    def _execute_with_retry(
        self,
        operation: Callable[[], T],
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
    ) -> T:
        """
        Execute a database operation with exponential backoff retry.

        Args:
            operation: The database operation to execute
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)

        Returns:
            The result of the operation

        Raises:
            The last exception if all retries fail
        """
        import time

        last_exception: Exception | None = None
        delay = base_delay

        for attempt in range(max_attempts):
            try:
                return operation()
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Check if this is a retryable error
                retryable_errors = [
                    "connection",
                    "timeout",
                    "temporarily unavailable",
                    "too many connections",
                    "connection refused",
                    "operational error",
                ]
                is_retryable = any(err in error_str for err in retryable_errors)

                if not is_retryable or attempt == max_attempts - 1:
                    raise

                logger.warning(
                    f"Database operation failed (attempt {attempt + 1}/{max_attempts}), retrying",
                    layer=self.name,
                    error=str(e),
                    delay=delay,
                )

                time.sleep(delay)
                delay = min(delay * 2, max_delay)

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected state in retry logic")

    def _index_exists(self, cur: Any, index_name: str) -> bool:
        """
        Check if an index already exists in the database.

        Args:
            cur: Database cursor to execute SQL statements
            index_name: Name of the index to check

        Returns:
            True if the index exists, False otherwise
        """
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE indexname = %s",
            (index_name,),
        )
        return cur.fetchone() is not None

    # Counter for periodic prompt history cleanup (every N inserts)
    _prompt_insert_count: int = 0
    _CLEANUP_INTERVAL: int = 10

    def _store_user_prompt(
        self,
        prompt: str,
        conversation_id: str,
        cache_hit: bool = False,
    ) -> bool:
        """
        Store user prompt in local conversation history.

        This is called after embeddings are generated to ensure the current prompt
        is not included in its own context.

        Prompt history is non-critical; failures are logged but don't crash the request.

        Periodically enforces max_prompt_history_length (every N inserts) to avoid
        running a DELETE query on every single insert.

        Args:
            prompt: The user's question/prompt
            conversation_id: The conversation ID
            cache_hit: Whether this prompt resulted in a cache hit

        Returns:
            True if prompt was stored successfully, False otherwise
        """
        prompt_table_name = self.prompt_history_table
        insert_sql = sql.SQL("""
            INSERT INTO {} 
            (genie_space_id, conversation_id, prompt, cache_hit)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (genie_space_id, conversation_id, prompt)
            DO UPDATE SET cache_hit = EXCLUDED.cache_hit, created_at = CURRENT_TIMESTAMP
        """).format(sql.Identifier(prompt_table_name))

        logger.debug(
            "Inserting prompt into history",
            layer=self.name,
            table=prompt_table_name,
            space_id=self.space_id,
            conversation_id=conversation_id,
            prompt_preview=prompt[:80] if len(prompt) > 80 else prompt,
            prompt_length=len(prompt),
            cache_hit=cache_hit,
        )

        try:
            # Use a single connection for both insert and periodic cleanup
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_sql, (self.space_id, conversation_id, prompt, cache_hit)
                    )

                    # Periodic cleanup: enforce limit every N inserts
                    self._prompt_insert_count += 1
                    if self._prompt_insert_count % self._CLEANUP_INTERVAL == 0:
                        self._enforce_prompt_history_limit(conversation_id, cur=cur)

            logger.info(
                "Stored user prompt in history",
                layer=self.name,
                table=prompt_table_name,
                conversation_id=conversation_id,
                prompt_preview=prompt[:50],
                cache_hit=cache_hit,
            )

            return True
        except Exception as e:
            logger.warning(
                f"Failed to store prompt in history (non-critical): {e}",
                layer=self.name,
                table=prompt_table_name,
                conversation_id=conversation_id,
            )
            return False

    def _enforce_prompt_history_limit(
        self, conversation_id: str, cur: Any | None = None
    ) -> int:
        """
        Delete oldest prompts if conversation exceeds max_prompt_history_length.

        Uses a single DELETE with subquery for efficiency. Can reuse an existing
        cursor to avoid acquiring a separate database connection.

        Args:
            conversation_id: The conversation ID to enforce limit for
            cur: Optional existing cursor to reuse (avoids extra connection checkout)

        Returns:
            Number of prompts deleted (0 if within limit)
        """
        max_length = self.max_prompt_history_length
        prompt_table_name = self.prompt_history_table

        # Delete prompts beyond the limit, keeping the most recent ones
        prompt_table_id = sql.Identifier(prompt_table_name)
        delete_sql = sql.SQL("""
            DELETE FROM {}
            WHERE genie_space_id = %s 
              AND conversation_id = %s
              AND created_at < (
                  SELECT created_at FROM {}
                  WHERE genie_space_id = %s 
                    AND conversation_id = %s
                  ORDER BY created_at DESC
                  LIMIT 1 OFFSET %s
              )
        """).format(prompt_table_id, prompt_table_id)

        params = (
            self.space_id,
            conversation_id,
            self.space_id,
            conversation_id,
            max_length - 1,
        )

        try:
            if cur is not None:
                # Reuse existing cursor (same connection/transaction)
                cur.execute(delete_sql, params)
                deleted = cur.rowcount if isinstance(cur.rowcount, int) else 0
            else:
                # Standalone call - acquire own connection
                with self._pool.connection() as conn:
                    with conn.cursor() as new_cur:
                        new_cur.execute(delete_sql, params)
                        deleted = (
                            new_cur.rowcount if isinstance(new_cur.rowcount, int) else 0
                        )

            if deleted > 0:
                logger.debug(
                    "Enforced prompt history limit",
                    layer=self.name,
                    table=prompt_table_name,
                    conversation_id=conversation_id,
                    max_length=max_length,
                    deleted=deleted,
                )
            return deleted
        except Exception as e:
            logger.debug(
                f"Failed to enforce prompt history limit (non-critical): {e}",
                layer=self.name,
                conversation_id=conversation_id,
            )
            return 0

    def _get_local_prompt_history(
        self,
        conversation_id: str,
        max_prompts: int | None = None,
    ) -> list[str]:
        """
        Retrieve recent user prompts from local storage.

        Uses SQL LIMIT for efficiency - only retrieves exactly the number
        of prompts needed for the context window, not all prompts.

        Args:
            conversation_id: The conversation ID to retrieve prompts for
            max_prompts: Maximum number of prompts to retrieve

        Returns:
            List of prompt strings in chronological order (oldest to newest)
        """
        if max_prompts is None:
            max_prompts = self.context_window_size

        prompt_table_name = self.prompt_history_table
        query_sql = sql.SQL("""
            SELECT prompt 
            FROM {}
            WHERE genie_space_id = %s 
              AND conversation_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """).format(sql.Identifier(prompt_table_name))

        logger.debug(
            "Querying prompt history",
            layer=self.name,
            table=prompt_table_name,
            space_id=self.space_id,
            conversation_id=conversation_id,
            max_prompts=max_prompts,
        )

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # LIMIT ensures we only fetch exactly what's needed
                cur.execute(query_sql, (self.space_id, conversation_id, max_prompts))
                rows: list[DbRow] = cur.fetchall()
                # Reverse to get chronological order (oldest to newest)
                prompts = [row["prompt"] for row in reversed(rows)]

                logger.info(
                    "Retrieved prompt history from database",
                    layer=self.name,
                    table=prompt_table_name,
                    conversation_id=conversation_id,
                    requested=max_prompts,
                    returned=len(prompts),
                    prompts_preview=[
                        p[:40] + "..." if len(p) > 40 else p for p in prompts
                    ],
                )

                return prompts

    def _update_prompt_cache_hit(
        self,
        conversation_id: str,
        prompt: str,
        cache_hit: bool,
        cache_entry_id: int | None = None,
    ) -> bool:
        """
        Update the cache_hit flag and cache_entry_id for a previously stored prompt.

        This is called after determining whether the prompt resulted in a cache hit.
        Updates the most recent prompt matching the given text.

        Args:
            conversation_id: The conversation ID
            prompt: The prompt text to update
            cache_hit: The cache hit status to set
            cache_entry_id: The ID of the cache entry that served this hit (for traceability)

        Returns:
            True if update was successful, False otherwise
        """
        prompt_table_name = self.prompt_history_table
        prompt_table_id = sql.Identifier(prompt_table_name)
        update_sql = sql.SQL("""
            UPDATE {}
            SET cache_hit = %s, cache_entry_id = %s
            WHERE genie_space_id = %s 
              AND conversation_id = %s 
              AND prompt = %s 
              AND created_at = (
                  SELECT MAX(created_at) 
                  FROM {}
                  WHERE genie_space_id = %s 
                    AND conversation_id = %s 
                    AND prompt = %s
              )
        """).format(prompt_table_id, prompt_table_id)

        logger.debug(
            "Updating prompt cache_hit flag and cache_entry_id",
            layer=self.name,
            table=prompt_table_name,
            space_id=self.space_id,
            conversation_id=conversation_id,
            prompt_preview=prompt[:50],
            new_cache_hit=cache_hit,
            cache_entry_id=cache_entry_id,
        )

        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        update_sql,
                        (
                            cache_hit,
                            cache_entry_id,
                            self.space_id,
                            conversation_id,
                            prompt,
                            self.space_id,
                            conversation_id,
                            prompt,
                        ),
                    )
                    # Handle rowcount safely (may be Mock in tests or None)
                    updated_rows = getattr(cur, "rowcount", 0)
                    if not isinstance(updated_rows, int):
                        updated_rows = 0

                    if updated_rows > 0:
                        logger.info(
                            "Updated prompt cache_hit flag and cache_entry_id in history",
                            layer=self.name,
                            table=prompt_table_name,
                            conversation_id=conversation_id,
                            prompt_preview=prompt[:50],
                            cache_hit=cache_hit,
                            cache_entry_id=cache_entry_id,
                            rows_updated=updated_rows,
                        )
                        return True
                    else:
                        logger.debug(
                            "No prompt found to update cache_hit flag (may be expected)",
                            layer=self.name,
                            table=prompt_table_name,
                            conversation_id=conversation_id,
                            prompt_preview=prompt[:50],
                        )
                        return False
        except Exception as e:
            logger.warning(
                f"Failed to update prompt cache_hit flag and cache_entry_id (non-critical): {e}",
                layer=self.name,
                table=prompt_table_name,
                conversation_id=conversation_id,
                cache_entry_id=cache_entry_id,
            )
            return False

    def _embed_question(
        self, question: str, conversation_id: str | None = None
    ) -> tuple[list[float], list[float], str]:
        """
        Generate dual embeddings using local prompt history for context.

        This method retrieves conversation history from local storage first,
        falling back to Genie API if local history is empty.

        Args:
            question: The question to embed
            conversation_id: Optional conversation ID for retrieving context

        Returns:
            Tuple of (question_embedding, context_embedding, conversation_context_string)
        """
        conversation_context = ""

        # If conversation context is enabled and available
        if conversation_id is not None and self.context_window_size > 0:
            try:
                # Try local prompt history first (FASTER, includes cache hits)
                recent_prompts = self._get_local_prompt_history(
                    conversation_id=conversation_id,
                    max_prompts=self.context_window_size,
                )

                logger.trace(
                    "Retrieved local prompt history",
                    layer=self.name,
                    prompts_count=len(recent_prompts),
                    conversation_id=conversation_id,
                )

                # Fallback to Genie API if local history empty and API available
                if not recent_prompts and self.workspace_client is not None:
                    logger.debug(
                        "Local prompt history empty, falling back to Genie API",
                        layer=self.name,
                        conversation_id=conversation_id,
                    )

                    conversation_messages = get_conversation_history(
                        workspace_client=self.workspace_client,
                        space_id=self.space_id,
                        conversation_id=conversation_id,
                        max_messages=self.context_window_size * 2,
                    )

                    if conversation_messages:
                        recent_messages = (
                            conversation_messages[-self.context_window_size :]
                            if len(conversation_messages) > self.context_window_size
                            else conversation_messages
                        )
                        recent_prompts = [
                            msg.content for msg in recent_messages if msg.content
                        ]

                # Build context string from prompts
                if recent_prompts:
                    context_parts: list[str] = []
                    for prompt in recent_prompts:
                        content: str = prompt
                        if len(content) > 500:
                            content = content[:500] + "..."
                        context_parts.append(f"Previous: {content}")

                    conversation_context = "\n".join(context_parts)

                    # Truncate if too long
                    estimated_tokens = len(conversation_context) / 4
                    if estimated_tokens > self.max_context_tokens:
                        target_chars = self.max_context_tokens * 4
                        conversation_context = (
                            conversation_context[:target_chars] + "..."
                        )

                    logger.trace(
                        "Using conversation context",
                        layer=self.name,
                        prompts_count=len(recent_prompts),
                        window_size=self.context_window_size,
                        source="local_db",
                    )
            except Exception as e:
                logger.warning(
                    "Failed to build conversation context, using question only",
                    layer=self.name,
                    error=str(e),
                )
                conversation_context = ""

        return self._generate_dual_embeddings(question, conversation_context)

    def get_prompt_history(
        self,
        conversation_id: str,
        max_prompts: int | None = None,
        include_cache_hits: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieve prompt history for a conversation with metadata.

        Public utility method for inspecting conversation history.

        Args:
            conversation_id: The conversation ID to retrieve
            max_prompts: Maximum number of prompts (None = all prompts)
            include_cache_hits: Whether to include prompts that hit cache

        Returns:
            List of prompt records with metadata (prompt, cache_hit, created_at)
        """
        self._setup()

        prompt_table_name = self.prompt_history_table

        cache_filter = (
            sql.SQL("AND cache_hit = false") if not include_cache_hits else sql.SQL("")
        )
        limit_clause = (
            sql.SQL("LIMIT {}").format(sql.Literal(max_prompts))
            if max_prompts
            else sql.SQL("")
        )

        query_sql = sql.SQL("""
            SELECT prompt, cache_hit, created_at
            FROM {}
            WHERE genie_space_id = %s 
              AND conversation_id = %s
              {}
            ORDER BY created_at ASC
            {}
        """).format(
            sql.Identifier(prompt_table_name),
            cache_filter,
            limit_clause,
        )

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_sql, (self.space_id, conversation_id))
                rows: list[DbRow] = cur.fetchall()

                return [
                    {
                        "prompt": row["prompt"],
                        "cache_hit": row["cache_hit"],
                        "created_at": row["created_at"],
                    }
                    for row in rows
                ]

    def export_prompt_history(
        self,
        conversation_id: str,
        output_format: str = "text",
    ) -> str:
        """
        Export prompt history for a conversation in various formats.

        Args:
            conversation_id: The conversation ID to export
            output_format: Format for export ("text", "json", "markdown")

        Returns:
            Formatted prompt history string
        """
        self._setup()

        history = self.get_prompt_history(conversation_id)

        if not history:
            return "No prompt history found."

        if output_format == "json":
            import json

            return json.dumps(history, indent=2, default=str)

        elif output_format == "markdown":
            lines = ["# Conversation History", ""]
            for i, entry in enumerate(history, 1):
                cache_mark = "HIT" if entry["cache_hit"] else "MISS"
                lines.append(f"## Prompt {i} [{cache_mark}]")
                lines.append(f"**Prompt**: {entry['prompt']}")
                lines.append(f"**Cache Hit**: {entry['cache_hit']}")
                lines.append(f"**Timestamp**: {entry['created_at']}")
                lines.append("")
            return "\n".join(lines)

        else:  # text format
            lines = [f"Conversation: {conversation_id}", ""]
            for i, entry in enumerate(history, 1):
                cache_mark = "[CACHE HIT]" if entry["cache_hit"] else "[GENIE]"
                lines.append(f"{i}. {cache_mark} {entry['prompt']}")
                lines.append(f"   Timestamp: {entry['created_at']}")
            return "\n".join(lines)

    def clear_prompt_history(self, conversation_id: str | None = None) -> int:
        """
        Clear prompt history for a conversation or entire space.

        Args:
            conversation_id: Specific conversation to clear (None = clear all for space)

        Returns:
            Number of prompts deleted
        """
        self._setup()

        prompt_table_name = self.prompt_history_table
        prompt_table_id = sql.Identifier(prompt_table_name)

        if conversation_id:
            delete_sql = sql.SQL("""
                DELETE FROM {}
                WHERE genie_space_id = %s AND conversation_id = %s
            """).format(prompt_table_id)
            params = (self.space_id, conversation_id)
        else:
            delete_sql = sql.SQL("""
                DELETE FROM {}
                WHERE genie_space_id = %s
            """).format(prompt_table_id)
            params = (self.space_id,)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(delete_sql, params)
                deleted: int = cur.rowcount

                logger.info(
                    "Cleared prompt history",
                    layer=self.name,
                    conversation_id=conversation_id or "all",
                    deleted_count=deleted,
                )

                return deleted

    def drop_tables(self) -> dict[str, bool]:
        """
        Drop both cache and prompt history tables.

        This is useful for test cleanup to avoid accumulating test tables.

        Returns:
            Dict with 'cache' and 'prompt_history' keys indicating success
        """
        self._setup()

        results: dict[str, bool] = {"cache": False, "prompt_history": False}

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # Drop cache table
                try:
                    cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                            sql.Identifier(self.table_name)
                        )
                    )
                    results["cache"] = True
                    logger.info(
                        "Dropped cache table",
                        layer=self.name,
                        table_name=self.table_name,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to drop cache table: {e}",
                        layer=self.name,
                        table_name=self.table_name,
                    )

                # Drop prompt history table
                try:
                    cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                            sql.Identifier(self.prompt_history_table)
                        )
                    )
                    results["prompt_history"] = True
                    logger.info(
                        "Dropped prompt history table",
                        layer=self.name,
                        table_name=self.prompt_history_table,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to drop prompt history table: {e}",
                        layer=self.name,
                        table_name=self.prompt_history_table,
                    )

        return results

    @property
    def size(self) -> int:
        """Current number of entries in the cache for this Genie space."""
        self._setup()
        count_sql = sql.SQL(
            "SELECT COUNT(*) as count FROM {} WHERE genie_space_id = %s"
        ).format(sql.Identifier(self.table_name))

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(count_sql, (self.space_id,))
                row: DbRow | None = cur.fetchone()
                return row.get("count", 0) if row else 0
