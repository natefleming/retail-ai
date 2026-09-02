"""User→thread session index for the dao-ai Console sidebar.

LangGraph checkpointers are thread-scoped and cannot list threads by user
(`BaseCheckpointSaver.alist()` has no user filter), and ``dao_ai_responses``
carries no ``user_id``. This small index records ``(user_id, thread_id, title,
timestamps)`` so the Console can list a user's past sessions. It reuses the same
``DatabaseModel`` (and therefore the same connection pool) as the LangGraph
checkpointer and :class:`~dao_ai.background.store.BackgroundStore`, so it adds no
extra connection footprint. Created idempotently on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import mlflow
from loguru import logger

from dao_ai.background.store import _valid_identifier
from dao_ai.config import DatabaseModel


@dataclass(frozen=True)
class SessionRef:
    """One row of the session index (a user's past conversation thread)."""

    thread_id: str
    title: Optional[str]
    updated_at: datetime


class SessionIndexStore:
    """Async upsert/list over the user-scoped session index table."""

    def __init__(
        self,
        database: DatabaseModel,
        *,
        table_name: str = "dao_ai_sessions",
    ) -> None:
        self.database = database
        self.table = _valid_identifier(table_name)
        self._schema_ready = False

    @mlflow.trace(name="sessions.store.ensure_schema", span_type="INTERNAL")
    async def ensure_schema(self) -> None:
        """Create the index table + (user_id, updated_at) index if absent."""
        if self._schema_ready:
            return
        ddl_table = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            user_id    TEXT NOT NULL,
            thread_id  TEXT NOT NULL,
            title      TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (user_id, thread_id)
        )
        """
        ddl_index = (
            f"CREATE INDEX IF NOT EXISTS idx_{self.table}_user_updated "
            f"ON {self.table} (user_id, updated_at DESC)"
        )
        await self.database.aexecute_update([ddl_table, ddl_index])
        self._schema_ready = True
        logger.info(
            "Session index table ready",
            database=self.database.name,
            table=self.table,
        )

    async def upsert_session(
        self,
        user_id: str,
        thread_id: str,
        title: Optional[str] = None,
    ) -> None:
        """Record/refresh a user's thread. The title is set once (first turn)
        and preserved on later turns; ``updated_at`` is always bumped so the
        sidebar orders by recency."""
        await self.ensure_schema()
        sql = f"""
        INSERT INTO {self.table} (user_id, thread_id, title, updated_at)
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (user_id, thread_id) DO UPDATE
            SET title = COALESCE({self.table}.title, EXCLUDED.title),
                updated_at = NOW()
        """
        await self.database.aexecute_update(sql, (user_id, thread_id, title))

    async def list_sessions(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionRef]:
        """Return a user's threads, most-recently-updated first."""
        await self.ensure_schema()
        sql = f"""
        SELECT thread_id, title, updated_at
        FROM {self.table}
        WHERE user_id = %s
        ORDER BY updated_at DESC
        LIMIT %s OFFSET %s
        """
        rows = await self.database.aexecute_query(sql, (user_id, limit, offset))
        return [
            SessionRef(
                thread_id=row["thread_id"],
                title=row["title"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]
