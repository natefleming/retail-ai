"""Lakebase-backed persistence for inbound channels.

Two responsibilities, two tables:

* **Dedup** — every Meta delivery is keyed on ``message_id`` with a
  ``UNIQUE`` constraint. Conflicting INSERTs return zero rows, signalling
  to the caller that the delivery has already been processed and should
  be acknowledged with 200 without re-running the agent.
* **Thread mapping** — ``wa_id`` (optionally scoped by
  ``phone_number_id``) is upserted to a stable LangGraph ``thread_id`` so
  follow-up messages from the same user resume the same conversation and
  hit the existing checkpointer/memory machinery.

When :class:`DatabaseModel` is None the store degrades to an in-process
:class:`dict`. That mode is intended for unit tests and local spikes —
production deployments MUST configure a database (see
``config/examples/21_channels/whatsapp.yaml``).

Schema is created idempotently on first use via :meth:`ensure_schema`.
The pool is shared with the LangGraph checkpointer and
:class:`LongRunningStore` when the same ``DatabaseModel`` is referenced.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional

import mlflow
from loguru import logger

from dao_ai.config import DatabaseModel


def _valid_identifier(name: str) -> str:
    """Reject SQL identifiers that aren't safe to interpolate.

    Table names come from config and are interpolated into DDL/DML
    without bind parameters (psycopg doesn't parameterize identifiers).
    """
    if not name or not all(c.isalnum() or c == "_" for c in name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


@dataclass(frozen=True)
class DedupResult:
    """Outcome of attempting to record an inbound message id."""

    inserted: bool
    """True if this is the first time we've seen this message_id."""


class ChannelStore:
    """Async dedup + thread mapping for inbound channels.

    When :attr:`database` is None the store keeps state in process
    dictionaries protected by an asyncio lock — useful for tests and
    local development. Pass a real :class:`DatabaseModel` for production.
    """

    def __init__(
        self,
        database: Optional[DatabaseModel],
        *,
        dedup_table_name: str = "dao_ai_whatsapp_inbound_dedup",
        threads_table_name: str = "dao_ai_whatsapp_threads",
    ) -> None:
        self.database = database
        self.dedup_table = _valid_identifier(dedup_table_name)
        self.threads_table = _valid_identifier(threads_table_name)
        self._schema_ready = False
        # In-memory fallback state (only used when database is None)
        self._mem_seen: set[str] = set()
        self._mem_threads: dict[str, str] = {}
        self._mem_lock = asyncio.Lock()

    @property
    def is_in_memory(self) -> bool:
        return self.database is None

    async def _pool(self):
        # Local import: keeps the in-memory mode free of psycopg dependency
        from dao_ai.memory.postgres import AsyncPostgresPoolManager

        assert self.database is not None
        return await AsyncPostgresPoolManager.get_pool(self.database)

    @mlflow.trace(name="channels.store.ensure_schema", span_type="INTERNAL")
    async def ensure_schema(self) -> None:
        """Create the dedup and threads tables if they don't exist."""
        if self._schema_ready or self.database is None:
            self._schema_ready = True
            return

        ddl_dedup = f"""
        CREATE TABLE IF NOT EXISTS {self.dedup_table} (
            message_id   TEXT PRIMARY KEY,
            channel      TEXT NOT NULL,
            received_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
        ddl_threads = f"""
        CREATE TABLE IF NOT EXISTS {self.threads_table} (
            thread_key   TEXT PRIMARY KEY,
            thread_id    TEXT NOT NULL,
            channel      TEXT NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """

        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(ddl_dedup)
                await cur.execute(ddl_threads)
            await conn.commit()

        self._schema_ready = True
        logger.info(
            "Channel store tables ready",
            dedup_table=self.dedup_table,
            threads_table=self.threads_table,
            database=self.database.name,
        )

    @mlflow.trace(name="channels.store.record_message", span_type="INTERNAL")
    async def record_message(
        self,
        *,
        message_id: str,
        channel: str,
    ) -> DedupResult:
        """Try to record ``message_id``. Returns ``inserted=False`` on conflict."""
        if self.database is None:
            async with self._mem_lock:
                if message_id in self._mem_seen:
                    return DedupResult(inserted=False)
                self._mem_seen.add(message_id)
                return DedupResult(inserted=True)

        sql = (
            f"INSERT INTO {self.dedup_table} (message_id, channel) "
            f"VALUES (%s, %s) ON CONFLICT (message_id) DO NOTHING"
        )
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (message_id, channel))
                inserted = (cur.rowcount or 0) > 0
            await conn.commit()
        return DedupResult(inserted=inserted)

    @mlflow.trace(name="channels.store.get_or_create_thread", span_type="INTERNAL")
    async def get_or_create_thread(
        self,
        *,
        thread_key: str,
        channel: str,
    ) -> str:
        """Look up the LangGraph ``thread_id`` for ``thread_key``, creating one if absent."""
        if self.database is None:
            async with self._mem_lock:
                existing = self._mem_threads.get(thread_key)
                if existing is not None:
                    return existing
                new_id = str(uuid.uuid4())
                self._mem_threads[thread_key] = new_id
                return new_id

        select_sql = (
            f"SELECT thread_id FROM {self.threads_table} WHERE thread_key = %s"
        )
        insert_sql = (
            f"INSERT INTO {self.threads_table} (thread_key, thread_id, channel) "
            f"VALUES (%s, %s, %s) ON CONFLICT (thread_key) DO UPDATE "
            f"SET updated_at = NOW() RETURNING thread_id"
        )
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(select_sql, (thread_key,))
                row = await cur.fetchone()
                if row is not None:
                    return str(row[0])

                new_id = str(uuid.uuid4())
                await cur.execute(insert_sql, (thread_key, new_id, channel))
                row = await cur.fetchone()
                assert row is not None
                resolved = str(row[0])
            await conn.commit()
        return resolved


__all__ = ["ChannelStore", "DedupResult"]
