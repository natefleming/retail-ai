"""Lakebase-backed persistence for long-running agent responses.

The store owns two tables (configurable names, defaults shown):

* ``dao_ai_responses`` — one row per kicked-off background request.
* ``dao_ai_response_messages`` — ordered event/item rows per response.

Both tables are created idempotently via :meth:`LongRunningStore.ensure_schema`
on first use. The pool is shared with the LangGraph checkpointer pool when the
same ``DatabaseModel`` is used, so no additional connection footprint is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Optional

import mlflow
from loguru import logger
from psycopg.rows import dict_row
from psycopg.types.json import Json

from dao_ai.config import DatabaseModel
from dao_ai.memory.postgres import AsyncPostgresPoolManager


class ResponseStatus(str, Enum):
    """Lifecycle state of a long-running response."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def terminal(cls) -> frozenset["ResponseStatus"]:
        return frozenset({cls.COMPLETED, cls.FAILED, cls.CANCELLED})

    @property
    def is_terminal(self) -> bool:
        return self in self.terminal()


@dataclass(frozen=True)
class ResponseRecord:
    """Row from the ``dao_ai_responses`` table."""

    response_id: str
    thread_id: str
    agent_task_id: Optional[str]
    status: ResponseStatus
    request_json: Optional[dict[str, Any]]
    error_json: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]


def _valid_identifier(name: str) -> str:
    """Return ``name`` if it's a safe SQL identifier, otherwise raise.

    Table names come from config and are interpolated into DDL/DML
    without bind parameters (psycopg doesn't parameterize identifiers).
    """
    if not name or not all(c.isalnum() or c == "_" for c in name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


class LongRunningStore:
    """Async CRUD over the long-running responses + messages tables."""

    def __init__(
        self,
        database: DatabaseModel,
        *,
        responses_table_name: str = "dao_ai_responses",
        messages_table_name: str = "dao_ai_response_messages",
    ) -> None:
        self.database = database
        self.responses_table = _valid_identifier(responses_table_name)
        self.messages_table = _valid_identifier(messages_table_name)
        self._schema_ready = False

    async def _pool(self):
        return await AsyncPostgresPoolManager.get_pool(self.database)

    @mlflow.trace(name="long_running.store.ensure_schema", span_type="INTERNAL")
    async def ensure_schema(self) -> None:
        """Create the two tables + index if they don't exist."""
        if self._schema_ready:
            return

        ddl_responses = f"""
        CREATE TABLE IF NOT EXISTS {self.responses_table} (
            response_id   TEXT PRIMARY KEY,
            thread_id     TEXT NOT NULL,
            agent_task_id TEXT,
            status        TEXT NOT NULL CHECK (status IN
                ('queued','in_progress','completed','failed','cancelled')),
            request_json  JSONB,
            error_json    JSONB,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at  TIMESTAMPTZ
        )
        """
        ddl_index = (
            f"CREATE INDEX IF NOT EXISTS idx_{self.responses_table}_updated_at "
            f"ON {self.responses_table} (updated_at)"
        )
        ddl_messages = f"""
        CREATE TABLE IF NOT EXISTS {self.messages_table} (
            response_id      TEXT NOT NULL,
            sequence_number  INTEGER NOT NULL DEFAULT 0,
            item             JSONB,
            stream_event     JSONB,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (response_id, sequence_number),
            FOREIGN KEY (response_id)
                REFERENCES {self.responses_table}(response_id)
                ON DELETE CASCADE
        )
        """

        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(ddl_responses)
                await cur.execute(ddl_index)
                await cur.execute(ddl_messages)
            await conn.commit()

        self._schema_ready = True
        logger.info(
            "Long-running response tables ready",
            responses=self.responses_table,
            messages=self.messages_table,
            database=self.database.name,
        )

    @mlflow.trace(name="long_running.store.create", span_type="INTERNAL")
    async def create(
        self,
        *,
        response_id: str,
        thread_id: str,
        request: Optional[dict[str, Any]] = None,
        status: ResponseStatus = ResponseStatus.IN_PROGRESS,
    ) -> None:
        sql = f"""
        INSERT INTO {self.responses_table}
            (response_id, thread_id, status, request_json)
        VALUES (%s, %s, %s, %s)
        """
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql,
                    (
                        response_id,
                        thread_id,
                        status.value,
                        Json(request) if request is not None else None,
                    ),
                )
            await conn.commit()
        logger.info(
            "Long-running response created",
            response_id=response_id,
            thread_id=thread_id,
            status=status.value,
        )

    async def set_agent_task_id(self, response_id: str, task_id: str) -> None:
        sql = (
            f"UPDATE {self.responses_table} SET agent_task_id = %s, updated_at = NOW() "
            f"WHERE response_id = %s"
        )
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (task_id, response_id))
            await conn.commit()

    @mlflow.trace(name="long_running.store.set_status", span_type="INTERNAL")
    async def set_status(
        self,
        response_id: str,
        status: ResponseStatus,
        *,
        error: Optional[dict[str, Any]] = None,
    ) -> None:
        completed_clause = ", completed_at = NOW()" if status.is_terminal else ""
        sql = (
            f"UPDATE {self.responses_table} "
            f"SET status = %s, error_json = %s, updated_at = NOW(){completed_clause} "
            f"WHERE response_id = %s"
        )
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql,
                    (
                        status.value,
                        Json(error) if error is not None else None,
                        response_id,
                    ),
                )
            await conn.commit()
        logger.info(
            "Long-running response status updated",
            response_id=response_id,
            status=status.value,
            has_error=error is not None,
        )

    async def mark_cancelled(self, response_id: str) -> None:
        await self.set_status(response_id, ResponseStatus.CANCELLED)

    @mlflow.trace(name="long_running.store.get", span_type="INTERNAL")
    async def get(self, response_id: str) -> Optional[ResponseRecord]:
        sql = (
            f"SELECT response_id, thread_id, agent_task_id, status, "
            f"request_json, error_json, created_at, updated_at, completed_at "
            f"FROM {self.responses_table} WHERE response_id = %s"
        )
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, (response_id,))
                row = await cur.fetchone()
        if row is None:
            return None
        return ResponseRecord(
            response_id=row["response_id"],
            thread_id=row["thread_id"],
            agent_task_id=row["agent_task_id"],
            status=ResponseStatus(row["status"]),
            request_json=_coerce_json(row["request_json"]),
            error_json=_coerce_json(row["error_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    async def append_event(
        self,
        response_id: str,
        event: dict[str, Any],
    ) -> int:
        """Append a streaming event row and return the assigned sequence number."""
        sql = f"""
        INSERT INTO {self.messages_table}
            (response_id, sequence_number, stream_event)
        VALUES (
            %s,
            COALESCE(
                (SELECT MAX(sequence_number) + 1 FROM {self.messages_table}
                 WHERE response_id = %s),
                0
            ),
            %s
        )
        RETURNING sequence_number
        """
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (response_id, response_id, Json(event)))
                row = await cur.fetchone()
                seq = int(row[0])
            # bump updated_at so pollers notice activity
            await conn.execute(
                f"UPDATE {self.responses_table} SET updated_at = NOW() "
                f"WHERE response_id = %s",
                (response_id,),
            )
            await conn.commit()
        return seq

    async def append_output(
        self,
        response_id: str,
        items: list[dict[str, Any]],
    ) -> None:
        """Append the final list of output items (non-streaming result)."""
        sql = f"""
        INSERT INTO {self.messages_table}
            (response_id, sequence_number, item)
        VALUES (
            %s,
            COALESCE(
                (SELECT MAX(sequence_number) + 1 FROM {self.messages_table}
                 WHERE response_id = %s),
                0
            ),
            %s
        )
        """
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                for item in items:
                    await cur.execute(sql, (response_id, response_id, Json(item)))
            await conn.commit()

    async def iter_events(
        self,
        response_id: str,
        *,
        cursor: int = 0,
    ) -> AsyncIterator[tuple[int, dict[str, Any]]]:
        """Yield ``(sequence_number, stream_event)`` tuples after ``cursor``."""
        sql = (
            f"SELECT sequence_number, stream_event FROM {self.messages_table} "
            f"WHERE response_id = %s AND sequence_number > %s "
            f"AND stream_event IS NOT NULL "
            f"ORDER BY sequence_number ASC"
        )
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, (response_id, cursor))
                async for row in cur:
                    yield int(row["sequence_number"]), _coerce_json(row["stream_event"])

    async def get_output(self, response_id: str) -> list[dict[str, Any]]:
        """Return all ``item`` rows for a response in order."""
        sql = (
            f"SELECT item FROM {self.messages_table} "
            f"WHERE response_id = %s AND item IS NOT NULL "
            f"ORDER BY sequence_number ASC"
        )
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, (response_id,))
                rows = await cur.fetchall()
        return [_coerce_json(r["item"]) for r in rows]


def _coerce_json(value: Any) -> Any:
    """JSON columns come back as dict in most configs but str in some; normalize."""
    if isinstance(value, (dict, list)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
