"""A2A :class:`TaskStore` implementations for dao-ai.

Two stores:

* :class:`a2a.server.tasks.InMemoryTaskStore` (from a2a-sdk) — process-local,
  tasks lost on restart.
* :class:`LakebaseTaskStore` — persists tasks in Lakebase (or any Postgres),
  reusing the same :class:`AsyncPostgresPoolManager` pool the LangGraph
  checkpointer and :class:`dao_ai.long_running.LongRunningStore` use whenever
  those point at the same :class:`dao_ai.config.DatabaseModel`. This lets A2A
  tasks survive worker restarts and stay consistent across replicas.

Selection is driven by :func:`build_task_store`, which reads
``config.app.a2a.task_store`` (an :class:`A2ATaskStoreModel`): absent
``database`` → in-memory, present → Lakebase. The A2A task store is
configured independently of :class:`dao_ai.config.LongRunningModel`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import mlflow
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.types import Task
from loguru import logger
from psycopg.rows import dict_row
from psycopg.types.json import Json

from dao_ai.long_running.store import _valid_identifier  # reuse the validator
from dao_ai.memory.postgres import AsyncPostgresPoolManager

if TYPE_CHECKING:
    from a2a.server.context import ServerCallContext

    from dao_ai.config import AppConfig, DatabaseModel


class LakebaseTaskStore(TaskStore):
    """Persist A2A :class:`Task` objects in a Lakebase (Postgres) table.

    Shares the connection pool with :class:`dao_ai.long_running.LongRunningStore`
    and the LangGraph Postgres checkpointer via
    :class:`dao_ai.memory.postgres.AsyncPostgresPoolManager`, so there is no
    additional connection footprint when ``app.long_running`` is also
    configured.

    The table is created idempotently on first use via :meth:`ensure_schema`.
    """

    def __init__(
        self,
        database: "DatabaseModel",
        *,
        table_name: str = "dao_ai_a2a_tasks",
    ) -> None:
        self.database = database
        self.table_name = _valid_identifier(table_name)
        self._schema_ready = False

    async def _pool(self):
        return await AsyncPostgresPoolManager.get_pool(self.database)

    @mlflow.trace(name="a2a.task_store.ensure_schema", span_type="INTERNAL")
    async def ensure_schema(self) -> None:
        """Create the tasks table + indexes if they don't exist.

        Skips DDL entirely when the table is already present so the
        connecting role only needs ``CREATE`` on schema public for the
        first-ever boot. In deployments where the platform or a
        provisioning notebook pre-creates the table (as
        ``03_provision_lakebase.py`` should for any persisted store),
        the agent SP can run with read/write-only privileges.
        """
        if self._schema_ready:
            logger.trace(
                "A2A task store schema cache hit; skipping probe",
                table=self.table_name,
                database=self.database.name,
            )
            return

        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                # Cheap existence probe; uses to_regclass which returns
                # NULL when the table doesn't exist instead of raising.
                await cur.execute(
                    "SELECT to_regclass(%s) AS reg",
                    (f"public.{self.table_name}",),
                )
                row = await cur.fetchone()
                if row is None:
                    # No rows shouldn't happen, but stay safe.
                    table_present = False
                elif isinstance(row, dict):
                    table_present = row.get("reg") is not None
                else:
                    table_present = row[0] is not None

                if not table_present:
                    ddl_tasks = f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        task_id    TEXT PRIMARY KEY,
                        context_id TEXT NOT NULL,
                        state      TEXT NOT NULL,
                        task_json  JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                    ddl_idx_ctx = (
                        f"CREATE INDEX IF NOT EXISTS "
                        f"idx_{self.table_name}_context_id "
                        f"ON {self.table_name} (context_id)"
                    )
                    ddl_idx_upd = (
                        f"CREATE INDEX IF NOT EXISTS "
                        f"idx_{self.table_name}_updated_at "
                        f"ON {self.table_name} (updated_at)"
                    )
                    await cur.execute(ddl_tasks)
                    await cur.execute(ddl_idx_ctx)
                    await cur.execute(ddl_idx_upd)
                    logger.info(
                        "A2A task store DDL applied",
                        table=self.table_name,
                        database=self.database.name,
                    )
            await conn.commit()

        self._schema_ready = True
        logger.success(
            "A2A task store schema ready",
            table=self.table_name,
            database=self.database.name,
            pre_existing=table_present,
        )

    @mlflow.trace(name="a2a.task_store.save", span_type="INTERNAL")
    async def save(
        self,
        task: Task,
        context: "Optional[ServerCallContext]" = None,
    ) -> None:
        """Upsert a task by ``task_id``."""
        await self.ensure_schema()
        sql = f"""
        INSERT INTO {self.table_name} (task_id, context_id, state, task_json)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (task_id) DO UPDATE
            SET context_id = EXCLUDED.context_id,
                state      = EXCLUDED.state,
                task_json  = EXCLUDED.task_json,
                updated_at = NOW()
        """
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql,
                    (
                        task.id,
                        task.context_id,
                        task.status.state.value,
                        Json(task.model_dump(mode="json")),
                    ),
                )
            await conn.commit()
        logger.debug(
            "A2A task persisted",
            table=self.table_name,
            task_id=task.id,
            context_id=task.context_id,
            state=task.status.state.value,
        )

    @mlflow.trace(name="a2a.task_store.get", span_type="INTERNAL")
    async def get(
        self,
        task_id: str,
        context: "Optional[ServerCallContext]" = None,
    ) -> Task | None:
        await self.ensure_schema()
        sql = f"SELECT task_json FROM {self.table_name} WHERE task_id = %s"
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, (task_id,))
                row = await cur.fetchone()
        if row is None:
            logger.trace(
                "A2A task store miss",
                table=self.table_name,
                task_id=task_id,
            )
            return None
        task = Task.model_validate(row["task_json"])
        logger.trace(
            "A2A task store hit",
            table=self.table_name,
            task_id=task_id,
            state=task.status.state.value,
        )
        return task

    @mlflow.trace(name="a2a.task_store.delete", span_type="INTERNAL")
    async def delete(
        self,
        task_id: str,
        context: "Optional[ServerCallContext]" = None,
    ) -> None:
        await self.ensure_schema()
        sql = f"DELETE FROM {self.table_name} WHERE task_id = %s"
        pool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (task_id,))
                rowcount = cur.rowcount
            await conn.commit()
        logger.debug(
            "A2A task deleted",
            table=self.table_name,
            task_id=task_id,
            rowcount=rowcount,
        )


def build_task_store(config: "AppConfig") -> TaskStore:
    """Pick a :class:`TaskStore` based on config.

    Resolution:

    * ``app.a2a.task_store.database`` is ``None`` (default) →
      :class:`InMemoryTaskStore`.
    * ``app.a2a.task_store.database`` is set →
      :class:`LakebaseTaskStore` against that :class:`DatabaseModel`,
      using ``app.a2a.task_store.table`` for the table name.

    Independent of ``app.long_running``: point this and the long-running
    store at the same :class:`DatabaseModel` to share a connection pool.
    """
    from dao_ai.apps.a2a.agent_card import effective_a2a

    a2a = effective_a2a(config)
    db = a2a.task_store.database

    if db is None:
        logger.info("A2A task store: in-memory")
        return InMemoryTaskStore()

    logger.info(
        "A2A task store: Lakebase",
        table=a2a.task_store.table,
        database=db.name,
    )
    return LakebaseTaskStore(database=db, table_name=a2a.task_store.table)
