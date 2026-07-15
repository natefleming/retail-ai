"""PostgreSQL memory storage implementations.

Standard PostgreSQL connections use ``psycopg_pool`` directly. The pool
managers here also dispatch to Lakebase pools when ``database.is_lakebase`` is
true; the Lakebase implementation lives in :mod:`dao_ai.memory.databricks` and
is loaded via deferred import so this module remains usable without
``databricks_ai_bridge`` installed.
"""

import asyncio
import atexit
import threading
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from dao_ai.config import CheckpointerModel, DatabaseModel, StoreModel
from dao_ai.memory.base import (
    CheckpointManagerBase,
    StoreManagerBase,
)

# ---------------------------------------------------------------------------
# Standard PostgreSQL pool (non-Lakebase)
# ---------------------------------------------------------------------------

_POOL_MAX_RETRIES = 3
_POOL_RETRY_DELAY_SECONDS = 5


def _create_pool(
    connection_params: dict[str, Any],
    database_name: str,
    max_pool_size: int,
    timeout_seconds: int,
    kwargs: dict,
) -> ConnectionPool:
    """Create a psycopg connection pool for standard PostgreSQL with retry logic."""
    import time

    logger.debug(
        "Creating connection pool",
        database=database_name,
        timeout=timeout_seconds,
    )

    last_error: Exception | None = None

    for attempt in range(1, _POOL_MAX_RETRIES + 1):
        pool = ConnectionPool(
            conninfo="",
            min_size=1,
            max_size=max_pool_size,
            open=False,
            timeout=timeout_seconds,
            check=ConnectionPool.check_connection,
            kwargs=lambda _p=connection_params, _k=kwargs: {**_k, **_p},
        )
        try:
            pool.open(wait=True, timeout=timeout_seconds)
            logger.success(
                "PostgreSQL connection pool created",
                database=database_name,
                pool_size=max_pool_size,
                attempt=attempt,
            )
            return pool
        except Exception as e:
            last_error = e
            logger.warning(
                "Pool open failed, retrying",
                database=database_name,
                attempt=attempt,
                max_retries=_POOL_MAX_RETRIES,
                error=str(e),
            )
            try:
                pool.close()
            except Exception:
                pass
            if attempt < _POOL_MAX_RETRIES:
                time.sleep(_POOL_RETRY_DELAY_SECONDS)

    raise last_error  # type: ignore[misc]


async def _create_async_pool(
    connection_params: dict[str, Any],
    database_name: str,
    max_pool_size: int,
    timeout_seconds: int,
    kwargs: dict,
) -> AsyncConnectionPool:
    """Create an async psycopg connection pool for standard PostgreSQL with retry logic."""
    logger.debug(
        "Creating async connection pool",
        database=database_name,
        timeout=timeout_seconds,
    )

    last_error: Exception | None = None

    for attempt in range(1, _POOL_MAX_RETRIES + 1):
        pool = AsyncConnectionPool(
            conninfo="",
            max_size=max_pool_size,
            open=False,
            timeout=timeout_seconds,
            check=AsyncConnectionPool.check_connection,
            kwargs=lambda _p=connection_params, _k=kwargs: {**_k, **_p},
        )
        try:
            await pool.open(wait=True, timeout=timeout_seconds)
            logger.success(
                "Async PostgreSQL connection pool created",
                database=database_name,
                pool_size=max_pool_size,
                attempt=attempt,
            )
            return pool
        except Exception as e:
            last_error = e
            logger.warning(
                "Async pool open failed, retrying",
                database=database_name,
                attempt=attempt,
                max_retries=_POOL_MAX_RETRIES,
                error=str(e),
            )
            try:
                await pool.close()
            except Exception:
                pass
            if attempt < _POOL_MAX_RETRIES:
                await asyncio.sleep(_POOL_RETRY_DELAY_SECONDS)

    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Async pool manager — shared pools keyed by (db name, event-loop id)
# ---------------------------------------------------------------------------


class AsyncPostgresPoolManager:
    """
    Asynchronous connection pool manager that shares pools by database config.

    For Lakebase databases, creates an ``AsyncLakebasePool`` which handles
    host resolution, credential rotation, and TCP keepalives automatically.
    For standard PostgreSQL, creates a ``psycopg_pool.AsyncConnectionPool``.
    """

    _pools: dict[str, AsyncConnectionPool] = {}
    _lakebase_pools: dict[str, Any] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    async def get_pool(cls, database: DatabaseModel) -> AsyncConnectionPool:
        loop_id: int = id(asyncio.get_running_loop())
        connection_key: str = f"{database.name}::{loop_id}"

        with cls._lock:
            if connection_key in cls._pools:
                logger.trace(
                    "Reusing existing async pool",
                    database=database.name,
                    loop_id=loop_id,
                )
                return cls._pools[connection_key]

        logger.debug("Creating new async pool", database=database.name)

        if database.is_lakebase:
            from dao_ai.memory.databricks import _create_async_lakebase_pool

            lakebase_pool = await _create_async_lakebase_pool(
                database,
                min_size=1,
                max_size=database.max_pool_size,
                timeout=float(database.timeout_seconds),
                check=AsyncConnectionPool.check_connection,
            )
            with cls._lock:
                cls._lakebase_pools[connection_key] = lakebase_pool
            pool = lakebase_pool.pool
            logger.success(
                "Async Lakebase pool created",
                database=database.name,
                project=database.project,
                pool_size=database.max_pool_size,
            )
        else:
            connection_params: dict[str, Any] = database.connection_params
            kwargs: dict[str, Any] = {
                "row_factory": dict_row,
                "autocommit": True,
            } | (database.connection_kwargs or {})

            pool = await _create_async_pool(
                connection_params=connection_params,
                database_name=database.name,
                max_pool_size=database.max_pool_size,
                timeout_seconds=database.timeout_seconds,
                kwargs=kwargs,
            )

        with cls._lock:
            cls._pools[connection_key] = pool
        return pool

    @classmethod
    async def close_all_pools(cls):
        with cls._lock:
            lakebase_snapshot = dict(cls._lakebase_pools)
            pool_snapshot = dict(cls._pools)
            cls._lakebase_pools.clear()
            cls._pools.clear()

        for key, lakebase_pool in lakebase_snapshot.items():
            try:
                await asyncio.wait_for(lakebase_pool.close(), timeout=2.0)
                logger.debug("Async Lakebase pool closed", pool=key)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                logger.warning(
                    "Error closing async Lakebase pool", pool=key, error=str(e)
                )

        for key, pool in pool_snapshot.items():
            if key in lakebase_snapshot:
                continue
            try:
                await asyncio.wait_for(pool.close(), timeout=2.0)
                logger.debug("Async PostgreSQL pool closed", pool=key)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                logger.warning("Error closing async pool", pool=key, error=str(e))


# ---------------------------------------------------------------------------
# Sync pool manager — used by Genie pg_vector cache
# ---------------------------------------------------------------------------


class PostgresPoolManager:
    """
    Synchronous connection pool manager.

    For Lakebase databases, creates a ``LakebasePool`` which handles
    host resolution, credential rotation, and TCP keepalives automatically.
    For standard PostgreSQL, creates a ``psycopg_pool.ConnectionPool``.
    """

    _pools: dict[str, ConnectionPool] = {}
    _lakebase_pools: dict[str, Any] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_pool(cls, database: DatabaseModel) -> ConnectionPool:
        connection_key: str = str(database.name)

        with cls._lock:
            if connection_key in cls._pools:
                logger.trace("Reusing existing PostgreSQL pool", database=database.name)
                return cls._pools[connection_key]

            logger.debug("Creating new PostgreSQL pool", database=database.name)

            if database.is_lakebase:
                from dao_ai.memory.databricks import _create_lakebase_pool

                lakebase_pool = _create_lakebase_pool(
                    database,
                    min_size=1,
                    max_size=database.max_pool_size,
                    timeout=float(database.timeout_seconds),
                    check=ConnectionPool.check_connection,
                )
                cls._lakebase_pools[connection_key] = lakebase_pool
                pool = lakebase_pool.pool
                logger.success(
                    "Lakebase pool created",
                    database=database.name,
                    project=database.project,
                    pool_size=database.max_pool_size,
                )
            else:
                connection_params: dict[str, Any] = database.connection_params
                kwargs: dict[str, Any] = {
                    "row_factory": dict_row,
                    "autocommit": True,
                } | (database.connection_kwargs or {})

                pool = _create_pool(
                    connection_params=connection_params,
                    database_name=database.name,
                    max_pool_size=database.max_pool_size,
                    timeout_seconds=database.timeout_seconds,
                    kwargs=kwargs,
                )

            # Validate connectivity
            try:
                with pool.connection(timeout=5.0) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
            except Exception as e:
                pool.close()
                cls._lakebase_pools.pop(connection_key, None)
                hint = (
                    f" Lakebase project '{database.project}'"
                    if database.is_lakebase
                    else f" host '{database.host}'"
                )
                raise ConnectionError(
                    f"Cannot connect to{hint}. "
                    f"Verify the instance is running and accessible: {e}"
                ) from e

            cls._pools[connection_key] = pool
            return pool

    @classmethod
    def close_all_pools(cls):
        with cls._lock:
            for key, lp in cls._lakebase_pools.items():
                try:
                    lp.close()
                    logger.debug("Lakebase pool closed", pool=key)
                except Exception as e:
                    logger.error("Error closing Lakebase pool", pool=key, error=str(e))
            cls._lakebase_pools.clear()

            for key, pool in cls._pools.items():
                try:
                    pool.close()
                    logger.debug("PostgreSQL pool closed", pool=key)
                except Exception as e:
                    logger.error(
                        "Error closing PostgreSQL pool", pool=key, error=str(e)
                    )
            cls._pools.clear()


# ---------------------------------------------------------------------------
# Standard PostgreSQL managers (non-Lakebase)
# ---------------------------------------------------------------------------


class AsyncPostgresStoreManager(StoreManagerBase):
    """Store manager for standard PostgreSQL using shared async connection pools."""

    def __init__(self, store_model: StoreModel):
        self.store_model = store_model
        self.pool: Optional[AsyncConnectionPool] = None
        self._store: Optional[BaseStore] = None
        self._setup_complete = False

    def store(self) -> BaseStore:
        if not self._setup_complete or not self._store:
            self._setup()
        if not self._store:
            raise RuntimeError("PostgresStore initialization failed")
        return self._store

    def _setup(self):
        if self._setup_complete:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()

        asyncio.run(self._async_setup())

    async def _async_setup(self):
        if self._setup_complete:
            return

        if not self.store_model.database:
            raise ValueError("Database configuration is required for PostgresStore")

        from langgraph.store.postgres.aio import AsyncPostgresStore

        self.pool = await self.store_model.database.aget_pool()
        self._store = AsyncPostgresStore(conn=self.pool)
        await self._store.setup()

        self._setup_complete = True
        logger.success(
            "Async PostgreSQL store initialized", store=self.store_model.name
        )


class AsyncPostgresCheckpointerManager(CheckpointManagerBase):
    """Checkpointer manager for standard PostgreSQL using shared async connection pools."""

    def __init__(self, checkpointer_model: CheckpointerModel):
        self.checkpointer_model = checkpointer_model
        self.pool: Optional[AsyncConnectionPool] = None
        self._checkpointer: Optional[BaseCheckpointSaver] = None
        self._setup_complete = False

    def checkpointer(self) -> BaseCheckpointSaver:
        if not self._setup_complete or not self._checkpointer:
            self._setup()
        if not self._checkpointer:
            raise RuntimeError("PostgresSaver initialization failed")
        return self._checkpointer

    def _setup(self):
        if self._setup_complete:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()

        asyncio.run(self._async_setup())

    async def _async_setup(self):
        if self._setup_complete:
            return

        if not self.checkpointer_model.database:
            raise ValueError("Database configuration is required for PostgresSaver")

        from langgraph.checkpoint.postgres.aio import AsyncShallowPostgresSaver

        self.pool = await self.checkpointer_model.database.aget_pool()
        self._checkpointer = AsyncShallowPostgresSaver(conn=self.pool)
        await self._checkpointer.setup()

        self._setup_complete = True
        logger.success(
            "Async PostgreSQL checkpointer initialized",
            checkpointer=self.checkpointer_model.name,
        )


# ---------------------------------------------------------------------------
# Shutdown hooks
# ---------------------------------------------------------------------------


def _shutdown_pools() -> None:
    try:
        PostgresPoolManager.close_all_pools()
        logger.debug("All synchronous PostgreSQL pools closed during shutdown")
    except Exception as e:
        logger.error(
            "Error closing synchronous PostgreSQL pools during shutdown", error=str(e)
        )


def _shutdown_async_pools() -> None:
    try:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(AsyncPostgresPoolManager.close_all_pools())
            logger.debug("Scheduled async pool closure in running event loop")
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(AsyncPostgresPoolManager.close_all_pools())
                logger.debug("All asynchronous PostgreSQL pools closed during shutdown")
            except Exception as inner_e:
                logger.warning(
                    "Could not close async pools cleanly during shutdown",
                    error=str(inner_e),
                )
    except Exception as e:
        logger.error(
            "Error closing asynchronous PostgreSQL pools during shutdown", error=str(e)
        )


atexit.register(_shutdown_pools)
atexit.register(_shutdown_async_pools)
