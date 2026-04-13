import asyncio
import atexit
import threading
from typing import Any, Optional

from databricks_ai_bridge.lakebase import AsyncLakebasePool, LakebasePool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import ShallowPostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncShallowPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from dao_ai.config import CheckpointerModel, DatabaseModel, StoreModel, value_of
from dao_ai.memory.base import (
    CheckpointManagerBase,
    StoreManagerBase,
)


class DaoLakebasePool(LakebasePool):
    """LakebasePool subclass that accepts an explicit username.

    The upstream LakebasePool infers the Postgres username via
    workspace_client.current_user.me(). On Databricks clusters this may
    return the cluster identity rather than the configured service principal.
    This subclass allows callers to provide the correct username directly.
    """

    def __init__(self, *, username: str, **kwargs):
        self._explicit_username = username
        super().__init__(**kwargs)

    def _infer_username(self) -> str:
        return self._explicit_username


class DaoAsyncLakebasePool(AsyncLakebasePool):
    """AsyncLakebasePool subclass that accepts an explicit username.

    See DaoLakebasePool for rationale.
    """

    def __init__(self, *, username: str, **kwargs):
        self._explicit_username = username
        super().__init__(**kwargs)

    def _infer_username(self) -> str:
        return self._explicit_username


def _lakebase_kwargs(database: DatabaseModel) -> dict:
    """Build LakebasePool kwargs from a DatabaseModel (provisioned only).

    LakebasePool from databricks-ai-bridge only supports provisioned Lakebase
    instances via instance_name. Autoscaling Lakebase uses standard PostgreSQL
    pools with connection_params (which resolves host/credentials via the
    Postgres autoscaling API).
    """
    if database.is_lakebase_autoscaling:
        raise ValueError(
            "LakebasePool does not support autoscaling Lakebase. "
            "Use connection_params with a standard PostgreSQL pool instead."
        )
    kwargs: dict = {"workspace_client": database.workspace_client}
    kwargs["instance_name"] = database.instance_name
    return kwargs


def _create_lakebase_pool(database: DatabaseModel, **pool_kwargs) -> LakebasePool:
    """Create a LakebasePool from a DatabaseModel.

    When the config provides an explicit client_id, uses DaoLakebasePool
    to ensure the correct Postgres username (on clusters, current_user.me()
    may return the cluster identity instead of the configured service principal).
    """
    kwargs = _lakebase_kwargs(database) | pool_kwargs
    client_id_value = value_of(database.client_id) if database.client_id else None

    if client_id_value:
        return DaoLakebasePool(username=client_id_value, **kwargs)
    return LakebasePool(**kwargs)


def _create_async_lakebase_pool(
    database: DatabaseModel, **pool_kwargs
) -> AsyncLakebasePool:
    """Create an AsyncLakebasePool from a DatabaseModel.

    See _create_lakebase_pool for rationale.
    """
    kwargs = _lakebase_kwargs(database) | pool_kwargs
    client_id_value = value_of(database.client_id) if database.client_id else None

    if client_id_value:
        return DaoAsyncLakebasePool(username=client_id_value, **kwargs)
    return AsyncLakebasePool(**kwargs)


_POOL_MAX_RETRIES = 3
_POOL_RETRY_DELAY_SECONDS = 5


def _create_pool(
    connection_params: dict[str, Any],
    database_name: str,
    max_pool_size: int,
    timeout_seconds: int,
    kwargs: dict,
) -> ConnectionPool:
    """Create a connection pool with retry logic for transient failures."""
    import time

    safe_params = {
        k: (str(v) if k != "password" else "***") for k, v in connection_params.items()
    }
    logger.debug(
        "Creating connection pool",
        database=database_name,
        timeout=timeout_seconds,
        **safe_params,
    )

    connection_kwargs = kwargs | connection_params
    last_error: Exception | None = None

    for attempt in range(1, _POOL_MAX_RETRIES + 1):
        pool = ConnectionPool(
            conninfo="",
            min_size=1,
            max_size=max_pool_size,
            open=False,
            timeout=timeout_seconds,
            kwargs=connection_kwargs,
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
    """Create an async connection pool with retry logic for transient failures."""
    safe_params = {
        k: (str(v) if k != "password" else "***") for k, v in connection_params.items()
    }
    logger.debug(
        "Creating async connection pool",
        database=database_name,
        timeout=timeout_seconds,
        **safe_params,
    )

    connection_kwargs = kwargs | connection_params
    last_error: Exception | None = None

    for attempt in range(1, _POOL_MAX_RETRIES + 1):
        pool = AsyncConnectionPool(
            conninfo="",
            max_size=max_pool_size,
            open=False,
            timeout=timeout_seconds,
            kwargs=connection_kwargs,
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


class AsyncPostgresPoolManager:
    """
    Asynchronous PostgreSQL connection pool manager that shares pools
    based on database configuration.

    For provisioned Lakebase (instance_name), uses AsyncLakebasePool from
    databricks_ai_bridge with automatic token rotation and host resolution.
    For autoscaling Lakebase (project) and standard PostgreSQL, uses
    psycopg_pool.AsyncConnectionPool with connection_params.
    """

    _pools: dict[str, AsyncConnectionPool] = {}
    _lakebase_pools: dict[str, AsyncLakebasePool] = {}
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def get_pool(cls, database: DatabaseModel) -> AsyncConnectionPool:
        connection_key: str = database.name

        async with cls._lock:
            if connection_key in cls._pools:
                logger.trace(
                    "Reusing existing async PostgreSQL pool", database=database.name
                )
                return cls._pools[connection_key]

            logger.debug("Creating new async PostgreSQL pool", database=database.name)

            if database.is_lakebase_provisioned:
                # AsyncLakebasePool handles token rotation and host resolution
                lakebase_pool = _create_async_lakebase_pool(
                    database,
                    min_size=1,
                    max_size=database.max_pool_size,
                    timeout=float(database.timeout_seconds),
                )
                await lakebase_pool.open()
                cls._lakebase_pools[connection_key] = lakebase_pool
                pool = lakebase_pool.pool
                logger.success(
                    "Async Lakebase provisioned pool created",
                    database=database.name,
                    instance=database.instance_name,
                    pool_size=database.max_pool_size,
                )
            else:
                # Autoscaling Lakebase and standard PostgreSQL use connection_params
                connection_params: dict[str, Any] = database.connection_params
                kwargs: dict[str, Any] = {
                    "row_factory": dict_row,
                    "autocommit": True,
                } | database.connection_kwargs or {}

                pool = await _create_async_pool(
                    connection_params=connection_params,
                    database_name=database.name,
                    max_pool_size=database.max_pool_size,
                    timeout_seconds=database.timeout_seconds,
                    kwargs=kwargs,
                )
                if database.is_lakebase_autoscaling:
                    logger.success(
                        "Async Lakebase autoscaling pool created",
                        database=database.name,
                        project=database.project,
                        pool_size=database.max_pool_size,
                    )

            cls._pools[connection_key] = pool
            return pool

    @classmethod
    async def close_pool(cls, database: DatabaseModel):
        connection_key: str = database.name

        async with cls._lock:
            # Close AsyncLakebasePool if it exists (handles underlying pool cleanup)
            if connection_key in cls._lakebase_pools:
                lakebase_pool = cls._lakebase_pools.pop(connection_key)
                await lakebase_pool.close()
                cls._pools.pop(connection_key, None)
                logger.debug("Async Lakebase pool closed", database=database.name)
            elif connection_key in cls._pools:
                pool = cls._pools.pop(connection_key)
                await pool.close()
                logger.debug("Async PostgreSQL pool closed", database=database.name)

    @classmethod
    async def close_all_pools(cls):
        async with cls._lock:
            # Close all AsyncLakebasePool instances first
            for connection_key, lakebase_pool in cls._lakebase_pools.items():
                try:
                    await asyncio.wait_for(lakebase_pool.close(), timeout=2.0)
                    logger.debug("Async Lakebase pool closed", pool=connection_key)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout closing async Lakebase pool, forcing closure",
                        pool=connection_key,
                    )
                except asyncio.CancelledError:
                    logger.warning(
                        "Async Lakebase pool closure cancelled (shutdown in progress)",
                        pool=connection_key,
                    )
                except Exception as e:
                    logger.error(
                        "Error closing async Lakebase pool",
                        pool=connection_key,
                        error=str(e),
                    )
            cls._lakebase_pools.clear()

            # Close any remaining standard async PostgreSQL pools
            for connection_key, pool in cls._pools.items():
                try:
                    await asyncio.wait_for(pool.close(), timeout=2.0)
                    logger.debug("Async PostgreSQL pool closed", pool=connection_key)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timeout closing async pool, forcing closure",
                        pool=connection_key,
                    )
                except asyncio.CancelledError:
                    logger.warning(
                        "Async pool closure cancelled (shutdown in progress)",
                        pool=connection_key,
                    )
                except Exception as e:
                    logger.error(
                        "Error closing async pool", pool=connection_key, error=str(e)
                    )
            cls._pools.clear()


class AsyncPostgresStoreManager(StoreManagerBase):
    """
    Manager for PostgresStore that uses shared connection pools.
    """

    def __init__(self, store_model: StoreModel):
        self.store_model = store_model
        self.pool: Optional[AsyncConnectionPool] = None
        self._store: Optional[AsyncPostgresStore] = None
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

        try:
            # Get shared pool
            self.pool = await AsyncPostgresPoolManager.get_pool(
                self.store_model.database
            )

            # Create store with the shared pool (using patched version)
            self._store = AsyncPostgresStore(conn=self.pool)

            await self._store.setup()

            self._setup_complete = True
            logger.success(
                "Async PostgreSQL store initialized", store=self.store_model.name
            )

        except Exception as e:
            logger.error(
                "Error setting up async PostgreSQL store",
                store=self.store_model.name,
                error=str(e),
            )
            raise


class AsyncPostgresCheckpointerManager(CheckpointManagerBase):
    """
    Manager for PostgresSaver that uses shared connection pools.
    """

    def __init__(self, checkpointer_model: CheckpointerModel):
        self.checkpointer_model = checkpointer_model
        self.pool: Optional[AsyncConnectionPool] = None
        self._checkpointer: Optional[AsyncShallowPostgresSaver] = None
        self._setup_complete = False

    def checkpointer(self) -> BaseCheckpointSaver:
        """
        Get the initialized checkpointer. Sets up the checkpointer if not already done.
        """
        if not self._setup_complete or not self._checkpointer:
            self._setup()

        if not self._checkpointer:
            raise RuntimeError("PostgresSaver initialization failed")

        return self._checkpointer

    def _setup(self):
        """
        Run the async setup synchronously. Uses nest_asyncio if an event loop
        is already running (e.g. Databricks serverless notebooks).
        """
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
        """
        Async version of setup for internal use.
        """
        if self._setup_complete:
            return

        if not self.checkpointer_model.database:
            raise ValueError("Database configuration is required for PostgresSaver")

        try:
            # Get shared pool
            self.pool = await AsyncPostgresPoolManager.get_pool(
                self.checkpointer_model.database
            )

            # Create checkpointer with the shared pool
            self._checkpointer = AsyncShallowPostgresSaver(conn=self.pool)
            await self._checkpointer.setup()

            self._setup_complete = True
            logger.success(
                "Async PostgreSQL checkpointer initialized",
                checkpointer=self.checkpointer_model.name,
            )

        except Exception as e:
            logger.error(
                "Error setting up async PostgreSQL checkpointer",
                checkpointer=self.checkpointer_model.name,
                error=str(e),
            )
            raise


class PostgresPoolManager:
    """
    Synchronous PostgreSQL connection pool manager that shares pools
    based on database configuration.

    For provisioned Lakebase (instance_name), uses LakebasePool from
    databricks_ai_bridge with automatic token rotation and host resolution.
    For autoscaling Lakebase (project) and standard PostgreSQL, uses
    psycopg_pool.ConnectionPool with connection_params.
    """

    _pools: dict[str, ConnectionPool] = {}
    _lakebase_pools: dict[str, LakebasePool] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_pool(cls, database: DatabaseModel) -> ConnectionPool:
        connection_key: str = str(database.name)

        with cls._lock:
            if connection_key in cls._pools:
                logger.trace("Reusing existing PostgreSQL pool", database=database.name)
                return cls._pools[connection_key]

            logger.debug("Creating new PostgreSQL pool", database=database.name)

            if database.is_lakebase_provisioned:
                # LakebasePool handles token rotation and host resolution
                lakebase_pool = _create_lakebase_pool(
                    database,
                    min_size=1,
                    max_size=database.max_pool_size,
                    timeout=float(database.timeout_seconds),
                )
                cls._lakebase_pools[connection_key] = lakebase_pool
                pool = lakebase_pool.pool
                logger.success(
                    "Lakebase provisioned pool created",
                    database=database.name,
                    instance=database.instance_name,
                    pool_size=database.max_pool_size,
                )
            else:
                # Autoscaling Lakebase and standard PostgreSQL use connection_params
                connection_params: dict[str, Any] = database.connection_params
                kwargs: dict[str, Any] = {
                    "row_factory": dict_row,
                    "autocommit": True,
                } | database.connection_kwargs or {}

                pool = _create_pool(
                    connection_params=connection_params,
                    database_name=database.name,
                    max_pool_size=database.max_pool_size,
                    timeout_seconds=database.timeout_seconds,
                    kwargs=kwargs,
                )
                if database.is_lakebase_autoscaling:
                    logger.success(
                        "Lakebase autoscaling pool created",
                        database=database.name,
                        project=database.project,
                        pool_size=database.max_pool_size,
                    )

            # Validate connectivity before caching the pool.
            try:
                with pool.connection(timeout=5.0) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
            except Exception as e:
                pool.close()
                cls._lakebase_pools.pop(connection_key, None)
                instance_hint = (
                    f" Lakebase instance '{database.instance_name}'"
                    if database.is_lakebase_provisioned
                    else f" Lakebase project '{database.project}'"
                    if database.is_lakebase_autoscaling
                    else f" host '{database.host}'"
                )
                raise ConnectionError(
                    f"Cannot connect to{instance_hint}. "
                    f"Verify the instance is running and accessible: {e}"
                ) from e

            cls._pools[connection_key] = pool
            return pool

    @classmethod
    def close_pool(cls, database: DatabaseModel):
        connection_key: str = database.name

        with cls._lock:
            # Close LakebasePool if it exists (handles underlying pool cleanup)
            if connection_key in cls._lakebase_pools:
                lakebase_pool = cls._lakebase_pools.pop(connection_key)
                lakebase_pool.close()
                cls._pools.pop(connection_key, None)
                logger.debug("Lakebase pool closed", database=database.name)
            elif connection_key in cls._pools:
                pool = cls._pools.pop(connection_key)
                pool.close()
                logger.debug("PostgreSQL pool closed", database=database.name)

    @classmethod
    def close_all_pools(cls):
        with cls._lock:
            # Close all LakebasePool instances first
            for connection_key, lakebase_pool in cls._lakebase_pools.items():
                try:
                    lakebase_pool.close()
                    logger.debug("Lakebase pool closed", pool=connection_key)
                except Exception as e:
                    logger.error(
                        "Error closing Lakebase pool",
                        pool=connection_key,
                        error=str(e),
                    )
            cls._lakebase_pools.clear()

            # Close any remaining standard PostgreSQL pools
            for connection_key, pool in cls._pools.items():
                # Skip if already closed via LakebasePool
                if connection_key not in cls._lakebase_pools:
                    try:
                        pool.close()
                        logger.debug("PostgreSQL pool closed", pool=connection_key)
                    except Exception as e:
                        logger.error(
                            "Error closing PostgreSQL pool",
                            pool=connection_key,
                            error=str(e),
                        )
            cls._pools.clear()


class PostgresStoreManager(StoreManagerBase):
    """
    Synchronous manager for PostgresStore that uses shared connection pools.
    """

    def __init__(self, store_model: StoreModel):
        self.store_model = store_model
        self.pool: Optional[ConnectionPool] = None
        self._store: Optional[PostgresStore] = None
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

        if not self.store_model.database:
            raise ValueError("Database configuration is required for PostgresStore")

        try:
            # Get shared pool
            self.pool = PostgresPoolManager.get_pool(self.store_model.database)

            # Create store with the shared pool
            self._store = PostgresStore(conn=self.pool)
            self._store.setup()

            self._setup_complete = True
            logger.success("PostgreSQL store initialized", store=self.store_model.name)

        except Exception as e:
            logger.error(
                "Error setting up PostgreSQL store",
                store=self.store_model.name,
                error=str(e),
            )
            raise


class PostgresCheckpointerManager(CheckpointManagerBase):
    """
    Synchronous manager for PostgresSaver that uses shared connection pools.
    """

    def __init__(self, checkpointer_model: CheckpointerModel):
        self.checkpointer_model = checkpointer_model
        self.pool: Optional[ConnectionPool] = None
        self._checkpointer: Optional[ShallowPostgresSaver] = None
        self._setup_complete = False

    def checkpointer(self) -> BaseCheckpointSaver:
        """
        Get the initialized checkpointer. Sets up the checkpointer if not already done.
        """
        if not self._setup_complete or not self._checkpointer:
            self._setup()

        if not self._checkpointer:
            raise RuntimeError("PostgresSaver initialization failed")

        return self._checkpointer

    def _setup(self):
        """
        Set up the checkpointer synchronously.
        """
        if self._setup_complete:
            return

        if not self.checkpointer_model.database:
            raise ValueError("Database configuration is required for PostgresSaver")

        try:
            # Get shared pool
            self.pool = PostgresPoolManager.get_pool(self.checkpointer_model.database)

            # Create checkpointer with the shared pool
            self._checkpointer = ShallowPostgresSaver(conn=self.pool)
            self._checkpointer.setup()

            self._setup_complete = True
            logger.success(
                "PostgreSQL checkpointer initialized",
                checkpointer=self.checkpointer_model.name,
            )

        except Exception as e:
            logger.error(
                "Error setting up PostgreSQL checkpointer",
                checkpointer=self.checkpointer_model.name,
                error=str(e),
            )
            raise


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
        # Try to get the current event loop first
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, create a task
            loop.create_task(AsyncPostgresPoolManager.close_all_pools())
            logger.debug("Scheduled async pool closure in running event loop")
        except RuntimeError:
            # No running loop, try to get or create one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    # Loop is closed, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(AsyncPostgresPoolManager.close_all_pools())
                logger.debug("All asynchronous PostgreSQL pools closed during shutdown")
            except Exception as inner_e:
                # If all else fails, just log the error
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
