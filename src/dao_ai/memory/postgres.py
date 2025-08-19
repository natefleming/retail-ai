import asyncio
import atexit
import threading
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import ShallowPostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncShallowPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from dao_ai.config import CheckpointerModel, DatabaseModel, StoreModel
from dao_ai.memory.base import (
    CheckpointManagerBase,
    StoreManagerBase,
)


class AsyncPostgresPoolManager:
    _pools: dict[str, AsyncConnectionPool] = {}
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def get_pool(cls, database: DatabaseModel) -> AsyncConnectionPool:
        connection_key: str = database.name
        connection_url: str = database.connection_url

        async with cls._lock:
            if connection_key in cls._pools:
                logger.debug(f"Reusing existing PostgreSQL pool for {database.name}")
                return cls._pools[connection_key]

            logger.debug(f"Creating new PostgreSQL pool for {database.name}")

            kwargs: dict[str, Any] = {
                "row_factory": dict_row,
                "autocommit": True,
            } | database.connection_kwargs or {}

            pool: AsyncConnectionPool = AsyncConnectionPool(
                conninfo=connection_url,
                max_size=database.max_pool_size,
                open=False,
                timeout=database.timeout_seconds,
                kwargs=kwargs,
            )

            try:
                await pool.open(wait=True, timeout=database.timeout_seconds)
                cls._pools[connection_key] = pool
                return pool
            except Exception as e:
                logger.error(
                    f"Failed to create PostgreSQL pool for {database.name}: {e}"
                )
                raise e

    @classmethod
    async def close_pool(cls, database: DatabaseModel):
        connection_key: str = database.name

        async with cls._lock:
            if connection_key in cls._pools:
                pool = cls._pools.pop(connection_key)
                await pool.close()
                logger.debug(f"Closed PostgreSQL pool for {database.name}")

    @classmethod
    async def close_all_pools(cls):
        async with cls._lock:
            for connection_key, pool in cls._pools.items():
                try:
                    await pool.close()
                    logger.debug(f"Closed PostgreSQL pool: {connection_key}")
                except Exception as e:
                    logger.error(f"Error closing pool {connection_key}: {e}")
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

            # Create store with the shared pool
            self._store = AsyncPostgresStore(conn=self.pool)
            await self._store.setup()

            self._setup_complete = True
            logger.debug(
                f"PostgresStore initialized successfully for {self.store_model.name}"
            )

        except Exception as e:
            logger.error(f"Error setting up PostgresStore: {e}")
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
        Run the async setup. Works in both sync and async contexts when nest_asyncio is applied.
        """
        if self._setup_complete:
            return

        # With nest_asyncio applied in notebooks, asyncio.run() works everywhere
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
            logger.debug(
                f"PostgresSaver initialized successfully for {self.checkpointer_model.name}"
            )

        except Exception as e:
            logger.error(f"Error setting up PostgresSaver: {e}")
            raise


class PostgresPoolManager:
    """
    Synchronous PostgreSQL connection pool manager that shares pools
    based on database configuration.
    """

    _pools: dict[str, ConnectionPool] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_pool(cls, database: DatabaseModel) -> ConnectionPool:
        connection_key: str = str(database.name)
        connection_url: str = database.connection_url

        with cls._lock:
            if connection_key in cls._pools:
                logger.debug(f"Reusing existing PostgreSQL pool for {database.name}")
                return cls._pools[connection_key]

            logger.debug(f"Creating new PostgreSQL pool for {database.name}")

            kwargs: dict[str, Any] = {
                "row_factory": dict_row,
                "autocommit": True,
            } | database.connection_kwargs or {}

            pool: ConnectionPool = ConnectionPool(
                conninfo=connection_url,
                max_size=database.max_pool_size,
                open=False,
                timeout=database.timeout_seconds,
                kwargs=kwargs,
            )

            try:
                pool.open(wait=True, timeout=database.timeout_seconds)
                cls._pools[connection_key] = pool
                return pool
            except Exception as e:
                logger.error(
                    f"Failed to create PostgreSQL pool for {database.name}: {e}"
                )
                raise e

    @classmethod
    def close_pool(cls, database: DatabaseModel):
        connection_key: str = database.name

        with cls._lock:
            if connection_key in cls._pools:
                pool = cls._pools.pop(connection_key)
                pool.close()
                logger.debug(f"Closed PostgreSQL pool for {database.name}")

    @classmethod
    def close_all_pools(cls):
        with cls._lock:
            for connection_key, pool in cls._pools.items():
                try:
                    pool.close()
                    logger.debug(f"Closed PostgreSQL pool: {connection_key}")
                except Exception as e:
                    logger.error(f"Error closing pool {connection_key}: {e}")
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
            logger.debug(
                f"PostgresStore initialized successfully for {self.store_model.name}"
            )

        except Exception as e:
            logger.error(f"Error setting up PostgresStore: {e}")
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
            logger.debug(
                f"PostgresSaver initialized successfully for {self.checkpointer_model.name}"
            )

        except Exception as e:
            logger.error(f"Error setting up PostgresSaver: {e}")
            raise


def _shutdown_pools():
    try:
        PostgresPoolManager.close_all_pools()
        logger.debug("Successfully closed all synchronous PostgreSQL pools")
    except Exception as e:
        logger.error(f"Error closing synchronous PostgreSQL pools during shutdown: {e}")


def _shutdown_async_pools():
    try:
        asyncio.run(AsyncPostgresPoolManager.close_all_pools())
        logger.debug("Successfully closed all asynchronous PostgreSQL pools")
    except Exception as e:
        logger.error(
            f"Error closing asynchronous PostgreSQL pools during shutdown: {e}"
        )


atexit.register(_shutdown_pools)
atexit.register(_shutdown_async_pools)
