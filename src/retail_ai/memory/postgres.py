"""
PostgreSQL-based Store and Checkpointer Managers with Shared Connection Pools

This module provides PostgreSQL implementations of StoreManager and CheckpointerManager
that share connection pools when using the same database configuration.

Key Features:
- Shared connection pools: Multiple managers using the same database configuration
  will share the same AsyncConnectionPool, reducing resource usage.
- Lazy initialization: Connection pools and stores/checkpointers are created only
  when first accessed via store() or checkpointer() methods.
- Event loop safe: Handles synchronous interface requirements while managing
  async PostgreSQL operations internally.
- Resource management: Provides cleanup methods for proper resource disposal.

Usage:
    # Create database configuration
    db_config = DatabaseModel(
        name="my_db",
        connection_url="postgresql://user:pass@localhost:5432/db",
        max_pool_size=20,
        timeout=5
    )

    # Create managers - they will share the same connection pool
    store_manager = PostgresStoreManager(StoreModel(
        name="my_store",
        type=StorageType.POSTGRES,
        database=db_config
    ))

    checkpointer_manager = PostgresCheckpointerManager(CheckpointerModel(
        name="my_checkpointer",
        type=StorageType.POSTGRES,
        database=db_config
    ))

    # Use them - pools will be created automatically
    store = store_manager.store()
    checkpointer = checkpointer_manager.checkpointer()

    # Cleanup when done (optional)
    await PostgresPoolManager.close_all_pools()
"""

import asyncio
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from retail_ai.config import CheckpointerModel, DatabaseModel, StoreModel
from retail_ai.memory.base import (
    CheckpointManagerBase,
    StoreManagerBase,
)


class PostgresPoolManager:
    _pools: dict[str, AsyncConnectionPool] = {}
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def get_pool(cls, database: DatabaseModel) -> AsyncConnectionPool:
        connection_key: str = (
            f"{database.connection_url}#{database.max_pool_size}#{database.timeout}"
        )

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
                conninfo=database.connection_url,
                max_size=database.max_pool_size,
                open=False,
                timeout=database.timeout,
                kwargs=kwargs,
            )

            try:
                await pool.open(wait=True, timeout=database.timeout)
                cls._pools[connection_key] = pool
                return pool
            except Exception as e:
                logger.error(
                    f"Failed to create PostgreSQL pool for {database.name}: {e}"
                )
                raise e

    @classmethod
    async def close_pool(cls, database: DatabaseModel):
        """
        Close a specific pool. Usually called during cleanup.
        """
        connection_key = (
            f"{database.connection_url}#{database.max_pool_size}#{database.timeout}"
        )

        async with cls._lock:
            if connection_key in cls._pools:
                pool = cls._pools.pop(connection_key)
                await pool.close()
                logger.debug(f"Closed PostgreSQL pool for {database.name}")

    @classmethod
    async def close_all_pools(cls):
        """
        Close all managed pools. Usually called during application shutdown.
        """
        async with cls._lock:
            for connection_key, pool in cls._pools.items():
                try:
                    await pool.close()
                    logger.debug(f"Closed PostgreSQL pool: {connection_key}")
                except Exception as e:
                    logger.error(f"Error closing pool {connection_key}: {e}")
            cls._pools.clear()


class PostgresStoreManager(StoreManagerBase):
    """
    Manager for PostgresStore that uses shared connection pools.
    """

    def __init__(self, store_model: StoreModel):
        self.store_model = store_model
        self.pool: Optional[AsyncConnectionPool] = None
        self._store: Optional[AsyncPostgresStore] = None
        self._setup_complete = False

    def store(self) -> BaseStore:
        """
        Get the initialized store. Sets up the store if not already done.
        """
        if not self._setup_complete or not self._store:
            self._setup()

        if not self._store:
            raise RuntimeError("PostgresStore initialization failed")

        return self._store

    def _setup(self):
        """
        Run the async setup in a synchronous context.
        """
        try:
            # Try to get the current event loop
            _ = asyncio.get_running_loop()
            # If we're in an async context, we can't block with run_until_complete
            raise RuntimeError(
                "PostgresStore setup cannot be called from within an async context. "
                "Please call setup() from a synchronous context first."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e):
                # No event loop is running, safe to create one
                asyncio.run(self._async_setup())
            else:
                # Re-raise the original error about async context
                raise

    async def _async_setup(self):
        """
        Async version of setup for internal use.
        """
        if self._setup_complete:
            return

        if not self.store_model.database:
            raise ValueError("Database configuration is required for PostgresStore")

        try:
            # Get shared pool
            self.pool = await PostgresPoolManager.get_pool(self.store_model.database)

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


class PostgresCheckpointerManager(CheckpointManagerBase):
    """
    Manager for PostgresSaver that uses shared connection pools.
    """

    def __init__(self, checkpointer_model: CheckpointerModel):
        self.checkpointer_model = checkpointer_model
        self.pool: Optional[AsyncConnectionPool] = None
        self._checkpointer: Optional[AsyncPostgresSaver] = None
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
        Run the async setup in a synchronous context.
        """
        try:
            # Try to get the current event loop
            _ = asyncio.get_running_loop()
            # If we're in an async context, we can't block with run_until_complete
            raise RuntimeError(
                "PostgresSaver setup cannot be called from within an async context. "
                "Please call setup() from a synchronous context first."
            )
        except RuntimeError as e:
            if "no running event loop" in str(e):
                # No event loop is running, safe to create one
                asyncio.run(self._async_setup())
            else:
                # Re-raise the original error about async context
                raise

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
            self.pool = await PostgresPoolManager.get_pool(
                self.checkpointer_model.database
            )

            # Create checkpointer with the shared pool
            self._checkpointer = AsyncPostgresSaver(conn=self.pool)
            await self._checkpointer.setup()

            self._setup_complete = True
            logger.debug(
                f"PostgresSaver initialized successfully for {self.checkpointer_model.name}"
            )

        except Exception as e:
            logger.error(f"Error setting up PostgresSaver: {e}")
            raise
