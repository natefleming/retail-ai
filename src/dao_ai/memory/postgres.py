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


def _create_pool_with_fallback(
    connection_params: dict[str, Any],
    database_name: str,
    max_pool_size: int,
    timeout_seconds: int,
    kwargs: dict,
    instance_name: str | None = None,
    workspace_client: Any | None = None,
) -> ConnectionPool:
    """
    Create a connection pool with fallback from explicit user to token identity.

    If 'user' is in connection_params and the first connection attempt fails,
    will retry without the user parameter to allow Lakebase to use token identity.
    """
    pool: ConnectionPool | None = None
    last_error: Exception | None = None
    has_user = "user" in connection_params

    # Separate connection params from connection kwargs (row_factory, autocommit, etc.)
    # Connection params go directly in kwargs for psycopg connection
    # Pool/connection behavior params stay in kwargs

    logger.debug(
        f"Connection params for {database_name}: {', '.join(k + '=' + (str(v) if k != 'password' else '***') for k, v in connection_params.items())}"
    )

    # Strategy 1: Try with explicit username if provided
    if has_user:
        try:
            logger.debug(
                f"Attempting connection with explicit username for {database_name}: user={connection_params['user']}"
            )
            # Merge connection_params into kwargs for psycopg
            connection_kwargs = kwargs | connection_params
            pool = ConnectionPool(
                conninfo="",  # Empty conninfo, params come from kwargs
                min_size=1,
                max_size=max_pool_size,
                open=False,
                timeout=timeout_seconds,
                kwargs=connection_kwargs,
            )
            pool.open(wait=True, timeout=timeout_seconds)
            logger.info(
                f"Successfully connected to {database_name} with explicit username: {connection_params['user']}"
            )
            return pool
        except Exception as e:
            logger.warning(
                f"plicit username '{connection_params['user']}': {e}. "
                f"Attempting fallback to token identity..."
            )
            last_error = e
            if pool:
                try:
                    pool.close()
                except Exception:
                    pass
            pool = None

    # Strategy 2: Try existing instance roles
    if instance_name and workspace_client:
        try:
            logger.debug(
                f"Fetching existing database instance roles for {instance_name}"
            )
            roles = list(
                workspace_client.database.list_database_instance_roles(instance_name)
            )
            logger.debug(f"Found {len(roles)} existing roles for {instance_name}")

            # Get connection parameters
            host: str = connection_params.get("host", "")
            port: int = connection_params.get("port", 5432)
            dbname: str = connection_params.get("dbname", "")
            password: str = connection_params.get("password", "")
            sslmode: str = connection_params.get("sslmode", "require")

            # Try each existing role
            for role in roles:
                if not role.name:
                    continue

                try:
                    logger.debug(
                        f"Attempting connection with role '{role.name}' for {database_name}"
                    )

                    # Build PostgreSQL URI with role name as username
                    conninfo = f"postgresql://{role.name}:{password}@{host}:{port}/{dbname}?sslmode={sslmode}"

                    pool = ConnectionPool(
                        conninfo=conninfo,
                        min_size=1,
                        max_size=max_pool_size,
                        open=False,
                        timeout=timeout_seconds,
                        kwargs=kwargs,
                    )
                    pool.open(wait=True, timeout=timeout_seconds)
                    logger.info(
                        f"Successfully connected to {database_name} using role '{role.name}'"
                    )
                    return pool
                except Exception as role_error:
                    logger.debug(
                        f"Failed to connect with role '{role.name}': {role_error}"
                    )
                    if pool:
                        try:
                            pool.close()
                        except Exception:
                            pass
                    continue

        except Exception as e:
            logger.warning(f"Failed to fetch or try instance roles: {e}")

    # Strategy 3: Try with empty username (let Lakebase use token identity)
    try:
        fallback_params = connection_params.copy()

        # Get connection parameters
        host: str = fallback_params.get("host", "")
        port: int = fallback_params.get("port", 5432)
        dbname: str = fallback_params.get("dbname", "")
        password: str = fallback_params.get("password", "")
        sslmode: str = fallback_params.get("sslmode", "require")

        # Build PostgreSQL URI with empty username to use token identity
        # Format: postgresql://:<password>@<host>:<port>/<dbname>?sslmode=<sslmode>
        conninfo = f"postgresql://:{password}@{host}:{port}/{dbname}?sslmode={sslmode}"

        logger.debug(
            f"Attempting connection using token identity for {database_name} with URI format (empty username)"
        )
        logger.debug(
            f"Connection URI: postgresql://:***@{host}:{port}/{dbname}?sslmode={sslmode}"
        )

        pool = ConnectionPool(
            conninfo=conninfo,  # Use PostgreSQL URI format
            min_size=1,
            max_size=max_pool_size,
            open=False,
            timeout=timeout_seconds,
            kwargs=kwargs,  # Only pool behavior kwargs (row_factory, autocommit)
        )
        pool.open(wait=True, timeout=timeout_seconds)
        logger.info(f"Successfully connected to {database_name} using token identity")
        return pool
    except Exception as e:
        logger.error(f"Failed to connect to {database_name}: {e}")
        if last_error:
            logger.error(f"Original error with explicit username: {last_error}")
        raise e


async def _create_async_pool_with_fallback(
    connection_params: dict[str, Any],
    database_name: str,
    max_pool_size: int,
    timeout_seconds: int,
    kwargs: dict,
    instance_name: str | None = None,
    workspace_client: Any | None = None,
) -> AsyncConnectionPool:
    """
    Create an async connection pool with fallback from explicit user to token identity.

    If 'user' is in connection_params and the first connection attempt fails,
    will retry without the user parameter to allow Lakebase to use token identity.
    """
    pool: AsyncConnectionPool | None = None
    last_error: Exception | None = None
    has_user = "user" in connection_params

    logger.debug(
        f"Connection params for {database_name}: {', '.join(k + '=' + (str(v) if k != 'password' else '***') for k, v in connection_params.items())}"
    )

    # Strategy 1: Try with explicit username if provided
    if has_user:
        try:
            logger.debug(
                f"Attempting async connection with explicit username for {database_name}: user={connection_params['user']}"
            )
            # Merge connection_params into kwargs for psycopg
            connection_kwargs = kwargs | connection_params
            pool = AsyncConnectionPool(
                conninfo="",  # Empty conninfo, params come from kwargs
                max_size=max_pool_size,
                open=False,
                timeout=timeout_seconds,
                kwargs=connection_kwargs,
            )
            await pool.open(wait=True, timeout=timeout_seconds)
            logger.info(
                f"Successfully connected to {database_name} with explicit username: {connection_params['user']}"
            )
            return pool
        except Exception as e:
            logger.warning(
                f"Failed to connect with explicit username '{connection_params['user']}': {e}. "
                f"Attempting fallback to token identity..."
            )
            last_error = e
            if pool:
                try:
                    await pool.close()
                except Exception:
                    pass
            pool = None

    # Strategy 2: Try existing instance roles
    if instance_name and workspace_client:
        try:
            logger.debug(
                f"Fetching existing database instance roles for {instance_name}"
            )
            roles = list(
                workspace_client.database.list_database_instance_roles(instance_name)
            )
            logger.debug(f"Found {len(roles)} existing roles for {instance_name}")

            # Get connection parameters
            host: str = connection_params.get("host", "")
            port: int = connection_params.get("port", 5432)
            dbname: str = connection_params.get("dbname", "")
            password: str = connection_params.get("password", "")
            sslmode: str = connection_params.get("sslmode", "require")

            # Try each existing role
            for role in roles:
                if not role.name:
                    continue

                try:
                    logger.debug(
                        f"Attempting async connection with role '{role.name}' for {database_name}"
                    )

                    # Build PostgreSQL URI with role name as username
                    conninfo = f"postgresql://{role.name}:{password}@{host}:{port}/{dbname}?sslmode={sslmode}"

                    pool = AsyncConnectionPool(
                        conninfo=conninfo,
                        max_size=max_pool_size,
                        open=False,
                        timeout=timeout_seconds,
                        kwargs=kwargs,
                    )
                    await pool.open(wait=True, timeout=timeout_seconds)
                    logger.info(
                        f"Successfully connected to {database_name} using role '{role.name}'"
                    )
                    return pool
                except Exception as role_error:
                    logger.debug(
                        f"Failed to connect with role '{role.name}': {role_error}"
                    )
                    if pool:
                        try:
                            await pool.close()
                        except Exception:
                            pass
                    continue

        except Exception as e:
            logger.warning(f"Failed to fetch or try instance roles: {e}")

    # Strategy 3: Try with empty username (let Lakebase use token identity)
    try:
        fallback_params = connection_params.copy()

        # Get connection parameters
        host: str = fallback_params.get("host", "")
        port: int = fallback_params.get("port", 5432)
        dbname: str = fallback_params.get("dbname", "")
        password: str = fallback_params.get("password", "")
        sslmode: str = fallback_params.get("sslmode", "require")

        # Build PostgreSQL URI with empty username to use token identity
        # Format: postgresql://:<password>@<host>:<port>/<dbname>?sslmode=<sslmode>
        conninfo = f"postgresql://:{password}@{host}:{port}/{dbname}?sslmode={sslmode}"

        logger.debug(
            f"Attempting async connection using token identity for {database_name} with URI format (empty username)"
        )
        logger.debug(
            f"Connection URI: postgresql://:***@{host}:{port}/{dbname}?sslmode={sslmode}"
        )

        pool = AsyncConnectionPool(
            conninfo=conninfo,  # Use PostgreSQL URI format
            max_size=max_pool_size,
            open=False,
            timeout=timeout_seconds,
            kwargs=kwargs,  # Only pool behavior kwargs (row_factory, autocommit)
        )
        await pool.open(wait=True, timeout=timeout_seconds)
        logger.info(f"Successfully connected to {database_name} using token identity")
        return pool
    except Exception as e:
        logger.error(f"Failed to connect to {database_name}: {e}")
        if last_error:
            logger.error(f"Original error with explicit username: {last_error}")
        raise e


class AsyncPostgresPoolManager:
    _pools: dict[str, AsyncConnectionPool] = {}
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def get_pool(cls, database: DatabaseModel) -> AsyncConnectionPool:
        connection_key: str = database.name
        connection_params: dict[str, Any] = database.connection_params

        async with cls._lock:
            if connection_key in cls._pools:
                logger.debug(f"Reusing existing PostgreSQL pool for {database.name}")
                return cls._pools[connection_key]

            logger.debug(f"Creating new PostgreSQL pool for {database.name}")

            kwargs: dict[str, Any] = {
                "row_factory": dict_row,
                "autocommit": True,
            } | database.connection_kwargs or {}

            # Use fallback connection strategy
            pool: AsyncConnectionPool = await _create_async_pool_with_fallback(
                connection_params=connection_params,
                database_name=database.name,
                max_pool_size=database.max_pool_size,
                timeout_seconds=database.timeout_seconds,
                kwargs=kwargs,
                instance_name=database.instance_name,
                workspace_client=database.workspace_client,
            )

            cls._pools[connection_key] = pool
            return pool

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
                    # Use a short timeout to avoid blocking on pool closure
                    await asyncio.wait_for(pool.close(), timeout=2.0)
                    logger.debug(f"Closed PostgreSQL pool: {connection_key}")
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout closing pool {connection_key}, forcing closure"
                    )
                except asyncio.CancelledError:
                    logger.warning(
                        f"Pool closure cancelled for {connection_key} (shutdown in progress)"
                    )
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

            # Create store with the shared pool (using patched version)
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
        connection_params: dict[str, Any] = database.connection_params

        with cls._lock:
            if connection_key in cls._pools:
                logger.debug(f"Reusing existing PostgreSQL pool for {database.name}")
                return cls._pools[connection_key]

            logger.debug(f"Creating new PostgreSQL pool for {database.name}")

            kwargs: dict[str, Any] = {
                "row_factory": dict_row,
                "autocommit": True,
            } | database.connection_kwargs or {}

            # Use fallback connection strategy
            pool: ConnectionPool = _create_pool_with_fallback(
                connection_params=connection_params,
                database_name=database.name,
                max_pool_size=database.max_pool_size,
                timeout_seconds=database.timeout_seconds,
                kwargs=kwargs,
                instance_name=database.instance_name,
                workspace_client=database.workspace_client,
            )

            cls._pools[connection_key] = pool
            return pool

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
                logger.debug("Successfully closed all asynchronous PostgreSQL pools")
            except Exception as inner_e:
                # If all else fails, just log the error
                logger.warning(
                    f"Could not close async pools cleanly during shutdown: {inner_e}"
                )
    except Exception as e:
        logger.error(
            f"Error closing asynchronous PostgreSQL pools during shutdown: {e}"
        )


atexit.register(_shutdown_pools)
atexit.register(_shutdown_async_pools)
