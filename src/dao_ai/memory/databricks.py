"""Lakebase-specific memory storage implementations.

Lakebase connections delegate to ``databricks_langchain.AsyncCheckpointSaver``
and ``databricks_langchain.AsyncDatabricksStore`` which wrap
``databricks_ai_bridge.lakebase.AsyncLakebasePool`` — that pool handles OAuth
token rotation, host resolution, TCP keepalives, and per-connection credential
refresh automatically.

This module owns:

- Lakebase pool helpers (``_lakebase_pool_kwargs``, ``_create_lakebase_pool``,
  ``_create_async_lakebase_pool``).
- Higher-level managers (``LakebaseCheckpointerManager``,
  ``LakebaseStoreManager``) that initialize the LangGraph checkpoint saver /
  store backed by Lakebase.

The Lakebase managers wrap the underlying ``AsyncCheckpointSaver`` /
``AsyncDatabricksStore`` in a lazy-init proxy: the AsyncLakebasePool open is
deferred to the first ``await`` call so the pool binds to whichever event loop
is calling. This works for both Model Serving (predict-time loop, with
``nest_asyncio``) and Databricks Apps (uvicorn's loop). Opening the pool at
sync model-load time would bind it to a transient loop that gets closed,
producing ``RuntimeError: got Future ... attached to a different loop`` on
later use.

The pool dispatchers in ``dao_ai.memory.postgres`` import the helpers below via
deferred imports to avoid pulling ``databricks_ai_bridge`` at module-load time.
"""

import asyncio
import threading
from typing import Any, AsyncIterator, Iterable, Literal, Sequence

from databricks_langchain import DatabricksEmbeddings
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    Item,
    NotProvided,
    Op,
    Result,
    SearchItem,
)
from loguru import logger
from psycopg_pool import AsyncConnectionPool

from dao_ai.config import CheckpointerModel, DatabaseModel, StoreModel
from dao_ai.memory.base import (
    CheckpointManagerBase,
    StoreManagerBase,
)

# ---------------------------------------------------------------------------
# Lakebase pool helpers (thin wrappers over databricks_ai_bridge)
# ---------------------------------------------------------------------------


def _lakebase_pool_kwargs(database: DatabaseModel) -> dict[str, Any]:
    """Build kwargs for ``LakebasePool`` / ``AsyncLakebasePool``."""
    branch = database.resolve_default_branch()
    return {
        "project": database.project,
        "branch": branch,
        "workspace_client": database.workspace_client,
    }


def _create_lakebase_pool(database: DatabaseModel, **extra: Any):
    """Create a sync ``LakebasePool`` from a ``DatabaseModel``."""
    from databricks_ai_bridge.lakebase import LakebasePool

    kwargs = _lakebase_pool_kwargs(database)
    kwargs.update(extra)
    return LakebasePool(**kwargs)


async def _create_async_lakebase_pool(database: DatabaseModel, **extra: Any):
    """Create and open an ``AsyncLakebasePool`` from a ``DatabaseModel``."""
    from databricks_ai_bridge.lakebase import AsyncLakebasePool

    kwargs = _lakebase_pool_kwargs(database)
    kwargs.update(extra)
    pool = AsyncLakebasePool(**kwargs)
    await pool.open()
    return pool


# ---------------------------------------------------------------------------
# Lazy-init wrappers
#
# These wrap the real ``databricks_langchain`` saver / store objects and defer
# the AsyncLakebasePool open + table setup to the first ``await`` call, so the
# pool is bound to the caller's event loop. This is critical because the
# manager's ``_setup`` is invoked synchronously (during model load in Model
# Serving, during agent factory build in Apps), and the AsyncLakebasePool's
# underlying psycopg_pool.AsyncConnectionPool latches its connection futures to
# the loop that opens it. Opening at sync time would bind the pool to a
# throwaway loop and break later use from the request loop.
# ---------------------------------------------------------------------------


class _LazyLakebaseCheckpointer(BaseCheckpointSaver):
    """Lazily opens the wrapped ``AsyncCheckpointSaver`` on first await call.

    Cache is keyed by ``id(asyncio.get_running_loop())`` so processes that hop
    loops — most commonly a Databricks App handling both a sync
    ``/invocations`` and a background ``/v1/responses`` — get one saver per
    loop instead of trying to reuse a Lock/pool bound to a different loop.
    The persisted checkpoint state lives in Lakebase, so per-loop savers
    still observe the same durable state.
    """

    def __init__(self, saver_kwargs: dict[str, Any], log_extra: dict[str, Any]):
        super().__init__()
        self._saver_kwargs = saver_kwargs
        self._log_extra = log_extra
        self._savers: dict[int, BaseCheckpointSaver] = {}
        self._init_locks: dict[int, asyncio.Lock] = {}
        self._dict_lock: threading.Lock = threading.Lock()

    async def _ensure(self) -> BaseCheckpointSaver:
        loop_id: int = id(asyncio.get_running_loop())
        saver = self._savers.get(loop_id)
        if saver is not None:
            return saver
        with self._dict_lock:
            init_lock = self._init_locks.get(loop_id)
            if init_lock is None:
                init_lock = asyncio.Lock()
                self._init_locks[loop_id] = init_lock
        async with init_lock:
            saver = self._savers.get(loop_id)
            if saver is None:
                from databricks_langchain import AsyncCheckpointSaver

                saver = AsyncCheckpointSaver(**self._saver_kwargs)
                await saver._lakebase.open()
                await saver.setup()
                logger.success(
                    "Lakebase checkpointer initialized (lazy)",
                    loop_id=loop_id,
                    **self._log_extra,
                )
                self._savers[loop_id] = saver
        return saver

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        return await (await self._ensure()).aget(config)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await (await self._ensure()).aget_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        saver = await self._ensure()
        async for item in saver.alist(
            config, filter=filter, before=before, limit=limit
        ):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await (await self._ensure()).aput(
            config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return await (await self._ensure()).aput_writes(
            config, writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id: str) -> None:
        return await (await self._ensure()).adelete_thread(thread_id)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        return await (await self._ensure()).acopy_thread(
            source_thread_id, target_thread_id
        )

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        return await (await self._ensure()).adelete_for_runs(run_ids)

    async def aprune(
        self, thread_ids: Sequence[str], *, strategy: str = "keep_latest"
    ) -> None:
        return await (await self._ensure()).aprune(thread_ids, strategy=strategy)


class _LazyLakebaseStore(BaseStore):
    """Lazily opens the wrapped ``AsyncDatabricksStore`` on first await call.

    Per-loop cache — see :class:`_LazyLakebaseCheckpointer` for rationale.
    """

    def __init__(self, store_kwargs: dict[str, Any], log_extra: dict[str, Any]):
        # Don't call super().__init__() because BaseStore is abstract on
        # ``batch`` / ``abatch`` — we satisfy those via overrides below.
        self._store_kwargs = store_kwargs
        self._log_extra = log_extra
        self._stores: dict[int, BaseStore] = {}
        self._init_locks: dict[int, asyncio.Lock] = {}
        self._dict_lock: threading.Lock = threading.Lock()

    async def _ensure(self) -> BaseStore:
        loop_id: int = id(asyncio.get_running_loop())
        store = self._stores.get(loop_id)
        if store is not None:
            return store
        with self._dict_lock:
            init_lock = self._init_locks.get(loop_id)
            if init_lock is None:
                init_lock = asyncio.Lock()
                self._init_locks[loop_id] = init_lock
        async with init_lock:
            store = self._stores.get(loop_id)
            if store is None:
                from databricks_langchain import AsyncDatabricksStore

                store = AsyncDatabricksStore(**self._store_kwargs)
                await store._lakebase.open()
                await store.setup()
                logger.success(
                    "Lakebase store initialized (lazy)",
                    loop_id=loop_id,
                    **self._log_extra,
                )
                self._stores[loop_id] = store
        return store

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await (await self._ensure()).abatch(ops)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        # The Lakebase store is async-only; batch() is required by BaseStore
        # but the agent never calls it from a sync context.
        raise NotImplementedError(
            "LakebaseStore is async-only; use abatch from an async context"
        )

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        return await (await self._ensure()).aget(
            namespace, key, refresh_ttl=refresh_ttl
        )

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        # Use the langgraph NOT_PROVIDED singleton (not a new NotProvided()
        # instance) so that the inner store's _ensure_ttl identity check
        # `if ttl is NOT_PROVIDED` succeeds. Constructing a fresh instance
        # here meant callers like langmem's manage_memory_tool would let
        # NotProvided() escape into PutOp.ttl, which langgraph's postgres
        # store then tried to `float()` and raised TypeError on.
        return await (await self._ensure()).aput(namespace, key, value, index, ttl=ttl)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        return await (await self._ensure()).adelete(namespace, key)

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        return await (await self._ensure()).asearch(
            namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )

    async def alist_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        return await (await self._ensure()).alist_namespaces(
            prefix=prefix,
            suffix=suffix,
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )


# ---------------------------------------------------------------------------
# Lakebase managers — delegate to databricks_langchain
# ---------------------------------------------------------------------------


class LakebaseCheckpointerManager(CheckpointManagerBase):
    """Checkpointer backed by ``databricks_langchain.AsyncCheckpointSaver``.

    Returns a :class:`_LazyLakebaseCheckpointer` that defers the
    AsyncLakebasePool open + ``saver.setup()`` to the first agent ``await``
    call. Sync ``_setup`` only validates config and assembles constructor
    kwargs — it does not touch the network or any event loop.
    """

    def __init__(self, checkpointer_model: CheckpointerModel):
        self.checkpointer_model = checkpointer_model
        self._checkpointer: BaseCheckpointSaver | None = None
        self._setup_complete = False

    def checkpointer(self) -> BaseCheckpointSaver:
        if not self._setup_complete or not self._checkpointer:
            self._setup()
        if not self._checkpointer:
            raise RuntimeError("LakebaseCheckpointerManager initialization failed")
        return self._checkpointer

    def _setup(self):
        if self._setup_complete:
            return

        database = self.checkpointer_model.database
        if database is None:
            raise ValueError(
                "Database configuration is required for Lakebase checkpointer"
            )

        branch = database.resolve_default_branch()
        saver_kwargs: dict[str, Any] = {
            "project": database.project,
            "branch": branch,
            "workspace_client": database.workspace_client,
            "min_size": 1,
            "max_size": database.max_pool_size,
            "timeout": float(database.timeout_seconds),
            "check": AsyncConnectionPool.check_connection,
        }

        self._checkpointer = _LazyLakebaseCheckpointer(
            saver_kwargs=saver_kwargs,
            log_extra={
                "checkpointer": self.checkpointer_model.name,
                "project": database.project,
                "branch": branch,
            },
        )
        self._setup_complete = True
        logger.debug(
            "Lakebase checkpointer registered (lazy open on first await)",
            checkpointer=self.checkpointer_model.name,
            project=database.project,
            branch=branch,
        )


class LakebaseStoreManager(StoreManagerBase):
    """Store backed by ``databricks_langchain.AsyncDatabricksStore``.

    Returns a :class:`_LazyLakebaseStore` that defers the AsyncLakebasePool
    open + ``store.setup()`` to the first agent ``await`` call. Sync
    ``_setup`` only assembles constructor kwargs.
    """

    def __init__(self, store_model: StoreModel):
        self.store_model = store_model
        self._store: BaseStore | None = None
        self._setup_complete = False

    def store(self) -> BaseStore:
        if not self._setup_complete or not self._store:
            self._setup()
        if not self._store:
            raise RuntimeError("LakebaseStoreManager initialization failed")
        return self._store

    def _setup(self):
        if self._setup_complete:
            return

        database = self.store_model.database
        if database is None:
            raise ValueError("Database configuration is required for Lakebase store")

        branch = database.resolve_default_branch()
        store_kwargs: dict[str, Any] = {
            "project": database.project,
            "branch": branch,
            "workspace_client": database.workspace_client,
            "min_size": 1,
            "max_size": database.max_pool_size,
            "timeout": float(database.timeout_seconds),
            "check": AsyncConnectionPool.check_connection,
        }

        if self.store_model.embedding_model is not None:
            embedding_endpoint = self.store_model.embedding_model.name
            embeddings = DatabricksEmbeddings(endpoint=embedding_endpoint)

            from dao_ai.memory.core import _resolve_embedding_dims

            embedding_dims = _resolve_embedding_dims(embeddings, self.store_model.dims)

            store_kwargs["embedding_endpoint"] = embedding_endpoint
            store_kwargs["embedding_dims"] = embedding_dims

            logger.debug(
                "Configuring store embeddings",
                endpoint=embedding_endpoint,
                dimensions=embedding_dims,
            )

        self._store = _LazyLakebaseStore(
            store_kwargs=store_kwargs,
            log_extra={
                "store": self.store_model.name,
                "project": database.project,
                "branch": branch,
                "embeddings_enabled": self.store_model.embedding_model is not None,
            },
        )
        self._setup_complete = True
        logger.debug(
            "Lakebase store registered (lazy open on first await)",
            store=self.store_model.name,
            project=database.project,
            branch=branch,
        )
