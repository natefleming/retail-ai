"""
Databricks-native memory storage implementations.

Provides CheckpointSaver and DatabricksStore implementations using
Databricks Lakebase for persistent storage, with async support.

See:
- https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_langchain.html#databricks_langchain.CheckpointSaver
- https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_langchain.html#databricks_langchain.DatabricksStore
"""

import asyncio
from collections.abc import AsyncIterator, Iterable, Sequence
from functools import partial
from typing import Any, Literal

from databricks_langchain import (
    CheckpointSaver as DatabricksCheckpointSaver,
)
from databricks_langchain import (
    DatabricksEmbeddings,
    DatabricksStore,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.store.base import BaseStore, Item, Op, Result, SearchItem
from loguru import logger

from dao_ai.config import (
    CheckpointerModel,
    StoreModel,
)
from dao_ai.memory.base import (
    CheckpointManagerBase,
    StoreManagerBase,
)

# Type alias for namespace path
NamespacePath = tuple[str, ...]

# Sentinel for not-provided values
NOT_PROVIDED = object()


class AsyncDatabricksCheckpointSaver(DatabricksCheckpointSaver):
    """
    Async wrapper for DatabricksCheckpointSaver.

    Provides async implementations of checkpoint methods by delegating
    to the sync methods using asyncio.to_thread().
    """

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async version of get_tuple."""
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        logger.debug(f"aget_tuple: Fetching checkpoint for thread_id={thread_id}")
        result = await asyncio.to_thread(self.get_tuple, config)
        if result:
            logger.debug(f"aget_tuple: Found checkpoint for thread_id={thread_id}")
        else:
            logger.debug(f"aget_tuple: No checkpoint found for thread_id={thread_id}")
        return result

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        """Async version of get."""
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        logger.debug(f"aget: Fetching checkpoint for thread_id={thread_id}")
        result = await asyncio.to_thread(self.get, config)
        if result:
            logger.debug(f"aget: Found checkpoint for thread_id={thread_id}")
        else:
            logger.debug(f"aget: No checkpoint found for thread_id={thread_id}")
        return result

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version of put."""
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        checkpoint_id = checkpoint.get("id", "unknown")
        logger.debug(
            f"aput: Saving checkpoint id={checkpoint_id} for thread_id={thread_id}"
        )
        result = await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions
        )
        logger.debug(f"aput: Checkpoint saved for thread_id={thread_id}")
        return result

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version of put_writes."""
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        logger.debug(
            f"aput_writes: Saving {len(writes)} writes for thread_id={thread_id}, "
            f"task_id={task_id}"
        )
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)
        logger.debug(f"aput_writes: Writes saved for thread_id={thread_id}")

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version of list."""
        thread_id = (
            config.get("configurable", {}).get("thread_id", "unknown")
            if config
            else "all"
        )
        logger.debug(
            f"alist: Listing checkpoints for thread_id={thread_id}, limit={limit}"
        )
        # Get all items from sync iterator in a thread
        items = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        logger.debug(f"alist: Found {len(items)} checkpoints for thread_id={thread_id}")
        for item in items:
            yield item

    async def adelete_thread(self, thread_id: str) -> None:
        """Async version of delete_thread."""
        logger.debug(f"adelete_thread: Deleting thread_id={thread_id}")
        await asyncio.to_thread(self.delete_thread, thread_id)
        logger.debug(f"adelete_thread: Thread deleted thread_id={thread_id}")


class AsyncDatabricksStore(DatabricksStore):
    """
    Async wrapper for DatabricksStore.

    Provides async implementations of store methods by delegating
    to the sync methods using asyncio.to_thread().
    """

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Async version of batch."""
        ops_list = list(ops)
        logger.debug(f"abatch: Executing {len(ops_list)} operations")
        result = await asyncio.to_thread(self.batch, ops_list)
        logger.debug(f"abatch: Completed {len(result)} operations")
        return result

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Async version of get."""
        ns_str = "/".join(namespace)
        logger.debug(f"aget: Fetching key={key} from namespace={ns_str}")
        result = await asyncio.to_thread(
            partial(self.get, namespace, key, refresh_ttl=refresh_ttl)
        )
        if result:
            logger.debug(f"aget: Found item key={key} in namespace={ns_str}")
        else:
            logger.debug(f"aget: No item found key={key} in namespace={ns_str}")
        return result

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None = None,
    ) -> None:
        """Async version of put."""
        ns_str = "/".join(namespace)
        logger.debug(f"aput: Storing key={key} in namespace={ns_str}")
        # Handle the ttl parameter - only pass if explicitly provided
        if ttl is not None:
            await asyncio.to_thread(
                partial(self.put, namespace, key, value, index, ttl=ttl)
            )
        else:
            await asyncio.to_thread(partial(self.put, namespace, key, value, index))
        logger.debug(f"aput: Stored key={key} in namespace={ns_str}")

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Async version of delete."""
        ns_str = "/".join(namespace)
        logger.debug(f"adelete: Deleting key={key} from namespace={ns_str}")
        await asyncio.to_thread(self.delete, namespace, key)
        logger.debug(f"adelete: Deleted key={key} from namespace={ns_str}")

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
        """Async version of search."""
        ns_str = "/".join(namespace_prefix)
        logger.debug(
            f"asearch: Searching namespace_prefix={ns_str}, query={query}, limit={limit}"
        )
        result = await asyncio.to_thread(
            partial(
                self.search,
                namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            )
        )
        logger.debug(f"asearch: Found {len(result)} items in namespace_prefix={ns_str}")
        return result

    async def alist_namespaces(
        self,
        *,
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """Async version of list_namespaces."""
        prefix_str = "/".join(prefix) if prefix else "all"
        logger.debug(
            f"alist_namespaces: Listing namespaces prefix={prefix_str}, limit={limit}"
        )
        result = await asyncio.to_thread(
            partial(
                self.list_namespaces,
                prefix=prefix,
                suffix=suffix,
                max_depth=max_depth,
                limit=limit,
                offset=offset,
            )
        )
        logger.debug(f"alist_namespaces: Found {len(result)} namespaces")
        return result


class DatabricksCheckpointerManager(CheckpointManagerBase):
    """
    Checkpointer manager using Databricks CheckpointSaver with async support.

    Uses AsyncDatabricksCheckpointSaver which wraps databricks_langchain.CheckpointSaver
    with async method implementations for LangGraph async streaming compatibility.

    Required configuration via CheckpointerModel.database:
    - instance_name: The Databricks Lakebase instance name
    - workspace_client: WorkspaceClient (supports OBO, service principal, or default auth)

    See: https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_langchain.html#databricks_langchain.CheckpointSaver
    """

    def __init__(self, checkpointer_model: CheckpointerModel):
        self.checkpointer_model = checkpointer_model
        self._checkpointer: BaseCheckpointSaver | None = None

    def checkpointer(self) -> BaseCheckpointSaver:
        if self._checkpointer is None:
            database = self.checkpointer_model.database
            if database is None:
                raise ValueError(
                    "Database configuration is required for Databricks checkpointer. "
                    "Please provide a 'database' field in the checkpointer configuration."
                )

            instance_name = database.instance_name
            workspace_client = database.workspace_client

            logger.debug(
                f"Creating AsyncDatabricksCheckpointSaver for instance: {instance_name}"
            )

            checkpointer = AsyncDatabricksCheckpointSaver(
                instance_name=instance_name,
                workspace_client=workspace_client,
            )

            # Setup the checkpointer (creates necessary tables if needed)
            logger.debug(f"Setting up checkpoint tables for instance: {instance_name}")
            checkpointer.setup()
            logger.debug(
                f"Checkpoint tables setup complete for instance: {instance_name}"
            )

            self._checkpointer = checkpointer

        return self._checkpointer


class DatabricksStoreManager(StoreManagerBase):
    """
    Store manager using Databricks DatabricksStore with async support.

    Uses AsyncDatabricksStore which wraps databricks_langchain.DatabricksStore
    with async method implementations for LangGraph async streaming compatibility.

    Required configuration via StoreModel.database:
    - instance_name: The Databricks Lakebase instance name
    - workspace_client: WorkspaceClient (supports OBO, service principal, or default auth)

    Optional configuration via StoreModel:
    - embedding_model: LLMModel for embeddings (will be converted to DatabricksEmbeddings)
    - dims: Embedding dimensions

    See: https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_langchain.html#databricks_langchain.DatabricksStore
    """

    def __init__(self, store_model: StoreModel):
        self.store_model = store_model
        self._store: BaseStore | None = None

    def store(self) -> BaseStore:
        if self._store is None:
            database = self.store_model.database
            if database is None:
                raise ValueError(
                    "Database configuration is required for Databricks store. "
                    "Please provide a 'database' field in the store configuration."
                )

            instance_name = database.instance_name
            workspace_client = database.workspace_client

            # Build embeddings configuration if embedding_model is provided
            embeddings: DatabricksEmbeddings | None = None
            embedding_dims: int | None = None

            if self.store_model.embedding_model is not None:
                embedding_endpoint = self.store_model.embedding_model.name
                embedding_dims = self.store_model.dims

                logger.debug(
                    f"Configuring embeddings: endpoint={embedding_endpoint}, dims={embedding_dims}"
                )

                embeddings = DatabricksEmbeddings(endpoint=embedding_endpoint)

            logger.debug(f"Creating AsyncDatabricksStore for instance: {instance_name}")

            store = AsyncDatabricksStore(
                instance_name=instance_name,
                workspace_client=workspace_client,
                embeddings=embeddings,
                embedding_dims=embedding_dims,
            )

            # Setup the store (creates necessary tables if needed)
            store.setup()
            self._store = store

        return self._store


__all__ = [
    "AsyncDatabricksCheckpointSaver",
    "AsyncDatabricksStore",
    "DatabricksCheckpointerManager",
    "DatabricksStoreManager",
]
