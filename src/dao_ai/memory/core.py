from typing import Any

from databricks_langchain import DatabricksEmbeddings
from langchain_core.embeddings.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from loguru import logger

from dao_ai.config import (
    CheckpointerModel,
    InferenceEndpointModel,
    StorageType,
    StoreModel,
)
from dao_ai.memory.base import (
    CheckpointManagerBase,
    StoreManagerBase,
)


def _resolve_embedding_dims(embeddings: Embeddings, configured_dims: int | None) -> int:
    """Resolve embedding dimensions, auto-detecting from the model if not configured.

    Follows the same pattern as the Genie contextual cache
    (see genie/cache/context_aware/base.py ``_initialize_embeddings``).
    """
    if configured_dims is not None:
        return configured_dims

    sample: list[float] = embeddings.embed_documents(["test"])[0]
    dims: int = len(sample)
    logger.debug("Auto-detected embedding dimensions", dims=dims)
    return dims


class InMemoryStoreManager(StoreManagerBase):
    def __init__(self, store_model: StoreModel):
        self.store_model = store_model

    def store(self) -> BaseStore:
        embedding_model: InferenceEndpointModel = self.store_model.embedding_model

        logger.debug(
            "Creating in-memory store", embeddings_enabled=embedding_model is not None
        )

        index: dict[str, Any] = None

        if embedding_model:
            embeddings: Embeddings = DatabricksEmbeddings(endpoint=embedding_model.name)

            def embed_texts(texts: list[str]) -> list[list[float]]:
                return embeddings.embed_documents(texts)

            dims: int = _resolve_embedding_dims(embeddings, self.store_model.dims)
            index = {"dims": dims, "embed": embed_texts}
            logger.debug(
                "Store embeddings configured",
                endpoint=embedding_model.name,
                dimensions=dims,
            )

        store: BaseStore = InMemoryStore(index=index)

        return store


class InMemoryCheckpointerManager(CheckpointManagerBase):
    def __init__(self, checkpointer_model: CheckpointerModel):
        self.checkpointer_model = checkpointer_model

    def checkpointer(self) -> BaseCheckpointSaver:
        return InMemorySaver()


class StoreManager:
    store_managers: dict[str, StoreManagerBase] = {}

    @classmethod
    def instance(cls, store_model: StoreModel) -> StoreManagerBase:
        store_manager: StoreManagerBase | None = None
        match store_model.storage_type:
            case StorageType.MEMORY:
                store_manager = cls.store_managers.get(store_model.name)
                if store_manager is None:
                    store_manager = InMemoryStoreManager(store_model)
                    cls.store_managers[store_model.name] = store_manager
            case StorageType.POSTGRES:
                cache_key = store_model.database.name
                store_manager = cls.store_managers.get(cache_key)
                if store_manager is None:
                    if store_model.database.is_lakebase:
                        from dao_ai.memory.databricks import LakebaseStoreManager

                        store_manager = LakebaseStoreManager(store_model)
                    else:
                        from dao_ai.memory.postgres import AsyncPostgresStoreManager

                        store_manager = AsyncPostgresStoreManager(store_model)
                    cls.store_managers[cache_key] = store_manager
            case StorageType.AGENT_MEMORY:
                # Key by the store config's own name, not the UC store full_name:
                # two StoreModels can target the same store under different scopes,
                # and scope is baked into the AgentMemoryStore at construction.
                cache_key = store_model.name
                store_manager = cls.store_managers.get(cache_key)
                if store_manager is None:
                    from dao_ai.memory.agent_memory import AgentMemoryStoreManager

                    store_manager = AgentMemoryStoreManager(store_model)
                    cls.store_managers[cache_key] = store_manager
            case _:
                raise ValueError(f"Unknown storage type: {store_model.storage_type}")

        return store_manager


class CheckpointManager:
    checkpoint_managers: dict[str, CheckpointManagerBase] = {}

    @classmethod
    def instance(cls, checkpointer_model: CheckpointerModel) -> CheckpointManagerBase:
        checkpointer_manager: CheckpointManagerBase | None = None
        match checkpointer_model.storage_type:
            case StorageType.MEMORY:
                checkpointer_manager = cls.checkpoint_managers.get(
                    checkpointer_model.name
                )
                if checkpointer_manager is None:
                    checkpointer_manager = InMemoryCheckpointerManager(
                        checkpointer_model
                    )
                    cls.checkpoint_managers[checkpointer_model.name] = (
                        checkpointer_manager
                    )
            case StorageType.POSTGRES:
                cache_key = checkpointer_model.database.name
                checkpointer_manager = cls.checkpoint_managers.get(cache_key)
                if checkpointer_manager is None:
                    if checkpointer_model.database.is_lakebase:
                        from dao_ai.memory.databricks import (
                            LakebaseCheckpointerManager,
                        )

                        checkpointer_manager = LakebaseCheckpointerManager(
                            checkpointer_model
                        )
                    else:
                        from dao_ai.memory.postgres import (
                            AsyncPostgresCheckpointerManager,
                        )

                        checkpointer_manager = AsyncPostgresCheckpointerManager(
                            checkpointer_model
                        )
                    cls.checkpoint_managers[cache_key] = checkpointer_manager
            case _:
                raise ValueError(
                    f"Unknown storage type: {checkpointer_model.storage_type}"
                )

        return checkpointer_manager
