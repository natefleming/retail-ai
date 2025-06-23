from typing import Any

from databricks_langchain import (
    DatabricksEmbeddings,
)
from langchain_core.embeddings.embeddings import Embeddings
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from loguru import logger

from dao_ai.config import (
    CheckpointerModel,
    LLMModel,
    StorageType,
    StoreModel,
)
from dao_ai.memory.base import (
    CheckpointManagerBase,
    StoreManagerBase,
)


class InMemoryStoreManager(StoreManagerBase):
    def __init__(self, store_model: StoreModel):
        self.store_model = store_model

    def store(self) -> BaseStore:
        logger.debug("Creating InMemory store")

        index: dict[str, Any] = None

        embedding_model: LLMModel = self.store_model.embedding_model

        if embedding_model:
            embeddings: Embeddings = DatabricksEmbeddings(endpoint=embedding_model.name)

            def embed_texts(texts: list[str]) -> list[list[float]]:
                return embeddings.embed_documents(texts)

            dims: int = self.store_model.dims
            index = {"dims": dims, "embed": embed_texts}

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
        store_manager: StoreManagerBase = None
        match store_model.type:
            case StorageType.MEMORY:
                store_manager = cls.store_managers.get(store_model.name)
                if store_manager is None:
                    store_manager = InMemoryStoreManager(store_model)
                    cls.store_managers[store_model.name] = store_manager
            case StorageType.POSTGRES:
                from dao_ai.memory.postgres import PostgresStoreManager

                store_manager = cls.store_managers.get(store_model.database.name)
                if store_manager is None:
                    store_manager = PostgresStoreManager(store_model)
                    cls.store_managers[store_model.database.name] = store_manager
            case _:
                raise ValueError(f"Unknown store type: {store_model.type}")

        return store_manager


class CheckpointManager:
    checkpoint_managers: dict[str, CheckpointManagerBase] = {}

    @classmethod
    def instance(cls, checkpointer_model: CheckpointerModel) -> CheckpointManagerBase:
        checkpointer_manager: CheckpointManagerBase = None
        match checkpointer_model.type:
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
                from dao_ai.memory.postgres import PostgresCheckpointerManager

                checkpointer_manager = cls.checkpoint_managers.get(
                    checkpointer_model.database.name
                )
                if checkpointer_manager is None:
                    checkpointer_manager = PostgresCheckpointerManager(
                        checkpointer_model
                    )
                    cls.checkpoint_managers[checkpointer_model.database.name] = (
                        checkpointer_manager
                    )
            case _:
                raise ValueError(f"Unknown store type: {checkpointer_model.type}")

        return checkpointer_manager
