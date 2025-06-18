from abc import ABC, abstractmethod

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore


class CheckpointManagerBase(ABC):
    @abstractmethod
    def checkpointer(self) -> BaseCheckpointSaver: ...


class StoreManagerBase(ABC):
    @abstractmethod
    def store(self) -> BaseStore: ...
