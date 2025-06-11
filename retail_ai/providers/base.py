from abc import ABC, abstractmethod
from typing import Any

from retail_ai.config import SchemaModel, VolumeModel


class ServiceProvider(ABC):
    @abstractmethod
    def create_catalog(self, schema: SchemaModel) -> Any: ...

    @abstractmethod
    def create_schema(self, schema: SchemaModel) -> Any: ...

    @abstractmethod
    def create_volume(self, schema: VolumeModel) -> Any: ...
