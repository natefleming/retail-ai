from abc import ABC, abstractmethod
from typing import Any

from retail_ai.config import (
    AppModel,
    DatasetModel,
    SchemaModel,
    UnityCatalogFunctionSqlModel,
    VectorStoreModel,
    VolumeModel,
)


class ServiceProvider(ABC):
    @abstractmethod
    def create_catalog(self, schema: SchemaModel) -> Any: ...

    @abstractmethod
    def create_schema(self, schema: SchemaModel) -> Any: ...

    @abstractmethod
    def create_volume(self, schema: VolumeModel) -> Any: ...

    @abstractmethod
    def create_dataset(self, dataset: DatasetModel) -> Any: ...

    @abstractmethod
    def create_vector_store(self, vector_store: VectorStoreModel) -> Any: ...

    @abstractmethod
    def get_vector_index(self, vector_store: VectorStoreModel) -> Any: ...

    @abstractmethod
    def create_sql_function(
        self, unity_catalog_function: UnityCatalogFunctionSqlModel
    ) -> Any: ...

    @abstractmethod
    def create_agent(self, agent: AppModel) -> Any: ...

    @abstractmethod
    def deploy_agent(self, config: AppModel) -> Any: ...

    @abstractmethod
    def create_montior(self, config: AppModel) -> Any: ...
