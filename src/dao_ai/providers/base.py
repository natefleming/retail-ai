from abc import ABC, abstractmethod
from typing import Any, Sequence

from dao_ai.config import (
    AppModel,
    DatasetModel,
    SchemaModel,
    UnityCatalogFunctionSqlModel,
    VectorStoreModel,
    VolumeModel,
)


class ServiceProvider(ABC):
    @abstractmethod
    def create_token(self) -> str: ...

    @abstractmethod
    def get_secret(
        self, secret_scope: str, secret_key: str, default_value: str | None = None
    ) -> str: ...

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
    def create_agent(
        self,
        agent: AppModel,
        additional_pip_reqs: Sequence[str],
        additional_code_paths: Sequence[str],
    ) -> Any: ...

    @abstractmethod
    def deploy_agent(self, config: AppModel) -> Any: ...
