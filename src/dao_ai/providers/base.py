from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

from dao_ai.config import (
    AppModel,
    DatasetModel,
    SchemaModel,
    ServingMode,
    UnityCatalogFunctionSqlModel,
    VectorStoreModel,
    VolumeModel,
)

if TYPE_CHECKING:
    from dao_ai.config import AppConfig


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
    def deploy_model_serving_agent(self, config: "AppConfig") -> Any:
        """Deploy agent to Databricks Model Serving endpoint."""
        ...

    @abstractmethod
    def deploy_apps_agent(
        self,
        config: "AppConfig",
        *,
        as_mcp: bool = False,
        development: bool | None = None,
    ) -> Any:
        """Deploy agent as a Databricks App (chat UI, or MCP server when
        ``as_mcp``)."""
        ...

    @abstractmethod
    def deploy_agent(
        self,
        config: "AppConfig",
        mode: ServingMode = ServingMode.MODEL_SERVING,
        development: bool | None = None,
        as_mcp: bool = False,
        with_connection: bool = False,
    ) -> Any:
        """
        Deploy agent using the specified serving platform.

        Args:
            config: The AppConfig containing deployment configuration
            mode: The serving platform (MODEL_SERVING or APPS)
            as_mcp: Serve over MCP instead of the chat UI (requires APPS)
            with_connection: Register a UC MCP connection after deploy
                (requires APPS + as_mcp)
        """
        ...
