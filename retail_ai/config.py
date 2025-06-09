import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models import LanguageModelLike
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from loguru import logger
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksResource,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksUCConnection,
    DatabricksVectorSearchIndex,
)
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


class HasFullName(ABC):
    @property
    @abstractmethod
    def full_name(self) -> str:
        pass


class IsDatabricksResource(ABC):
    on_behalf_of_user: Optional[bool] = False

    @abstractmethod
    def as_resource(self) -> DatabricksResource:
        pass


class Privilege(str, Enum):
    ALL_PRIVILEGES = "ALL_PRIVILEGES"
    USE_CATALOG = "USE_CATALOG"
    USE_SCHEMA = "USE_SCHEMA"
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    MODIFY = "MODIFY"
    CREATE = "CREATE"
    USAGE = "USAGE"
    CREATE_SCHEMA = "CREATE_SCHEMA"
    CREATE_TABLE = "CREATE_TABLE"
    CREATE_VIEW = "CREATE_VIEW"
    CREATE_FUNCTION = "CREATE_FUNCTION"
    CREATE_EXTERNAL_LOCATION = "CREATE_EXTERNAL_LOCATION"
    CREATE_STORAGE_CREDENTIAL = "CREATE_STORAGE_CREDENTIAL"
    CREATE_MATERIALIZED_VIEW = "CREATE_MATERIALIZED_VIEW"
    CREATE_TEMPORARY_FUNCTION = "CREATE_TEMPORARY_FUNCTION"
    EXECUTE = "EXECUTE"
    READ_FILES = "READ_FILES"
    WRITE_FILES = "WRITE_FILES"


class PermissionModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    principals: list[str] = Field(default_factory=list)
    privileges: list[Privilege]


class SchemaModel(BaseModel, HasFullName):
    catalog_name: str
    schema_name: str
    permissions: list[PermissionModel]

    @property
    def full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}"


class TableModel(BaseModel, HasFullName, IsDatabricksResource):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name

    def as_resource(self) -> DatabricksResource:
        return DatabricksTable(
            table_name=self.full_name, on_behalf_of_user=self.on_behalf_of_user
        )


class LLMModel(BaseModel, IsDatabricksResource):
    name: str
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 8192
    fallbacks: Optional[list[str]] = Field(default_factory=list)

    def as_resource(self) -> DatabricksResource:
        return DatabricksServingEndpoint(
            endpoint_name=self.name, on_behalf_of_user=self.on_behalf_of_user
        )

    def as_chat_model(self) -> LanguageModelLike:
        chat_client: LanguageModelLike = ChatDatabricks(
            model=self.name, temperature=self.temperature
        )
        fallbacks: Sequence[LanguageModelLike] = [
            ChatDatabricks(model=f, temperature=self.temperature)
            for f in self.fallbacks
            if f != self.name
        ]
        if fallbacks:
            chat_client = chat_client.with_fallbacks(fallbacks)
        return chat_client


class VectorSearchEndpointType(str, Enum):
    STANDARD = "STANDARD"
    OPTIMIZED_STORAGE = "OPTIMIZED_STORAGE"


class VectorSearchEndpoint(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    name: str
    type: VectorSearchEndpointType


class IndexModel(BaseModel, HasFullName, IsDatabricksResource):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name

    def as_resource(self) -> DatabricksResource:
        return DatabricksVectorSearchIndex(
            index_name=self.full_name, on_behalf_of_user=self.on_behalf_of_user
        )


class VectorStoreModel(BaseModel, IsDatabricksResource):
    embedding_model: LLMModel
    endpoint: VectorSearchEndpoint
    index: IndexModel
    source_table: TableModel
    primary_key: str
    doc_uri: Optional[str] = None
    embedding_source_column: str
    columns: list[str]

    def as_resource(self) -> DatabricksResource:
        return self.index.as_resource()


class GenieRoomModel(BaseModel, IsDatabricksResource):
    name: str
    description: str
    space_id: str

    def as_resource(self) -> DatabricksResource:
        return DatabricksGenieSpace(
            genie_space_id=self.space_id, on_behalf_of_user=self.on_behalf_of_user
        )


class VolumeModel(BaseModel, HasFullName):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class FunctionModel(BaseModel, HasFullName, IsDatabricksResource):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name

    def as_resource(self) -> DatabricksResource:
        return DatabricksFunction(
            function_name=self.full_name, on_behalf_of_user=self.on_behalf_of_user
        )


class ConnectionModel(BaseModel, HasFullName, IsDatabricksResource):
    name: str

    @property
    def full_name(self) -> str:
        return self.name

    def as_resource(self) -> DatabricksResource:
        return DatabricksUCConnection(
            connection_name=self.name, on_behalf_of_user=self.on_behalf_of_user
        )


class WarehouseModel(BaseModel, IsDatabricksResource):
    name: str
    description: str
    warehouse_id: str

    def as_resource(self) -> DatabricksResource:
        return DatabricksSQLWarehouse(
            warehouse_id=self.warehouse_id, on_behalf_of_user=self.on_behalf_of_user
        )


class DatabaseModel(BaseModel):
    name: str
    connection_url: Optional[str] = None
    connection_kwargs: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, context) -> None:
        if not self.connection_url:
            if "PGCONNECTION_STRING" in os.environ:
                self.connection_url = os.getenv("PGCONNECTION_STRING")
            else:
                pg_host: str = os.getenv("PGHOST", "localhost")
                pg_port: str = os.getenv("PGPORT", "5432")
                pg_database: str = os.getenv("PGDATABASE", "postgres")
                pg_user: str = os.getenv("PGUSER", "postgres")
                pg_password: str = os.getenv("PGPASSWORD", "")

                self.connection_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}?sslmode=require"


class SearchParametersModel(BaseModel):
    num_results: Optional[int] = 10
    filters: Optional[dict[str, Any]] = Field(default_factory=dict)
    query_type: Optional[str] = "ANN"


class RetrieverModel(BaseModel):
    vector_store: VectorStoreModel
    columns: list[str]
    search_parameters: SearchParametersModel


class FunctionType(str, Enum):
    PYTHON = "python"
    FACTORY = "factory"
    UNITY_CATALOG = "unity_catalog"
    MCP = "mcp"


class BaseFunctionModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    type: FunctionType
    name: str

    @field_serializer("type")
    def serialize_type(self, value) -> str:
        # Handle both enum objects and already-converted strings
        if isinstance(value, FunctionType):
            return value.value
        return str(value)


class PythonFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True)
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    type: FunctionType = FunctionType.PYTHON

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class FactoryFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True)
    args: dict[str, Any] = Field(default_factory=dict)
    type: FunctionType = FunctionType.FACTORY

    @property
    def full_name(self) -> str:
        return self.name


class TransportType(str, Enum):
    STREAMABLE_HTTP = "streamable_http"
    STDIO = "stdio"


class McpFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True)
    type: FunctionType = FunctionType.MCP

    transport: TransportType
    command: Optional[str] = "python"
    url: Optional[str] = None
    args: list[str] = Field(default_factory=list)

    @property
    def full_name(self) -> str:
        return self.name

    @model_validator(mode="after")
    def validate_mutually_exclusive(self):
        if self.transport == TransportType.STREAMABLE_HTTP and not self.url:
            raise ValueError("url must be provided for STREAMABLE_HTTP transport")
        if self.transport == TransportType.STDIO and not self.command:
            raise ValueError("command must not be provided for STDIO transport")
        if self.transport == TransportType.STDIO and not self.args:
            raise ValueError("args must not be provided for STDIO transport")
        return self


class UnityCatalogFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True)
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    type: FunctionType = FunctionType.UNITY_CATALOG

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class ToolModel(BaseModel):
    name: str
    function: (
        PythonFunctionModel
        | FactoryFunctionModel
        | UnityCatalogFunctionModel
        | McpFunctionModel
    )


class GuardrailsModel(BaseModel):
    name: str
    model: LLMModel
    prompt: str


class StorageType(str, Enum):
    POSTGRES = "postgres"
    MEMORY = "memory"


class CheckpointerModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    name: str
    type: StorageType
    database: DatabaseModel


class StoreModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    name: str
    embedding_model: Optional[LLMModel] = None
    type: Optional[StorageType] = StorageType.MEMORY
    dims: Optional[int] = 1536
    database: Optional[DatabaseModel] = None

    def as_store(self) -> BaseStore:
        store: BaseStore = (
            self._as_postgres_store()
            if self.type == StorageType.POSTGRES
            else self._as_in_memory_store()
        )

        return store

    def _as_in_memory_store(self) -> BaseStore:
        logger.debug("Creating InMemory store")
        embeddings: Embeddings = DatabricksEmbeddings(
            endpoint=self.embedding_model.name
        )

        def embed_texts(texts: list[str]) -> list[list[float]]:
            return embeddings.embed_documents(texts)

        store: BaseStore = InMemoryStore(
            index={"dims": self.dims, "embed": embed_texts}
        )

        return store

    def _as_postgres_store(self) -> BaseStore:
        logger.debug("Creating Postgres store")
        if not self.database:
            raise ValueError("Database must be provided for Postgres store")

        index: dict[str, Any] = {}
        if self.embedding_model:
            embeddings: Embeddings = DatabricksEmbeddings(
                endpoint=self.embedding_model.name
            )

            def embed_texts(texts: list[str]) -> list[list[float]]:
                return embeddings.embed_documents(texts)

            index = {"dims": self.dims, "embed": embed_texts}

        logger.debug(f"connection_url: {self.database.connection_url}")
        with PostgresStore.from_conn_string(
            conn_string=self.database.connection_url, index=index
        ) as store:
            store.setup()

        return store


class MemoryModel(BaseModel):
    checkpointer: Optional[CheckpointerModel] = None
    store: Optional[StoreModel] = None


class AgentModel(BaseModel):
    name: str
    description: str
    model: LLMModel
    tools: list[ToolModel] = Field(default_factory=list)
    guardrails: list[GuardrailsModel] = Field(default_factory=list)
    memory: Optional[MemoryModel] = None
    prompt: str
    handoff_prompt: Optional[str] = None


class SupervisorModel(BaseModel):
    model: LLMModel
    default_agent: AgentModel | str


class SwarmModel(BaseModel):
    model: LLMModel
    default_agent: AgentModel | str
    handoffs: Optional[dict[str, Optional[list[AgentModel | str]]]] = Field(default_factory=dict)


class OrchestrationModel(BaseModel):
    supervisor: Optional[SupervisorModel] = None
    swarm: Optional[SwarmModel] = None

    @model_validator(mode="after")
    def validate_mutually_exclusive(self):
        if self.supervisor is not None and self.swarm is not None:
            raise ValueError("Cannot specify both supervisor and swarm")
        if self.supervisor is None and self.swarm is None:
            raise ValueError("Must specify either supervisor or swarm")
        return self


class RegisteredModelModel(BaseModel, HasFullName):
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class Entitlement(str, Enum):
    CAN_MANAGE = "CAN_MANAGE"
    CAN_QUERY = "CAN_QUERY"
    CAN_VIEW = "CAN_VIEW"
    CAN_REVIEW = "CAN_REVIEW"
    NO_PERMISSIONS = "NO_PERMISSIONS"


class AppPermissionModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    principals: list[str] = Field(default_factory=list)
    entitlements: list[Entitlement]


class LogLevel(str, Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class WorkloadSize(str, Enum):
    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"


class AppModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    log_level: LogLevel
    registered_model: RegisteredModelModel
    endpoint_name: str
    tags: Optional[dict[str, Any]] = Field(default_factory=dict)
    scale_to_zero: Optional[bool] = True
    environment_vars: Optional[dict[str, Any]] = Field(default_factory=dict)
    budget_policy_id: Optional[str] = None
    workload_size: Optional[WorkloadSize] = "Small"
    permissions: list[AppPermissionModel]
    agents: list[AgentModel] = Field(default_factory=list)
    orchestration: OrchestrationModel


class EvaluationModel(BaseModel):
    model: LLMModel
    table: TableModel
    num_evals: int


class DatasetFormat(str, Enum):
    CSV = "csv"
    DELTA = "delta"
    JSON = "json"
    PARQUET = "parquet"
    ORC = "orc"
    SQL = "sql"


class DatasetModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    table: TableModel
    ddl: str
    data: str
    format: DatasetFormat
    read_options: Optional[dict[str, Any]] = Field(default_factory=dict)


class UnityCatalogFunctionSqlTestModel(BaseModel):
    parameters: Optional[dict[str, Any]] = Field(default_factory=dict)


class UnityCatalogFunctionSqlModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    function: UnityCatalogFunctionModel
    ddl: str
    test: Optional[UnityCatalogFunctionSqlTestModel] = None


class ResourcesModel(BaseModel):
    llms: dict[str, LLMModel] = Field(default_factory=dict)
    vector_stores: dict[str, VectorStoreModel] = Field(default_factory=dict)
    genie_rooms: dict[str, GenieRoomModel] = Field(default_factory=dict)
    tables: dict[str, TableModel] = Field(default_factory=dict)
    volumes: dict[str, VolumeModel] = Field(default_factory=dict)
    functions: dict[str, FunctionModel] = Field(default_factory=dict)
    warehouses: dict[str, WarehouseModel] = Field(default_factory=dict)
    databases: dict[str, DatabaseModel] = Field(default_factory=dict)
    connections: dict[str, ConnectionModel] = Field(default_factory=dict)


class AppConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    schemas: dict[str, SchemaModel]
    resources: ResourcesModel
    retrievers: dict[str, RetrieverModel] = Field(default_factory=dict)
    tools: dict[str, ToolModel] = Field(default_factory=dict)
    guardrails: dict[str, GuardrailsModel] = Field(default_factory=dict)
    memory: Optional[MemoryModel] = None
    agents: dict[str, AgentModel] = Field(default_factory=dict)
    app: AppModel
    evaluation: Optional[EvaluationModel] = None
    datasets: Optional[list[DatasetModel]] = Field(default_factory=list)
    unity_catalog_functions: Optional[list[UnityCatalogFunctionSqlModel]] = Field(
        default_factory=list
    )

    def find_agents(
        self, predicate: Callable[[AgentModel], bool] | None = None
    ) -> Sequence[AgentModel]:
        """
        Find agents in the configuration that match a given predicate.

        Args:
            predicate: A callable that takes an AgentModel and returns True if it matches.

        Returns:
            A list of AgentModel instances that match the predicate.
        """
        if predicate is None:

            def _null_predicate(agent: AgentModel) -> bool:
                return True

            predicate = _null_predicate

        return [agent for agent in self.agents.values() if predicate(agent)]

    def find_tools(
        self, predicate: Callable[[ToolModel], bool] | None = None
    ) -> Sequence[AgentModel]:
        """
        Find agents in the configuration that match a given predicate.

        Args:
            predicate: A callable that takes an AgentModel and returns True if it matches.

        Returns:
            A list of AgentModel instances that match the predicate.
        """
        if predicate is None:

            def _null_predicate(tool: ToolModel) -> bool:
                return True

            predicate = _null_predicate

        return [tool for tool in self.tools.values() if predicate(tool)]

    def find_guardrails(
        self, predicate: Callable[[GuardrailsModel], bool] | None = None
    ) -> Sequence[AgentModel]:
        """
        Find agents in the configuration that match a given predicate.

        Args:
            predicate: A callable that takes an AgentModel and returns True if it matches.

        Returns:
            A list of AgentModel instances that match the predicate.
        """
        if predicate is None:

            def _null_predicate(guardrails: GuardrailsModel) -> bool:
                return True

            predicate = _null_predicate

        return [
            guardrail for guardrail in self.guardrails.values() if predicate(guardrail)
        ]
