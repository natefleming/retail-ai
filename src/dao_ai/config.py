import os
from abc import ABC, abstractmethod
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    Union,
)

from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import (
    CredentialsStrategy,
    ModelServingUserCredentials,
)
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex
from databricks_langchain import (
    ChatDatabricks,
    DatabricksFunctionClient,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from loguru import logger
from mlflow.models import ModelConfig
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


class HasValue(ABC):
    @abstractmethod
    def as_value(self) -> Any: ...


def value_of(value: HasValue | str | int | float | bool) -> Any:
    if isinstance(value, HasValue):
        value = value.as_value()
    return value


class HasFullName(ABC):
    @property
    @abstractmethod
    def full_name(self) -> str: ...


class IsDatabricksResource(ABC):
    on_behalf_of_user: Optional[bool] = False

    @abstractmethod
    def as_resource(self) -> DatabricksResource: ...

    @property
    @abstractmethod
    def api_scopes(self) -> Sequence[str]: ...

    @property
    def workspace_client(self) -> WorkspaceClient:
        credentials_strategy: CredentialsStrategy = None
        if self.on_behalf_of_user:
            credentials_strategy = ModelServingUserCredentials()
        return WorkspaceClient(credentials_strategy=credentials_strategy)


class EnvironmentVariableModel(BaseModel, HasValue):
    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )
    env: str
    default_value: Optional[Any] = None

    def as_value(self) -> Any:
        logger.debug(f"Fetching environment variable: {self.env}")
        value: Any = os.environ.get(self.env, self.default_value)
        return value

    def __str__(self) -> str:
        return self.env


class SecretVariableModel(BaseModel, HasValue):
    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )
    scope: str
    secret: str
    default_value: Optional[Any] = None

    def as_value(self) -> Any:
        logger.debug(f"Fetching secret: {self.scope}/{self.secret}")
        from dao_ai.providers.databricks import DatabricksProvider

        provider: DatabricksProvider = DatabricksProvider()
        value: Any = provider.get_secret(self.scope, self.secret, self.default_value)
        return value

    def __str__(self) -> str:
        return "{{secrets/" + f"{self.scope}/{self.secret}" + "}}"


class PrimitiveVariableModel(BaseModel, HasValue):
    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )

    value: Union[str, int, float, bool]

    def as_value(self) -> Any:
        return self.value

    @field_serializer("value")
    def serialize_value(self, value: Any) -> str:
        return str(value)

    @model_validator(mode="after")
    def validate_value(self) -> "PrimitiveVariableModel":
        if not isinstance(self.as_value(), (str, int, float, bool)):
            raise ValueError("Value must be a primitive type (str, int, float, bool)")
        return self


class CompositeVariableModel(BaseModel, HasValue):
    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )
    default_value: Optional[Any] = None
    variables: list[
        EnvironmentVariableModel
        | SecretVariableModel
        | PrimitiveVariableModel
        | str
        | int
        | float
        | bool
    ] = Field(default_factory=list)

    def as_value(self) -> Any:
        logger.debug("Evaluating composite variable...")
        value: Any = None
        for v in self.variables:
            value = value_of(v)
            if value is not None:
                return value
        return self.default_value


type AnyVariable = (
    CompositeVariableModel
    | EnvironmentVariableModel
    | SecretVariableModel
    | PrimitiveVariableModel
    | str
    | int
    | float
    | bool
)


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
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    principals: list[str] = Field(default_factory=list)
    privileges: list[Privilege]


class SchemaModel(BaseModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    catalog_name: str
    schema_name: str
    permissions: Optional[list[PermissionModel]] = Field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}"

    def create(self, w: WorkspaceClient | None = None) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w)
        provider.create_schema(self)


class TableModel(BaseModel, HasFullName, IsDatabricksResource):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name

    @property
    def api_scopes(self) -> Sequence[str]:
        return []

    def as_resource(self) -> DatabricksResource:
        return DatabricksTable(
            table_name=self.full_name, on_behalf_of_user=self.on_behalf_of_user
        )


class LLMModel(BaseModel, IsDatabricksResource):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 8192
    fallbacks: Optional[list[Union[str, "LLMModel"]]] = Field(default_factory=list)

    @property
    def api_scopes(self) -> Sequence[str]:
        return [
            "serving.serving-endpoints",
        ]

    def as_resource(self) -> DatabricksResource:
        return DatabricksServingEndpoint(
            endpoint_name=self.name, on_behalf_of_user=self.on_behalf_of_user
        )

    def as_chat_model(self) -> LanguageModelLike:
        chat_client: LanguageModelLike = ChatDatabricks(
            model=self.name, temperature=self.temperature, max_tokens=self.max_tokens
        )
        fallbacks: Sequence[LanguageModelLike] = []
        for fallback in self.fallbacks:
            fallback: str | LLMModel
            if isinstance(fallback, str):
                fallback = LLMModel(
                    name=fallback,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            if fallback.name == self.name:
                continue
            fallback_model: LanguageModelLike = fallback.as_chat_model()
            fallbacks.append(fallback_model)

        if fallbacks:
            chat_client = chat_client.with_fallbacks(fallbacks)

        return chat_client


class VectorSearchEndpointType(str, Enum):
    STANDARD = "STANDARD"
    OPTIMIZED_STORAGE = "OPTIMIZED_STORAGE"


class VectorSearchEndpoint(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    type: VectorSearchEndpointType


class IndexModel(BaseModel, HasFullName, IsDatabricksResource):
    model_config = ConfigDict()
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def api_scopes(self) -> Sequence[str]:
        return [
            "vectorsearch.vector-search-indexes",
        ]

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
    model_config = ConfigDict()
    embedding_model: LLMModel
    endpoint: VectorSearchEndpoint
    index: IndexModel
    source_table: TableModel
    primary_key: str
    doc_uri: Optional[str] = None
    embedding_source_column: str
    columns: list[str]

    @property
    def api_scopes(self) -> Sequence[str]:
        return [
            "vectorsearch.vector-search-endpoints",
        ] + self.index.api_scopes

    def as_resource(self) -> DatabricksResource:
        return self.index.as_resource()

    def as_index(self, vsc: VectorSearchClient | None = None) -> VectorSearchIndex:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(vsc=vsc)
        index: VectorSearchIndex = provider.get_vector_index(self)
        return index

    def create(self, vsc: VectorSearchClient | None = None) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(vsc=vsc)
        provider.create_vector_store(self)


class GenieRoomModel(BaseModel, IsDatabricksResource):
    model_config = ConfigDict()
    name: str
    description: Optional[str] = None
    space_id: str

    @property
    def api_scopes(self) -> Sequence[str]:
        return [
            "dashboards.genie",
        ]

    def as_resource(self) -> DatabricksResource:
        return DatabricksGenieSpace(
            genie_space_id=self.space_id, on_behalf_of_user=self.on_behalf_of_user
        )


class VolumeModel(BaseModel, HasFullName):
    model_config = ConfigDict()
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name

    def create(self, w: WorkspaceClient | None = None) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w)
        provider.create_volume(self)


class FunctionModel(BaseModel, HasFullName, IsDatabricksResource):
    model_config = ConfigDict()
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

    @property
    def api_scopes(self) -> Sequence[str]:
        return ["sql.statement-execution"]


class ConnectionModel(BaseModel, HasFullName, IsDatabricksResource):
    model_config = ConfigDict()
    name: str

    @property
    def full_name(self) -> str:
        return self.name

    @property
    def api_scopes(self) -> Sequence[str]:
        return [
            "catalog.connections",
        ]

    def as_resource(self) -> DatabricksResource:
        return DatabricksUCConnection(
            connection_name=self.name, on_behalf_of_user=self.on_behalf_of_user
        )


class WarehouseModel(BaseModel, IsDatabricksResource):
    model_config = ConfigDict()
    name: str
    description: Optional[str] = None
    warehouse_id: str

    @property
    def api_scopes(self) -> Sequence[str]:
        return [
            "sql.warehouses",
        ]

    def as_resource(self) -> DatabricksResource:
        return DatabricksSQLWarehouse(
            warehouse_id=self.warehouse_id, on_behalf_of_user=self.on_behalf_of_user
        )


class DatabaseModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    description: Optional[str] = None
    host: Optional[AnyVariable]
    database: Optional[AnyVariable] = "databricks_postgres"
    port: Optional[AnyVariable] = 5432
    connection_kwargs: Optional[dict[str, Any]] = Field(default_factory=dict)
    max_pool_size: Optional[int] = 10
    timeout_seconds: Optional[int] = 5
    user: Optional[AnyVariable] = None
    password: Optional[AnyVariable] = None
    client_id: Optional[AnyVariable] = None
    client_secret: Optional[AnyVariable] = None
    workspace_host: Optional[AnyVariable] = None

    @model_validator(mode="after")
    def validate_auth_methods(self):
        oauth_fields: Sequence[Any] = [
            self.workspace_host,
            self.client_id,
            self.client_secret,
        ]
        has_oauth: bool = all(field is not None for field in oauth_fields)

        pat_fields: Sequence[Any] = [self.user, self.password]
        has_user_auth: bool = all(field is not None for field in pat_fields)

        if has_oauth and has_user_auth:
            raise ValueError(
                "Cannot use both OAuth and user authentication methods. "
                "Please provide either OAuth credentials or user credentials."
            )

        if not has_oauth and not has_user_auth:
            raise ValueError(
                "At least one authentication method must be provided: "
                "either OAuth credentials (workspace_host, client_id, client_secret) "
                "or user credentials (user, password)."
            )

        return self

    @property
    def connection_url(self) -> str:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        username: str | None = None

        if self.client_id and self.client_secret and self.workspace_host:
            username = value_of(self.client_id)
        else:
            username = value_of(self.user)

        host: str = value_of(self.host)
        port: int = value_of(self.port)
        database: str = value_of(self.database)

        provider: ServiceProvider = DatabricksProvider(
            client_id=value_of(self.client_id),
            client_secret=value_of(self.client_secret),
            workspace_host=value_of(self.workspace_host),
            pat=value_of(self.password),
        )
        token: str = provider.create_token()

        return (
            f"postgresql://{username}:{token}@{host}:{port}/{database}?sslmode=require"
        )


class SearchParametersModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    num_results: Optional[int] = 10
    filters: Optional[dict[str, Any]] = Field(default_factory=dict)
    query_type: Optional[str] = "ANN"


class RetrieverModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    vector_store: VectorStoreModel
    columns: list[str]
    search_parameters: SearchParametersModel


class FunctionType(str, Enum):
    PYTHON = "python"
    FACTORY = "factory"
    UNITY_CATALOG = "unity_catalog"
    MCP = "mcp"


class HumanInTheLoopModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    review_prompt: str
    interupt_config: dict[str, Any] = Field(default_factory=dict)


class BaseFunctionModel(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True,
        discriminator="type",
    )
    type: FunctionType
    name: str
    human_in_the_loop: Optional[HumanInTheLoopModel] = None

    @field_serializer("type")
    def serialize_type(self, value) -> str:
        # Handle both enum objects and already-converted strings
        if isinstance(value, FunctionType):
            return value.value
        return str(value)


class PythonFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.PYTHON] = FunctionType.PYTHON

    @property
    def full_name(self) -> str:
        return self.name

    def as_tool(self, **kwargs: Any) -> Callable[..., Any]:
        from dao_ai.tools import create_python_tool

        return create_python_tool(self, **kwargs)


class FactoryFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    args: Optional[dict[str, Any]] = Field(default_factory=dict)
    type: Literal[FunctionType.FACTORY] = FunctionType.FACTORY

    @property
    def full_name(self) -> str:
        return self.name

    def as_tool(self, **kwargs: Any) -> Callable[..., Any]:
        from dao_ai.tools import create_factory_tool

        return create_factory_tool(self, **kwargs)


class TransportType(str, Enum):
    STREAMABLE_HTTP = "streamable_http"
    STDIO = "stdio"


class McpFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.MCP] = FunctionType.MCP

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

    def as_tool(self, **kwargs: Any) -> BaseTool:
        from dao_ai.tools import create_mcp_tool

        return create_mcp_tool(self)


class UnityCatalogFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    type: Literal[FunctionType.UNITY_CATALOG] = FunctionType.UNITY_CATALOG

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name

    def as_tool(self, **kwargs: Any) -> BaseTool:
        from dao_ai.tools import create_uc_tool

        return create_uc_tool(self)


AnyTool: TypeAlias = (
    Union[
        PythonFunctionModel,
        FactoryFunctionModel,
        UnityCatalogFunctionModel,
        McpFunctionModel,
    ]
    | str
)


class ToolModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    function: AnyTool


class GuardrailModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    model: LLMModel
    prompt: str
    num_retries: Optional[int] = 3


class StorageType(str, Enum):
    POSTGRES = "postgres"
    MEMORY = "memory"


class CheckpointerModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    type: Optional[StorageType] = StorageType.MEMORY
    database: Optional[DatabaseModel] = None

    @model_validator(mode="after")
    def validate_postgres_requires_database(self):
        if self.type == StorageType.POSTGRES and not self.database:
            raise ValueError("Database must be provided when storage type is POSTGRES")
        return self

    def as_checkpointer(self) -> BaseCheckpointSaver:
        from dao_ai.memory import CheckpointManager

        checkpointer: BaseCheckpointSaver = CheckpointManager.instance(
            self
        ).checkpointer()

        return checkpointer


class StoreModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    embedding_model: Optional[LLMModel] = None
    type: Optional[StorageType] = StorageType.MEMORY
    dims: Optional[int] = 1536
    database: Optional[DatabaseModel] = None
    namespace: Optional[str] = None

    @model_validator(mode="after")
    def validate_postgres_requires_database(self):
        if self.type == StorageType.POSTGRES and not self.database:
            raise ValueError("Database must be provided when storage type is POSTGRES")
        return self

    def as_store(self) -> BaseStore:
        from dao_ai.memory import StoreManager

        store: BaseStore = StoreManager.instance(self).store()
        return store


class MemoryModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    checkpointer: Optional[CheckpointerModel] = None
    store: Optional[StoreModel] = None


FunctionHook: TypeAlias = PythonFunctionModel | FactoryFunctionModel | str


class AgentModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    description: Optional[str] = None
    model: LLMModel
    tools: list[ToolModel] = Field(default_factory=list)
    guardrails: list[GuardrailModel] = Field(default_factory=list)
    memory: Optional[MemoryModel] = None
    prompt: str
    handoff_prompt: Optional[str] = None
    create_agent_hook: Optional[FunctionHook] = None
    pre_agent_hook: Optional[FunctionHook] = None
    post_agent_hook: Optional[FunctionHook] = None


class SupervisorModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel
    memory: Optional[MemoryModel] = None


class SwarmModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel
    default_agent: Optional[AgentModel | str] = None
    handoffs: Optional[dict[str, Optional[list[AgentModel | str]]]] = Field(
        default_factory=dict
    )
    memory: Optional[MemoryModel] = None


class OrchestrationModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
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
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
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
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
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


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    role: MessageRole
    content: str


class ChatPayload(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    messages: list[Message]
    custom_inputs: dict


class SummarizationModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel
    retained_message_count: Optional[int] = None
    max_tokens: Optional[int] = None

    @model_validator(mode="after")
    def validate_mutually_exclusive(self):
        if self.retained_message_count and self.max_tokens:
            raise ValueError(
                "Cannot specify both retained_message_count and max_tokens. "
                "Please provide only one of these parameters."
            )
        if not self.retained_message_count and not self.max_tokens:
            self.retained_message_count = 5  # Default value if none is provided

        return self


class AppModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    description: Optional[str] = None
    log_level: Optional[LogLevel] = "WARNING"
    registered_model: RegisteredModelModel
    endpoint_name: Optional[str] = None
    tags: Optional[dict[str, Any]] = Field(default_factory=dict)
    scale_to_zero: Optional[bool] = True
    environment_vars: Optional[dict[str, Any]] = Field(default_factory=dict)
    budget_policy_id: Optional[str] = None
    workload_size: Optional[WorkloadSize] = "Small"
    permissions: Optional[list[AppPermissionModel]] = Field(default_factory=list)
    agents: list[AgentModel] = Field(default_factory=list)
    orchestration: OrchestrationModel
    alias: Optional[str] = None
    message_hooks: Optional[FunctionHook | list[FunctionHook]] = Field(
        default_factory=list
    )
    input_example: Optional[ChatPayload] = None
    summarization: Optional[SummarizationModel] = None

    @model_validator(mode="after")
    def validate_agents_not_empty(self):
        if not self.agents:
            raise ValueError("agents must contain at least one item")
        return self

    @model_validator(mode="after")
    def set_default_endpoint_name(self):
        if self.endpoint_name is None:
            self.endpoint_name = self.name
        return self

    @model_validator(mode="after")
    def set_default_agent(self):
        default_agent_name = self.agents[0].name

        if self.orchestration.swarm and not self.orchestration.swarm.default_agent:
            self.orchestration.swarm.default_agent = default_agent_name

        return self


class EvaluationModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
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


class VolumePathModel(BaseModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    volume: VolumeModel
    path: str

    @property
    def full_name(self) -> str:
        if self.volume.schema_model:
            catalog_name: str = self.volume.schema_model.catalog_name
            schema_name: str = self.volume.schema_model.schema_name
            volume_name: str = self.volume.name
            return f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{self.path}"
        return f"/Volumes/{volume_name}/{self.path}"


class DatasetModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    table: Optional[TableModel] = None
    ddl: Optional[str | VolumeModel] = None
    data: Optional[str | VolumePathModel] = None
    format: Optional[DatasetFormat] = None
    read_options: Optional[dict[str, Any]] = Field(default_factory=dict)
    table_schema: Optional[str] = None

    def create(self, w: WorkspaceClient | None = None) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w)
        provider.create_dataset(self)


class UnityCatalogFunctionSqlTestModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    parameters: Optional[dict[str, Any]] = Field(default_factory=dict)


class UnityCatalogFunctionSqlModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    function: UnityCatalogFunctionModel
    ddl: str
    test: Optional[UnityCatalogFunctionSqlTestModel] = None

    def create(
        self,
        w: WorkspaceClient | None = None,
        dfs: DatabricksFunctionClient | None = None,
    ) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w, dfs=dfs)
        provider.create_sql_function(self)


class ResourcesModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
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
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schemas: dict[str, SchemaModel]
    resources: ResourcesModel
    retrievers: dict[str, RetrieverModel] = Field(default_factory=dict)
    tools: dict[str, ToolModel] = Field(default_factory=dict)
    guardrails: dict[str, GuardrailModel] = Field(default_factory=dict)
    memory: Optional[MemoryModel] = None
    agents: dict[str, AgentModel] = Field(default_factory=dict)
    app: AppModel
    evaluation: Optional[EvaluationModel] = None
    datasets: Optional[list[DatasetModel]] = Field(default_factory=list)
    unity_catalog_functions: Optional[list[UnityCatalogFunctionSqlModel]] = Field(
        default_factory=list
    )
    providers: Optional[dict[type | str, Any]] = None

    @classmethod
    def from_file(cls, path: PathLike) -> "AppConfig":
        path = Path(path).as_posix()
        logger.debug(f"Loading config from {path}")
        model_config: ModelConfig = ModelConfig(development_config=path)
        config: AppConfig = AppConfig(**model_config.to_dict())
        return config

    def display_graph(self) -> None:
        from dao_ai.graph import create_dao_ai_graph
        from dao_ai.models import display_graph

        display_graph(create_dao_ai_graph(config=self))

    def save_image(self, path: PathLike) -> None:
        from dao_ai.graph import create_dao_ai_graph
        from dao_ai.models import save_image

        logger.info(f"Saving image to {path}")
        save_image(create_dao_ai_graph(config=self), path=path)

    def create_agent(
        self,
        w: WorkspaceClient | None = None,
        *,
        additional_pip_reqs: Sequence[str] = [],
        additional_code_paths: Sequence[str] = [],
    ) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w)
        provider.create_agent(
            self,
            additional_pip_reqs=additional_pip_reqs,
            additional_code_paths=additional_code_paths,
        )

    def deploy_agent(
        self,
        w: WorkspaceClient | None = None,
    ) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w)
        provider.deploy_agent(self)

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
    ) -> Sequence[ToolModel]:
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
        self, predicate: Callable[[GuardrailModel], bool] | None = None
    ) -> Sequence[GuardrailModel]:
        """
        Find agents in the configuration that match a given predicate.

        Args:
            predicate: A callable that takes an AgentModel and returns True if it matches.

        Returns:
            A list of AgentModel instances that match the predicate.
        """
        if predicate is None:

            def _null_predicate(guardrails: GuardrailModel) -> bool:
                return True

            predicate = _null_predicate

        return [
            guardrail for guardrail in self.guardrails.values() if predicate(guardrail)
        ]
