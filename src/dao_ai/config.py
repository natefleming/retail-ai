import atexit
import importlib
import os
import sys
from abc import ABC, abstractmethod
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
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
from databricks.sdk.service.catalog import FunctionInfo, TableInfo
from databricks.sdk.service.database import DatabaseInstance
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex
from databricks_langchain import (
    DatabricksFunctionClient,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables.base import RunnableLike
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from loguru import logger
from mlflow.genai.datasets import EvaluationDataset, create_dataset, get_dataset
from mlflow.genai.prompts import PromptVersion, load_prompt
from mlflow.models import ModelConfig
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksLakebase,
    DatabricksResource,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksUCConnection,
    DatabricksVectorSearchIndex,
)
from mlflow.pyfunc import ChatModel, ResponsesAgent
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)


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
    def as_resources(self) -> Sequence[DatabricksResource]: ...

    @property
    @abstractmethod
    def api_scopes(self) -> Sequence[str]: ...

    @property
    def workspace_client(self) -> WorkspaceClient:
        credentials_strategy: CredentialsStrategy = None
        if self.on_behalf_of_user:
            credentials_strategy = ModelServingUserCredentials()
        logger.debug(
            f"Creating WorkspaceClient with credentials strategy: {credentials_strategy}"
        )
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
    options: list[
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
        for v in self.options:
            value = value_of(v)
            if value is not None:
                return value
        return self.default_value


AnyVariable: TypeAlias = (
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
    name: Optional[str] = None

    @model_validator(mode="after")
    def validate_name_or_schema_required(self) -> "TableModel":
        if not self.name and not self.schema_model:
            raise ValueError(
                "Either 'name' or 'schema_model' must be provided for TableModel"
            )
        return self

    @property
    def full_name(self) -> str:
        if self.schema_model:
            name: str = ""
            if self.name:
                name = f".{self.name}"
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}{name}"
        return self.name

    @property
    def api_scopes(self) -> Sequence[str]:
        return []

    def as_resources(self) -> Sequence[DatabricksResource]:
        resources: list[DatabricksResource] = []

        excluded_suffixes: Sequence[str] = [
            "_payload",
            "_assessment_logs",
            "_request_logs",
        ]

        excluded_prefixes: Sequence[str] = ["trace_logs_"]

        if self.name:
            resources.append(
                DatabricksTable(
                    table_name=self.full_name, on_behalf_of_user=self.on_behalf_of_user
                )
            )
        else:
            w: WorkspaceClient = self.workspace_client
            schema_full_name: str = self.schema_model.full_name
            tables: Iterator[TableInfo] = w.tables.list(
                catalog_name=self.schema_model.catalog_name,
                schema_name=self.schema_model.schema_name,
            )
            resources.extend(
                [
                    DatabricksTable(
                        table_name=f"{schema_full_name}.{table.name}",
                        on_behalf_of_user=self.on_behalf_of_user,
                    )
                    for table in tables
                    if not any(
                        table.name.endswith(suffix) for suffix in excluded_suffixes
                    )
                    and not any(
                        table.name.startswith(prefix) for prefix in excluded_prefixes
                    )
                ]
            )

        return resources


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

    @property
    def uri(self) -> str:
        return f"databricks:/{self.name}"

    def as_resources(self) -> Sequence[DatabricksResource]:
        return [
            DatabricksServingEndpoint(
                endpoint_name=self.name, on_behalf_of_user=self.on_behalf_of_user
            )
        ]

    def as_chat_model(self) -> LanguageModelLike:
        # Retrieve langchain chat client from workspace client to enable OBO
        # ChatOpenAI does not allow additional inputs at the moment, so we cannot use it directly
        # chat_client: LanguageModelLike = self.as_open_ai_client()

        # Create ChatDatabricksWrapper instance directly
        from dao_ai.chat_models import ChatDatabricksFiltered

        chat_client: LanguageModelLike = ChatDatabricksFiltered(
            model=self.name, temperature=self.temperature, max_tokens=self.max_tokens
        )
        # chat_client: LanguageModelLike = ChatDatabricks(
        #     model=self.name, temperature=self.temperature, max_tokens=self.max_tokens
        # )

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

    def as_open_ai_client(self) -> LanguageModelLike:
        chat_client: ChatOpenAI = (
            self.workspace_client.serving_endpoints.get_langchain_chat_open_ai_client(
                model=self.name
            )
        )
        chat_client.temperature = self.temperature
        chat_client.max_tokens = self.max_tokens

        return chat_client


class VectorSearchEndpointType(str, Enum):
    STANDARD = "STANDARD"
    OPTIMIZED_STORAGE = "OPTIMIZED_STORAGE"


class VectorSearchEndpoint(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    type: VectorSearchEndpointType = VectorSearchEndpointType.STANDARD


class IndexModel(BaseModel, HasFullName, IsDatabricksResource):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
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

    def as_resources(self) -> Sequence[DatabricksResource]:
        return [
            DatabricksVectorSearchIndex(
                index_name=self.full_name, on_behalf_of_user=self.on_behalf_of_user
            )
        ]


class GenieRoomModel(BaseModel, IsDatabricksResource):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    description: Optional[str] = None
    space_id: AnyVariable

    @property
    def api_scopes(self) -> Sequence[str]:
        return [
            "dashboards.genie",
        ]

    def as_resources(self) -> Sequence[DatabricksResource]:
        return [
            DatabricksGenieSpace(
                genie_space_id=value_of(self.space_id),
                on_behalf_of_user=self.on_behalf_of_user,
            )
        ]

    @model_validator(mode="after")
    def update_space_id(self):
        self.space_id = value_of(self.space_id)
        return self


class VolumeModel(BaseModel, HasFullName, IsDatabricksResource):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
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

    @property
    def api_scopes(self) -> Sequence[str]:
        return ["files.files", "catalog.volumes"]

    def as_resources(self) -> Sequence[DatabricksResource]:
        return []


class VolumePathModel(BaseModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    volume: Optional[VolumeModel] = None
    path: Optional[str] = None

    @model_validator(mode="after")
    def validate_path_or_volume(self) -> "VolumePathModel":
        if not self.volume and not self.path:
            raise ValueError("Either 'volume' or 'path' must be provided")
        return self

    @property
    def full_name(self) -> str:
        if self.volume and self.volume.schema_model:
            catalog_name: str = self.volume.schema_model.catalog_name
            schema_name: str = self.volume.schema_model.schema_name
            volume_name: str = self.volume.name
            path = f"/{self.path}" if self.path else ""
            return f"/Volumes/{catalog_name}/{schema_name}/{volume_name}{path}"
        return self.path

    def as_path(self) -> Path:
        return Path(self.full_name)

    def create(self, w: WorkspaceClient | None = None) -> None:
        from dao_ai.providers.databricks import DatabricksProvider

        if self.volume:
            self.volume.create(w=w)

        provider: DatabricksProvider = DatabricksProvider(w=w)
        provider.create_path(self)


class VectorStoreModel(BaseModel, IsDatabricksResource):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    embedding_model: Optional[LLMModel] = None
    index: Optional[IndexModel] = None
    endpoint: Optional[VectorSearchEndpoint] = None
    source_table: TableModel
    source_path: Optional[VolumePathModel] = None
    checkpoint_path: Optional[VolumePathModel] = None
    primary_key: Optional[str] = None
    columns: Optional[list[str]] = Field(default_factory=list)
    doc_uri: Optional[str] = None
    embedding_source_column: str

    @model_validator(mode="after")
    def set_default_embedding_model(self):
        if not self.embedding_model:
            self.embedding_model = LLMModel(name="databricks-gte-large-en")
        return self

    @model_validator(mode="after")
    def set_default_primary_key(self):
        if self.primary_key is None:
            from dao_ai.providers.databricks import DatabricksProvider

            provider: DatabricksProvider = DatabricksProvider()
            primary_key: Sequence[str] | None = provider.find_primary_key(
                self.source_table
            )
            if not primary_key:
                raise ValueError(
                    "Missing field primary_key and unable to find an appropriate primary_key."
                )
            if len(primary_key) > 1:
                raise ValueError(
                    f"Table {self.source_table.full_name} has more than one primary key: {primary_key}"
                )
            self.primary_key = primary_key[0] if primary_key else None

        return self

    @model_validator(mode="after")
    def set_default_index(self):
        if self.index is None:
            name: str = f"{self.source_table.name}_index"
            self.index = IndexModel(schema=self.source_table.schema_model, name=name)
        return self

    @model_validator(mode="after")
    def set_default_endpoint(self):
        if self.endpoint is None:
            from dao_ai.providers.databricks import (
                DatabricksProvider,
                with_available_indexes,
            )

            provider: DatabricksProvider = DatabricksProvider()
            logger.debug("Finding endpoint for existing index...")
            endpoint_name: str | None = provider.find_endpoint_for_index(self.index)
            if endpoint_name is None:
                logger.debug("Finding first endpoint with available indexes...")
                endpoint_name = provider.find_vector_search_endpoint(
                    with_available_indexes
                )
            if endpoint_name is None:
                logger.debug("No endpoint found, creating a new name...")
                endpoint_name = (
                    f"{self.source_table.schema_model.catalog_name}_endpoint"
                )
            logger.debug(f"Using endpoint: {endpoint_name}")
            self.endpoint = VectorSearchEndpoint(name=endpoint_name)

        return self

    @property
    def api_scopes(self) -> Sequence[str]:
        return [
            "vectorsearch.vector-search-endpoints",
            "serving.serving-endpoints",
        ] + self.index.api_scopes

    def as_resources(self) -> Sequence[DatabricksResource]:
        return self.index.as_resources()

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


class FunctionModel(BaseModel, HasFullName, IsDatabricksResource):
    model_config = ConfigDict()
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: Optional[str] = None

    @model_validator(mode="after")
    def validate_name_or_schema_required(self) -> "FunctionModel":
        if not self.name and not self.schema_model:
            raise ValueError(
                "Either 'name' or 'schema_model' must be provided for FunctionModel"
            )
        return self

    @property
    def full_name(self) -> str:
        if self.schema_model:
            name: str = ""
            if self.name:
                name = f".{self.name}"
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}{name}"
        return self.name

    def as_resources(self) -> Sequence[DatabricksResource]:
        resources: list[DatabricksResource] = []
        if self.name:
            resources.append(
                DatabricksFunction(
                    function_name=self.full_name,
                    on_behalf_of_user=self.on_behalf_of_user,
                )
            )
        else:
            w: WorkspaceClient = self.workspace_client
            schema_full_name: str = self.schema_model.full_name
            functions: Iterator[FunctionInfo] = w.functions.list(
                catalog_name=self.schema_model.catalog_name,
                schema_name=self.schema_model.schema_name,
            )
            resources.extend(
                [
                    DatabricksFunction(
                        function_name=f"{schema_full_name}.{function.name}",
                        on_behalf_of_user=self.on_behalf_of_user,
                    )
                    for function in functions
                ]
            )

        return resources

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
            "serving.serving-endpoints",
            "mcp.genie",
            "mcp.functions",
            "mcp.vectorsearch",
            "mcp.external",
        ]

    def as_resources(self) -> Sequence[DatabricksResource]:
        return [
            DatabricksUCConnection(
                connection_name=self.name, on_behalf_of_user=self.on_behalf_of_user
            )
        ]


class WarehouseModel(BaseModel, IsDatabricksResource):
    model_config = ConfigDict()
    name: str
    description: Optional[str] = None
    warehouse_id: AnyVariable

    @property
    def api_scopes(self) -> Sequence[str]:
        return [
            "sql.warehouses",
            "sql.statement-execution",
        ]

    def as_resources(self) -> Sequence[DatabricksResource]:
        return [
            DatabricksSQLWarehouse(
                warehouse_id=value_of(self.warehouse_id),
                on_behalf_of_user=self.on_behalf_of_user,
            )
        ]

    @model_validator(mode="after")
    def update_warehouse_id(self):
        self.warehouse_id = value_of(self.warehouse_id)
        return self


class DatabaseModel(BaseModel, IsDatabricksResource):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    instance_name: Optional[str] = None
    description: Optional[str] = None
    host: Optional[AnyVariable] = None
    database: Optional[AnyVariable] = "databricks_postgres"
    port: Optional[AnyVariable] = 5432
    connection_kwargs: Optional[dict[str, Any]] = Field(default_factory=dict)
    max_pool_size: Optional[int] = 10
    timeout_seconds: Optional[int] = 10
    capacity: Optional[Literal["CU_1", "CU_2"]] = "CU_2"
    node_count: Optional[int] = None
    user: Optional[AnyVariable] = None
    password: Optional[AnyVariable] = None
    client_id: Optional[AnyVariable] = None
    client_secret: Optional[AnyVariable] = None
    workspace_host: Optional[AnyVariable] = None

    @property
    def api_scopes(self) -> Sequence[str]:
        return []

    def as_resources(self) -> Sequence[DatabricksResource]:
        return [
            DatabricksLakebase(
                database_instance_name=self.instance_name,
                on_behalf_of_user=self.on_behalf_of_user,
            )
        ]

    @model_validator(mode="after")
    def update_instance_name(self):
        if self.instance_name is None:
            self.instance_name = self.name

        return self

    @model_validator(mode="after")
    def update_user(self):
        if self.client_id or self.user:
            return self

        self.user = self.workspace_client.current_user.me().user_name
        if not self.user:
            raise ValueError(
                "Unable to determine current user. Please provide a user name or OAuth credentials."
            )

        return self

    @model_validator(mode="after")
    def update_host(self):
        if self.host is not None:
            return self

        existing_instance: DatabaseInstance = (
            self.workspace_client.database.get_database_instance(
                name=self.instance_name
            )
        )
        self.host = existing_instance.read_write_dns
        return self

    @model_validator(mode="after")
    def validate_auth_methods(self):
        oauth_fields: Sequence[Any] = [
            self.workspace_host,
            self.client_id,
            self.client_secret,
        ]
        has_oauth: bool = all(field is not None for field in oauth_fields)

        pat_fields: Sequence[Any] = [self.user]
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
    def connection_params(self) -> dict[str, Any]:
        """
        Get database connection parameters as a dictionary.

        Returns a dict with connection parameters suitable for psycopg ConnectionPool.
        If username is configured, it will be included; otherwise it will be omitted
        to allow Lakebase to authenticate using the token's identity.
        """
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        username: str | None = None

        if self.client_id and self.client_secret and self.workspace_host:
            username = value_of(self.client_id)
        elif self.user:
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

        token: str = provider.lakebase_password_provider(self.instance_name)

        # Build connection parameters dictionary
        params: dict[str, Any] = {
            "dbname": database,
            "host": host,
            "port": port,
            "password": token,
            "sslmode": "require",
        }

        # Only include user if explicitly configured
        if username:
            params["user"] = username
            logger.debug(
                f"Connection params: dbname={database} user={username} host={host} port={port} password=******** sslmode=require"
            )
        else:
            logger.debug(
                f"Connection params: dbname={database} host={host} port={port} password=******** sslmode=require (using token identity)"
            )

        return params

    @property
    def connection_url(self) -> str:
        """
        Get database connection URL as a string (for backwards compatibility).

        Note: It's recommended to use connection_params instead for better flexibility.
        """
        params = self.connection_params
        parts = [f"{k}={v}" for k, v in params.items()]
        return " ".join(parts)

    def create(self, w: WorkspaceClient | None = None) -> None:
        from dao_ai.providers.databricks import DatabricksProvider

        provider: DatabricksProvider = DatabricksProvider()
        provider.create_lakebase(self)
        provider.create_lakebase_instance_role(self)


class SearchParametersModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    num_results: Optional[int] = 10
    filters: Optional[dict[str, Any]] = Field(default_factory=dict)
    query_type: Optional[str] = "ANN"


class RetrieverModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    vector_store: VectorStoreModel
    columns: Optional[list[str]] = Field(default_factory=list)
    search_parameters: SearchParametersModel = Field(
        default_factory=SearchParametersModel
    )

    @model_validator(mode="after")
    def set_default_columns(self):
        if not self.columns:
            columns: Sequence[str] = self.vector_store.columns
            self.columns = columns
        return self


class FunctionType(str, Enum):
    PYTHON = "python"
    FACTORY = "factory"
    UNITY_CATALOG = "unity_catalog"
    MCP = "mcp"


class HumanInTheLoopActionType(str, Enum):
    """Supported action types for human-in-the-loop interactions."""

    ACCEPT = "accept"
    EDIT = "edit"
    RESPONSE = "response"
    DECLINE = "decline"


class HumanInTheLoopModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    review_prompt: str = "Please review the tool call"
    interrupt_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
            "allow_decline": True,
        }
    )
    decline_message: str = "Tool call declined by user"
    custom_actions: Optional[dict[str, str]] = Field(default_factory=dict)


class BaseFunctionModel(ABC, BaseModel):
    model_config = ConfigDict(
        use_enum_values=True,
        discriminator="type",
    )
    type: FunctionType
    name: str
    human_in_the_loop: Optional[HumanInTheLoopModel] = None

    @abstractmethod
    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]: ...

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

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_python_tool

        return [create_python_tool(self, **kwargs)]


class FactoryFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    args: Optional[dict[str, Any]] = Field(default_factory=dict)
    type: Literal[FunctionType.FACTORY] = FunctionType.FACTORY

    @property
    def full_name(self) -> str:
        return self.name

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_factory_tool

        return [create_factory_tool(self, **kwargs)]

    @model_validator(mode="after")
    def update_args(self):
        for key, value in self.args.items():
            self.args[key] = value_of(value)
        return self


class TransportType(str, Enum):
    STREAMABLE_HTTP = "streamable_http"
    STDIO = "stdio"


class McpFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.MCP] = FunctionType.MCP

    transport: TransportType = TransportType.STREAMABLE_HTTP
    command: Optional[str] = "python"
    url: Optional[AnyVariable] = None
    connection: Optional[ConnectionModel] = None
    headers: dict[str, AnyVariable] = Field(default_factory=dict)
    args: list[str] = Field(default_factory=list)
    pat: Optional[AnyVariable] = None
    client_id: Optional[AnyVariable] = None
    client_secret: Optional[AnyVariable] = None
    workspace_host: Optional[AnyVariable] = None

    @property
    def full_name(self) -> str:
        return self.name

    @field_serializer("transport")
    def serialize_transport(self, value) -> str:
        if isinstance(value, TransportType):
            return value.value
        return str(value)

    @model_validator(mode="after")
    def validate_mutually_exclusive(self):
        if self.transport == TransportType.STREAMABLE_HTTP and not (
            self.url or self.connection
        ):
            raise ValueError(
                "url or connection must be provided for STREAMABLE_HTTP transport"
            )
        if self.transport == TransportType.STDIO and not self.command:
            raise ValueError("command must not be provided for STDIO transport")
        if self.transport == TransportType.STDIO and not self.args:
            raise ValueError("args must not be provided for STDIO transport")
        return self

    @model_validator(mode="after")
    def update_url(self):
        self.url = value_of(self.url)
        return self

    @model_validator(mode="after")
    def update_headers(self):
        for key, value in self.headers.items():
            self.headers[key] = value_of(value)
        return self

    @model_validator(mode="after")
    def validate_auth_methods(self):
        oauth_fields: Sequence[Any] = [
            self.client_id,
            self.client_secret,
        ]
        has_oauth: bool = all(field is not None for field in oauth_fields)

        pat_fields: Sequence[Any] = [self.pat]
        has_user_auth: bool = all(field is not None for field in pat_fields)

        if has_oauth and has_user_auth:
            raise ValueError(
                "Cannot use both OAuth and user authentication methods. "
                "Please provide either OAuth credentials or user credentials."
            )

        if (has_oauth or has_user_auth) and not self.workspace_host:
            raise ValueError(
                "Workspace host must be provided when using OAuth or user credentials."
            )

        return self

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_mcp_tools

        return create_mcp_tools(self)


class UnityCatalogFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    partial_args: Optional[dict[str, AnyVariable]] = Field(default_factory=dict)
    type: Literal[FunctionType.UNITY_CATALOG] = FunctionType.UNITY_CATALOG

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_uc_tools

        return create_uc_tools(self)


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


class PromptModel(BaseModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str
    description: Optional[str] = None
    default_template: Optional[str] = None
    alias: Optional[str] = None
    version: Optional[int] = None
    tags: Optional[dict[str, Any]] = Field(default_factory=dict)

    @property
    def template(self) -> str:
        from dao_ai.providers.databricks import DatabricksProvider

        provider: DatabricksProvider = DatabricksProvider()
        prompt_version = provider.get_prompt(self)
        return prompt_version.to_single_brace_format()

    @property
    def full_name(self) -> str:
        prompt_name: str = self.name
        if self.schema_model:
            prompt_name = f"{self.schema_model.full_name}.{prompt_name}"
        return prompt_name

    @property
    def uri(self) -> str:
        prompt_uri: str = f"prompts:/{self.full_name}"

        if self.alias:
            prompt_uri = f"prompts:/{self.full_name}@{self.alias}"
        elif self.version:
            prompt_uri = f"prompts:/{self.full_name}/{self.version}"
        else:
            prompt_uri = f"prompts:/{self.full_name}@latest"

        return prompt_uri

    def as_prompt(self) -> PromptVersion:
        prompt_version: PromptVersion = load_prompt(self.uri)
        return prompt_version

    @model_validator(mode="after")
    def validate_mutually_exclusive(self):
        if self.alias and self.version:
            raise ValueError("Cannot specify both alias and version")
        return self


class AgentModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    description: Optional[str] = None
    model: LLMModel
    tools: list[ToolModel] = Field(default_factory=list)
    guardrails: list[GuardrailModel] = Field(default_factory=list)
    prompt: Optional[str | PromptModel] = None
    handoff_prompt: Optional[str] = None
    create_agent_hook: Optional[FunctionHook] = None
    pre_agent_hook: Optional[FunctionHook] = None
    post_agent_hook: Optional[FunctionHook] = None


class SupervisorModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel
    tools: list[ToolModel] = Field(default_factory=list)
    prompt: Optional[str] = None


class SwarmModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel
    default_agent: Optional[AgentModel | str] = None
    handoffs: Optional[dict[str, Optional[list[AgentModel | str]]]] = Field(
        default_factory=dict
    )


class OrchestrationModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    supervisor: Optional[SupervisorModel] = None
    swarm: Optional[SwarmModel] = None
    memory: Optional[MemoryModel] = None

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
    input: Optional[list[Message]] = None
    messages: Optional[list[Message]] = None
    custom_inputs: Optional[dict] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mutual_exclusion_and_alias(self) -> "ChatPayload":
        """Handle dual field support with automatic aliasing."""
        # If both fields are provided and they're the same, that's okay (redundant but valid)
        if self.input is not None and self.messages is not None:
            # Allow if they're identical (redundant specification)
            if self.input == self.messages:
                return self
            # If they're different, prefer input and copy to messages
            else:
                self.messages = self.input
                return self

        # If neither field is provided, that's an error
        if self.input is None and self.messages is None:
            raise ValueError("Must specify either 'input' or 'messages' field.")

        # Create alias: copy messages to input if input is None
        if self.input is None and self.messages is not None:
            self.input = self.messages

        # Create alias: copy input to messages if messages is None
        elif self.messages is None and self.input is not None:
            self.messages = self.input

        return self


class ChatHistoryModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel
    max_tokens: int = 256
    max_tokens_before_summary: Optional[int] = None
    max_messages_before_summary: Optional[int] = None
    max_summary_tokens: int = 255

    @model_validator(mode="after")
    def validate_max_summary_tokens(self) -> "ChatHistoryModel":
        if self.max_summary_tokens >= self.max_tokens:
            raise ValueError(
                f"max_summary_tokens ({self.max_summary_tokens}) must be less than max_tokens ({self.max_tokens})"
            )
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
    environment_vars: Optional[dict[str, AnyVariable]] = Field(default_factory=dict)
    budget_policy_id: Optional[str] = None
    workload_size: Optional[WorkloadSize] = "Small"
    permissions: Optional[list[AppPermissionModel]] = Field(default_factory=list)
    agents: list[AgentModel] = Field(default_factory=list)

    orchestration: Optional[OrchestrationModel] = None
    alias: Optional[str] = None
    initialization_hooks: Optional[FunctionHook | list[FunctionHook]] = Field(
        default_factory=list
    )
    shutdown_hooks: Optional[FunctionHook | list[FunctionHook]] = Field(
        default_factory=list
    )
    message_hooks: Optional[FunctionHook | list[FunctionHook]] = Field(
        default_factory=list
    )
    input_example: Optional[ChatPayload] = None
    chat_history: Optional[ChatHistoryModel] = None
    code_paths: list[str] = Field(default_factory=list)
    pip_requirements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_agents_not_empty(self):
        if not self.agents:
            raise ValueError("At least one agent must be specified")
        return self

    @model_validator(mode="after")
    def update_environment_vars(self):
        for key, value in self.environment_vars.items():
            if isinstance(value, SecretVariableModel):
                updated_value = str(value)
            else:
                updated_value = value_of(value)

            self.environment_vars[key] = updated_value
        return self

    @model_validator(mode="after")
    def set_default_orchestration(self):
        if self.orchestration is None:
            if len(self.agents) > 1:
                default_agent: AgentModel = self.agents[0]
                self.orchestration = OrchestrationModel(
                    supervisor=SupervisorModel(model=default_agent.model)
                )
            elif len(self.agents) == 1:
                default_agent: AgentModel = self.agents[0]
                self.orchestration = OrchestrationModel(
                    swarm=SwarmModel(
                        model=default_agent.model, default_agent=default_agent
                    )
                )
            else:
                raise ValueError("At least one agent must be specified")

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

    @model_validator(mode="after")
    def add_code_paths_to_sys_path(self):
        for code_path in self.code_paths:
            parent_path: str = str(Path(code_path).parent)
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
                logger.debug(f"Added code path to sys.path: {parent_path}")
        importlib.invalidate_caches()
        return self


class GuidelineModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    guidelines: list[str]


class EvaluationModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel
    table: TableModel
    num_evals: int
    agent_description: Optional[str] = None
    question_guidelines: Optional[str] = None
    custom_inputs: dict[str, Any] = Field(default_factory=dict)
    guidelines: list[GuidelineModel] = Field(default_factory=list)



class EvaluationDatasetInputsModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    question: str

class EvaluationDatasetExpectationsModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    expected_response: Optional[str] = None
    expected_facts: Optional[list[str]] = None
    
    
    @model_validator(mode="after")
    def validate_mutually_exclusive(self):
        if self.expected_response is not None and self.expected_facts is not None:
            raise ValueError("Cannot specify both expected_response and expected_facts")
        return self

class EvaluationDatasetEntryModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    inputs: EvaluationDatasetInputsModel
    expectations: EvaluationDatasetExpectationsModel


class EvaluationDatasetModel(BaseModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str
    data: Optional[list[EvaluationDatasetEntryModel]] = Field(default_factory=list)

    def as_dataset(self, w: WorkspaceClient | None = None) -> EvaluationDataset:
        evaluation_dataset: EvaluationDataset
        try:
            evaluation_dataset = get_dataset(name=self.full_name)
        except Exception:
            logger.warning(f"Dataset {self.full_name} not found, creating new dataset")
            evaluation_dataset = create_dataset(name=self.full_name)

            if self.data:
                logger.debug(
                    f"Merging {len(self.data)} entries into dataset {self.full_name}"
                )
                evaluation_dataset.merge_records([e.model_dump() for e in self.data])

        return evaluation_dataset  # If inputs and expectations are provided, create/update the dataset

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class PromptOptimizationModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    prompt: PromptModel
    agent: AgentModel | str
    dataset: (
        EvaluationDatasetModel | str
    )  # Reference to dataset name (looked up in OptimizationsModel.training_datasets or MLflow)
    reflection_model: Optional[LLMModel | str] = None
    num_candidates: Optional[int] = 5
    max_steps: Optional[int] = 3
    scorer_model: Optional[LLMModel | str] = None
    temperature: Optional[float] = 0.0

    def optimize(self, w: WorkspaceClient | None = None) -> "PromptModel":
        """
        Optimize the prompt using MLflow's prompt optimization.

        Args:
            w: Optional WorkspaceClient for Databricks operations

        Returns:
            PromptModel: The optimized prompt model with new URI
        """
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w)
        optimized_prompt: PromptModel = provider.optimize_prompt(self)
        return optimized_prompt


class OptimizationsModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    training_datasets: dict[str, EvaluationDatasetModel] = Field(default_factory=dict)
    prompt_optimizations: dict[str, PromptOptimizationModel] = Field(
        default_factory=dict
    )

    def get_dataset(self, dataset_name: str) -> Any:
        """
        Get a dataset by name, either from training_datasets or from MLflow.

        Args:
            dataset_name: Name of the dataset to retrieve

        Returns:
            EvaluationDataset: The MLflow evaluation dataset
        """
        # First, check if dataset is defined in training_datasets
        if dataset_name in self.training_datasets:
            logger.debug(
                f"Found dataset '{dataset_name}' in training_datasets, calling as_dataset()"
            )
            return self.training_datasets[dataset_name].as_dataset()

        # Otherwise, look it up from MLflow
        logger.debug(
            f"Dataset '{dataset_name}' not in training_datasets, looking up in MLflow"
        )
        from mlflow.genai.datasets import get_dataset

        return get_dataset(name=dataset_name)

    def optimize(self, w: WorkspaceClient | None = None) -> dict[str, PromptModel]:
        """
        Optimize all prompts in this configuration.

        This method:
        1. Ensures all training datasets are created/registered in MLflow
        2. Runs each prompt optimization

        Args:
            w: Optional WorkspaceClient for Databricks operations

        Returns:
            dict[str, PromptModel]: Dictionary mapping optimization names to optimized prompts
        """
        # First, ensure all training datasets are created/registered in MLflow
        logger.info(f"Ensuring {len(self.training_datasets)} training datasets exist")
        for dataset_name, dataset_model in self.training_datasets.items():
            logger.debug(f"Creating/updating dataset: {dataset_name}")
            dataset_model.as_dataset()

        # Run optimizations
        results: dict[str, PromptModel] = {}
        for name, optimization in self.prompt_optimizations.items():
            results[name] = optimization.optimize(w)
        return results


class DatasetFormat(str, Enum):
    CSV = "csv"
    DELTA = "delta"
    JSON = "json"
    PARQUET = "parquet"
    ORC = "orc"
    SQL = "sql"
    EXCEL = "excel"


class DatasetModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    table: Optional[TableModel] = None
    ddl: Optional[str | VolumeModel] = None
    data: Optional[str | VolumePathModel] = None
    format: Optional[DatasetFormat] = None
    read_options: Optional[dict[str, Any]] = Field(default_factory=dict)
    table_schema: Optional[str] = None
    parameters: Optional[dict[str, Any]] = Field(default_factory=dict)

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
    parameters: Optional[dict[str, Any]] = Field(default_factory=dict)
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
    variables: dict[str, AnyVariable] = Field(default_factory=dict)
    schemas: dict[str, SchemaModel] = Field(default_factory=dict)
    resources: Optional[ResourcesModel] = None
    retrievers: dict[str, RetrieverModel] = Field(default_factory=dict)
    tools: dict[str, ToolModel] = Field(default_factory=dict)
    guardrails: dict[str, GuardrailModel] = Field(default_factory=dict)
    memory: Optional[MemoryModel] = None
    prompts: dict[str, PromptModel] = Field(default_factory=dict)
    agents: dict[str, AgentModel] = Field(default_factory=dict)
    app: Optional[AppModel] = None
    evaluation: Optional[EvaluationModel] = None
    optimizations: Optional[OptimizationsModel] = None
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

        config.initialize()

        atexit.register(config.shutdown)

        return config

    def initialize(self) -> None:
        from dao_ai.hooks.core import create_hooks

        if self.app and self.app.log_level:
            logger.remove()
            logger.add(sys.stderr, level=self.app.log_level)

        logger.debug("Calling initialization hooks...")
        initialization_functions: Sequence[Callable[..., Any]] = create_hooks(
            self.app.initialization_hooks
        )
        for initialization_function in initialization_functions:
            logger.debug(
                f"Running initialization hook: {initialization_function.__name__}"
            )
            initialization_function(self)

    def shutdown(self) -> None:
        from dao_ai.hooks.core import create_hooks

        logger.debug("Calling shutdown hooks...")
        shutdown_functions: Sequence[Callable[..., Any]] = create_hooks(
            self.app.shutdown_hooks
        )
        for shutdown_function in shutdown_functions:
            logger.debug(f"Running shutdown hook: {shutdown_function.__name__}")
            try:
                shutdown_function(self)
            except Exception as e:
                logger.error(
                    f"Error during shutdown hook {shutdown_function.__name__}: {e}"
                )

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
    ) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w)
        provider.create_agent(self)

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

    def as_graph(self) -> CompiledStateGraph:
        from dao_ai.graph import create_dao_ai_graph

        graph: CompiledStateGraph = create_dao_ai_graph(config=self)
        return graph

    def as_chat_model(self) -> ChatModel:
        from dao_ai.models import create_agent

        graph: CompiledStateGraph = self.as_graph()
        app: ChatModel = create_agent(graph)
        return app

    def as_responses_agent(self) -> ResponsesAgent:
        from dao_ai.models import create_responses_agent

        graph: CompiledStateGraph = self.as_graph()
        app: ResponsesAgent = create_responses_agent(graph)
        return app
