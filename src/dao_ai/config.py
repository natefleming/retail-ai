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
    Self,
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
    ChatDatabricks,
    DatabricksEmbeddings,
    DatabricksFunctionClient,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, messages_from_dict
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
from mlflow.types.responses import (
    ResponsesAgentRequest,
)
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


class ServicePrincipalModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )
    client_id: AnyVariable
    client_secret: AnyVariable


class IsDatabricksResource(ABC, BaseModel):
    """
    Base class for Databricks resources with authentication support.

    Authentication Options:
    ----------------------
    1. **On-Behalf-Of User (OBO)**: Set on_behalf_of_user=True to use the
       calling user's identity via ModelServingUserCredentials.

    2. **Service Principal (OAuth M2M)**: Provide service_principal or
       (client_id + client_secret + workspace_host) for service principal auth.

    3. **Personal Access Token (PAT)**: Provide pat (and optionally workspace_host)
       to authenticate with a personal access token.

    4. **Ambient Authentication**: If no credentials provided, uses SDK defaults
       (environment variables, notebook context, etc.)

    Authentication Priority:
    1. OBO (on_behalf_of_user=True)
    2. Service Principal (client_id + client_secret + workspace_host)
    3. PAT (pat + workspace_host)
    4. Ambient/default authentication
    """

    model_config = ConfigDict(use_enum_values=True)

    on_behalf_of_user: Optional[bool] = False
    service_principal: Optional[ServicePrincipalModel] = None
    client_id: Optional[AnyVariable] = None
    client_secret: Optional[AnyVariable] = None
    workspace_host: Optional[AnyVariable] = None
    pat: Optional[AnyVariable] = None

    @abstractmethod
    def as_resources(self) -> Sequence[DatabricksResource]: ...

    @property
    @abstractmethod
    def api_scopes(self) -> Sequence[str]: ...

    @model_validator(mode="after")
    def _expand_service_principal(self) -> Self:
        """Expand service_principal into client_id and client_secret if provided."""
        if self.service_principal is not None:
            if self.client_id is None:
                self.client_id = self.service_principal.client_id
            if self.client_secret is None:
                self.client_secret = self.service_principal.client_secret
        return self

    @model_validator(mode="after")
    def _validate_auth_not_mixed(self) -> Self:
        """Validate that OAuth and PAT authentication are not both provided."""
        has_oauth: bool = self.client_id is not None and self.client_secret is not None
        has_pat: bool = self.pat is not None

        if has_oauth and has_pat:
            raise ValueError(
                "Cannot use both OAuth and user authentication methods. "
                "Please provide either OAuth credentials or user credentials."
            )
        return self

    @property
    def workspace_client(self) -> WorkspaceClient:
        """
        Get a WorkspaceClient configured with the appropriate authentication.

        Authentication priority:
        1. If on_behalf_of_user is True, uses ModelServingUserCredentials (OBO)
        2. If service principal credentials are configured (client_id, client_secret,
           workspace_host), uses OAuth M2M
        3. If PAT is configured, uses token authentication
        4. Otherwise, uses default/ambient authentication
        """
        from dao_ai.utils import normalize_host

        # Check for OBO first (highest priority)
        if self.on_behalf_of_user:
            credentials_strategy: CredentialsStrategy = ModelServingUserCredentials()
            logger.debug(
                f"Creating WorkspaceClient for {self.__class__.__name__} "
                f"with OBO credentials strategy"
            )
            return WorkspaceClient(credentials_strategy=credentials_strategy)

        # Check for service principal credentials
        client_id_value: str | None = (
            value_of(self.client_id) if self.client_id else None
        )
        client_secret_value: str | None = (
            value_of(self.client_secret) if self.client_secret else None
        )
        workspace_host_value: str | None = (
            normalize_host(value_of(self.workspace_host))
            if self.workspace_host
            else None
        )

        if client_id_value and client_secret_value and workspace_host_value:
            logger.debug(
                f"Creating WorkspaceClient for {self.__class__.__name__} with service principal: "
                f"client_id={client_id_value}, host={workspace_host_value}"
            )
            return WorkspaceClient(
                host=workspace_host_value,
                client_id=client_id_value,
                client_secret=client_secret_value,
                auth_type="oauth-m2m",
            )

        # Check for PAT authentication
        pat_value: str | None = value_of(self.pat) if self.pat else None
        if pat_value:
            logger.debug(
                f"Creating WorkspaceClient for {self.__class__.__name__} with PAT"
            )
            return WorkspaceClient(
                host=workspace_host_value,
                token=pat_value,
                auth_type="pat",
            )

        # Default: use ambient authentication
        logger.debug(
            f"Creating WorkspaceClient for {self.__class__.__name__} "
            "with default/ambient authentication"
        )
        return WorkspaceClient()


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
    principals: list[ServicePrincipalModel | str] = Field(default_factory=list)
    privileges: list[Privilege]

    @model_validator(mode="after")
    def resolve_principals(self) -> Self:
        """Resolve ServicePrincipalModel objects to their client_id."""
        resolved: list[str] = []
        for principal in self.principals:
            if isinstance(principal, ServicePrincipalModel):
                resolved.append(value_of(principal.client_id))
            else:
                resolved.append(principal)
        self.principals = resolved
        return self


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


class TableModel(IsDatabricksResource, HasFullName):
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


class LLMModel(IsDatabricksResource):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 8192
    fallbacks: Optional[list[Union[str, "LLMModel"]]] = Field(default_factory=list)
    use_responses_api: Optional[bool] = Field(
        default=False,
        description="Use Responses API for ResponsesAgent endpoints",
    )

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
        chat_client: LanguageModelLike = ChatDatabricks(
            model=self.name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            use_responses_api=self.use_responses_api,
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

    def as_open_ai_client(self) -> LanguageModelLike:
        chat_client: ChatOpenAI = (
            self.workspace_client.serving_endpoints.get_langchain_chat_open_ai_client(
                model=self.name
            )
        )
        chat_client.temperature = self.temperature
        chat_client.max_tokens = self.max_tokens

        return chat_client

    def as_embeddings_model(self) -> Embeddings:
        return DatabricksEmbeddings(endpoint=self.name)


class VectorSearchEndpointType(str, Enum):
    STANDARD = "STANDARD"
    OPTIMIZED_STORAGE = "OPTIMIZED_STORAGE"


class VectorSearchEndpoint(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    type: VectorSearchEndpointType = VectorSearchEndpointType.STANDARD

    @field_serializer("type")
    def serialize_type(self, value: VectorSearchEndpointType) -> str:
        """Ensure enum is serialized to string value."""
        if isinstance(value, VectorSearchEndpointType):
            return value.value
        return str(value)


class IndexModel(IsDatabricksResource, HasFullName):
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


class GenieRoomModel(IsDatabricksResource):
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
    def update_space_id(self) -> Self:
        self.space_id = value_of(self.space_id)
        return self


class VolumeModel(IsDatabricksResource, HasFullName):
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


class VectorStoreModel(IsDatabricksResource):
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
    def set_default_embedding_model(self) -> Self:
        if not self.embedding_model:
            self.embedding_model = LLMModel(name="databricks-gte-large-en")
        return self

    @model_validator(mode="after")
    def set_default_primary_key(self) -> Self:
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
    def set_default_index(self) -> Self:
        if self.index is None:
            name: str = f"{self.source_table.name}_index"
            self.index = IndexModel(schema=self.source_table.schema_model, name=name)
        return self

    @model_validator(mode="after")
    def set_default_endpoint(self) -> Self:
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


class FunctionModel(IsDatabricksResource, HasFullName):
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


class ConnectionModel(IsDatabricksResource, HasFullName):
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


class WarehouseModel(IsDatabricksResource):
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
    def update_warehouse_id(self) -> Self:
        self.warehouse_id = value_of(self.warehouse_id)
        return self


class DatabaseType(str, Enum):
    POSTGRES = "postgres"
    LAKEBASE = "lakebase"


class DatabaseModel(IsDatabricksResource):
    """
    Configuration for a Databricks Lakebase (PostgreSQL) database instance.

    Authentication is inherited from IsDatabricksResource. Additionally supports:
    - user/password: For user-based database authentication

    Database Type:
    - lakebase: Databricks-managed Lakebase instance (authentication optional, supports ambient auth)
    - postgres: Standard PostgreSQL database (authentication required)

    Example Service Principal Configuration:
    ```yaml
    databases:
      my_lakebase:
        name: my-database
        type: lakebase
        service_principal:
          client_id:
            env: SERVICE_PRINCIPAL_CLIENT_ID
          client_secret:
            scope: my-scope
            secret: sp-client-secret
        workspace_host:
          env: DATABRICKS_HOST
    ```

    Example User Configuration:
    ```yaml
    databases:
      my_lakebase:
        name: my-database
        type: lakebase
        user: my-user@databricks.com
    ```

    Example Ambient Authentication (Lakebase only):
    ```yaml
    databases:
      my_lakebase:
        name: my-database
        type: lakebase
        on_behalf_of_user: true
    ```
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    type: Optional[DatabaseType] = DatabaseType.LAKEBASE
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
    # Database-specific auth (user identity for DB connection)
    user: Optional[AnyVariable] = None
    password: Optional[AnyVariable] = None

    @field_serializer("type")
    def serialize_type(self, value: DatabaseType | None) -> str | None:
        """Serialize the database type enum to its string value."""
        return value.value if value is not None else None

    @property
    def api_scopes(self) -> Sequence[str]:
        return ["database.database-instances"]

    def as_resources(self) -> Sequence[DatabricksResource]:
        return [
            DatabricksLakebase(
                database_instance_name=self.instance_name,
                on_behalf_of_user=self.on_behalf_of_user,
            )
        ]

    @model_validator(mode="after")
    def update_instance_name(self) -> Self:
        if self.instance_name is None:
            self.instance_name = self.name
        return self

    @model_validator(mode="after")
    def update_user(self) -> Self:
        # Skip if using OBO (passive auth), explicit credentials, or explicit user
        if self.on_behalf_of_user or self.client_id or self.user or self.pat:
            return self

        # For postgres, we need explicit user credentials
        # For lakebase with no auth, ambient auth is allowed
        if self.type == DatabaseType.POSTGRES:
            # Try to determine current user for local development
            try:
                self.user = self.workspace_client.current_user.me().user_name
            except Exception as e:
                logger.warning(
                    f"Could not determine current user for PostgreSQL database: {e}. "
                    f"Please provide explicit user credentials."
                )
        else:
            # For lakebase, try to determine current user but don't fail if we can't
            try:
                self.user = self.workspace_client.current_user.me().user_name
            except Exception:
                # If we can't determine user and no explicit auth, that's okay
                # for lakebase with ambient auth - credentials will be injected at runtime
                pass

        return self

    @model_validator(mode="after")
    def update_host(self) -> Self:
        if self.host is not None:
            return self

        # Try to fetch host from existing instance
        # This may fail for OBO/ambient auth during model logging (before deployment)
        try:
            existing_instance: DatabaseInstance = (
                self.workspace_client.database.get_database_instance(
                    name=self.instance_name
                )
            )
            self.host = existing_instance.read_write_dns
        except Exception as e:
            # For lakebase with OBO/ambient auth, we can't fetch at config time
            # The host will need to be provided explicitly or fetched at runtime
            if self.type == DatabaseType.LAKEBASE and self.on_behalf_of_user:
                logger.debug(
                    f"Could not fetch host for database {self.instance_name} "
                    f"(Lakebase with OBO mode - will be resolved at runtime): {e}"
                )
            else:
                raise ValueError(
                    f"Could not fetch host for database {self.instance_name}. "
                    f"Please provide the 'host' explicitly or ensure the instance exists: {e}"
                )
        return self

    @model_validator(mode="after")
    def validate_auth_methods(self) -> Self:
        oauth_fields: Sequence[Any] = [
            self.workspace_host,
            self.client_id,
            self.client_secret,
        ]
        has_oauth: bool = all(field is not None for field in oauth_fields)
        has_user_auth: bool = self.user is not None
        has_obo: bool = self.on_behalf_of_user is True
        has_pat: bool = self.pat is not None

        # Count how many auth methods are configured
        auth_methods_count: int = sum([has_oauth, has_user_auth, has_obo, has_pat])

        if auth_methods_count > 1:
            raise ValueError(
                "Cannot mix authentication methods. "
                "Please provide exactly one of: "
                "on_behalf_of_user=true (for passive auth in model serving), "
                "OAuth credentials (service_principal or client_id + client_secret + workspace_host), "
                "PAT (personal access token), "
                "or user credentials (user)."
            )

        # For postgres type, at least one auth method must be configured
        # For lakebase type, auth is optional (supports ambient authentication)
        if self.type == DatabaseType.POSTGRES and auth_methods_count == 0:
            raise ValueError(
                "PostgreSQL databases require explicit authentication. "
                "Please provide one of: "
                "OAuth credentials (workspace_host, client_id, client_secret), "
                "service_principal with workspace_host, "
                "PAT (personal access token), "
                "or user credentials (user)."
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
        import uuid as _uuid

        from databricks.sdk.service.database import DatabaseCredential

        username: str | None = None

        if self.client_id and self.client_secret and self.workspace_host:
            username = value_of(self.client_id)
        elif self.user:
            username = value_of(self.user)
        # For OBO mode, no username is needed - the token identity is used

        # Resolve host - may need to fetch at runtime for OBO mode
        host_value: Any = self.host
        if host_value is None and self.on_behalf_of_user:
            # Fetch host at runtime for OBO mode
            existing_instance: DatabaseInstance = (
                self.workspace_client.database.get_database_instance(
                    name=self.instance_name
                )
            )
            host_value = existing_instance.read_write_dns

        if host_value is None:
            raise ValueError(
                f"Database host not configured for {self.instance_name}. "
                "Please provide 'host' explicitly."
            )

        host: str = value_of(host_value)
        port: int = value_of(self.port)
        database: str = value_of(self.database)

        # Use the resource's own workspace_client to generate the database credential
        w: WorkspaceClient = self.workspace_client
        cred: DatabaseCredential = w.database.generate_database_credential(
            request_id=str(_uuid.uuid4()),
            instance_names=[self.instance_name],
        )
        token: str = cred.token

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

        # Use provided workspace client or fall back to resource's own workspace_client
        if w is None:
            w = self.workspace_client
        provider: DatabricksProvider = DatabricksProvider(w=w)
        provider.create_lakebase(self)
        provider.create_lakebase_instance_role(self)


class GenieLRUCacheParametersModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    capacity: int = 1000
    time_to_live_seconds: int | None = (
        60 * 60 * 24
    )  # 1 day default, None or negative = never expires
    warehouse: WarehouseModel


class GenieSemanticCacheParametersModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    time_to_live_seconds: int | None = (
        60 * 60 * 24
    )  # 1 day default, None or negative = never expires
    similarity_threshold: float = 0.85  # Minimum similarity for question matching (L2 distance converted to 0-1 scale)
    context_similarity_threshold: float = 0.80  # Minimum similarity for context matching (L2 distance converted to 0-1 scale)
    question_weight: Optional[float] = (
        0.6  # Weight for question similarity in combined score (0-1). If not provided, computed as 1 - context_weight
    )
    context_weight: Optional[float] = (
        None  # Weight for context similarity in combined score (0-1). If not provided, computed as 1 - question_weight
    )
    embedding_model: str | LLMModel = "databricks-gte-large-en"
    embedding_dims: int | None = None  # Auto-detected if None
    database: DatabaseModel
    warehouse: WarehouseModel
    table_name: str = "genie_semantic_cache"
    context_window_size: int = 3  # Number of previous turns to include for context
    max_context_tokens: int = (
        2000  # Maximum context length to prevent extremely long embeddings
    )

    @model_validator(mode="after")
    def compute_and_validate_weights(self) -> Self:
        """
        Compute missing weight and validate that question_weight + context_weight = 1.0.

        Either question_weight or context_weight (or both) can be provided.
        The missing one will be computed as 1.0 - provided_weight.
        If both are provided, they must sum to 1.0.
        """
        if self.question_weight is None and self.context_weight is None:
            # Both missing - use defaults
            self.question_weight = 0.6
            self.context_weight = 0.4
        elif self.question_weight is None:
            # Compute question_weight from context_weight
            if not (0.0 <= self.context_weight <= 1.0):
                raise ValueError(
                    f"context_weight must be between 0.0 and 1.0, got {self.context_weight}"
                )
            self.question_weight = 1.0 - self.context_weight
        elif self.context_weight is None:
            # Compute context_weight from question_weight
            if not (0.0 <= self.question_weight <= 1.0):
                raise ValueError(
                    f"question_weight must be between 0.0 and 1.0, got {self.question_weight}"
                )
            self.context_weight = 1.0 - self.question_weight
        else:
            # Both provided - validate they sum to 1.0
            total_weight = self.question_weight + self.context_weight
            if not abs(total_weight - 1.0) < 0.0001:  # Allow small floating point error
                raise ValueError(
                    f"question_weight ({self.question_weight}) + context_weight ({self.context_weight}) "
                    f"must equal 1.0 (got {total_weight}). These weights determine the relative importance "
                    f"of question vs context similarity in the combined score."
                )

        return self


class SearchParametersModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    num_results: Optional[int] = 10
    filters: Optional[dict[str, Any]] = Field(default_factory=dict)
    query_type: Optional[str] = "ANN"


class RerankParametersModel(BaseModel):
    """
    Configuration for reranking retrieved documents using FlashRank.

    FlashRank provides fast, local reranking without API calls using lightweight
    cross-encoder models. Reranking improves retrieval quality by reordering results
    based on semantic relevance to the query.

    Typical workflow:
    1. Retrieve more documents than needed (e.g., 50 via num_results)
    2. Rerank all retrieved documents
    3. Return top_n best matches (e.g., 5)

    Example:
        ```yaml
        retriever:
          search_parameters:
            num_results: 50  # Retrieve more candidates
          rerank:
            model: ms-marco-MiniLM-L-12-v2
            top_n: 5  # Return top 5 after reranking
        ```

    Available models (from fastest to most accurate):
    - "ms-marco-TinyBERT-L-2-v2" (fastest, smallest)
    - "ms-marco-MiniLM-L-6-v2"
    - "ms-marco-MiniLM-L-12-v2" (default, good balance)
    - "rank-T5-flan" (most accurate, slower)
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    model: str = Field(
        default="ms-marco-MiniLM-L-12-v2",
        description="FlashRank model name. Default provides good balance of speed and accuracy.",
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Number of documents to return after reranking. If None, uses search_parameters.num_results.",
    )
    cache_dir: Optional[str] = Field(
        default="/tmp/flashrank_cache",
        description="Directory to cache downloaded model weights.",
    )
    columns: Optional[list[str]] = Field(
        default_factory=list, description="Columns to rerank using DatabricksReranker"
    )


class RetrieverModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    vector_store: VectorStoreModel
    columns: Optional[list[str]] = Field(default_factory=list)
    search_parameters: SearchParametersModel = Field(
        default_factory=SearchParametersModel
    )
    rerank: Optional[RerankParametersModel | bool] = Field(
        default=None,
        description="Optional reranking configuration. Set to true for defaults, or provide ReRankParametersModel for custom settings.",
    )

    @model_validator(mode="after")
    def set_default_columns(self) -> Self:
        if not self.columns:
            columns: Sequence[str] = self.vector_store.columns
            self.columns = columns
        return self

    @model_validator(mode="after")
    def set_default_reranker(self) -> Self:
        """Convert bool to ReRankParametersModel with defaults."""
        if isinstance(self.rerank, bool) and self.rerank:
            self.rerank = RerankParametersModel()
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
    def update_args(self) -> Self:
        for key, value in self.args.items():
            self.args[key] = value_of(value)
        return self


class TransportType(str, Enum):
    STREAMABLE_HTTP = "streamable_http"
    STDIO = "stdio"


class McpFunctionModel(BaseFunctionModel, IsDatabricksResource, HasFullName):
    """
    MCP Function Model with authentication inherited from IsDatabricksResource.

    Authentication for MCP connections uses the same options as other resources:
    - Service Principal (client_id + client_secret + workspace_host)
    - PAT (pat + workspace_host)
    - OBO (on_behalf_of_user)
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.MCP] = FunctionType.MCP
    transport: TransportType = TransportType.STREAMABLE_HTTP
    command: Optional[str] = "python"
    url: Optional[AnyVariable] = None
    headers: dict[str, AnyVariable] = Field(default_factory=dict)
    args: list[str] = Field(default_factory=list)
    # MCP-specific fields
    connection: Optional[ConnectionModel] = None
    functions: Optional[SchemaModel] = None
    genie_room: Optional[GenieRoomModel] = None
    sql: Optional[bool] = None
    vector_search: Optional[VectorStoreModel] = None

    @property
    def api_scopes(self) -> Sequence[str]:
        """API scopes for MCP connections."""
        return [
            "serving.serving-endpoints",
            "mcp.genie",
            "mcp.functions",
            "mcp.vectorsearch",
            "mcp.external",
        ]

    def as_resources(self) -> Sequence[DatabricksResource]:
        """MCP functions don't declare static resources."""
        return []

    @property
    def full_name(self) -> str:
        return self.name

    def _get_workspace_host(self) -> str:
        """
        Get the workspace host, either from config or from workspace client.

        If connection is provided, uses its workspace client.
        Otherwise, falls back to the default Databricks host.

        Returns:
            str: The workspace host URL with https:// scheme and without trailing slash
        """
        from dao_ai.utils import get_default_databricks_host, normalize_host

        # Try to get workspace_host from config
        workspace_host: str | None = (
            normalize_host(value_of(self.workspace_host))
            if self.workspace_host
            else None
        )

        # If no workspace_host in config, get it from workspace client
        if not workspace_host:
            # Use connection's workspace client if available
            if self.connection:
                workspace_host = normalize_host(
                    self.connection.workspace_client.config.host
                )
            else:
                # get_default_databricks_host already normalizes the host
                workspace_host = get_default_databricks_host()

        if not workspace_host:
            raise ValueError(
                "Could not determine workspace host. "
                "Please set workspace_host in config or DATABRICKS_HOST environment variable."
            )

        # Remove trailing slash
        return workspace_host.rstrip("/")

    @property
    def mcp_url(self) -> str:
        """
        Get the MCP URL for this function.

        Returns the URL based on the configured source:
        - If url is set, returns it directly
        - If connection is set, constructs URL from connection
        - If genie_room is set, constructs Genie MCP URL
        - If sql is set, constructs DBSQL MCP URL (serverless)
        - If vector_search is set, constructs Vector Search MCP URL
        - If functions is set, constructs UC Functions MCP URL

        URL patterns (per https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp):
        - Genie: https://{host}/api/2.0/mcp/genie/{space_id}
        - DBSQL: https://{host}/api/2.0/mcp/sql (serverless, workspace-level)
        - Vector Search: https://{host}/api/2.0/mcp/vector-search/{catalog}/{schema}
        - UC Functions: https://{host}/api/2.0/mcp/functions/{catalog}/{schema}
        - Connection: https://{host}/api/2.0/mcp/external/{connection_name}
        """
        # Direct URL provided
        if self.url:
            return self.url

        # Get workspace host (from config, connection, or default workspace client)
        workspace_host: str = self._get_workspace_host()

        # UC Connection
        if self.connection:
            connection_name: str = self.connection.name
            return f"{workspace_host}/api/2.0/mcp/external/{connection_name}"

        # Genie Room
        if self.genie_room:
            space_id: str = value_of(self.genie_room.space_id)
            return f"{workspace_host}/api/2.0/mcp/genie/{space_id}"

        # DBSQL MCP server (serverless, workspace-level)
        if self.sql:
            return f"{workspace_host}/api/2.0/mcp/sql"

        # Vector Search
        if self.vector_search:
            if (
                not self.vector_search.index
                or not self.vector_search.index.schema_model
            ):
                raise ValueError(
                    "vector_search must have an index with a schema (catalog/schema) configured"
                )
            catalog: str = self.vector_search.index.schema_model.catalog_name
            schema: str = self.vector_search.index.schema_model.schema_name
            return f"{workspace_host}/api/2.0/mcp/vector-search/{catalog}/{schema}"

        # UC Functions MCP server
        if self.functions:
            catalog: str = self.functions.catalog_name
            schema: str = self.functions.schema_name
            return f"{workspace_host}/api/2.0/mcp/functions/{catalog}/{schema}"

        raise ValueError(
            "No URL source configured. Provide one of: url, connection, genie_room, "
            "sql, vector_search, or functions"
        )

    @field_serializer("transport")
    def serialize_transport(self, value) -> str:
        if isinstance(value, TransportType):
            return value.value
        return str(value)

    @model_validator(mode="after")
    def validate_mutually_exclusive(self) -> "McpFunctionModel":
        """Validate that exactly one URL source is provided."""
        # Count how many URL sources are provided
        url_sources: list[tuple[str, Any]] = [
            ("url", self.url),
            ("connection", self.connection),
            ("genie_room", self.genie_room),
            ("sql", self.sql),
            ("vector_search", self.vector_search),
            ("functions", self.functions),
        ]

        provided_sources: list[str] = [
            name for name, value in url_sources if value is not None
        ]

        if self.transport == TransportType.STREAMABLE_HTTP:
            if len(provided_sources) == 0:
                raise ValueError(
                    "For STREAMABLE_HTTP transport, exactly one of the following must be provided: "
                    "url, connection, genie_room, sql, vector_search, or functions"
                )
            if len(provided_sources) > 1:
                raise ValueError(
                    f"For STREAMABLE_HTTP transport, only one URL source can be provided. "
                    f"Found: {', '.join(provided_sources)}. "
                    f"Please provide only one of: url, connection, genie_room, sql, vector_search, or functions"
                )

        if self.transport == TransportType.STDIO:
            if not self.command:
                raise ValueError("command must be provided for STDIO transport")
            if not self.args:
                raise ValueError("args must be provided for STDIO transport")

        return self

    @model_validator(mode="after")
    def update_url(self) -> "McpFunctionModel":
        self.url = value_of(self.url)
        return self

    @model_validator(mode="after")
    def update_headers(self) -> "McpFunctionModel":
        for key, value in self.headers.items():
            self.headers[key] = value_of(value)
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
    def validate_mutually_exclusive(self) -> Self:
        if self.alias and self.version:
            raise ValueError("Cannot specify both alias and version")
        return self


class GuardrailModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    model: str | LLMModel
    prompt: str | PromptModel
    num_retries: Optional[int] = 3

    @model_validator(mode="after")
    def validate_llm_model(self) -> Self:
        if isinstance(self.model, str):
            self.model = LLMModel(name=self.model)
        return self


class MiddlewareModel(BaseModel):
    """Configuration for middleware that can be applied to agents.

    Middleware is defined at the AppConfig level and can be referenced by name
    in agent configurations using YAML anchors for reusability.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Fully qualified name of the middleware factory function"
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the middleware factory function",
    )

    @model_validator(mode="after")
    def resolve_args(self) -> Self:
        """Resolve any variable references in args."""
        for key, value in self.args.items():
            self.args[key] = value_of(value)
        return self


class StorageType(str, Enum):
    POSTGRES = "postgres"
    MEMORY = "memory"
    LAKEBASE = "lakebase"


class CheckpointerModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    type: Optional[StorageType] = StorageType.MEMORY
    database: Optional[DatabaseModel] = None

    @model_validator(mode="after")
    def validate_storage_requires_database(self) -> Self:
        if (
            self.type in [StorageType.POSTGRES, StorageType.LAKEBASE]
            and not self.database
        ):
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
    def validate_postgres_requires_database(self) -> Self:
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
    """
    Configuration model for an agent in the DAO AI framework.

    Agents combine an LLM with tools and middleware to create systems that can
    reason about tasks, decide which tools to use, and iteratively work towards solutions.

    Middleware replaces the previous pre_agent_hook and post_agent_hook patterns,
    providing a more flexible and composable way to customize agent behavior.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    description: Optional[str] = None
    model: LLMModel
    tools: list[ToolModel] = Field(default_factory=list)
    guardrails: list[GuardrailModel] = Field(default_factory=list)
    prompt: Optional[str | PromptModel] = None
    handoff_prompt: Optional[str] = None
    middleware: list[MiddlewareModel] = Field(
        default_factory=list,
        description="List of middleware to apply to this agent",
    )

    def as_runnable(self) -> RunnableLike:
        from dao_ai.nodes import create_agent_node

        return create_agent_node(self)

    def as_responses_agent(self) -> ResponsesAgent:
        from dao_ai.models import create_responses_agent

        graph: CompiledStateGraph = self.as_runnable()
        return create_responses_agent(graph)


class SupervisorModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel
    tools: list[ToolModel] = Field(default_factory=list)
    prompt: Optional[str] = None
    middleware: list[MiddlewareModel] = Field(
        default_factory=list,
        description="List of middleware to apply to the supervisor",
    )


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
    def validate_mutually_exclusive(self) -> Self:
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
    principals: list[ServicePrincipalModel | str] = Field(default_factory=list)
    entitlements: list[Entitlement]

    @model_validator(mode="after")
    def resolve_principals(self) -> Self:
        """Resolve ServicePrincipalModel objects to their client_id."""
        resolved: list[str] = []
        for principal in self.principals:
            if isinstance(principal, ServicePrincipalModel):
                resolved.append(value_of(principal.client_id))
            else:
                resolved.append(principal)
        self.principals = resolved
        return self


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

    @model_validator(mode="after")
    def ensure_thread_id(self) -> "ChatPayload":
        """Ensure thread_id or conversation_id is present in configurable, generating UUID if needed."""
        import uuid

        if self.custom_inputs is None:
            self.custom_inputs = {}

        # Get or create configurable section
        configurable: dict[str, Any] = self.custom_inputs.get("configurable", {})

        # Check if thread_id or conversation_id exists
        has_thread_id = configurable.get("thread_id") is not None
        has_conversation_id = configurable.get("conversation_id") is not None

        # If neither is provided, generate a UUID for conversation_id
        if not has_thread_id and not has_conversation_id:
            configurable["conversation_id"] = str(uuid.uuid4())
            self.custom_inputs["configurable"] = configurable

        return self

    def as_messages(self) -> Sequence[BaseMessage]:
        return messages_from_dict(
            [{"type": m.role, "content": m.content} for m in self.messages]
        )

    def as_agent_request(self) -> ResponsesAgentRequest:
        from mlflow.types.responses_helpers import Message as _Message

        return ResponsesAgentRequest(
            input=[_Message(role=m.role, content=m.content) for m in self.messages],
            custom_inputs=self.custom_inputs,
        )


class ChatHistoryModel(BaseModel):
    """
    Configuration for chat history summarization.

    Attributes:
        model: The LLM to use for generating summaries.
        max_tokens: Maximum tokens to keep after summarization (the "keep" threshold).
            After summarization, recent messages totaling up to this many tokens are preserved.
        max_tokens_before_summary: Token threshold that triggers summarization.
            When conversation exceeds this, summarization runs. Mutually exclusive with
            max_messages_before_summary. If neither is set, defaults to max_tokens * 10.
        max_messages_before_summary: Message count threshold that triggers summarization.
            When conversation exceeds this many messages, summarization runs.
            Mutually exclusive with max_tokens_before_summary.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel
    max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum tokens to keep after summarization",
    )
    max_tokens_before_summary: Optional[int] = Field(
        default=None,
        gt=0,
        description="Token threshold that triggers summarization",
    )
    max_messages_before_summary: Optional[int] = Field(
        default=None,
        gt=0,
        description="Message count threshold that triggers summarization",
    )


class AppModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    service_principal: Optional[ServicePrincipalModel] = None
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
    input_example: Optional[ChatPayload] = None
    chat_history: Optional[ChatHistoryModel] = None
    code_paths: list[str] = Field(default_factory=list)
    pip_requirements: list[str] = Field(default_factory=list)
    python_version: Optional[str] = Field(
        default="3.12",
        description="Python version for Model Serving deployment. Defaults to 3.12 "
        "which is supported by Databricks Model Serving. This allows deploying from "
        "environments with different Python versions (e.g., Databricks Apps with 3.11).",
    )

    @model_validator(mode="after")
    def set_databricks_env_vars(self) -> Self:
        """Set Databricks environment variables for Model Serving.

        Sets DATABRICKS_HOST, DATABRICKS_CLIENT_ID, and DATABRICKS_CLIENT_SECRET.
        Values explicitly provided in environment_vars take precedence.
        """
        from dao_ai.utils import get_default_databricks_host

        # Set DATABRICKS_HOST if not already provided
        if "DATABRICKS_HOST" not in self.environment_vars:
            host: str | None = get_default_databricks_host()
            if host:
                self.environment_vars["DATABRICKS_HOST"] = host

        # Set service principal credentials if provided
        if self.service_principal is not None:
            if "DATABRICKS_CLIENT_ID" not in self.environment_vars:
                self.environment_vars["DATABRICKS_CLIENT_ID"] = (
                    self.service_principal.client_id
                )
            if "DATABRICKS_CLIENT_SECRET" not in self.environment_vars:
                self.environment_vars["DATABRICKS_CLIENT_SECRET"] = (
                    self.service_principal.client_secret
                )
        return self

    @model_validator(mode="after")
    def validate_agents_not_empty(self) -> Self:
        if not self.agents:
            raise ValueError("At least one agent must be specified")
        return self

    @model_validator(mode="after")
    def resolve_environment_vars(self) -> Self:
        for key, value in self.environment_vars.items():
            updated_value: str
            if isinstance(value, SecretVariableModel):
                updated_value = str(value)
            else:
                updated_value = value_of(value)

            self.environment_vars[key] = updated_value
        return self

    @model_validator(mode="after")
    def set_default_orchestration(self) -> Self:
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
    def set_default_endpoint_name(self) -> Self:
        if self.endpoint_name is None:
            self.endpoint_name = self.name
        return self

    @model_validator(mode="after")
    def set_default_agent(self) -> Self:
        default_agent_name: str = self.agents[0].name

        if self.orchestration.swarm and not self.orchestration.swarm.default_agent:
            self.orchestration.swarm.default_agent = default_agent_name

        return self

    @model_validator(mode="after")
    def add_code_paths_to_sys_path(self) -> Self:
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


class EvaluationDatasetExpectationsModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    expected_response: Optional[str] = None
    expected_facts: Optional[list[str]] = None

    @model_validator(mode="after")
    def validate_mutually_exclusive(self) -> Self:
        if self.expected_response is not None and self.expected_facts is not None:
            raise ValueError("Cannot specify both expected_response and expected_facts")
        return self


class EvaluationDatasetEntryModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    inputs: ChatPayload
    expectations: EvaluationDatasetExpectationsModel

    def to_mlflow_format(self) -> dict[str, Any]:
        """
        Convert to MLflow evaluation dataset format.

        Flattens the expectations fields to the top level alongside inputs,
        which is the format expected by MLflow's Correctness scorer.

        Returns:
            dict: Flattened dictionary with inputs and expectation fields at top level
        """
        result: dict[str, Any] = {"inputs": self.inputs.model_dump()}

        # Flatten expectations to top level for MLflow compatibility
        if self.expectations.expected_response is not None:
            result["expected_response"] = self.expectations.expected_response
        if self.expectations.expected_facts is not None:
            result["expected_facts"] = self.expectations.expected_facts

        return result


class EvaluationDatasetModel(BaseModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: str
    data: Optional[list[EvaluationDatasetEntryModel]] = Field(default_factory=list)
    overwrite: Optional[bool] = False

    def as_dataset(self, w: WorkspaceClient | None = None) -> EvaluationDataset:
        evaluation_dataset: EvaluationDataset
        needs_creation: bool = False

        try:
            evaluation_dataset = get_dataset(name=self.full_name)
            if self.overwrite:
                logger.warning(f"Overwriting dataset {self.full_name}")
                workspace_client: WorkspaceClient = w if w else WorkspaceClient()
                logger.debug(f"Dropping table: {self.full_name}")
                workspace_client.tables.delete(full_name=self.full_name)
                needs_creation = True
        except Exception:
            logger.warning(
                f"Dataset {self.full_name} not found, will create new dataset"
            )
            needs_creation = True

        # Create dataset if needed (either new or after overwrite)
        if needs_creation:
            evaluation_dataset = create_dataset(name=self.full_name)
            if self.data:
                logger.debug(
                    f"Merging {len(self.data)} entries into dataset {self.full_name}"
                )
                # Use to_mlflow_format() to flatten expectations for MLflow compatibility
                evaluation_dataset.merge_records(
                    [e.to_mlflow_format() for e in self.data]
                )

        return evaluation_dataset

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class PromptOptimizationModel(BaseModel):
    """Configuration for prompt optimization using GEPA.

    GEPA (Generative Evolution of Prompts and Agents) is an evolutionary
    optimizer that uses reflective mutation to improve prompts based on
    evaluation feedback.

    Example:
        prompt_optimization:
          name: optimize_my_prompt
          prompt: *my_prompt
          agent: *my_agent
          dataset: *my_training_dataset
          reflection_model: databricks-meta-llama-3-3-70b-instruct
          num_candidates: 50
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    prompt: Optional[PromptModel] = None
    agent: AgentModel
    dataset: EvaluationDatasetModel  # Training dataset with examples
    reflection_model: Optional[LLMModel | str] = None
    num_candidates: Optional[int] = 50

    def optimize(self, w: WorkspaceClient | None = None) -> PromptModel:
        """
        Optimize the prompt using GEPA.

        Args:
            w: Optional WorkspaceClient (not used, kept for API compatibility)

        Returns:
            PromptModel: The optimized prompt model
        """
        from dao_ai.optimization import OptimizationResult, optimize_prompt

        # Get reflection model name
        reflection_model_name: str | None = None
        if self.reflection_model:
            if isinstance(self.reflection_model, str):
                reflection_model_name = self.reflection_model
            else:
                reflection_model_name = self.reflection_model.uri

        # Ensure prompt is set
        prompt = self.prompt
        if prompt is None:
            raise ValueError(
                f"Prompt optimization '{self.name}' requires a prompt to be set"
            )

        result: OptimizationResult = optimize_prompt(
            prompt=prompt,
            agent=self.agent,
            dataset=self.dataset,
            reflection_model=reflection_model_name,
            num_candidates=self.num_candidates or 50,
            register_if_improved=True,
        )

        return result.optimized_prompt

    @model_validator(mode="after")
    def set_defaults(self) -> Self:
        # If no prompt is specified, try to use the agent's prompt
        if self.prompt is None:
            if isinstance(self.agent.prompt, PromptModel):
                self.prompt = self.agent.prompt
            else:
                raise ValueError(
                    f"Prompt optimization '{self.name}' requires either an explicit prompt "
                    f"or an agent with a prompt configured"
                )

        return self


class OptimizationsModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    training_datasets: dict[str, EvaluationDatasetModel] = Field(default_factory=dict)
    prompt_optimizations: dict[str, PromptOptimizationModel] = Field(
        default_factory=dict
    )

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
    service_principals: dict[str, ServicePrincipalModel] = Field(default_factory=dict)
    schemas: dict[str, SchemaModel] = Field(default_factory=dict)
    resources: Optional[ResourcesModel] = None
    retrievers: dict[str, RetrieverModel] = Field(default_factory=dict)
    tools: dict[str, ToolModel] = Field(default_factory=dict)
    guardrails: dict[str, GuardrailModel] = Field(default_factory=dict)
    middleware: dict[str, MiddlewareModel] = Field(default_factory=dict)
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
        vsc: "VectorSearchClient | None" = None,
        pat: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        workspace_host: str | None = None,
    ) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(
            w=w,
            vsc=vsc,
            pat=pat,
            client_id=client_id,
            client_secret=client_secret,
            workspace_host=workspace_host,
        )
        provider.create_agent(self)

    def deploy_agent(
        self,
        w: WorkspaceClient | None = None,
        vsc: "VectorSearchClient | None" = None,
        pat: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        workspace_host: str | None = None,
    ) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(
            w=w,
            vsc=vsc,
            pat=pat,
            client_id=client_id,
            client_secret=client_secret,
            workspace_host=workspace_host,
        )
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
