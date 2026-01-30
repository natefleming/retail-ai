import atexit
import importlib
import os
import sys
from abc import ABC, abstractmethod
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
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

if TYPE_CHECKING:
    from dao_ai.state import Context

from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import (
    CredentialsStrategy,
    ModelServingUserCredentials,
)
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.apps import App
from databricks.sdk.service.catalog import FunctionInfo, TableInfo
from databricks.sdk.service.dashboards import GenieSpace
from databricks.sdk.service.database import DatabaseInstance
from databricks.sdk.service.sql import GetWarehouseResponse
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex
from databricks_langchain import (
    ChatDatabricks,
    DatabricksEmbeddings,
    DatabricksFunctionClient,
)
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
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
    DatabricksApp,
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
    PrivateAttr,
    field_serializer,
    model_validator,
)

from dao_ai.utils import normalize_name


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
    def validate_value(self) -> Self:
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
       calling user's identity. Implementation varies by deployment:
       - Databricks Apps: Uses X-Forwarded-Access-Token from request headers
       - Model Serving: Uses ModelServingUserCredentials

    2. **Service Principal (OAuth M2M)**: Provide service_principal or
       (client_id + client_secret + workspace_host) for service principal auth.

    3. **Personal Access Token (PAT)**: Provide pat (and optionally workspace_host)
       to authenticate with a personal access token.

    4. **Ambient Authentication**: If no credentials provided, uses SDK defaults
       (environment variables, notebook context, etc.)

    Authentication Priority:
    1. OBO (on_behalf_of_user=True)
       - Checks for forwarded headers (Databricks Apps)
       - Falls back to ModelServingUserCredentials (Model Serving)
    2. Service Principal (client_id + client_secret + workspace_host)
    3. PAT (pat + workspace_host)
    4. Ambient/default authentication

    Note: When on_behalf_of_user=True, the agent acts as the calling user regardless
    of deployment target. In Databricks Apps, this uses X-Forwarded-Access-Token
    automatically captured by MLflow AgentServer. In Model Serving, this uses
    ModelServingUserCredentials. Forwarded headers are ONLY used when
    on_behalf_of_user=True.
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

        A new client is created on each access.

        Authentication priority:
        1. On-Behalf-Of User (on_behalf_of_user=True):
           - Uses ModelServingUserCredentials (Model Serving)
           - For Databricks Apps with headers, use workspace_client_from(context)
        2. Service Principal (client_id + client_secret + workspace_host)
        3. PAT (pat + workspace_host)
        4. Ambient/default authentication
        """
        from dao_ai.utils import normalize_host

        # Check for OBO first (highest priority)
        if self.on_behalf_of_user:
            credentials_strategy: CredentialsStrategy = ModelServingUserCredentials()
            logger.debug(
                f"Creating WorkspaceClient for {self.__class__.__name__} "
                f"with OBO credentials strategy (Model Serving)"
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

        if client_id_value and client_secret_value:
            # If workspace_host is not provided, check DATABRICKS_HOST env var first,
            # then fall back to WorkspaceClient().config.host
            if not workspace_host_value:
                workspace_host_value = os.getenv("DATABRICKS_HOST")
                if not workspace_host_value:
                    workspace_host_value = WorkspaceClient().config.host

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

    def workspace_client_from(self, context: "Context | None") -> WorkspaceClient:
        """
        Get a WorkspaceClient using headers from the provided Context.

        Use this method from tools that have access to ToolRuntime[Context].
        This allows OBO authentication to work in Databricks Apps where headers
        are captured at request entry and passed through the Context.

        Args:
            context: Runtime context containing headers for OBO auth.
                     If None or no headers, falls back to workspace_client property.

        Returns:
            WorkspaceClient configured with appropriate authentication.
        """
        from dao_ai.utils import normalize_host

        logger.trace(
            "workspace_client_from called",
            context=context,
            on_behalf_of_user=self.on_behalf_of_user,
        )

        # Check if we have headers in context for OBO
        if context and context.headers and self.on_behalf_of_user:
            headers = context.headers
            # Try both lowercase and title-case header names (HTTP headers are case-insensitive)
            forwarded_token: str = headers.get(
                "x-forwarded-access-token"
            ) or headers.get("X-Forwarded-Access-Token")

            if forwarded_token:
                forwarded_user = headers.get("x-forwarded-user") or headers.get(
                    "X-Forwarded-User", "unknown"
                )
                logger.debug(
                    f"Creating WorkspaceClient for {self.__class__.__name__} "
                    f"with OBO using forwarded token from Context",
                    forwarded_user=forwarded_user,
                )
                # Use workspace_host if configured, otherwise SDK will auto-detect
                workspace_host_value: str | None = (
                    normalize_host(value_of(self.workspace_host))
                    if self.workspace_host
                    else None
                )
                return WorkspaceClient(
                    host=workspace_host_value,
                    token=forwarded_token,
                    auth_type="pat",
                )

        # Fall back to existing workspace_client property
        return self.workspace_client


class DeploymentTarget(str, Enum):
    """Target platform for agent deployment."""

    MODEL_SERVING = "model_serving"
    """Deploy to Databricks Model Serving endpoint."""

    APPS = "apps"
    """Deploy as a Databricks App."""


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
    catalog_name: AnyVariable
    schema_name: AnyVariable
    permissions: Optional[list[PermissionModel]] = Field(default_factory=list)

    @model_validator(mode="after")
    def resolve_variables(self) -> Self:
        """Resolve AnyVariable fields to their actual string values."""
        self.catalog_name = value_of(self.catalog_name)
        self.schema_name = value_of(self.schema_name)
        return self

    @property
    def full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}"

    def create(self, w: WorkspaceClient | None = None) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w)
        provider.create_schema(self)


class DatabricksAppModel(IsDatabricksResource, HasFullName):
    """
    Configuration for a Databricks App resource.

    The `name` is the unique instance name of the Databricks App within the workspace.
    The `url` is dynamically retrieved from the workspace client by calling
    `apps.get(name)` and returning the app's URL.

    Example:
        ```yaml
        resources:
          apps:
            my_app:
              name: my-databricks-app
        ```
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    """The unique instance name of the Databricks App in the workspace."""

    @property
    def url(self) -> str:
        """
        Retrieve the URL of the Databricks App from the workspace.

        Returns:
            The URL of the deployed Databricks App.

        Raises:
            RuntimeError: If the app is not found or URL is not available.
        """
        app: App = self.workspace_client.apps.get(self.name)
        if app.url is None:
            raise RuntimeError(
                f"Databricks App '{self.name}' does not have a URL. "
                "The app may not be deployed yet."
            )
        return app.url

    @property
    def full_name(self) -> str:
        return self.name

    @property
    def api_scopes(self) -> Sequence[str]:
        return ["apps.apps"]

    def as_resources(self) -> Sequence[DatabricksResource]:
        return [
            DatabricksApp(app_name=self.name, on_behalf_of_user=self.on_behalf_of_user)
        ]


class TableModel(IsDatabricksResource, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: Optional[str] = None

    @model_validator(mode="after")
    def validate_name_or_schema_required(self) -> Self:
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

    def exists(self) -> bool:
        """Check if the table exists in Unity Catalog.

        Returns:
            True if the table exists, False otherwise.
        """
        try:
            self.workspace_client.tables.get(full_name=self.full_name)
            return True
        except NotFound:
            logger.debug(f"Table not found: {self.full_name}")
            return False
        except Exception as e:
            logger.warning(f"Error checking table existence for {self.full_name}: {e}")
            return False

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
    description: Optional[str] = None
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
    """Model representing a Databricks Vector Search index."""

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

    def exists(self) -> bool:
        """Check if this vector search index exists.

        Returns:
            True if the index exists, False otherwise.
        """
        try:
            self.workspace_client.vector_search_indexes.get_index(self.full_name)
            return True
        except NotFound:
            logger.debug(f"Index not found: {self.full_name}")
            return False
        except Exception as e:
            logger.warning(f"Error checking index existence for {self.full_name}: {e}")
            return False


class FunctionModel(IsDatabricksResource, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(default=None, alias="schema")
    name: Optional[str] = None

    @model_validator(mode="after")
    def validate_name_or_schema_required(self) -> Self:
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

    def exists(self) -> bool:
        """Check if the function exists in Unity Catalog.

        Returns:
            True if the function exists, False otherwise.
        """
        try:
            self.workspace_client.functions.get(name=self.full_name)
            return True
        except NotFound:
            logger.debug(f"Function not found: {self.full_name}")
            return False
        except Exception as e:
            logger.warning(
                f"Error checking function existence for {self.full_name}: {e}"
            )
            return False

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


class WarehouseModel(IsDatabricksResource):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: Optional[str] = None
    description: Optional[str] = None
    warehouse_id: AnyVariable

    _warehouse_details: Optional[GetWarehouseResponse] = PrivateAttr(default=None)

    def _get_warehouse_details(self) -> GetWarehouseResponse:
        if self._warehouse_details is None:
            self._warehouse_details = self.workspace_client.warehouses.get(
                id=value_of(self.warehouse_id)
            )
        return self._warehouse_details

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

    @model_validator(mode="after")
    def populate_name(self) -> Self:
        """Populate name from warehouse details if not provided."""
        if self.warehouse_id and not self.name:
            try:
                warehouse_details = self._get_warehouse_details()
                if warehouse_details.name:
                    self.name = warehouse_details.name
            except Exception as e:
                logger.debug(f"Could not fetch details from warehouse: {e}")
        return self


class GenieRoomModel(IsDatabricksResource):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: Optional[str] = None
    description: Optional[str] = None
    space_id: AnyVariable

    _space_details: Optional[GenieSpace] = PrivateAttr(default=None)

    def _get_space_details(self) -> GenieSpace:
        if self._space_details is None:
            self._space_details = self.workspace_client.genie.get_space(
                space_id=self.space_id, include_serialized_space=True
            )
        return self._space_details

    def _parse_serialized_space(self) -> dict[str, Any]:
        """Parse the serialized_space JSON string and return the parsed data."""
        import json

        space_details = self._get_space_details()
        if not space_details.serialized_space:
            return {}

        try:
            return json.loads(space_details.serialized_space)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse serialized_space: {e}")
            return {}

    @property
    def warehouse(self) -> Optional[WarehouseModel]:
        """Extract warehouse information from the Genie space.

        Returns:
            WarehouseModel instance if warehouse_id is available, None otherwise.
        """
        space_details: GenieSpace = self._get_space_details()

        if not space_details.warehouse_id:
            return None

        try:
            response: GetWarehouseResponse = self.workspace_client.warehouses.get(
                space_details.warehouse_id
            )
            warehouse_name: str = response.name or space_details.warehouse_id

            warehouse_model = WarehouseModel(
                name=warehouse_name,
                warehouse_id=space_details.warehouse_id,
                on_behalf_of_user=self.on_behalf_of_user,
                service_principal=self.service_principal,
                client_id=self.client_id,
                client_secret=self.client_secret,
                workspace_host=self.workspace_host,
                pat=self.pat,
            )

            return warehouse_model
        except Exception as e:
            logger.warning(
                f"Failed to fetch warehouse details for {space_details.warehouse_id}: {e}"
            )
            return None

    @property
    def tables(self) -> list[TableModel]:
        """Extract tables from the serialized Genie space.

        Databricks Genie stores tables in: data_sources.tables[].identifier
        Only includes tables that actually exist in Unity Catalog.
        """
        parsed_space = self._parse_serialized_space()
        tables_list: list[TableModel] = []

        # Primary structure: data_sources.tables with 'identifier' field
        if "data_sources" in parsed_space:
            data_sources = parsed_space["data_sources"]
            if isinstance(data_sources, dict) and "tables" in data_sources:
                tables_data = data_sources["tables"]
                if isinstance(tables_data, list):
                    for table_item in tables_data:
                        table_name: str | None = None
                        if isinstance(table_item, dict):
                            # Standard Databricks structure uses 'identifier'
                            table_name = table_item.get("identifier") or table_item.get(
                                "name"
                            )
                        elif isinstance(table_item, str):
                            table_name = table_item

                        if table_name:
                            table_model = TableModel(
                                name=table_name,
                                on_behalf_of_user=self.on_behalf_of_user,
                                service_principal=self.service_principal,
                                client_id=self.client_id,
                                client_secret=self.client_secret,
                                workspace_host=self.workspace_host,
                                pat=self.pat,
                            )

                            # Verify the table exists before adding
                            if not table_model.exists():
                                continue

                            tables_list.append(table_model)

        return tables_list

    @property
    def functions(self) -> list[FunctionModel]:
        """Extract functions from the serialized Genie space.

        Databricks Genie stores functions in multiple locations:
        - instructions.sql_functions[].identifier (SQL functions)
        - data_sources.functions[].identifier (other functions)
        Only includes functions that actually exist in Unity Catalog.
        """
        parsed_space = self._parse_serialized_space()
        functions_list: list[FunctionModel] = []
        seen_functions: set[str] = set()

        def add_function_if_exists(function_name: str) -> None:
            """Helper to add a function if it exists and hasn't been added."""
            if function_name in seen_functions:
                return

            seen_functions.add(function_name)
            function_model = FunctionModel(
                name=function_name,
                on_behalf_of_user=self.on_behalf_of_user,
                service_principal=self.service_principal,
                client_id=self.client_id,
                client_secret=self.client_secret,
                workspace_host=self.workspace_host,
                pat=self.pat,
            )

            # Verify the function exists before adding
            if not function_model.exists():
                return

            functions_list.append(function_model)

        # Primary structure: instructions.sql_functions with 'identifier' field
        if "instructions" in parsed_space:
            instructions = parsed_space["instructions"]
            if isinstance(instructions, dict) and "sql_functions" in instructions:
                sql_functions_data = instructions["sql_functions"]
                if isinstance(sql_functions_data, list):
                    for function_item in sql_functions_data:
                        if isinstance(function_item, dict):
                            # SQL functions use 'identifier' field
                            function_name = function_item.get(
                                "identifier"
                            ) or function_item.get("name")
                            if function_name:
                                add_function_if_exists(function_name)

        # Secondary structure: data_sources.functions with 'identifier' field
        if "data_sources" in parsed_space:
            data_sources = parsed_space["data_sources"]
            if isinstance(data_sources, dict) and "functions" in data_sources:
                functions_data = data_sources["functions"]
                if isinstance(functions_data, list):
                    for function_item in functions_data:
                        function_name: str | None = None
                        if isinstance(function_item, dict):
                            # Standard Databricks structure uses 'identifier'
                            function_name = function_item.get(
                                "identifier"
                            ) or function_item.get("name")
                        elif isinstance(function_item, str):
                            function_name = function_item

                        if function_name:
                            add_function_if_exists(function_name)

        return functions_list

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

    @model_validator(mode="after")
    def populate_name_and_description(self) -> Self:
        """Populate name and description from GenieSpace if not provided."""
        if self.space_id and (not self.name or not self.description):
            try:
                space_details = self._get_space_details()
                if not self.name and space_details.title:
                    self.name = space_details.title
                if not self.description and space_details.description:
                    self.description = space_details.description
            except Exception as e:
                logger.debug(f"Could not fetch details from Genie space: {e}")
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
    def validate_path_or_volume(self) -> Self:
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
    """
    Configuration model for a Databricks Vector Search store.

    Supports two modes:
    1. **Use Existing Index**: Provide only `index` (fully qualified name).
       Used for querying an existing vector search index at runtime.
    2. **Provisioning Mode**: Provide `source_table` + `embedding_source_column`.
       Used for creating a new vector search index.

    Examples:
        Minimal configuration (use existing index):
        ```yaml
        vector_stores:
          products_search:
            index:
              name: catalog.schema.my_index
        ```

        Full provisioning configuration:
        ```yaml
        vector_stores:
          products_search:
            source_table:
              schema: *my_schema
              name: products
            embedding_source_column: description
            endpoint:
              name: my_endpoint
        ```
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # RUNTIME: Only index is truly required for querying existing indexes
    index: Optional[IndexModel] = None

    # PROVISIONING ONLY: Required when creating a new index
    source_table: Optional[TableModel] = None
    embedding_source_column: Optional[str] = None
    embedding_model: Optional[LLMModel] = None
    endpoint: Optional[VectorSearchEndpoint] = None

    # OPTIONAL: For both modes
    source_path: Optional[VolumePathModel] = None
    checkpoint_path: Optional[VolumePathModel] = None
    primary_key: Optional[str] = None
    columns: Optional[list[str]] = Field(default_factory=list)
    doc_uri: Optional[str] = None

    @model_validator(mode="after")
    def validate_configuration_mode(self) -> Self:
        """
        Validate that configuration is valid for either:
        - Use existing mode: index is provided
        - Provisioning mode: source_table + embedding_source_column provided
        """
        has_index = self.index is not None
        has_source_table = self.source_table is not None
        has_embedding_col = self.embedding_source_column is not None

        # Must have at least index OR source_table
        if not has_index and not has_source_table:
            raise ValueError(
                "Either 'index' (for existing indexes) or 'source_table' "
                "(for provisioning) must be provided"
            )

        # If provisioning mode, need embedding_source_column
        if has_source_table and not has_embedding_col:
            raise ValueError(
                "embedding_source_column is required when source_table is provided (provisioning mode)"
            )

        return self

    @model_validator(mode="after")
    def set_default_embedding_model(self) -> Self:
        # Only set default embedding model in provisioning mode
        if self.source_table is not None and not self.embedding_model:
            self.embedding_model = LLMModel(name="databricks-gte-large-en")
        return self

    @model_validator(mode="after")
    def set_default_primary_key(self) -> Self:
        # Only auto-discover primary key in provisioning mode
        if self.primary_key is None and self.source_table is not None:
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
        # Only generate index from source_table in provisioning mode
        if self.index is None and self.source_table is not None:
            name: str = f"{self.source_table.name}_index"
            self.index = IndexModel(schema=self.source_table.schema_model, name=name)
        return self

    @model_validator(mode="after")
    def set_default_endpoint(self) -> Self:
        # Only find/create endpoint in provisioning mode
        if self.endpoint is None and self.source_table is not None:
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
        from dao_ai.providers.databricks import DatabricksProvider

        provider: DatabricksProvider = DatabricksProvider(vsc=vsc)
        index: VectorSearchIndex = provider.get_vector_index(self)
        return index

    def create(self, vsc: VectorSearchClient | None = None) -> None:
        """
        Create or validate the vector search index.

        Behavior depends on configuration mode:
        - **Provisioning Mode** (source_table provided): Creates the index
        - **Use Existing Mode** (only index provided): Validates the index exists

        Args:
            vsc: Optional VectorSearchClient instance

        Raises:
            ValueError: If configuration is invalid or index doesn't exist
        """
        from dao_ai.providers.databricks import DatabricksProvider

        provider: DatabricksProvider = DatabricksProvider(vsc=vsc)

        if self.source_table is not None:
            self._create_new_index(provider)
        else:
            self._validate_existing_index(provider)

    def _validate_existing_index(self, provider: Any) -> None:
        """Validate that an existing index is accessible."""
        if self.index is None:
            raise ValueError("index is required for 'use existing' mode")

        if self.index.exists():
            logger.info(
                "Vector search index exists and ready",
                index_name=self.index.full_name,
            )
        else:
            raise ValueError(
                f"Index '{self.index.full_name}' does not exist. "
                "Provide 'source_table' to provision it."
            )

    def _create_new_index(self, provider: Any) -> None:
        """Create a new vector search index from source table."""
        if self.embedding_source_column is None:
            raise ValueError("embedding_source_column is required for provisioning")
        if self.endpoint is None:
            raise ValueError("endpoint is required for provisioning")
        if self.index is None:
            raise ValueError("index is required for provisioning")

        provider.create_vector_store(self)


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


class DatabaseModel(IsDatabricksResource):
    """
    Configuration for database connections supporting both Databricks Lakebase and standard PostgreSQL.

    Authentication is inherited from IsDatabricksResource. Additionally supports:
    - user/password: For user-based database authentication

    Connection Types (determined by fields provided):
    - Databricks Lakebase: Provide `instance_name` (authentication optional, supports ambient auth)
    - Standard PostgreSQL: Provide `host` (authentication required via user/password)

    Note: For Lakebase connections, `name` is optional and defaults to `instance_name`.
    For PostgreSQL connections, `name` is required.

    Example Databricks Lakebase (minimal):
    ```yaml
    databases:
      my_lakebase:
        instance_name: my-lakebase-instance  # name defaults to instance_name
    ```

    Example Databricks Lakebase with Service Principal:
    ```yaml
    databases:
      my_lakebase:
        instance_name: my-lakebase-instance
        service_principal:
          client_id:
            env: SERVICE_PRINCIPAL_CLIENT_ID
          client_secret:
            scope: my-scope
            secret: sp-client-secret
        workspace_host:
          env: DATABRICKS_HOST
    ```

    Example Databricks Lakebase with Ambient Authentication:
    ```yaml
    databases:
      my_lakebase:
        instance_name: my-lakebase-instance
        on_behalf_of_user: true
    ```

    Example Standard PostgreSQL:
    ```yaml
    databases:
      my_postgres:
        name: my-database
        host: my-postgres-host.example.com
        port: 5432
        database: my_db
        user: my_user
        password:
          env: PGPASSWORD
    ```
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: Optional[str] = None
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

    @property
    def api_scopes(self) -> Sequence[str]:
        return ["database.database-instances"]

    @property
    def is_lakebase(self) -> bool:
        """Returns True if this is a Databricks Lakebase connection (instance_name provided)."""
        return self.instance_name is not None

    def as_resources(self) -> Sequence[DatabricksResource]:
        if self.is_lakebase:
            return [
                DatabricksLakebase(
                    database_instance_name=self.instance_name,
                    on_behalf_of_user=self.on_behalf_of_user,
                )
            ]
        return []

    @model_validator(mode="after")
    def validate_connection_type(self) -> Self:
        """Validate connection configuration based on type.

        - If instance_name is provided: Databricks Lakebase connection
          (host is optional - will be fetched from API if not provided)
        - If only host is provided: Standard PostgreSQL connection
          (must not have instance_name)
        """
        if not self.instance_name and not self.host:
            raise ValueError(
                "Either instance_name (Databricks Lakebase) or host (PostgreSQL) must be provided."
            )
        return self

    @model_validator(mode="after")
    def populate_name_from_instance_name(self) -> Self:
        """Populate name from instance_name if not provided for Lakebase connections."""
        if self.name is None and self.instance_name:
            self.name = self.instance_name
        elif self.name is None:
            raise ValueError(
                "Either 'name' or 'instance_name' must be provided for DatabaseModel."
            )
        return self

    @model_validator(mode="after")
    def update_user(self) -> Self:
        # Skip if using OBO (passive auth), explicit credentials, or explicit user
        if self.on_behalf_of_user or self.client_id or self.user or self.pat:
            return self

        # For standard PostgreSQL, we need explicit user credentials
        # For Lakebase with no auth, ambient auth is allowed
        if not self.is_lakebase:
            # Standard PostgreSQL - try to determine current user for local development
            try:
                self.user = self.workspace_client.current_user.me().user_name
            except Exception as e:
                logger.warning(
                    f"Could not determine current user for PostgreSQL database: {e}. "
                    f"Please provide explicit user credentials."
                )
        else:
            # For Lakebase, try to determine current user but don't fail if we can't
            try:
                self.user = self.workspace_client.current_user.me().user_name
            except Exception:
                # If we can't determine user and no explicit auth, that's okay
                # for Lakebase with ambient auth - credentials will be injected at runtime
                pass

        return self

    @model_validator(mode="after")
    def update_host(self) -> Self:
        # Lakebase uses instance_name directly via databricks_langchain - host not needed
        if self.is_lakebase:
            return self

        # For standard PostgreSQL, host must be provided by the user
        # (enforced by validate_connection_type)
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

        # For standard PostgreSQL (host-based), at least one auth method must be configured
        # For Lakebase (instance_name-based), auth is optional (supports ambient authentication)
        if not self.is_lakebase and auth_methods_count == 0:
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

        For Lakebase: Uses Databricks-generated credentials (token-based auth).
        For standard PostgreSQL: Uses provided user/password credentials.
        """
        import uuid as _uuid

        from databricks.sdk.service.database import DatabaseCredential

        host: str
        port: int
        database: str
        username: str | None = None
        password_value: str | None = None

        # Resolve host - fetch from API at runtime for Lakebase if not provided
        host_value: Any = self.host
        if host_value is None and self.is_lakebase:
            # Fetch host from Lakebase instance API
            existing_instance: DatabaseInstance = (
                self.workspace_client.database.get_database_instance(
                    name=self.instance_name
                )
            )
            host_value = existing_instance.read_write_dns

        if host_value is None:
            instance_or_name = self.instance_name if self.is_lakebase else self.name
            raise ValueError(
                f"Database host not configured for {instance_or_name}. "
                "Please provide 'host' explicitly."
            )

        host = value_of(host_value)
        port = value_of(self.port)
        database = value_of(self.database)

        if self.is_lakebase:
            # Lakebase: Use Databricks-generated credentials
            if self.client_id and self.client_secret and self.workspace_host:
                username = value_of(self.client_id)
            elif self.user:
                username = value_of(self.user)
            # For OBO mode, no username is needed - the token identity is used

            # Generate Databricks database credential (token)
            w: WorkspaceClient = self.workspace_client
            cred: DatabaseCredential = w.database.generate_database_credential(
                request_id=str(_uuid.uuid4()),
                instance_names=[self.instance_name],
            )
            password_value = cred.token
        else:
            # Standard PostgreSQL: Use provided credentials
            if self.user:
                username = value_of(self.user)
            if self.password:
                password_value = value_of(self.password)

            if not username or not password_value:
                raise ValueError(
                    f"Standard PostgreSQL databases require both 'user' and 'password'. "
                    f"Database: {self.name}"
                )

        # Build connection parameters dictionary
        params: dict[str, Any] = {
            "dbname": database,
            "host": host,
            "port": port,
            "password": password_value,
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


# Memory estimation for capacity planning:
# - Each entry: ~20KB (8KB question embedding + 8KB context embedding + 4KB strings/overhead)
# - 1,000 entries: ~20MB (0.4% of 8GB)
# - 5,000 entries: ~100MB (2% of 8GB)
# - 10,000 entries: ~200MB (4-5% of 8GB) - default for ~30 users
# - 20,000 entries: ~400MB (8-10% of 8GB)
# Default 10,000 entries provides ~330 queries per user for 30 users.
class GenieInMemorySemanticCacheParametersModel(BaseModel):
    """
    Configuration for in-memory semantic cache (no database required).

    This cache stores embeddings and cache entries entirely in memory, providing
    semantic similarity matching without requiring external database dependencies
    like PostgreSQL or Databricks Lakebase.

    Default settings are tuned for ~30 users on an 8GB machine:
    - Capacity: 10,000 entries (~200MB memory, ~330 queries per user)
    - Eviction: LRU (Least Recently Used) - keeps frequently accessed queries
    - TTL: 1 week (accommodates weekly work patterns and batch jobs)
    - Memory overhead: ~4-5% of 8GB system

    The LRU eviction strategy ensures hot queries stay cached while cold queries
    are evicted, providing better hit rates than FIFO eviction.

    For larger deployments or memory-constrained environments, adjust capacity and TTL accordingly.

    Use this when:
    - No external database access is available
    - Single-instance deployments (cache not shared across instances)
    - Cache persistence across restarts is not required
    - Cache sizes are moderate (hundreds to low thousands of entries)

    For multi-instance deployments or large cache sizes, use GenieSemanticCacheParametersModel
    with PostgreSQL backend instead.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    time_to_live_seconds: int | None = (
        60 * 60 * 24 * 7
    )  # 1 week default (604800 seconds), None or negative = never expires
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
    warehouse: WarehouseModel
    capacity: int | None = (
        10000  # Maximum cache entries. ~200MB for 10000 entries (1024-dim embeddings). LRU eviction when full. None = unlimited (not recommended for production).
    )
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


class InstructionAwareRerankModel(BaseModel):
    """
    LLM-based reranking considering user instructions and constraints.

    Use fast models (GPT-3.5, Haiku, Llama 3 8B) to minimize latency (~100ms).
    Runs AFTER FlashRank as an additional constraint-aware reranking stage.
    Skipped for 'standard' mode when auto_bypass=true in router config.

    Example:
        ```yaml
        rerank:
          model: ms-marco-MiniLM-L-12-v2
          top_n: 20
          instruction_aware:
            model: *fast_llm
            instructions: |
              Prioritize results matching price and brand constraints.
            top_n: 10
        ```
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    model: Optional["LLMModel"] = Field(
        default=None,
        description="LLM for instruction reranking (fast model recommended)",
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Custom reranking instructions for constraint prioritization",
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Number of documents to return after instruction reranking",
    )


class RerankParametersModel(BaseModel):
    """
    Configuration for reranking retrieved documents.

    Supports three reranking options that can be combined:
    1. FlashRank (local cross-encoder) - set `model`
    2. Databricks server-side reranking - set `columns`
    3. LLM instruction-aware reranking - set `instruction_aware`

    Example with Databricks columns + instruction-aware (no FlashRank):
        ```yaml
        rerank:
          columns:                    # Databricks server-side reranking
            - product_name
            - brand_name
          instruction_aware:          # LLM-based constraint reranking
            model: *fast_llm
            instructions: "Prioritize by brand preferences"
            top_n: 10
        ```

    Example with FlashRank:
        ```yaml
        rerank:
          model: ms-marco-MiniLM-L-12-v2  # FlashRank model
          top_n: 10
        ```

    Available FlashRank models (see https://github.com/PrithivirajDamodaran/FlashRank):
    - "ms-marco-TinyBERT-L-2-v2" (~4MB, fastest)
    - "ms-marco-MiniLM-L-12-v2" (~34MB, best cross-encoder)
    - "rank-T5-flan" (~110MB, best non cross-encoder)
    - "ms-marco-MultiBERT-L-12" (~150MB, multilingual 100+ languages)
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    model: Optional[str] = Field(
        default=None,
        description="FlashRank model name. If None, FlashRank is not used (use columns for Databricks reranking).",
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Number of documents to return after reranking. If None, uses search_parameters.num_results.",
    )
    cache_dir: Optional[str] = Field(
        default="~/.dao_ai/cache/flashrank",
        description="Directory to cache downloaded model weights. Supports tilde expansion (e.g., ~/.dao_ai).",
    )
    columns: Optional[list[str]] = Field(
        default_factory=list, description="Columns to rerank using DatabricksReranker"
    )
    instruction_aware: Optional[InstructionAwareRerankModel] = Field(
        default=None,
        description="Optional LLM-based reranking stage after FlashRank",
    )


class FilterItem(BaseModel):
    """A metadata filter for vector search.

    Filters constrain search results by matching column values.
    Use column names from the provided schema description.
    """

    model_config = ConfigDict(extra="forbid")
    key: str = Field(
        description=(
            "Column name with optional operator suffix. "
            "Operators: (none) for equality, NOT for exclusion, "
            "< <= > >= for numeric comparison, "
            "LIKE for token match, NOT LIKE to exclude tokens."
        )
    )
    value: Union[str, int, float, bool, list[Union[str, int, float, bool]]] = Field(
        description=(
            "The filter value matching the column type. "
            "Use an array for IN-style matching multiple values."
        )
    )


class SearchQuery(BaseModel):
    """A single search query with optional metadata filters.

    Represents one focused search intent extracted from the user's request.
    The text should be a natural language query optimized for semantic search.
    Filters constrain results to match specific metadata values.
    """

    model_config = ConfigDict(extra="forbid")
    text: str = Field(
        description=(
            "Natural language search query text optimized for semantic similarity. "
            "Should be focused on a single search intent. "
            "Do NOT include filter criteria in the text; use the filters field instead."
        )
    )
    filters: Optional[list[FilterItem]] = Field(
        default=None,
        description=(
            "Metadata filters to constrain search results. "
            "Set to null if no filters apply. "
            "Extract filter values from explicit constraints in the user query."
        ),
    )


class DecomposedQueries(BaseModel):
    """Decomposed search queries extracted from a user request.

    Break down complex user queries into multiple focused search queries.
    Each query targets a distinct search intent with appropriate filters.
    Generate 1-3 queries depending on the complexity of the user request.
    """

    model_config = ConfigDict(extra="forbid")
    queries: list[SearchQuery] = Field(
        description=(
            "List of search queries extracted from the user request. "
            "Each query should target a distinct search intent. "
            "Order queries by importance, with the most relevant first."
        )
    )


class ColumnInfo(BaseModel):
    """Column metadata for dynamic schema generation in structured output.

    When provided, column information is embedded directly into the JSON schema
    that with_structured_output sends to the LLM, improving filter accuracy.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Column name as it appears in the database")
    type: Literal["string", "number", "boolean", "datetime"] = Field(
        default="string",
        description="Column data type for value validation",
    )
    operators: list[str] = Field(
        default=["", "NOT", "<", "<=", ">", ">=", "LIKE", "NOT LIKE"],
        description="Valid filter operators for this column",
    )


class InstructedRetrieverModel(BaseModel):
    """
    Configuration for instructed retrieval with query decomposition and RRF merging.

    Instructed retrieval decomposes user queries into multiple subqueries with
    metadata filters, executes them in parallel, and merges results using
    Reciprocal Rank Fusion (RRF) before reranking.

    Example:
        ```yaml
        retriever:
          vector_store: *products_vector_store
          instructed:
            decomposition_model: *fast_llm
            schema_description: |
              Products table: product_id, brand_name, category, price, updated_at
              Filter operators: {"col": val}, {"col >": val}, {"col NOT": val}
            columns:
              - name: brand_name
                type: string
              - name: price
                type: number
                operators: ["", "<", "<=", ">", ">="]
            constraints:
              - "Prefer recent products"
            max_subqueries: 3
            examples:
              - query: "cheap drills"
                filters: {"price <": 100}
        ```
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    decomposition_model: Optional["LLMModel"] = Field(
        default=None,
        description="LLM for query decomposition (smaller/faster model recommended)",
    )
    schema_description: str = Field(
        description="Column names, types, and valid filter syntax for the LLM"
    )
    columns: Optional[list[ColumnInfo]] = Field(
        default=None,
        description=(
            "Structured column info for dynamic schema generation. "
            "When provided, column names are embedded in the JSON schema for better LLM accuracy."
        ),
    )
    constraints: Optional[list[str]] = Field(
        default=None, description="Default constraints to always apply"
    )
    max_subqueries: int = Field(
        default=3, description="Maximum number of parallel subqueries"
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant (lower values weight top ranks more heavily)",
    )
    examples: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Few-shot examples for domain-specific filter translation",
    )
    normalize_filter_case: Optional[Literal["uppercase", "lowercase"]] = Field(
        default=None,
        description="Auto-normalize filter string values to uppercase or lowercase",
    )


class RouterModel(BaseModel):
    """
    Select internal execution mode based on query characteristics.

    Use fast models (GPT-3.5, Haiku, Llama 3 8B) to minimize latency (~50-100ms).
    Routes to internal modes within the same retriever, not external retrievers.
    Cross-index routing belongs at the agent/tool-selection level.

    Execution Modes:
    - "standard": Single similarity_search() for simple keyword/product searches
    - "instructed": Decompose -> Parallel Search -> RRF for constrained queries

    Example:
        ```yaml
        retriever:
          router:
            model: *fast_llm
            default_mode: standard
            auto_bypass: true
        ```
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    model: Optional["LLMModel"] = Field(
        default=None,
        description="LLM for routing decision (fast model recommended)",
    )
    default_mode: Literal["standard", "instructed"] = Field(
        default="standard",
        description="Fallback mode if routing fails",
    )
    auto_bypass: bool = Field(
        default=True,
        description="Skip Instruction Reranker and Verifier for standard mode",
    )


class VerificationResult(BaseModel):
    """Verification of whether search results satisfy the user's constraints.

    Analyze the retrieved results against the original query and any explicit
    constraints to determine if a retry with modified filters is needed.
    """

    model_config = ConfigDict(extra="forbid")

    passed: bool = Field(
        description="True if results satisfy the user's query intent and constraints."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the verification decision, from 0.0 (uncertain) to 1.0 (certain).",
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Explanation of why verification passed or failed. Include specific issues found.",
    )
    suggested_filter_relaxation: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Suggested filter modifications for retry. "
            "Keys are column names, values indicate changes (e.g., 'REMOVE', 'WIDEN', or new values)."
        ),
    )
    unmet_constraints: Optional[list[str]] = Field(
        default=None,
        description="List of user constraints that the results failed to satisfy.",
    )


class VerifierModel(BaseModel):
    """
    Validate results against user constraints with structured feedback.

    Use fast models (GPT-3.5, Haiku, Llama 3 8B) to minimize latency (~50-100ms).
    Skipped for 'standard' mode when auto_bypass=true in router config.
    Returns structured feedback for intelligent retry, not blind retry.

    Example:
        ```yaml
        retriever:
          verifier:
            model: *fast_llm
            on_failure: warn_and_retry
            max_retries: 1
        ```
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    model: Optional["LLMModel"] = Field(
        default=None,
        description="LLM for verification (fast model recommended)",
    )
    on_failure: Literal["warn", "retry", "warn_and_retry"] = Field(
        default="warn",
        description="Behavior when verification fails",
    )
    max_retries: int = Field(
        default=1,
        description="Maximum retry attempts before returning with warning",
    )


class RankedDocument(BaseModel):
    """Single ranked document."""

    index: int = Field(description="Document index from input list")
    score: float = Field(description="0.0-1.0 relevance score")
    reason: str = Field(default="", description="Why this score")


class RankingResult(BaseModel):
    """Reranking output."""

    rankings: list[RankedDocument] = Field(
        default_factory=list,
        description="Ranked documents, highest score first",
    )


class RetrieverModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    vector_store: VectorStoreModel
    columns: Optional[list[str]] = Field(default_factory=list)
    search_parameters: SearchParametersModel = Field(
        default_factory=SearchParametersModel
    )
    router: Optional[RouterModel] = Field(
        default=None,
        description="Optional query router for selecting execution mode (standard vs instructed).",
    )
    rerank: Optional[RerankParametersModel | bool] = Field(
        default=None,
        description="Optional reranking configuration. Set to true for defaults, or provide ReRankParametersModel for custom settings.",
    )
    instructed: Optional[InstructedRetrieverModel] = Field(
        default=None,
        description="Optional instructed retrieval with query decomposition and RRF merging.",
    )
    verifier: Optional[VerifierModel] = Field(
        default=None,
        description="Optional result verification with structured feedback for retry.",
    )

    @model_validator(mode="after")
    def set_default_columns(self) -> Self:
        if not self.columns:
            columns: Sequence[str] = self.vector_store.columns
            self.columns = columns
        return self

    @model_validator(mode="after")
    def set_default_reranker(self) -> Self:
        """Convert bool to ReRankParametersModel with defaults.

        When rerank: true is used, sets the default FlashRank model
        (ms-marco-MiniLM-L-12-v2) to enable reranking.
        """
        if isinstance(self.rerank, bool) and self.rerank:
            self.rerank = RerankParametersModel(model="ms-marco-MiniLM-L-12-v2")
        return self


class FunctionType(str, Enum):
    PYTHON = "python"
    FACTORY = "factory"
    UNITY_CATALOG = "unity_catalog"
    MCP = "mcp"


class HumanInTheLoopModel(BaseModel):
    """
    Configuration for Human-in-the-Loop tool approval.

    This model configures when and how tools require human approval before execution.
    It maps to LangChain's HumanInTheLoopMiddleware.

    LangChain supports three decision types:
    - "approve": Execute tool with original arguments
    - "edit": Modify arguments before execution
    - "reject": Skip execution with optional feedback message
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    review_prompt: Optional[str] = Field(
        default=None,
        description="Message shown to the reviewer when approval is requested",
    )

    allowed_decisions: list[Literal["approve", "edit", "reject"]] = Field(
        default_factory=lambda: ["approve", "edit", "reject"],
        description="List of allowed decision types for this tool",
    )

    @model_validator(mode="after")
    def validate_and_normalize_decisions(self) -> Self:
        """Validate and normalize allowed decisions."""
        if not self.allowed_decisions:
            raise ValueError("At least one decision type must be allowed")

        # Remove duplicates while preserving order
        seen = set()
        unique_decisions = []
        for decision in self.allowed_decisions:
            if decision not in seen:
                seen.add(decision)
                unique_decisions.append(decision)
        self.allowed_decisions = unique_decisions

        return self


class BaseFunctionModel(ABC, BaseModel):
    model_config = ConfigDict(
        use_enum_values=True,
        discriminator="type",
    )
    type: FunctionType
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
    name: str

    @property
    def full_name(self) -> str:
        return self.name

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_python_tool

        return [create_python_tool(self, **kwargs)]


class FactoryFunctionModel(BaseFunctionModel, HasFullName):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.FACTORY] = FunctionType.FACTORY
    name: str
    args: Optional[dict[str, Any]] = Field(default_factory=dict)

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


class McpFunctionModel(BaseFunctionModel, IsDatabricksResource):
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
    app: Optional[DatabricksAppModel] = None
    connection: Optional[ConnectionModel] = None
    functions: Optional[SchemaModel] = None
    genie_room: Optional[GenieRoomModel] = None
    sql: Optional[bool] = None
    vector_search: Optional[VectorStoreModel] = None
    # Tool filtering
    include_tools: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of tool names or glob patterns to include from the MCP server. "
            "If specified, only tools matching these patterns will be loaded. "
            "Supports glob patterns: * (any chars), ? (single char), [abc] (char set). "
            "Examples: ['execute_query', 'list_*', 'get_?_data']"
        ),
    )
    exclude_tools: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of tool names or glob patterns to exclude from the MCP server. "
            "Tools matching these patterns will not be loaded. "
            "Takes precedence over include_tools. "
            "Supports glob patterns: * (any chars), ? (single char), [abc] (char set). "
            "Examples: ['drop_*', 'delete_*', 'execute_ddl']"
        ),
    )

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
        - If app is set, retrieves URL from Databricks App via workspace client
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
        - Databricks App: Retrieved dynamically from workspace
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

        # Databricks App - MCP endpoint is at {app_url}/mcp
        # Try McpFunctionModel's workspace_client first (which may have credentials),
        # then fall back to DatabricksAppModel.url property (which uses its own workspace_client)
        if self.app:
            from databricks.sdk.service.apps import App

            app_url: str | None = None

            # First, try using McpFunctionModel's workspace_client
            try:
                app: App = self.workspace_client.apps.get(self.app.name)
                app_url = app.url
                logger.trace(
                    "Got app URL using McpFunctionModel workspace_client",
                    app_name=self.app.name,
                    url=app_url,
                )
            except Exception as e:
                logger.debug(
                    "Failed to get app URL using McpFunctionModel workspace_client, "
                    "trying DatabricksAppModel.url property",
                    app_name=self.app.name,
                    error=str(e),
                )

            # Fall back to DatabricksAppModel.url property
            if not app_url:
                try:
                    app_url = self.app.url
                    logger.trace(
                        "Got app URL using DatabricksAppModel.url property",
                        app_name=self.app.name,
                        url=app_url,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Databricks App '{self.app.name}' does not have a URL. "
                        "The app may not be deployed yet, or credentials may be invalid. "
                        f"Error: {e}"
                    ) from e

            return f"{app_url.rstrip('/')}/mcp"

        # Vector Search
        if self.vector_search:
            if (
                not self.vector_search.index
                or not self.vector_search.index.schema_model
            ):
                raise ValueError(
                    "vector_search must have an index with a schema (catalog/schema) configured"
                )
            catalog: str = value_of(self.vector_search.index.schema_model.catalog_name)
            schema: str = value_of(self.vector_search.index.schema_model.schema_name)
            return f"{workspace_host}/api/2.0/mcp/vector-search/{catalog}/{schema}"

        # UC Functions MCP server
        if self.functions:
            catalog: str = value_of(self.functions.catalog_name)
            schema: str = value_of(self.functions.schema_name)
            return f"{workspace_host}/api/2.0/mcp/functions/{catalog}/{schema}"

        raise ValueError(
            "No URL source configured. Provide one of: url, app, connection, genie_room, "
            "sql, vector_search, or functions"
        )

    @field_serializer("transport")
    def serialize_transport(self, value: TransportType) -> str:
        """Serialize transport enum to string."""
        if isinstance(value, TransportType):
            return value.value
        return str(value)

    @model_validator(mode="after")
    def validate_mutually_exclusive(self) -> Self:
        """Validate that exactly one URL source is provided."""
        # Count how many URL sources are provided
        url_sources: list[tuple[str, Any]] = [
            ("url", self.url),
            ("app", self.app),
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
                    "url, app, connection, genie_room, sql, vector_search, or functions"
                )
            if len(provided_sources) > 1:
                raise ValueError(
                    f"For STREAMABLE_HTTP transport, only one URL source can be provided. "
                    f"Found: {', '.join(provided_sources)}. "
                    f"Please provide only one of: url, app, connection, genie_room, sql, vector_search, or functions"
                )

        if self.transport == TransportType.STDIO:
            if not self.command:
                raise ValueError("command must be provided for STDIO transport")
            if not self.args:
                raise ValueError("args must be provided for STDIO transport")

        return self

    @model_validator(mode="after")
    def update_url(self) -> Self:
        """Resolve AnyVariable to concrete value for URL."""
        if self.url is not None:
            resolved_value: Any = value_of(self.url)
            # Cast to string since URL must be a string
            self.url = str(resolved_value) if resolved_value else None
        return self

    @model_validator(mode="after")
    def update_headers(self) -> Self:
        """Resolve AnyVariable to concrete values for headers."""
        for key, value in self.headers.items():
            resolved_value: Any = value_of(value)
            # Headers must be strings
            self.headers[key] = str(resolved_value) if resolved_value else ""
        return self

    @model_validator(mode="after")
    def validate_tool_filters(self) -> Self:
        """Validate tool filter configuration."""
        from loguru import logger

        # Warn if both are empty lists (explicit but pointless)
        if self.include_tools is not None and len(self.include_tools) == 0:
            logger.warning(
                "include_tools is empty list - no tools will be loaded. "
                "Remove field to load all tools."
            )

        if self.exclude_tools is not None and len(self.exclude_tools) == 0:
            logger.warning(
                "exclude_tools is empty list - has no effect. "
                "Remove field or add patterns."
            )

        return self

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_mcp_tools

        return create_mcp_tools(self)


class UnityCatalogFunctionModel(BaseFunctionModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.UNITY_CATALOG] = FunctionType.UNITY_CATALOG
    resource: FunctionModel
    partial_args: Optional[dict[str, AnyVariable]] = Field(default_factory=dict)

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
    auto_register: bool = Field(
        default=False,
        description="Whether to automatically register the default_template to the prompt registry. "
        "If False, the prompt will only be loaded from the registry (never created/updated). "
        "Defaults to True for backward compatibility.",
    )

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


class CheckpointerModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str
    database: Optional[DatabaseModel] = None

    @property
    def storage_type(self) -> StorageType:
        """Infer storage type from database presence."""
        return StorageType.POSTGRES if self.database else StorageType.MEMORY

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
    dims: Optional[int] = 1536
    database: Optional[DatabaseModel] = None
    namespace: Optional[str] = None

    @property
    def storage_type(self) -> StorageType:
        """Infer storage type from database presence."""
        return StorageType.POSTGRES if self.database else StorageType.MEMORY

    def as_store(self) -> BaseStore:
        from dao_ai.memory import StoreManager

        store: BaseStore = StoreManager.instance(self).store()
        return store


class MemoryModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    checkpointer: Optional[CheckpointerModel] = None
    store: Optional[StoreModel] = None


FunctionHook: TypeAlias = PythonFunctionModel | FactoryFunctionModel | str


class ResponseFormatModel(BaseModel):
    """
    Configuration for structured response formats.

    The response_schema field accepts either a type or a string:
    - Type (Pydantic model, dataclass, etc.): Used directly for structured output
    - String: First attempts to load as a fully qualified type name, falls back to JSON schema string

    This unified approach simplifies the API while maintaining flexibility.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    use_tool: Optional[bool] = Field(
        default=None,
        description=(
            "Strategy for structured output: "
            "None (default) = auto-detect from model capabilities, "
            "False = force ProviderStrategy (native), "
            "True = force ToolStrategy (function calling)"
        ),
    )
    response_schema: Optional[str | type] = Field(
        default=None,
        description="Type or string for response format. String attempts FQN import, falls back to JSON schema.",
    )

    def as_strategy(self) -> ProviderStrategy | ToolStrategy:
        """
        Convert response_schema to appropriate LangChain strategy.

        Returns:
            - None if no response_schema configured
            - Raw schema/type for auto-detection (when use_tool=None)
            - ToolStrategy wrapping the schema (when use_tool=True)
            - ProviderStrategy wrapping the schema (when use_tool=False)

        Raises:
            ValueError: If response_schema is a JSON schema string that cannot be parsed
        """

        if self.response_schema is None:
            return None

        schema = self.response_schema

        # Handle type schemas (Pydantic, dataclass, etc.)
        if self.is_type_schema:
            if self.use_tool is None:
                # Auto-detect: Pass schema directly, let LangChain decide
                return schema
            elif self.use_tool is True:
                # Force ToolStrategy (function calling)
                return ToolStrategy(schema)
            else:  # use_tool is False
                # Force ProviderStrategy (native structured output)
                return ProviderStrategy(schema)

        # Handle JSON schema strings
        elif self.is_json_schema:
            import json

            try:
                schema_dict = json.loads(schema)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON schema string: {e}") from e

            # Apply same use_tool logic as type schemas
            if self.use_tool is None:
                # Auto-detect
                return schema_dict
            elif self.use_tool is True:
                # Force ToolStrategy
                return ToolStrategy(schema_dict)
            else:  # use_tool is False
                # Force ProviderStrategy
                return ProviderStrategy(schema_dict)

        return None

    @model_validator(mode="after")
    def validate_response_schema(self) -> Self:
        """
        Validate and convert response_schema.

        Processing logic:
        1. If None: no response format specified
        2. If type: use directly as structured output type
        3. If str: try to load as FQN using type_from_fqn
           - Success: response_schema becomes the loaded type
           - Failure: keep as string (treated as JSON schema)

        After validation, response_schema is one of:
        - None (no schema)
        - type (use for structured output)
        - str (JSON schema)

        Returns:
            Self with validated response_schema
        """
        if self.response_schema is None:
            return self

        # If already a type, return
        if isinstance(self.response_schema, type):
            return self

        # If it's a string, try to load as type, fallback to json_schema
        if isinstance(self.response_schema, str):
            from dao_ai.utils import type_from_fqn

            try:
                resolved_type = type_from_fqn(self.response_schema)
                self.response_schema = resolved_type
                logger.debug(
                    f"Resolved response_schema string to type: {resolved_type}"
                )
                return self
            except (ValueError, ImportError, AttributeError, TypeError) as e:
                # Keep as string - it's a JSON schema
                logger.debug(
                    f"Could not resolve '{self.response_schema}' as type: {e}. "
                    f"Treating as JSON schema string."
                )
                return self

        # Invalid type
        raise ValueError(
            f"response_schema must be None, type, or str, got {type(self.response_schema)}"
        )

    @property
    def is_type_schema(self) -> bool:
        """Returns True if response_schema is a type (not JSON schema string)."""
        return isinstance(self.response_schema, type)

    @property
    def is_json_schema(self) -> bool:
        """Returns True if response_schema is a JSON schema string (not a type)."""
        return isinstance(self.response_schema, str)


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
    response_format: Optional[ResponseFormatModel | type | str] = None

    @model_validator(mode="after")
    def validate_response_format(self) -> Self:
        """
        Validate and normalize response_format.

        Accepts:
        - None (no response format)
        - ResponseFormatModel (already validated)
        - type (Pydantic model, dataclass, etc.) - converts to ResponseFormatModel
        - str (FQN or json_schema) - converts to ResponseFormatModel (smart fallback)

        ResponseFormatModel handles the logic of trying FQN import and falling back to JSON schema.
        """
        if self.response_format is None or isinstance(
            self.response_format, ResponseFormatModel
        ):
            return self

        # Convert type or str to ResponseFormatModel
        # ResponseFormatModel's validator will handle the smart type loading and fallback
        if isinstance(self.response_format, (type, str)):
            self.response_format = ResponseFormatModel(
                response_schema=self.response_format
            )
            return self

        # Invalid type
        raise ValueError(
            f"response_format must be None, ResponseFormatModel, type, or str, "
            f"got {type(self.response_format)}"
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
    default_agent: Optional[AgentModel | str] = None
    middleware: list[MiddlewareModel] = Field(
        default_factory=list,
        description="List of middleware to apply to all agents in the swarm",
    )
    handoffs: Optional[dict[str, Optional[list[AgentModel | str]]]] = Field(
        default_factory=dict
    )


class OrchestrationModel(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    supervisor: Optional[SupervisorModel] = None
    swarm: Optional[SwarmModel | Literal[True]] = None
    memory: Optional[MemoryModel] = None

    @model_validator(mode="after")
    def validate_and_normalize(self) -> Self:
        """Validate orchestration and normalize swarm shorthand."""
        # Convert swarm: true to SwarmModel()
        if self.swarm is True:
            self.swarm = SwarmModel()

        # Validate mutually exclusive
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
    deployment_target: Optional[DeploymentTarget] = Field(
        default=None,
        description="Default deployment target. If not specified, defaults to MODEL_SERVING. "
        "Can be overridden via CLI --target flag. Options: 'model_serving' or 'apps'.",
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
                    swarm=SwarmModel(default_agent=default_agent)
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
    """
    Configuration for MLflow GenAI evaluation.

    Attributes:
        model: LLM model used as the judge for LLM-based scorers (e.g., Guidelines, Safety).
               This model evaluates agent responses during evaluation.
        table: Table to store evaluation results.
        num_evals: Number of evaluation samples to generate.
        agent_description: Description of the agent for evaluation data generation.
        question_guidelines: Guidelines for generating evaluation questions.
        custom_inputs: Custom inputs to pass to the agent during evaluation.
        guidelines: List of guideline configurations for Guidelines scorers.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel = Field(
        ..., description="LLM model used as the judge for LLM-based evaluation scorers"
    )
    table: TableModel
    num_evals: int
    agent_description: Optional[str] = None
    question_guidelines: Optional[str] = None
    custom_inputs: dict[str, Any] = Field(default_factory=dict)
    guidelines: list[GuidelineModel] = Field(default_factory=list)

    @property
    def judge_model_endpoint(self) -> str:
        """
        Get the judge model endpoint string for MLflow scorers.

        Returns:
            Endpoint string in format 'databricks:/model-name'
        """
        return f"databricks:/{self.model.name}"


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
    function: FunctionModel
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
    apps: dict[str, DatabricksAppModel] = Field(default_factory=dict)

    @model_validator(mode="after")
    def update_genie_warehouses(self) -> Self:
        """
        Automatically populate warehouses from genie_rooms.

        Warehouses are extracted from each Genie room and added to the
        resources if they don't already exist (based on warehouse_id).
        """
        if not self.genie_rooms:
            return self

        # Process warehouses from all genie rooms
        for genie_room in self.genie_rooms.values():
            genie_room: GenieRoomModel
            warehouse: Optional[WarehouseModel] = genie_room.warehouse

            if warehouse is None:
                continue

            # Check if warehouse already exists based on warehouse_id
            warehouse_exists: bool = any(
                existing_warehouse.warehouse_id == warehouse.warehouse_id
                for existing_warehouse in self.warehouses.values()
            )

            if not warehouse_exists:
                warehouse_key: str = normalize_name(
                    "_".join([genie_room.name, warehouse.warehouse_id])
                )
                self.warehouses[warehouse_key] = warehouse
                logger.trace(
                    "Added warehouse from Genie room",
                    room=genie_room.name,
                    warehouse=warehouse.warehouse_id,
                    key=warehouse_key,
                )

        return self

    @model_validator(mode="after")
    def update_genie_tables(self) -> Self:
        """
        Automatically populate tables from genie_rooms.

        Tables are extracted from each Genie room and added to the
        resources if they don't already exist (based on full_name).
        """
        if not self.genie_rooms:
            return self

        # Process tables from all genie rooms
        for genie_room in self.genie_rooms.values():
            genie_room: GenieRoomModel
            for table in genie_room.tables:
                table: TableModel
                table_exists: bool = any(
                    existing_table.full_name == table.full_name
                    for existing_table in self.tables.values()
                )
                if not table_exists:
                    table_key: str = normalize_name(
                        "_".join([genie_room.name, table.full_name])
                    )
                    self.tables[table_key] = table
                    logger.trace(
                        "Added table from Genie room",
                        room=genie_room.name,
                        table=table.name,
                        key=table_key,
                    )

        return self

    @model_validator(mode="after")
    def update_genie_functions(self) -> Self:
        """
        Automatically populate functions from genie_rooms.

        Functions are extracted from each Genie room and added to the
        resources if they don't already exist (based on full_name).
        """
        if not self.genie_rooms:
            return self

        # Process functions from all genie rooms
        for genie_room in self.genie_rooms.values():
            genie_room: GenieRoomModel
            for function in genie_room.functions:
                function: FunctionModel
                function_exists: bool = any(
                    existing_function.full_name == function.full_name
                    for existing_function in self.functions.values()
                )
                if not function_exists:
                    function_key: str = normalize_name(
                        "_".join([genie_room.name, function.full_name])
                    )
                    self.functions[function_key] = function
                    logger.trace(
                        "Added function from Genie room",
                        room=genie_room.name,
                        function=function.name,
                        key=function_key,
                    )

        return self


class AppConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    version: Optional[str] = None
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

    # Private attribute to track the source config file path (set by from_file)
    _source_config_path: str | None = None

    @classmethod
    def from_file(cls, path: PathLike) -> "AppConfig":
        path = Path(path).as_posix()
        logger.debug(f"Loading config from {path}")
        model_config: ModelConfig = ModelConfig(development_config=path)
        config: AppConfig = AppConfig(**model_config.to_dict())

        # Store the source config path for later use (e.g., Apps deployment)
        config._source_config_path = path

        config.initialize()

        atexit.register(config.shutdown)

        return config

    @property
    def source_config_path(self) -> str | None:
        """Get the source config file path if loaded via from_file."""
        return self._source_config_path

    def initialize(self) -> None:
        from dao_ai.hooks.core import create_hooks
        from dao_ai.logging import configure_logging

        if self.app and self.app.log_level:
            configure_logging(level=self.app.log_level)

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
        target: DeploymentTarget | None = None,
        w: WorkspaceClient | None = None,
        vsc: "VectorSearchClient | None" = None,
        pat: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        workspace_host: str | None = None,
    ) -> None:
        """
        Deploy the agent to the specified target.

        Target resolution follows this priority:
        1. Explicit `target` parameter (if provided)
        2. `app.deployment_target` from config file (if set)
        3. Default: MODEL_SERVING

        Args:
            target: The deployment target (MODEL_SERVING or APPS). If None, uses
                config.app.deployment_target or defaults to MODEL_SERVING.
            w: Optional WorkspaceClient instance
            vsc: Optional VectorSearchClient instance
            pat: Optional personal access token for authentication
            client_id: Optional client ID for service principal authentication
            client_secret: Optional client secret for service principal authentication
            workspace_host: Optional workspace host URL
        """
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        # Resolve target using hybrid logic:
        # 1. Explicit parameter takes precedence
        # 2. Fall back to config.app.deployment_target
        # 3. Default to MODEL_SERVING
        resolved_target: DeploymentTarget
        if target is not None:
            resolved_target = target
        elif self.app is not None and self.app.deployment_target is not None:
            resolved_target = self.app.deployment_target
        else:
            resolved_target = DeploymentTarget.MODEL_SERVING

        provider: ServiceProvider = DatabricksProvider(
            w=w,
            vsc=vsc,
            pat=pat,
            client_id=client_id,
            client_secret=client_secret,
            workspace_host=workspace_host,
        )
        provider.deploy_agent(self, target=resolved_target)

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
