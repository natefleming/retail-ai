import atexit
import importlib
import os
import re
import sys
from abc import ABC, abstractmethod
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterator,
    Literal,
    Optional,
    Self,
    Sequence,
    TypeAlias,
    Union,
)

if TYPE_CHECKING:
    from dao_ai.genie.cache.context_aware.optimization import (
        ContextAwareCacheEvalDataset,
        ThresholdOptimizationResult,
    )
    from dao_ai.state import Context

from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import (
    CredentialsStrategy,
    ModelServingUserCredentials,
)
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.apps import App
from databricks.sdk.service.catalog import FunctionInfo, TableInfo
from databricks.sdk.service.dashboards import GenieListSpacesResponse, GenieSpace
from databricks.sdk.service.database import DatabaseInstance
from databricks.sdk.service.sql import EndpointInfo, GetWarehouseResponse
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
from mlflow.genai.datasets import (
    EvaluationDataset,
    create_dataset,
    delete_dataset,
    get_dataset,
)
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
    field_validator,
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
    """A variable resolved from an environment variable at runtime."""

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )
    env: str = Field(
        description="Environment variable name to read at runtime.",
    )
    default_value: Optional[Any] = Field(
        default=None,
        description="Fallback value used when the environment variable is not set.",
    )

    def as_value(self) -> Any:
        logger.debug(f"Fetching environment variable: {self.env}")
        value: Any = os.environ.get(self.env, self.default_value)
        if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
            logger.warning(
                f"Environment variable {self.env} contains an unresolved template "
                f"reference: {value}. Treating as unresolved."
            )
            return self.default_value
        return value

    def __str__(self) -> str:
        return self.env


class SecretVariableModel(BaseModel, HasValue):
    """A variable resolved from a Databricks secret scope at runtime."""

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )
    scope: str = Field(
        description="Databricks secret scope name.",
    )
    secret: str = Field(
        description="Secret key within the scope.",
    )
    default_value: Optional[Any] = Field(
        default=None,
        description="Fallback value used when the secret cannot be retrieved.",
    )

    def as_value(self) -> Any:
        logger.debug(f"Fetching secret: {self.scope}/{self.secret}")
        from dao_ai.providers.databricks import DatabricksProvider

        provider: DatabricksProvider = DatabricksProvider()
        value: Any = provider.get_secret(self.scope, self.secret, self.default_value)
        return value

    def __str__(self) -> str:
        return "{{secrets/" + f"{self.scope}/{self.secret}" + "}}"


class PrimitiveVariableModel(BaseModel, HasValue):
    """A variable holding a literal primitive value (string, int, float, or bool)."""

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )

    value: Union[str, int, float, bool] = Field(
        description="Literal value (string, integer, float, or boolean).",
    )

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
    """A variable that tries multiple sources in order, returning the first non-None value."""

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )
    default_value: Optional[Any] = Field(
        default=None,
        description="Fallback value used when all options resolve to None.",
    )
    options: list[
        EnvironmentVariableModel
        | SecretVariableModel
        | PrimitiveVariableModel
        | str
        | int
        | float
        | bool
    ] = Field(
        default_factory=list,
        description="Ordered list of variable sources tried until one returns a non-None value.",
    )

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
    """Databricks service principal credentials for OAuth M2M authentication."""

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )
    client_id: AnyVariable = Field(
        description="OAuth application (client) ID for the service principal.",
    )
    client_secret: AnyVariable = Field(
        description="OAuth client secret for the service principal.",
    )


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

    on_behalf_of_user: Optional[bool] = Field(
        default=False,
        description="Use the calling user's identity (OBO). Works in Model Serving and Databricks Apps.",
    )
    service_principal: Optional[ServicePrincipalModel] = Field(
        default=None,
        description="Service principal for OAuth M2M authentication. Expands to client_id and client_secret.",
    )
    client_id: Optional[AnyVariable] = Field(
        default=None,
        description="OAuth client ID for service principal authentication.",
    )
    client_secret: Optional[AnyVariable] = Field(
        default=None,
        description="OAuth client secret for service principal authentication.",
    )
    workspace_host: Optional[AnyVariable] = Field(
        default=None,
        description="Databricks workspace URL (e.g., 'https://my-workspace.cloud.databricks.com').",
    )
    pat: Optional[AnyVariable] = Field(
        default=None,
        description="Personal access token for PAT-based authentication.",
    )

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
    """Unity Catalog privilege types for granting access to resources."""

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
    """A grant of Unity Catalog privileges to one or more principals."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    principals: list[ServicePrincipalModel | str] = Field(
        default_factory=list,
        description="Users, groups, or service principals receiving the privileges.",
    )
    privileges: list[Privilege] = Field(
        description="Unity Catalog privileges to grant (e.g., SELECT, EXECUTE, USE_SCHEMA).",
    )

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
    """Unity Catalog schema reference (catalog + schema) used to qualify tables, functions, and prompts."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    catalog_name: AnyVariable = Field(
        description="Unity Catalog catalog name.",
    )
    schema_name: AnyVariable = Field(
        description="Unity Catalog schema name within the catalog.",
    )
    permissions: Optional[list[PermissionModel]] = Field(
        default_factory=list,
        description="Permissions to grant on this schema during provisioning.",
    )

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
    name: str = Field(
        description="The unique instance name of the Databricks App in the workspace.",
    )

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
    """Unity Catalog table reference. Provide a fully qualified name or a schema + table name."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(
        default=None,
        alias="schema",
        description="Schema reference qualifying the table. If omitted, name must be fully qualified.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Table name (short) or fully qualified name (catalog.schema.table).",
    )

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
    """Configuration for an LLM served via a Databricks Model Serving endpoint."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Serving endpoint name (e.g., 'databricks-meta-llama-3-3-70b-instruct').",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of this model configuration.",
    )
    temperature: Optional[float] = Field(
        default=0.1,
        description="Sampling temperature controlling output randomness (0.0 = deterministic, 1.0 = creative).",
    )
    max_tokens: Optional[int] = Field(
        default=8192,
        description="Maximum number of tokens in the model response.",
    )
    fallbacks: Optional[list[Union[str, "LLMModel"]]] = Field(
        default_factory=list,
        description="Ordered list of fallback model names or LLMModel configs tried on primary model failure.",
    )
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
    """Vector search endpoint compute profile."""

    STANDARD = "STANDARD"
    OPTIMIZED_STORAGE = "OPTIMIZED_STORAGE"


class VectorSearchEndpoint(BaseModel):
    """Vector search endpoint that hosts one or more vector search indexes."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Vector search endpoint name in the workspace.",
    )
    type: VectorSearchEndpointType = Field(
        default=VectorSearchEndpointType.STANDARD,
        description="Endpoint type: STANDARD or OPTIMIZED_STORAGE.",
    )

    @field_serializer("type")
    def serialize_type(self, value: VectorSearchEndpointType) -> str:
        """Ensure enum is serialized to string value."""
        if isinstance(value, VectorSearchEndpointType):
            return value.value
        return str(value)


class IndexModel(IsDatabricksResource, HasFullName):
    """Model representing a Databricks Vector Search index."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(
        default=None,
        alias="schema",
        description="Schema reference qualifying the index name.",
    )
    name: str = Field(
        description="Index name (short) or fully qualified name (catalog.schema.index).",
    )

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
    """Unity Catalog function reference. Provide a fully qualified name or a schema + function name."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(
        default=None,
        alias="schema",
        description="Schema reference qualifying the function. If omitted, name must be fully qualified.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Function name (short) or fully qualified name (catalog.schema.function).",
    )

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
    """SQL warehouse configuration. Provide either a name or warehouse_id."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: Optional[str] = Field(
        default=None,
        description="SQL warehouse display name. Resolved to warehouse_id automatically.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of this warehouse.",
    )
    warehouse_id: Optional[AnyVariable] = Field(
        default=None,
        description="SQL warehouse ID. If omitted, looked up by name.",
    )

    _warehouse_details: Optional[GetWarehouseResponse] = PrivateAttr(default=None)

    def _get_warehouse_details(self) -> GetWarehouseResponse:
        if self._warehouse_details is None:
            self._warehouse_details = self.workspace_client.warehouses.get(
                id=value_of(self.warehouse_id)
            )
        return self._warehouse_details

    def _resolve_warehouse_id_by_name(self, name: str) -> str:
        """Look up a warehouse ID by iterating all warehouses and matching by name."""
        logger.info(f"Resolving warehouse by name: '{name}'")
        warehouses: Iterator[EndpointInfo] = self.workspace_client.warehouses.list()
        for warehouse in warehouses:
            if warehouse.name == name:
                logger.info(f"Resolved warehouse '{name}' to id '{warehouse.id}'")
                return warehouse.id
        raise ValueError(
            f"No warehouse found with name '{name}'. "
            "Verify the name matches an existing SQL warehouse in your workspace."
        )

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
    def resolve_warehouse_by_name(self) -> Self:
        """Resolve warehouse_id from name when only name is provided."""
        if self.warehouse_id:
            self.warehouse_id = value_of(self.warehouse_id)
            return self
        if self.name:
            self.warehouse_id = self._resolve_warehouse_id_by_name(self.name)
            return self
        raise ValueError(
            "Either 'warehouse_id' or 'name' must be provided for WarehouseModel."
        )

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
    """Databricks Genie space configuration for natural-language SQL exploration."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: Optional[str] = Field(
        default=None,
        description="Display name for the Genie room. Auto-populated from the space if omitted.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the Genie room. Auto-populated from the space if omitted.",
    )
    space_id: Optional[AnyVariable] = Field(
        default=None,
        description="Databricks Genie space ID. If omitted, looked up by name.",
    )

    _space_details: Optional[GenieSpace] = PrivateAttr(default=None)

    def _get_space_details(self) -> GenieSpace | None:
        """Fetch Genie space details from the API.

        Returns:
            GenieSpace if successful, None if the API call fails (e.g., due to
            permission issues in model serving environments).
        """
        if self._space_details is None:
            try:
                self._space_details = self.workspace_client.genie.get_space(
                    space_id=self.space_id, include_serialized_space=True
                )
            except Exception as e:
                logger.debug(
                    "Could not fetch Genie space details (this is expected in model serving)",
                    space_id=self.space_id,
                    error=str(e),
                )
                return None
        return self._space_details

    def _resolve_space_id_by_name(self, name: str) -> str:
        """Look up a Genie space ID by iterating all spaces and matching by title."""
        logger.info(f"Resolving Genie space by name: '{name}'")
        page_token: Optional[str] = None
        while True:
            response: GenieListSpacesResponse = self.workspace_client.genie.list_spaces(
                page_token=page_token
            )
            if response.spaces:
                for space in response.spaces:
                    if space.title == name:
                        logger.info(
                            f"Resolved Genie space '{name}' to space_id '{space.space_id}'"
                        )
                        return space.space_id
            if not response.next_page_token:
                break
            page_token = response.next_page_token
        raise ValueError(
            f"No Genie space found with title '{name}'. "
            "Verify the name matches an existing Genie space in your workspace."
        )

    def _parse_serialized_space(self) -> dict[str, Any]:
        """Parse the serialized_space JSON string and return the parsed data."""
        import json

        space_details = self._get_space_details()
        if space_details is None or not space_details.serialized_space:
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
        space_details = self._get_space_details()

        if space_details is None or not space_details.warehouse_id:
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
    def resolve_space_by_name(self) -> Self:
        """Resolve space_id from name when only name is provided."""
        if self.space_id:
            self.space_id = value_of(self.space_id)
            return self
        if self.name:
            self.space_id = self._resolve_space_id_by_name(self.name)
            return self
        raise ValueError(
            "Either 'space_id' or 'name' must be provided for GenieRoomModel."
        )

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
    """Unity Catalog volume reference for file storage."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(
        default=None,
        alias="schema",
        description="Schema reference qualifying the volume name.",
    )
    name: str = Field(
        description="Volume name (short) or fully qualified name (catalog.schema.volume).",
    )

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
    """A path within a Unity Catalog volume (e.g., /Volumes/catalog/schema/volume/subdir)."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    volume: Optional[VolumeModel] = Field(
        default=None,
        description="Volume reference. Combined with path to form the full /Volumes/... path.",
    )
    path: Optional[str] = Field(
        default=None,
        description="Relative path within the volume, or an absolute /Volumes/... path if volume is omitted.",
    )

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

    index: Optional[IndexModel] = Field(
        default=None,
        description="Vector search index to query. Required for runtime; auto-generated in provisioning mode.",
    )

    source_table: Optional[TableModel] = Field(
        default=None,
        description="Source table for provisioning a new vector search index. Omit when using an existing index.",
    )
    embedding_source_column: Optional[str] = Field(
        default=None,
        description="Column in the source table containing text to embed. Required in provisioning mode.",
    )
    embedding_model: Optional[LLMModel] = Field(
        default=None,
        description="Embedding model endpoint. Defaults to 'databricks-gte-large-en' in provisioning mode.",
    )
    endpoint: Optional[VectorSearchEndpoint] = Field(
        default=None,
        description="Vector search endpoint hosting the index. Auto-detected in provisioning mode.",
    )

    source_path: Optional[VolumePathModel] = Field(
        default=None,
        description="Volume path for source data files (alternative to source_table).",
    )
    checkpoint_path: Optional[VolumePathModel] = Field(
        default=None,
        description="Volume path for sync checkpoint storage.",
    )
    primary_key: Optional[str] = Field(
        default=None,
        description="Primary key column in the source table. Auto-detected if omitted.",
    )
    columns: Optional[list[str]] = Field(
        default_factory=list,
        description="Columns to include in search results.",
    )
    doc_uri: Optional[str] = Field(
        default=None,
        description="Column name containing document URIs for provenance tracking.",
    )

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
    """Unity Catalog connection for external data sources and MCP servers."""

    model_config = ConfigDict()
    name: str = Field(
        description="Unity Catalog connection name.",
    )

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
    name: Optional[str] = Field(
        default=None,
        description="Logical database name. For Lakebase, defaults to instance_name.",
    )
    instance_name: Optional[str] = Field(
        default=None,
        description="Databricks Lakebase instance name. Mutually exclusive with host (standard PostgreSQL).",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of this database connection.",
    )
    host: Optional[AnyVariable] = Field(
        default=None,
        description="PostgreSQL host address. For Lakebase, auto-fetched from the instance API.",
    )
    database: Optional[AnyVariable] = Field(
        default="databricks_postgres",
        description="Database name within the PostgreSQL server.",
    )
    port: Optional[AnyVariable] = Field(
        default=5432,
        description="PostgreSQL port number.",
    )
    connection_kwargs: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Extra keyword arguments passed to the connection pool.",
    )
    max_pool_size: Optional[int] = Field(
        default=10,
        description="Maximum number of connections in the pool.",
    )
    timeout_seconds: Optional[int] = Field(
        default=10,
        description="Connection timeout in seconds.",
    )
    capacity: Optional[Literal["CU_1", "CU_2"]] = Field(
        default="CU_2",
        description="Lakebase compute capacity tier (CU_1 or CU_2).",
    )
    node_count: Optional[int] = Field(
        default=None,
        description="Number of Lakebase compute nodes for horizontal scaling.",
    )
    user: Optional[AnyVariable] = Field(
        default=None,
        description="Database username. For Lakebase, auto-detected from workspace identity.",
    )
    password: Optional[AnyVariable] = Field(
        default=None,
        description="Database password. For Lakebase, a token is generated automatically.",
    )

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
    """Configuration for a simple LRU (Least Recently Used) Genie response cache."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    capacity: int = Field(
        default=1000,
        description="Maximum number of cached responses before LRU eviction.",
    )
    time_to_live_seconds: int | None = Field(
        default=60 * 60 * 24,
        description="Cache entry TTL in seconds. None or negative = entries never expire. Default: 1 day.",
    )
    warehouse: WarehouseModel = Field(
        description="SQL warehouse used by the Genie API for query execution.",
    )


class GenieContextAwareCacheParametersBase(BaseModel):
    """
    Base configuration shared by all context-aware cache backends.

    This base class contains the shared fields for similarity matching,
    embedding generation, and context window configuration that are common
    to both the PostgreSQL-backed and in-memory context-aware cache implementations.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    time_to_live_seconds: int | None = Field(
        default=60 * 60 * 24,
        description="Cache entry TTL in seconds. None or negative = entries never expire. Default: 1 day.",
    )
    similarity_threshold: float = Field(
        default=0.85,
        description="Minimum similarity score (0-1) for question matching.",
    )
    context_similarity_threshold: float = Field(
        default=0.80,
        description="Minimum similarity score (0-1) for conversation context matching.",
    )
    question_weight: Optional[float] = Field(
        default=0.6,
        description="Weight for question similarity in the combined score (0-1). Computed as 1 - context_weight if omitted.",
    )
    context_weight: Optional[float] = Field(
        default=None,
        description="Weight for context similarity in the combined score (0-1). Computed as 1 - question_weight if omitted.",
    )
    embedding_model: str | LLMModel = Field(
        default="databricks-gte-large-en",
        description="Embedding model endpoint for generating similarity vectors.",
    )
    embedding_dims: int | None = Field(
        default=None,
        description="Embedding vector dimensions. Auto-detected from the model if not set.",
    )
    warehouse: WarehouseModel = Field(
        description="SQL warehouse used by the Genie API for query execution.",
    )
    context_window_size: int = Field(
        default=4,
        description="Number of previous conversation turns included as context for matching.",
    )
    max_context_tokens: int = Field(
        default=2000,
        description="Maximum token length for context to prevent oversized embeddings.",
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


class GenieContextAwareCacheParametersModel(GenieContextAwareCacheParametersBase):
    """
    Configuration for PostgreSQL-backed context-aware cache.

    Extends the base context-aware cache configuration with database-specific
    fields for table management and prompt history tracking.
    """

    database: DatabaseModel = Field(
        description="PostgreSQL or Lakebase database for persistent cache storage.",
    )
    table_name: str = Field(
        default="genie_context_aware_cache",
        description="Table name for storing cache entries in the database.",
    )
    prompt_history_table: str = Field(
        default="genie_prompt_history",
        description="Table name for storing prompt history used in context-aware matching.",
    )
    max_prompt_history_length: int = Field(
        default=50,
        description="Maximum number of prompts to keep per conversation.",
    )
    use_genie_api_for_history: bool = Field(
        default=False,
        description="Fall back to the Genie API when local prompt history is empty.",
    )
    prompt_history_ttl_seconds: int | None = Field(
        default=None,
        description="TTL for prompt history entries in seconds. None = use the cache TTL.",
    )
    ivfflat_lists: int | None = Field(
        default=None,
        description="Number of IVFFlat index lists for pg_vector. None = auto-computed as max(100, sqrt(row_count)).",
    )
    ivfflat_probes: int | None = Field(
        default=None,
        description="Number of IVFFlat lists to probe per query. None = auto-computed as max(10, sqrt(lists)).",
    )
    ivfflat_candidates: int = Field(
        default=20,
        description="Number of top-K candidates retrieved before Python-side reranking.",
    )

    @field_validator("table_name", "prompt_history_table")
    @classmethod
    def validate_sql_identifier(cls, v: str) -> str:
        """Validate that table names are safe SQL identifiers to prevent injection."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError(
                f"Invalid SQL identifier: {v!r}. "
                "Table names must start with a letter or underscore and contain "
                "only letters, digits, and underscores."
            )
        return v


# Memory estimation for capacity planning:
# - Each entry: ~20KB (8KB question embedding + 8KB context embedding + 4KB strings/overhead)
# - 1,000 entries: ~20MB (0.4% of 8GB)
# - 5,000 entries: ~100MB (2% of 8GB)
# - 10,000 entries: ~200MB (4-5% of 8GB) - default for ~30 users
# - 20,000 entries: ~400MB (8-10% of 8GB)
# Default 10,000 entries provides ~330 queries per user for 30 users.
class GenieInMemoryContextAwareCacheParametersModel(
    GenieContextAwareCacheParametersBase
):
    """
    Configuration for in-memory context-aware cache (no database required).

    This cache stores embeddings and cache entries entirely in memory, providing
    context-aware similarity matching without requiring external database dependencies
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

    For multi-instance deployments or large cache sizes, use GenieContextAwareCacheParametersModel
    with PostgreSQL backend instead.
    """

    time_to_live_seconds: int | None = Field(
        default=60 * 60 * 24 * 7,
        description="Cache entry TTL in seconds. Default: 1 week (604800s). None or negative = never expires.",
    )
    capacity: int | None = Field(
        default=10000,
        description="Maximum cache entries (~200MB at 10000 with 1024-dim embeddings). LRU eviction when full. None = unlimited.",
    )
    context_window_size: int = Field(
        default=3,
        description="Number of previous conversation turns included as context for matching.",
    )


class SearchParametersModel(BaseModel):
    """Tuning parameters for vector similarity search queries."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    num_results: Optional[int] = Field(
        default=10,
        description="Number of results to return per search query.",
    )
    filters: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Static metadata filters applied to every search (key-value pairs).",
    )
    query_type: Optional[str] = Field(
        default="ANN",
        description="Search algorithm type: 'ANN' (approximate nearest neighbor) or 'HYBRID'.",
    )


class InstructionAwareRerankModel(BaseModel):
    """
    LLM-based reranking considering user instructions and constraints.

    Use fast models (GPT-3.5, Haiku, Llama 3 8B) to minimize latency (~100ms).
    Runs AFTER FlashRank as an additional constraint-aware reranking stage.
    Skipped for 'standard' mode when auto_bypass=true in router config.

    Example:
        ```yaml
        instructed:
          columns:
            - name: brand_name
              type: string
          rerank:
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

    Supports two reranking options that can be combined:
    1. FlashRank (local cross-encoder) - set `model`
    2. Databricks server-side reranking - set `columns`

    For LLM instruction-aware reranking, use `instructed.rerank` instead.

    Example with FlashRank:
        ```yaml
        rerank:
          model: ms-marco-MiniLM-L-12-v2  # FlashRank model
          top_n: 10
        ```

    Example with Databricks columns:
        ```yaml
        rerank:
          columns:
            - product_name
            - brand_name
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

    The optional ``description`` field lets users annotate a column with semantic
    context (e.g. example values, business meaning).  Descriptions are embedded
    into JSON schemas and prompt context that pipeline components (decomposition,
    routing, verification, reranking) generate from the column metadata.
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
    description: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable description of the column for LLM context. "
            "Include example values or business meaning to improve filter accuracy "
            "(e.g. 'Brand/manufacturer (MILWAUKEE, DEWALT, etc.)')."
        ),
    )


class DecompositionModel(BaseModel):
    """
    Query decomposition settings for instructed retrieval.

    Decomposes complex user queries into multiple focused subqueries with
    metadata filters, executed in parallel and merged using Reciprocal Rank Fusion (RRF).

    Example:
        ```yaml
        instructed:
          decomposition:
            model: *fast_llm
            max_subqueries: 3
            rrf_k: 60
            normalize_filter_case: uppercase
            examples:
              - query: "cheap drills"
                filters: {"price <": 100}
        ```
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    model: Optional["LLMModel"] = Field(
        default=None,
        description="LLM for query decomposition (smaller/faster model recommended)",
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


class InstructedRetrieverModel(BaseModel):
    """
    Configuration for instructed retrieval with query decomposition and RRF merging.

    Groups all schema-aware, LLM-driven features: query decomposition, instruction-aware
    reranking, query routing, and result verification. These features share schema context
    (columns, constraints) and are co-located here to enforce that dependency at the type
    level.

    Column metadata is the single source of truth for schema context. Each pipeline
    component (decomposition, routing, verification, reranking) generates the specific
    context it needs from the structured ``columns`` data:
    - Decomposition embeds column info into the JSON schema for ``with_structured_output``
    - Routing generates a compact column summary
    - Verification generates a context with column descriptions (no operator syntax)
    - Reranking uses column names and types for instruction generation

    Example:
        ```yaml
        retriever:
          vector_store: *products_vector_store
          instructed:
            columns:
              - name: brand_name
                type: string
                description: "Brand/manufacturer (MILWAUKEE, DEWALT, etc.)"
              - name: price
                type: number
                operators: ["", "<", "<=", ">", ">="]
                description: "Product price in USD"
            constraints:
              - "Prefer recent products"
            decomposition:
              model: *fast_llm
              max_subqueries: 3
              examples:
                - query: "cheap drills"
                  filters: {"price <": 100}
            rerank:
              model: *fast_llm
              instructions: "Prioritize by brand preferences"
              top_n: 10
            router:
              model: *fast_llm
              default_mode: standard
            verifier:
              model: *fast_llm
              on_failure: warn_and_retry
        ```
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    columns: list[ColumnInfo] = Field(
        description=(
            "Structured column info used by all pipeline components. "
            "Column names, types, operators, and descriptions are embedded into "
            "JSON schemas and prompts for each component automatically."
        ),
    )
    constraints: Optional[list[str]] = Field(
        default=None, description="Default constraints to always apply"
    )
    decomposition: Optional[DecompositionModel] = Field(
        default=None,
        description="Query decomposition settings for breaking complex queries into subqueries.",
    )
    rerank: Optional[InstructionAwareRerankModel] = Field(
        default=None,
        description="Optional LLM-based instruction-aware reranking stage.",
    )
    router: Optional["RouterModel"] = Field(
        default=None,
        description="Optional query router for selecting execution mode (standard vs instructed).",
    )
    verifier: Optional["VerifierModel"] = Field(
        default=None,
        description="Optional result verification with structured feedback for retry.",
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
          instructed:
            columns:
              - name: brand_name
                type: string
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
          instructed:
            columns:
              - name: brand_name
                type: string
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
    """Retriever combining a vector store with search parameters, reranking, and instructed retrieval."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    vector_store: VectorStoreModel = Field(
        description="Vector search index configuration used for similarity search.",
    )
    columns: Optional[list[str]] = Field(
        default_factory=list,
        description="Columns to return from search results. Defaults to the vector store's columns.",
    )
    search_parameters: SearchParametersModel = Field(
        default_factory=SearchParametersModel,
        description="Search tuning: number of results, query type, and metadata filters.",
    )
    rerank: Optional[RerankParametersModel | bool] = Field(
        default=None,
        description="Optional reranking configuration. Set to true for defaults, or provide RerankParametersModel for custom settings.",
    )
    instructed: Optional[InstructedRetrieverModel] = Field(
        default=None,
        description="Optional instructed retrieval with query decomposition, instruction-aware reranking, routing, and verification.",
    )

    @model_validator(mode="after")
    def set_default_columns(self) -> Self:
        if not self.columns:
            columns: Sequence[str] = self.vector_store.columns
            self.columns = columns
        return self

    @model_validator(mode="after")
    def set_default_reranker(self) -> Self:
        """Convert bool to RerankParametersModel with defaults.

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
    INLINE = "inline"


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
        seen: set[str] = set()
        unique_decisions: list[Literal["approve", "edit", "reject"]] = []
        for decision in self.allowed_decisions:
            if decision not in seen:
                seen.add(decision)
                unique_decisions.append(decision)
        self.allowed_decisions = unique_decisions

        return self


class BaseFunctionModel(ABC, BaseModel):
    """Base class for all function/tool implementations (Python, factory, inline, MCP, UC)."""

    model_config = ConfigDict(
        use_enum_values=True,
        discriminator="type",
    )
    type: FunctionType = Field(
        description="Function type discriminator (python, factory, inline, mcp, unity_catalog).",
    )
    human_in_the_loop: Optional[HumanInTheLoopModel] = Field(
        default=None,
        description="Human-in-the-loop approval configuration for this tool.",
    )

    @abstractmethod
    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]: ...

    @field_serializer("type")
    def serialize_type(self, value) -> str:
        # Handle both enum objects and already-converted strings
        if isinstance(value, FunctionType):
            return value.value
        return str(value)


class PythonFunctionModel(BaseFunctionModel, HasFullName):
    """A tool implemented as a Python function, imported by fully qualified name."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.PYTHON] = Field(
        default=FunctionType.PYTHON,
        description="Function type discriminator. Must be 'python'.",
    )
    name: str = Field(
        description="Fully qualified Python function name (e.g., 'my_package.tools.my_tool').",
    )

    @property
    def full_name(self) -> str:
        return self.name

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_python_tool

        return [create_python_tool(self, **kwargs)]


class FactoryFunctionModel(BaseFunctionModel, HasFullName):
    """A tool created by calling a factory function with optional arguments."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.FACTORY] = Field(
        default=FunctionType.FACTORY,
        description="Function type discriminator. Must be 'factory'.",
    )
    name: str = Field(
        description="Fully qualified factory function name that returns a tool or list of tools.",
    )
    args: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Keyword arguments passed to the factory function.",
    )

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


class InlineFunctionModel(BaseFunctionModel):
    """
    Inline function model for defining tool code directly in YAML configuration.

    This allows you to define simple tools without creating separate Python files.
    The code should define a function decorated with @tool from langchain.tools.

    SECURITY WARNING: This model uses exec() to execute arbitrary Python code
    from the YAML configuration. Only load configurations from trusted sources.
    A malicious configuration could execute arbitrary code on the host system.

    Example YAML:
        tools:
          calculator:
            name: calculator
            function:
              type: inline
              code: |
                from langchain.tools import tool

                @tool
                def calculator(expression: str) -> str:
                    '''Evaluate a mathematical expression.'''
                    return str(eval(expression))

    The code block must:
    - Import @tool from langchain.tools
    - Define exactly one function decorated with @tool
    - The function name becomes the tool name
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.INLINE] = Field(
        default=FunctionType.INLINE,
        description="Function type discriminator. Must be 'inline'.",
    )
    code: str = Field(
        ...,
        description="Python code defining a tool function decorated with @tool",
    )

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        """Execute the inline code and return the tool(s) defined in it.

        SECURITY WARNING: This method uses exec() to run arbitrary Python code.
        Only use with trusted configuration sources.
        """
        from langchain_core.tools import BaseTool

        logger.warning(
            "Executing inline tool code - ensure this code comes from a trusted source",
            code_preview=self.code[:100],
        )

        # Create a namespace for executing the code
        namespace: dict[str, Any] = {}

        # Execute the code in the namespace
        try:
            exec(self.code, namespace)  # noqa: S102
        except Exception as e:
            raise ValueError(f"Failed to execute inline tool code: {e}") from e

        # Find all tools (functions decorated with @tool) in the namespace
        tools: list[RunnableLike] = []
        for name, obj in namespace.items():
            if isinstance(obj, BaseTool):
                tools.append(obj)

        if not tools:
            raise ValueError(
                "Inline code must define at least one function decorated with @tool. "
                "Make sure to import and use: from langchain.tools import tool"
            )

        logger.debug(
            "Created inline tools",
            tool_names=[t.name for t in tools if hasattr(t, "name")],
        )
        return tools


class TransportType(str, Enum):
    """MCP transport protocol."""

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
    type: Literal[FunctionType.MCP] = Field(
        default=FunctionType.MCP,
        description="Function type discriminator. Must be 'mcp'.",
    )
    transport: TransportType = Field(
        default=TransportType.STREAMABLE_HTTP,
        description="MCP transport protocol: streamable_http (default) or stdio.",
    )
    command: Optional[str] = Field(
        default="python",
        description="Executable command for STDIO transport (e.g., 'python', 'node').",
    )
    url: Optional[AnyVariable] = Field(
        default=None,
        description="Direct MCP server URL. Mutually exclusive with app, connection, genie_room, sql, vector_search, functions.",
    )
    headers: dict[str, AnyVariable] = Field(
        default_factory=dict,
        description="HTTP headers sent with MCP requests (e.g., authorization tokens).",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Command-line arguments for STDIO transport.",
    )
    app: Optional[DatabricksAppModel] = Field(
        default=None,
        description="Databricks App whose /mcp endpoint serves MCP tools.",
    )
    connection: Optional[ConnectionModel] = Field(
        default=None,
        description="Unity Catalog connection for external MCP servers.",
    )
    functions: Optional[SchemaModel] = Field(
        default=None,
        description="Unity Catalog schema whose functions are exposed as MCP tools.",
    )
    genie_room: Optional[GenieRoomModel] = Field(
        default=None,
        description="Genie space exposed as an MCP server for natural-language SQL.",
    )
    sql: Optional[bool] = Field(
        default=None,
        description="Enable the Databricks SQL MCP server (serverless, workspace-level).",
    )
    vector_search: Optional[VectorStoreModel] = Field(
        default=None,
        description="Vector search index exposed as an MCP server.",
    )
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
    """A tool backed by a Unity Catalog SQL function."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.UNITY_CATALOG] = Field(
        default=FunctionType.UNITY_CATALOG,
        description="Function type discriminator. Must be 'unity_catalog'.",
    )
    resource: FunctionModel = Field(
        description="Unity Catalog function reference.",
    )
    partial_args: Optional[dict[str, AnyVariable]] = Field(
        default_factory=dict,
        description="Pre-filled arguments automatically injected when the function is called.",
    )

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_uc_tools

        return create_uc_tools(self)


AnyTool: TypeAlias = (
    Union[
        PythonFunctionModel,
        FactoryFunctionModel,
        InlineFunctionModel,
        UnityCatalogFunctionModel,
        McpFunctionModel,
    ]
    | str
)


class ToolModel(BaseModel):
    """A named tool binding an identifier to a function implementation."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Display name for the tool shown to the LLM during function calling.",
    )
    function: AnyTool = Field(
        description="Function implementation: Python, factory, inline, Unity Catalog, MCP, or a reference string.",
    )


class PromptModel(BaseModel, HasFullName):
    """A prompt backed by the MLflow Prompt Registry with versioning and alias support."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(
        default=None,
        alias="schema",
        description="Unity Catalog schema qualifying the prompt name (catalog.schema.name).",
    )
    name: str = Field(
        description="Prompt name in the MLflow Prompt Registry.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description stored with the prompt in the registry.",
    )
    default_template: Optional[str] = Field(
        default=None,
        description="Inline template text registered when auto_register is true and no registry entry exists.",
    )
    alias: Optional[str] = Field(
        default=None,
        description="Prompt alias to load (e.g., 'latest', 'champion'). Mutually exclusive with version.",
    )
    version: Optional[int] = Field(
        default=None,
        description="Specific prompt version number to load. Mutually exclusive with alias.",
    )
    tags: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Key-value tags attached to the prompt version in the registry.",
    )
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
    def jinja_template(self) -> str:
        """Return the template in Jinja2 format (with {{ }} variables).

        Unlike ``template`` which converts to single-brace Python format,
        this property ensures the template uses Jinja2 double-brace
        variables (e.g. ``{{ inputs }}``, ``{{ outputs }}``) required by
        MLflow judges.

        If the registry stores the older single-brace format
        (``{inputs}``), the known MLflow judge variables are automatically
        converted to double-brace Jinja2 syntax.
        """
        import re

        from dao_ai.providers.databricks import DatabricksProvider

        provider: DatabricksProvider = DatabricksProvider()
        prompt_version = provider.get_prompt(self)
        raw_template: str = prompt_version.template

        # Convert single-brace MLflow judge variables to Jinja2 double-brace
        # format when the template was stored in legacy format.
        _JUDGE_VARS = ("inputs", "outputs", "trace", "expectations", "conversation")
        for var in _JUDGE_VARS:
            # Match {var} but NOT {{var}} (already Jinja2)
            raw_template = re.sub(
                r"(?<!\{)\{" + var + r"\}(?!\})",
                "{{ " + var + " }}",
                raw_template,
            )

        return raw_template

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
    """Configuration for a guardrail.

    Guardrails evaluate agent responses against quality or safety criteria.
    Two configuration modes are supported:

    1. **Custom (LLM-judge)** -- provide *model* and *prompt*.  A
       ``JudgeScorer`` is created using ``mlflow.genai.judges.make_judge``.
    2. **Scorer-based** -- provide *scorer* (and optionally *scorer_args*).
       Any ``mlflow.genai.scorers.base.Scorer`` class can be used,
       including built-in ``GuardrailsScorer`` validators such as
       ``ToxicLanguage`` and ``DetectPII``.

    The two modes are mutually exclusive.

    Attributes:
        name: Name identifying this guardrail.
        model: LLM model for the judge.  Accepts a string (model name) or
            ``LLMModel``.  Required when using the custom judge mode.
        prompt: Evaluation instructions using ``{{ inputs }}`` and
            ``{{ outputs }}`` template variables.  Required when using
            the custom judge mode.
        scorer: Fully qualified name of an MLflow ``Scorer`` class
            (e.g. ``"mlflow.genai.scorers.guardrails.DetectPII"``).
            Required when using the scorer-based mode.
        scorer_args: Keyword arguments forwarded to the scorer constructor
            (e.g. ``{"pii_entities": ["CREDIT_CARD", "SSN"]}``).
        hub: Optional guardrails-ai hub URI for the scorer validator
            (e.g. ``"hub://guardrails/toxic_language"``).  When set,
            the validator is auto-installed at startup if the
            ``GUARDRAILSAI_API_KEY`` environment variable is present.
            Only valid when *scorer* is also set.
        num_retries: Maximum retry attempts when evaluation fails (default: 3).
        fail_on_error: If True, block responses when the evaluation call
            itself errors (e.g. scorer exception, network timeout).
            If False (default), let responses through on evaluation
            errors.
        max_context_length: Max character length for extracted tool context
            (default: 8000).
        apply_to: When to run this guardrail.  ``"input"`` runs before
            the model (on user messages), ``"output"`` runs after the
            model (on agent responses), ``"both"`` runs in both places
            (default: ``"both"``).
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Name identifying this guardrail.",
    )
    model: Optional[str | LLMModel] = Field(
        default=None,
        description="LLM model for the judge. Required for custom judge mode.",
    )
    prompt: Optional[str | PromptModel] = Field(
        default=None,
        description="Evaluation instructions using {{ inputs }} and {{ outputs }} template variables. Required for custom judge mode.",
    )
    scorer: Optional[str] = Field(
        default=None,
        description="Fully qualified name of an MLflow Scorer class (e.g., 'mlflow.genai.scorers.guardrails.DetectPII'). Required for scorer-based mode.",
    )
    scorer_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the scorer constructor (e.g., {'pii_entities': ['CREDIT_CARD', 'SSN']}).",
    )
    hub: Optional[str] = Field(
        default=None,
        description="Guardrails-AI hub URI for auto-installing the scorer validator (e.g., 'hub://guardrails/toxic_language'). Requires scorer.",
    )
    num_retries: Optional[int] = Field(
        default=3,
        description="Maximum retry attempts when the evaluation call fails.",
    )
    fail_on_error: Optional[bool] = Field(
        default=False,
        description="If true, block responses when the evaluation itself errors. If false, let responses through on errors.",
    )
    max_context_length: Optional[int] = Field(
        default=8000,
        description="Maximum character length for extracted tool context passed to the guardrail.",
    )
    apply_to: Literal["input", "output", "both"] = Field(
        default="both",
        description="When to run: 'input' (before model), 'output' (after model), or 'both'.",
    )

    @model_validator(mode="after")
    def validate_guardrail_type(self) -> Self:
        has_scorer: bool = self.scorer is not None
        has_judge: bool = self.model is not None or self.prompt is not None

        if has_scorer and has_judge:
            raise ValueError(
                "Cannot specify both 'scorer' and 'model'/'prompt'. "
                "Use either scorer-based or custom judge configuration."
            )
        if not has_scorer and not has_judge:
            raise ValueError(
                "Either 'scorer' or both 'model' and 'prompt' must be provided."
            )
        if not has_scorer and (self.model is None or self.prompt is None):
            raise ValueError(
                "Both 'model' and 'prompt' are required for custom judge guardrails."
            )
        if self.hub is not None and not has_scorer:
            raise ValueError(
                "'hub' requires 'scorer' to be set. The hub URI identifies "
                "the guardrails-ai hub package for the scorer validator."
            )
        return self

    @model_validator(mode="after")
    def validate_llm_model(self) -> Self:
        if self.model is not None and isinstance(self.model, str):
            self.model = LLMModel(name=self.model)
        return self

    def as_scorer(self) -> Any:
        """Return an MLflow ``Scorer`` instance for this guardrail.

        For scorer-based guardrails, imports and instantiates the class
        referenced by ``self.scorer`` with ``self.scorer_args``.  When
        ``self.hub`` is set, the hub validator is auto-installed first.

        For LLM-judge guardrails, creates a ``JudgeScorer`` wrapping
        ``mlflow.genai.judges.make_judge`` with the resolved prompt and
        model endpoint.
        """
        if self.scorer:
            if self.hub:
                from dao_ai.guardrails_hub import ensure_single_hub_validator

                ensure_single_hub_validator(self.hub)

            from dao_ai.utils import load_function

            scorer_cls = load_function(self.scorer)
            return scorer_cls(**self.scorer_args)

        from dao_ai.middleware._prompt_utils import resolve_prompt
        from dao_ai.middleware.guardrails import JudgeScorer

        template: str = resolve_prompt(self.prompt, jinja=True)
        return JudgeScorer(
            name=self.name,
            instructions=template,
            model=self.model.uri,
        )


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
    """Conversation state checkpointer for persisting LangGraph thread state across turns."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Unique name for this checkpointer instance.",
    )
    database: Optional[DatabaseModel] = Field(
        default=None,
        description="Database for persistent storage. If omitted, uses in-memory storage (lost on restart).",
    )

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
    """Long-term memory store for cross-thread memories (user profiles, preferences, episodes)."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Unique name for this store instance.",
    )
    embedding_model: Optional[LLMModel] = Field(
        default=None,
        description="Embedding model for semantic memory search. Required for vector-based recall.",
    )
    dims: Optional[int] = Field(
        default=None,
        description="Embedding dimensions. Auto-detected from the model if not set.",
    )
    database: Optional[DatabaseModel] = Field(
        default=None,
        description="Database for persistent storage. If omitted, uses in-memory storage (lost on restart).",
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Namespace prefix for memory keys, enabling multi-tenant isolation.",
    )

    @property
    def storage_type(self) -> StorageType:
        """Infer storage type from database presence."""
        return StorageType.POSTGRES if self.database else StorageType.MEMORY

    def as_store(self) -> BaseStore:
        from dao_ai.memory import StoreManager

        store: BaseStore = StoreManager.instance(self).store()
        return store


MemorySchemaName: TypeAlias = Literal["user_profile", "preference", "episode"]


class MemoryExtractionModel(BaseModel):
    """Configuration for automatic memory extraction and injection.

    Controls how the system automatically extracts memories from
    conversations and injects relevant context into prompts.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    schemas: Optional[list[MemorySchemaName]] = Field(
        default=None,
        description=(
            "Schema names for structured extraction "
            "(e.g. ['user_profile', 'preference', 'episode']). "
            "When None, uses unstructured string memories."
        ),
    )
    instructions: Optional[str] = Field(
        default=None,
        description=(
            "Custom extraction instructions guiding what to remember. "
            "When None, uses langmem's default instructions."
        ),
    )
    auto_inject: bool = Field(
        default=True,
        description=(
            "Automatically search and inject relevant memories into "
            "the system prompt before each model call."
        ),
    )
    auto_inject_limit: int = Field(
        default=5,
        description="Maximum number of memories to inject into the prompt.",
    )
    background_extraction: bool = Field(
        default=True,
        description=(
            "Extract memories in a background thread after each "
            "conversation turn (no latency impact on responses)."
        ),
    )
    extraction_model: Optional[LLMModel] = Field(
        default=None,
        description=(
            "Separate LLM for memory extraction. Can be a smaller, "
            "cheaper model. When None, uses the agent's primary model."
        ),
    )
    query_model: Optional[LLMModel] = Field(
        default=None,
        description=(
            "Separate LLM for optimizing memory search queries. "
            "When None, embeds the raw user message directly."
        ),
    )


class MemoryModel(BaseModel):
    """Memory configuration combining state checkpointing, long-term memory storage, and automatic extraction."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    checkpointer: Optional[CheckpointerModel] = Field(
        default=None,
        description="Checkpointer for persisting conversation thread state across turns.",
    )
    store: Optional[StoreModel] = Field(
        default=None,
        description="Long-term memory store for cross-thread knowledge (profiles, preferences, episodes).",
    )
    extraction: Optional[MemoryExtractionModel] = Field(
        default=None,
        description="Automatic memory extraction and injection settings.",
    )


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
    name: str = Field(
        description="Unique agent name used for identification in multi-agent orchestration.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description shown when the LLM selects handoff targets.",
    )
    model: LLMModel = Field(
        description="LLM model configuration (serving endpoint name, temperature, etc.).",
    )
    tools: list[ToolModel] = Field(
        default_factory=list,
        description="Tools available to this agent during reasoning.",
    )
    guardrails: list[GuardrailModel] = Field(
        default_factory=list,
        description="Guardrails that evaluate this agent's inputs and/or outputs.",
    )
    prompt: Optional[str | PromptModel] = Field(
        default=None,
        description="System prompt as an inline string or a PromptModel referencing the MLflow Prompt Registry.",
    )
    handoff_prompt: Optional[str] = Field(
        default=None,
        description="Additional instructions appended to the prompt during multi-agent handoff.",
    )
    middleware: list[MiddlewareModel] = Field(
        default_factory=list,
        description="List of middleware to apply to this agent.",
    )
    response_format: Optional[ResponseFormatModel | type | str] = Field(
        default=None,
        description="Structured output format (Pydantic type, JSON schema, or ResponseFormatModel).",
    )

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
        from dao_ai.prompts import get_cached_prompt_versions

        graph: CompiledStateGraph = self.as_runnable()
        prompt_versions = get_cached_prompt_versions()
        return create_responses_agent(graph, prompt_versions=prompt_versions)


class SupervisorModel(BaseModel):
    """Configuration for the supervisor agent in a supervisor orchestration pattern."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel = Field(
        description="LLM model used by the supervisor to route tasks to sub-agents.",
    )
    tools: list[ToolModel] = Field(
        default_factory=list,
        description="Tools available directly to the supervisor agent.",
    )
    prompt: Optional[str | PromptModel] = Field(
        default=None,
        description="System prompt for the supervisor agent.",
    )
    middleware: list[MiddlewareModel] = Field(
        default_factory=list,
        description="List of middleware to apply to the supervisor.",
    )


class HandoffRouteModel(BaseModel):
    """
    Configuration model for a handoff route in a swarm.

    A handoff route specifies a target agent and whether the handoff should be
    deterministic (always route to this agent) or agentic (LLM decides via tool call).

    When ``is_deterministic`` is ``True``, the source agent will **always** transfer
    control to this target agent after completing its turn, without requiring the
    LLM to invoke a handoff tool. This is useful for pipeline-style workflows
    where the routing order is predetermined.

    When ``is_deterministic`` is ``False`` (the default), a handoff tool is created
    for the target agent and the LLM decides when to invoke it. This is the
    standard agentic handoff behavior.

    Example YAML::

        handoffs:
          triage_agent:
            - agent: billing_agent
              is_deterministic: true
          billing_agent:
            - support_agent            # shorthand for agentic handoff
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    agent: AgentModel | str = Field(
        description="The target agent to hand off to, specified as an AgentModel or agent name string.",
    )
    is_deterministic: bool = Field(
        default=False,
        description=(
            "When true, the handoff is deterministic: control always transfers to this "
            "agent after the source agent completes its turn, without LLM tool-call routing. "
            "When false (default), a handoff tool is created and the LLM decides when to invoke it."
        ),
    )


class SwarmModel(BaseModel):
    """Configuration for swarm-style multi-agent orchestration with agent-to-agent handoffs."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    default_agent: Optional[AgentModel | str] = Field(
        default=None,
        description="The initial agent that receives user messages. Defaults to the first agent.",
    )
    middleware: list[MiddlewareModel] = Field(
        default_factory=list,
        description="List of middleware to apply to all agents in the swarm",
    )
    handoffs: Optional[
        dict[str, Optional[list[AgentModel | str | HandoffRouteModel]]]
    ] = Field(
        default_factory=dict,
        description=(
            "Mapping of agent names to their allowed handoff targets. "
            "Each target can be an agent name (str), an AgentModel, or a "
            "HandoffRouteModel for deterministic routing. "
            "Use null (~) to allow handoffs to all agents."
        ),
    )


class OrchestrationModel(BaseModel):
    """Multi-agent orchestration configuration. Exactly one of supervisor or swarm must be specified."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    supervisor: Optional[SupervisorModel] = Field(
        default=None,
        description="Supervisor pattern: a central LLM routes tasks to sub-agents.",
    )
    swarm: Optional[SwarmModel | Literal[True]] = Field(
        default=None,
        description="Swarm pattern: agents hand off to each other via tool calls. Set to true for defaults.",
    )
    memory: Optional[MemoryModel] = Field(
        default=None,
        description="Memory configuration scoped to the orchestration layer (checkpointer, store, extraction).",
    )

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
    """Unity Catalog registered model where the agent artifact is logged."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(
        default=None,
        alias="schema",
        description="Schema reference qualifying the model name.",
    )
    name: str = Field(
        description="Registered model name (short) or fully qualified (catalog.schema.model).",
    )

    @property
    def full_name(self) -> str:
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name


class Entitlement(str, Enum):
    """Access control entitlements for serving endpoints and apps."""

    CAN_MANAGE = "CAN_MANAGE"
    CAN_QUERY = "CAN_QUERY"
    CAN_VIEW = "CAN_VIEW"
    CAN_REVIEW = "CAN_REVIEW"
    NO_PERMISSIONS = "NO_PERMISSIONS"


class AppPermissionModel(BaseModel):
    """Access control entry granting entitlements to principals on a serving endpoint."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    principals: list[ServicePrincipalModel | str] = Field(
        default_factory=list,
        description="Users, groups, or service principals receiving the entitlements.",
    )
    entitlements: list[Entitlement] = Field(
        description="Entitlements to grant (CAN_MANAGE, CAN_QUERY, CAN_VIEW, CAN_REVIEW).",
    )

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
    """Logging verbosity level."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class WorkloadSize(str, Enum):
    """Model Serving workload size controlling compute resources."""

    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"


class MessageRole(str, Enum):
    """Role of a message in a chat conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single chat message with a role and content."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    role: MessageRole = Field(
        description="Message role: user, assistant, or system.",
    )
    content: str = Field(
        description="Message text content.",
    )


class ChatPayload(BaseModel):
    """Chat request payload containing messages and optional custom inputs."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    input: Optional[list[Message]] = Field(
        default=None,
        description="Chat messages (alias for 'messages'). Provide either input or messages.",
    )
    messages: Optional[list[Message]] = Field(
        default=None,
        description="Chat messages (alias for 'input'). Provide either messages or input.",
    )
    custom_inputs: Optional[dict] = Field(
        default_factory=dict,
        description="Extra inputs forwarded to the agent (e.g., configurable with thread_id).",
    )

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
    model: LLMModel = Field(
        description="LLM used to generate conversation summaries.",
    )
    max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum tokens to keep after summarization.",
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


class GuidelineModel(BaseModel):
    """A named set of evaluation guidelines used by the Guidelines scorer."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Unique name for this guideline set.",
    )
    guidelines: list[str] = Field(
        description="List of guideline statements the scorer evaluates responses against.",
    )


class MonitoringModel(BaseModel):
    """
    Configuration for production monitoring of GenAI scorers.

    Controls which scorers are registered and at what sampling rates against
    production traces stored in Unity Catalog via the MLflow 3 scorer
    lifecycle API.

    Attributes:
        sample_rate: Sampling rate for built-in scorers. Defaults to 1.0 (100%).
        scorers: Optional list of built-in scorer names to enable. When omitted,
            all built-in scorers are registered (safety, completeness,
            relevance_to_query, tool_call_efficiency).
        guidelines: Optional list of guideline configurations for Guidelines
            scorers used in production monitoring.
        guidelines_sample_rate: Sampling rate for Guidelines scorers, which
            invoke an LLM judge per trace and are more expensive. Defaults
            to 0.5 (50%).
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Sampling rate for built-in scorers (0.0–1.0)",
    )
    scorers: Optional[list[str | GuardrailModel]] = Field(
        default=None,
        description="Built-in scorer names, glob patterns, or GuardrailModel references to enable. "
        "Built-in options: safety, completeness, relevance_to_query, tool_call_efficiency. "
        "Supports glob patterns: '*' (all built-in scorers), 'safe*', etc. "
        "GuardrailModel entries are converted to scorers via as_scorer(). "
        "Defaults to all built-in scorers when omitted.",
    )
    guidelines: list[GuidelineModel] = Field(
        default_factory=list,
        description="Guideline configurations for production monitoring Guidelines scorers.",
    )
    guidelines_sample_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Sampling rate for Guidelines scorers (0.0–1.0)",
    )


class TraceLocationModel(BaseModel):
    """Unity Catalog location for storing MLflow traces in OTEL-format Delta tables.

    Accepts either a SchemaModel reference (aliased to "schema") or a string
    in "catalog.schema" format. When configured on AppModel, traces are stored
    in UC Delta tables via set_experiment_trace_location().
    """

    OTEL_TABLE_SUFFIXES: ClassVar[Sequence[str]] = (
        "mlflow_experiment_trace_otel_spans",
        "mlflow_experiment_trace_otel_logs",
        "mlflow_experiment_trace_otel_metrics",
        "mlflow_experiment_trace_metadata",
        "mlflow_experiment_trace_unified",
    )

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", populate_by_name=True
    )
    schema_model: SchemaModel = Field(
        alias="schema",
        description="Unity Catalog schema (catalog.schema) where OTEL trace tables are stored.",
    )
    warehouse: Union[WarehouseModel, str] = Field(
        description="SQL warehouse for creating views and querying traces. "
        "Accepts a WarehouseModel reference or a warehouse ID string.",
    )

    @model_validator(mode="before")
    @classmethod
    def parse_string_schema(cls, data: Any) -> Any:
        """Accept 'catalog.schema' string shorthand."""
        if isinstance(data, str):
            parts = data.split(".")
            if len(parts) != 2:
                raise ValueError(
                    "trace_location string must be 'catalog_name.schema_name'"
                )
            return {"schema": {"catalog_name": parts[0], "schema_name": parts[1]}}
        return data

    @property
    def warehouse_id(self) -> str:
        """Resolve warehouse to a warehouse ID string."""
        if isinstance(self.warehouse, WarehouseModel):
            return value_of(self.warehouse.warehouse_id)
        return self.warehouse

    @property
    def catalog_name(self) -> str:
        return value_of(self.schema_model.catalog_name)

    @property
    def schema_name(self) -> str:
        return value_of(self.schema_model.schema_name)

    def as_resources(self) -> Sequence[DatabricksResource]:
        """Return DatabricksTable resources for the OTEL trace tables.

        Model serving needs SELECT on these tables for set_experiment_trace_location()
        to succeed at startup. Including them as system resources ensures the
        auth policy grants the serving identity appropriate permissions.
        """
        schema_prefix = f"{self.catalog_name}.{self.schema_name}"
        return [
            DatabricksTable(table_name=f"{schema_prefix}.{suffix}")
            for suffix in self.OTEL_TABLE_SUFFIXES
        ]


class AppModel(BaseModel):
    """Application-level configuration for deployment, model registration, and orchestration."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Unique application name used for the serving endpoint and model registration.",
    )
    service_principal: Optional[ServicePrincipalModel] = Field(
        default=None,
        description="Service principal credentials injected as environment variables during Model Serving deployment.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the application.",
    )
    log_level: Optional[LogLevel] = Field(
        default="WARNING",
        description="Logging verbosity level (TRACE, DEBUG, INFO, WARNING, ERROR).",
    )
    registered_model: RegisteredModelModel = Field(
        description="Unity Catalog registered model where the agent is logged.",
    )
    endpoint_name: Optional[str] = Field(
        default=None,
        description="Model Serving endpoint name. Defaults to the app name if not specified.",
    )
    tags: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Key-value tags attached to the registered model version.",
    )
    scale_to_zero: Optional[bool] = Field(
        default=True,
        description="Whether the serving endpoint scales to zero when idle.",
    )
    environment_vars: Optional[dict[str, AnyVariable]] = Field(
        default_factory=dict,
        description="Environment variables set on the serving endpoint or Databricks App.",
    )
    budget_policy_id: Optional[str] = Field(
        default=None,
        description="Databricks budget policy ID for cost attribution.",
    )
    workload_size: Optional[WorkloadSize] = Field(
        default="Small",
        description="Model Serving workload size (Small, Medium, Large).",
    )
    permissions: Optional[list[AppPermissionModel]] = Field(
        default_factory=list,
        description="Access control list for the serving endpoint.",
    )
    agents: list[AgentModel] = Field(
        default_factory=list,
        description="List of agent definitions. At least one is required.",
    )

    orchestration: Optional[OrchestrationModel] = Field(
        default=None,
        description="Multi-agent orchestration mode (supervisor or swarm). Auto-configured if omitted.",
    )
    alias: Optional[str] = Field(
        default=None,
        description="Model version alias (e.g., 'champion') assigned after registration.",
    )
    initialization_hooks: Optional[FunctionHook | list[FunctionHook]] = Field(
        default_factory=list,
        description="Functions called once at startup after config is loaded.",
    )
    shutdown_hooks: Optional[FunctionHook | list[FunctionHook]] = Field(
        default_factory=list,
        description="Functions called on graceful shutdown.",
    )
    input_example: Optional[ChatPayload] = Field(
        default=None,
        description="Example chat payload logged alongside the model for documentation and testing.",
    )
    chat_history: Optional[ChatHistoryModel] = Field(
        default=None,
        description="Chat history summarization settings to manage long conversations.",
    )
    code_paths: list[str] = Field(
        default_factory=list,
        description="Additional Python file paths bundled with the model artifact.",
    )
    pip_requirements: list[str] = Field(
        default_factory=list,
        description="Extra pip packages installed in the serving environment.",
    )
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
    trace_location: Optional[TraceLocationModel] = Field(
        default=None,
        description="Unity Catalog location for storing MLflow traces in OTEL-format Delta tables. "
        "Accepts a schema reference or 'catalog.schema' string. "
        "When set, set_experiment_trace_location() is called at startup for both "
        "Model Serving and Databricks Apps deployments.",
    )
    monitoring: Optional[MonitoringModel] = Field(
        default=None,
        description="Production monitoring configuration. When present, scorers are "
        "registered to continuously evaluate production traces. Works with both "
        "experiment-based traces and UC OTEL trace tables. When trace_location is "
        "also configured, the SQL warehouse from trace_location is used for monitoring.",
    )

    @model_validator(mode="after")
    def set_databricks_env_vars(self) -> Self:
        """Set Databricks environment variables for Model Serving.

        Sets DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET,
        and OTEL trace destination env vars when trace_location is configured.
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

        # Set OTEL trace destination env vars when trace_location is configured
        if self.trace_location is not None:
            if "MLFLOW_TRACING_DESTINATION" not in self.environment_vars:
                self.environment_vars["MLFLOW_TRACING_DESTINATION"] = (
                    f"{self.trace_location.catalog_name}.{self.trace_location.schema_name}"
                )
            if "MLFLOW_TRACING_SQL_WAREHOUSE_ID" not in self.environment_vars:
                self.environment_vars["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = (
                    self.trace_location.warehouse_id
                )
        return self

    @model_validator(mode="after")
    def validate_agents_not_empty(self) -> Self:
        if not self.agents:
            raise ValueError("At least one agent must be specified")
        return self

    @staticmethod
    def _find_secret_source(
        value: Any,
    ) -> "SecretVariableModel | None":
        """Return the SecretVariableModel if *value* is one, or wraps one as
        the first option of a CompositeVariableModel.

        This mirrors the logic in ``_resolve_variable_type`` used by
        Databricks Apps deployment so that Model Serving ``environment_vars``
        consistently receive the ``{{secrets/scope/key}}`` format the
        serving infrastructure expects.
        """
        if isinstance(value, SecretVariableModel):
            return value
        if isinstance(value, CompositeVariableModel) and value.options:
            first = value.options[0]
            if isinstance(first, SecretVariableModel):
                return first
        return None

    @model_validator(mode="after")
    def resolve_environment_vars(self) -> Self:
        for key, value in self.environment_vars.items():
            updated_value: str
            secret_source = self._find_secret_source(value)
            if secret_source is not None:
                updated_value = str(secret_source)
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


class EvaluationModel(BaseModel):
    """
    Configuration for MLflow GenAI offline evaluation.

    Attributes:
        model: LLM model used as the judge for LLM-based scorers (e.g., Guidelines, Safety).
               This model evaluates agent responses during evaluation.
        table: Table to store evaluation results.
        num_evals: Number of evaluation samples to generate.
        replace: If True, drop and recreate the evaluation table and dataset.
            If False, reuse existing resources. Defaults to False.
        agent_description: Description of the agent for evaluation data generation.
        question_guidelines: Guidelines for generating evaluation questions.
        custom_inputs: Custom inputs to pass to the agent during evaluation.
        guidelines: List of guideline configurations for Guidelines scorers.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: LLMModel = Field(
        ..., description="LLM model used as the judge for LLM-based evaluation scorers."
    )
    table: TableModel = Field(
        description="Unity Catalog table where evaluation results are stored.",
    )
    num_evals: int = Field(
        description="Number of evaluation samples to generate from the agent.",
    )
    replace: bool = Field(
        default=False,
        description="If True, drop and recreate the evaluation table and dataset. "
        "If False, reuse existing resources.",
    )
    agent_description: Optional[str] = Field(
        default=None,
        description="Description of the agent used when generating synthetic evaluation questions.",
    )
    question_guidelines: Optional[str] = Field(
        default=None,
        description="Guidelines for the synthetic question generator (e.g., topic focus, difficulty).",
    )
    custom_inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra key-value inputs forwarded to the agent during evaluation runs.",
    )
    guidelines: list[GuidelineModel] = Field(
        default_factory=list,
        description="Guideline configurations for Guidelines scorers used during evaluation.",
    )

    @property
    def judge_model_endpoint(self) -> str:
        """
        Get the judge model endpoint string for MLflow scorers.

        Returns:
            Endpoint string in format 'databricks:/model-name'
        """
        return f"databricks:/{self.model.name}"


class EvaluationDatasetExpectationsModel(BaseModel):
    """Expected outcomes for an evaluation entry. Provide one of expected_response or expected_facts."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    expected_response: Optional[str] = Field(
        default=None,
        description="Full expected response text for correctness scoring.",
    )
    expected_facts: Optional[list[str]] = Field(
        default=None,
        description="List of facts the response should contain for fact-based correctness scoring.",
    )

    @model_validator(mode="after")
    def validate_mutually_exclusive(self) -> Self:
        if self.expected_response is not None and self.expected_facts is not None:
            raise ValueError("Cannot specify both expected_response and expected_facts")
        return self


class EvaluationDatasetEntryModel(BaseModel):
    """A single evaluation example pairing input messages with expected outcomes."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    inputs: ChatPayload = Field(
        description="Chat messages to send to the agent as evaluation input.",
    )
    expectations: EvaluationDatasetExpectationsModel = Field(
        description="Expected response or facts for scoring the agent's output.",
    )

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
    """An MLflow evaluation dataset containing input/expectation pairs."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    schema_model: Optional[SchemaModel] = Field(
        default=None,
        alias="schema",
        description="Schema reference qualifying the dataset name.",
    )
    name: str = Field(
        description="Dataset name in the MLflow registry.",
    )
    data: Optional[list[EvaluationDatasetEntryModel]] = Field(
        default_factory=list,
        description="Inline evaluation entries merged into the dataset on creation.",
    )
    overwrite: Optional[bool] = Field(
        default=False,
        description="If true, delete and recreate the dataset. If false, reuse the existing one.",
    )

    def as_dataset(self) -> EvaluationDataset:
        evaluation_dataset: EvaluationDataset
        needs_creation: bool = False

        try:
            evaluation_dataset = get_dataset(name=self.full_name)
            if self.overwrite:
                logger.warning(f"Overwriting dataset {self.full_name}")
                delete_dataset(name=self.full_name)
                needs_creation = True
        except Exception:
            logger.warning(
                f"Dataset {self.full_name} not found, will create new dataset"
            )
            needs_creation = True

        if needs_creation:
            evaluation_dataset = create_dataset(name=self.full_name)
            if self.data:
                logger.debug(
                    f"Merging {len(self.data)} entries into dataset {self.full_name}"
                )
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
    name: str = Field(
        description="Unique name for this optimization run.",
    )
    prompt: Optional[PromptModel] = Field(
        default=None,
        description="Prompt to optimize. If omitted, uses the agent's prompt.",
    )
    agent: AgentModel = Field(
        description="Agent whose prompt is being optimized.",
    )
    dataset: EvaluationDatasetModel = Field(
        description="Training dataset with input/expectation pairs for fitness evaluation.",
    )
    reflection_model: Optional[LLMModel | str] = Field(
        default=None,
        description="LLM used for reflective mutation during GEPA optimization.",
    )
    num_candidates: Optional[int] = Field(
        default=50,
        description="Number of candidate prompts to evaluate per optimization run.",
    )

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
    """Container for prompt and cache threshold optimization configurations."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    training_datasets: dict[str, EvaluationDatasetModel] = Field(
        default_factory=dict,
        description="Named training datasets used by optimization runs.",
    )
    prompt_optimizations: dict[str, PromptOptimizationModel] = Field(
        default_factory=dict,
        description="Named prompt optimization configurations using GEPA.",
    )
    cache_threshold_optimizations: dict[str, "ContextAwareCacheOptimizationModel"] = (
        Field(
            default_factory=dict,
            description="Named cache threshold optimization configurations using Bayesian optimization.",
        )
    )

    def optimize(self, w: WorkspaceClient | None = None) -> dict[str, Any]:
        """
        Optimize all prompts and cache thresholds in this configuration.

        This method:
        1. Ensures all training datasets are created/registered in MLflow
        2. Runs each prompt optimization
        3. Runs each cache threshold optimization

        Args:
            w: Optional WorkspaceClient for Databricks operations

        Returns:
            dict[str, Any]: Dictionary with 'prompts' and 'cache_thresholds' keys
                containing the respective optimization results
        """
        # First, ensure all training datasets are created/registered in MLflow
        logger.info(f"Ensuring {len(self.training_datasets)} training datasets exist")
        for dataset_name, dataset_model in self.training_datasets.items():
            logger.debug(f"Creating/updating dataset: {dataset_name}")
            dataset_model.as_dataset()

        # Run prompt optimizations
        prompt_results: dict[str, PromptModel] = {}
        for name, optimization in self.prompt_optimizations.items():
            prompt_results[name] = optimization.optimize(w)

        # Run cache threshold optimizations
        cache_results: dict[str, Any] = {}
        for name, optimization in self.cache_threshold_optimizations.items():
            cache_results[name] = optimization.optimize(w)

        return {"prompts": prompt_results, "cache_thresholds": cache_results}


class ContextAwareCacheEvalEntryModel(BaseModel):
    """Single evaluation entry for context-aware cache threshold optimization.

    Represents a pair of question/context combinations to evaluate
    whether the cache should return a hit or miss.

    Example:
        entry:
          question: "What are total sales?"
          question_embedding: [0.1, 0.2, ...]  # Pre-computed
          context: "Previous: Show me revenue"
          context_embedding: [0.1, 0.2, ...]
          cached_question: "Show total sales"
          cached_question_embedding: [0.1, 0.2, ...]
          cached_context: "Previous: Show me revenue"
          cached_context_embedding: [0.1, 0.2, ...]
          expected_match: true
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    question: str = Field(
        description="Incoming user question to evaluate against the cache.",
    )
    question_embedding: list[float] = Field(
        description="Pre-computed embedding vector for the question.",
    )
    context: str = Field(
        default="",
        description="Conversation context accompanying the question.",
    )
    context_embedding: list[float] = Field(
        default_factory=list,
        description="Pre-computed embedding vector for the context.",
    )
    cached_question: str = Field(
        description="Previously cached question to compare against.",
    )
    cached_question_embedding: list[float] = Field(
        description="Pre-computed embedding vector for the cached question.",
    )
    cached_context: str = Field(
        default="",
        description="Context that was stored with the cached question.",
    )
    cached_context_embedding: list[float] = Field(
        default_factory=list,
        description="Pre-computed embedding vector for the cached context.",
    )
    expected_match: Optional[bool] = Field(
        default=None,
        description="Whether the pair should be a cache hit (true) or miss (false). None = use LLM judge.",
    )


class ContextAwareCacheEvalDatasetModel(BaseModel):
    """Dataset for context-aware cache threshold optimization.

    Contains pairs of questions/contexts to evaluate whether thresholds
    correctly identify semantic matches.

    Example:
        dataset:
          name: my_cache_eval_dataset
          description: "Evaluation data for cache tuning"
          entries:
            - question: "What are total sales?"
              # ... entry fields
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Unique name for this evaluation dataset.",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the evaluation dataset.",
    )
    entries: list[ContextAwareCacheEvalEntryModel] = Field(
        default_factory=list,
        description="List of question/context pair entries for evaluation.",
    )

    def as_eval_dataset(self) -> "ContextAwareCacheEvalDataset":
        """Convert to internal evaluation dataset format."""
        from dao_ai.genie.cache.context_aware.optimization import (
            ContextAwareCacheEvalDataset,
            ContextAwareCacheEvalEntry,
        )

        entries = [
            ContextAwareCacheEvalEntry(
                question=e.question,
                question_embedding=e.question_embedding,
                context=e.context,
                context_embedding=e.context_embedding,
                cached_question=e.cached_question,
                cached_question_embedding=e.cached_question_embedding,
                cached_context=e.cached_context,
                cached_context_embedding=e.cached_context_embedding,
                expected_match=e.expected_match,
            )
            for e in self.entries
        ]

        return ContextAwareCacheEvalDataset(
            name=self.name,
            entries=entries,
            description=self.description,
        )


class ContextAwareCacheOptimizationModel(BaseModel):
    """Configuration for context-aware cache threshold optimization.

    Uses Optuna Bayesian optimization to find optimal threshold values
    that maximize cache hit accuracy (F1 score by default).

    Example:
        optimizations:
          cache_threshold_optimizations:
            my_optimization:
              name: optimize_cache_thresholds
              cache_parameters: *my_cache_params
              dataset: *my_eval_dataset
              judge_model: databricks-meta-llama-3-3-70b-instruct
              n_trials: 50
              metric: f1
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Unique name for this optimization run (used as the Optuna study name).",
    )
    cache_parameters: Optional[GenieContextAwareCacheParametersModel] = Field(
        default=None,
        description="Cache configuration whose thresholds serve as the starting point for optimization.",
    )
    dataset: ContextAwareCacheEvalDatasetModel = Field(
        description="Evaluation dataset with question/context pairs and expected match labels.",
    )
    judge_model: Optional[LLMModel | str] = Field(
        default="databricks-meta-llama-3-3-70b-instruct",
        description="LLM judge for evaluating match quality when expected_match is None.",
    )
    n_trials: int = Field(
        default=50,
        description="Number of Bayesian optimization trials to run.",
    )
    metric: Literal["f1", "precision", "recall", "fbeta"] = Field(
        default="f1",
        description="Optimization metric to maximize (f1, precision, recall, or fbeta).",
    )
    beta: float = Field(
        default=1.0,
        description="Beta parameter for the fbeta metric (higher = favor recall over precision).",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible optimization results.",
    )

    def optimize(
        self, w: WorkspaceClient | None = None
    ) -> "ThresholdOptimizationResult":
        """
        Optimize context-aware cache thresholds.

        Args:
            w: Optional WorkspaceClient (not used, kept for API compatibility)

        Returns:
            ThresholdOptimizationResult with optimized thresholds
        """
        from dao_ai.genie.cache.context_aware.optimization import (
            ThresholdOptimizationResult,
            optimize_context_aware_cache_thresholds,
        )

        # Convert dataset
        eval_dataset = self.dataset.as_eval_dataset()

        # Get original thresholds from cache_parameters
        original_thresholds: dict[str, float] | None = None
        if self.cache_parameters:
            original_thresholds = {
                "similarity_threshold": self.cache_parameters.similarity_threshold,
                "context_similarity_threshold": self.cache_parameters.context_similarity_threshold,
                "question_weight": self.cache_parameters.question_weight or 0.6,
            }

        # Get judge model
        judge_model_name: str
        if isinstance(self.judge_model, str):
            judge_model_name = self.judge_model
        elif self.judge_model:
            judge_model_name = self.judge_model.uri
        else:
            judge_model_name = "databricks-meta-llama-3-3-70b-instruct"

        result: ThresholdOptimizationResult = optimize_context_aware_cache_thresholds(
            dataset=eval_dataset,
            original_thresholds=original_thresholds,
            judge_model=judge_model_name,
            n_trials=self.n_trials,
            metric=self.metric,
            beta=self.beta,
            register_if_improved=True,
            study_name=self.name,
            seed=self.seed,
        )

        return result


class DatasetFormat(str, Enum):
    """Supported data file formats for dataset loading."""

    CSV = "csv"
    DELTA = "delta"
    JSON = "json"
    PARQUET = "parquet"
    ORC = "orc"
    SQL = "sql"
    EXCEL = "excel"


class DatasetModel(BaseModel):
    """A dataset definition for provisioning a table with DDL and seed data."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    table: Optional[TableModel] = Field(
        default=None,
        description="Target table where the dataset is materialized.",
    )
    ddl: Optional[str | VolumeModel] = Field(
        default=None,
        description="SQL DDL statement or Volume reference containing the CREATE TABLE statement.",
    )
    data: Optional[str | VolumePathModel] = Field(
        default=None,
        description="Seed data as inline SQL INSERT statements or a VolumePath to a data file.",
    )
    format: Optional[DatasetFormat] = Field(
        default=None,
        description="Data file format when loading from a file (csv, json, parquet, delta, etc.).",
    )
    read_options: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Spark read options passed when loading data files (e.g., {'header': 'true'}).",
    )
    table_schema: Optional[str] = Field(
        default=None,
        description="Explicit Spark schema string for the data file (e.g., 'id INT, name STRING').",
    )
    parameters: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Variable substitution parameters for DDL and data templates.",
    )

    def create(self, w: WorkspaceClient | None = None) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w)
        provider.create_dataset(self)


class UnityCatalogFunctionSqlTestModel(BaseModel):
    """Test configuration for validating a Unity Catalog SQL function after creation."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    parameters: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Parameter values to pass when invoking the function for testing.",
    )


class UnityCatalogFunctionSqlModel(BaseModel):
    """A Unity Catalog SQL function definition with DDL, parameters, and optional test."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    function: FunctionModel = Field(
        description="Unity Catalog function reference (target location).",
    )
    ddl: str = Field(
        description="SQL DDL statement defining the function (CREATE OR REPLACE FUNCTION ...).",
    )
    parameters: Optional[dict[str, Any]] = Field(
        default_factory=dict,
        description="Variable substitution parameters for the DDL template.",
    )
    test: Optional[UnityCatalogFunctionSqlTestModel] = Field(
        default=None,
        description="Optional test to run after creating the function to verify it works.",
    )

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
    """Databricks resource declarations used by agents and tools.

    Each resource type is a named dictionary so entries can be referenced
    elsewhere in the config via YAML anchors.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    llms: dict[str, LLMModel] = Field(
        default_factory=dict,
        description="LLM serving endpoint configurations keyed by name.",
    )
    vector_stores: dict[str, VectorStoreModel] = Field(
        default_factory=dict,
        description="Vector search index configurations for semantic retrieval.",
    )
    genie_rooms: dict[str, GenieRoomModel] = Field(
        default_factory=dict,
        description="Databricks Genie space configurations for natural-language SQL.",
    )
    tables: dict[str, TableModel] = Field(
        default_factory=dict,
        description="Unity Catalog table references.",
    )
    volumes: dict[str, VolumeModel] = Field(
        default_factory=dict,
        description="Unity Catalog volume references for file storage.",
    )
    functions: dict[str, FunctionModel] = Field(
        default_factory=dict,
        description="Unity Catalog function references.",
    )
    warehouses: dict[str, WarehouseModel] = Field(
        default_factory=dict,
        description="SQL warehouse configurations for query execution.",
    )
    databases: dict[str, DatabaseModel] = Field(
        default_factory=dict,
        description="Database connection configurations (Lakebase or standard PostgreSQL).",
    )
    connections: dict[str, ConnectionModel] = Field(
        default_factory=dict,
        description="Unity Catalog connection references for MCP and external data sources.",
    )
    apps: dict[str, DatabricksAppModel] = Field(
        default_factory=dict,
        description="Databricks App references used as MCP endpoints or tool backends.",
    )

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
    """Top-level configuration for a DAO AI application.

    Defines all resources, agents, tools, and deployment settings
    needed to build and deploy an AI agent on Databricks.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    version: Optional[str] = Field(
        default=None,
        description="Configuration schema version for forward compatibility.",
    )
    variables: dict[str, AnyVariable] = Field(
        default_factory=dict,
        description="Named variables (env vars, secrets, literals, composites) reusable via YAML anchors.",
    )
    service_principals: dict[str, ServicePrincipalModel] = Field(
        default_factory=dict,
        description="Named service principals for OAuth M2M authentication with Databricks resources.",
    )
    schemas: dict[str, SchemaModel] = Field(
        default_factory=dict,
        description="Unity Catalog schema references (catalog + schema) used by tables, functions, and prompts.",
    )
    resources: Optional[ResourcesModel] = Field(
        default=None,
        description="Databricks resource declarations: LLMs, vector stores, Genie rooms, tables, warehouses, databases, and more.",
    )
    retrievers: dict[str, RetrieverModel] = Field(
        default_factory=dict,
        description="Named retriever configurations combining a vector store with search parameters and optional reranking.",
    )
    tools: dict[str, ToolModel] = Field(
        default_factory=dict,
        description="Named tool definitions (Python, factory, inline, Unity Catalog, or MCP) available to agents.",
    )
    guardrails: dict[str, GuardrailModel] = Field(
        default_factory=dict,
        description="Named guardrail configurations for evaluating agent responses against quality or safety criteria.",
    )
    middleware: dict[str, MiddlewareModel] = Field(
        default_factory=dict,
        description="Named middleware definitions that can be applied to agents for cross-cutting concerns.",
    )
    memory: Optional[MemoryModel] = Field(
        default=None,
        description="Global memory configuration (checkpointer, store, extraction) shared across agents.",
    )
    prompts: dict[str, PromptModel] = Field(
        default_factory=dict,
        description="Named prompt definitions backed by the MLflow Prompt Registry.",
    )
    agents: dict[str, AgentModel] = Field(
        default_factory=dict,
        description="Named agent definitions combining an LLM model with tools, guardrails, and middleware.",
    )
    app: Optional[AppModel] = Field(
        default=None,
        description="Application-level settings: deployment target, model registration, permissions, and orchestration.",
    )
    evaluation: Optional[EvaluationModel] = Field(
        default=None,
        description="Offline evaluation configuration using MLflow GenAI scorers and a judge model.",
    )
    optimizations: Optional[OptimizationsModel] = Field(
        default=None,
        description="Prompt and cache threshold optimization configurations.",
    )
    datasets: Optional[list[DatasetModel]] = Field(
        default_factory=list,
        description="Dataset definitions for provisioning tables with DDL and seed data.",
    )
    unity_catalog_functions: Optional[list[UnityCatalogFunctionSqlModel]] = Field(
        default_factory=list,
        description="Unity Catalog SQL function definitions to create during provisioning.",
    )
    providers: Optional[dict[type | str, Any]] = Field(
        default=None,
        description="Custom provider overrides for dependency injection (advanced usage).",
    )

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
        from dao_ai.guardrails_hub import ensure_guardrails_hub
        from dao_ai.hooks.core import create_hooks
        from dao_ai.logging import configure_logging

        if self.app and self.app.log_level:
            configure_logging(level=self.app.log_level)

        ensure_guardrails_hub(self)

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
        from dao_ai.prompts import get_cached_prompt_versions

        graph: CompiledStateGraph = self.as_graph()
        prompt_versions = get_cached_prompt_versions()
        app: ResponsesAgent = create_responses_agent(
            graph, prompt_versions=prompt_versions
        )
        return app
