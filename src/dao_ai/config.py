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
    Mapping,
    Optional,
    Protocol,
    Self,
    Sequence,
    TypeAlias,
    Union,
)

import yaml

if TYPE_CHECKING:
    from dao_ai.genie.cache.context_aware.optimization import (
        ContextAwareCacheEvalDataset,
        ThresholdOptimizationResult,
    )
    from dao_ai.state import Context

from a2a.types import SecurityScheme
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import (
    CredentialsStrategy,
    ModelServingUserCredentials,
)
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.apps import App
from databricks.sdk.service.catalog import FunctionInfo, TableInfo
from databricks.sdk.service.dashboards import GenieListSpacesResponse, GenieSpace
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
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)

from dao_ai.config_vars import (
    ParameterDeclarationModel,
    substitute_params,
    substitute_workspace_refs,
)
from dao_ai.resource_protocol import (
    ManagedResource,
    Provisionable,
)
from dao_ai.utils import normalize_name


class HasValue(ABC):
    @abstractmethod
    def as_value(self) -> Any: ...


def value_of(value: HasValue | str | int | float | bool) -> Any:
    if isinstance(value, HasValue):
        value = value.as_value()
    return value


_PARAMETER_REF_PATTERN: re.Pattern[str] = re.compile(
    r"^\$\{(?:var|param)\.([A-Za-z_][A-Za-z0-9_]*)\}$"
)


def is_parameter(value: Any) -> bool:
    """Return True if ``value`` is a ``${var.NAME}`` or ``${param.NAME}`` reference.

    Useful for tooling that needs to distinguish operator-supplied parameters
    from literal values in a YAML field — e.g., a provisioning task that
    should only forward task-values for fields backed by a CLI/env parameter.
    """
    return isinstance(value, str) and bool(_PARAMETER_REF_PATTERN.match(value.strip()))


def parameter_name(value: Any) -> str | None:
    """Return the parameter name if ``value`` is a ``${var.NAME}`` / ``${param.NAME}`` reference, else None."""
    if not isinstance(value, str):
        return None
    m: re.Match[str] | None = _PARAMETER_REF_PATTERN.match(value.strip())
    return m.group(1) if m else None


class TaskValuesLike(Protocol):
    """Minimal duck-typed shape of ``dbutils.jobs.taskValues``.

    Lets :meth:`AppConfig.from_file` pull resolved parameter values from
    an upstream Databricks job task without importing ``dbutils`` in tests.
    """

    def get(
        self,
        *,
        taskKey: str,
        key: str,
        default: Any = None,
        debugValue: Any = None,
    ) -> Any: ...


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

    _resolved: bool = PrivateAttr(default=False)

    @abstractmethod
    def as_resources(self) -> Sequence[DatabricksResource]: ...

    @property
    @abstractmethod
    def api_scopes(self) -> Sequence[str]: ...

    def ensure_resolved(self) -> None:
        """Perform deferred API calls (e.g., name resolution, detail fetching).

        Called during AppConfig.initialize(), not during config parsing.
        Subclasses should call super().ensure_resolved() first to set the
        _resolved flag before making API calls that depend on it.
        """
        self._resolved = True

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

    BOTH = "both"
    """Deploy to both Model Serving and Apps."""


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


class SchemaModel(BaseModel, HasFullName, Provisionable):
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
        return ["sql.statement-execution"]

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


class BestOfNConfig(BaseModel):
    """Opt-in best-of-N + LLM-as-judge wrapper around an InferenceEndpointModel.

    When attached to an InferenceEndpointModel, every model invocation fans out N parallel
    candidate generations at elevated temperature, then asks a judge model to
    score the candidates. The wrapper returns the highest-scoring candidate
    verbatim (no synthesis).

    Cost: roughly N+1 LLM calls per protected step. Generator calls run in
    parallel so wall-clock latency is approximately one generator call plus
    one judge call. Token cost is ~N x baseline plus the judge.

    Diversity is non-negotiable: with low generator temperature, the N
    candidates collapse to near-identical outputs and best-of-N degenerates
    into best-of-1 with extra cost. The wrapper enforces an effective
    generator temperature of max(InferenceEndpointModel.temperature, 0.7) unless
    `temperature_override` is set.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    n: int = Field(
        default=8,
        description="Number of parallel candidate generations. Must be in [1, 16].",
    )
    judge: Union[str, "InferenceEndpointModel"] = Field(
        description="Judge model: a serving endpoint name (string) or a full InferenceEndpointModel config.",
    )
    temperature_override: Optional[float] = Field(
        default=None,
        description=(
            "If set, the parallel candidate calls use this temperature regardless "
            "of the generator's configured temperature. If unset, the wrapper uses "
            "max(InferenceEndpointModel.temperature, 0.7) so candidates have meaningful diversity."
        ),
    )

    @field_validator("n")
    @classmethod
    def _validate_n(cls, v: int) -> int:
        if v < 1 or v > 16:
            raise ValueError(f"best_of_n.n must be in [1, 16], got {v}")
        return v


class AIGatewayChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant for the Databricks AI Gateway.

    The Gateway's OpenAI-compatible validator rejects ``name`` on
    ``user`` / ``assistant`` / ``system`` messages with::

        400 BAD_REQUEST: messages.N.name: Extra inputs are not permitted

    LangGraph's supervisor pattern attaches a ``name`` field to agent
    AIMessages for routing (see
    :func:`dao_ai.orchestration.core.filter_messages_for_agent`), so we
    strip it at the request-payload boundary instead of in orchestration
    where it carries real semantics for ChatDatabricks and other backends.
    ``role: "tool"`` / ``role: "function"`` messages are left untouched.
    """

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        payload: dict = super()._get_request_payload(input_, stop=stop, **kwargs)
        for msg in payload.get("messages", []) or []:
            if isinstance(msg, dict) and msg.get("role") in (
                "user",
                "assistant",
                "system",
            ):
                msg.pop("name", None)
        return payload


class InferenceEndpointModel(IsDatabricksResource):
    """Configuration for a Databricks Model Serving endpoint used for inference.

    This is the single config type for *any* serving endpoint dao-ai calls at
    runtime — not just chat LLMs. The same class backs:

    - Chat LLMs declared under ``resources.models`` (e.g. claude-sonnet,
      meta-llama-3-3-70b-instruct).
    - Embedding endpoints referenced by ``VectorStoreModel.embedding_model``
      (e.g. databricks-gte-large-en).
    - Judge / extraction / reflection / query models inside
      ``BestOfNConfig.judge``, ``MemoryConfig.extraction_model``,
      ``MemoryConfig.query_model``, ``DeepAgentOrchestrationConfig.judge_model``,
      ``DeepAgentOrchestrationConfig.reflection_model``.
    - Custom agent endpoints (any model packaged behind
      ``/serving-endpoints/<name>/invocations``).

    The previous class name ``LLMModel`` is still importable as a module-level
    alias (``LLMModel = InferenceEndpointModel``) for backward compatibility;
    the legacy name will be removed in a future major release.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Serving endpoint name (e.g., 'databricks-gpt-5-4-mini').",
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
    fallbacks: Optional[list[Union[str, "InferenceEndpointModel"]]] = Field(
        default_factory=list,
        description="Ordered list of fallback endpoint names or InferenceEndpointModel configs tried on primary failure.",
    )
    use_responses_api: Optional[bool] = Field(
        default=False,
        description="Use Responses API for ResponsesAgent endpoints",
    )
    disable_streaming: bool = Field(
        default=False,
        description="Disable streaming for this model. Required when the Foundation Model endpoint has output guardrails enabled.",
    )
    ai_gateway: bool = Field(
        default=False,
        description=(
            "Route through the Databricks AI Gateway "
            "(/ai-gateway/mlflow/v1/chat/completions) instead of "
            "/serving-endpoints/<name>/invocations. When True, `name` is "
            "sent as the OpenAI-style model id in the request body. "
            "AI Gateway is OpenAI-compatible chat completions only — not "
            "for embeddings, Responses API, or non-chat endpoints."
        ),
    )
    best_of_n: Optional[BestOfNConfig] = Field(
        default=None,
        description=(
            "Opt-in best-of-N + LLM-as-judge wrapper. When set, every invocation of "
            "this model fans out N parallel candidates and a judge picks the winner. "
            "Forces disable_streaming=True. See BestOfNConfig for cost implications."
        ),
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

    @model_validator(mode="after")
    def _validate_ai_gateway_compatibility(self) -> Self:
        if not self.ai_gateway:
            return self
        if self.use_responses_api:
            raise ValueError(
                "ai_gateway=True is incompatible with use_responses_api=True. "
                "AI Gateway exposes only the OpenAI-compatible "
                "/chat/completions path; Responses API endpoints must stay on "
                "/serving-endpoints/<name>/invocations."
            )
        # NOTE: on_behalf_of_user + ai_gateway is permitted pending live
        # verification. If a workspace returns 401/403 on AI Gateway with an
        # OBO token, gate it here with a ValueError.
        return self

    def _ai_gateway_host(self) -> str:
        """Return the workspace host without trailing slash."""
        from dao_ai.utils import normalize_host

        return normalize_host(self.workspace_client.config.host).rstrip("/")

    def _ai_gateway_token_provider(self) -> Callable[[], str]:
        """Return a callable that resolves a fresh bearer token per call.

        ``ChatOpenAI`` / the ``openai`` SDK invoke this provider on every
        request, so PAT, service-principal (OAuth-M2M), and OBO tokens are
        always refreshed through the SDK's auth ladder instead of being
        captured once at construction time.
        """
        wc: WorkspaceClient = self.workspace_client

        def _provider() -> str:
            headers: Mapping[str, str] = wc.config.authenticate()
            auth: str = headers.get("Authorization", "")
            if not auth.lower().startswith("bearer "):
                raise RuntimeError(
                    f"Could not extract bearer token for {self.name!r}; "
                    f"authenticate() returned keys: {list(headers)}"
                )
            return auth.split(" ", 1)[1]

        return _provider

    def _resolve_ai_gateway_credentials(self) -> tuple[str, Callable[[], str]]:
        """Return ``(host, token_provider)`` for AI Gateway HTTP calls."""
        return self._ai_gateway_host(), self._ai_gateway_token_provider()

    def chat_model_for_workspace_client(
        self,
        workspace_client: WorkspaceClient,
        *,
        disable_streaming: bool | None = None,
    ) -> LanguageModelLike:
        """Build a chat client bound to a specific ``WorkspaceClient``.

        Used by OBO call sites that need to swap in a user-scoped
        ``WorkspaceClient`` per request. Respects ``self.ai_gateway`` so OBO
        traffic still routes through the AI Gateway path when enabled.
        """
        from dao_ai.utils import normalize_host

        effective_disable_streaming: bool = (
            self.disable_streaming if disable_streaming is None else disable_streaming
        )

        if self.ai_gateway:
            host: str = normalize_host(workspace_client.config.host).rstrip("/")

            def token_provider() -> str:
                headers: Mapping[str, str] = workspace_client.config.authenticate()
                auth: str = headers.get("Authorization", "")
                if not auth.lower().startswith("bearer "):
                    raise RuntimeError(
                        f"Could not extract bearer token for {self.name!r}; "
                        f"authenticate() returned keys: {list(headers)}"
                    )
                return auth.split(" ", 1)[1]

            return AIGatewayChatOpenAI(
                model=self.name,
                base_url=f"{host}/ai-gateway/mlflow/v1",
                api_key=token_provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                streaming=not effective_disable_streaming,
            )

        return ChatDatabricks(
            model=self.name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            use_responses_api=self.use_responses_api,
            disable_streaming=effective_disable_streaming,
            workspace_client=workspace_client,
        )

    def as_chat_model(self) -> LanguageModelLike:
        # When best_of_n is enabled, force streaming off — the wrapper has to
        # buffer the full candidate response before it can hand it to the judge.
        effective_disable_streaming = self.disable_streaming
        if self.best_of_n is not None and not self.disable_streaming:
            logger.debug(
                "best_of_n is enabled; forcing disable_streaming=True for {name}",
                name=self.name,
            )
            effective_disable_streaming = True

        chat_client: LanguageModelLike
        if self.ai_gateway:
            host, token_provider = self._resolve_ai_gateway_credentials()
            chat_client = AIGatewayChatOpenAI(
                model=self.name,
                base_url=f"{host}/ai-gateway/mlflow/v1",
                api_key=token_provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                streaming=not effective_disable_streaming,
            )
        else:
            chat_client = ChatDatabricks(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                use_responses_api=self.use_responses_api,
                disable_streaming=effective_disable_streaming,
            )

        fallbacks: Sequence[LanguageModelLike] = []
        for fallback in self.fallbacks:
            fallback: str | InferenceEndpointModel
            if isinstance(fallback, str):
                fallback = InferenceEndpointModel(
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

        if self.best_of_n is not None:
            # Lazy import to keep config.py free of langchain_core chat-model
            # imports beyond what already lives at module top.
            from dao_ai.best_of_n import BestOfNChatModel

            judge_cfg = self.best_of_n.judge
            if isinstance(judge_cfg, str):
                judge_cfg = InferenceEndpointModel(name=judge_cfg)
            judge_chat_model = judge_cfg.as_chat_model()

            chat_client = BestOfNChatModel.from_components(
                generator=chat_client,
                judge=judge_chat_model,
                n=self.best_of_n.n,
                generator_temperature=self.temperature,
                temperature_override=self.best_of_n.temperature_override,
            )

        return chat_client

    def as_open_ai_client(self) -> LanguageModelLike:
        chat_client: ChatOpenAI
        if self.ai_gateway:
            host, token_provider = self._resolve_ai_gateway_credentials()
            chat_client = AIGatewayChatOpenAI(
                model=self.name,
                base_url=f"{host}/ai-gateway/mlflow/v1",
                api_key=token_provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            chat_client = self.workspace_client.serving_endpoints.get_langchain_chat_open_ai_client(
                model=self.name
            )
            chat_client.temperature = self.temperature
            chat_client.max_tokens = self.max_tokens

        return chat_client

    def as_embeddings_model(self) -> Embeddings:
        return DatabricksEmbeddings(endpoint=self.name)


# Backward-compatible alias. The class was renamed from LLMModel to
# InferenceEndpointModel to reflect its real scope (every Databricks Model
# Serving endpoint dao-ai calls — chat LLMs, embeddings, judges, extraction /
# reflection / query models, custom agent endpoints). Customer code importing
# the old name (`from dao_ai.config import LLMModel`) keeps working unchanged;
# `isinstance(x, LLMModel)` continues to return True because both names point
# at the same class. The legacy name will be removed in a future major release.
LLMModel = InferenceEndpointModel

# Resolve the forward reference BestOfNConfig.judge -> InferenceEndpointModel
# now that the class is in scope. Without this, instantiating BestOfNConfig
# with a dict judge config would fail with a forward-reference error.
BestOfNConfig.model_rebuild()


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
    target_qps: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Target queries-per-second for the endpoint. STANDARD only. "
            "Scales endpoint compute linearly; capacity changes take effect "
            "the next time an index on this endpoint is created or synced. "
            "Public Preview. Honored at endpoint-creation time only — if "
            "the endpoint already exists, this value is ignored."
        ),
    )

    @field_serializer("type")
    def serialize_type(self, value: VectorSearchEndpointType) -> str:
        """Ensure enum is serialized to string value."""
        if isinstance(value, VectorSearchEndpointType):
            return value.value
        return str(value)

    @model_validator(mode="after")
    def validate_target_qps_only_on_standard(self) -> Self:
        """Reject target_qps on non-STANDARD endpoints (SDK constraint)."""
        if (
            self.target_qps is not None
            and self.type != VectorSearchEndpointType.STANDARD
        ):
            raise ValueError(
                f"target_qps is only supported on STANDARD endpoints, not {self.type!r}"
            )
        return self


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
        description="SQL warehouse ID. Required when on_behalf_of_user is true. If omitted, looked up by name.",
    )

    _warehouse_details: Optional[GetWarehouseResponse] = PrivateAttr(default=None)

    def _get_warehouse_details(self) -> GetWarehouseResponse | None:
        if self._warehouse_details is None:
            if not self._resolved:
                return None
            self._warehouse_details = self.workspace_client.warehouses.get(
                id=value_of(self.warehouse_id)
            )
        return self._warehouse_details

    def _resolve_warehouse_id_by_name(self, name: str) -> str:
        """Look up a warehouse ID by iterating all warehouses and matching by name.

        Raises:
            ValueError: if zero or more than one warehouse match ``name``.
                The Databricks docs claim warehouse names "must be unique
                within an org," but the platform does **not** enforce this
                — duplicates are routinely created. Pass ``warehouse_id``
                directly to disambiguate.
        """
        logger.info(f"Resolving warehouse by name: '{name}'")
        matches: list[str] = [
            warehouse.id
            for warehouse in self.workspace_client.warehouses.list()
            if warehouse.name == name
        ]
        if not matches:
            raise ValueError(
                f"No warehouse found with name '{name}'. "
                "Verify the name matches an existing SQL warehouse in your workspace."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple warehouses ({len(matches)}) found with name '{name}': "
                f"{matches}. Pass warehouse_id directly to disambiguate."
            )
        logger.info(f"Resolved warehouse '{name}' to id '{matches[0]}'")
        return matches[0]

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
        """Validate warehouse config. Actual name→ID resolution deferred to ensure_resolved()."""
        if self.warehouse_id:
            self.warehouse_id = value_of(self.warehouse_id)
            return self
        if self.on_behalf_of_user:
            raise ValueError(
                "warehouse_id is required when on_behalf_of_user is True. "
                "Name-based lookup cannot authenticate in Model Serving at startup."
            )
        if not self.name:
            raise ValueError(
                "Either 'warehouse_id' or 'name' must be provided for WarehouseModel."
            )
        return self

    def ensure_resolved(self) -> None:
        """Resolve warehouse name→ID and populate details via API."""
        if self._resolved:
            return
        super().ensure_resolved()
        # Resolve name → warehouse_id if needed
        if not self.warehouse_id and self.name:
            self.warehouse_id = self._resolve_warehouse_id_by_name(self.name)
        # Populate name from warehouse details if missing
        if self.warehouse_id and not self.name:
            try:
                warehouse_details = self._get_warehouse_details()
                if warehouse_details and warehouse_details.name:
                    self.name = warehouse_details.name
            except Exception as e:
                logger.debug(f"Could not fetch details from warehouse: {e}")


class GenieColumnConfig(BaseModel):
    """Per-column metadata registered with a Genie data source.

    Mirrors the ``data_sources.tables[].column_configs[]`` entries in
    a Genie space's ``serialized_space`` payload. ``synonyms`` lets the
    LLM map natural-language terms to the underlying column.

    Uses ``extra="allow"`` so unmodeled server metadata round-trips
    cleanly through ``GenieRoomModel.refresh() → create()``.
    """

    model_config = ConfigDict(use_enum_values=True, extra="allow")
    name: str = Field(description="Column name as defined in Unity Catalog.")
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description shown to the model when reasoning about this column.",
    )
    synonyms: Optional[list[str]] = Field(
        default=None,
        description="Alternate names users may use for this column.",
    )
    excluded: bool = Field(
        default=False,
        description="If true, hide this column from Genie when answering questions.",
    )
    sample_values: bool = Field(
        default=True,
        description="If true, surface example values for this column to the model.",
    )
    build_value_dictionary: bool = Field(
        default=False,
        description="If true, build a value dictionary so Genie can match user terms against actual values.",
    )


class GenieTableSource(BaseModel):
    """A Unity Catalog table or view registered as a Genie data source."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")
    table: TableModel = Field(
        description="Reference to the UC table, view, or materialized view (reuses TableModel).",
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the table presented to the Genie LLM.",
    )
    column_configs: Optional[list[GenieColumnConfig]] = Field(
        default=None,
        description="Per-column metadata (synonyms, descriptions, exclusions).",
    )


class GenieMetricViewSource(BaseModel):
    """A Unity Catalog metric view registered as a Genie data source."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")
    table: TableModel = Field(
        description="Reference to the metric view (UC three-part name via TableModel).",
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the metric view presented to the Genie LLM.",
    )


class GenieSqlFunctionSource(BaseModel):
    """A Unity Catalog SQL function registered as a Genie trusted asset."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")
    function: FunctionModel = Field(
        description="Reference to the UC function (reuses FunctionModel).",
    )


class GenieSqlParameter(BaseModel):
    """A named parameter on a trusted example SQL query."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")
    name: str = Field(description="Parameter name as referenced in the SQL.")
    type_hint: Literal["STRING", "INTEGER", "DATE"] = Field(
        default="STRING",
        description="Declared parameter type so Genie can route values correctly.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the parameter's meaning.",
    )


class GenieExampleSql(BaseModel):
    """A trusted example question + SQL pair Genie uses for few-shot guidance."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")
    question: str = Field(description="Natural-language question.")
    sql: str = Field(description="Authoritative SQL answering the question.")
    parameters: Optional[list[GenieSqlParameter]] = Field(
        default=None,
        description="Optional parameters bound to the SQL.",
    )
    usage_guidance: Optional[str] = Field(
        default=None,
        description="When the model should reuse this example.",
    )


class GenieRelationshipType(str, Enum):
    """Relationship cardinality used to annotate Genie join specs."""

    ONE_TO_ONE = "ONE_TO_ONE"
    ONE_TO_MANY = "ONE_TO_MANY"
    MANY_TO_ONE = "MANY_TO_ONE"
    MANY_TO_MANY = "MANY_TO_MANY"


class GenieJoinSpec(BaseModel):
    """A trusted join relationship between two Genie data sources."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")
    left: TableModel = Field(description="Left table in the join.")
    left_alias: Optional[str] = Field(
        default=None, description="Optional alias for the left table."
    )
    right: TableModel = Field(description="Right table in the join.")
    right_alias: Optional[str] = Field(
        default=None, description="Optional alias for the right table."
    )
    sql: str = Field(
        description="Join condition expression (e.g., 'orders.customer_id = customers.id').",
    )
    relationship_type: Optional[GenieRelationshipType] = Field(
        default=None,
        description="Cardinality annotation Genie can reason about.",
    )
    comment: Optional[str] = Field(
        default=None,
        description="Free-form description of when the join applies.",
    )


class GenieSqlSnippet(BaseModel):
    """A reusable SQL snippet (filter / expression / measure) registered with a Genie space."""

    model_config = ConfigDict(use_enum_values=True, extra="allow")
    display_name: str = Field(description="Human-readable name shown to authors.")
    sql: str = Field(description="The SQL fragment.")
    instruction: Optional[str] = Field(
        default=None,
        description="When the model should use this snippet.",
    )
    synonyms: Optional[list[str]] = Field(
        default=None,
        description="Alternate phrasings users may employ to reference this snippet.",
    )


class GenieBenchmarkQuestion(BaseModel):
    """A benchmark question + expected SQL pair stored on the Genie space.

    Mirrors ``benchmarks.questions[]`` in the serialized space and lets
    teams ship offline evaluation data alongside the space configuration.
    """

    model_config = ConfigDict(use_enum_values=True, extra="allow")
    question: str = Field(description="Benchmark natural-language question.")
    expected_sql: str = Field(description="Expected SQL answer for evaluation.")


class GenieEntitlementLevel(str, Enum):
    """Workspace permission levels supported on Genie spaces."""

    CAN_VIEW = "CAN_VIEW"
    CAN_RUN = "CAN_RUN"
    CAN_EDIT = "CAN_EDIT"
    CAN_MANAGE = "CAN_MANAGE"


class GenieEntitlement(BaseModel):
    """A grant of workspace-level permissions on the Genie space.

    Principals may be users (email), groups, or service principals
    (application ID, or a ``ServicePrincipalModel``).
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    principals: list[ServicePrincipalModel | str] = Field(
        default_factory=list,
        description="Users (email), groups, or service principals to grant the permission to.",
    )
    permission_level: GenieEntitlementLevel = Field(
        description="Genie permission level to grant (CAN_VIEW, CAN_RUN, CAN_EDIT, CAN_MANAGE).",
    )

    @model_validator(mode="after")
    def resolve_principals(self) -> Self:
        """Resolve ``ServicePrincipalModel`` entries to their client_id."""
        resolved: list[str] = []
        for principal in self.principals:
            if isinstance(principal, ServicePrincipalModel):
                resolved.append(value_of(principal.client_id))
            else:
                resolved.append(principal)
        self.principals = resolved
        return self


class GenieRoomModel(IsDatabricksResource, ManagedResource):
    """Databricks Genie space configuration for natural-language SQL exploration.

    Supports two modes:

    1. **Use Existing Space**: Provide ``space_id`` (or ``name`` for lookup) to
       reference an existing Genie space at runtime. Tables, functions, and
       warehouses are auto-discovered from the live ``serialized_space``.

    2. **Provisioning Mode**: Provide ``warehouse`` plus any of
       ``table_sources``, ``metric_view_sources``, ``function_sources``,
       ``text_instructions``, ``example_sqls``, ``join_specs``,
       ``sql_filters``/``sql_expressions``/``sql_measures``,
       ``sample_questions``, ``benchmarks``, and ``entitlements`` to declare
       the full space configuration. Calling :meth:`create` provisions a new
       space (or updates an existing one when ``space_id`` is set).
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: Optional[str] = Field(
        default=None,
        description="Display name (title) for the Genie space. Auto-populated from the space if omitted.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the Genie room. Auto-populated from the space if omitted.",
    )
    space_id: Optional[AnyVariable] = Field(
        default=None,
        description=(
            "Databricks-assigned Genie space identifier. The only field "
            "guaranteed unique by the platform; titles are not enforced unique. "
            "Lifecycle: "
            "(a) Reference mode — set by the user in YAML; readable immediately. "
            "(b) Spec mode — left None, populated by .create() once the space "
            "is provisioned. Downstream consumers reading *room_anchor.space_id "
            "must wait until .create() has run. "
            "Required when on_behalf_of_user is true (name-based lookup cannot "
            "authenticate at Model Serving startup)."
        ),
    )
    parent_path: Optional[AnyVariable] = Field(
        default=None,
        description="Workspace folder path used when provisioning a new space (e.g., '/Users/me@example.com/genie').",
    )
    warehouse: Optional[WarehouseModel] = Field(
        default=None,
        description="SQL warehouse the Genie space queries against. Required for provisioning. For existing-space references, call :meth:`discover_warehouse` to fetch the warehouse attached to the live space.",
    )
    table_sources: Optional[list[GenieTableSource]] = Field(
        default=None,
        description="UC tables/views registered as Genie data sources (with optional column metadata).",
    )
    metric_view_sources: Optional[list[GenieMetricViewSource]] = Field(
        default=None,
        description="UC metric views registered as Genie data sources.",
    )
    function_sources: Optional[list[GenieSqlFunctionSource]] = Field(
        default=None,
        description="UC SQL functions registered as Genie trusted assets.",
    )
    text_instructions: Optional[list[str]] = Field(
        default=None,
        description="Free-form instructions Genie always considers when reasoning.",
    )
    example_sqls: Optional[list[GenieExampleSql]] = Field(
        default=None,
        description="Trusted example question→SQL pairs.",
    )
    join_specs: Optional[list[GenieJoinSpec]] = Field(
        default=None,
        description="Trusted join relationships between data sources.",
    )
    sql_filters: Optional[list[GenieSqlSnippet]] = Field(
        default=None,
        description="Reusable SQL filter snippets.",
    )
    sql_expressions: Optional[list[GenieSqlSnippet]] = Field(
        default=None,
        description="Reusable SQL expression snippets.",
    )
    sql_measures: Optional[list[GenieSqlSnippet]] = Field(
        default=None,
        description="Reusable SQL measure snippets.",
    )
    sample_questions: Optional[list[str]] = Field(
        default=None,
        description="Suggested sample questions surfaced in the Genie UI.",
    )
    benchmarks: Optional[list[GenieBenchmarkQuestion]] = Field(
        default=None,
        description="Benchmark questions + expected SQL stored with the space for offline evaluation.",
    )
    entitlements: Optional[list[GenieEntitlement]] = Field(
        default=None,
        description="Workspace permissions to grant on the Genie space (applied during .create()).",
    )

    _space_details: Optional[GenieSpace] = PrivateAttr(default=None)
    _raw_space_id: Optional[str] = PrivateAttr(default=None)
    """The original ``space_id`` YAML value before parameter substitution.

    Populated by :meth:`AppConfig.from_file` from the pre-substitution
    raw YAML. Useful for tooling that needs to know whether the field
    was backed by a ``${var.X}`` reference — e.g., a provisioning task
    that should forward a resolved value back to that same parameter.
    Use :func:`is_parameter` and :func:`parameter_name` to inspect.
    """

    @property
    def raw_space_id(self) -> Optional[str]:
        """The original (pre-substitution) ``space_id`` YAML value, or ``None``.

        Use with :func:`is_parameter` / :func:`parameter_name` to detect
        whether the field was bound to a ``${var.X}`` parameter.
        """
        return self._raw_space_id

    def _get_space_details(self) -> GenieSpace | None:
        """Fetch Genie space details from the API.

        Returns:
            GenieSpace if successful, None if not yet resolved or if the API
            call fails (e.g., due to permission issues in model serving).
        """
        if self._space_details is None:
            if not self._resolved:
                return None
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
        """Look up a Genie space ID by iterating all spaces and matching by title.

        Raises:
            ValueError: if zero spaces or more than one space match ``name``.
                Genie titles are not enforced unique by Databricks; pass
                ``space_id`` directly to disambiguate.
        """
        logger.info(f"Resolving Genie space by name: '{name}'")
        matches: list[str] = []
        page_token: Optional[str] = None
        while True:
            response: GenieListSpacesResponse = self.workspace_client.genie.list_spaces(
                page_token=page_token
            )
            if response.spaces:
                for space in response.spaces:
                    if space.title == name:
                        matches.append(space.space_id)
            if not response.next_page_token:
                break
            page_token = response.next_page_token

        if not matches:
            raise ValueError(
                f"No Genie space found with title '{name}'. "
                "Verify the name matches an existing Genie space in your workspace."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple Genie spaces ({len(matches)}) found with title '{name}': "
                f"{matches}. Genie titles are not enforced unique by Databricks; "
                "pass space_id directly to disambiguate."
            )
        logger.info(f"Resolved Genie space '{name}' to space_id '{matches[0]}'")
        return matches[0]

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

    def discover_warehouse(self) -> Optional[WarehouseModel]:
        """Fetch the SQL warehouse attached to the live Genie space.

        Used for existing-space references: looks up the space via the
        Databricks SDK and returns a :class:`WarehouseModel` for whatever
        warehouse is currently bound to it. Returns ``None`` if the space
        cannot be inspected, has no warehouse, or the warehouse details
        call fails. This does not mutate :attr:`warehouse` — assign the
        result explicitly if you want to cache it on the model.
        """
        space_details = self._get_space_details()

        if space_details is None or not space_details.warehouse_id:
            return None

        try:
            response: GetWarehouseResponse = self.workspace_client.warehouses.get(
                space_details.warehouse_id
            )
            warehouse_name: str = response.name or space_details.warehouse_id

            return WarehouseModel(
                name=warehouse_name,
                warehouse_id=space_details.warehouse_id,
                on_behalf_of_user=self.on_behalf_of_user,
                service_principal=self.service_principal,
                client_id=self.client_id,
                client_secret=self.client_secret,
                workspace_host=self.workspace_host,
                pat=self.pat,
            )
        except Exception as e:
            logger.warning(
                f"Failed to fetch warehouse details for {space_details.warehouse_id}: {e}"
            )
            return None

    @property
    def tables(self) -> list[TableModel]:
        """Extract UC table-like resources from the serialized Genie space.

        Genie stores table-like dependencies under ``data_sources``:

        - ``data_sources.tables[].identifier`` — regular UC tables and views.
        - ``data_sources.metric_views[].identifier`` — AI/BI Genie metric
          views (semantic layer). These are UC-first-class objects and
          accept the same ``SELECT`` grant as tables, so the bundle treats
          them identically for permission purposes.

        Only includes entries that actually exist in Unity Catalog so a
        stale reference in ``serialized_space`` doesn't break bundle
        generation. If the live space cannot be inspected (e.g., before
        provisioning), falls back to ``table_sources`` /
        ``metric_view_sources`` declared on this model so resource
        discovery still works during the same provisioning run.
        """
        parsed_space = self._parse_serialized_space()
        tables_list: list[TableModel] = []

        if not parsed_space and (self.table_sources or self.metric_view_sources):
            for source in self.table_sources or []:
                tables_list.append(source.table)
            for source in self.metric_view_sources or []:
                tables_list.append(source.table)
            return tables_list

        data_sources = parsed_space.get("data_sources")
        if not isinstance(data_sources, dict):
            return tables_list

        # Both tables and metric_views follow the same [{identifier: ...}]
        # shape and both are grantable as TABLE securables.
        for ds_key in ("tables", "metric_views"):
            items = data_sources.get(ds_key)
            if not isinstance(items, list):
                continue
            for item in items:
                table_name: str | None = None
                if isinstance(item, dict):
                    table_name = item.get("identifier") or item.get("name")
                elif isinstance(item, str):
                    table_name = item

                if not table_name:
                    continue

                table_model = TableModel(
                    name=table_name,
                    on_behalf_of_user=self.on_behalf_of_user,
                    service_principal=self.service_principal,
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    workspace_host=self.workspace_host,
                    pat=self.pat,
                )
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
        Only includes functions that actually exist in Unity Catalog. Falls
        back to ``function_sources`` declared on this model when the live
        space cannot be inspected (e.g., before provisioning).
        """
        parsed_space = self._parse_serialized_space()
        functions_list: list[FunctionModel] = []
        seen_functions: set[str] = set()

        if not parsed_space and self.function_sources:
            for source in self.function_sources:
                if source.function.full_name in seen_functions:
                    continue
                seen_functions.add(source.function.full_name)
                functions_list.append(source.function)
            return functions_list

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
        """Validate genie config. Actual name→ID resolution deferred to ensure_resolved()."""
        if self.space_id:
            self.space_id = value_of(self.space_id)
            return self
        if self.on_behalf_of_user:
            raise ValueError(
                "space_id is required when on_behalf_of_user is True. "
                "Name-based lookup cannot authenticate in Model Serving at startup."
            )
        if not self.name:
            raise ValueError(
                "Either 'space_id' or 'name' must be provided for GenieRoomModel."
            )
        return self

    @property
    def has_provisioning_config(self) -> bool:
        """True when any provisioning field is set on this model.

        Used to distinguish discovery-mode usage (just reference an existing
        space) from provisioning-mode usage (declare the space contents and
        call :meth:`create`). In provisioning mode we tolerate a missing
        space at resolve time because :meth:`create` will materialize it.
        """
        return any(
            field is not None and field != []
            for field in (
                self.warehouse,
                self.table_sources,
                self.metric_view_sources,
                self.function_sources,
                self.text_instructions,
                self.example_sqls,
                self.join_specs,
                self.sql_filters,
                self.sql_expressions,
                self.sql_measures,
                self.sample_questions,
                self.benchmarks,
                self.entitlements,
                self.parent_path,
            )
        )

    def ensure_resolved(self) -> None:
        """Resolve space name→ID and populate details via API."""
        if self._resolved:
            return
        super().ensure_resolved()
        # Resolve name → space_id if needed. When provisioning fields are
        # declared, a missing space is expected (it will be created on demand
        # by :meth:`create`), so we skip rather than raise.
        if not self.space_id and self.name:
            try:
                self.space_id = self._resolve_space_id_by_name(self.name)
            except ValueError:
                if self.has_provisioning_config:
                    logger.debug(
                        "Genie space not found by name; will be created when create() is invoked",
                        name=self.name,
                    )
                else:
                    raise
        # Populate name and description from space details if missing
        if self.space_id and (not self.name or not self.description):
            try:
                space_details = self._get_space_details()
                if space_details:
                    if not self.name and space_details.title:
                        self.name = space_details.title
                    if not self.description and space_details.description:
                        self.description = space_details.description
            except Exception as e:
                logger.debug(f"Could not fetch details from Genie space: {e}")

    def _build_serialized_space(self) -> dict[str, Any]:
        """Build the ``serialized_space`` JSON payload from this model's fields.

        The shape mirrors the JSON Genie persists internally (data_sources,
        instructions, benchmarks, etc.). Stable hex IDs are derived from the
        natural keys of each entry so re-running provisioning produces the
        same payload (important for the diff check in :meth:`create`).
        """
        import hashlib

        def _stable_id(*parts: str) -> str:
            digest = hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()
            return digest[:32]

        payload: dict[str, Any] = {"version": 2}

        # config.sample_questions
        if self.sample_questions:
            payload["config"] = {
                "sample_questions": [
                    {
                        "id": _stable_id("sample_question", question),
                        "question": [question],
                    }
                    for question in self.sample_questions
                ]
            }

        # data_sources.tables and data_sources.metric_views
        data_sources: dict[str, Any] = {}
        if self.table_sources:
            tables_payload: list[dict[str, Any]] = []
            for source in self.table_sources:
                entry: dict[str, Any] = {"identifier": source.table.full_name}
                if source.description:
                    entry["description"] = [source.description]
                if source.column_configs:
                    cc_payload = [
                        {
                            "column_name": cc.name,
                            **(
                                {"description": [cc.description]}
                                if cc.description
                                else {}
                            ),
                            "enable_format_assistance": cc.sample_values,
                            "exclude": cc.excluded,
                            **({"synonyms": cc.synonyms} if cc.synonyms else {}),
                            "enable_entity_matching": cc.build_value_dictionary,
                        }
                        for cc in source.column_configs
                    ]
                    entry["column_configs"] = cc_payload
                tables_payload.append(entry)
            data_sources["tables"] = tables_payload

        if self.metric_view_sources:
            mv_payload = [
                {
                    "identifier": source.table.full_name,
                    **(
                        {"description": [source.description]}
                        if source.description
                        else {}
                    ),
                }
                for source in self.metric_view_sources
            ]
            data_sources["metric_views"] = mv_payload

        if data_sources:
            payload["data_sources"] = data_sources

        # instructions.{text_instructions, example_question_sqls, sql_functions, join_specs, sql_snippets}
        instructions: dict[str, Any] = {}
        if self.text_instructions:
            combined_id = _stable_id("text_instruction", *self.text_instructions)
            instructions["text_instructions"] = [
                {
                    "id": combined_id,
                    "content": self.text_instructions,
                }
            ]
        if self.example_sqls:
            example_payload: list[dict[str, Any]] = []
            for example in self.example_sqls:
                entry = {
                    "id": _stable_id("example_sql", example.question, example.sql),
                    "question": [example.question],
                    "sql": [example.sql],
                }
                if example.parameters:
                    entry["parameters"] = [
                        {
                            "name": p.name,
                            "type_hint": p.type_hint,
                            **(
                                {"description": [p.description]}
                                if p.description
                                else {}
                            ),
                        }
                        for p in example.parameters
                    ]
                if example.usage_guidance:
                    entry["usage_guidance"] = [example.usage_guidance]
                example_payload.append(entry)
            instructions["example_question_sqls"] = example_payload
        if self.function_sources:
            instructions["sql_functions"] = [
                {
                    "id": _stable_id("sql_function", source.function.full_name),
                    "identifier": source.function.full_name,
                }
                for source in self.function_sources
            ]
        if self.join_specs:
            join_payload: list[dict[str, Any]] = []
            for spec in self.join_specs:
                left: dict[str, str] = {"identifier": spec.left.full_name}
                right: dict[str, str] = {"identifier": spec.right.full_name}
                entry: dict[str, Any] = {
                    "id": _stable_id(
                        "join_spec", spec.left.full_name, spec.right.full_name, spec.sql
                    ),
                    "left": left,
                    "right": right,
                    "sql": [spec.sql],
                }
                if spec.comment:
                    entry["comment"] = [spec.comment]
                join_payload.append(entry)
            instructions["join_specs"] = join_payload

        snippets: dict[str, Any] = {}
        for snippet_key, snippet_list in (
            ("filters", self.sql_filters),
            ("expressions", self.sql_expressions),
            ("measures", self.sql_measures),
        ):
            if not snippet_list:
                continue
            label_key = "display_name" if snippet_key == "filters" else "alias"
            snippets[snippet_key] = [
                {
                    "id": _stable_id(snippet_key, snippet.display_name, snippet.sql),
                    "sql": [snippet.sql],
                    label_key: snippet.display_name,
                    **(
                        {"instruction": [snippet.instruction]}
                        if snippet.instruction
                        else {}
                    ),
                    **({"synonyms": snippet.synonyms} if snippet.synonyms else {}),
                }
                for snippet in snippet_list
            ]
        if snippets:
            instructions["sql_snippets"] = snippets

        if instructions:
            payload["instructions"] = instructions

        # benchmarks.questions
        if self.benchmarks:
            payload["benchmarks"] = {
                "questions": [
                    {
                        "id": _stable_id("benchmark", b.question, b.expected_sql),
                        "question": [b.question],
                        "answer": [{"format": "SQL", "content": [b.expected_sql]}],
                    }
                    for b in self.benchmarks
                ]
            }

        self._sort_payload_lists(payload)
        return payload

    @staticmethod
    def _sort_payload_lists(obj: Any) -> None:
        """Recursively sort list-of-dicts in the serialized payload.

        The Genie v2 API requires all repeated entries to be sorted by
        their natural key (``id``, ``identifier``, ``column_name``, or
        ``display_name``).
        """
        if isinstance(obj, dict):
            for value in obj.values():
                GenieRoomModel._sort_payload_lists(value)
        elif isinstance(obj, list):
            for item in obj:
                GenieRoomModel._sort_payload_lists(item)
            if obj and isinstance(obj[0], dict):
                for key in ("column_name", "identifier", "id", "display_name"):
                    if key in obj[0]:
                        obj.sort(key=lambda x: (x.get(key, ""),))
                        break

    def refresh(
        self,
        *,
        force: bool = False,
        payload: dict[str, Any] | None = None,
    ) -> Self:
        """Hydrate provisioning fields from the live ``serialized_space``.

        Inverse of :meth:`_build_serialized_space`. Mutates the structured
        provisioning fields (``table_sources``, ``text_instructions``, etc.)
        in place from the JSON payload stored on the Genie space, so the
        same model can then be locally edited and pushed back via
        :meth:`create`.

        Args:
            force: If True, invalidate the cached ``_space_details`` before
                re-fetching. Use after a write so subsequent reads see the
                post-write state.
            payload: Optional pre-parsed serialized_space dict. Bypasses the
                network entirely — primarily for tests and for callers that
                already hold the JSON.

        Returns:
            self, for chaining.
        """
        if payload is None:
            if force:
                self._space_details = None
            payload = self._parse_serialized_space()

        if not payload:
            return self

        self._apply_serialized_space(payload)
        return self

    def _apply_serialized_space(self, payload: dict[str, Any]) -> None:
        """Write a parsed ``serialized_space`` payload into the model fields."""

        # config.sample_questions
        cfg = payload.get("config")
        if isinstance(cfg, dict):
            sample = cfg.get("sample_questions")
            if isinstance(sample, list):
                self.sample_questions = [
                    _unwrap_text(item.get("question"))
                    if isinstance(item, dict)
                    else None
                    for item in sample
                ]
                self.sample_questions = [q for q in self.sample_questions if q]

        data_sources = payload.get("data_sources")
        if isinstance(data_sources, dict):
            tables_data = data_sources.get("tables")
            if isinstance(tables_data, list):
                self.table_sources = [
                    self._table_source_from_payload(entry)
                    for entry in tables_data
                    if isinstance(entry, (dict, str))
                ]
            metric_views_data = data_sources.get("metric_views")
            if isinstance(metric_views_data, list):
                self.metric_view_sources = [
                    GenieMetricViewSource(
                        table=TableModel(name=_identifier_of(entry)),
                        description=_unwrap_text(entry.get("description"))
                        if isinstance(entry, dict)
                        else None,
                    )
                    for entry in metric_views_data
                    if _identifier_of(entry)
                ]

        instructions = payload.get("instructions")
        if isinstance(instructions, dict):
            text_instructions = instructions.get("text_instructions")
            if isinstance(text_instructions, list):
                self.text_instructions = [
                    _unwrap_text(item.get("content"))
                    for item in text_instructions
                    if isinstance(item, dict) and item.get("content")
                ]
                self.text_instructions = [t for t in self.text_instructions if t]

            example_sqls = instructions.get("example_question_sqls")
            if isinstance(example_sqls, list):
                self.example_sqls = [
                    self._example_sql_from_payload(entry)
                    for entry in example_sqls
                    if isinstance(entry, dict)
                ]

            sql_functions = instructions.get("sql_functions")
            if isinstance(sql_functions, list):
                self.function_sources = [
                    GenieSqlFunctionSource(
                        function=FunctionModel(name=_identifier_of(entry))
                    )
                    for entry in sql_functions
                    if _identifier_of(entry)
                ]

            join_specs = instructions.get("join_specs")
            if isinstance(join_specs, list):
                self.join_specs = [
                    self._join_spec_from_payload(entry)
                    for entry in join_specs
                    if isinstance(entry, dict)
                ]

            snippets = instructions.get("sql_snippets")
            if isinstance(snippets, dict):
                for snippet_key, target_attr in (
                    ("filters", "sql_filters"),
                    ("expressions", "sql_expressions"),
                    ("measures", "sql_measures"),
                ):
                    items = snippets.get(snippet_key)
                    if isinstance(items, list):
                        setattr(
                            self,
                            target_attr,
                            [
                                self._snippet_from_payload(entry)
                                for entry in items
                                if isinstance(entry, dict)
                            ],
                        )

        benchmarks = payload.get("benchmarks")
        if isinstance(benchmarks, dict):
            questions = benchmarks.get("questions")
            if isinstance(questions, list):
                self.benchmarks = [
                    self._benchmark_from_payload(entry)
                    for entry in questions
                    if isinstance(entry, dict)
                ]

    @staticmethod
    def _table_source_from_payload(entry: Any) -> "GenieTableSource":
        if isinstance(entry, str):
            return GenieTableSource(table=TableModel(name=entry))
        identifier = entry.get("identifier") or entry.get("name")
        column_configs_raw = entry.get("column_configs") or []
        column_configs: list[GenieColumnConfig] = []
        for cc in column_configs_raw:
            if not isinstance(cc, dict):
                continue
            column_configs.append(
                GenieColumnConfig(
                    name=cc.get("column_name") or cc.get("name") or "",
                    description=_unwrap_text(cc.get("description")),
                    synonyms=cc.get("synonyms") or None,
                    excluded=bool(cc.get("exclude", False)),
                    sample_values=bool(
                        cc.get(
                            "enable_format_assistance",
                            cc.get("get_example_values", True),
                        )
                    ),
                    build_value_dictionary=bool(
                        cc.get(
                            "enable_entity_matching",
                            cc.get("build_value_dictionary", False),
                        )
                    ),
                )
            )
        return GenieTableSource(
            table=TableModel(name=identifier),
            description=_unwrap_text(entry.get("description")),
            column_configs=column_configs or None,
        )

    @staticmethod
    def _example_sql_from_payload(entry: dict[str, Any]) -> "GenieExampleSql":
        params_raw = entry.get("parameters") or []
        parameters: list[GenieSqlParameter] = []
        for p in params_raw:
            if not isinstance(p, dict):
                continue
            parameters.append(
                GenieSqlParameter(
                    name=p.get("name", ""),
                    type_hint=p.get("type_hint", "STRING"),
                    description=_unwrap_text(p.get("description")),
                )
            )
        return GenieExampleSql(
            question=_unwrap_text(entry.get("question")) or "",
            sql=_unwrap_text(entry.get("sql")) or "",
            parameters=parameters or None,
            usage_guidance=_unwrap_text(entry.get("usage_guidance")),
        )

    @staticmethod
    def _join_spec_from_payload(entry: dict[str, Any]) -> "GenieJoinSpec":
        sql_text: str = _unwrap_text(entry.get("sql")) or ""
        relationship_type: GenieRelationshipType | None = None
        rt_match = re.search(r"\s*--rt=([A-Z_]+)--\s*$", sql_text)
        if rt_match:
            try:
                relationship_type = GenieRelationshipType(rt_match.group(1))
            except ValueError:
                relationship_type = None
            sql_text = sql_text[: rt_match.start()].rstrip()
        left = entry.get("left") or {}
        right = entry.get("right") or {}
        return GenieJoinSpec(
            left=TableModel(name=left.get("identifier", "")),
            left_alias=left.get("alias"),
            right=TableModel(name=right.get("identifier", "")),
            right_alias=right.get("alias"),
            sql=sql_text,
            relationship_type=relationship_type,
            comment=_unwrap_text(entry.get("comment")),
        )

    @staticmethod
    def _snippet_from_payload(entry: dict[str, Any]) -> "GenieSqlSnippet":
        return GenieSqlSnippet(
            display_name=entry.get("display_name", ""),
            sql=_unwrap_text(entry.get("sql")) or "",
            instruction=_unwrap_text(entry.get("instruction")),
            synonyms=entry.get("synonyms") or None,
        )

    @staticmethod
    def _benchmark_from_payload(entry: dict[str, Any]) -> "GenieBenchmarkQuestion":
        question_text: str = _unwrap_text(entry.get("question")) or ""
        answers = entry.get("answer") or []
        expected_sql: str = ""
        if answers and isinstance(answers, list):
            first = answers[0]
            if isinstance(first, dict):
                expected_sql = _unwrap_text(first.get("content")) or ""
        return GenieBenchmarkQuestion(
            question=question_text,
            expected_sql=expected_sql,
        )

    @classmethod
    def from_space(
        cls,
        space_id: str,
        *,
        w: WorkspaceClient | None = None,
        **auth_kwargs: Any,
    ) -> Self:
        """Construct a fully-hydrated ``GenieRoomModel`` from an existing space.

        Convenience factory equivalent to::

            room = GenieRoomModel(space_id=space_id, **auth_kwargs)
            room.ensure_resolved()
            room.refresh()

        Args:
            space_id: The Databricks Genie space identifier.
            w: Optional pre-built ``WorkspaceClient``. When omitted, falls
                back to ambient/auth credentials configured on the model.
            **auth_kwargs: Forwarded to ``GenieRoomModel.__init__`` for
                callers using PAT / service-principal auth.

        Returns:
            A new ``GenieRoomModel`` instance with structured fields
            populated from the live space.
        """
        instance = cls(space_id=space_id, **auth_kwargs)
        instance.ensure_resolved()
        if w is not None:
            instance._space_details = w.genie.get_space(
                space_id=space_id, include_serialized_space=True
            )
        instance.refresh()
        return instance

    @classmethod
    def from_space_id(
        cls,
        space_id: Optional[str],
        *,
        w: WorkspaceClient | None = None,
        **auth_kwargs: Any,
    ) -> Optional[Self]:
        """Tolerant variant of :meth:`from_space`: returns ``None`` when the
        space does not exist (or when ``space_id`` is empty/None) instead of
        raising.

        Use this when a caller has a *candidate* space_id and wants to know
        whether it refers to a live space — e.g., a provisioning task that
        will create a fresh space if the candidate is stale or unset.

        Args:
            space_id: A candidate space id (may be empty or None).
            w: Optional pre-built ``WorkspaceClient``.
            **auth_kwargs: Forwarded to :meth:`from_space`.

        Returns:
            A populated ``GenieRoomModel`` if the space exists, else ``None``.
        """
        if not space_id:
            return None
        try:
            return cls.from_space(space_id, w=w, **auth_kwargs)
        except NotFound:
            return None

    @classmethod
    def from_name(
        cls,
        name: Optional[str],
        *,
        w: WorkspaceClient | None = None,
        **auth_kwargs: Any,
    ) -> Optional[Self]:
        """Find a Genie space by its title. Returns a hydrated
        ``GenieRoomModel`` for the **most-recently created** match, or
        ``None`` when no space matches.

        Unlike :meth:`_resolve_space_id_by_name`, this method is *tolerant*
        of duplicate titles — Genie does not enforce unique titles and
        re-running provisioning workflows can leave multiple spaces with
        the same name. We pick the freshest match rather than raise, so
        repeated runs of a provisioning task converge on a single space.

        Args:
            name: Genie space title to look up (may be empty or None).
            w: Optional pre-built ``WorkspaceClient``.
            **auth_kwargs: Forwarded to :meth:`from_space`.

        Returns:
            A populated ``GenieRoomModel`` if at least one space matches,
            else ``None``.
        """
        if not name:
            return None
        client: WorkspaceClient = w or WorkspaceClient()
        matches: list = []
        page_token: Optional[str] = None
        while True:
            resp = client.genie.list_spaces(page_token=page_token)
            if resp.spaces:
                for sp in resp.spaces:
                    if sp.title == name:
                        matches.append(sp)
            if not resp.next_page_token:
                break
            page_token = resp.next_page_token
        if not matches:
            return None
        matches.sort(key=lambda s: getattr(s, "created_time", 0) or 0, reverse=True)
        return cls.from_space(matches[0].space_id, w=client, **auth_kwargs)

    def create(self, w: WorkspaceClient | None = None) -> None:
        """Create or update this Genie space.

        Behavior:

        - **Create**: When ``space_id`` is not set after :meth:`ensure_resolved`,
          a new space is provisioned via ``WorkspaceClient.genie.create_space``
          and the resulting ID is written back to ``self.space_id``.
        - **Update**: When ``space_id`` is set, the live ``serialized_space``
          is fetched and compared against the locally-built payload. If
          anything changed, ``WorkspaceClient.genie.update_space`` is called.
        - **Entitlements**: After create/update, any configured
          :class:`GenieEntitlement` grants are applied via
          ``WorkspaceClient.permissions.set``.

        Provisioning requires :attr:`warehouse` to be set.
        """
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w)
        provider.create_genie_space(self)


def _unwrap_text(value: Any) -> str | None:
    """Genie stores most string fields as one-element lists. Unwrap to a plain string."""
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    if isinstance(value, str):
        return value
    return None


def _identifier_of(entry: Any) -> str | None:
    """Pull the ``identifier`` from a ``data_sources.*[]`` entry, with fallbacks."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("identifier") or entry.get("name")
    return None


class VolumeModel(IsDatabricksResource, HasFullName, Provisionable):
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


class VolumePathModel(BaseModel, HasFullName, Provisionable):
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


class SkillModel(BaseModel):
    """A deepagents skill — a directory of Markdown content that teaches a deep_agent how to do a task.

    A skill at minimum contains ``SKILL.md`` (deepagents convention). It may also contain
    ``AGENTS.md`` for memory, plus arbitrary supporting files referenced by the SKILL.

    Skills can live in two places, both expressed via the ``path`` field:

    * **Local** — a plain string relative path under the project root
      (e.g. ``skills/research``). The directory is bundled with the model
      artifact via ``code_paths`` and shipped with both Model Serving and
      Databricks Apps deployments.
    * **Volume** — a ``VolumePathModel`` referencing a Unity Catalog volume
      (``/Volumes/<cat>/<schema>/<vol>/...``). The path is read directly at
      runtime and the volume is wired as a deployment resource for permission
      grants. Use this when skills are governed centrally.

    A raw absolute string starting with ``/Volumes/`` is auto-promoted to a
    ``VolumePathModel`` by the pre-validator, so users may copy-paste paths
    from the UC explorer without writing the structured form. The structured
    form (``volume:`` + ``path:``) is preferred for governed skills because
    it gives the deployment pipeline a typed handle to the volume.

    Skills are referenced from ``orchestration.deep_agent.skills`` (or per-subagent
    ``subagents[].skills``) and resolved to filesystem paths that deepagents'
    ``SkillsMiddleware`` can load.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Unique skill name. Used by deepagents' SkillsMiddleware as the skill identifier.",
    )
    path: str | VolumePathModel = Field(
        description=(
            "Skill source directory. Either a local relative path string "
            "(e.g. ``skills/research``) or a ``VolumePathModel`` referencing "
            "a Unity Catalog volume. Raw ``/Volumes/...`` strings are auto-promoted "
            "to ``VolumePathModel`` by the pre-validator."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of what this skill does. Surfaced in docs and traces.",
    )

    @model_validator(mode="before")
    @classmethod
    def _promote_volume_path_string(cls, data: Any) -> Any:
        """Auto-promote raw ``/Volumes/...`` strings to a ``VolumePathModel``.

        This lets users paste an absolute UC volume path verbatim without
        having to spell out the ``volume:`` + ``path:`` structure. The
        structured form remains preferred for governed skills.
        """
        if isinstance(data, dict):
            raw_path = data.get("path")
            if isinstance(raw_path, str) and raw_path.startswith("/Volumes/"):
                data = {**data, "path": {"path": raw_path}}
        return data

    @property
    def is_volume_backed(self) -> bool:
        """True if ``path`` is a Unity Catalog volume reference."""
        return isinstance(self.path, VolumePathModel)

    @property
    def runtime_path(self) -> str:
        """The filesystem path passed to ``deepagents.create_deep_agent``.

        For local skills this is the raw relative string. For volume skills
        this is the fully-qualified ``/Volumes/<cat>/<schema>/<vol>/<sub>``
        path composed by ``VolumePathModel.full_name``.
        """
        if isinstance(self.path, VolumePathModel):
            return self.path.full_name
        return self.path

    def as_resources(self) -> Sequence[DatabricksResource]:
        """Emit deployment resources for the underlying volume, if any.

        Local skills emit nothing — they're shipped via ``code_paths``. Volume
        skills delegate to their underlying ``VolumeModel`` so existing
        volume-permission logic is reused without duplication.
        """
        if isinstance(self.path, VolumePathModel) and self.path.volume:
            return self.path.volume.as_resources()
        return []

    def as_middleware(self) -> "MiddlewareModel":
        """Build a ``MiddlewareModel`` invoking the SkillsMiddleware factory.

        Returns a single-source ``MiddlewareModel`` pointed at
        :func:`dao_ai.middleware.skills.create_skills_middleware`. Used by
        :class:`AgentModel` to convert ``agent.skills`` entries into middleware
        on the agent's stack at config-load time.

        Path resolution: ``sources`` is set to the *parent* of the skill leaf
        (the deepagents SkillsMiddleware source-dir convention — it lists
        subdirs and reads ``SKILL.md`` from each). For local skills the
        leaf is resolved against the runtime anchors (env var, CWD, ``sys.path``)
        at call time so the absolute path is baked into the MiddlewareModel.
        Volume-backed skills use the volume root.

        For the filesystem backend, ``root_dir="/"`` is used because the
        runtime resolver returns absolute paths.
        """
        # Lazy import to avoid touching dao_ai.skills at class definition time
        # (would create an import cycle via config → skills → config).
        from dao_ai.skills import _resolve_runtime_path

        if isinstance(self.path, VolumePathModel):
            # Parent of /Volumes/.../<skill> is the volume root or a subdir
            # of it — pass that as the source dir.
            full = self.runtime_path.rstrip("/")
            parent = full.rsplit("/", 1)[0] or "/"
            return MiddlewareModel(
                name="dao_ai.middleware.skills.create_skills_middleware",
                args={
                    "sources": [parent],
                    "backend_type": "volume",
                    "volume_path": self.path,
                },
            )

        # Local skill: resolve to absolute parent dir against runtime anchors.
        resolved = _resolve_runtime_path(self.path)
        # Fall back to the raw path if not found (will log a warning at
        # SkillsMiddleware.ls() time but won't crash agent build).
        leaf = str(resolved) if resolved is not None else self.path
        parent = leaf.rstrip("/").rsplit("/", 1)[0] or "/"
        return MiddlewareModel(
            name="dao_ai.middleware.skills.create_skills_middleware",
            args={
                "sources": [parent],
                "backend_type": "filesystem",
                "root_dir": "/",
            },
        )


class VectorStoreModel(IsDatabricksResource, ManagedResource):
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
    embedding_model: Optional[InferenceEndpointModel] = Field(
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

    _index_details: Optional[dict[str, Any]] = PrivateAttr(default=None)

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
            self.embedding_model = InferenceEndpointModel(
                name="databricks-gte-large-en"
            )
        return self

    def ensure_resolved(self) -> None:
        """Auto-discover primary key in provisioning mode via API."""
        if self._resolved:
            return
        super().ensure_resolved()
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

    def refresh(
        self,
        *,
        force: bool = False,
        vsc: VectorSearchClient | None = None,
        details: dict[str, Any] | None = None,
    ) -> Self:
        """Hydrate fields from a live vector search index's ``describe()`` response.

        Used in "existing index" mode: takes a model with just an ``index``
        reference and populates ``source_table``, ``embedding_source_column``,
        ``embedding_model``, ``endpoint``, ``primary_key``, and ``columns``
        from the live index spec.

        Args:
            force: If True, invalidate the cached describe response before
                re-fetching.
            vsc: Optional ``VectorSearchClient`` to use for the lookup.
            details: Optional pre-fetched describe dict (for tests / callers
                that already hold the response).

        Returns:
            self, for chaining.
        """
        if self.index is None:
            raise ValueError(
                "VectorStoreModel.refresh() requires an 'index' reference."
            )

        if details is None:
            if force:
                self._index_details = None
            if self._index_details is None:
                from dao_ai.providers.databricks import DatabricksProvider

                provider: DatabricksProvider = DatabricksProvider(vsc=vsc)
                live_index: VectorSearchIndex = provider.get_vector_index(self)
                self._index_details = live_index.describe()
            details = self._index_details

        if not isinstance(details, dict):
            return self

        delta_spec = details.get("delta_sync_index_spec") or {}
        source_table_name = delta_spec.get("source_table")
        if source_table_name:
            self.source_table = TableModel(
                name=source_table_name,
                on_behalf_of_user=self.on_behalf_of_user,
                service_principal=self.service_principal,
                client_id=self.client_id,
                client_secret=self.client_secret,
                workspace_host=self.workspace_host,
                pat=self.pat,
            )

        embedding_source_columns = delta_spec.get("embedding_source_columns") or []
        if embedding_source_columns:
            first = embedding_source_columns[0]
            if isinstance(first, dict):
                self.embedding_source_column = first.get("name")
                model_endpoint_name = first.get("embedding_model_endpoint_name")
                if model_endpoint_name:
                    self.embedding_model = InferenceEndpointModel(
                        name=model_endpoint_name
                    )

        endpoint_name = details.get("endpoint_name")
        if endpoint_name:
            self.endpoint = VectorSearchEndpoint(name=endpoint_name)

        primary_key = details.get("primary_key")
        if primary_key:
            self.primary_key = primary_key

        columns_to_sync = delta_spec.get("columns_to_sync")
        if columns_to_sync:
            self.columns = list(columns_to_sync)

        return self

    @classmethod
    def from_index(
        cls,
        index_name: str,
        *,
        vsc: VectorSearchClient | None = None,
        **auth_kwargs: Any,
    ) -> Self:
        """Construct a fully-hydrated ``VectorStoreModel`` from an existing index.

        Convenience factory equivalent to::

            vs = VectorStoreModel(index=IndexModel(name=index_name), **auth_kwargs)
            vs.ensure_resolved()
            vs.refresh(vsc=vsc)

        Args:
            index_name: Fully qualified UC index name (``catalog.schema.index``).
            vsc: Optional ``VectorSearchClient``.
            **auth_kwargs: Forwarded to ``VectorStoreModel.__init__``.

        Returns:
            A new ``VectorStoreModel`` with structured fields populated.
        """
        instance = cls(index=IndexModel(name=index_name), **auth_kwargs)
        instance.ensure_resolved()
        instance.refresh(vsc=vsc)
        return instance

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
    Configuration for database connections supporting Databricks Lakebase (autoscaling) and standard PostgreSQL.

    Authentication is inherited from IsDatabricksResource. Additionally supports:
    - user/password: For user-based database authentication

    Connection Types (determined by fields provided):
    - Lakebase (autoscaling): Provide `project` (and optionally `branch`).
      ``instance_name`` is accepted as a deprecated alias for ``project``.
    - Standard PostgreSQL: Provide `host` (authentication required via user/password)

    For Lakebase connections, `name` defaults to `project`.
    For PostgreSQL connections, `name` is required.

    Example Lakebase (minimal):
    ```yaml
    databases:
      my_lakebase:
        project: my-lakebase-project  # name defaults to project
    ```

    Example Lakebase with branch and Service Principal:
    ```yaml
    databases:
      my_lakebase:
        project: my-lakebase-project
        branch: main                  # optional, auto-resolved if omitted
        client_id:
          env: SERVICE_PRINCIPAL_CLIENT_ID
        client_secret:
          scope: my-scope
          secret: sp-client-secret
        workspace_host:
          env: DATABRICKS_HOST
    ```

    Example Lakebase with Ambient Authentication:
    ```yaml
    databases:
      my_lakebase:
        project: my-lakebase-project
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
        description="Logical database name. For Lakebase, defaults to project.",
    )
    project: Optional[str] = Field(
        default=None,
        description="Lakebase autoscaling project name.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of this database connection.",
    )
    host: Optional[AnyVariable] = Field(
        default=None,
        description="PostgreSQL host address. Not needed for Lakebase.",
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
        default=None,
        description=(
            "Pool-level timeout in seconds (how long to wait for a free connection). "
            "Defaults to 120 for Lakebase (to allow endpoint wake-up) "
            "and 30 for other database types."
        ),
    )
    connect_timeout: Optional[int] = Field(
        default=None,
        description=(
            "TCP-level connection timeout in seconds passed to libpq via psycopg. "
            "Limits how long a new connection attempt waits for the database to respond. "
            "Defaults to 30 for Lakebase (suspended endpoints need wake-up time) "
            "and 10 for other database types."
        ),
    )
    # --- Lakebase fields (only valid with project) ---
    branch: Optional[str] = Field(
        default=None,
        description="Lakebase branch name. If omitted, the default branch is auto-resolved.",
    )
    autoscaling_min_cu: Optional[int] = Field(
        default=2,
        description="Minimum compute units for autoscaling Lakebase.",
    )
    autoscaling_max_cu: Optional[int] = Field(
        default=4,
        description="Maximum compute units for autoscaling Lakebase.",
    )
    suspend_timeout_seconds: Optional[int] = Field(
        default=600,
        description=(
            "Seconds of inactivity before the Lakebase endpoint suspends. "
            "Valid range is 60-604800 (1 min to 1 week). "
            "Set to 0 or negative to disable suspension (always on)."
        ),
    )
    # --- Common auth fields ---
    user: Optional[AnyVariable] = Field(
        default=None,
        description="Database username. For Lakebase, auto-detected from workspace identity.",
    )
    password: Optional[AnyVariable] = Field(
        default=None,
        description="Database password. For Lakebase, a token is generated automatically.",
    )

    @model_validator(mode="before")
    @classmethod
    def _alias_instance_name_to_project(cls, data: Any) -> Any:
        """Accept ``instance_name`` as a deprecated alias for ``project``."""
        if isinstance(data, dict) and "instance_name" in data:
            import warnings

            warnings.warn(
                "DatabaseModel field 'instance_name' is deprecated. "
                "Use 'project' instead — Lakebase only supports autoscaling going forward.",
                DeprecationWarning,
                stacklevel=4,
            )
            if "project" not in data or data["project"] is None:
                data["project"] = data.pop("instance_name")
            else:
                data.pop("instance_name")
        return data

    @property
    def api_scopes(self) -> Sequence[str]:
        if self.is_lakebase:
            return ["postgres"]
        return []

    @property
    def is_lakebase(self) -> bool:
        """Returns True if this is a Databricks Lakebase connection."""
        return self.project is not None

    @property
    def is_lakebase_autoscaling(self) -> bool:
        """Alias for ``is_lakebase`` — all Lakebase is autoscaling now."""
        return self.is_lakebase

    def as_resources(self) -> Sequence[DatabricksResource]:
        # Lakebase autoscaling projects intentionally do NOT emit a
        # DatabricksLakebase MLflow resource. MLflow's DatabricksLakebase
        # resource only supports the deprecated provisioned-instance shape;
        # logging it for an autoscaling project makes the Model Serving
        # endpoint fail to start with:
        #
        #   NOT_FOUND: Database instance is not found. Please ensure all
        #   resource dependencies for the server entity are valid...
        #
        # MLflow team confirmed this gap is not planned for the time being
        # (https://github.com/mlflow/mlflow/issues/22452, 2026-04-10).
        # The recommended workaround is to manage Lakebase auth from within
        # the agent code using OAuth M2M (set ``client_id`` and
        # ``client_secret`` on the DatabaseModel, or surface them via the
        # secret-scope-wrapped pattern used in the workshop's Lab 7 YAML).
        #
        # The Apps deploy path is unaffected -- it uses the platform's
        # ``postgres`` resource binding (see
        # ``dao_ai.apps.resources._extract_database_resources``), which
        # supports autoscaling natively.
        #
        # Standalone PostgreSQL hosts have no Databricks-managed resource
        # binding and likewise return [].
        if self.is_lakebase and self.project:
            logger.debug(
                "Lakebase database is autoscaling -- skipping DatabricksLakebase "
                "MLflow resource emission. Use OAuth client_id/client_secret on "
                "the DatabaseModel for Model Serving auth. See "
                "https://github.com/mlflow/mlflow/issues/22452.",
                project=self.project,
            )
        return []

    @model_validator(mode="after")
    def validate_connection_type(self) -> Self:
        """Validate that either ``project`` (Lakebase) or ``host`` (PostgreSQL) is provided."""
        if not self.project and not self.host:
            raise ValueError(
                "One of 'project' (Lakebase) or 'host' (PostgreSQL) must be provided."
            )
        return self

    @model_validator(mode="after")
    def resolve_timeout_seconds(self) -> Self:
        """Set default timeout_seconds based on database type.

        Lakebase endpoints may be suspended and need 30-60s to wake up,
        so they get a longer default timeout (120s).
        """
        if self.timeout_seconds is None:
            self.timeout_seconds = 120 if self.is_lakebase else 30
        return self

    @model_validator(mode="after")
    def resolve_connect_timeout(self) -> Self:
        """Set default connect_timeout based on database type.

        This is the TCP-level timeout (libpq ``connect_timeout``), distinct
        from the pool-level ``timeout_seconds``.  Lakebase endpoints may
        be suspended and need extra time to accept the TCP handshake.
        """
        if self.connect_timeout is None:
            self.connect_timeout = 30 if self.is_lakebase else 10
        return self

    @model_validator(mode="after")
    def populate_name(self) -> Self:
        """Populate name from project if not provided."""
        if self.name is None and self.project:
            self.name = self.project
        elif self.name is None:
            raise ValueError(
                "Either 'name' or 'project' must be provided for DatabaseModel."
            )
        return self

    def _resolve_user(self) -> None:
        """Resolve current user via API. Called from ensure_resolved()."""
        if self.on_behalf_of_user or self.client_id or self.user or self.pat:
            return

        if not self.is_lakebase:
            try:
                self.user = self.workspace_client.current_user.me().user_name
            except Exception as e:
                logger.warning(
                    f"Could not determine current user for PostgreSQL database: {e}. "
                    f"Please provide explicit user credentials."
                )
        else:
            try:
                self.user = self.workspace_client.current_user.me().user_name
            except Exception:
                pass

    def ensure_resolved(self) -> None:
        """Resolve user via API."""
        if self._resolved:
            return
        super().ensure_resolved()
        self._resolve_user()

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

        # Standard PostgreSQL requires explicit auth; Lakebase supports ambient auth
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

    def resolve_default_branch(self) -> str:
        """Return the configured branch, or resolve the project's default branch from the API.

        Returns the branch id (e.g. ``"production"``), not the full resource path.
        """
        if self.branch:
            return self.branch
        project_name = f"projects/{self.project}"
        w: WorkspaceClient = self.workspace_client
        branches = list(w.postgres.list_branches(project_name))
        if not branches:
            raise ValueError(
                f"No branches found for Lakebase project '{self.project}'."
            )
        default_branch = next(
            (b for b in branches if b.status and b.status.default),
            branches[0],
        )
        return default_branch.name.rsplit("/", 1)[-1]

    @property
    def connection_params(self) -> dict[str, Any]:
        """
        Get database connection parameters for **standard PostgreSQL only**.

        Lakebase connections should use ``databricks_langchain``
        ``AsyncCheckpointSaver`` / ``AsyncDatabricksStore`` directly — they
        manage host resolution and credential rotation internally.

        Raises ``ValueError`` if called on a Lakebase database.
        """
        if self.is_lakebase:
            raise ValueError(
                "connection_params is not supported for Lakebase databases. "
                "Use databricks_langchain AsyncCheckpointSaver / AsyncDatabricksStore instead."
            )

        host_value: Any = self.host
        if host_value is None:
            raise ValueError(
                f"Database host not configured for {self.name}. "
                "Please provide 'host' explicitly."
            )

        host: str = value_of(host_value)
        port: int = value_of(self.port)
        database: str = value_of(self.database)
        username: str | None = value_of(self.user) if self.user else None
        password_value: str | None = value_of(self.password) if self.password else None

        if not username or not password_value:
            raise ValueError(
                f"Standard PostgreSQL databases require both 'user' and 'password'. "
                f"Database: {self.name}"
            )

        params: dict[str, Any] = {
            "dbname": database,
            "host": host,
            "port": port,
            "user": username,
            "password": password_value,
            "sslmode": "require",
            "connect_timeout": self.connect_timeout,
        }

        logger.debug(
            f"Connection params: dbname={database} user={username} host={host} "
            f"port={port} password=******** sslmode=require connect_timeout={self.connect_timeout}"
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

        if w is None:
            w = self.workspace_client
        provider: DatabricksProvider = DatabricksProvider(w=w)
        if self.is_lakebase:
            provider.create_lakebase_autoscaling(self)
            provider.create_lakebase_autoscaling_role(self)


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
    invalidate_on_empty_result: bool = Field(
        default=False,
        description=(
            "When true, cached SQL that returns an empty result set is treated as stale: "
            "the cache entry is invalidated and the question is re-sent to Genie. "
            "Useful when cached queries contain date-relative expressions like CURRENT_DATE() "
            "that become invalid as underlying data ages."
        ),
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
    embedding_model: str | InferenceEndpointModel = Field(
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
    invalidate_on_empty_result: bool = Field(
        default=False,
        description=(
            "When true, cached SQL that returns an empty result set is treated as stale: "
            "the cache entry is invalidated and the question is re-sent to Genie. "
            "Useful when cached queries contain date-relative expressions like CURRENT_DATE() "
            "that become invalid as underlying data ages."
        ),
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

    model: Optional["InferenceEndpointModel"] = Field(
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

    model: Optional["InferenceEndpointModel"] = Field(
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

    model: Optional["InferenceEndpointModel"] = Field(
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

    model: Optional["InferenceEndpointModel"] = Field(
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

    LangChain supports four decision types:
    - "approve": Execute tool with original arguments
    - "edit": Modify arguments before execution
    - "reject": Skip execution with optional feedback message
    - "respond": Reply with a message in place of executing the tool. The
      reviewer's text becomes a synthetic ToolMessage that the LLM consumes
      as if the tool had returned it. Distinct from "reject", which skips
      the tool and ends the turn.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    review_prompt: Optional[str] = Field(
        default=None,
        description="Message shown to the reviewer when approval is requested",
    )

    allowed_decisions: list[Literal["approve", "edit", "reject", "respond"]] = Field(
        default_factory=lambda: ["approve", "edit", "reject"],
        description=(
            "List of allowed decision types for this tool. Defaults to "
            "``['approve', 'edit', 'reject']``; add ``'respond'`` to let the "
            "reviewer reply in place of executing the tool."
        ),
    )

    @model_validator(mode="after")
    def validate_and_normalize_decisions(self) -> Self:
        """Validate and normalize allowed decisions."""
        if not self.allowed_decisions:
            raise ValueError("At least one decision type must be allowed")

        # Remove duplicates while preserving order
        seen: set[str] = set()
        unique_decisions: list[Literal["approve", "edit", "reject", "respond"]] = []
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

        result = create_factory_tool(self, **kwargs)
        if isinstance(result, (list, tuple)):
            return list(result)
        return [result]

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
            ``InferenceEndpointModel``.  Required when using the custom judge mode.
        prompt: Evaluation instructions using ``{{ inputs }}`` and
            ``{{ outputs }}`` template variables.  Required when using
            the custom judge mode.
        scorer: Fully qualified name of an MLflow ``Scorer`` class
            (e.g. ``"mlflow.genai.scorers.guardrails.DetectPII"``).
            Required when using the scorer-based mode.
        scorer_args: Keyword arguments forwarded to the scorer constructor
            (e.g. ``{"pii_entities": ["CREDIT_CARD", "SSN"]}``).
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
    model: Optional[str | InferenceEndpointModel] = Field(
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
        return self

    @model_validator(mode="after")
    def validate_llm_model(self) -> Self:
        if self.model is not None and isinstance(self.model, str):
            self.model = InferenceEndpointModel(name=self.model)
        return self

    def as_scorer(self) -> Any:
        """Return an MLflow ``Scorer`` instance for this guardrail.

        For scorer-based guardrails, imports and instantiates the class
        referenced by ``self.scorer`` with ``self.scorer_args``.

        For LLM-judge guardrails, creates a ``JudgeScorer`` wrapping
        ``mlflow.genai.judges.make_judge`` with the resolved prompt and
        model endpoint.
        """
        if self.scorer:
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
    embedding_model: Optional[InferenceEndpointModel] = Field(
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
    supervisor_auto_inject: bool = Field(
        default=False,
        description=(
            "Whether to inject memories into the supervisor's context. "
            "Disabled by default since the supervisor only routes requests "
            "and does not need memory context."
        ),
    )
    background_extraction: bool = Field(
        default=True,
        description=(
            "Extract memories in a background thread after each "
            "conversation turn (no latency impact on responses)."
        ),
    )
    extraction_model: Optional[InferenceEndpointModel] = Field(
        default=None,
        description=(
            "Separate LLM for memory extraction. Can be a smaller, "
            "cheaper model. When None, uses the agent's primary model."
        ),
    )
    query_model: Optional[InferenceEndpointModel] = Field(
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
    model: InferenceEndpointModel = Field(
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
    skills: list[SkillModel | str] = Field(
        default_factory=list,
        description=(
            "Skills available to this agent. Each entry produces a SkillsMiddleware "
            "(via :meth:`SkillModel.as_middleware`) appended to the agent's middleware "
            "stack. Strings are resolved against ``config.resources.skills``; inline "
            "``SkillModel`` entries are accepted directly. Works uniformly across "
            "supervisor, swarm, and deep_agent orchestration — including when this "
            "AgentModel becomes an implicit sub-agent under deep_agent."
        ),
    )
    response_format: Optional[ResponseFormatModel | type | str] = Field(
        default=None,
        description="Structured output format (Pydantic type, JSON schema, or ResponseFormatModel).",
    )
    recursion_limit: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of graph supersteps (LLM call + tool execution cycles) "
            "allowed per agent invocation. Prevents runaway iteration loops in tool-calling "
            "agents. When None, uses LangGraph's default (25)."
        ),
    )
    requires: list[str] = Field(
        default_factory=list,
        description=(
            "Names of prerequisite agents that must have run (i.e. produced an "
            "AIMessage tagged with their name) before this agent can be reached "
            "via a swarm handoff. When unmet, the handoff tool returns a refusal "
            "ToolMessage and 'active_agent' stays unchanged so the LLM can "
            "self-correct on its next step. Semantics: any-order, all-of. "
            "Currently enforced in the swarm pattern only; the supervisor pattern "
            "ignores this field. Cross-agent validation (unknown name, "
            "self-reference, cycles in the requires DAG) runs at config-build time."
        ),
    )

    @model_validator(mode="after")
    def validate_requires_no_self_reference(self) -> Self:
        """Reject ``requires`` entries that reference the agent itself."""
        if self.name in self.requires:
            raise ValueError(
                f"Agent '{self.name}' has a self-reference in 'requires'. "
                f"An agent cannot require itself as a prerequisite."
            )
        return self

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
    model: InferenceEndpointModel = Field(
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
    max_hops: int = Field(
        default=25,
        ge=1,
        description=(
            "Cross-agent hop ceiling for the parent swarm graph. Two agents "
            "agentic-handing-off to each other are bounded only by this "
            "value (set via LangGraph's recursion_limit on the compiled "
            "parent graph). Independent of per-worker recursion limits. "
            "Defaults to LangGraph's own default of 25."
        ),
    )

    @model_validator(mode="after")
    def validate_no_deterministic_handoff_in_cycle(self) -> Self:
        """Reject swarm configs where a deterministic edge participates in a cycle.

        A deterministic handoff transfers control unconditionally on every
        traversal. If any cycle in the handoff graph contains at least one
        deterministic edge, the workflow can run forever -- the deterministic
        edge guarantees re-entry, and the agentic edges that close the cycle
        only need an LLM whose prompt occasionally fires the handoff tool to
        keep the loop going.

        This validator runs at config load time and rejects the pattern with
        a clear cycle path so the user can break or reconfigure the cycle
        before any compute is spent.

        Allowed:
          * No cycle (``A -det-> B`` with no path back to A).
          * A cycle of all-agentic edges (LLMs can choose to terminate).

        Rejected:
          * Any cycle containing at least one deterministic edge.
        """
        if not self.handoffs:
            return self

        # Build edge list: list[(source_name, target_name, is_deterministic)]
        edges: list[tuple[str, str, bool]] = []
        for source, targets in self.handoffs.items():
            if not targets:
                continue
            for entry in targets:
                if isinstance(entry, HandoffRouteModel):
                    target_obj = entry.agent
                    is_det = entry.is_deterministic
                else:
                    target_obj = entry
                    is_det = False
                # Resolve AgentModel -> name; pass strings through.
                target_name: str = (
                    target_obj.name if hasattr(target_obj, "name") else str(target_obj)
                )
                # Skip self-references; they're handled (and rejected for
                # deterministic) at swarm-build time, not here.
                if target_name == source:
                    continue
                edges.append((source, target_name, is_det))

        # Adjacency list keyed by source -> list[(target, is_deterministic)]
        adj: dict[str, list[tuple[str, bool]]] = {}
        for u, v, det in edges:
            adj.setdefault(u, []).append((v, det))

        # For each deterministic edge (u, v), is there a path v -> ... -> u?
        # If yes, that path + the (u, v) edge forms a cycle containing a
        # deterministic edge.
        def find_path(start: str, goal: str) -> list[str] | None:
            """BFS for any path from start to goal. Returns the node sequence
            including endpoints, or None if no path exists."""
            if start == goal:
                return [start]
            from collections import deque

            queue: deque[tuple[str, list[str]]] = deque([(start, [start])])
            visited: set[str] = {start}
            while queue:
                node, path = queue.popleft()
                for nxt, _det in adj.get(node, []):
                    if nxt == goal:
                        return path + [nxt]
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, path + [nxt]))
            return None

        for u, v, det in edges:
            if not det:
                continue
            return_path: list[str] | None = find_path(v, u)
            if return_path is None:
                continue
            # Cycle = u -det-> v -> ... -> u. Format with edge annotations.
            full_path: list[str] = [u] + return_path  # u -> v -> ... -> u
            # Annotate the deterministic edge so the message is unambiguous.
            edge_str = f"{u} =[deterministic]=> " + " -> ".join(full_path[1:])
            raise ValueError(
                "Swarm has a deterministic handoff inside a cycle: "
                f"{edge_str}. Deterministic edges fire unconditionally on every "
                "traversal, so any path back to the source forms a runaway loop. "
                "Either remove the return path or make the deterministic edge "
                "agentic."
            )

        return self


class FilesystemPermissionModel(BaseModel):
    """Filesystem permission rule for deepagents tools.

    Mirrors ``deepagents.middleware.permissions.FilesystemPermission`` (a TypedDict).
    Rules are evaluated in declaration order; the first match wins. If no rule
    matches, the call is allowed.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    paths: list[str] = Field(
        description="Path patterns covered by this rule (e.g. ``['/skills/**', '/tmp/*']``).",
    )
    mode: Literal["allow", "deny"] = Field(
        default="allow",
        description="Whether to allow or deny operations on paths matching the patterns.",
    )
    operations: Optional[list[Literal["read", "write"]]] = Field(
        default=None,
        description=(
            "File operations this rule applies to. Defaults to both read and write "
            "when omitted."
        ),
    )


class BackendModel(BaseModel):
    """Storage/execution backend for a deep_agent.

    Mirrors deepagents' ``BackendProtocol`` factory pattern. ``name`` is a
    fully-qualified factory function or class (e.g. ``deepagents.backends.StateBackend``)
    and ``args`` are passed as keyword arguments. If omitted, deepagents'
    default ``StateBackend`` is used.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Fully qualified name of the backend class or factory function.",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the backend factory.",
    )

    @model_validator(mode="after")
    def resolve_args(self) -> Self:
        """Resolve any variable references in args (mirrors MiddlewareModel)."""
        for key, value in self.args.items():
            self.args[key] = value_of(value)
        return self


class SubAgentModel(BaseModel):
    """A deepagents sub-agent invoked by the main deep_agent via the ``task`` tool.

    Mirrors ``deepagents.SubAgent`` (a ``TypedDict``) but lifted into Pydantic so
    every field can accept dao-ai primitives:

    * ``model`` — string serving-endpoint name OR an ``InferenceEndpointModel``
    * ``tools`` — list of strings (looked up in ``config.tools``) OR ``ToolModel`` entries
    * ``skills`` — list of skill paths OR named ``SkillModel`` references
    * ``system_prompt`` — string OR ``PromptModel`` (resolved through ``make_prompt``)

    Required fields per deepagents are ``name``, ``description``, and
    ``system_prompt``. Optional fields override the parent deep_agent's defaults.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Unique sub-agent identifier. The main agent uses this when calling the ``task`` tool.",
    )
    description: str = Field(
        description="What this sub-agent does. Used by the main agent to decide when to delegate.",
    )
    system_prompt: str | PromptModel = Field(
        description="Instructions for the sub-agent. Inline string or MLflow Prompt Registry reference.",
    )
    tools: list[ToolModel | str] = Field(
        default_factory=list,
        description="Tools available to this sub-agent. Strings are resolved against ``config.tools``.",
    )
    model: Optional[InferenceEndpointModel | str] = Field(
        default=None,
        description=(
            "LLM model. Inherits from the parent deep_agent if omitted. "
            "Strings are passed verbatim to ``langchain.chat_models.init_chat_model`` "
            "(e.g. ``'openai:gpt-4o'``)."
        ),
    )
    middleware: list[MiddlewareModel] = Field(
        default_factory=list,
        description="Additional middleware applied to this sub-agent on top of the deepagents base stack.",
    )
    interrupt_on: dict[str, bool | HumanInTheLoopModel] = Field(
        default_factory=dict,
        description=(
            "Per-tool human-in-the-loop config. ``true`` enables defaults; a "
            "``HumanInTheLoopModel`` customizes the review prompt and allowed "
            "decisions. Same model used for tool-level ``human_in_the_loop:`` "
            "annotations (one concept, one shape)."
        ),
    )
    skills: list[SkillModel | str] = Field(
        default_factory=list,
        description="Skill source directories or SkillModel references scoped to this sub-agent.",
    )
    permissions: list[FilesystemPermissionModel] = Field(
        default_factory=list,
        description="Filesystem permission rules. Replace (not extend) the parent's rules when present.",
    )
    response_format: Optional[ResponseFormatModel | type | str] = Field(
        default=None,
        description="Structured output format. Same semantics as ``AgentModel.response_format``.",
    )


class DeepAgentModel(BaseModel):
    """Configuration for the ``deep_agent`` orchestration pattern.

    Wraps `deepagents.create_deep_agent` so every parameter is declarative in YAML
    and dao-ai primitives can be substituted wherever the deepagents API takes a
    string or callable.

    Layered with the existing ``OrchestrationModel.memory`` block: the
    ``checkpointer`` and ``store`` passed to ``create_deep_agent`` are derived
    from that block, not redeclared here.

    See ``orchestration/deep_agent.py::create_deep_agent_graph`` for resolution
    semantics.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    model: Optional[InferenceEndpointModel | str] = Field(
        default=None,
        description=(
            "Primary LLM. ``InferenceEndpointModel`` is resolved via ``as_chat_model()``; "
            "strings pass through to ``init_chat_model`` (e.g. ``'openai:gpt-4o'``). "
            "Defaults to deepagents' default (``claude-sonnet-4-6``) when omitted."
        ),
    )
    tools: list[ToolModel | str] = Field(
        default_factory=list,
        description=(
            "Tools merged with deepagents' built-in suite (todo, filesystem, execute, task). "
            "Strings are resolved against ``config.tools``."
        ),
    )
    system_prompt: Optional[str | PromptModel] = Field(
        default=None,
        description=(
            "System prompt prepended to deepagents' base prompt. "
            "Inline string or MLflow ``PromptModel`` (resolved via ``make_prompt``)."
        ),
    )
    middleware: list[MiddlewareModel] = Field(
        default_factory=list,
        description="User middleware inserted between the deepagents base stack and tail stack.",
    )
    subagents: list[SubAgentModel | AgentModel | str] = Field(
        default_factory=list,
        # The before-validator below also accepts dict[str, SubAgentModel | AgentModel]
        # and rewrites it to the list form, injecting the dict key as `name`.
        description=(
            "Sub-agents callable via the ``task`` tool. Three forms accepted: "
            "(1) inline ``SubAgentModel`` dict, (2) string referencing an entry in ``app.agents`` by name, "
            "(3) full ``AgentModel`` (carries over name/description/prompt/tools/model/middleware/response_format)."
        ),
    )
    skills: list[SkillModel | str] = Field(
        default_factory=list,
        description=(
            "Skill source paths exposed via deepagents' ``SkillsMiddleware``. "
            "Strings are inline relative paths; ``SkillModel`` references provide "
            "named, governed skills with optional Unity Catalog volume backing."
        ),
    )
    instruction_files: list[str] = Field(
        default_factory=list,
        description=(
            "Paths to ``AGENTS.md``-style instruction files loaded into the system "
            "prompt at startup (deepagents' MemoryMiddleware feature). Renamed from "
            "``memory`` to avoid collision with ``OrchestrationModel.memory`` "
            "(checkpointer/store/extraction). Despite deepagents' upstream naming "
            "these files are static instructions, not runtime memory."
        ),
    )
    permissions: list[FilesystemPermissionModel] = Field(
        default_factory=list,
        description="Filesystem permission rules applied to the main agent and inherited by sub-agents.",
    )
    response_format: Optional[ResponseFormatModel | type | str] = Field(
        default=None,
        description="Structured output format. Same semantics as ``AgentModel.response_format``.",
    )
    interrupt_on: dict[str, bool | HumanInTheLoopModel] = Field(
        default_factory=dict,
        description=(
            "Per-tool human-in-the-loop config. ``true`` enables defaults; "
            "a ``HumanInTheLoopModel`` customizes the review prompt and "
            "allowed decisions."
        ),
    )
    backend: Optional[BackendModel] = Field(
        default=None,
        description="Backend factory. If omitted, deepagents uses ``StateBackend()``.",
    )
    context_schema: Optional[str] = Field(
        default=None,
        description=(
            "Fully qualified name of a TypedDict/dataclass class defining run-scoped context. "
            "Resolved via importlib at graph-build time."
        ),
    )
    recursion_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Per-run graph recursion limit. ``None`` uses LangGraph's default (25).",
    )
    debug: bool = Field(
        default=False,
        description="Enable verbose debug output from deepagents.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Human-readable name attached to the compiled graph. Useful in MLflow trace dashboards.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_subagents(cls, data: Any) -> Any:
        """Accept ``subagents`` as a dict and rewrite to the list form.

        Allows the more idiomatic YAML shape:

        .. code-block:: yaml

            subagents:
              research:
                description: ...
                system_prompt: ...

        which gets normalized to the equivalent list form ``[{name: research, ...}]``
        before the regular validator runs. Mirrors the dict-keyed pattern used by
        ``resources.models``, ``tools``, and ``swarm.handoffs``.
        """
        if not isinstance(data, dict):
            return data
        subs = data.get("subagents")
        if not isinstance(subs, dict):
            return data
        normalized: list[Any] = []
        for name, spec in subs.items():
            if isinstance(spec, dict) and "name" not in spec:
                spec = {**spec, "name": name}
            normalized.append(spec)
        return {**data, "subagents": normalized}


class OrchestrationModel(BaseModel):
    """Multi-agent orchestration configuration.

    Exactly one of ``supervisor``, ``swarm``, ``deep_agent`` may be set. If none
    are set (e.g. the user only supplies ``memory:``), AppConfig auto-picks a
    sensible router based on the number of agents — ``deep_agent`` is opt-in
    only and never auto-selected.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    supervisor: Optional[SupervisorModel] = Field(
        default=None,
        description="Supervisor pattern: a central LLM routes tasks to sub-agents.",
    )
    swarm: Optional[SwarmModel | Literal[True]] = Field(
        default=None,
        description="Swarm pattern: agents hand off to each other via tool calls. Set to true for defaults.",
    )
    deep_agent: Optional[DeepAgentModel] = Field(
        default=None,
        description=(
            "Deep_agent pattern: a single planning agent with built-in todo/filesystem/shell/sub-agent tools, "
            "plus first-class skills and memory. Wraps ``deepagents.create_deep_agent``."
        ),
    )
    memory: Optional[MemoryModel] = Field(
        default=None,
        description="Memory configuration scoped to the orchestration layer (checkpointer, store, extraction).",
    )
    output_mode: Literal["full_history", "last_message"] = Field(
        default="last_message",
        description=(
            "How an agent's response flows back into parent state. "
            "``last_message`` (default) returns only the final AI response "
            "from each agent, isolating downstream consumers (supervisor or "
            "peer swarm agents) from worker-side message corruption — "
            "middleware mutations that interleave system/tool messages, "
            "parallel tool calls that some strict-validation LLMs reject in "
            "history, or orphan tool_result blocks. "
            "``full_history`` returns the agent's full local history, "
            "preserving cross-agent tool context (one agent can see another's "
            "structured tool outputs) at the cost of exposing downstream "
            "consumers to any worker-side malformed messages. Override per "
            "app via ``orchestration.output_mode: full_history`` when cross-"
            "agent tool context is required and the worker-side message-"
            "assembly path is known to be clean."
        ),
    )

    @model_validator(mode="after")
    def validate_and_normalize(self) -> Self:
        """Validate orchestration and normalize swarm shorthand."""
        # Convert swarm: true to SwarmModel()
        if self.swarm is True:
            self.swarm = SwarmModel()

        # At most one orchestration mode may be set. Allowing none -- AppConfig
        # fills in a default router (supervisor for >1 agent, swarm for 1) so
        # that `orchestration: { memory: ... }` is valid for single-agent apps
        # that only want the memory wiring. deep_agent is opt-in only and is
        # never auto-selected.
        active_modes: list[str] = [
            name
            for name, value in (
                ("supervisor", self.supervisor),
                ("swarm", self.swarm),
                ("deep_agent", self.deep_agent),
            )
            if value is not None
        ]
        if len(active_modes) > 1:
            raise ValueError(
                f"Cannot specify more than one orchestration mode at a time; got {active_modes}."
            )
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
    model: InferenceEndpointModel = Field(
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


class LongRunningModel(BaseModel):
    """Opt-in long-running agent configuration.

    Enables Responses-API-compatible kickoff/poll/cancel on top of any
    dao-ai agent, persisted in the referenced Lakebase ``database``.
    When present on ``AppModel``, requests can set ``background: true``
    (or ``custom_inputs.operation: retrieve|cancel``) on ``/invocations``;
    in Databricks Apps, strict ``/v1/responses*`` FastAPI routes are
    additionally mounted and translate to the same contract.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    database: DatabaseModel = Field(
        description="Lakebase (or PostgreSQL) database used to persist response rows and stream events. May be the same database used by the checkpointer.",
    )
    default_background: bool = Field(
        default=False,
        description="If True, requests are treated as long-running even when background is not explicitly set.",
    )
    max_duration_seconds: int = Field(
        default=1800,
        ge=1,
        description="Hard cap on any single background run. Tasks exceeding this are marked failed.",
    )
    poll_interval_seconds: float = Field(
        default=1.0,
        gt=0.0,
        description="Interval (seconds) used by streaming retrieve to poll the database for new events.",
    )
    responses_table_name: str = Field(
        default="dao_ai_responses",
        description="Name of the Lakebase table that stores one row per response.",
    )
    messages_table_name: str = Field(
        default="dao_ai_response_messages",
        description="Name of the Lakebase table that stores streamed events / final items.",
    )


class A2ATaskStoreModel(BaseModel):
    """A2A protocol task persistence configuration.

    Mirrors the dao-ai idiom shared by :class:`CheckpointerModel` and
    :class:`StoreModel`: an optional :class:`DatabaseModel` toggles the
    backing store. Absent → in-memory (tasks lost on restart); present →
    Lakebase/Postgres, persisted in ``table``. This is independent of
    :class:`LongRunningModel` — the two concepts (A2A task lifecycle vs
    Responses-API kickoff/poll/cancel) are configured separately.

    When the same ``DatabaseModel`` is referenced here, on
    ``memory.checkpointer.database``, and on ``app.long_running.database``,
    :class:`dao_ai.memory.postgres.AsyncPostgresPoolManager` dedupes by
    connection-string value, so all three share a single connection pool.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    database: Optional[DatabaseModel] = Field(
        default=None,
        description=(
            "Database for persistent task storage. When omitted, A2A tasks "
            "are held in process memory and lost on restart. When set, "
            "tasks persist in the configured ``table`` and share the "
            "AsyncPostgresPoolManager pool with the LangGraph checkpointer "
            "and LongRunningStore whenever those point at the same "
            "DatabaseModel."
        ),
    )
    table: str = Field(
        default="dao_ai_a2a_tasks",
        description="Table name for task persistence. Ignored when `database` is None.",
    )

    @property
    def storage_type(self) -> StorageType:
        """Infer storage type from database presence."""
        return StorageType.POSTGRES if self.database else StorageType.MEMORY


class ProviderModel(BaseModel):
    """Service-provider information advertised on the A2A Agent Card.

    Mirrors :class:`a2a.types.AgentProvider`. Both fields are required by
    the A2A spec.
    """

    model_config = ConfigDict(extra="forbid")
    organization: str = Field(
        description="Name of the organization providing the agent (e.g., 'Databricks Field Engineering').",
    )
    url: str = Field(
        description="URL for the provider's site or relevant documentation.",
    )


class A2ASkillModel(BaseModel):
    """Single skill advertised on the A2A Agent Card."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(description="Stable skill identifier (often the sub-agent name).")
    name: str = Field(description="Human-readable skill name.")
    description: Optional[str] = Field(
        default=None, description="Short summary of what this skill does."
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags shown on the Agent Card for skill discovery.",
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Example prompts illustrating how to invoke the skill.",
    )
    input_modes: Optional[list[str]] = Field(
        default=None,
        description="Supported input MIME types for this skill. Falls back to A2AModel.default_input_modes.",
    )
    output_modes: Optional[list[str]] = Field(
        default=None,
        description="Supported output MIME types for this skill. Falls back to A2AModel.default_output_modes.",
    )


class A2AModel(BaseModel):
    """Google A2A (Agent2Agent) protocol endpoint configuration.

    Every Databricks Apps deployment exposes the following routes by default
    (``AppModel.a2a`` defaults to a fresh ``A2AModel`` with ``enabled=True``):

    * ``GET  /.well-known/agent-card.json``  — Agent Card discovery.
    * ``POST /a2a``                          — JSON-RPC 2.0 (message/send,
      message/stream, tasks/get, tasks/list, tasks/cancel, tasks/subscribe).

    These run alongside the existing OpenAI Responses contract on
    ``/invocations`` and ``/v1/responses*`` — both protocols share the same
    LangGraph instance via :meth:`AppConfig.as_graph` and the same
    checkpointer, so conversations are consistent across contracts.

    Set ``enabled: false`` to skip mounting the A2A routes entirely.

    Task persistence is configured via :attr:`task_store` — an
    :class:`A2ATaskStoreModel` whose ``database`` field toggles between
    in-memory (default) and Lakebase-backed storage. This is independent
    of :class:`LongRunningModel`; point both at the same
    :class:`DatabaseModel` to share the connection pool.

    Skills and security schemes are derived from the rest of the config
    (one skill per entry in ``config.agents``; bearer scheme with an OBO-
    aware description when :attr:`on_behalf_of_user` is True). Override
    either by setting the field explicitly here. The
    :attr:`security_schemes` field is typed against a2a-sdk's
    :class:`a2a.types.SecurityScheme` discriminated union, so malformed
    schemes fail at config load instead of at request time.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(
        default=True,
        description="When False, A2A routes are NOT mounted on the Databricks Apps FastAPI app.",
    )
    server_url: Optional[str] = Field(
        default=None,
        description="Public base URL advertised on the Agent Card. If unset, derived from "
        "$DATABRICKS_APP_URL at startup; otherwise a relative '/a2a' URL is published.",
    )
    skills: Optional[list[A2ASkillModel]] = Field(
        default=None,
        description="Overrides the Agent Card 'skills' list. When unset, one skill is "
        "derived per entry in config.agents.",
    )
    security_schemes: Optional[dict[str, SecurityScheme]] = Field(
        default=None,
        description="Overrides the Agent Card 'securitySchemes'. Keys are scheme names; "
        "values are validated against a2a-sdk's SecurityScheme discriminated union at "
        "config-load time. See ``dao_ai.apps.a2a.security`` for ready-made constants "
        "(BEARER_DATABRICKS_PAT, BEARER_DATABRICKS_M2M, BEARER_DATABRICKS_OBO) and "
        "factories (oauth2_databricks_authorization_code, oauth2_databricks_obo, "
        "openid_connect_databricks, api_key_header). When unset, derived from "
        ":attr:`on_behalf_of_user` (bearer scheme with OBO-aware description).",
    )
    default_input_modes: list[str] = Field(
        default_factory=lambda: ["text/plain", "application/json"],
        description="Default supported input MIME types on the Agent Card.",
    )
    default_output_modes: list[str] = Field(
        default_factory=lambda: ["text/plain", "application/json"],
        description="Default supported output MIME types on the Agent Card.",
    )
    task_store: A2ATaskStoreModel = Field(
        default_factory=A2ATaskStoreModel,
        description="A2A task persistence configuration. Defaults to an empty "
        "A2ATaskStoreModel (no database → InMemoryTaskStore). Set ``task_store.database`` "
        "to a DatabaseModel to persist tasks in Lakebase. Independent of app.long_running.",
    )
    on_behalf_of_user: Optional[bool] = Field(
        default=None,
        description="Three-state advisory controlling how the Agent Card advertises "
        "On-Behalf-Of-User (OBO) auth. None (default) → auto-derive: True iff any "
        "Databricks resource in the config has ``on_behalf_of_user=True``. True → "
        "force-advertise OBO (Agent Card emits oauth2 + bearer schemes). False → "
        "force-suppress (Agent Card emits a single PAT/M2M bearer scheme). This flag "
        "does NOT toggle OBO on any resource; resource-level OBO is still configured "
        "per resource via the resource's own ``on_behalf_of_user`` field.",
    )
    streaming: bool = Field(
        default=True,
        description="Advertise streaming support (message/stream JSON-RPC) on the Agent "
        "Card capabilities object. dao-ai's a2a-sdk integration supports streaming, so "
        "this defaults True.",
    )
    push_notifications: bool = Field(
        default=False,
        description="Advertise push-notification webhook support on the Agent Card "
        "capabilities object. dao-ai does not currently implement A2A push notifications, "
        "so this defaults False; flip to True only after wiring an external notifier.",
    )
    state_transition_history: Optional[bool] = Field(
        default=None,
        description="Advertise task state-transition history retention on the Agent Card "
        "capabilities object. None (default) auto-derives: True iff the app has a "
        "configured A2A task store backed by a database OR an orchestration checkpointer "
        "that persists task state across requests. Set explicitly to override.",
    )
    provider: Optional[ProviderModel] = Field(
        default=None,
        description="Optional service-provider block shown on the Agent Card. "
        "Recommended for production agents so callers can identify the maintainer.",
    )
    documentation_url: Optional[str] = Field(
        default=None,
        description="Optional URL to public documentation for this agent. Surfaced on "
        "the Agent Card so callers can find usage docs.",
    )
    icon_url: Optional[str] = Field(
        default=None,
        description="Optional URL to a hosted icon image for this agent. Surfaced on "
        "the Agent Card.",
    )


class AppModel(BaseModel):
    """Application-level configuration for deployment, model registration, and orchestration."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        max_length=30,
        min_length=2,
        description="Unique application name used for the serving endpoint and model registration. Must be 2-30 characters.",
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
    registered_model: Optional[RegisteredModelModel] = Field(
        default=None,
        description="Unity Catalog registered model where the agent is logged. Required for model registration and deployment.",
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
    enable_chat_proxy: Optional[bool] = Field(
        default=True,
        description="Whether the MLflow AgentServer enables the chat proxy endpoint for Databricks Apps.",
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
    long_running: Optional[LongRunningModel] = Field(
        default=None,
        description="Opt-in long-running agent configuration. When set, the ResponsesAgent "
        "is wrapped so that requests with background=True or custom_inputs.operation are "
        "persisted in the referenced Lakebase database. In Databricks Apps, strict "
        "Responses API routes (/v1/responses, /v1/responses/{id}, /v1/responses/{id}/cancel) "
        "are additionally exposed. See config/examples/19_long_running_agents/.",
    )
    a2a: A2AModel = Field(
        default_factory=A2AModel,
        description="Google A2A protocol endpoint configuration for Databricks Apps "
        "deployments. Defaults to a fresh A2AModel — enabled with sensible defaults "
        "(skills derived from sub-agents, bearer scheme derived from "
        "a2a.on_behalf_of_user). Set a2a.enabled=false to opt out. Ignored for Model "
        "Serving deployments. See A2AModel for the full schema.",
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
        if self.agents:
            return self
        # Under the deep_agent orchestration the "main agent" IS the
        # orchestration block itself (its model/system_prompt/tools), so
        # ``app.agents`` may be empty — any agents declared here will be
        # treated as implicit sub-agents by create_deep_agent_graph.
        if self.orchestration is not None and self.orchestration.deep_agent is not None:
            return self
        raise ValueError("At least one agent must be specified")

    @model_validator(mode="after")
    def validate_agent_requires(self) -> Self:
        """Validate cross-agent ``requires`` references on every agent.

        Checks:
          * Each name in any agent's ``requires`` references a declared agent.
          * The ``requires`` DAG is acyclic. A cycle is unsatisfiable -- no
            traversal can ever satisfy it -- so we reject at config-build time.

        Self-reference is caught earlier on ``AgentModel`` itself.
        """
        agent_names: set[str] = {a.name for a in self.agents}

        # Unknown-name check.
        for agent in self.agents:
            for required in agent.requires:
                if required not in agent_names:
                    raise ValueError(
                        f"Agent '{agent.name}' has a 'requires' entry "
                        f"'{required}' that does not match any declared agent. "
                        f"Known agents: {sorted(agent_names)}."
                    )

        # Cycle detection over the requires DAG via DFS with a recursion stack.
        graph: dict[str, list[str]] = {a.name: list(a.requires) for a in self.agents}
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {name: WHITE for name in graph}

        def visit(node: str, path: list[str]) -> None:
            color[node] = GRAY
            for nxt in graph.get(node, []):
                if color[nxt] == GRAY:
                    cycle_start = path.index(nxt) if nxt in path else 0
                    cycle = path[cycle_start:] + [nxt]
                    raise ValueError(
                        f"Cycle detected in agent 'requires' DAG: "
                        f"{' -> '.join(cycle)}. A cycle is unsatisfiable -- no "
                        f"path through the swarm can ever satisfy a circular "
                        f"prerequisite. Break the cycle by removing one of the "
                        f"'requires' entries."
                    )
                if color[nxt] == WHITE:
                    visit(nxt, path + [nxt])
            color[node] = BLACK

        for name in graph:
            if color[name] == WHITE:
                visit(name, [name])

        return self

    @model_validator(mode="after")
    def validate_registered_model_required_for_serving(self) -> Self:
        """Ensure registered_model is provided when deployment target is not apps."""
        if (
            self.registered_model is None
            and self.deployment_target != DeploymentTarget.APPS
        ):
            raise ValueError(
                "registered_model is required when deployment_target is not 'apps'. "
                "Either add a registered_model section or set deployment_target to 'apps'."
            )
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
        # deep_agent solo (no app.agents) is allowed — the orchestration
        # block itself acts as the main agent and there's nothing to default.
        if not self.agents:
            if (
                self.orchestration is not None
                and self.orchestration.deep_agent is not None
            ):
                return self
            raise ValueError("At least one agent must be specified")

        if self.orchestration is None:
            self.orchestration = OrchestrationModel()

        # If no orchestration mode is set (e.g. user only supplied
        # `orchestration: { memory: ... }`), pick a sensible default based
        # on agent count. ``deep_agent`` is opt-in only and is never
        # auto-selected; it counts as "set" if present so the auto-fill skips.
        if (
            self.orchestration.supervisor is None
            and self.orchestration.swarm is None
            and self.orchestration.deep_agent is None
        ):
            default_agent: AgentModel = self.agents[0]
            if len(self.agents) > 1:
                self.orchestration.supervisor = SupervisorModel(
                    model=default_agent.model
                )
            else:
                self.orchestration.swarm = SwarmModel(default_agent=default_agent)

        return self

    @model_validator(mode="after")
    def set_default_endpoint_name(self) -> Self:
        if self.endpoint_name is None:
            self.endpoint_name = self.name
        return self

    @model_validator(mode="after")
    def set_default_agent(self) -> Self:
        # Only meaningful when agents are declared. Under deep_agent solo
        # (empty ``app.agents``) the swarm default_agent path is unreachable.
        if not self.agents:
            return self
        default_agent_name: str = self.agents[0].name

        if self.orchestration.swarm and not self.orchestration.swarm.default_agent:
            self.orchestration.swarm.default_agent = default_agent_name

        return self

    @model_validator(mode="after")
    def validate_no_deterministic_handoff_to_constrained(self) -> Self:
        """Reject deterministic handoffs to agents that declare ``requires``.

        Deterministic edges fire unconditionally via static ``add_edge`` in
        the swarm graph. Layering a runtime ``requires`` check on top is a
        contradiction -- the edge would either fire and bypass the check, or
        the check would block an edge that's supposed to be unconditional.

        Catch this at config-build time and direct the user to use an
        agentic handoff for constrained targets instead.
        """
        if not self.orchestration or not self.orchestration.swarm:
            return self
        if not self.orchestration.swarm.handoffs:
            return self

        # Build a name -> requires map so we can resolve target requires by
        # name regardless of how the handoff entry referenced the agent.
        requires_by_name: dict[str, list[str]] = {
            a.name: list(a.requires) for a in self.agents
        }

        for source, targets in self.orchestration.swarm.handoffs.items():
            if not targets:
                continue
            for entry in targets:
                if isinstance(entry, HandoffRouteModel):
                    target_obj = entry.agent
                    is_det = entry.is_deterministic
                else:
                    target_obj = entry
                    is_det = False
                if not is_det:
                    continue
                target_name: str = (
                    target_obj.name if hasattr(target_obj, "name") else str(target_obj)
                )
                target_requires: list[str] = requires_by_name.get(target_name, [])
                if target_requires:
                    raise ValueError(
                        f"Agent '{source}' has a deterministic handoff to "
                        f"'{target_name}', but '{target_name}' declares "
                        f"requires={target_requires}. Deterministic edges "
                        f"fire unconditionally and cannot honor a runtime "
                        f"prerequisite check. Use an agentic handoff "
                        f"(default) for constrained targets."
                    )
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
    model: InferenceEndpointModel = Field(
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
          reflection_model: databricks-gpt-5-4-mini
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
    reflection_model: Optional[InferenceEndpointModel | str] = Field(
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
              judge_model: databricks-gpt-5-4-mini
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
    judge_model: Optional[InferenceEndpointModel | str] = Field(
        default="databricks-gpt-5-4-mini",
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
            judge_model_name = "databricks-gpt-5-4-mini"

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

    # `validation_alias=AliasChoices("models", "llms")` accepts both YAML
    # keys at parse time while keeping `models` as the canonical Python
    # field name (and therefore the canonical JSON-schema property). This
    # is the right tool for a *rename with back-compat* — `Field(alias=...)`
    # would have flipped the schema's canonical name to `llms`, defeating
    # the rename for IDE / linting purposes.
    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
    )
    models: dict[str, InferenceEndpointModel] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("models", "llms"),
        description=(
            "Databricks Model Serving endpoint configurations keyed by name. "
            "Holds chat LLMs, embedding models, judge/extraction/reflection/query "
            "models, and custom agent endpoints — anything reachable via "
            "/serving-endpoints/<name>/invocations. Renamed from `llms` in dao-ai 0.1.75; "
            "the legacy key is still accepted via validation alias and will be removed in a future major release."
        ),
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
    skills: dict[str, SkillModel] = Field(
        default_factory=dict,
        description=(
            "Reusable deepagents skills keyed by name. Referenced from "
            "``orchestration.deep_agent.skills`` and ``subagents[].skills``. "
            "Local skills ship via ``code_paths``; volume-backed skills are wired as deployment resources."
        ),
    )

    @property
    def llms(self) -> dict[str, InferenceEndpointModel]:
        """Deprecated alias for :attr:`models`.

        Old customer code accessing ``config.resources.llms`` keeps working
        for now. Use :attr:`models` for new code. The alias will be removed
        in a future major release.
        """
        import warnings

        warnings.warn(
            "ResourcesModel.llms is a deprecated alias for ResourcesModel.models; "
            "use `.models` instead. The alias will be removed in a future major release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.models

    @model_validator(mode="after")
    def update_genie_warehouses(self) -> Self:
        """
        Automatically populate warehouses from genie_rooms.

        Warehouses are extracted from each Genie room and added to the
        resources if they don't already exist (based on warehouse_id).
        """
        if not self.genie_rooms:
            return self

        # Process warehouses from all genie rooms. Prefer the explicitly
        # configured warehouse; fall back to discovery for existing-space
        # references that don't declare one inline.
        for genie_room in self.genie_rooms.values():
            genie_room: GenieRoomModel
            warehouse: Optional[WarehouseModel] = (
                genie_room.warehouse or genie_room.discover_warehouse()
            )

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
    parameters: dict[str, ParameterDeclarationModel] = Field(
        default_factory=dict,
        description=(
            "Declared input parameters for load-time substitution. Reference "
            "with ${param.NAME} or ${var.NAME} (interchangeable aliases) "
            "anywhere in any string value. Resolved by AppConfig.from_file "
            "from CLI --var, process env, declared default, or inline "
            "${var.NAME:-fallback} - in that order."
        ),
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

    # Private attributes set by from_file
    _source_config_path: str | None = None
    _rendered_yaml: str | None = None
    _substitution_vars: dict[str, str] | None = None
    _raw_yaml_dict: dict[str, Any] | None = None
    _initialized: bool = False

    @model_validator(mode="after")
    def _translate_agent_skills_to_middleware(self) -> Self:
        """Convert each agent's ``skills`` list into ``MiddlewareModel`` entries on ``middleware``.

        Runs once at AppConfig load time. For each agent in ``self.agents``
        (the dict registry — also covers ``self.app.agents`` via YAML anchor
        identity), each entry in ``agent.skills`` is rewritten:

        * Inline ``SkillModel`` → ``SkillModel.as_middleware()`` (returns a
          ``MiddlewareModel`` calling ``dao_ai.middleware.skills.create_skills_middleware``)
        * String → look up in ``self.resources.skills`` then call ``as_middleware()``

        After translation ``agent.skills`` is cleared so re-running the
        validator (e.g. if the model is rebuilt) is idempotent. This is also
        why the runtime never needs to know about ``agent.skills`` — by the
        time ``create_agent_node`` runs, skills are middleware.
        """
        agents_seen: set[int] = set()

        def _translate_one(agent: AgentModel) -> None:
            if id(agent) in agents_seen:
                return
            agents_seen.add(id(agent))
            if not agent.skills:
                return
            for entry in agent.skills:
                if isinstance(entry, str):
                    if not self.resources or entry not in self.resources.skills:
                        raise ValueError(
                            f"Agent '{agent.name}' references unknown skill '{entry}'. "
                            f"Add a SkillModel under resources.skills or use an inline anchor."
                        )
                    skill_model = self.resources.skills[entry]
                else:
                    skill_model = entry
                agent.middleware.append(skill_model.as_middleware())
            agent.skills = []

        for agent in (self.agents or {}).values():
            _translate_one(agent)
        if self.app and self.app.agents:
            for agent in self.app.agents:
                _translate_one(agent)

        return self

    @classmethod
    def from_file(
        cls,
        path: PathLike,
        *,
        params: Optional[Mapping[str, str]] = None,
        task_values: Optional[TaskValuesLike] = None,
        task_key: Optional[str] = None,
        initialize: bool = True,
    ) -> "AppConfig":
        """Load an AppConfig from a YAML file with optional parameter substitution.

        Top-level ``parameters:`` declarations are parsed first and used to
        resolve ``${param.NAME}`` and ``${var.NAME}`` references in the rest
        of the YAML (the two prefixes are interchangeable aliases).
        Resolution precedence per reference is CLI ``params`` > ``task_values``
        > process env > declared ``default`` > inline ``${var.NAME:-fallback}``
        > error.

        Args:
            path: Path to the YAML config file.
            params: Optional mapping of parameter name to literal string value,
                used to override env-var and default lookups for
                ``${param.NAME}`` / ``${var.NAME}`` references.
            task_values: Optional :class:`TaskValuesLike` (e.g.
                ``dbutils.jobs.taskValues``). When provided, every declared
                parameter is probed via ``task_values.get(taskKey=task_key,
                key=NAME, ...)`` and any non-empty result is folded into the
                substitution map. ``params`` overrides ``task_values`` per
                key. Requires ``task_key``.
            task_key: Upstream task key whose taskValues should be probed.
                Required when ``task_values`` is given.
            initialize: Whether to call :meth:`initialize` after loading.

        Raises:
            ValueError: If ``task_values`` is given without ``task_key``.
            ConfigVariableError: If any required parameter cannot be resolved
                or any reference is undeclared (when a ``parameters:`` block
                is present).
        """
        path = Path(path).as_posix()
        logger.debug(f"Loading config from {path}")

        raw_text: str = Path(path).read_text()

        # Resolve ${workspace.*} refs first so they can appear inside
        # parameter defaults (e.g. default: /Users/${workspace.current_user.userName}/...).
        workspace_resolved_text: str = substitute_workspace_refs(raw_text, source=path)

        raw_dict: dict[str, Any] = yaml.safe_load(workspace_resolved_text) or {}
        decl_block: dict[str, Any] = raw_dict.get("parameters", {}) or {}
        declarations: dict[str, ParameterDeclarationModel] = {
            name: ParameterDeclarationModel(**(spec or {}))
            for name, spec in decl_block.items()
        }

        task_value_params: dict[str, str] = {}
        if task_values is not None:
            if not task_key:
                raise ValueError(
                    "AppConfig.from_file: task_values requires task_key "
                    "(the upstream task whose taskValues should be probed)."
                )
            for name in declarations:
                try:
                    val: Any = task_values.get(
                        taskKey=task_key, key=name, default="", debugValue=""
                    )
                except Exception as e:
                    logger.debug(
                        f"task_values.get raised for {name!r} on taskKey={task_key!r}: {e}"
                    )
                    continue
                if val:
                    task_value_params[name] = str(val)

        # CLI params win over taskValues (CLI is the most explicit source).
        merged_params: dict[str, str] = {**task_value_params, **(params or {})}

        rendered_text: str = substitute_params(
            workspace_resolved_text,
            declarations=declarations,
            cli_vars=merged_params or None,
            source=path,
        )
        rendered_dict: dict[str, Any] = yaml.safe_load(rendered_text) or {}

        model_config: ModelConfig = ModelConfig(development_config=rendered_dict)
        config: AppConfig = AppConfig(**model_config.to_dict())

        config._source_config_path = path
        config._rendered_yaml = rendered_text
        config._substitution_vars = dict(merged_params) if merged_params else None
        # Stash the pre-substitution dict so tooling can recover which
        # YAML fields were backed by ``${var.X}`` references.
        config._raw_yaml_dict = raw_dict

        # Wire pre-substitution values onto specific models that need to
        # introspect their own parameter bindings at runtime. Genie rooms
        # carry the raw space_id so provisioning tasks can detect a
        # ``${var.X}`` binding via ``is_parameter(room.raw_space_id)``.
        if config.resources is not None and config.resources.genie_rooms:
            raw_rooms: dict[str, Any] = (
                raw_dict.get("resources", {}).get("genie_rooms", {}) or {}
            )
            for room_key, room in config.resources.genie_rooms.items():
                raw_room: dict[str, Any] = raw_rooms.get(room_key) or {}
                room._raw_space_id = raw_room.get("space_id")

        if initialize:
            config.initialize()

        return config

    @property
    def source_config_path(self) -> str | None:
        """Get the source config file path if loaded via from_file."""
        return self._source_config_path

    @property
    def rendered_yaml(self) -> str | None:
        """Get the YAML text after ${param.NAME} substitution, if loaded via from_file."""
        return self._rendered_yaml

    @property
    def substitution_vars(self) -> dict[str, str] | None:
        """Get the explicit substitution vars used at load time, if any."""
        return self._substitution_vars

    def _resolve_all_resources(self) -> None:
        """Walk the config tree and call ensure_resolved() on all IsDatabricksResource instances."""

        def _walk(obj: Any) -> None:
            if isinstance(obj, IsDatabricksResource):
                obj.ensure_resolved()
            if isinstance(obj, BaseModel):
                for field_name in obj.model_fields:
                    value = getattr(obj, field_name, None)
                    if value is None:
                        continue
                    if isinstance(value, list):
                        for item in value:
                            _walk(item)
                    elif isinstance(value, dict):
                        for item in value.values():
                            _walk(item)
                    else:
                        _walk(value)

        _walk(self)

    def initialize(self) -> None:
        if self._initialized:
            return

        from dao_ai.hooks.core import create_hooks
        from dao_ai.logging import configure_logging

        if self.app and self.app.log_level:
            configure_logging(level=self.app.log_level)

        self._resolve_all_resources()

        logger.debug("Calling initialization hooks...")
        initialization_functions: Sequence[Callable[..., Any]] = create_hooks(
            self.app.initialization_hooks
        )
        for initialization_function in initialization_functions:
            logger.debug(
                f"Running initialization hook: {initialization_function.__name__}"
            )
            initialization_function(self)

        self._initialized = True
        atexit.register(self.shutdown)

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

        long_running = self.app.long_running if self.app else None
        if long_running is not None:
            from dao_ai.long_running import (
                LongRunningResponsesAgent,
                LongRunningStore,
            )

            store = LongRunningStore(
                database=long_running.database,
                responses_table_name=long_running.responses_table_name,
                messages_table_name=long_running.messages_table_name,
            )
            app = LongRunningResponsesAgent(
                inner=app,
                store=store,
                max_duration_seconds=long_running.max_duration_seconds,
                poll_interval_seconds=long_running.poll_interval_seconds,
                default_background=long_running.default_background,
            )

        return app
