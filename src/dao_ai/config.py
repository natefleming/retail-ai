import atexit
import hashlib
import importlib
import os
import re
import sys
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from enum import Enum
from functools import cache
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    AsyncIterator,
    Callable,
    Final,
    Iterable,
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
    from dao_ai.audit import LakebaseAuditSink
    from dao_ai.genie.cache.context_aware.optimization import (
        ContextAwareCacheEvalDataset,
        ThresholdOptimizationResult,
    )
    from dao_ai.state import Context

from databricks.ai_search.client import VectorSearchClient
from databricks.ai_search.index import VectorSearchIndex
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
    Discriminator,
    Field,
    PrivateAttr,
    Tag,
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
from dao_ai.sources import ConfigSource, ResolvedConfig, SourceLike
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
        # All declared sources (secret scopes, env vars, ...) resolved to None,
        # so we fall back to default_value. This is a common silent-misconfig
        # trap: e.g. a `--direct` deploy where the container's env/scope isn't
        # populated falls back to a stale hardcoded default (a dead genie
        # space_id). Warn when a non-None default is actually used so the
        # fallback is visible instead of failing mysteriously downstream.
        if self.default_value is not None:
            logger.warning(
                "Composite variable resolved to its default_value — all "
                "declared sources (scope/env) returned None. Using default; "
                "verify the intended source is populated in this environment.",
                default_value=self.default_value,
                option_kinds=[type(v).__name__ for v in self.options],
            )
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


APP_RESOURCE_DESCRIPTION_MAX_LENGTH: Final[int] = 200
"""Max length for descriptions on IsDatabricksResource subclasses. Matches the
Databricks Apps platform limit on ``AppResource.description``; overlong values
are rejected at deploy time, so we reject at config-load time instead."""


def clip_resource_description(description: str) -> str:
    """Clip a *discovered* description to :data:`APP_RESOURCE_DESCRIPTION_MAX_LENGTH`.

    The limit exists to reject an overlong description the operator *wrote*. A
    description read back off a live workspace object is not theirs to shorten,
    and it has to survive a round trip: both deploy paths write the resolved
    value back into a config the container re-validates — Model Serving through
    the logged ``model_config``, Apps through the baked YAML — so an unclipped
    200+ char Genie space description would pass at deploy time and then fail
    the *container's* ``max_length`` at startup with ``string_too_long``.

    Clip rather than drop: the description feeds the Genie tool description a
    supervisor routes on, so a truncated one is still useful signal. The ellipsis
    marks it as truncated.
    """
    if len(description) <= APP_RESOURCE_DESCRIPTION_MAX_LENGTH:
        return description
    return description[: APP_RESOURCE_DESCRIPTION_MAX_LENGTH - 1].rstrip() + "…"


class ServicePrincipalModel(BaseModel):
    """Databricks service principal credentials for OAuth M2M authentication."""

    model_config = ConfigDict(
        frozen=True,
        use_enum_values=True,
    )
    name: Optional[str] = Field(
        default=None,
        description=(
            "Workspace display name of the service principal. Used by "
            "`dao-ai service-principal provision` to create or reuse the SP; "
            "defaults to '<app.name>-<key>' where <key> is this entry's name "
            "under `service_principals`. Set it explicitly to bind a config to "
            "an SP that already exists."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "What this service principal is for — which resources it owns and "
            "why it is separate from the others. Documentation only: the "
            "Databricks service-principal API has no description field, so this "
            "is never sent to the workspace."
        ),
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

        # Operator-set client_id / client_secret that failed to resolve is the
        # silent-misconfiguration footgun that bit us during the FEVM testing
        # for the MCP server (PR #105/#106): the YAML references a secret-scope
        # path, the App SP doesn't have READ on the scope, value_of() returns
        # None, we fall through to ambient (the App's auto-SP) instead of the
        # stable SP the operator intended. Surface this loudly.
        if self.client_id is not None and client_id_value is None:
            logger.warning(
                "dao_ai.auth.client_id.unresolved",
                resource=self.__class__.__name__,
                client_id_spec=str(self.client_id)[:300],
                note=(
                    "client_id is configured but value_of() returned None — "
                    "falling back to PAT or ambient auth. On Databricks Apps "
                    "this means the App's auto-injected SP, not the stable SP "
                    "you intended. Likely causes: (a) the App SP doesn't have "
                    "READ on the referenced secret scope, (b) the env-var "
                    "fallback isn't set, or (c) the scope/key was renamed. "
                    "For DatabaseModel + Lakebase, this is the most common "
                    "cause of 'permission denied for sequence' errors on the "
                    "Postgres semantic cache. See docs/mcp_server.md."
                ),
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

    def workspace_client_from(
        self,
        context: "Context | None",
        *,
        strict: bool = False,
    ) -> WorkspaceClient:
        """
        Get a WorkspaceClient using headers from the provided Context.

        Use this method from tools that have access to ToolRuntime[Context].
        This allows OBO authentication to work in Databricks Apps where headers
        are captured at request entry and passed through the Context.

        Args:
            context: Runtime context containing headers for OBO auth.
                     If None or no headers, falls back to workspace_client property.
            strict: If True and ``self.on_behalf_of_user`` is set but no
                    forwarded user token is available in the context, raise
                    :class:`dao_ai.auth.OBONotAvailableError` instead of
                    silently falling back to the service-principal identity.
                    Dispatcher call sites that invoke a target on a user's
                    behalf should pass ``strict=True`` so misconfigured OBO
                    surfaces at first-call rather than running as the SP.

                    Note on deploy target semantics: inside Databricks Apps,
                    the proxy unconditionally forwards
                    ``x-forwarded-access-token`` on every authenticated user
                    request, so ``strict=True`` rarely fires there — the
                    token is present even when other agents in the call
                    chain didn't ask for OBO. Strict-mode is most useful in
                    Model Serving deploys (no Apps proxy → genuine absence
                    of the forwarded header) and in M2M-only contexts.

        Returns:
            WorkspaceClient configured with appropriate authentication.

        Raises:
            OBONotAvailableError: when ``strict=True`` and OBO was requested
                but no forwarded user token is in the call context.
        """
        from dao_ai.auth import OBONotAvailableError
        from dao_ai.utils import normalize_host

        logger.trace(
            "workspace_client_from called",
            context=context,
            on_behalf_of_user=self.on_behalf_of_user,
            strict=strict,
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

        # OBO was requested but no forwarded token reached us. Either the
        # calling agent isn't running with OBO, or its middleware didn't
        # propagate the headers. In strict mode this is a misconfiguration
        # we should fail fast on; in lenient mode (current default) we fall
        # through to the SP-identity workspace_client.
        if strict and self.on_behalf_of_user:
            raise OBONotAvailableError(
                resource_name=getattr(self, "name", None),
                field=f"{self.__class__.__name__}.on_behalf_of_user",
            )

        # Fall back to existing workspace_client property
        return self.workspace_client


class ServingMode(str, Enum):
    """Hosting PLATFORM for agent deployment (a deploy-action parameter — NOT a
    field on AppConfig).

    This is the ``--mode`` axis: *where* the agent runs. The wire protocol is a
    separate axis — see the ``as_mcp`` deploy parameter (CLI ``--as-mcp``), which
    serves the agent over MCP on the Apps platform instead of the chat UI. MCP is
    deliberately NOT a member here: it is not a third platform, it is the same
    Databricks Apps runtime speaking a different protocol.
    """

    MODEL_SERVING = "model_serving"
    """Deploy to a Databricks Model Serving endpoint (no DAB bundle)."""

    APPS = "apps"
    """Deploy as a Databricks App from a DAB bundle — chat UI by default, or the
    MCP server when the deploy sets ``as_mcp``."""


#: Prefix applied to the deployed App name when serving over MCP. The chat App
#: and the MCP server are both Databricks Apps built from the SAME config, so
#: without a distinct name one would silently replace the other. The ``mcp-``
#: form is also what Databricks Multi-Agent Supervisor pattern-matches on when
#: auto-discovering MCP-hosted Apps across an account.
MCP_APP_PREFIX: str = "mcp-"


def app_name_for(name: str, *, as_mcp: bool = False) -> str:
    """Normalize a config ``app.name`` to its deployed Databricks App name.

    Lowercases and hyphenates (the form every Apps API call must use), then
    applies :data:`MCP_APP_PREFIX` when the deployment serves MCP. Single source
    of truth for the deployed name — the bundle writers, the SDK deploy path, log
    retrieval, the readiness pollers, and teardown all resolve through here.

    Prefixing is **idempotent**: a config that already names its app ``mcp-*``
    (the convention dao-ai's own docs recommend, since Multi-Agent Supervisor
    pattern-matches it) is returned unchanged rather than becoming
    ``mcp-mcp-*``.

    Deliberately a function over a raw name rather than a method on
    :class:`AppModel`, so callers holding only a name (and the duck-typed app
    objects used in tests) can resolve it without constructing a full model.
    """
    normalized: str = name.lower().replace("_", "-")
    if not as_mcp or normalized.startswith(MCP_APP_PREFIX):
        return normalized
    return f"{MCP_APP_PREFIX}{normalized}"


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


class ChatUnityAIGateway(ChatDatabricks):
    """``ChatDatabricks`` subclass routed through the Databricks AI Gateway.

    Functionally identical to ``ChatDatabricks(use_ai_gateway=True)`` — exists
    so that MLflow trace spans surface a distinct class name (LangChain's
    MLflow autolog labels spans by ``self.__class__.__name__``) and so
    ``_llm_type`` reports a distinct identifier, making AI-Gateway-routed
    calls visually distinguishable from plain ``ChatDatabricks`` calls in
    the trace UI.

    The historical ``name``-field strip workaround is no longer needed:
    ``ChatDatabricks._convert_message_to_dict`` already drops the ``name``
    field at the request-payload boundary (databricks-langchain >= 0.20).
    """

    use_ai_gateway: bool = True

    @property
    def _llm_type(self) -> str:
        return "chat-unity-ai-gateway"


class InferenceEndpointModel(IsDatabricksResource, HasFullName):
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

    # ``populate_by_name`` so the attribute name ``schema_model`` is accepted on
    # input alongside the config-facing alias ``schema``. Without it, a plain
    # ``model_dump()`` (no ``by_alias=True``) emits ``schema_model`` and the
    # result no longer re-validates against ``extra="forbid"`` — breaking any
    # dump/reload round-trip on a config containing a model. The deploy bake
    # dumps with ``by_alias=True`` and so already round-trips, but nothing
    # forces every caller to.
    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", populate_by_name=True
    )
    schema_model: Optional[SchemaModel] = Field(
        default=None,
        alias="schema",
        description=(
            "Schema qualifying a UC-securable model name, for models addressed "
            "through the Unity AI Gateway as UC securables (e.g. catalog "
            "`system`, schema `ai`). When set, `name` is the short model name "
            "and `full_name` resolves to `<catalog>.<schema>.<name>`. Omit it "
            "to pass a serving endpoint name — or an already-qualified model "
            "name — in `name` directly. Requires `use_ai_gateway: true`: a "
            "three-level name is only addressable on the gateway, and the "
            "`/serving-endpoints/<name>/invocations` path answers 404 for one."
        ),
    )
    name: str = Field(
        description=(
            "Serving endpoint name (e.g. 'databricks-gpt-5-4-mini'), a "
            "UC-securable model name (e.g. 'system.ai.claude-sonnet-4-5'), or "
            "the short model name when `schema` is set. A UC-securable name "
            "requires `use_ai_gateway: true` however it is spelled — the "
            "`/serving-endpoints/<name>/invocations` path answers 404 for a "
            "three-level name."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        max_length=APP_RESOURCE_DESCRIPTION_MAX_LENGTH,
        description="Human-readable description of this model configuration.",
    )
    temperature: Optional[float] = Field(
        default=None,
        description=(
            "Sampling temperature. When unset, dao-ai omits `temperature` "
            "from the outbound payload entirely, so the serving endpoint "
            "uses its own default. This is required by reasoning-mode "
            "endpoints (e.g. Anthropic Sonnet 5) that reject the parameter "
            "outright. Set explicitly (e.g. 0.1 for deterministic, 1.0 for "
            "creative) to override the endpoint default."
        ),
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description=(
            "Maximum tokens in the model response. When unset, dao-ai "
            "omits `max_tokens` from the outbound payload so the serving "
            "endpoint uses its own default. Set explicitly to cap output "
            "length. Ignored by the Responses API (endpoint enforces its "
            "own limit)."
        ),
    )
    extra_params: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Extra request parameters forwarded verbatim to the serving "
            "endpoint via the chat client's ``extra_params`` (e.g. "
            "``{reasoning_effort: low}`` on gpt-oss reasoning models to cap the "
            "reasoning preamble and cut latency, or other endpoint-specific "
            "knobs). Passed on both the standard and AI-Gateway paths."
        ),
    )
    fallbacks: Optional[list[Union[str, "InferenceEndpointModel"]]] = Field(
        default_factory=list,
        description="Ordered list of fallback endpoint names or InferenceEndpointModel configs tried on primary failure.",
    )
    use_responses_api: Optional[bool] = Field(
        default=False,
        description=(
            "Use the Responses API instead of chat completions. Composes with "
            "`use_ai_gateway`: /ai-gateway/mlflow/v1/responses answers 200 for "
            "every model tested. Caveat, per-model and client-side: OpenAI-"
            "family models (gpt-5-4, gpt-5-4-mini, gpt-5-mini) return "
            "input_tokens_details/output_tokens_details and work end to end, "
            "while gpt-oss-120b and claude-sonnet-4-5 omit them and langchain "
            "then fails on usage_metadata — use /chat/completions for those. "
            "Second caveat, server-side and model-independent: the gateway's "
            "/responses translation layer cannot parse a `function_call` "
            "content item, so any turn that makes a tool call fails with "
            "INVALID_PARAMETER_VALUE. That includes every supervisor/swarm "
            "handoff, so pair `use_ai_gateway` with `use_responses_api` only "
            "for agents that call no tools. "
            "To reach a custom ResponsesAgent serving endpoint, leave "
            "`use_ai_gateway` false: the gateway serves only Foundation Model "
            "and UC-securable models and 404s on a custom endpoint."
        ),
    )
    disable_streaming: bool = Field(
        default=False,
        description="Disable streaming for this model. Required when the Foundation Model endpoint has output guardrails enabled.",
    )
    use_ai_gateway: bool = Field(
        default=False,
        validation_alias=AliasChoices("use_ai_gateway", "ai_gateway"),
        description=(
            "Route through the Databricks AI Gateway (/ai-gateway/mlflow/v1) "
            "instead of /serving-endpoints/<name>/invocations. When True, "
            "`name` is sent as the OpenAI-style model id in the request body. "
            "Serves both /chat/completions and /responses, so this composes "
            "with `use_responses_api` — see that field for the per-model "
            "caveat. Addresses Foundation Model and UC-securable models only, "
            "never a custom serving endpoint. Not for embeddings or other "
            "non-chat endpoints. Renamed from `ai_gateway` to match the "
            "databricks-langchain kwarg it feeds; the legacy key is still "
            "accepted and will be removed in a future major release."
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

    @model_validator(mode="after")
    def validate_schema_qualification(self) -> Self:
        """Reject the two combinations that cannot work at runtime.

        Both fail as an opaque 404 on first invocation otherwise, so they are
        caught here where the message can name the offending key.

        The second check keys on the resolved ``full_name`` rather than on
        ``schema``, because the two spellings are equivalent: ``schema`` plus a
        short name and a dotted ``name`` produce the same identifier, so
        checking only the former let the latter through to the 404 it exists to
        prevent.
        """
        if self.schema_model is not None and "." in self.name:
            raise ValueError(
                f"Model '{self.name}' is already fully qualified, so 'schema' "
                f"(catalog '{self.schema_model.catalog_name}', schema "
                f"'{self.schema_model.schema_name}') would produce "
                f"'{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}'. "
                "Provide either a fully qualified 'name' or 'schema' plus the "
                "short model name — not both."
            )

        if self.is_uc_securable and not self.use_ai_gateway:
            raise ValueError(
                f"Model '{self.full_name}' is a UC-securable model name, which "
                "is only addressable through the Unity AI Gateway — the "
                "/serving-endpoints/<name>/invocations path answers 404 for a "
                "three-level name. Set 'use_ai_gateway: true', or address a "
                "serving endpoint instead (e.g. "
                "'databricks-claude-sonnet-4-5')."
            )

        return self

    @property
    def api_scopes(self) -> Sequence[str]:
        return [
            "serving.serving-endpoints",
        ]

    @property
    def full_name(self) -> str:
        """The model identifier dao-ai sends to the serving layer.

        With ``schema`` set, the UC-securable three-level name. Without it,
        ``name`` verbatim — which is what every existing config resolves to,
        whether that is a serving endpoint name (``databricks-claude-sonnet-4-5``)
        or an already-qualified model name (``system.ai.claude-sonnet-4-5``).
        """
        if self.schema_model:
            return f"{self.schema_model.catalog_name}.{self.schema_model.schema_name}.{self.name}"
        return self.name

    @property
    def uri(self) -> str:
        return f"databricks:/{self.full_name}"

    @property
    def is_uc_securable(self) -> bool:
        """Whether this model is addressed as a UC securable, not an endpoint.

        Both deploy targets need this distinction and neither can resolve a
        three-level name against ``/api/2.0/serving-endpoints``, so the rule
        lives here rather than being spelled out at each site.
        """
        return "." in self.full_name

    def as_resources(self) -> Sequence[DatabricksResource]:
        # A UC-securable model name is not a serving endpoint, and MLflow has no
        # resource type for one — only DatabricksServingEndpoint, whose
        # endpoint_name the platform resolves against
        # /api/2.0/serving-endpoints. Verified on a live workspace: neither
        # ``system.ai.claude-sonnet-4-5`` nor the short ``claude-sonnet-4-5``
        # resolves there, so either spelling would put an unresolvable name into
        # the deploy manifest and auth policy. Emit nothing instead; access to a
        # UC-securable model is governed by UC grants on the model, and the
        # gateway's OBO scope is emitted separately (apps/resources.py).
        if self.is_uc_securable:
            return []
        return [
            DatabricksServingEndpoint(
                endpoint_name=self.name, on_behalf_of_user=self.on_behalf_of_user
            )
        ]

    def chat_model_for_workspace_client(
        self,
        workspace_client: WorkspaceClient,
        *,
        disable_streaming: bool | None = None,
    ) -> LanguageModelLike:
        """Build a chat client bound to a specific ``WorkspaceClient``.

        Used by OBO call sites that need to swap in a user-scoped
        ``WorkspaceClient`` per request. Respects ``self.use_ai_gateway`` so
        OBO traffic still routes through the AI Gateway path when enabled.

        NOTE: ``on_behalf_of_user`` + ``use_ai_gateway`` is permitted. If a
        workspace ever returns 401/403 on the AI Gateway with an OBO token,
        gate the combination in a validator on this model.
        """
        effective_disable_streaming: bool = (
            self.disable_streaming if disable_streaming is None else disable_streaming
        )

        cls = ChatUnityAIGateway if self.use_ai_gateway else ChatDatabricks
        return cls(
            model=self.full_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            use_responses_api=self.use_responses_api,
            disable_streaming=effective_disable_streaming,
            workspace_client=workspace_client,
            **({"extra_params": self.extra_params} if self.extra_params else {}),
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

        cls = ChatUnityAIGateway if self.use_ai_gateway else ChatDatabricks
        chat_client: LanguageModelLike = cls(
            model=self.full_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            use_responses_api=self.use_responses_api,
            disable_streaming=effective_disable_streaming,
            **({"extra_params": self.extra_params} if self.extra_params else {}),
        )

        fallbacks: Sequence[LanguageModelLike] = []
        for fallback in self.fallbacks:
            fallback: str | InferenceEndpointModel
            if isinstance(fallback, str):
                fallback = InferenceEndpointModel(
                    name=fallback,
                    # A bare string carries no routing of its own, so it inherits
                    # the primary's: without this a UC-securable fallback name is
                    # built with the default False and rejected by
                    # validate_schema_qualification, and any fallback silently
                    # drops off the gateway path the primary is on.
                    use_ai_gateway=self.use_ai_gateway,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_params=self.extra_params,
                )
            if fallback.full_name == self.full_name:
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
                # Same as fallbacks: a bare judge string inherits the primary's
                # routing so a UC-securable name builds and reaches the gateway.
                judge_cfg = InferenceEndpointModel(
                    name=judge_cfg, use_ai_gateway=self.use_ai_gateway
                )
            judge_chat_model = judge_cfg.as_chat_model()

            chat_client = BestOfNChatModel.from_components(
                generator=chat_client,
                judge=judge_chat_model,
                n=self.best_of_n.n,
                generator_temperature=self.temperature,
                temperature_override=self.best_of_n.temperature_override,
            )

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


class AiSearchEndpointType(str, Enum):
    """AI Search endpoint compute profile.

    (Formerly ``VectorSearchEndpointType`` — Databricks rebranded Vector
    Search to AI Search. The old name remains as an alias for backwards
    compatibility; it will eventually be deprecated.)
    """

    STANDARD = "STANDARD"
    OPTIMIZED_STORAGE = "OPTIMIZED_STORAGE"


class AiSearchEndpoint(BaseModel):
    """AI Search endpoint that hosts one or more indexes.

    (Formerly ``VectorSearchEndpoint``. See :class:`AiSearchEndpointType`.)
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="AI Search endpoint name in the workspace.",
    )
    type: AiSearchEndpointType = Field(
        default=AiSearchEndpointType.STANDARD,
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
    def serialize_type(self, value: AiSearchEndpointType) -> str:
        """Ensure enum is serialized to string value."""
        if isinstance(value, AiSearchEndpointType):
            return value.value
        return str(value)

    @model_validator(mode="after")
    def validate_target_qps_only_on_standard(self) -> Self:
        """Reject target_qps on non-STANDARD endpoints (SDK constraint)."""
        if self.target_qps is not None and self.type != AiSearchEndpointType.STANDARD:
            raise ValueError(
                f"target_qps is only supported on STANDARD endpoints, not {self.type!r}"
            )
        return self


# Backwards-compatible aliases — Vector Search naming will eventually be
# deprecated. Both names refer to the same class.
VectorSearchEndpointType = AiSearchEndpointType
VectorSearchEndpoint = AiSearchEndpoint


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
        max_length=APP_RESOURCE_DESCRIPTION_MAX_LENGTH,
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


_RELATIONSHIP_MARKER_PREFIX = "FROM_RELATIONSHIP_TYPE_"
_RELATIONSHIP_MARKER_RE = re.compile(r"--rt=([A-Z_]+)--")


def _relationship_marker(relationship_type: Any) -> str:
    """Encode a join's cardinality the way Genie's export proto expects.

    ``instructions.join_specs[].sql`` must carry exactly two elements: the
    join condition, then this marker. Posting the condition on its own is
    rejected with ``Failed to parse export proto: <condition> (of class
    java.lang.String)``, so an undeclared cardinality still has to say so.
    """
    value = getattr(relationship_type, "value", relationship_type) or "UNSPECIFIED"
    return f"--rt={_RELATIONSHIP_MARKER_PREFIX}{value}--"


def _relationship_type_from_marker(marker: str) -> "GenieRelationshipType | None":
    """Decode a ``--rt=…--`` marker, or ``None`` if it carries no cardinality."""
    match = _RELATIONSHIP_MARKER_RE.search(marker or "")
    if not match:
        return None
    try:
        return GenieRelationshipType(
            match.group(1).removeprefix(_RELATIONSHIP_MARKER_PREFIX)
        )
    except ValueError:
        return None


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


# Sort priority for ``GenieRoomModel._sort_payload_lists`` — ordered from
# most-specific to least-specific. Entries that carry multiple of these keys
# are sorted by ALL of them as a composite tuple (in priority order),
# matching the Genie v2 export-proto validator's expectations. In particular,
# ``instructions.sql_functions`` entries carry both ``id`` and ``identifier``
# and the validator demands sort by ``(id, identifier)`` (not by either
# field alone). Module-level because Pydantic treats leading-underscore
# class attributes as private model attrs.
_GENIE_SORT_KEY_PRIORITY: tuple[str, ...] = (
    "id",
    "identifier",
    "column_name",
    "display_name",
)


GENIE_AGENT_DEFAULT_TIMEOUT_SECONDS: int = 300
"""Default httpx client timeout for the Genie Agent Mode streaming call.

The Databricks server-side response timeout is 90 minutes; this bounds the
client only. Shared by :meth:`GenieRoomModel.as_chat_model` and the
:attr:`GenieAgentModel.timeout_seconds` field default so the two cannot drift.
"""


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
        max_length=APP_RESOURCE_DESCRIPTION_MAX_LENGTH,
        description="Description of the Genie room. Auto-populated from the space if omitted.",
    )
    space_id: Optional[AnyVariable] = Field(
        default=None,
        validation_alias=AliasChoices("space_id", "agent_id"),
        description=(
            "Databricks-assigned Genie space identifier. The only field "
            "guaranteed unique by the platform; titles are not enforced unique. "
            "Also accepted under the alias ``agent_id`` — Databricks renamed "
            "Genie Spaces to Genie Agents, and the Agent Mode API refers to "
            "the same 32-char hex value as ``agent_id``. Both YAML keys "
            "resolve to this attribute. "
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
                # The serialized payload requires CAN_EDIT, while ``title`` and
                # ``description`` need only CAN_RUN — the level a deployed
                # identity typically holds. Retry without it so the tool
                # description still gets the space's own text; only the sample
                # questions are lost, and ``_parse_serialized_space`` already
                # treats an absent payload as "no questions".
                try:
                    self._space_details = self.workspace_client.genie.get_space(
                        space_id=self.space_id
                    )
                except Exception as retry_error:
                    logger.debug(
                        "Could not fetch Genie space details without the "
                        "serialized payload either",
                        space_id=self.space_id,
                        error=str(retry_error),
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
        # Populate name, description and sample questions from space details if
        # missing. All three feed the Genie *tool* description, which is how a
        # supervisor tells two Genie tools apart — so back-filling them lets a
        # bare-``space_id`` room still advertise itself usefully. This also runs
        # at deploy time, which bakes the values into the logged model_config so
        # Model Serving needs no Genie call at model load.
        if self.space_id and (
            not self.name or not self.description or not self.sample_questions
        ):
            try:
                space_details = self._get_space_details()
                if space_details:
                    if not self.name and space_details.title:
                        self.name = space_details.title
                    if not self.description and space_details.description:
                        self.description = clip_resource_description(
                            space_details.description
                        )
                if not self.sample_questions:
                    # Reads the same cached _space_details, so no extra API call.
                    discovered: list[str] | None = self._sample_questions_from_payload(
                        self._parse_serialized_space()
                    )
                    if discovered:
                        self.sample_questions = discovered
            except Exception as e:
                logger.debug(f"Could not fetch details from Genie space: {e}")

    @property
    def _agent_id(self) -> str:
        """Resolve this room's Genie agent/space id, or raise if unset.

        Static-only by design: ``space_id`` (or its ``agent_id`` alias), else
        the ``DATABRICKS_GENIE_SPACE_ID`` env var. Deliberately does *not* fall
        back to :meth:`_resolve_space_id_by_name` — that is a live Genie call,
        and this runs on the chat-model build path, including Model Serving
        startup where it cannot authenticate.
        """
        space_id: AnyVariable = self.space_id or os.environ.get(
            "DATABRICKS_GENIE_SPACE_ID"
        )
        if isinstance(space_id, dict):
            space_id = CompositeVariableModel(**space_id)
        resolved: Any = value_of(space_id)
        if not resolved:
            raise ValueError(
                "GenieRoomModel: unable to resolve agent_id. Set space_id (or "
                "its alias agent_id), or the DATABRICKS_GENIE_SPACE_ID env var."
            )
        return str(resolved)

    def chat_model_for_workspace_client(
        self,
        workspace_client: WorkspaceClient,
        *,
        conversation_id: "str | None" = None,
        timeout_seconds: int = GENIE_AGENT_DEFAULT_TIMEOUT_SECONDS,
    ) -> LanguageModelLike:
        """Build a Genie Agent chat model bound to a specific workspace client.

        Consumed via :class:`GenieAgentModel` by
        :class:`dao_ai.middleware.genie_agent.GenieAgentMiddleware`, which swaps
        in a user-scoped client (OBO) and the prior Genie ``conversation_id``
        per request, in one step.
        """
        from dao_ai.genie.agent_chat_model import GenieAgentChatModel

        return GenieAgentChatModel(
            agent_id=self._agent_id,
            workspace_client=workspace_client,
            conversation_id=conversation_id,
            timeout_seconds=timeout_seconds,
        )

    def as_chat_model(
        self,
        *,
        timeout_seconds: int = GENIE_AGENT_DEFAULT_TIMEOUT_SECONDS,
    ) -> LanguageModelLike:
        """Build the streaming Genie chat model using the ambient/room client."""
        return self.chat_model_for_workspace_client(
            self.workspace_client, timeout_seconds=timeout_seconds
        )

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
                if spec.left_alias:
                    left["alias"] = spec.left_alias
                right: dict[str, str] = {"identifier": spec.right.full_name}
                if spec.right_alias:
                    right["alias"] = spec.right_alias
                entry: dict[str, Any] = {
                    "id": _stable_id(
                        "join_spec", spec.left.full_name, spec.right.full_name, spec.sql
                    ),
                    "left": left,
                    "right": right,
                    "sql": [spec.sql, _relationship_marker(spec.relationship_type)],
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
            # Genie's export-proto validator now requires ``display_name`` on
            # every snippet type (filters/expressions/measures); historically
            # the schema only required ``alias`` on expressions+measures.
            # Emit both so the payload satisfies both old and new validators.
            snippets[snippet_key] = [
                {
                    "id": _stable_id(snippet_key, snippet.display_name, snippet.sql),
                    "sql": [snippet.sql],
                    "display_name": snippet.display_name,
                    **(
                        {"alias": snippet.display_name}
                        if snippet_key != "filters"
                        else {}
                    ),
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
        their natural key. For entries with multiple identifying fields
        (e.g. ``sql_functions`` entries carry both ``id`` and ``identifier``),
        the validator demands a composite sort over all of them — sorting by
        only one of the fields trips the ``Invalid export proto: ... must be
        sorted by (id, identifier)`` error.
        """
        if isinstance(obj, dict):
            for value in obj.values():
                GenieRoomModel._sort_payload_lists(value)
        elif isinstance(obj, list):
            for item in obj:
                GenieRoomModel._sort_payload_lists(item)
            if obj and isinstance(obj[0], dict):
                sort_keys = tuple(k for k in _GENIE_SORT_KEY_PRIORITY if k in obj[0])
                if sort_keys:
                    obj.sort(key=lambda x: tuple(x.get(k, "") for k in sort_keys))

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

    @staticmethod
    def _sample_questions_from_payload(payload: dict[str, Any]) -> list[str] | None:
        """Read ``config.sample_questions`` out of a ``serialized_space`` payload.

        Returns ``None`` when the payload carries no sample-question block, so
        callers can tell "not present" from "present but empty". Shared by
        :meth:`_apply_serialized_space` and :meth:`ensure_resolved` so the two
        cannot drift.
        """
        cfg = payload.get("config")
        if not isinstance(cfg, dict):
            return None
        sample = cfg.get("sample_questions")
        if not isinstance(sample, list):
            return None
        questions: list[str | None] = [
            _unwrap_text(item.get("question")) if isinstance(item, dict) else None
            for item in sample
        ]
        return [q for q in questions if q]

    def _apply_serialized_space(self, payload: dict[str, Any]) -> None:
        """Write a parsed ``serialized_space`` payload into the model fields."""

        # config.sample_questions
        sample_questions = self._sample_questions_from_payload(payload)
        if sample_questions is not None:
            self.sample_questions = sample_questions

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
        sql_parts: list[str] = entry.get("sql") or []
        if isinstance(sql_parts, str):
            sql_parts = [sql_parts]
        sql_text: str = sql_parts[0] if sql_parts else ""
        # Current export format keeps the cardinality marker in its own
        # element; spaces written by older builds append it to the condition.
        relationship_type: GenieRelationshipType | None = (
            _relationship_type_from_marker(sql_parts[-1])
            if len(sql_parts) > 1
            else None
        )
        rt_match = re.search(r"\s*--rt=([A-Z_]+)--\s*$", sql_text)
        if rt_match:
            if relationship_type is None:
                relationship_type = _relationship_type_from_marker(rt_match.group(0))
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


class GenieAgentModel(BaseModel):
    """A Genie Agent used as an agent's reasoning model (streaming brain).

    Wraps a :class:`GenieRoomModel` and exposes the same duck-typed surface
    ``AgentModel``/``OBOModelMiddleware`` expect from a model resource, so a
    plain agent with no tools becomes a natively-streaming "Genie specialist"
    that a supervisor can route to like any other sub-agent.

    Unlike :class:`InferenceEndpointModel`, a Genie Agent is not a serving
    endpoint: :meth:`as_chat_model` returns a
    :class:`dao_ai.genie.agent_chat_model.GenieAgentChatModel` that streams the
    Genie Agent Mode API (``POST /api/2.0/genie/agents/{agent_id}/responses``)
    rather than a ``ChatDatabricks`` pointed at ``/serving-endpoints``. This is
    a wrapper (composition), not a subclass of ``InferenceEndpointModel`` — a
    Genie Agent has no ``temperature``/``max_tokens``/``use_ai_gateway`` semantics
    and must never be substitutable into embedding/judge/rerank slots.

    Authentication and OBO are delegated wholesale to the wrapped
    ``genie_room`` (an :class:`IsDatabricksResource`), so there is exactly one
    OBO flag (``genie_room.on_behalf_of_user``) and no chance of a second flag
    drifting out of sync.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    genie_room: GenieRoomModel = Field(
        description=(
            "Genie space/agent to invoke. Accepts either ``space_id`` or "
            "``agent_id`` (aliases) for the 32-char hex identifier, and carries "
            "the warehouse and OBO (``on_behalf_of_user``) configuration."
        ),
    )
    timeout_seconds: int = Field(
        default=GENIE_AGENT_DEFAULT_TIMEOUT_SECONDS,
        description=(
            "httpx client timeout in seconds for the streaming call. The "
            "Databricks server-side response timeout is 90 minutes."
        ),
    )

    @property
    def _agent_id(self) -> str:
        """Resolve the wrapped room's agent/space id, or raise if unset."""
        return self.genie_room._agent_id

    @property
    def name(self) -> str:
        """Model name for logging / trace ``ResourceInfo`` (the agent_id)."""
        return self._agent_id

    @property
    def on_behalf_of_user(self) -> bool:
        """Delegate OBO to the wrapped room (the single source of truth)."""
        return bool(self.genie_room.on_behalf_of_user)

    def workspace_client_from(
        self, context: "Context | None", *, strict: bool = False
    ) -> WorkspaceClient:
        """Delegate to the wrapped room so OBO resolves identically to tools."""
        return self.genie_room.workspace_client_from(context, strict=strict)

    def chat_model_for_workspace_client(
        self,
        workspace_client: WorkspaceClient,
        *,
        conversation_id: "str | None" = None,
    ) -> LanguageModelLike:
        """Build a chat model bound to a specific (e.g. OBO) workspace client.

        Consumed by :class:`dao_ai.middleware.genie_agent.GenieAgentMiddleware`
        to swap in a user-scoped client (OBO) and the prior Genie
        ``conversation_id`` per request, in one step. Delegates to the room,
        adding only this wrapper's ``timeout_seconds`` override.
        """
        return self.genie_room.chat_model_for_workspace_client(
            workspace_client,
            conversation_id=conversation_id,
            timeout_seconds=self.timeout_seconds,
        )

    def as_chat_model(self) -> LanguageModelLike:
        """Build the streaming Genie chat model using the ambient/room client."""
        return self.genie_room.as_chat_model(timeout_seconds=self.timeout_seconds)


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

        ``sources`` is the *parent* of the skill leaf, per the deepagents
        SkillsMiddleware source-dir convention — it lists a source's subdirs and
        reads ``SKILL.md`` from each.

        **Pure: no filesystem access.** This runs during model validation, so it
        runs on whichever machine loaded the config — which for a provisioning
        job or a git-sourced deploy is not the machine that will run the agent.
        Probing the filesystem here and storing what it found baked the loader's
        own directory into the config, and ``create_agent`` then serialized that
        dead path into the model artifact; the endpoint came up healthy and
        skill-less. So a local skill's parent stays relative and is resolved at
        graph build time by ``create_skills_middleware``, against the config's
        directory among other anchors. ``root_dir="/"`` is retained for the
        filesystem backend because resolution yields absolute paths, which the
        backend passes through regardless of root.

        Volume-backed skills use the volume root, which is already absolute and
        machine-independent, so nothing needs deferring.
        """
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

        # Local skill: the parent of the declared leaf, still relative. A leaf
        # with no directory component ("product-lookup") yields "." — "the
        # anchor itself is the source dir" — which the resolver understands.
        # ``PurePosixPath`` and not ``rsplit`` so that case does not collapse to
        # ``/`` and start naming the filesystem root.
        parent = str(PurePosixPath(self.path.rstrip("/")).parent)
        return MiddlewareModel(
            name="dao_ai.middleware.skills.create_skills_middleware",
            args={
                "sources": [parent],
                "backend_type": "filesystem",
                "root_dir": "/",
            },
        )


class AiSearchVectorStoreModel(IsDatabricksResource, ManagedResource):
    """
    Configuration model for a Databricks AI Search vector store.

    (Formerly ``VectorStoreModel`` / ``AiSearchIndexModel``. Databricks
    rebranded Vector Search to AI Search; ``AiSearchVectorStoreModel`` is
    the current name — chosen to parallel ``LakebaseVectorStoreModel`` so
    both retriever backends can co-exist under ``resources.vector_stores``.
    Both legacy names remain as aliases defined at the end of the class
    body for backwards compatibility.)

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
    # Discriminator field for the ``AnyVectorStore`` union. Plain-string
    # Literal (not the enum instance) to keep yaml.safe_dump round-tripping
    # clean — same pattern as ``AnyRetriever``.
    type: Literal["ai_search"] = Field(
        default="ai_search",
        description="Discriminator. Must be 'ai_search' (or omitted — defaults).",
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

    # NOTE: endpoint auto-discovery is intentionally NOT done here. It used to
    # live in a ``@model_validator(mode="after")``, but validators run at config
    # PARSE time — which includes serving/MCP-server boot (``AppConfig.from_file``
    # / ``initialize()``). Discovering the endpoint requires a
    # ``VectorSearchClient``, whose bare constructor hard-requires explicit creds
    # and crashes ambient serving auth. The endpoint is a PROVISIONING-only
    # concern (the runtime retrieval path uses ``index.full_name`` and never
    # reads ``endpoint``), so discovery is deferred to ``_create_new_index``.

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

        # Build an authenticated VectorSearchClient from this model's own
        # workspace client when the caller didn't supply one — a bare
        # ``VectorSearchClient()`` raises ``InvalidInputException`` under SP /
        # ambient CLI-profile auth. Same ambient-auth extraction the runtime
        # retrieval path uses (``_vsc_for_refresh`` → ``_client_args_from_ambient_wc``).
        if vsc is None:
            from dao_ai.tools.vector_search import _vsc_for_refresh

            vsc = _vsc_for_refresh(self)

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

        # Resolve provisioning-mode fields (e.g. primary_key) up front. Idempotent
        # (guarded by ``self._resolved``) so this is a no-op when the config was
        # already resolved during ``AppConfig.initialize()``. Endpoint discovery
        # is NOT part of ``ensure_resolved`` — it happens in ``_create_new_index``
        # so serving/MCP boot never builds a VectorSearchClient.
        self.ensure_resolved()

        # Build the VectorSearchClient from this model's own workspace client so
        # provisioning (endpoint discovery + index create) authenticates under
        # every auth mode — including ambient CLI-profile / Serverless-v5, where a
        # bare ``VectorSearchClient()`` raises ``InvalidInputException``. Reuses
        # the same ambient-auth extraction the runtime retrieval path uses
        # (``_vsc_for_refresh`` → ``_client_args_from_ambient_wc``). A caller-
        # supplied ``vsc`` still takes precedence.
        if vsc is None:
            from dao_ai.tools.vector_search import _vsc_for_refresh

            vsc = _vsc_for_refresh(self)

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
        """Create a new vector search index from source table.

        Discovers the target endpoint on demand when one was not configured.
        This is deliberately here (provisioning) rather than in a validator or
        ``ensure_resolved`` so that serving/MCP-server boot — which parses and
        resolves the config but never provisions — makes zero VectorSearch API
        calls and never needs VectorSearch credentials.
        """
        if self.embedding_source_column is None:
            raise ValueError("embedding_source_column is required for provisioning")
        if self.index is None:
            raise ValueError("index is required for provisioning")

        if self.endpoint is None:
            from dao_ai.providers.databricks import with_available_indexes

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

        provider.create_vector_store(self)


# Backwards-compatible aliases — both legacy names point at the same class.
# Python code importing either continues to work. Prefer
# ``AiSearchVectorStoreModel`` in new code.
AiSearchIndexModel = AiSearchVectorStoreModel
VectorStoreModel = AiSearchVectorStoreModel


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
        # ``catalog.connections`` and ``serving.serving-endpoints`` cover the
        # SP-side surface; OBO emission expands ``catalog.connections`` to
        # include the ``mcp.external`` companion automatically (see
        # apps/resources.py:API_SCOPE_TO_USER_SCOPES).
        return [
            "catalog.connections",
            "serving.serving-endpoints",
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
        max_length=APP_RESOURCE_DESCRIPTION_MAX_LENGTH,
        description="Human-readable description of this database connection.",
    )
    host: Optional[AnyVariable] = Field(
        default=None,
        description="PostgreSQL host address. Not needed for Lakebase.",
    )
    database: Optional[AnyVariable] = Field(
        default="databricks_postgres",
        description=(
            "PostgreSQL-level database name (used in connection strings / "
            "``PGDATABASE`` at runtime). Distinct from ``database_id``, "
            "which is the Databricks resource id used in Apps resource "
            "paths. Convention for auto-provisioned Lakebase projects is "
            "underscored (``databricks_postgres``); for custom projects "
            "the pg-level name can be anything (e.g. ``vllm_analytics``). "
            "As an escape hatch, if this value is set to a full resource "
            "path (starting with ``projects/``), it takes precedence over "
            "``database_id`` when constructing the Apps resource binding."
        ),
    )
    database_id: Optional[AnyVariable] = Field(
        default=None,
        description=(
            "Databricks Lakebase resource id — the hyphenated identifier at "
            "the end of the resource path "
            "``projects/<project>/branches/<branch>/databases/<database_id>``. "
            "Distinct from ``database`` (the pg-level DB name). "
            "When left unset (the common case) the bundle generator "
            "auto-detects the resource id by calling "
            "``postgres.list_databases()`` and matching by pg-name, then "
            "returning the resource's ``.name``. Set explicitly when you "
            "want to skip that lookup (e.g. offline bundle generation) or "
            "to point at a specific custom-provisioned database resource. "
            "Set the value to whatever appears as ``database_id`` under "
            "``databricks api get /api/2.0/postgres/projects/<p>/branches/<b>/databases``. "
            "If unset AND the SDK lookup fails, the resolver falls back to "
            "``databricks-postgres`` (Databricks' auto-provisioning default) "
            "and logs a WARNING."
        ),
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
        """Provision this database (Lakebase project + service-principal role).

        Args:
            w: Workspace client to provision *with*. Defaults to ambient
                authentication (the notebook/CLI caller), NOT
                ``self.workspace_client`` — creating a Lakebase project and
                granting a Postgres role require ``Can Manage`` on the project,
                which the database's own service principal does not have. The
                SP remains the role *subject*; see
                :meth:`DatabricksProvider.create_lakebase_autoscaling_role`.
        """
        from dao_ai.providers.databricks import DatabricksProvider

        if w is None:
            w = WorkspaceClient()
        provider: DatabricksProvider = DatabricksProvider(w=w)
        if self.is_lakebase:
            provider.create_lakebase_autoscaling(self)
            provider.create_lakebase_autoscaling_role(self)

    def execute_update(
        self,
        statements: str | Sequence[str],
        parameters: Sequence[Any] | None = None,
    ) -> None:
        """Execute one or more write SQL statements against this database.

        Convenience wrapper around the synchronous connection pool
        (:class:`dao_ai.memory.postgres.PostgresPoolManager`). Runs
        every statement in a single transaction; commits on success and
        rolls back on error. Intended for ad-hoc DDL/DML from notebooks
        and setup scripts — not a substitute for a proper ORM or
        parameterized bulk-write path.

        For read statements (SELECT, RETURNING, SHOW) use
        :meth:`execute_query` instead — this method drops any result set.

        Args:
            statements: A single SQL string OR a sequence of SQL strings.
                Strings are executed verbatim (no additional splitting
                on ``;``). When passing a sequence, ``parameters`` must
                be ``None`` — parameter binding is only supported when
                a single statement is provided.
            parameters: Optional positional parameters passed to
                ``cursor.execute()``. Only valid when ``statements`` is
                a single string.
        """
        from dao_ai.memory.postgres import PostgresPoolManager

        if isinstance(statements, str):
            stmts: list[str] = [statements]
        else:
            stmts = list(statements)
            if parameters is not None:
                raise ValueError(
                    "`parameters` is only supported when `statements` is a "
                    "single string; got a sequence."
                )
        if not stmts:
            return

        pool = PostgresPoolManager.get_pool(self)
        with pool.connection() as conn:
            try:
                with conn.cursor() as cur:
                    for stmt in stmts:
                        if parameters is not None:
                            cur.execute(stmt, parameters)
                        else:
                            cur.execute(stmt)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def execute_query(
        self,
        query: str,
        parameters: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read query and return the fetched rows.

        Companion to :meth:`execute_update` for the read direction. Takes
        exactly one SELECT-style statement and returns its rows as
        ``list[dict]`` keyed by column name (Lakebase pools use
        ``row_factory=dict_row``). Returns an empty list when the query
        yields no rows.

        Args:
            query: The SQL query text (single statement).
            parameters: Optional positional parameters passed to
                ``cursor.execute()``.
        """
        from dao_ai.memory.postgres import PostgresPoolManager

        pool = PostgresPoolManager.get_pool(self)
        with pool.connection() as conn, conn.cursor() as cur:
            if parameters is not None:
                cur.execute(query, parameters)
            else:
                cur.execute(query)
            if cur.description is None:
                return []
            return list(cur.fetchall())

    def execute_many(
        self,
        query: str,
        param_seq: Iterable[Sequence[Any]],
    ) -> None:
        """Execute a parameterized write statement across many rows.

        Wraps psycopg's ``cursor.executemany()`` — one server round-trip
        for all rows, much faster than a Python-level loop of
        ``execute_update`` calls. Intended for bulk INSERT / UPDATE
        with pre-computed parameter tuples.

        Runs inside a single transaction. Commits on success, rolls
        back on error.

        .. code-block:: python

            database.execute_many(
                "UPDATE kb_articles SET embedding = %s::vector WHERE id = %s",
                [(vec, row_id) for row_id, vec in zip(ids, vectors)],
            )
        """
        from dao_ai.memory.postgres import PostgresPoolManager

        pool = PostgresPoolManager.get_pool(self)
        with pool.connection() as conn:
            try:
                with conn.cursor() as cur:
                    cur.executemany(query, param_seq)
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @contextmanager
    def connect(self) -> Iterator[Any]:
        """Yield a psycopg cursor scoped to a single transaction.

        Companion escape hatch to :meth:`execute_update` /
        :meth:`execute_query` for the cases where those aren't enough —
        multi-statement transactions that need row-level control,
        streaming ``fetchmany`` loops, ``executemany`` on precomputed
        rows, or any workflow psycopg handles natively.

        Auto-commits when the ``with`` block exits normally and rolls
        back on any exception. Hides the
        :class:`dao_ai.memory.postgres.PostgresPoolManager` import so
        notebook code has a single-line entry point:

        .. code-block:: python

            with database.connect() as cur:
                cur.execute("SELECT id, passage FROM t WHERE embedding IS NULL")
                rows = cur.fetchall()
                if rows:
                    vectors = embedder.embed_documents([r["passage"] for r in rows])
                    for row, vec in zip(rows, vectors):
                        cur.execute(
                            "UPDATE t SET embedding = %s::vector WHERE id = %s",
                            (vec, row["id"]),
                        )
        """
        from dao_ai.memory.postgres import PostgresPoolManager

        pool = PostgresPoolManager.get_pool(self)
        with pool.connection() as conn:
            try:
                with conn.cursor() as cur:
                    yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    async def aget_pool(self) -> Any:
        """Return the shared async :class:`psycopg_pool.AsyncConnectionPool` for this database.

        Thin async accessor over
        :class:`dao_ai.memory.postgres.AsyncPostgresPoolManager` — every
        async caller in dao-ai (audit receipts, checkpointer, memory
        store, background queue, A2A task store) should route pool
        acquisition through this method so ``DatabaseModel`` remains the
        single entry point for database access.

        Prefer :meth:`aexecute_query` / :meth:`aexecute_update` /
        :meth:`aexecute_many` / :meth:`aconnect` for typical work;
        reach for the raw pool only when you need psycopg-specific
        control (e.g. streaming reads, ``COPY``, or composing
        ``sql.Composed`` inside a shared cursor).
        """
        from dao_ai.memory.postgres import AsyncPostgresPoolManager

        return await AsyncPostgresPoolManager.get_pool(self)

    async def aexecute_update(
        self,
        statements: str | Sequence[str],
        parameters: Sequence[Any] | None = None,
    ) -> None:
        """Async twin of :meth:`execute_update`.

        Runs one or more write SQL statements against the shared async
        pool (:class:`dao_ai.memory.postgres.AsyncPostgresPoolManager`).
        Async Lakebase pools run with ``autocommit=True``, so each
        statement commits individually; on error the exception
        propagates.

        Args:
            statements: A single SQL string OR a sequence of SQL
                strings. When passing a sequence, ``parameters`` must
                be ``None``.
            parameters: Optional positional parameters passed to
                ``cursor.execute()``. Only valid when ``statements`` is
                a single string.
        """
        if isinstance(statements, str):
            stmts: list[str] = [statements]
        else:
            stmts = list(statements)
            if parameters is not None:
                raise ValueError(
                    "`parameters` is only supported when `statements` is a "
                    "single string; got a sequence."
                )
        if not stmts:
            return

        pool = await self.aget_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                for stmt in stmts:
                    if parameters is not None:
                        await cur.execute(stmt, parameters)
                    else:
                        await cur.execute(stmt)

    async def aexecute_query(
        self,
        query: Any,
        parameters: Sequence[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Async twin of :meth:`execute_query`.

        Runs a single read query and returns rows as ``list[dict]``
        keyed by column name (async pools use
        ``row_factory=dict_row``). Returns an empty list when the
        query yields no rows.

        Args:
            query: The SQL query — either a plain ``str`` or a
                ``psycopg.sql.Composable`` (``sql.SQL`` / ``sql.Composed``)
                for identifier-safe composition.
            parameters: Optional positional parameters passed to
                ``cursor.execute()``.
        """
        from psycopg.rows import dict_row

        pool = await self.aget_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                if parameters is not None:
                    await cur.execute(query, parameters)
                else:
                    await cur.execute(query)
                if cur.description is None:
                    return []
                return list(await cur.fetchall())

    async def aexecute_many(
        self,
        query: str,
        param_seq: Iterable[Sequence[Any]],
    ) -> None:
        """Async twin of :meth:`execute_many`.

        Wraps psycopg's ``cursor.executemany()`` on the shared async
        pool. Async Lakebase pools run with ``autocommit=True``.
        """
        pool = await self.aget_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(query, param_seq)

    @asynccontextmanager
    async def aconnect(self) -> AsyncIterator[Any]:
        """Async twin of :meth:`connect`.

        Yields a psycopg async cursor scoped to a single connection
        acquired from the shared async pool. Async Lakebase pools run
        with ``autocommit=True``, so the caller gets each statement
        auto-committed.

        .. code-block:: python

            async with database.aconnect() as cur:
                await cur.execute("SELECT ... WHERE embedding IS NULL")
                rows = await cur.fetchall()
        """
        pool = await self.aget_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                yield cur


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
    type: Literal["string", "number", "boolean", "datetime", "array"] = Field(
        default="string",
        description=(
            "Column data type. ``array`` denotes an ARRAY<primitive> column "
            "on the index; on Databricks VS Standard endpoints these filter "
            "via element containment (equality only). The element type is "
            "irrelevant for filtering, so a single ``array`` value covers "
            "ARRAY<STRING>, ARRAY<INT>, etc."
        ),
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


class RetrieverType(str, Enum):
    """Discriminator values for the ``AnyRetriever`` discriminated union.

    Values match the corresponding ``FunctionType`` tool aliases so YAML
    ``retrievers:`` entries and ``tools[].function.type`` entries speak the
    same vocabulary.
    """

    AI_SEARCH = "ai_search"
    LAKEBASE_SEARCH = "lakebase_search"


class BaseRetrieverModel(ABC, BaseModel):
    """Common surface for all retriever configs.

    Every concrete retriever exposes ``columns`` + ``search_parameters`` and
    knows how to produce a ``StructuredTool`` via :meth:`as_tools`. Concrete
    subclasses add a ``type`` discriminator field (Literal-narrowed) and a
    ``vector_store`` field whose type is retriever-specific.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    columns: Optional[list[str | ColumnInfo]] = Field(
        default_factory=list,
        description=(
            "Columns to expose to the LLM as filterable + returnable. Accepts "
            "either bare strings (name only — types are discovered from the "
            "index at build time) or ColumnInfo objects that declare name / "
            "type / operators / description. Hand-declared ColumnInfo items "
            "are authoritative and skip build-time discovery. The two forms "
            "may be mixed in one list."
        ),
    )
    search_parameters: SearchParametersModel = Field(
        default_factory=SearchParametersModel,
        description="Search tuning: number of results, query type, and metadata filters.",
    )
    rerank: Optional[RerankParametersModel | bool] = Field(
        default=None,
        description=(
            "Optional FlashRank cross-encoder reranking pass over the "
            "retriever's raw hits. Set to ``true`` for defaults, or provide "
            "a ``RerankParametersModel`` for a specific model + ``top_n``. "
            "For LLM instruction-aware reranking use ``instructed.rerank`` "
            "instead."
        ),
    )
    instructed: Optional[InstructedRetrieverModel] = Field(
        default=None,
        description=(
            "Optional instructed-retrieval pipeline: LLM decomposes the "
            "query into constraint-aware subqueries, runs them in parallel, "
            "RRF-merges results, applies an LLM instruction-aware rerank, "
            "and (optionally) verifies + retries. Both retriever backends "
            "share the pipeline implementation in "
            "``dao_ai.tools.instructed_pipeline``."
        ),
    )

    @model_validator(mode="after")
    def _set_default_reranker(self) -> Self:
        """``rerank: true`` → default FlashRank model. Shared by all backends."""
        if isinstance(self.rerank, bool) and self.rerank:
            self.rerank = RerankParametersModel(model="ms-marco-MiniLM-L-12-v2")
        return self

    @abstractmethod
    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        """Build the retrieval tool(s) from this config.

        Subclasses delegate to the matching factory
        (``create_ai_search_tool`` / ``create_lakebase_search_tool``).
        Signature mirrors :meth:`BaseFunctionModel.as_tools`.
        """
        ...


class AiSearchRetrieverModel(BaseRetrieverModel):
    """Retriever backed by a Databricks AI Search (formerly Vector Search) index.

    (Formerly ``RetrieverModel``. Renamed to disambiguate now that Lakebase
    Postgres has its own retriever config — see :class:`LakebaseRetrieverModel`.)
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # String Literal instead of Literal[RetrieverType.AI_SEARCH] — Pydantic's
    # use_enum_values doesn't coerce enum instances inside a Literal-typed
    # field during model_dump, which broke yaml.safe_dump round-tripping.
    # The enum values (:class:`RetrieverType`) still document the vocabulary.
    type: Literal["ai_search"] = Field(
        default="ai_search",
        description="Discriminator. Must be 'ai_search' (or omitted — defaults).",
    )
    vector_store: VectorStoreModel = Field(
        description="AI Search / Vector Search index configuration used for similarity search.",
    )

    @model_validator(mode="after")
    def set_default_columns(self) -> Self:
        if not self.columns:
            self.columns = list(self.vector_store.columns or [])
        return self

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_ai_search_tool

        return [create_ai_search_tool(retriever=self, **kwargs)]


class LakebaseVectorStoreModel(BaseModel):
    """Configuration for a Databricks Lakebase Postgres table used for retrieval.

    Points at a table that has ``lakebase_vector`` (and optionally
    ``lakebase_text``) extensions installed and appropriate indexes
    built. Call :meth:`provision` to idempotently create the extensions,
    table, and indexes if they don't exist yet.

    Example (existing table, hybrid-ready):
    ```yaml
    vector_store:
      database:
        project: my-lakebase-project
      table: kb_articles
      content_column: passage
      embedding_column: embedding
      tsvector_column: passage_tsv
      embedding_model: databricks-gte-large-en
      metadata_columns: [category, source_url]
      bm25_index_name: kb_articles_passage_bm25
      distance_metric: cosine
    ```
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # Discriminator field for the ``AnyVectorStore`` union. Plain-string
    # Literal to keep yaml.safe_dump round-tripping clean — same pattern
    # as ``AnyRetriever``.
    type: Literal["lakebase_search"] = Field(
        default="lakebase_search",
        description="Discriminator. Must be 'lakebase_search'.",
    )
    database: DatabaseModel = Field(
        description=(
            "Lakebase (or PostgreSQL) database connection. Reuses the "
            "existing DatabaseModel — pool + auth handled by "
            "``dao_ai.memory.postgres`` pool managers."
        ),
    )
    schema_name: str = Field(
        default="public",
        description="Postgres schema containing the retrieval table.",
    )
    table: str = Field(
        description="Source table with vector (and optionally tsvector) columns.",
    )
    id_column: str = Field(
        default="id",
        description="Primary-key column exposed as ``Document.metadata['id']``.",
    )
    content_column: str = Field(
        description="Text column returned as ``Document.page_content``.",
    )
    embedding_column: str = Field(
        description=(
            "``VECTOR(N)`` column indexed by ``lakebase_ann``. "
            "Dimension must match the embedding model."
        ),
    )
    tsvector_column: Optional[str] = Field(
        default=None,
        description=(
            "``tsvector`` column indexed by ``lakebase_bm25``. "
            "Required for BM25 and HYBRID query types."
        ),
    )
    metadata_columns: Optional[list[str]] = Field(
        default_factory=list,
        description="Additional columns surfaced on ``Document.metadata``.",
    )
    embedding_model: InferenceEndpointModel = Field(
        description=(
            "Embedding endpoint used to embed the query at retrieval time. "
            "Accepts a bare endpoint name (e.g. ``databricks-gte-large-en``) "
            "which is coerced to ``InferenceEndpointModel(name=<str>)``."
        ),
    )
    bm25_index_name: Optional[str] = Field(
        default=None,
        description=(
            "Regclass name required by ``to_bm25query(..., <index>::regclass)``. "
            "Auto-derived from ``<schema>.<table>_<tsvector_column>_bm25`` if omitted."
        ),
    )
    distance_metric: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description=(
            "Vector distance operator: ``cosine`` (``<=>`` / ``vector_cosine_ops``), "
            "``l2`` (``<->`` / ``vector_l2_ops``), ``ip`` (``<#>`` / ``vector_ip_ops``)."
        ),
    )
    tsv_language: str = Field(
        default="english",
        description="Text-search dictionary passed to ``to_tsvector(<lang>, ...)``.",
    )

    @field_validator("embedding_model", mode="before")
    @classmethod
    def _coerce_embedding_model(cls, v: Any) -> Any:
        """Accept a bare endpoint name string as shorthand."""
        if isinstance(v, str):
            return InferenceEndpointModel(name=v)
        return v

    @model_validator(mode="after")
    def _derive_bm25_index_name(self) -> Self:
        if self.bm25_index_name is None and self.tsvector_column is not None:
            self.bm25_index_name = (
                f"{self.schema_name}.{self.table}_{self.tsvector_column}_bm25"
            )
        return self

    # -- IsDatabricksResource-shaped delegation --------------------------
    # LakebaseVectorStoreModel is not itself a Databricks resource — its
    # auth genuinely lives on the nested ``database``. Delegating the
    # three IsDatabricksResource members that deploy paths iterate
    # (``as_resources()``, ``api_scopes``, ``on_behalf_of_user``) keeps
    # ``resources.vector_stores`` polymorphic without lying about class
    # identity or forcing multiple inheritance. Callers that need to
    # dispatch further (e.g. only emit ``vector-search-index`` bundle
    # resources for AI Search) still do so via ``isinstance``.

    @property
    def on_behalf_of_user(self) -> bool:
        return self.database.on_behalf_of_user

    @property
    def api_scopes(self) -> Sequence[str]:
        return self.database.api_scopes

    def as_resources(self) -> Sequence["DatabricksResource"]:
        return self.database.as_resources()

    def provision(
        self,
        *,
        dimension: int,
        metadata_column_types: dict[str, str] | None = None,
        id_column_type: str = "text",
    ) -> None:
        """Idempotently create the Postgres extensions + table + indexes.

        Runs ``CREATE EXTENSION IF NOT EXISTS`` for ``lakebase_vector``
        (and ``lakebase_text`` when ``tsvector_column`` is configured),
        then ``CREATE TABLE IF NOT EXISTS`` for the retrieval table with
        columns matching this vector-store config, then
        ``CREATE INDEX IF NOT EXISTS`` for the ``lakebase_ann`` index on
        ``embedding_column`` (and ``lakebase_bm25`` on ``tsvector_column``
        when configured). Every statement is safe to re-run — no rows or
        indexes are dropped.

        The generated table has this shape (all columns nullable except
        the id + content column):

        - ``{id_column} {id_column_type} PRIMARY KEY``
        - ``{content_column} text NOT NULL``
        - ``{embedding_column} vector({dimension})``
        - Optional: ``{tsvector_column} tsvector GENERATED ALWAYS AS
          (to_tsvector('{tsv_language}', {content_column})) STORED``
        - Each metadata column typed via ``metadata_column_types``
          (defaults to ``text`` when unspecified).

        Args:
            dimension: Vector dimension of the ``embedding_column``. Must
                match the ``embedding_model`` endpoint's output shape
                (e.g. ``1024`` for ``databricks-gte-large-en``). No
                auto-detection — the caller knows the endpoint.
            metadata_column_types: Optional ``{name: pg_type}`` overrides
                for entries in ``metadata_columns``. Any name not listed
                defaults to ``text``. Use standard Postgres type names
                (``int``, ``bigint``, ``numeric``, ``timestamp``, etc.).
            id_column_type: Postgres type for the primary-key column.
                Defaults to ``text`` to match the ``kb_articles`` seed
                convention; use ``bigint`` / ``uuid`` etc. as needed.

        This method is idempotent. It does NOT drop the existing table
        or its data — pointing ``provision()`` at an existing table with
        a different schema is a no-op (Postgres accepts the ``CREATE
        TABLE IF NOT EXISTS`` silently even when the shape differs). To
        rebuild from scratch, drop the table manually first.
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be a positive int; got {dimension!r}")
        meta_types = metadata_column_types or {}
        qualified = f"{self.schema_name}.{self.table}"

        stmts: list[str] = ["CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE;"]
        if self.tsvector_column:
            stmts.append("CREATE EXTENSION IF NOT EXISTS lakebase_text;")

        cols: list[str] = [
            f"{self.id_column} {id_column_type} PRIMARY KEY",
            f"{self.content_column} text NOT NULL",
            f"{self.embedding_column} vector({dimension})",
        ]
        for col in self.metadata_columns or []:
            col_type = meta_types.get(col, "text")
            cols.append(f"{col} {col_type}")
        if self.tsvector_column:
            cols.append(
                f"{self.tsvector_column} tsvector GENERATED ALWAYS AS "
                f"(to_tsvector('{self.tsv_language}', {self.content_column})) STORED"
            )
        stmts.append(
            f"CREATE TABLE IF NOT EXISTS {qualified} (\n    "
            + ",\n    ".join(cols)
            + "\n);"
        )

        # ANN index — name follows the same convention the tool uses when
        # reading back (`<table>_<column>_ann`). vector_cosine_ops matches
        # the default `distance_metric`; overridden below when needed.
        ops_map = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "ip": "vector_ip_ops",
        }
        ops = ops_map[self.distance_metric]
        ann_index = f"{self.table}_{self.embedding_column}_ann"
        stmts.append(
            f"CREATE INDEX IF NOT EXISTS {ann_index} "
            f"ON {qualified} USING lakebase_ann ({self.embedding_column} {ops});"
        )
        if self.tsvector_column:
            bm25_index = f"{self.table}_{self.tsvector_column}_bm25"
            stmts.append(
                f"CREATE INDEX IF NOT EXISTS {bm25_index} "
                f"ON {qualified} USING lakebase_bm25 ({self.tsvector_column});"
            )
        self.database.execute_update(stmts)


class LakebaseRetrieverModel(BaseRetrieverModel):
    """Full Lakebase retriever config — vector store plus search parameters."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    type: Literal["lakebase_search"] = Field(
        default="lakebase_search",
        description="Discriminator. Must be 'lakebase_search'.",
    )
    vector_store: LakebaseVectorStoreModel = Field(
        description="Lakebase table + columns + embedding model.",
    )

    @model_validator(mode="after")
    def _set_default_columns(self) -> Self:
        if not self.columns:
            self.columns = list(self.vector_store.metadata_columns or [])
        return self

    @model_validator(mode="after")
    def _bm25_requires_tsvector(self) -> Self:
        qt = (self.search_parameters.query_type or "ANN").upper()
        if qt in {"BM25", "HYBRID"} and self.vector_store.tsvector_column is None:
            raise ValueError(
                f"query_type={qt!r} requires 'tsvector_column' on the vector store."
            )
        return self

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_lakebase_search_tool

        return [create_lakebase_search_tool(retriever=self, **kwargs)]


def _retriever_discriminator(v: Any) -> str:
    """Callable discriminator for :data:`AnyRetriever`.

    Pydantic v2 tagged unions do not honour Literal defaults on missing tag
    fields — the tag must be present in the raw input. This callable
    normalizes both dict input (YAML) and already-instantiated models to a
    discriminator string, defaulting to ``"ai_search"`` when ``type`` is
    absent. That's the load-bearing back-compat hook that lets existing
    YAML ``retrievers:`` entries (which never specified ``type:``) keep
    parsing as AI Search retrievers.
    """
    if isinstance(v, BaseRetrieverModel):
        return v.type if isinstance(v.type, str) else v.type.value
    if isinstance(v, dict):
        raw = v.get("type")
        if raw is None:
            return RetrieverType.AI_SEARCH.value
        return raw.value if isinstance(raw, RetrieverType) else str(raw)
    raise ValueError(f"cannot infer retriever discriminator from {type(v).__name__}")


# Discriminated union — Pydantic dispatches to the concrete class using the
# callable above. Existing YAML retriever entries that omit ``type`` default
# to ``AiSearchRetrieverModel`` for back-compat. Each Union member needs an
# explicit ``Tag`` because the discriminator is callable (Pydantic v2
# requirement).
AnyRetriever: TypeAlias = Annotated[
    Union[
        Annotated[AiSearchRetrieverModel, Tag(RetrieverType.AI_SEARCH.value)],
        Annotated[LakebaseRetrieverModel, Tag(RetrieverType.LAKEBASE_SEARCH.value)],
    ],
    Discriminator(_retriever_discriminator),
]


def _vector_store_discriminator(v: Any) -> str:
    """Callable discriminator for :data:`AnyVectorStore`.

    Mirrors :func:`_retriever_discriminator`. Defaults to ``"ai_search"``
    when the ``type`` field is absent — load-bearing back-compat hook
    that lets existing YAML ``resources.vector_stores:`` entries (which
    never specified ``type:``) keep parsing as
    :class:`AiSearchVectorStoreModel`.
    """
    if isinstance(v, (AiSearchVectorStoreModel, LakebaseVectorStoreModel)):
        return v.type if isinstance(v.type, str) else v.type.value
    if isinstance(v, dict):
        raw = v.get("type")
        if raw is None:
            return "ai_search"
        return raw.value if hasattr(raw, "value") else str(raw)
    raise ValueError(f"cannot infer vector_store discriminator from {type(v).__name__}")


# Discriminated union — Pydantic dispatches to the concrete class using
# the callable above. Existing YAML entries under
# ``resources.vector_stores`` that omit ``type`` default to
# :class:`AiSearchVectorStoreModel` for back-compat.
AnyVectorStore: TypeAlias = Annotated[
    Union[
        Annotated[AiSearchVectorStoreModel, Tag("ai_search")],
        Annotated[LakebaseVectorStoreModel, Tag("lakebase_search")],
    ],
    Discriminator(_vector_store_discriminator),
]


class FunctionType(str, Enum):
    PYTHON = "python"
    FACTORY = "factory"
    UNITY_CATALOG = "unity_catalog"
    MCP = "mcp"
    INLINE = "inline"
    GENIE = "genie"
    # Vector Search was renamed to AI Search by Databricks. Both YAML
    # values (``vector_search`` and ``ai_search``) route to the same
    # tool model; ``vector_search`` will eventually be deprecated.
    VECTOR_SEARCH = "vector_search"
    AI_SEARCH = "ai_search"
    LAKEBASE_SEARCH = "lakebase_search"
    SEARCH = "search"
    APP = "app"
    SERVING_ENDPOINT = "serving_endpoint"
    A2A = "a2a"
    SQL = "sql"


class ParamSource(str, Enum):
    """Where a SQL statement parameter's value comes from at runtime."""

    LLM = "llm"
    CONTEXT = "context"


class StatementParam(BaseModel):
    """A single bound parameter for a SQL statement tool.

    Values are bound natively (warehouse: ``:name`` markers; Lakebase / Postgres:
    ``%(name)s`` markers) — never interpolated into the SQL string. ``source``
    decides where the value comes from: ``llm`` params are surfaced in the tool
    schema the model sees, while ``context`` params are resolved server-side from
    the runtime ``Context`` and are never exposed to the model.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Parameter/marker name as it appears in the SQL statement.",
    )
    type: Literal["string", "int", "float", "bool"] = Field(
        default="string",
        description="Declared value type; shapes the LLM-facing tool schema.",
    )
    source: ParamSource = Field(
        default=ParamSource.LLM,
        description=(
            "Value source: 'llm' (model supplies it, appears in the tool schema) "
            "or 'context' (bound from runtime Context, hidden from the model)."
        ),
    )
    required: bool = Field(
        default=True,
        description="Whether a value must be present; a missing required value errors.",
    )
    default: Optional[Any] = Field(
        default=None,
        description="Fallback value applied when the parameter is not supplied.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human/LLM-facing description of the parameter.",
    )
    context_key: Optional[str] = Field(
        default=None,
        description=(
            "For source='context': the Context attribute to read. Defaults to the "
            "parameter 'name' when omitted."
        ),
    )


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


class AuditModel(BaseModel):
    """
    Configuration for tamper-evident audit receipts on tool invocations.

    Presence of this block on a tool's ``function`` enables auditing for that
    tool. When absent, no audit behavior is added and the runtime path is
    bit-for-bit unchanged. Typically declared once as a YAML anchor
    (``audit: &audit_sink { ... }``) and referenced from every tool that
    should be audited (``audit: *audit_sink``).

    Receipts are written to a Lakebase table specified by ``database`` and
    ``table``; the destination table is created idempotently on first use
    with an append-only trigger. Audit works with or without
    ``human_in_the_loop`` on the same tool:

    - Tool with ``audit`` only: an execution receipt is recorded on every
      tool invocation (who/what/when/args_hash).
    - Tool with ``audit`` and ``human_in_the_loop``: the receipt is enriched
      with approval fields (decision, approver, nonce, args_hash binding)
      and the tool call is aborted fail-closed if the args hash differs
      between interrupt time and execution time.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    database: DatabaseModel = Field(
        ...,
        description=(
            "Lakebase database that stores audit receipts and, when combined "
            "with human_in_the_loop, single-use approval nonces. Reuse the "
            "same DatabaseModel anchor used for the HITL checkpointer to "
            "avoid provisioning a second Lakebase."
        ),
    )
    table: str = Field(
        default="audit_receipts",
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        max_length=48,
        description=(
            "Table name (unqualified) for audit receipts within the "
            "configured Lakebase database. Must match the Postgres "
            "unquoted-identifier grammar ``^[A-Za-z_][A-Za-z0-9_]*$`` and "
            "be at most 48 characters so derived identifiers (indexes, "
            "trigger names, function name — all suffixed with up to 15 "
            "characters) stay within Postgres' 63-char identifier limit. "
            "The table is created idempotently on first write."
        ),
    )
    nonce_ttl_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description=(
            "Server-issued nonce lifetime (seconds) for approvals. Resume "
            "attempts arriving after this window are rejected fail-closed. "
            "Only applies when the same tool also has human_in_the_loop set."
        ),
    )

    def as_audit_sink(self) -> "LakebaseAuditSink":
        """Return the audit sink instance for this configuration.

        Lazy-imports the sink implementation so the disabled path never
        loads the audit module or its dependencies.
        """
        from dao_ai.audit import AuditSinkManager

        return AuditSinkManager.for_config(self)


class ToolCallLimitModel(BaseModel):
    """
    Configuration for capping how many times a tool may be called.

    Presence of this block on a tool's ``function`` registers a
    ``ToolCallLimitMiddleware`` for that tool on every agent that uses it, so
    the limit follows the tool rather than being re-declared on each agent's
    ``middleware`` list. Typically declared once as a YAML anchor
    (``call_limit: &tool_limit { ... }``) and referenced from every tool that
    should share the same cap.

    A bare integer is accepted as shorthand for ``run_limit`` with
    ``exit_behavior='continue'`` (see ``BaseFunctionModel.call_limit``); the
    object form below gives full control. At least one of ``run_limit`` or
    ``thread_limit`` must be set. This maps to LangChain's
    ``ToolCallLimitMiddleware`` via ``create_tool_call_limit_middleware``.
    """

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", populate_by_name=True
    )

    run_limit: Optional[int] = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("run_limit", "turn_limit"),
        description=(
            "Maximum number of times this tool may be called within a single "
            "agent invocation (one user message). Resets each run and requires "
            "no checkpointer. Alias: ``turn_limit``."
        ),
    )
    thread_limit: Optional[int] = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("thread_limit", "conversation_limit"),
        description=(
            "Maximum number of times this tool may be called across an entire "
            "conversation (thread). Requires a checkpointer, which DAO AI "
            "agents have via memory. Alias: ``conversation_limit``."
        ),
    )
    exit_behavior: Literal["continue", "error", "end"] = Field(
        default="continue",
        description=(
            "Behavior when a limit is reached: 'continue' blocks the call and "
            "returns an error message so the agent can try another approach "
            "(recommended); 'error' raises immediately; 'end' stops execution "
            "gracefully (single-tool scenarios only)."
        ),
    )

    @model_validator(mode="after")
    def validate_at_least_one_limit(self) -> Self:
        """Require at least one of run_limit or thread_limit to be set."""
        if self.run_limit is None and self.thread_limit is None:
            raise ValueError(
                "At least one of run_limit or thread_limit must be specified."
            )
        return self


class ModelCallLimitModel(BaseModel):
    """
    Configuration for capping how many LLM (model) calls an agent may make.

    The model-call analogue of :class:`ToolCallLimitModel`. Presence of this
    block on an agent's ``call_limit`` registers a ``ModelCallLimitMiddleware``
    for that agent, bounding the ReAct loop directly — a more discoverable,
    co-located alternative to hand-wiring
    ``dao_ai.middleware.create_model_call_limit_middleware`` into the agent's
    ``middleware`` list. Complements :attr:`AgentModel.recursion_limit` (the
    graph-superstep backstop): ``run_limit`` is the graceful per-invocation cap.

    A bare integer is accepted as shorthand for ``run_limit`` (see
    ``AgentModel.call_limit``); the object form below gives full control. At
    least one of ``run_limit`` or ``thread_limit`` must be set. This maps to
    ``create_model_call_limit_middleware``.

    Unlike the tool variant, ``exit_behavior`` only accepts ``'end'`` (default)
    or ``'error'`` — a model-call limit has no per-tool ``'continue'`` semantics
    (there is no alternative tool to fall back to).
    """

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", populate_by_name=True
    )

    run_limit: Optional[int] = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("run_limit", "turn_limit"),
        description=(
            "Maximum number of model (LLM) calls this agent may make within a "
            "single invocation (one user message). Resets each run and requires "
            "no checkpointer. Alias: ``turn_limit``."
        ),
    )
    thread_limit: Optional[int] = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices("thread_limit", "conversation_limit"),
        description=(
            "Maximum number of model (LLM) calls this agent may make across an "
            "entire conversation (thread). Requires a checkpointer, which DAO AI "
            "agents have via memory. Alias: ``conversation_limit``."
        ),
    )
    exit_behavior: Literal["error", "end"] = Field(
        default="end",
        description=(
            "Behavior when a limit is reached: 'end' stops execution gracefully "
            "(recommended); 'error' raises immediately. Unlike tool-call limits, "
            "'continue' is not supported for model-call limits."
        ),
    )

    @model_validator(mode="after")
    def validate_at_least_one_limit(self) -> Self:
        """Require at least one of run_limit or thread_limit to be set."""
        if self.run_limit is None and self.thread_limit is None:
            raise ValueError(
                "At least one of run_limit or thread_limit must be specified."
            )
        return self


class BaseFunctionModel(ABC, BaseModel):
    """Base class for all function/tool implementations (Python, factory, inline, MCP, UC)."""

    model_config = ConfigDict(
        use_enum_values=True,
        discriminator="type",
    )
    type: FunctionType = Field(
        description="Function type discriminator (python, factory, inline, mcp, unity_catalog, genie, ai_search, vector_search (legacy alias for ai_search), search, agent, a2a).",
    )
    human_in_the_loop: Optional[HumanInTheLoopModel] = Field(
        default=None,
        description="Human-in-the-loop approval configuration for this tool.",
    )
    audit: Optional[AuditModel] = Field(
        default=None,
        description=(
            "Optional tamper-evident audit trail for this tool. Presence of "
            "this block enables audit-receipt logging on every invocation; "
            "absence leaves the runtime path unchanged. Typically declared "
            "once as a YAML anchor and referenced from every tool that "
            "should be audited. Composes with human_in_the_loop to produce "
            "approval receipts with args-hash binding and fail-closed "
            "execution."
        ),
    )
    call_limit: Optional[int | ToolCallLimitModel] = Field(
        default=None,
        description=(
            "Shortcut to cap how many times this tool may be called. A bare "
            "integer sets run_limit (per-invocation) with "
            "exit_behavior='continue'; an object gives full control over "
            "run_limit/thread_limit/exit_behavior. Applied automatically to "
            "every agent that uses this tool, in addition to any explicit "
            "tool-call-limit middleware configured on the agent."
        ),
    )

    @field_validator("call_limit", mode="before")
    @classmethod
    def _normalize_call_limit(cls, value: Any) -> Any:
        """Normalize a bare integer shortcut into a ToolCallLimitModel dict.

        Runs before union validation so a bare int becomes ``run_limit``.
        ``bool`` is a subclass of ``int`` but is never a valid limit, so
        reject it explicitly rather than silently coercing ``True`` to
        ``run_limit=1``.
        """
        if isinstance(value, bool):
            raise ValueError(
                "call_limit must be a positive integer or an object, not a bool."
            )
        if isinstance(value, int):
            return {"run_limit": value}
        return value

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


class McpResourceModel(BaseModel):
    """A static resource exposed by dao-ai's own MCP server via ``resources/list``.

    Resources are read-only URIs the server publishes for client discovery
    (system prompts, curated document snippets, etc). Content is served as
    text — for binary payloads, register at the server-side rather than
    declaring here.
    """

    model_config = ConfigDict(extra="forbid")
    uri: str = Field(
        ...,
        description="Resource URI advertised on resources/list, e.g. 'dao-ai://prompts/system'.",
    )
    name: str = Field(
        ...,
        description="Human-readable resource name shown in client UIs.",
    )
    description: Optional[str] = Field(
        default=None,
        description="What the resource contains — surfaced in the resources/list response.",
    )
    mime_type: str = Field(
        default="text/plain",
        description="MIME type of the resource payload. Defaults to text/plain.",
    )
    content: str = Field(
        ...,
        description="Static text content returned by the server on resources/read for this URI.",
    )


class McpPromptArgumentModel(BaseModel):
    """A single argument on an McpPromptModel."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        ...,
        description="Argument name — must match a {placeholder} in the template.",
    )
    description: Optional[str] = Field(
        default=None,
        description="What the argument represents. Shown to clients requesting the prompt.",
    )
    required: bool = Field(
        default=False,
        description="Whether the client must supply this argument on prompts/get. Optional args are empty-string when omitted.",
    )


class McpPromptModel(BaseModel):
    """A prompt template exposed by dao-ai's own MCP server via ``prompts/list``.

    Clients call ``prompts/get`` with argument values; the server returns the
    rendered template as a single user-role message. Placeholders in the
    template use Python format-string syntax, e.g. ``{customer_name}``.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        ...,
        description="Prompt identifier used on prompts/list and prompts/get.",
    )
    description: Optional[str] = Field(
        default=None,
        description="What the prompt does — surfaced in prompts/list.",
    )
    template: str = Field(
        ...,
        description="Prompt template with Python format-string placeholders (e.g. 'Hello, {name}!').",
    )
    arguments: list[McpPromptArgumentModel] = Field(
        default_factory=list,
        description="Arguments the client can supply on prompts/get. Empty list means the template takes no arguments.",
    )


class McpServerCapabilitiesModel(BaseModel):
    """Advanced capabilities emitted BY dao-ai when deployed as an MCP server.

    Client-side capabilities are on ``McpFunctionModel.capabilities``. This
    model is the server-side complement — declaring what the dao-ai MCP
    server itself publishes and how it reports progress/logging to callers.

    When None on ``AppModel`` (the default), dao-ai's MCP server publishes
    only the single agent-as-tool surface — no static resources or prompts,
    no progress notifications.
    """

    model_config = ConfigDict(extra="forbid")
    progress: bool = Field(
        default=True,
        description="Emit progress notifications from LangGraph astream_events during agent execution. Requires the caller to supply a progressToken via _meta on tools/call.",
    )
    resources: list[McpResourceModel] = Field(
        default_factory=list,
        description="Static resources published via resources/list. Empty list means no resources are advertised.",
    )
    prompts: list[McpPromptModel] = Field(
        default_factory=list,
        description="Prompt templates published via prompts/list. Empty list means no prompts are advertised.",
    )


class McpCapabilitiesModel(BaseModel):
    """Advanced MCP capabilities for an MCP client (McpFunctionModel).

    When None (the default on McpFunctionModel), the classic
    MultiServerMCPClient path runs with no callbacks or interceptors —
    guaranteeing byte-for-byte behavior parity with the pre-capabilities
    version.

    When set, dao-ai wires langchain-mcp-adapters ``Callbacks`` and
    ``ToolCallInterceptor`` middleware around the client.
    """

    model_config = ConfigDict(extra="forbid")
    progress: bool = Field(
        default=False,
        description="Consume progress notifications from the MCP server; forward as MLflow span events on the enclosing tool span.",
    )
    elicitation: Optional[Literal["hitl", "reject"]] = Field(
        default=None,
        description="Handle server-initiated elicitation/create. 'hitl' raises a LangGraph interrupt whose resume value becomes the ElicitResult; 'reject' returns action='cancel' without prompting.",
    )
    structured_output: bool = Field(
        default=True,
        description="Prefer CallToolResult.structuredContent and expand resource_link items into MLflow span attributes via a ToolCallInterceptor. Additive only; falls back to text extraction when structuredContent is absent.",
    )


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
        description="Direct MCP server URL. Mutually exclusive with app, connection, genie_room, genie, sql, vector_search, functions.",
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
    genie: Optional[bool] = Field(
        default=None,
        description="Enable the workspace-wide Databricks Genie MCP server (queries across all Genie spaces in the workspace, no space_id required).",
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
    meta: Optional[dict[str, AnyVariable]] = Field(
        default=None,
        description=(
            "Per-MCP-server `_meta` parameters (MCP spec). Sent as `_meta` on every "
            "tools/call request to this server. Common Databricks keys: "
            "warehouse_id (DBSQL); num_results, filters, query_type, columns, "
            "score_threshold, include_score, columns_to_rerank (Vector Search). "
            "Values support AnyVariable (env vars, secrets, defaults)."
        ),
    )
    capabilities: Optional[McpCapabilitiesModel] = Field(
        default=None,
        description=(
            "Advanced MCP capabilities (progress, logging, elicitation, structured "
            "output, sampling, roots). When None the classic MultiServerMCPClient "
            "path is used with no callbacks or interceptors — zero regression from "
            "the pre-capabilities behavior. See McpCapabilitiesModel."
        ),
    )

    @property
    def api_scopes(self) -> Sequence[str]:
        """API scopes for an MCP connection.

        Sub-type aware: each managed-MCP endpoint kind declares the native
        platform scope for the resource it fronts, and OBO emission derives the
        ``mcp.*`` companion scope from it (see
        apps/resources.py:API_SCOPE_TO_USER_SCOPES). Without this an OBO MCP
        tool contributes no usable user-API scope and OBO calls to the endpoint
        are under-scoped.

        Mapping (mirrors :meth:`mcp_url`):
          - genie_room / genie      → ``dashboards.genie``      (→ genie, mcp.genie)
          - sql / functions         → ``sql.warehouses``        (→ sql, mcp.functions)
          - vector_search           → ``vectorsearch.vector-search-indexes``
                                                                (→ vector-search, mcp.vectorsearch)
          - connection              → ``catalog.connections``   (→ catalog.connections, mcp.external)
          - url / app (opaque)      → ``serving.serving-endpoints`` (best-effort default)
        """
        if self.genie_room is not None or self.genie:
            return ["dashboards.genie"]
        if self.sql or self.functions is not None:
            return ["sql.warehouses"]
        if self.vector_search is not None:
            return ["vectorsearch.vector-search-indexes"]
        if self.connection is not None:
            return ["catalog.connections"]
        # Direct url / Databricks App MCP: endpoint kind is opaque here, so fall
        # back to the generic serving scope.
        return ["serving.serving-endpoints"]

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
        - If genie is set, constructs workspace-wide Genie MCP URL (no space_id)
        - If sql is set, constructs DBSQL MCP URL (serverless)
        - If vector_search is set, constructs Vector Search MCP URL
        - If functions is set, constructs UC Functions MCP URL

        URL patterns (per https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp):
        - Genie (per-space): https://{host}/api/2.0/mcp/genie/{space_id}
        - Genie (workspace-wide): https://{host}/api/2.0/mcp/genie (all spaces)
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

        # Genie Room (per-space)
        if self.genie_room:
            space_id: str = value_of(self.genie_room.space_id)
            return f"{workspace_host}/api/2.0/mcp/genie/{space_id}"

        # Workspace-wide Genie MCP server (serverless, no space_id)
        if self.genie:
            return f"{workspace_host}/api/2.0/mcp/genie"

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
            "genie, sql, vector_search, or functions"
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
            ("genie", self.genie),
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
                    "url, app, connection, genie_room, genie, sql, vector_search, or functions"
                )
            if len(provided_sources) > 1:
                raise ValueError(
                    f"For STREAMABLE_HTTP transport, only one URL source can be provided. "
                    f"Found: {', '.join(provided_sources)}. "
                    f"Please provide only one of: url, app, connection, genie_room, genie, sql, vector_search, or functions"
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


class GenieToolModel(BaseFunctionModel):
    """First-class Genie tool that delegates to ``dao_ai.tools.create_genie_tool``.

    Equivalent to ``type: factory + name: dao_ai.tools.create_genie_tool``, but
    with typed fields so beginners get IDE autocomplete and JSON-schema
    validation. Returns a single uncached tool when no caching is configured,
    or a ``GenieToolkit`` (query + feedback tools) when any cache is set or
    ``enable_feedback=True``.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.GENIE] = Field(
        default=FunctionType.GENIE,
        description="Function type discriminator. Must be 'genie'.",
    )
    genie_room: GenieRoomModel = Field(
        description="Genie space configuration.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Tool name visible to the LLM. Defaults to 'genie_tool'.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Tool description shown to the LLM during function calling.",
    )
    persist_conversation: bool = Field(
        default=True,
        description="Persist conversation IDs across calls for multi-turn Genie chats.",
    )
    truncate_results: bool = Field(
        default=False,
        description="Truncate large query results returned by Genie.",
    )
    preserve_question: bool = Field(
        default=False,
        description=(
            "When true, instruct the calling LLM to pass the user's question to "
            "Genie exactly as asked — no rephrasing, decomposition, or added "
            "qualifiers. Constrains the question sent *to* Genie, not Genie's "
            "answer. Shapes the tool description and the question-argument "
            "annotation. Default false preserves the existing 'ask simple, clear "
            "questions' behavior."
        ),
    )
    include_example_questions: bool = Field(
        default=False,
        description=(
            "When true, append the Genie space's example questions to the tool "
            "description, giving the supervisor concrete routing signal. Opt-in "
            "in every case: default false never appends them, whether or not a "
            "'description' is set on this tool, so nothing reaches the prompt "
            "that you did not ask for."
        ),
    )
    lru_cache: Optional[GenieLRUCacheParametersModel] = Field(
        default=None,
        description="LRU cache configuration for fast exact-match SQL caching.",
    )
    context_aware_cache: Optional[GenieContextAwareCacheParametersModel] = Field(
        default=None,
        description="PostgreSQL/Lakebase context-aware (semantic) cache configuration.",
    )
    in_memory_context_aware_cache: Optional[
        GenieInMemoryContextAwareCacheParametersModel
    ] = Field(
        default=None,
        description="In-memory context-aware (semantic) cache configuration.",
    )
    max_consecutive_cache_hits: Optional[int] = Field(
        default=None,
        description=(
            "Circuit breaker: auto-invalidate after this many consecutive "
            "identical cache hits. None disables. Suggested value: 3."
        ),
    )
    enable_feedback: bool = Field(
        default=False,
        description=(
            "Force toolkit mode (with feedback tool) even when no cache is "
            "configured. Implicitly true whenever any cache is set."
        ),
    )

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_genie_tool
        from dao_ai.tools.genie import GenieToolkit

        result = create_genie_tool(
            genie_room=self.genie_room,
            name=self.name,
            description=self.description,
            persist_conversation=self.persist_conversation,
            truncate_results=self.truncate_results,
            preserve_question=self.preserve_question,
            include_example_questions=self.include_example_questions,
            lru_cache_parameters=self.lru_cache,
            context_aware_cache_parameters=self.context_aware_cache,
            in_memory_context_aware_cache_parameters=self.in_memory_context_aware_cache,
            max_consecutive_cache_hits=self.max_consecutive_cache_hits,
            enable_feedback=self.enable_feedback,
        )
        if isinstance(result, GenieToolkit):
            return result.get_tools()
        return [result]


class AiSearchToolModel(BaseFunctionModel):
    """First-class AI Search tool that delegates to ``dao_ai.tools.create_ai_search_tool``.

    (Formerly ``VectorSearchToolModel``. Databricks rebranded Vector Search
    to AI Search; the old class name remains as an alias.)

    Equivalent to ``type: factory + name: dao_ai.tools.create_ai_search_tool``,
    but with typed fields. Exactly one of ``retriever`` or ``vector_store`` is
    required. Accepts either ``type: ai_search`` (new) or ``type: vector_search``
    (legacy) in YAML.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.VECTOR_SEARCH, FunctionType.AI_SEARCH] = Field(
        default=FunctionType.AI_SEARCH,
        description=(
            "Function type discriminator. Accepts 'ai_search' (preferred) "
            "or 'vector_search' (legacy alias)."
        ),
    )
    retriever: Optional[AiSearchRetrieverModel] = Field(
        default=None,
        description="Full retriever configuration with search parameters and reranking. Mutually exclusive with vector_store.",
    )
    vector_store: Optional[AiSearchVectorStoreModel] = Field(
        default=None,
        description="Direct AI Search vector-store reference (uses default search parameters). Mutually exclusive with retriever.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Tool name visible to the LLM.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Tool description shown to the LLM during function calling.",
    )

    @model_validator(mode="after")
    def _retriever_or_vector_store(self) -> Self:
        if self.retriever is None and self.vector_store is None:
            raise ValueError(
                "AiSearchToolModel requires exactly one of 'retriever' or 'vector_store'."
            )
        if self.retriever is not None and self.vector_store is not None:
            raise ValueError(
                "AiSearchToolModel cannot accept both 'retriever' and 'vector_store'."
            )
        return self

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_ai_search_tool

        return [
            create_ai_search_tool(
                retriever=self.retriever,
                vector_store=self.vector_store,
                name=self.name,
                description=self.description,
            )
        ]


# Backwards-compatible alias — Vector Search naming will eventually be
# deprecated. Both names refer to the same class.
VectorSearchToolModel = AiSearchToolModel


class LakebaseSearchToolModel(BaseFunctionModel):
    """First-class Lakebase retrieval tool that delegates to ``dao_ai.tools.create_lakebase_search_tool``.

    Retrieves from a Databricks Lakebase Postgres table using the
    ``lakebase_vector`` and (optionally) ``lakebase_text`` extensions.
    Exactly one of ``retriever`` or ``vector_store`` is required.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.LAKEBASE_SEARCH] = Field(
        default=FunctionType.LAKEBASE_SEARCH,
        description="Function type discriminator. Must be 'lakebase_search'.",
    )
    retriever: Optional[LakebaseRetrieverModel] = Field(
        default=None,
        description=(
            "Full Lakebase retriever configuration with search parameters. "
            "Mutually exclusive with ``vector_store``."
        ),
    )
    vector_store: Optional[LakebaseVectorStoreModel] = Field(
        default=None,
        description=(
            "Direct Lakebase table reference (uses default search parameters). "
            "Mutually exclusive with ``retriever``."
        ),
    )
    name: Optional[str] = Field(
        default=None,
        description="Tool name visible to the LLM.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Tool description shown to the LLM during function calling.",
    )

    @model_validator(mode="after")
    def _retriever_or_vector_store(self) -> Self:
        if self.retriever is None and self.vector_store is None:
            raise ValueError(
                "LakebaseSearchToolModel requires exactly one of 'retriever' or 'vector_store'."
            )
        if self.retriever is not None and self.vector_store is not None:
            raise ValueError(
                "LakebaseSearchToolModel cannot accept both 'retriever' and 'vector_store'."
            )
        return self

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_lakebase_search_tool

        return [
            create_lakebase_search_tool(
                retriever=self.retriever,
                vector_store=self.vector_store,
                name=self.name,
                description=self.description,
            )
        ]


class SqlToolModel(BaseFunctionModel):
    """First-class SQL tool that delegates to ``dao_ai.tools.sql.create_execute_statement_tool``.

    Equivalent to ``type: factory + name:
    dao_ai.tools.sql.create_execute_statement_tool``, but with typed fields. Runs
    a fixed SQL statement (optionally with bound ``params``) against a SQL
    warehouse or a Lakebase / Postgres database. Exactly one of ``warehouse`` or
    ``database`` is required. Parameter values are bound natively (``:name`` for a
    warehouse, ``%(name)s`` for Lakebase) — never interpolated into the SQL.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.SQL] = Field(
        default=FunctionType.SQL,
        description="Function type discriminator. Must be 'sql'.",
    )
    warehouse: Optional[WarehouseModel] = Field(
        default=None,
        description=(
            "SQL warehouse to run the statement against (use ':name' bind "
            "markers). Mutually exclusive with 'database'; exactly one is required."
        ),
    )
    database: Optional[DatabaseModel] = Field(
        default=None,
        description=(
            "Lakebase / Postgres database to run the statement against (use "
            "'%(name)s' bind markers). Mutually exclusive with 'warehouse'; "
            "exactly one is required."
        ),
    )
    statement: str = Field(
        description="SQL statement to execute.",
    )
    params: Optional[list[StatementParam]] = Field(
        default=None,
        description=(
            "Optional bound parameters. 'llm'-sourced params appear in the tool "
            "schema; 'context'-sourced params bind from the runtime Context."
        ),
    )
    name: Optional[str] = Field(
        default=None,
        description="Tool name visible to the LLM. Defaults to 'execute_sql_tool'.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Tool description shown to the LLM during function calling.",
    )

    @model_validator(mode="after")
    def _require_exactly_one_target(self) -> Self:
        if bool(self.warehouse) == bool(self.database):
            raise ValueError(
                "SqlToolModel requires exactly one of 'warehouse' or 'database'."
            )
        return self

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools.sql import create_execute_statement_tool

        return [
            create_execute_statement_tool(
                target=self.warehouse or self.database,
                statement=self.statement,
                params=self.params,
                name=self.name or "execute_sql_tool",
                description=self.description,
            )
        ]


class SearchToolModel(BaseFunctionModel):
    """First-class web search tool that delegates to ``dao_ai.tools.create_search_tool``.

    Equivalent to ``type: factory + name: dao_ai.tools.create_search_tool``.
    No configuration required.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.SEARCH] = Field(
        default=FunctionType.SEARCH,
        description="Function type discriminator. Must be 'search'.",
    )

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_search_tool

        return [create_search_tool()]


class AppToolModel(BaseFunctionModel):
    """First-class tool that calls a Databricks App as a tool.

    Supervisor API ``app`` contract. The target is a Databricks App,
    which may host an agent (ResponsesAgent or otherwise) or any other
    HTTP service that speaks OpenAI Responses or Chat Completions.

    Routes via ``DatabricksOpenAI(workspace_client=...)`` against the
    App's ``/v1/responses`` or ``/v1/chat/completions`` route (selected
    by ``api:``).

    Default ``api:`` behavior: when ``api`` is unset (None), the wire
    shape is discovered lazily on first invocation via
    ``GET <app_url>/agent/info`` (the MLflow Agent Server self-describe
    route). Falls back to ``"responses"`` if discovery returns no
    signal. Setting ``api:`` explicitly skips discovery entirely.

    For other target kinds see ``type: serving_endpoint`` (Model
    Serving), ``type: a2a`` (Google A2A protocol), ``type: mcp`` (MCP
    apps, ``mcp-`` prefix).
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.APP] = Field(
        default=FunctionType.APP,
        description="Function type discriminator. Must be 'app'.",
    )
    app: DatabricksAppModel = Field(
        description=(
            "Databricks App resource to call. Required. dao-ai dispatches "
            "via DatabricksOpenAI(workspace_client=...) using "
            "model='apps/<name>'. OBO is auto-derived from "
            "app.on_behalf_of_user. MCP apps (mcp- prefix) are rejected — "
            "use type: mcp instead."
        ),
    )
    api: Optional[Literal["responses", "completions"]] = Field(
        default=None,
        description=(
            "OpenAI API contract to use against the App:\n"
            "- 'responses' — POST /v1/responses (canonical for "
            "mlflow.agents ResponsesAgent deployments).\n"
            "- 'completions' — POST /v1/chat/completions (apps that "
            "expose the OpenAI Chat Completions route).\n"
            "- None (default) — lazy-probe GET <app_url>/agent/info on "
            "first invocation; fall back to 'responses' if discovery "
            "returns no signal. Setting this field skips discovery "
            "entirely."
        ),
    )
    name: Optional[str] = Field(
        default=None,
        description="Tool name visible to the LLM. Defaults to the app's name.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Tool description shown to the LLM during function calling.",
    )

    @model_validator(mode="after")
    def _reject_mcp_prefix(self) -> Self:
        if self.app.name.startswith("mcp-"):
            raise ValueError(
                f"AppToolModel: app '{self.app.name}' looks like an MCP "
                f"app (mcp- prefix). Use 'type: mcp' with 'app:' instead "
                f"of 'type: app'."
            )
        return self

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_app_dispatcher

        return [
            create_app_dispatcher(
                app=self.app,
                api=self.api,
                default_api="responses",
                name=self.name or self.app.name,
                description=self.description,
            )
        ]


class ServingEndpointToolModel(BaseFunctionModel):
    """First-class tool that calls a Databricks Model Serving endpoint.

    Supervisor API ``serving_endpoint`` contract. Covers FMAPI /
    Foundation Model API endpoints (Chat Completions) AND UC-registered
    agents deployed to Model Serving (ResponsesAgent). The wire shape
    is selected by ``api:``.

    Default ``api:`` behavior: when ``api`` is unset (None), the wire
    shape is discovered lazily on first invocation via
    ``WorkspaceClient.serving_endpoints.get(name).task``; mapped to
    ``"responses"`` for ``task="agent/v1/responses"`` and
    ``"completions"`` for ``task="llm/v1/chat"``. Falls back to
    ``"completions"`` if discovery returns no signal. Setting ``api:``
    explicitly skips discovery entirely.

    ``endpoint`` accepts two shapes:

    - **String / variable** (sugar) — just the endpoint name. dao-ai
      promotes it to a minimal ``InferenceEndpointModel`` internally.
    - **Full ``InferenceEndpointModel``** — when you need
      ``temperature``, ``max_tokens``, ``use_ai_gateway``, or
      ``on_behalf_of_user`` on the endpoint itself.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.SERVING_ENDPOINT] = Field(
        default=FunctionType.SERVING_ENDPOINT,
        description="Function type discriminator. Must be 'serving_endpoint'.",
    )
    endpoint: Union[InferenceEndpointModel, AnyVariable] = Field(
        description=(
            "Model Serving endpoint to call. Accepts either an endpoint "
            "name string (sugar) or a full InferenceEndpointModel with "
            "temperature / max_tokens / use_ai_gateway / on_behalf_of_user. "
            "For a UC-registered agent endpoint or a Knowledge Assistant "
            "this is the endpoint name (e.g., 'ka-customer-reviews', "
            "'hardware_store_dao'). For FMAPI it's the foundation model "
            "endpoint name (e.g., 'databricks-claude-sonnet-4-5')."
        ),
    )
    api: Optional[Literal["responses", "completions"]] = Field(
        default=None,
        description=(
            "OpenAI API contract to use against the endpoint:\n"
            "- 'responses' — ChatDatabricks(use_responses_api=True). "
            "Required for UC-registered agent endpoints (task = "
            "'agent/v1/responses').\n"
            "- 'completions' — ChatDatabricks (legacy default). "
            "Required for FMAPI endpoints (task = 'llm/v1/chat').\n"
            "- None (default) — lazy-probe "
            "serving_endpoints.get(name).task on first invocation and "
            "map to the right contract; fall back to 'completions' if "
            "discovery returns no signal. Setting this field skips "
            "discovery entirely."
        ),
    )
    name: Optional[str] = Field(
        default=None,
        description="Tool name visible to the LLM. Defaults to the endpoint name.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Tool description shown to the LLM during function calling.",
    )
    on_behalf_of_user: Optional[bool] = Field(
        default=None,
        description=(
            "If True, call the endpoint on behalf of the calling user by "
            "forwarding their bearer token. If False or None, the agent's "
            "service principal calls. Ignored when ``endpoint`` is a full "
            "InferenceEndpointModel that already sets on_behalf_of_user."
        ),
    )

    def _resolved_llm(self) -> "InferenceEndpointModel":
        """Normalize the endpoint field to an InferenceEndpointModel.

        String / variable inputs are promoted to a minimal model. A
        full InferenceEndpointModel is returned as-is.
        """
        if isinstance(self.endpoint, InferenceEndpointModel):
            return self.endpoint
        endpoint_name: str = str(value_of(self.endpoint))
        return InferenceEndpointModel(
            name=endpoint_name,
            on_behalf_of_user=bool(self.on_behalf_of_user)
            if self.on_behalf_of_user is not None
            else None,
        )

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_serving_endpoint_dispatcher

        llm: InferenceEndpointModel = self._resolved_llm()
        return [
            create_serving_endpoint_dispatcher(
                llm=llm,
                api=self.api,
                default_api="completions",
                name=self.name or llm.name,
                description=self.description,
            )
        ]


class A2AToolModel(BaseFunctionModel):
    """First-class tool that calls a Google A2A v0.3 agent.

    Equivalent to ``type: factory + name: dao_ai.tools.create_a2a_agent_tool``,
    but with typed fields. Two configuration modes match the underlying
    factory:

    - **Mode 1 (external A2A)**: set ``endpoint`` + ``auth_type``
      (default ``bearer``). Use for Vertex AI Agent Engine, Crew.ai, Google
      ADK, or any third-party A2A agent outside Databricks Apps.
    - **Mode 2 (Databricks App)**: set ``app`` referencing a
      ``DatabricksAppModel``. ``auth_type`` defaults from
      ``app.on_behalf_of_user``: True → ``forwarded_user_token``;
      False/None → ``databricks_app_sp``.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    type: Literal[FunctionType.A2A] = Field(
        default=FunctionType.A2A,
        description="Function type discriminator. Must be 'a2a'.",
    )
    endpoint: Optional[AnyVariable] = Field(
        default=None,
        description=(
            "Base URL of the A2A agent (e.g., 'https://agent.example.com'). "
            "Mode 1 entry point. Mutually exclusive with ``app`` (``app`` wins "
            "if both are provided)."
        ),
    )
    app: Optional[DatabricksAppModel] = Field(
        default=None,
        description=(
            "Databricks App resource for the remote dao-ai app. Mode 2 entry "
            "point. Endpoint and auth mode derived from the bound app."
        ),
    )
    auth: Optional[AnyVariable] = Field(
        default=None,
        description=(
            "Auth material resolved to a string. Required for ``bearer`` and "
            "``gcp_service_account``. Ignored by ``none``, "
            "``forwarded_user_token``, and ``databricks_app_sp``."
        ),
    )
    auth_type: Optional[
        Literal[
            "bearer",
            "gcp_service_account",
            "none",
            "forwarded_user_token",
            "databricks_app_sp",
        ]
    ] = Field(
        default=None,
        description=(
            "Auth mode. Default is 'bearer' in Mode 1 and derived from "
            "``app.on_behalf_of_user`` in Mode 2. Passing this in Mode 2 "
            "overrides the app-derived default."
        ),
    )
    streaming: bool = Field(
        default=True,
        description=(
            "If true (default) the A2A client negotiates streaming; responses "
            "are still aggregated internally and returned as one string."
        ),
    )
    timeout_seconds: int = Field(
        default=300,
        description="httpx client timeout in seconds.",
    )
    card_path: Optional[str] = Field(
        default=None,
        description=(
            "Primary agent-card discovery path relative to ``endpoint``. "
            "Defaults to the current spec's '/.well-known/agent-card.json'."
        ),
    )
    card_fallback_path: Optional[str] = Field(
        default=None,
        description=(
            "Fallback agent-card path if the primary 404s. Defaults to the "
            "pre-1.0 spec's '/.well-known/agent.json'. Pass empty string to "
            "disable fallback."
        ),
    )
    user_id: Optional[AnyVariable] = Field(
        default=None,
        description=(
            "Static value forwarded as ``Message.metadata['dao_ai.user_id']``. "
            "If omitted, falls back to ``runtime.context.user_id``."
        ),
    )
    extra_metadata: Optional[dict[str, AnyVariable]] = Field(
        default=None,
        description="Static metadata merged into ``Message.metadata`` on every call.",
    )
    name: Optional[AnyVariable] = Field(
        default=None,
        description="Tool name visible to the LLM. Defaults to 'a2a_agent'.",
    )
    description: Optional[AnyVariable] = Field(
        default=None,
        description="Tool description shown to the LLM during function calling.",
    )

    @model_validator(mode="after")
    def _endpoint_or_app(self) -> Self:
        if self.endpoint is None and self.app is None:
            raise ValueError("A2AToolModel requires one of 'endpoint' or 'app'.")
        return self

    def as_tools(self, **kwargs: Any) -> Sequence[RunnableLike]:
        from dao_ai.tools import create_a2a_agent_tool

        factory_kwargs: dict[str, Any] = {
            "endpoint": self.endpoint,
            "app": self.app,
            "auth": self.auth,
            "auth_type": self.auth_type,
            "streaming": self.streaming,
            "timeout_seconds": self.timeout_seconds,
            "user_id": self.user_id,
            "extra_metadata": self.extra_metadata,
            "name": self.name,
            "description": self.description,
        }
        if self.card_path is not None:
            factory_kwargs["card_path"] = self.card_path
        if self.card_fallback_path is not None:
            factory_kwargs["card_fallback_path"] = (
                self.card_fallback_path if self.card_fallback_path else None
            )
        return [create_a2a_agent_tool(**factory_kwargs)]


AnyTool: TypeAlias = (
    Union[
        PythonFunctionModel,
        FactoryFunctionModel,
        InlineFunctionModel,
        UnityCatalogFunctionModel,
        McpFunctionModel,
        GenieToolModel,
        AiSearchToolModel,
        LakebaseSearchToolModel,
        SqlToolModel,
        SearchToolModel,
        AppToolModel,
        ServingEndpointToolModel,
        A2AToolModel,
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


class PromptModel(BaseModel):
    """A named, reusable prompt defined inline in configuration.

    Prompts are first-class config objects so they can be declared once and
    referenced as YAML anchors/aliases by agents, guardrails, and supervisors.
    The template text is carried inline; there is no registry round-trip.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    name: str = Field(
        description="Identifier for the prompt, used as a label in logs and traces.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the prompt.",
    )
    template: str = Field(
        description="The prompt template text, with optional {variable} placeholders.",
    )

    @property
    def jinja_template(self) -> str:
        """Return the template in Jinja2 format (with {{ }} variables).

        Unlike ``template`` which uses single-brace Python format, this
        property ensures the known MLflow judge variables (e.g.
        ``{{ inputs }}``, ``{{ outputs }}``) required by MLflow judges use
        double-brace Jinja2 syntax. Templates already written in single-brace
        format for those variables are converted automatically.
        """
        import re

        raw_template: str = self.template

        # Convert single-brace MLflow judge variables to Jinja2 double-brace
        # format when the template was written in single-brace format.
        _JUDGE_VARS = ("inputs", "outputs", "trace", "expectations", "conversation")
        for var in _JUDGE_VARS:
            # Match {var} but NOT {{var}} (already Jinja2)
            raw_template = re.sub(
                r"(?<!\{)\{" + var + r"\}(?!\})",
                "{{ " + var + " }}",
                raw_template,
            )

        return raw_template


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


def _accepted_keys(field_name: str, field: Any) -> set[str]:
    """Every YAML key that populates ``field`` — its name plus any alias."""
    keys: set[str] = {field_name}
    alias: Any = getattr(field, "validation_alias", None)
    if isinstance(alias, str):
        keys.add(alias)
    elif isinstance(alias, AliasChoices):
        keys.update(choice for choice in alias.choices if isinstance(choice, str))
    return keys


@cache
def _genie_room_only_keys() -> frozenset[str]:
    """Keys that can only belong to a ``GenieRoomModel``, never an endpoint.

    Derived from the two models' fields — including each field's accepted
    aliases, which is what contributes ``agent_id`` (the alias of
    ``space_id``) — so the set cannot drift as fields are added to either
    class. Shared keys (``name``, ``description``, ``on_behalf_of_user``, the
    auth fields) are excluded by construction: they cannot disambiguate.
    """
    endpoint_keys: set[str] = set()
    for field_name, field in InferenceEndpointModel.model_fields.items():
        endpoint_keys |= _accepted_keys(field_name, field)

    room_keys: set[str] = set()
    for field_name, field in GenieRoomModel.model_fields.items():
        room_keys |= _accepted_keys(field_name, field)

    return frozenset(room_keys - endpoint_keys)


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
    model: InferenceEndpointModel | GenieAgentModel = Field(
        description=(
            "Reasoning model for this agent. Either an ``InferenceEndpointModel`` "
            "(a Model Serving chat endpoint) or a ``GenieAgentModel`` (a Genie "
            "Agent used as a streaming brain — typically with ``tools: []`` to "
            "make this a Genie specialist sub-agent)."
        ),
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
    call_limit: Optional[int | ModelCallLimitModel] = Field(
        default=None,
        description=(
            "Shortcut to cap how many model (LLM) calls this agent may make. A "
            "bare integer sets run_limit (per-invocation) with "
            "exit_behavior='end'; an object gives full control over "
            "run_limit/thread_limit/exit_behavior. Registers a "
            "ModelCallLimitMiddleware for this agent — the discoverable, "
            "co-located equivalent of hand-wiring "
            "``create_model_call_limit_middleware`` into ``middleware``. The "
            "model-call analogue of a tool's ``call_limit``; complements "
            "``recursion_limit`` (the graph-superstep backstop)."
        ),
    )

    @field_validator("call_limit", mode="before")
    @classmethod
    def _normalize_call_limit(cls, value: Any) -> Any:
        """Normalize a bare integer shortcut into a ModelCallLimitModel dict.

        Runs before union validation so a bare int becomes ``run_limit``.
        ``bool`` is a subclass of ``int`` but is never a valid limit, so reject
        it explicitly rather than silently coercing ``True`` to ``run_limit=1``.
        """
        if isinstance(value, bool):
            raise ValueError(
                "call_limit must be a positive integer or an object, not a bool."
            )
        if isinstance(value, int):
            return {"run_limit": value}
        return value

    handoff: Optional[bool] = Field(
        default=None,
        description=(
            "Only meaningful when ``model`` is a Genie space (``GenieAgentModel``) "
            "used as a worker under a SUPERVISOR. A Genie brain runs its tool loop "
            "server-side and never emits a client tool call; under a supervisor, "
            "dao-ai makes it hand control back deterministically — like every other "
            "worker — by injecting a ``handoff_to_supervisor`` tool call after it "
            "answers, so the supervisor can chain another agent in the same turn. "
            "LLM-free. Under a supervisor this is the DEFAULT (``None`` and ``true`` "
            "both hand back); set ``handoff: false`` to opt OUT and make the brain a "
            "terminal graph sink (it answers and the turn ends; the supervisor only "
            "re-routes on the next turn). It has NO effect outside the supervisor "
            "pattern: a standalone single-agent Genie has nothing to hand back to, "
            "and swarm routing uses an ``is_deterministic`` handoff route at the "
            "graph level instead. Ignored for non-Genie models (they hand off "
            "through their own tool loop)."
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
    internal: bool = Field(
        default=False,
        description=(
            "Marks this agent as an internal / plumbing agent. Its "
            "``AIMessage`` outputs are filtered out of the conversation "
            "history passed to NON-internal (customer-facing) agents' LLM "
            "context. Other internal agents still see them — e.g., a "
            "planner can still read a supervisor's intent classification. "
            "The agent itself always sees its own prior messages on "
            "re-entry.\n\n"
            "The agent still runs, still writes to shared state, and still "
            "appears in MLflow traces — only downstream customer-facing "
            "agents' LLM view is scoped away, so internal-plumbing text "
            "(intent classifications, routing decisions, working notes) "
            "cannot leak as if it were a customer-facing response. "
            "Typical use: swarm supervisors, planners, routers; anything "
            "whose output is intended for downstream consumption rather "
            "than the customer. Composer / response-formatter agents "
            "should stay at the default ``False``."
        ),
    )

    @field_validator("model", mode="before")
    @classmethod
    def _wrap_bare_genie_room(cls, value: Any) -> Any:
        """Auto-wrap a bare Genie room assigned to ``model`` into a ``GenieAgentModel``.

        Ergonomic sugar so a registered room anchor can be assigned directly::

            model: *retail_genie_room          # bare room  → GenieAgentModel
            model: {genie_room: *room, timeout_seconds: 600}  # explicit wrapper
            model: {name: databricks-claude-sonnet-4-5}       # LLM (untouched)

        Coercion is by shape, not by smart-union guessing:

        * a :class:`GenieRoomModel` instance, or
        * a dict that is not already the wrapper shape (no ``genie_room`` key)
          and carries at least one key only a room can have — see
          :func:`_genie_room_only_keys`, which covers ``space_id``/``agent_id``
          as well as the provisioning fields (``table_sources``, ``warehouse``,
          ``text_instructions``, …) that a managed room is declared with,

        is rewritten to ``{"genie_room": value}``. Everything else is left
        untouched for the normal union to resolve.

        A dict of only *shared* keys (notably ``{"name": ...}``, valid for both
        classes) stays ambiguous and resolves to ``InferenceEndpointModel``;
        :meth:`AppConfig._validate_genie_agent_rooms_registered` catches that
        case by cross-referencing ``resources.genie_rooms``, since shape alone
        cannot decide it.

        The bare form uses the default ``timeout_seconds``; use the explicit
        ``{genie_room: ...}`` wrapper to set invocation knobs.
        """
        if isinstance(value, GenieRoomModel):
            return {"genie_room": value}
        if isinstance(value, dict) and "genie_room" not in value:
            if _genie_room_only_keys() & value.keys():
                return {"genie_room": value}
        return value

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

    @model_validator(mode="after")
    def validate_genie_brain_bindings(self) -> Self:
        """Reject config a Genie brain would silently discard.

        ``GenieAgentChatModel.bind_tools`` returns the model unchanged — Genie
        Agent Mode runs its own tool loop server-side and exposes no way to
        declare client tools — so declared ``tools`` are registered in the
        agent's ToolNode and can never be called, and a ``response_format``
        binding is dropped on the floor. Either config builds cleanly and
        surfaces only as a wrong answer at runtime, so both are refused here,
        where the message can name the agent and the offending tools.
        """
        if not isinstance(self.model, GenieAgentModel):
            if self.handoff is not None:
                raise ValueError(
                    f"Agent '{self.name}' sets 'handoff' but its model is not a "
                    f"Genie space. 'handoff' opts a Genie brain into deterministic "
                    f"handback to a supervisor; a non-Genie agent already hands off "
                    f"through its own tool loop. Remove 'handoff'."
                )
            return self

        if self.tools:
            tool_names: str = ", ".join(tool.name for tool in self.tools)
            raise ValueError(
                f"Agent '{self.name}' uses a Genie space as its model, which "
                f"runs its own tool loop server-side and can never call a "
                f"client tool — but it declares tools: {tool_names}. Remove "
                f"them, or give the agent an LLM model and reach Genie through "
                f"a `type: genie` tool instead."
            )

        if self.response_format is not None:
            raise ValueError(
                f"Agent '{self.name}' uses a Genie space as its model, which "
                f"streams narrative markdown and cannot be bound to a "
                f"response_format. Remove response_format, or give the agent "
                f"an LLM model and reach Genie through a `type: genie` tool "
                f"instead."
            )

        return self

    def as_runnable(self) -> RunnableLike:
        from dao_ai.nodes import create_agent_node

        return create_agent_node(self)

    def as_responses_agent(self) -> ResponsesAgent:
        from dao_ai.models import create_responses_agent

        graph: CompiledStateGraph = self.as_runnable()
        return create_responses_agent(
            graph,
            tool_models=self.tools,
        )


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

    A single entry expresses either a **single-target** handoff or a
    **parallel fan-out cohort**. These two shapes are mutually exclusive on
    the same entry.

    Single-target handoff — set ``agent`` and optionally ``is_deterministic``:

    - **Agentic** (default): a handoff tool is created for the target agent
      and the LLM decides when to invoke it via a tool call.
    - **Deterministic** (``is_deterministic=True``): control always transfers
      to this agent after the source agent completes its turn, without LLM
      tool-call routing.

    Parallel fan-out cohort — set ``agents`` (list) and ``join``:

    - A per-sibling parallel handoff tool is created for each entry in
      ``agents``. When the LLM invokes multiple of these tools in a single
      turn, LangGraph schedules the targeted siblings in the same superstep
      (true concurrent execution). All siblings converge on the shared
      ``join`` agent via a static edge in the parent graph; superstep
      semantics run the join exactly once after every fired sibling
      completes.

    Example YAML — mixed shapes on one source::

        handoffs:
          triage_agent:
            - agents:
                - pricing_agent
                - inventory_agent
                - policy_agent
              join: synthesizer_agent            # shared join for the cohort
            - escalation_agent                    # agentic single-target
            - agent: emergency_agent
              is_deterministic: true              # deterministic single-target

    Validation (enforced at config load time):

    - ``agent`` and ``agents`` are mutually exclusive; exactly one must be set.
    - ``agents`` requires ``join``; ``join`` requires ``agents``.
    - ``is_deterministic`` is only valid on single-target entries.
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    agent: Optional[AgentModel | str] = Field(
        default=None,
        description=(
            "Single-target handoff. Set this OR ``agents`` (mutually exclusive)."
        ),
    )
    is_deterministic: bool = Field(
        default=False,
        description=(
            "Single-target only. When true, control always transfers to ``agent`` "
            "after the source completes its turn, without LLM tool-call routing. "
            "When false (default), a handoff tool is created and the LLM decides "
            "when to invoke it. Ignored on cohort entries (``agents``)."
        ),
    )
    agents: Optional[list[AgentModel | str]] = Field(
        default=None,
        description=(
            "Parallel fan-out cohort — the sibling agents the source may invoke "
            "concurrently in a single LLM turn. Requires ``join``. Mutually "
            "exclusive with ``agent``."
        ),
    )
    join: Optional[AgentModel | str] = Field(
        default=None,
        description=(
            "Shared join agent for a parallel fan-out cohort. All fired "
            "siblings in ``agents`` converge here via a static edge; the join "
            "runs exactly once after all siblings complete. Required when "
            "``agents`` is set; must be omitted otherwise."
        ),
    )

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Enforce single-target XOR cohort shape.

        - Exactly one of ``agent`` or ``agents`` must be set.
        - ``agents`` requires ``join``; ``join`` requires ``agents``.
        - ``is_deterministic`` is not meaningful on cohort entries.
        - Cohort must have at least two siblings (a "cohort" of one is a
          plain single-target handoff — reject as a config error so the
          user picks the right shape).
        - Cohort siblings must be distinct — duplicate names route the
          same handoff tool twice.
        - The join must not also appear among the cohort siblings.
        """
        single_target: bool = self.agent is not None
        cohort: bool = self.agents is not None

        if single_target and cohort:
            raise ValueError(
                "Handoff entry cannot set both ``agent`` and ``agents``. "
                "Use ``agent`` for a single-target handoff, or ``agents`` + "
                "``join`` for a parallel fan-out cohort."
            )
        if not single_target and not cohort:
            raise ValueError(
                "Handoff entry must set either ``agent`` (single-target) or "
                "``agents`` + ``join`` (parallel fan-out cohort)."
            )

        if cohort:
            if self.join is None:
                raise ValueError(
                    "Handoff cohort (``agents``) requires ``join`` naming the "
                    "shared reducer agent all siblings converge on."
                )
            if self.is_deterministic:
                raise ValueError(
                    "``is_deterministic`` is not meaningful on a cohort entry "
                    "(``agents``). The cohort's ``join`` is always reached "
                    "deterministically after every fired sibling completes."
                )
            if len(self.agents) < 2:
                raise ValueError(
                    "Handoff cohort (``agents``) must have at least two "
                    "siblings. For a single handoff use ``agent`` instead."
                )
            sibling_names: list[str] = [
                a.name if isinstance(a, AgentModel) else a for a in self.agents
            ]
            if len(set(sibling_names)) != len(sibling_names):
                dupes: list[str] = [
                    n for n in sibling_names if sibling_names.count(n) > 1
                ]
                raise ValueError(
                    f"Handoff cohort siblings must be distinct; duplicates: "
                    f"{sorted(set(dupes))}."
                )
            join_name: str = (
                self.join.name if isinstance(self.join, AgentModel) else self.join
            )
            if join_name in sibling_names:
                raise ValueError(
                    f"Handoff cohort join '{join_name}' cannot also be a "
                    "sibling. The join must be distinct so LangGraph can fan "
                    "siblings into the join without a self-edge."
                )
        else:
            # single-target: ``join`` is not meaningful
            if self.join is not None:
                raise ValueError(
                    "``join`` is only valid on a cohort entry (``agents``). "
                    "For a single-target handoff, omit ``join``."
                )
        return self


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

    @staticmethod
    def _target_name(target: "AgentModel | str") -> str:
        """Return the agent name for a handoff target, regardless of shape."""
        return target.name if isinstance(target, AgentModel) else target

    @staticmethod
    def _iter_edges(
        entry: "AgentModel | str | HandoffRouteModel",
    ) -> list[tuple[str, str]]:
        """Return the ``(target_name, kind)`` edges implied by one handoff entry.

        A single-target entry produces one edge. A cohort entry produces N+1
        edges: one ``parallel`` edge per sibling in ``agents`` (source →
        sibling) and one ``deterministic`` edge for ``join`` (source →
        join, though at runtime the join is reached *through* the siblings —
        we surface it here so the cycle detector treats it uniformly).

        ``kind ∈ {agentic, deterministic, parallel}``.
        """
        if isinstance(entry, HandoffRouteModel):
            if entry.agents is not None:
                edges: list[tuple[str, str]] = [
                    (SwarmModel._target_name(a), "parallel") for a in entry.agents
                ]
                edges.append((SwarmModel._target_name(entry.join), "deterministic"))
                return edges
            kind: str = "deterministic" if entry.is_deterministic else "agentic"
            return [(SwarmModel._target_name(entry.agent), kind)]
        return [(SwarmModel._target_name(entry), "agentic")]

    @model_validator(mode="after")
    def validate_parallel_cohort_shape(self) -> Self:
        """Cross-entry validation for parallel fan-out cohorts.

        Per-entry shape (``agents`` requires ``join``, at least two distinct
        siblings, no sibling == join, etc.) is enforced by
        ``HandoffRouteModel.validate_shape``. This validator only handles
        invariants that span multiple entries in the swarm:

        - Cohort self-reference: source ∉ its own ``agents`` (a source can't
          fan out to itself — the cohort would deadlock its own turn).
        - Cross-cohort collision: an agent can't be a sibling in two cohorts
          with *different* joins.
        - Nested fan-out: a sibling can't also be the source of another
          cohort. Nested fan-out is not supported — the outer join would
          be unreachable once the inner cohort fires.
        """
        if not self.handoffs:
            return self

        sibling_to_join: dict[str, str] = {}
        parallel_sources: set[str] = set()

        for source, targets in self.handoffs.items():
            if not targets:
                continue
            for entry in targets:
                if not isinstance(entry, HandoffRouteModel) or entry.agents is None:
                    continue
                # Cohort entry.
                sibling_names: list[str] = [self._target_name(a) for a in entry.agents]
                join_name: str = self._target_name(entry.join)
                if source in sibling_names:
                    raise ValueError(
                        f"Agent '{source}' cannot appear in its own parallel "
                        f"cohort (``agents``). A source can't fan out to itself."
                    )
                if source == join_name:
                    raise ValueError(
                        f"Agent '{source}' cannot be its own cohort ``join``. "
                        "Pick a distinct join agent."
                    )
                parallel_sources.add(source)
                for sibling in sibling_names:
                    existing_join: str | None = sibling_to_join.get(sibling)
                    if existing_join is not None and existing_join != join_name:
                        raise ValueError(
                            f"Agent '{sibling}' is a sibling in cohorts with "
                            f"different join targets ('{existing_join}' and "
                            f"'{join_name}'). A sibling may belong to at most "
                            "one cohort."
                        )
                    sibling_to_join[sibling] = join_name

        # Reject nested fan-out — a sibling that is itself a cohort source.
        # Allowing this silently would leave the outer join unreachable
        # whenever the inner cohort fires.
        nested: set[str] = set(sibling_to_join) & parallel_sources
        if nested:
            offender: str = sorted(nested)[0]
            raise ValueError(
                f"Agent '{offender}' is both a cohort sibling and the source "
                "of its own cohort. Nested parallel fan-out is not supported "
                "— the outer join would be unreachable once the inner cohort "
                "fires. Restructure so any agent is at most one of: cohort "
                "source, cohort sibling, cohort join."
            )
        return self

    @model_validator(mode="after")
    def validate_no_deterministic_handoff_in_cycle(self) -> Self:
        """Reject swarm configs where a deterministic or parallel edge participates in a cycle.

        A deterministic handoff transfers control unconditionally on every
        traversal. A parallel handoff, once the LLM invokes it, also transfers
        control unconditionally — and the shared join edge from every parallel
        sibling is a static edge in the parent graph. Both edge classes are
        treated as "unconditional" for cycle detection: if any cycle in the
        handoff graph contains at least one such edge, the workflow can run
        forever, because the loop is closed by the unconditional edge and the
        agentic edges that make up the rest of the cycle only need an LLM
        whose prompt occasionally fires the handoff tool to keep it going.

        This validator runs at config load time and rejects the pattern with
        a clear cycle path so the user can break or reconfigure the cycle
        before any compute is spent.

        Allowed:
          * No cycle (``A -det-> B`` with no path back to A).
          * A cycle of all-agentic edges (LLMs can choose to terminate).

        Rejected:
          * Any cycle containing at least one deterministic OR parallel edge.
        """
        if not self.handoffs:
            return self

        # Build edge list: list[(source, target, kind)] where kind is
        # "deterministic", "parallel", or "agentic". Cohort entries expand
        # to N parallel edges (source -> sibling) plus one deterministic
        # edge for the join, so the cycle detector treats all
        # unconditional edges uniformly.
        edges: list[tuple[str, str, str]] = []
        for source, targets in self.handoffs.items():
            if not targets:
                continue
            for entry in targets:
                for target_name, kind in self._iter_edges(entry):
                    # Skip self-references; per-entry validators already
                    # flag them.
                    if target_name == source:
                        continue
                    edges.append((source, target_name, kind))

        # Adjacency list keyed by source -> list[(target, kind)]
        adj: dict[str, list[tuple[str, str]]] = {}
        for u, v, kind in edges:
            adj.setdefault(u, []).append((v, kind))

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
                for nxt, _kind in adj.get(node, []):
                    if nxt == goal:
                        return path + [nxt]
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, path + [nxt]))
            return None

        for u, v, kind in edges:
            if kind == "agentic":
                continue
            return_path: list[str] | None = find_path(v, u)
            if return_path is None:
                continue
            full_path: list[str] = [u] + return_path
            edge_str = f"{u} =[{kind}]=> " + " -> ".join(full_path[1:])
            raise ValueError(
                f"Swarm has a {kind} handoff inside a cycle: "
                f"{edge_str}. {kind.capitalize()} edges fire unconditionally on every "
                "traversal, so any path back to the source forms a runaway loop. "
                "Either remove the return path or make the "
                f"{kind} edge agentic."
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
            "Inline string or ``PromptModel`` (resolved via ``make_prompt``)."
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
    interrupt_model: Optional[InferenceEndpointModel] = Field(
        default=None,
        description=(
            "LLM used to parse a user's natural-language Human-in-the-Loop "
            "interrupt response (approve/reject/edit) into structured "
            "decisions. When unset, dao-ai derives it from the router: the "
            "supervisor's model, else the swarm default agent's model, else "
            "the first declared agent's model. Set this to pin a specific "
            "endpoint for interrupt parsing (e.g. a cheaper/faster model). "
            "The resolved choice and its source are logged at build time."
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
    """Compute size controlling deployment resources.

    Shared by both deployment targets. Model Serving accepts Small/Medium/Large
    natively; XLarge is an Apps-only tier (Databricks Apps supports MEDIUM,
    LARGE, XLARGE and has no Small tier). When XLarge is used with the Model
    Serving target it is clamped to Large (its largest size).
    """

    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"
    XLARGE = "XLarge"


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

    When configured on ``AppModel``, traces are stored in UC Delta tables via
    ``mlflow.set_experiment(experiment_id=..., trace_location=UnityCatalog(...))``.
    MLflow materializes four Delta tables per (schema, prefix) pair:
    ``<catalog>.<schema>.<prefix>_otel_{spans,logs,metrics,annotations}``.

    When ``table_prefix`` is not explicitly set, MLflow falls back to the
    ``experiment_id`` at runtime — producing a per-experiment table set
    out of the box (e.g. ``2931483616868130_otel_spans``). Set
    ``table_prefix`` to namespace tables when multiple experiments share
    a single UC schema and you want each agent's traces in its own table
    set (e.g. ``table_prefix: sales_genie`` yields
    ``sales_genie_otel_spans``).

    Deploy ordering:

    1. ``_link_experiment_trace_location`` calls
       ``mlflow.set_experiment(trace_location=UnityCatalog(...))``,
       which materializes the four OTEL tables on the configured
       warehouse (auto-starts STOPPED warehouses; waits up to 1200s).
    2. ``build_auth_policy`` then includes the four tables in the
       deployed model's ``SystemAuthPolicy.resources`` via
       :meth:`as_resources`, so ``agents.deploy`` auto-grants the
       Model Serving SP USE_CATALOG / USE_SCHEMA / SELECT / MODIFY
       on each table — no manual post-deploy ``GRANT`` step required.
    """

    model_config = ConfigDict(
        use_enum_values=True, extra="forbid", populate_by_name=True
    )
    schema_model: SchemaModel = Field(
        alias="schema",
        description="Unity Catalog schema (catalog.schema) where OTEL trace tables are stored.",
    )
    warehouse: Union[WarehouseModel, AnyVariable] = Field(
        description="SQL warehouse for creating views and querying traces. "
        "Accepts a WarehouseModel reference, a bare warehouse-id string, or "
        "an AnyVariable (env var / secret / composite / primitive) — useful "
        "when the warehouse id is environment-specific or held in a secret "
        "scope rather than baked into the YAML.",
    )
    table_prefix: Optional[AnyVariable] = Field(
        default=None,
        description=(
            "Optional table-prefix passed into MLflow's ``UnityCatalog`` "
            "trace location. When set, OTEL trace tables are named "
            "``<catalog>.<schema>.<table_prefix>_otel_{spans,logs,metrics}`` "
            "instead of the unprefixed default — useful when multiple agents "
            "share a single trace schema and you want each agent's traces in "
            "its own table set. Leave unset (None) to fall back to the shared "
            "default (all experiments writing to this schema share one table "
            "set, distinguished by experiment_id rows inside the tables)."
        ),
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
        """Resolve warehouse to a warehouse ID string.

        Handles all warehouse field shapes: WarehouseModel (resolve through
        the embedded warehouse_id), AnyVariable (env/secret/composite/
        primitive — resolved via value_of), or plain str (passed through
        by value_of unchanged).
        """
        if isinstance(self.warehouse, WarehouseModel):
            return value_of(self.warehouse.warehouse_id)
        return value_of(self.warehouse)

    @property
    def catalog_name(self) -> str:
        return value_of(self.schema_model.catalog_name)

    @property
    def schema_name(self) -> str:
        return value_of(self.schema_model.schema_name)

    @property
    def resolved_table_prefix(self) -> Optional[str]:
        """Resolve ``table_prefix`` (if set) to a concrete string via value_of.

        Returns None when no prefix is configured. When None is passed to
        MLflow's ``UnityCatalog`` constructor (or omitted), MLflow falls
        back to the literal default ``mlflow_experiment_trace`` — see
        ``mlflow.entities.trace_location._UC_SCHEMA_DEFAULT_SPANS_TABLE_NAME``
        (``= "mlflow_experiment_trace_otel_spans"``) and the matching logs
        constant. The tables end up at
        ``<catalog>.<schema>.mlflow_experiment_trace_otel_{spans,logs,metrics}``
        and are SHARED across every experiment linked to that schema with no
        prefix; rows are partitioned by ``experiment_id`` column. Set
        ``table_prefix`` to get a dedicated table set per agent.
        """
        if self.table_prefix is None:
            return None
        resolved = value_of(self.table_prefix)
        return resolved if resolved else None

    def as_resources(self) -> Sequence[DatabricksResource]:
        """OTEL trace tables intentionally NOT declared as auth_policy resources.

        MLflow trace persistence from Model Serving endpoints created via
        ``agents.deploy`` is a known Databricks platform limitation:

        * The endpoint's runtime tracing writer uses the auto-generated
          per-endpoint system SP for authentication.
        * That SP is not exposed by any Databricks API and cannot be
          granted UC permissions directly — the platform rejects the
          grant.
        * ``agents.deploy(resources=…)`` declarations trigger auto-auth
          only for the inference-time ``generate-temporary-credentials``
          path (vector-search reads, UC function calls, etc.) — not for
          the tracing writer's static-credential path. Empirically
          verified: declaring the OTEL tables as ``DatabricksTable``
          resources does NOT propagate any grants to the tracing SP.

        On Databricks Apps, trace persistence works — the App's runtime
        SP is a normal user-visible SP (``fresh_app.service_principal_client_id``)
        which dao-ai grants explicitly after ``apps.create_and_wait``
        via ``_grant_trace_permissions_to_principal``.

        Return ``[]`` here so nothing is added to the Model Serving
        ``system_auth_policy.resources`` — declaring the tables would
        add register-time overhead without helping trace persistence.
        """
        return []


class ExperimentModel(BaseModel):
    """Reference an MLflow experiment by name and/or id.

    Mirrors the ``WarehouseModel`` idiom (`config.py:1310`): both ``name``
    and ``id`` are optional but at least one is required. Actual resolution
    is lazy — the id-from-name lookup (and optional create) happens in
    :meth:`ensure_resolved`, not at config-load time, so this model is safe
    to construct in environments without an active MLflow client.

    Precedence:
        * ``id`` wins when both are set — used directly, no MLflow lookup.
        * ``name`` triggers ``mlflow.get_experiment_by_name(name)``; if
          missing and ``create_if_not_exists`` is True (default), the
          experiment is created and its id captured back into ``self.id``.

    Independent of ``service_principal`` and ``trace_location`` — any
    combination of the three is a valid config. When ``AppModel.experiment``
    is omitted entirely, dao-ai falls back to the historical default:
    ``/Users/<deployer_email>/<app.name>`` (create-if-missing).
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    name: Optional[AnyVariable] = Field(
        default=None,
        description=(
            "Full workspace path of the MLflow experiment "
            "(e.g. ``/Shared/team/agent_traces``). When ``id`` is unset, "
            "dao-ai resolves the id from this name at deploy time — "
            "creating the experiment if missing (see ``create_if_not_exists``)."
        ),
    )
    id: Optional[AnyVariable] = Field(
        default=None,
        description=(
            "Numeric MLflow experiment id. When set, takes precedence over "
            "``name`` and is used verbatim (no MLflow lookup, no create). "
            "Use when an admin has pre-provisioned + linked the experiment "
            "and the deployer knows the id."
        ),
    )
    create_if_not_exists: bool = Field(
        default=True,
        description=(
            "When True (default), ``ensure_resolved()`` creates the "
            "experiment at ``name`` when ``id`` is unset and the name "
            "does not resolve. Set to False when an admin has already "
            "provisioned the experiment and the deployer lacks create "
            "rights on the parent workspace path."
        ),
    )

    _resolved: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def require_name_or_id(self) -> Self:
        if self.name is None and self.id is None:
            raise ValueError(
                "ExperimentModel: at least one of 'name' or 'id' must be set."
            )
        return self

    @property
    def resolved_name(self) -> Optional[str]:
        return value_of(self.name) if self.name is not None else None

    @property
    def resolved_id(self) -> Optional[str]:
        return str(value_of(self.id)) if self.id is not None else None

    def create(self, w: WorkspaceClient | None = None) -> None:
        """Ensure the referenced MLflow experiment exists and ``self.id``
        is populated.

        Matches the dao-ai ``Model.create(w)`` convention (see
        ``SchemaModel.create`` / ``VolumeModel.create``): fire-and-forget
        wrapper that delegates to the provider. Idempotent — repeated
        calls after the first no-op.

        Callers that need the resolved ``mlflow.entities.Experiment``
        object should use :meth:`DatabricksProvider.create_experiment`
        (or call ``mlflow.get_experiment(self.resolved_id)`` after this
        returns).
        """
        from dao_ai.providers.databricks import DatabricksProvider

        provider = DatabricksProvider(w=w)
        provider.create_experiment(self)


class BackgroundModel(BaseModel):
    """Opt-in background agent configuration.

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
    default_enabled: bool = Field(
        default=False,
        description="If True, requests are treated as background even when ``background: true`` is not explicitly set on the request.",
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
    :class:`BackgroundModel` — the two concepts (A2A task lifecycle vs
    Responses-API kickoff/poll/cancel) are configured separately.

    When the same ``DatabaseModel`` is referenced here, on
    ``memory.checkpointer.database``, and on ``app.background.database``,
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
            "and BackgroundStore whenever those point at the same "
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
    of :class:`BackgroundModel`; point both at the same
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
    security_schemes: Optional[dict[str, Any]] = Field(
        default=None,
        description="Overrides the Agent Card 'securitySchemes'. Keys are scheme names; "
        "values are validated against a2a-sdk's SecurityScheme discriminated union at "
        "config-load time (requires the 'a2a' extra). See ``dao_ai.apps.a2a.security`` "
        "for ready-made constants (BEARER_DATABRICKS_PAT, BEARER_DATABRICKS_M2M, "
        "BEARER_DATABRICKS_OBO) and factories (oauth2_databricks_authorization_code, "
        "oauth2_databricks_obo, openid_connect_databricks, api_key_header). When unset, "
        "derived from :attr:`on_behalf_of_user` (bearer scheme with OBO-aware description).",
    )

    @field_validator("security_schemes", mode="after")
    @classmethod
    def _validate_security_schemes(
        cls, value: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Validate scheme values against a2a-sdk's SecurityScheme union.

        Typed as ``dict[str, Any]`` so ``dao_ai.config`` imports without the
        optional ``a2a`` extra installed. When schemes are actually supplied we
        require the extra and validate each value against the real discriminated
        union, preserving the original fail-at-load-time contract.
        """
        if not value:
            return value

        from dao_ai._extras import require_extra

        require_extra("a2a", feature="A2A security schemes")
        from a2a.types import SecurityScheme
        from pydantic import TypeAdapter

        # TypeAdapter validates both raw dicts and already-constructed
        # SecurityScheme instances against the discriminated union.
        adapter: TypeAdapter = TypeAdapter(SecurityScheme)
        return {key: adapter.validate_python(scheme) for key, scheme in value.items()}

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
        "to a DatabaseModel to persist tasks in Lakebase. Independent of app.background.",
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
    space: Optional[str] = Field(
        default=None,
        description=(
            "Name of an existing Databricks App Space to assign this app to. "
            "Spaces govern shared resources, user_api_scopes, and the runtime "
            "service principal across multiple apps. App Spaces is currently in "
            "Private Preview; the space must already exist in the target workspace "
            "(create via Terraform `databricks_app_space` or "
            "`WorkspaceClient.apps.create_space()`). dao-ai does not create spaces."
        ),
    )
    workload_size: Optional[WorkloadSize] = Field(
        default="Small",
        description=(
            "Compute size for the deployment. Applies to both targets. "
            "Model Serving accepts Small/Medium/Large (XLarge is clamped to "
            "Large). For the Databricks Apps target this is coerced to the Apps "
            "compute_size domain: Small/Medium leave the platform default "
            "(MEDIUM, ~2 vCPU / 6 GB — Apps has no Small tier and existing apps "
            "are not resized), Large → LARGE, XLarge → XLARGE. Apps compute does "
            "not scale to zero."
        ),
    )
    workers: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Number of backend worker processes for the Databricks Apps server. "
            "Leave UNSET (the default) to auto-size at runtime to the container's "
            "available CPUs — the simplest config: pick the Apps compute size and "
            "dao-ai matches the worker count to it. For workers>1 the backend runs "
            "under gunicorn with ``preload_app`` so the agent graph is built ONCE "
            "in the arbiter and copy-on-write shared across forked workers (cheap "
            "memory, and workers are ready immediately rather than each rebuilding "
            "the graph — which on the older uvicorn spawn path exceeded the "
            "multi-worker startup window and crash-looped). Set an explicit "
            "integer to override the auto-sizing. Emitted as the "
            "``DAO_AI_APP_WORKERS`` env var (only when set) and forwarded to the "
            "backend as ``--workers``. Apps-target only (no effect on Model "
            "Serving, which manages its own worker pool via workload_size)."
        ),
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
        description="Additional Python files/directories (relative to the config "
        "file's directory; absolute paths pass through) shipped with EVERY "
        "deployment target: Model Serving via ``log_model(code_paths=...)``, "
        "Databricks Apps by uploading them next to the config (importable via "
        "``sys.path``), and the ``workflow generate`` job by staging them into the "
        "bundle. Use for custom ``type: python`` tool modules or agent code. "
        "(Apps bundles also still support hand-packaged code under ``src/<package>/``.)",
    )
    resource_paths: list[str] = Field(
        default_factory=list,
        description="Additional Databricks Asset Bundle resource files (``*.yml``, "
        "relative to the config file's directory; absolute paths pass through) "
        "copied into the generated bundle's ``resources/`` directory, where DABs' "
        "``include: [resources/*.yml]`` merges them at deploy. Use to add your own "
        "Jobs, Pipelines, or other resources alongside the generated bundle without "
        "editing any generated file — works identically on the agent, mcp, and "
        "workflow nouns. As a convention (like ``src/`` for code), any ``*.yml`` "
        "under a ``resources/`` directory colocated with the config is shipped "
        "automatically with no declaration here. These are user-owned: copied once "
        "and never overwritten by a rebuild (pass ``--overwrite`` to re-copy).",
    )
    pip_requirements: list[str] = Field(
        default_factory=list,
        description="Extra pip packages for your custom code, installed alongside "
        "dao-ai in every deployment target (Model Serving conda env; Apps/MCP "
        "requirements.txt; pipeline job environment).",
    )
    python_version: Optional[str] = Field(
        default="3.12",
        description="Python version for Model Serving deployment. Defaults to 3.12 "
        "which is supported by Databricks Model Serving. This lets the serving "
        "container's Python be pinned independently of the environment running the "
        "deploy (a local machine, CI, or a job may be on a different Python version).",
    )
    trace_location: Optional[TraceLocationModel] = Field(
        default=None,
        description="Unity Catalog location for storing MLflow traces in OTEL-format Delta tables. "
        "Accepts a schema reference or 'catalog.schema' string, with an optional "
        "``table_prefix`` to namespace the OTEL tables. When set, "
        "``mlflow.set_experiment(trace_location=UnityCatalog(...))`` is called at startup "
        "and at deploy time for both Model Serving and Databricks Apps deployments.",
    )
    mcp_server: Optional[McpServerCapabilitiesModel] = Field(
        default=None,
        description=(
            "Server-side MCP capabilities exposed when dao-ai is deployed as an "
            "MCP server (``dao-ai agent build --as-mcp``). Declares static resources, "
            "prompt templates, and whether progress + logging notifications are "
            "emitted from the agent tool. When None the server publishes only "
            "the single agent-as-tool surface with no notifications."
        ),
    )
    experiment: Optional[ExperimentModel] = Field(
        default=None,
        description=(
            "Reference an existing MLflow experiment by name or id. "
            "Independent of ``service_principal`` and ``trace_location`` — "
            "any combination is a valid config. When omitted, dao-ai uses "
            "``/Users/<deployer_email>/<app.name>`` (create-if-missing). "
            "When set, ``MLFLOW_EXPERIMENT_ID`` is injected on the "
            "deployed workload — from ``experiment.id`` when explicit, or "
            "resolved via ``experiment.name`` at deploy time."
        ),
    )
    manage_permissions: bool = Field(
        default=True,
        description=(
            "When True (default), dao-ai attempts UC and MLflow-experiment "
            "grants on the runtime service principal at deploy time. Set "
            "to False when an admin has already provisioned the SP + "
            "experiment + all grants and the deployer lacks GRANT rights "
            "— avoids noisy 403 warnings during deploy."
        ),
    )
    monitoring: Optional[MonitoringModel] = Field(
        default=None,
        description="Production monitoring configuration. When present, scorers are "
        "registered to continuously evaluate production traces. Works with both "
        "experiment-based traces and UC OTEL trace tables. When trace_location is "
        "also configured, the SQL warehouse from trace_location is used for monitoring.",
    )
    background: Optional[BackgroundModel] = Field(
        default=None,
        description="Opt-in background agent configuration. When set, the ResponsesAgent "
        "is wrapped so that requests with background=True or custom_inputs.operation are "
        "persisted in the referenced Lakebase database. In Databricks Apps, strict "
        "Responses API routes (/v1/responses, /v1/responses/{id}, /v1/responses/{id}/cancel) "
        "are additionally exposed. See examples/18_background_agents/.",
    )
    a2a: A2AModel = Field(
        default_factory=A2AModel,
        description="Google A2A protocol endpoint configuration for Databricks Apps "
        "deployments. Defaults to a fresh A2AModel — enabled with sensible defaults "
        "(skills derived from sub-agents, bearer scheme derived from "
        "a2a.on_behalf_of_user). Set a2a.enabled=false to opt out. Ignored for Model "
        "Serving deployments. See A2AModel for the full schema.",
    )

    def apps_compute_size(self) -> Optional[str]:
        """Map ``workload_size`` to a Databricks Apps ``compute_size``.

        Returns ``None`` to leave the Apps platform default (MEDIUM) — this is
        the case for Small and Medium, so existing apps are never resized on
        redeploy and the simplest configs keep the current behavior. Large and
        XLarge map to the corresponding Apps tiers. Apps has no Small tier and
        does not scale to zero.
        """
        return {"Large": "LARGE", "XLarge": "XLARGE"}.get(self.workload_size)

    def serving_workload_size(self) -> Optional[str]:
        """Clamp ``workload_size`` to the Model Serving domain.

        Model Serving has no XLarge tier, so XLarge is clamped to Large (its
        largest size). Small/Medium/Large pass through unchanged. Returns
        ``None`` when ``workload_size`` is unset (explicit null).
        """
        return "Large" if self.workload_size == "XLarge" else self.workload_size

    @model_validator(mode="after")
    def set_databricks_env_vars(self) -> Self:
        """Set Databricks environment variables for Model Serving / Apps.

        Each field triggers its own env-var injection independently — SP,
        experiment, and trace_location are three separate concerns and any
        combination is a valid config. Values already present in
        ``environment_vars`` are preserved (setdefault semantics).

        * ``service_principal`` → ``DATABRICKS_CLIENT_ID`` / ``DATABRICKS_CLIENT_SECRET``.
        * ``experiment.id`` → ``MLFLOW_EXPERIMENT_ID`` (only when id is
          explicit at config-load; the ``name`` and auto-derive cases are
          resolved at deploy time in ``deploy_*_agent``).
        * ``trace_location`` → ``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` (documented).
          ``MLFLOW_TRACING_DESTINATION`` is intentionally NOT set — Databricks
          docs do not use it, and MLflow's ``_get_trace_location_from_env``
          parses the ``<catalog>.<schema>`` string as legacy
          ``UCSchemaLocation`` with the hardcoded default table name
          ``mlflow_experiment_trace_otel_spans``, shadowing the correct
          experiment-linked ``UnityCatalog`` (see docs.databricks.com/aws/en/
          mlflow3/genai/tracing/trace-unity-catalog for the recommended
          pattern). Trace-location routing at runtime relies on MLflow's
          own fallback resolver reading the experiment's linked
          ``UnityCatalog`` from the tracking store.
        """
        from dao_ai.utils import get_default_databricks_host

        if "DATABRICKS_HOST" not in self.environment_vars:
            host: str | None = get_default_databricks_host()
            if host:
                self.environment_vars["DATABRICKS_HOST"] = host

        if self.service_principal is not None:
            self.environment_vars.setdefault(
                "DATABRICKS_CLIENT_ID", self.service_principal.client_id
            )
            self.environment_vars.setdefault(
                "DATABRICKS_CLIENT_SECRET", self.service_principal.client_secret
            )

        if self.experiment is not None and self.experiment.resolved_id:
            self.environment_vars.setdefault(
                "MLFLOW_EXPERIMENT_ID", self.experiment.resolved_id
            )

        if self.trace_location is not None:
            # ``warehouse_id`` is None here when the warehouse is given by NAME
            # — name→id resolution is deferred to WarehouseModel.ensure_resolved()
            # (a live API call we must not force at config-load, or offline
            # ``generate`` breaks). Injecting None poisons environment_vars: the
            # schema re-validation inside the Model Serving pyfunc rejects a None
            # value. Only set the var when the id is already concrete; a
            # name-based warehouse therefore requires pinning ``warehouse_id``
            # for Model Serving trace routing (Apps/local resolve it at runtime).
            warehouse_id: Optional[str] = self.trace_location.warehouse_id
            if warehouse_id:
                self.environment_vars.setdefault(
                    "MLFLOW_TRACING_SQL_WAREHOUSE_ID",
                    warehouse_id,
                )
            else:
                # Expected for a name-based warehouse (id resolves later);
                # fine for Apps/local, but Model Serving needs the id in the
                # endpoint env for trace routing. Log so a missing-MS-traces
                # investigation isn't left guessing.
                logger.debug(
                    "trace_location warehouse id not yet resolved at "
                    "config-load; skipping MLFLOW_TRACING_SQL_WAREHOUSE_ID "
                    "env injection. Pin trace_location.warehouse_id (not "
                    "name) for Model Serving trace routing.",
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
                # The supervisor routes with an LLM. A Genie-brain agent
                # (GenieAgentModel) has none, so borrow the first agent's model
                # that is one; with no such agent there is nothing to route
                # with and the config has to say what it wants.
                supervisor_model: InferenceEndpointModel | None = next(
                    (
                        agent.model
                        for agent in self.agents
                        if isinstance(agent.model, InferenceEndpointModel)
                    ),
                    None,
                )
                if supervisor_model is None:
                    raise ValueError(
                        "No default orchestration: every agent's model is a Genie "
                        "space, so there is no LLM for a supervisor to route with. "
                        "Declare `orchestration.supervisor.model` explicitly. A "
                        "swarm of Genie brains routes only with "
                        "`is_deterministic: true` handoffs — a Genie model "
                        "discards the agentic handoff tools, so `active_agent` "
                        "would never be written and every turn would land on the "
                        "default agent."
                    )
                self.orchestration.supervisor = SupervisorModel(model=supervisor_model)
            else:
                self.orchestration.swarm = SwarmModel(default_agent=default_agent)

        return self

    @model_validator(mode="after")
    def validate_swarm_brain_handoffs(self) -> Self:
        """Reject an agentic handoff *out of* a Genie-brain agent in a swarm.

        ``_handoffs_for_agent`` gives the source agent its agentic handoff tools
        as ``additional_tools``. A Genie model discards them, so the brain emits
        no tool call, ``active_agent`` is never written, and the swarm router
        lands every later turn back on the same agent — everything past the brain
        is unreachable, with no error anywhere. A deterministic handoff is a real
        parent-graph edge and still works; so does a brain that is a leaf.

        The check must mirror how the runtime resolves handoffs, not just the
        keys the author spelled out: ``_handoffs_for_agent`` defaults an agent
        *absent* from the dict (and every agent when the dict is empty) to
        agentic handoffs to all agents (``handoffs.get(name, config.app.agents)``).
        So a brain is a leaf only when its outbound list is declared *empty* —
        omission is the dead-swarm case, and is rejected the same as an explicit
        agentic handoff. A self-handoff is not a route away, so a lone-brain
        swarm (whose default resolves to just itself) is left alone.
        """
        if self.orchestration is None or self.orchestration.swarm is None:
            return self

        brain_names: set[str] = {
            agent.name
            for agent in self.agents
            if isinstance(agent.model, GenieAgentModel)
        }
        if not brain_names:
            return self

        handoffs = self.orchestration.swarm.handoffs or {}

        def _target_names(
            entry: "AgentModel | str | HandoffRouteModel",
        ) -> set[str]:
            def _name(a: "AgentModel | str") -> str:
                return a.name if isinstance(a, AgentModel) else a

            if isinstance(entry, HandoffRouteModel):
                if entry.agents:
                    return {_name(a) for a in entry.agents}
                return {_name(entry.agent)} if entry.agent is not None else set()
            return {_name(entry)}

        for brain_name in brain_names:
            # Resolve exactly as the runtime does: a missing key defaults to
            # every agent (legacy peer-to-peer swarm), an explicit list is used
            # verbatim, and an explicit None/[] makes the brain a leaf.
            effective = handoffs.get(brain_name, self.agents)
            for entry in effective or ():
                if isinstance(entry, HandoffRouteModel) and entry.is_deterministic:
                    continue
                # A self-handoff is not a route away, so it does not strand the
                # swarm; only an agentic handoff to another agent does.
                if _target_names(entry) - {brain_name}:
                    raise ValueError(
                        f"Swarm agent '{brain_name}' uses a Genie space as its "
                        f"model, which discards the agentic handoff tools it is "
                        f"given — it could never route away, so every later turn "
                        f"would land back on it and the rest of the swarm would "
                        f"be unreachable. Declare each handoff out of it with "
                        f"`is_deterministic: true`, declare its handoffs empty "
                        f"(`{brain_name}: []`) to make it a leaf, or give the "
                        f"agent an LLM model and reach Genie through a "
                        f"`type: genie` tool instead. (An agent omitted from "
                        f"`handoffs` defaults to agentic handoffs to every "
                        f"agent, so omission is not leaf-ness.)"
                    )

        return self

    @model_validator(mode="after")
    def set_default_endpoint_name(self) -> Self:
        if self.endpoint_name is None:
            self.endpoint_name = self.name
        return self

    @property
    def app_resource_name(self) -> str:
        """Workspace Databricks App name derived from ``name``.

        Lowercased with underscores replaced by hyphens. This is the name the
        app is deployed under (see ``dao_ai.apps.bundle``); log retrieval and
        any other App API call must use this form, not the raw ``name``.

        Chat-protocol name only. For an MCP deployment call
        :func:`app_name_for(name, as_mcp=True)`, which adds the ``mcp-`` prefix.
        """
        return app_name_for(self.name)

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


class OptimizationsModel(BaseModel):
    """Container for cache threshold optimization configurations and the
    training datasets used by optimization and offline-evaluation runs."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    training_datasets: dict[str, EvaluationDatasetModel] = Field(
        default_factory=dict,
        description="Named training datasets used by optimization and evaluation runs.",
    )
    cache_threshold_optimizations: dict[str, "ContextAwareCacheOptimizationModel"] = (
        Field(
            default_factory=dict,
            description="Named cache threshold optimization configurations using Bayesian optimization.",
        )
    )

    def optimize(self, w: WorkspaceClient | None = None) -> dict[str, Any]:
        """
        Optimize all cache thresholds in this configuration.

        This method:
        1. Ensures all training datasets are created/registered in MLflow
        2. Runs each cache threshold optimization

        Args:
            w: Optional WorkspaceClient for Databricks operations

        Returns:
            dict[str, Any]: Dictionary with a 'cache_thresholds' key containing
                the respective optimization results
        """
        # First, ensure all training datasets are created/registered in MLflow
        logger.info(f"Ensuring {len(self.training_datasets)} training datasets exist")
        for dataset_name, dataset_model in self.training_datasets.items():
            logger.debug(f"Creating/updating dataset: {dataset_name}")
            dataset_model.as_dataset()

        # Run cache threshold optimizations
        cache_results: dict[str, Any] = {}
        for name, optimization in self.cache_threshold_optimizations.items():
            cache_results[name] = optimization.optimize(w)

        return {"cache_thresholds": cache_results}


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


def _resolve_relative_to_base(value: str, base_path: str | None) -> Path:
    """Resolve a config asset path against its config directory.

    ``value`` is a filesystem path from a ``ddl``/``data`` field. Absolute
    values pass through unchanged. Relative values resolve against
    ``base_path`` (the config file's directory) when set — this is what lets
    ``ddl: functions/x.sql`` find the file colocated with the config — and
    against the process CWD otherwise (the legacy notebook-CWD behaviour).
    """
    path: Path = Path(value)
    if path.is_absolute() or base_path is None:
        return path
    return (Path(base_path) / path).resolve()


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
    # Directory the relative ``ddl``/``data`` paths resolve against — stamped by
    # ``AppConfig.from_file`` with the config file's own directory so assets can
    # be colocated with the config (``ddl: functions/x.sql``). ``None`` falls
    # back to the process CWD, preserving the legacy notebook-CWD behaviour.
    _base_path: Optional[str] = PrivateAttr(default=None)
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

    def resolve_asset_path(self, value: str) -> Path:
        """Resolve a relative ``ddl``/``data`` path against the config's dir.

        Absolute paths pass through unchanged. Relative paths resolve against
        ``_base_path`` (the config file's directory, stamped by
        ``AppConfig.from_file``) when set, else the process CWD.
        """
        return _resolve_relative_to_base(value, self._base_path)

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
    # See ``DatasetModel._base_path`` — stamped by ``AppConfig.from_file`` so a
    # relative ``ddl`` resolves against the config file's own directory.
    _base_path: Optional[str] = PrivateAttr(default=None)

    def resolve_asset_path(self, value: str) -> Path:
        """Resolve the relative ``ddl`` path against the config's directory."""
        return _resolve_relative_to_base(value, self._base_path)

    def create(
        self,
        w: WorkspaceClient | None = None,
        dfs: DatabricksFunctionClient | None = None,
    ) -> None:
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        provider: ServiceProvider = DatabricksProvider(w=w, dfs=dfs)
        provider.create_sql_function(self)


def _credential_fingerprint(room: "GenieRoomModel") -> str:
    """A stable, opaque id for the identity ``room`` would look a space up as.

    Used only as a cache key, so it never has to be reversible — and it is
    hashed precisely so that it cannot become a leak if it is ever logged or
    serialized. ``pat``/``client_secret`` are normally secret *references*
    (scope + key), but a config may inline a literal, and a raw token must not
    survive in a set key.
    """
    declaration: str = repr(
        (
            room.service_principal,
            room.client_id,
            room.client_secret,
            room.workspace_host,
            room.pat,
        )
    )
    return hashlib.sha256(declaration.encode()).hexdigest()[:16]


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
    vector_stores: dict[str, AnyVectorStore] = Field(
        default_factory=dict,
        description=(
            "Named vector-store configurations, keyed by name. Each entry "
            "is a discriminated union — the ``type`` field selects between "
            "``AiSearchVectorStoreModel`` (default, `type: ai_search`) and "
            "``LakebaseVectorStoreModel`` (`type: lakebase_search`). "
            "Existing YAMLs that omit ``type`` continue to parse as AI Search "
            "vector stores."
        ),
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

    _genie_warehouses_backfilled: set[tuple[str, str]] = PrivateAttr(
        default_factory=set
    )
    """``(space_id, identity)`` pairs whose warehouse the backfill already found.

    Keyed on the *identity* too, not the space alone: a room may carry its own
    ``service_principal``/``client_id``/``pat``/``workspace_host``, so two rooms
    over one space can look it up as different principals. One that cannot read
    the space must not stop another that can.
    """

    def backfill_genie_warehouses(self) -> Self:
        """Add each Genie room's SQL warehouse to :attr:`warehouses`.

        Prefers the explicitly configured ``warehouse:``; falls back to
        :meth:`GenieRoomModel.discover_warehouse` for existing-space references
        that don't declare one inline. Every consumer of a Genie warehouse reads
        it from here rather than from the room: the service-principal grant plan
        (the ``CAN_USE`` grant), the Model Serving auth policy, and the generated
        App resource list.

        Called twice, deliberately. The ``mode="after"`` validator below covers
        inline warehouses, which need no network. Discovery, though, cannot work
        during validation — it goes through ``_get_space_details``, which returns
        ``None`` while the room is unresolved, and rooms are resolved later, in
        :meth:`AppConfig._resolve_all_resources`. So that method calls this again
        once resolution has run, which is the only point where discovery can
        succeed — and it is the choke point every deploy path funnels through,
        not just ``initialize()``.

        Idempotent, and never raises: a room whose space cannot be inspected is
        skipped, because a deploy must not fail over a best-effort lookup.
        """
        if not self.genie_rooms:
            return self

        # Discovery is a `get_space` + `warehouses.get` per room. Rooms often
        # share a space (different tool descriptions over the same data), so
        # only look the space up for the first room that needs it — and remember
        # which spaces already yielded a warehouse, so a later call is free
        # rather than merely harmless. Spaces that yielded *nothing* are
        # deliberately not remembered: that is exactly the pre-resolution pass,
        # which the post-resolution one has to retry.
        seen_spaces: set[tuple[str, str]] = set(self._genie_warehouses_backfilled)

        for room_key, genie_room in self.genie_rooms.items():
            genie_room: GenieRoomModel

            # An inline ``warehouse:`` is a declaration, not a lookup, so it is
            # taken before any of the gates below — including for
            # ``on_behalf_of_user`` rooms, whose warehouse still has to reach
            # ``warehouses``: ``generate_user_api_scopes`` reads OBO warehouses
            # from there to emit the ``sql`` scope the forwarded user token needs,
            # and ``build_auth_policy`` routes them to the ``UserAuthPolicy``.
            # Dropping it here is what would break OBO Genie.
            warehouse: Optional[WarehouseModel] = genie_room.warehouse
            discovered: bool = warehouse is None
            space_id: Optional[str] = None
            cache_key: Optional[tuple[str, str]] = None

            if warehouse is None:
                # Everything from here on is a live lookup.
                #
                # OBO rooms query as the calling user, whose own warehouse access
                # applies, so there is nothing to discover a warehouse *for*: the
                # app's service principal is never granted CAN_USE on their
                # behalf. (An OBO room that declares no warehouse inline
                # therefore contributes no ``sql`` scope — unchanged from before
                # discovery existed. Declare the warehouse inline to get one.)
                if genie_room.on_behalf_of_user:
                    continue

                try:
                    space_id = (
                        value_of(genie_room.space_id) if genie_room.space_id else None
                    )
                    if space_id is not None:
                        cache_key = (space_id, _credential_fingerprint(genie_room))
                        if cache_key in seen_spaces:
                            continue
                    warehouse = genie_room.discover_warehouse()
                except Exception as e:
                    logger.debug(
                        "Could not determine the warehouse for Genie room",
                        room=room_key,
                        error=str(e),
                    )
                    continue

            if warehouse is None or not warehouse.warehouse_id:
                continue

            # Only *now* is the space known to have yielded something. Marking it
            # before the lookup would let one room's failure (no permission, a
            # transient 5xx) suppress the retry another room — or a later pass —
            # is entitled to.
            if cache_key is not None:
                seen_spaces.add(cache_key)
                self._genie_warehouses_backfilled.add(cache_key)

            # Hand the room the warehouse it just yielded. Without this, discovery
            # is invisible to anything that re-parses the config: every fresh load
            # (Model Serving model load, Apps container startup) would re-issue
            # ``get_space`` + ``warehouses.get`` per room and then throw the answer
            # away at the dedupe check below, paying cold-start latency — and a
            # ``logger.warning`` per 403 — for a result the baked config already
            # holds. It also gives ``_extract_genie_warehouse_resources`` its
            # documented fallback: that extractor reads ``genie.warehouse``, which
            # nothing else populates.
            if discovered:
                genie_room.warehouse = warehouse

            # Already present (either from a previous call or declared by hand).
            if any(
                existing.warehouse_id == warehouse.warehouse_id
                for existing in self.warehouses.values()
            ):
                continue

            # Key off the room's mapping key, not ``genie_room.name`` — the name
            # is None for a bare-``space_id`` room until resolution runs, while
            # the key is always present and stable. The ``_warehouse`` suffix is
            # load-bearing: a warehouse key and a Genie room key both become an
            # App resource *name*, and the DABs bundle has no uniquify pass — a bare
            # ``room_key`` would emit two resources called e.g. ``retail_genie``,
            # one genie-space and one sql-warehouse, with the ``value_from``
            # bindings then ambiguous. It also matches what
            # ``_extract_genie_warehouse_resources`` has always named this same
            # warehouse, so when both paths emit it the existing
            # dedupe-by-identity pass collapses them.
            warehouse_key: str = normalize_name(f"{room_key}_warehouse")
            # A hand-declared resource may already own that key; suffix rather
            # than clobber it. Genie room keys share the App resource namespace,
            # so they count as taken too (rooms keyed ``sales`` and
            # ``sales_warehouse`` in one config). Compare *normalized* forms on
            # both sides: ``warehouse_key`` is already normalized, so a config
            # key written ``Sales_Warehouse`` would otherwise slip past the guard
            # and collide once the App resource namespace folds their case.
            taken: set[str] = {
                normalize_name(key)
                for key in (*self.warehouses.keys(), *self.genie_rooms.keys())
            }
            if warehouse_key in taken:
                warehouse_key = normalize_name(
                    "_".join([room_key, str(value_of(warehouse.warehouse_id))])
                )

            self.warehouses[warehouse_key] = warehouse
            logger.trace(
                "Added warehouse from Genie room",
                room=room_key,
                warehouse=warehouse.warehouse_id,
                key=warehouse_key,
            )

        return self

    @model_validator(mode="after")
    def update_genie_warehouses(self) -> Self:
        """Populate :attr:`warehouses` from any inline Genie ``warehouse:``.

        See :meth:`backfill_genie_warehouses`, which this delegates to and which
        ``AppConfig._resolve_all_resources`` calls again once the rooms are
        resolved.
        """
        return self.backfill_genie_warehouses()

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


def _reject_relative_assets_for_remote_config(
    config: "AppConfig", *, source: str
) -> None:
    """Raise if a URL-loaded config declares assets only a local tree could supply.

    A remote config has no directory to resolve relative paths against, so
    ``ddl: functions/x.sql`` cannot be found. Failing here — naming the field —
    beats resolving against the process CWD and surfacing a FileNotFoundError
    several provisioning steps later.

    ``code_paths`` and ``skills`` matter for a second reason. Resolving a remote
    config's ``code_paths`` would put a directory named by a remote document onto
    ``sys.path``; resolving its skills would read whichever local ``skills/`` tree
    the process happened to be standing next to and splice that Markdown into the
    agent's prompt. Both let a document that cannot be trusted to name local files
    do exactly that, so both are refused rather than resolved.
    """

    def _is_relative(value: object) -> bool:
        if not isinstance(value, str) or not value:
            return False
        if value.startswith(("/", "dbfs:", "s3:", "abfss:", "gs:")):
            return False
        # Inline SQL rather than a path — ``data:`` accepts either.
        if "\n" in value or value.rstrip().endswith(";"):
            return False
        return True

    offenders: list[str] = []
    for dataset in config.datasets or []:
        for field_name in ("ddl", "data"):
            value = getattr(dataset, field_name, None)
            if _is_relative(value):
                offenders.append(f"datasets[].{field_name}: {value}")
    for fn in config.unity_catalog_functions or []:
        value = getattr(fn, "ddl", None)
        if _is_relative(value):
            offenders.append(f"unity_catalog_functions[].ddl: {value}")
    if config.app is not None:
        for code_path in config.app.code_paths or []:
            if _is_relative(code_path):
                offenders.append(f"app.code_paths: {code_path}")

    # Skills are declared as ``path:`` but stored as middleware ``sources`` by the
    # time this runs, so read them where they end up. Volume-backed skills are
    # absolute and legitimately remote-safe; ``_is_relative`` already excludes them.
    from dao_ai.skills import (
        _declared_skill_path,
        _iter_agent_skill_sources,
        _iter_deep_agent_skills,
    )

    for skill_source in _iter_agent_skill_sources(config):
        if _is_relative(skill_source):
            offenders.append(f"skills[].path: {skill_source}")
    # deep_agent / subagent skills keep their spec form (they are resolved at
    # graph build, not translated to middleware), so check them separately. Both
    # forms have to be read through ``_declared_skill_path``: a bare string is
    # either a key into ``resources.skills`` — whose target carries the path that
    # matters — or an inline relative path, and narrowing to ``SkillModel`` alone
    # let ``skills: [skills/research]`` walk straight past a guard whose whole
    # purpose is to stop a remote document from naming a local directory.
    for spec in _iter_deep_agent_skills(config):
        declared: str | None = _declared_skill_path(spec, config)
        if _is_relative(declared):
            label: str = spec.name if isinstance(spec, SkillModel) else str(spec)
            offenders.append(f"deep_agent skill {label!r}: {declared}")
    # ``instruction_files`` are spliced into the system prompt verbatim, so a
    # remote config naming a relative one is the same untrusted-document-reads-
    # local-Markdown problem as skills, with none of the indirection.
    if config.app is not None and config.app.orchestration is not None:
        deep_agent = config.app.orchestration.deep_agent
        for entry in (deep_agent.instruction_files if deep_agent else []) or []:
            if _is_relative(entry):
                offenders.append(f"deep_agent.instruction_files: {entry}")

    if offenders:
        raise ValueError(
            f"Config loaded from {source} declares relative paths that cannot be "
            "resolved — a URL serves one file, with no directory to anchor them to:"
            + "".join(f"\n  - {o}" for o in offenders)
            + "\nLoad it as a git locator instead, which brings the whole project "
            "tree along and resolves these normally:"
            + f"\n  {_git_locator_hint(str(source))}"
            + "\nAlternatively use absolute paths or a Unity Catalog Volume "
            "reference, or download the project and pass a local path."
        )


def _git_locator_hint(url: str) -> str:
    """Rewrite a raw/blob GitHub config URL into the equivalent git locator.

    The URL already carries owner, repo, ref, and in-repo path — everything a
    locator needs — so the suggestion can be exact and copy-pasteable rather than
    a generic syntax reminder. Falls back to the generic form for other hosts.
    """
    from urllib.parse import urlparse

    generic: str = "git+https://<host>/<owner>/<repo>@<ref>#<path/to/config.yaml>"
    parsed = urlparse(url)
    parts: list[str] = parsed.path.lstrip("/").split("/")

    if parsed.netloc == "raw.githubusercontent.com" and len(parts) >= 4:
        owner, repo, ref, *rest = parts
        # A raw URL may spell the ref as `refs/heads/<branch>`.
        if ref == "refs" and len(rest) >= 2:
            ref, rest = rest[1], rest[2:]
        return f"git+https://github.com/{owner}/{repo}@{ref}#{'/'.join(rest)}"

    if parsed.netloc in ("github.com", "www.github.com") and len(parts) >= 5:
        owner, repo, kind, ref, *rest = parts
        if kind in ("blob", "raw"):
            return f"git+https://github.com/{owner}/{repo}@{ref}#{'/'.join(rest)}"

    return generic


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
    retrievers: dict[str, AnyRetriever] = Field(
        default_factory=dict,
        description=(
            "Named retriever configurations, keyed by name. Each entry is a "
            "discriminated union — the ``type`` field selects between "
            "``AiSearchRetrieverModel`` (default, `type: ai_search`) and "
            "``LakebaseRetrieverModel`` (`type: lakebase_search`). Existing "
            "YAMLs that omit ``type`` continue to parse as AI Search retrievers."
        ),
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
        description="Named, reusable prompt definitions referenced by agents via YAML anchors.",
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
    # Immutable revision of the source the config was loaded from (a git commit
    # SHA), when it has one. Folded into the bundle checksum so bumping a git ref
    # re-stages even when the config text is byte-identical and only a colocated
    # asset (ddl/data) changed.
    _source_git_sha: str | None = None
    # The config's actual path on disk. Identical to ``_source_config_path`` for a
    # local file, but for a git source that field holds the locator (what the user
    # typed) while this holds the file inside the checkout.
    _local_config_path: str | None = None
    _rendered_yaml: str | None = None
    _substitution_vars: dict[str, str] | None = None
    _raw_yaml_dict: dict[str, Any] | None = None
    # The ${workspace.*}-resolved but ${param/var}-UNsubstituted source text, the
    # parsed parameter declarations, and the operator-supplied param names (CLI
    # --param/--var only — NOT taskValues/env/defaults). The workflow staging path
    # reuses these to re-render with a `defer` set (preserve unprovided Genie-room
    # space_id refs) without re-reading the source file. See write_pipeline_bundle.
    _workspace_resolved_yaml: str | None = None
    _declarations: dict[str, ParameterDeclarationModel] | None = None
    _operator_supplied_params: set[str] | None = None
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

    @model_validator(mode="after")
    def _validate_genie_agent_rooms_registered(self) -> Self:
        """Require every ``GenieAgentModel`` agent's room to be registered
        under ``resources.genie_rooms``.

        A ``GenieAgentModel`` is a plain wrapper — it is not an
        ``IsDatabricksResource`` and is never collected for deploy grants.
        The Genie deploy resource (``genie-space`` + ``dashboards.genie``
        scope) is emitted only from ``resources.genie_rooms``. If an agent's
        Genie room is inlined solely under ``agent.model.genie_room`` and not
        registered, the bundle deploys with NO Genie grant and the agent 403s
        at runtime — a silent failure. Catch it at config-load instead.

        Matching is by resolved ``agent_id``/``space_id`` (not object
        identity), so it holds whether the room is shared via a YAML anchor or
        written inline with the same id. Rooms whose id cannot be resolved
        statically (name-only, resolved by a live lookup) are skipped — they
        cannot be checked without an API call.

        Also catches the inverse mistake: a name-only room assigned bare to
        ``model:``. ``{name: X}`` is a valid shape for *both* union members, so
        :meth:`AgentModel._wrap_bare_genie_room` cannot coerce it and it lands
        as an ``InferenceEndpointModel`` pointing at a serving endpoint that
        does not exist — clean at load, a runtime failure later. Here the room
        registry is visible, so a name that matches a registered room is
        reported instead.
        """
        registered_ids: set[str] = set()
        registered_names: set[str] = set()
        if self.resources and self.resources.genie_rooms:
            for room in self.resources.genie_rooms.values():
                raw_id: Any = room.space_id
                resolved: Any = value_of(raw_id) if raw_id is not None else None
                if resolved:
                    registered_ids.add(str(resolved))
                room_name: Any = value_of(room.name) if room.name is not None else None
                if room_name:
                    registered_names.add(str(room_name))

        seen: set[int] = set()

        def _check_one(agent: AgentModel) -> None:
            if id(agent) in seen:
                return
            seen.add(id(agent))
            model = agent.model
            if isinstance(model, InferenceEndpointModel):
                # Only the genuinely ambiguous shape: a bare ``{name: X}``. Any
                # endpoint-specific key (temperature, max_tokens, …) means the
                # user meant an endpoint, so leave those alone.
                if model.model_fields_set == {"name"} and (
                    str(value_of(model.name)) in registered_names
                ):
                    raise ValueError(
                        f"Agent '{agent.name}' assigns model: "
                        f"{{name: '{value_of(model.name)}'}}, which matches the "
                        f"Genie room of that name under resources.genie_rooms — "
                        f"but a bare {{name: ...}} is indistinguishable from a "
                        f"serving endpoint, so it resolved to an inference "
                        f"endpoint and will fail at runtime. Write "
                        f"'model: {{genie_room: *your_room_anchor}}' to use the "
                        f"room as a Genie Agent, or give the room an "
                        f"agent_id/space_id so it can be assigned bare."
                    )
                return
            if not isinstance(model, GenieAgentModel):
                return
            raw_id = model.genie_room.space_id
            resolved = value_of(raw_id) if raw_id is not None else None
            if not resolved:
                # Name-only room; can't verify without a live lookup.
                return
            if str(resolved) not in registered_ids:
                raise ValueError(
                    f"Agent '{agent.name}' uses a GenieAgentModel whose Genie "
                    f"room (agent_id/space_id '{resolved}') is not registered "
                    f"under resources.genie_rooms. Register it there (e.g. via "
                    f"a YAML anchor shared with agent.model.genie_room) so the "
                    f"deploy emits the genie-space grant; otherwise the agent "
                    f"will fail with PERMISSION_DENIED at runtime."
                )

        for agent in (self.agents or {}).values():
            _check_one(agent)
        if self.app and self.app.agents:
            for agent in self.app.agents:
                _check_one(agent)

        return self

    @classmethod
    def _coerce_source(
        cls,
        source: "SourceLike",
        *,
        expected: type["ConfigSource"] | None,
    ) -> "ConfigSource":
        """Normalize a spec-or-source to a :class:`ConfigSource`.

        Every ``from_*`` method accepts either a spec string (classified here) or
        an already-constructed source (useful when it needs options a bare string
        cannot express, e.g. ``GitSource(spec, token=...)``). They differ only in
        the ``expected`` type they pin.

        Args:
            source: A path, URL, git locator, or a :class:`ConfigSource`.
            expected: The source type the calling method accepts, or ``None`` to
                accept any (``from_file``/``from_source``).

        Raises:
            ValueError: if ``source`` is not the ``expected`` kind.
        """
        from dao_ai.sources import ConfigSource, resolve_source

        if isinstance(source, ConfigSource):
            if expected is not None and not isinstance(source, expected):
                raise ValueError(
                    f"Expected {expected.__name__}, got {type(source).__name__}. "
                    "Use AppConfig.from_source to accept any source type."
                )
            return source

        if expected is None:
            return resolve_source(source)

        if not expected.handles(str(source)):
            raise ValueError(f"Not a valid {expected.__name__} spec: {str(source)!r}")
        return expected(str(source))

    @classmethod
    def from_source(cls, source: "SourceLike", **kwargs: Any) -> "AppConfig":
        """Load an AppConfig from any supported source.

        Accepts a filesystem path, an ``http(s)`` URL, a git locator, or an
        explicitly-constructed :class:`ConfigSource`. The named ``from_file`` /
        ``from_url`` / ``from_git`` methods are typed front doors onto this,
        each validating that the source is the kind it advertises.

        Args:
            source: A path, URL, git locator, or :class:`ConfigSource`.
            **kwargs: Forwarded to :meth:`from_file` (``params``,
                ``task_values``, ``task_key``, ``initialize``).
        """
        return cls.from_file(source, **kwargs)

    @classmethod
    def from_git(cls, locator: "SourceLike", **kwargs: Any) -> "AppConfig":
        """Load an AppConfig from a git repository, bringing its whole tree along.

        Where :meth:`from_url` fetches a single YAML — and so cannot resolve
        relative ``ddl`` / ``data`` / ``code_paths`` — this materializes the repo
        into a local cache, so every colocated-asset convention resolves exactly
        as it does for a local project::

            git+https://github.com/<owner>/<repo>@<ref>#path/to/agent.yaml
            gh:<owner>/<repo>@<ref>#path/to/agent.yaml

        The ref and in-repo path are both optional: the remote's default HEAD is
        used when no ref is given, and the config is auto-discovered when the path
        is a directory or omitted (ambiguity is an error naming the candidates).

        Pass a :class:`~dao_ai.git_source.GitSource` instead of a string to set a
        token, cache directory, or force a refresh.

        A git locator runs the repository's code — a config can ship Python via
        ``code_paths`` / ``src/`` — exactly as cloning it and running dao-ai
        locally would. The resolved commit SHA is logged on every load. Pin a tag
        or SHA for repositories you do not control.

        Args:
            locator: A git locator, or a :class:`~dao_ai.git_source.GitSource`.
            **kwargs: Forwarded to :meth:`from_file` (``params``,
                ``task_values``, ``task_key``, ``initialize``).

        Raises:
            ValueError: if ``locator`` is not a git locator, if the repository or
                ref cannot be fetched, or if the config cannot be identified.
        """
        from dao_ai.git_source import GitSource

        return cls.from_file(cls._coerce_source(locator, expected=GitSource), **kwargs)

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "AppConfig":
        """Load an AppConfig from an ``http(s)`` URL.

        A convenience alias for :meth:`from_file`, which accepts URLs directly —
        useful when the caller wants the intent to be explicit, or wants a URL
        validated as such rather than falling through to a filesystem read.

        GitHub file-viewer links are rewritten to their raw equivalent, so both
        of these work::

            https://github.com/<owner>/<repo>/blob/main/config.yaml
            https://raw.githubusercontent.com/<owner>/<repo>/main/config.yaml

        Only YAML is parsed; remote content is never executed. Because a remote
        config has no local directory, it cannot resolve relative ``ddl`` /
        ``data`` / ``code_paths`` entries — declaring one raises ``ValueError``.

        Args:
            url: The config URL, or a :class:`UrlSource`.
            **kwargs: Forwarded to :meth:`from_file` (``params``,
                ``task_values``, ``task_key``, ``initialize``).

        Raises:
            ValueError: if ``url`` is not an http(s) URL, if it cannot be
                fetched, or if the config declares unresolvable relative paths.
        """
        from dao_ai.config_source import is_remote_config
        from dao_ai.sources import ConfigSource, UrlSource

        # Keep the long-standing message for a plain non-URL string; _coerce_source
        # handles the wrong-source-type case.
        if not isinstance(url, ConfigSource) and not is_remote_config(url):
            raise ValueError(
                f"Not an http(s) URL: {url!r}. Use AppConfig.from_file for "
                "filesystem paths."
            )
        return cls.from_file(cls._coerce_source(url, expected=UrlSource), **kwargs)

    @classmethod
    def from_file(
        cls,
        # `SourceLike` (= str | PathLike | ConfigSource), not bare `PathLike`:
        # every caller in the pipeline notebooks passes a plain string (a widget
        # value), and `str` is not a `PathLike` — the narrower annotation was a
        # type error at all 11 of them. The `ConfigSource` arm additionally lets a
        # caller pass a constructed source (see :meth:`from_source`).
        path: SourceLike,
        *,
        params: Optional[Mapping[str, str]] = None,
        task_values: Optional[TaskValuesLike] = None,
        task_key: Optional[str] = None,
        initialize: bool = True,
    ) -> "AppConfig":
        """Load an AppConfig from a YAML file with optional parameter substitution.

        The general-purpose loader, and the single implementation every other
        ``from_*`` delegates to. Accepts a filesystem path, an ``http(s)`` URL, a
        git locator, or an explicit :class:`~dao_ai.sources.ConfigSource`; the
        source is classified by :func:`~dao_ai.sources.resolve_source`. Unlike
        :meth:`from_url` / :meth:`from_git` it validates nothing, since it has
        always accepted whatever it was handed.

        Top-level ``parameters:`` declarations are parsed first and used to
        resolve ``${param.NAME}`` and ``${var.NAME}`` references in the rest
        of the YAML (the two prefixes are interchangeable aliases).
        Resolution precedence per reference is CLI ``params`` > ``task_values``
        > process env > declared ``default`` > inline ``${var.NAME:-fallback}``
        > error.

        Args:
            path: Path to the YAML config file, a URL, a git locator, or a
                :class:`~dao_ai.sources.ConfigSource`.
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
        # Sources differ in exactly two ways — how the text is read, and whether
        # there is a local directory to anchor relative assets and code_paths
        # against. Everything downstream is identical. See dao_ai.sources.
        source: ConfigSource = cls._coerce_source(path, expected=None)
        logger.debug(f"Loading config from {source}")
        resolved: ResolvedConfig = source.load()

        raw_text: str = resolved.text
        path = resolved.origin
        base_path: Optional[str] = (
            str(resolved.base_path) if resolved.base_path is not None else None
        )

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
        config._source_git_sha = resolved.revision
        # The config's real location on disk, which for a git source is the file
        # inside the checkout while `_source_config_path` keeps the locator (what
        # the user typed, and what messages should echo). Consumers that need the
        # actual path — e.g. deriving a staging dir keyed by repo — use this.
        config._local_config_path = (
            str(resolved.local_path) if resolved.local_path is not None else None
        )
        config._rendered_yaml = rendered_text
        config._substitution_vars = dict(merged_params) if merged_params else None
        # Preserve inputs the workflow staging path needs to re-render with a
        # `defer` set (unprovided Genie-room space_id refs). operator_supplied is
        # the CLI --param/--var subset only, distinct from merged_params (which
        # also folds in taskValues) — the defer decision keys off what the
        # operator explicitly passed, not resolved values/defaults.
        config._workspace_resolved_yaml = workspace_resolved_text
        config._declarations = declarations
        config._operator_supplied_params = set(params.keys()) if params else set()

        # Stamp each provisioning model with the config's own directory so its
        # relative ``ddl``/``data`` paths resolve against the config location
        # (assets colocated with the config), not the process CWD. Absolute
        # paths and Volume references are unaffected.
        if base_path is not None:
            for dataset in config.datasets or []:
                dataset._base_path = base_path
            for fn in config.unity_catalog_functions or []:
                fn._base_path = base_path

            # Put custom code (``app.code_paths``) on ``sys.path`` now that the
            # config directory is known, resolving each entry against it. The
            # ``add_code_paths_to_sys_path`` validator ran at construction (before
            # ``_source_config_path`` was set) and could only anchor at the process
            # CWD; this makes config-relative custom modules importable for every
            # consumer that loads a config from a file — the deploy notebook's
            # ``display_graph``/``create_agent``, the Apps runtime, and any tool
            # resolution via ``load_function`` — regardless of the process CWD.
            from dao_ai.code_paths import (
                prepend_code_paths_to_sys_path,
                prepend_src_to_sys_path,
            )

            prepend_code_paths_to_sys_path(config)
            # Convention: a colocated ``src/`` dir auto-ships its packages; put it
            # on sys.path so ``src/foo/bar.py`` imports as ``foo.bar`` for every
            # consumer (deploy notebook display_graph/create_agent, Apps runtime,
            # load_function).
            prepend_src_to_sys_path(config)

            # Skills need no equivalent fixup: their ``sources`` stay relative in
            # the config and are resolved against the config's directory at graph
            # build time (``skills.skill_anchors``). Rewriting them here instead
            # would bake *this* machine's paths into the config that
            # ``create_agent`` then serializes into the model artifact.
        else:
            # A remote config has no local directory, so anything it declares by
            # relative path is unresolvable. Say so loudly rather than silently
            # anchoring at the process CWD and failing much later with a
            # FileNotFoundError. This is also a security boundary: code_paths and
            # skills from a remote config would otherwise put remote-authored
            # directories on sys.path.
            _reject_relative_assets_for_remote_config(config, source=str(path))
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
                # ``agent_id`` is an alias of ``space_id`` on GenieRoomModel
                # (the Genie Agent Mode API renamed the concept); accept
                # either key when snapshotting the pre-substitution value.
                room._raw_space_id = raw_room.get("space_id") or raw_room.get(
                    "agent_id"
                )

        if initialize:
            config.initialize()

        return config

    @property
    def source_config_path(self) -> str | None:
        """Get the source config file path if loaded via from_file."""
        return self._source_config_path

    @property
    def local_config_path(self) -> str | None:
        """The config's real path on disk, or ``None`` when nothing local backs it.

        Prefer this over :attr:`source_config_path` for anything that touches the
        filesystem — resolving a colocated asset, anchoring ``skills``/
        ``code_paths``, testing whether the config came out of the checkout cache.
        ``source_config_path`` holds the *locator* when loaded via
        :meth:`from_git` (what the user typed, which messages should echo), and a
        locator is not a path. For a local file the two are identical; for a URL
        neither is a path, so the URL falls through and callers fail on it as they
        always have.
        """
        return self._local_config_path or self._source_config_path

    @property
    def rendered_yaml(self) -> str | None:
        """Get the YAML text after ${param.NAME} substitution, if loaded via from_file."""
        return self._rendered_yaml

    @property
    def substitution_vars(self) -> dict[str, str] | None:
        """Get the explicit substitution vars used at load time, if any."""
        return self._substitution_vars

    def _resolve_all_resources(self) -> None:
        """Walk the config tree and call ensure_resolved() on all IsDatabricksResource instances.

        Genie warehouse discovery runs here, at the tail, because it only works
        *after* resolution: the ``ResourcesModel`` validator runs while the rooms
        are still unresolved, where ``discover_warehouse`` can only return None.
        This is the one choke point every deploy path funnels through —
        ``initialize()``, and the CLI paths that load with ``initialize=False``
        and call this directly (``agent up --mode apps``, ``--direct``,
        ``agent build``, ``service-principal grant -c``) — so every consumer
        downstream sees the discovered warehouse: the SP grant plan's ``CAN_USE``,
        the Model Serving auth policy, and the App resource + env-var lists.
        """

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

        if self.resources is not None:
            self.resources.backfill_genie_warehouses()

    def assert_provided_params_satisfied(self) -> None:
        """Guard non-workflow deploy paths against unsatisfied ``provided`` params.

        A parameter declared ``provided: true`` is furnished dynamically at run
        time (like a build tool's 'provided' dependency scope) — in dao-ai today
        that is the ``workflow`` job's provisioning tasks forwarding values via
        taskValues (e.g. a Genie space id), but the flag is generic and
        independent of any consumer. The apps / mcp / model_serving deploy paths
        have NO such run-time step, so a ``provided`` param that was neither
        supplied by the operator (``--param``/``--var``) nor given a ``default``
        fallback resolves to an empty placeholder and would silently deploy a
        broken binding. Fail loudly instead.

        No-op on the workflow path (which never calls this — it defers such params
        for its run-time step). Params with a ``default`` or an operator value are
        satisfied and pass. Independent of genie / any specific field.
        """
        declarations = self._declarations or {}
        supplied: set[str] = self._operator_supplied_params or set()
        unsatisfied: list[str] = [
            name
            for name, decl in declarations.items()
            if decl.provided and decl.default is None and name not in supplied
        ]
        if unsatisfied:
            joined = ", ".join(sorted(unsatisfied))
            raise ValueError(
                f"Parameter(s) declared `provided: true` have no value on a "
                f"non-workflow deploy path: {joined}. These paths cannot furnish "
                "values at run time. Supply each one (e.g. --param "
                f"{sorted(unsatisfied)[0]}=<value>), give it a `default`, or run "
                "`dao-ai workflow up` to provision it."
            )

    def initialize(self) -> None:
        if self._initialized:
            return

        from dao_ai.hooks.core import create_hooks
        from dao_ai.logging import configure_logging

        if self.app and self.app.log_level:
            configure_logging(level=self.app.log_level)

        # Also back-fills each Genie room's discovered warehouse — see there.
        self._resolve_all_resources()

        # ``app`` is Optional — app-less configs (e.g. an ``optimizations``-only
        # config for cache-threshold tuning) have no initialization hooks. Guard
        # like the log_level check above; without this, from_file() on such a
        # config raises AttributeError on ``self.app.initialization_hooks``.
        if self.app is not None:
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

        # ``app`` is Optional — app-less configs have no shutdown hooks. Guard
        # to match initialize(); otherwise shutdown() (called from from_file)
        # raises AttributeError on ``self.app.shutdown_hooks``.
        if self.app is None:
            return

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
        development: bool | None = None,
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
        provider.create_agent(self, development=development)

    def deploy_agent(
        self,
        mode: ServingMode | None = None,
        w: WorkspaceClient | None = None,
        vsc: "VectorSearchClient | None" = None,
        pat: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        workspace_host: str | None = None,
        development: bool | None = None,
        as_mcp: bool = False,
    ) -> None:
        """
        Deploy the agent using the specified serving mode.

        Mode resolution: the caller supplies ``mode`` (the CLI defaults it).
        If not provided, defaults to MODEL_SERVING.

        Args:
            mode: The serving platform (MODEL_SERVING or APPS). If None,
                defaults to MODEL_SERVING.
            w: Optional WorkspaceClient instance
            vsc: Optional VectorSearchClient instance
            pat: Optional personal access token for authentication
            client_id: Optional client ID for service principal authentication
            client_secret: Optional client secret for service principal authentication
            workspace_host: Optional workspace host URL
            development: Ship local dao-ai source/wheel (True), the PyPI package
                (False), or auto-detect from the install type (None).
            as_mcp: Serve the agent over MCP instead of the chat UI. Valid only
                with ``mode=APPS`` (MCP runs on the Apps runtime); deploys under
                the ``mcp-`` prefixed App name.
        """
        from dao_ai.providers.base import ServiceProvider
        from dao_ai.providers.databricks import DatabricksProvider

        # Mode is a deploy-action parameter, not config: the caller supplies it
        # (the CLI defaults it). No AppConfig fallback.
        resolved_mode: ServingMode = mode or ServingMode.MODEL_SERVING

        provider: ServiceProvider = DatabricksProvider(
            w=w,
            vsc=vsc,
            pat=pat,
            client_id=client_id,
            client_secret=client_secret,
            workspace_host=workspace_host,
        )
        provider.deploy_agent(
            self, mode=resolved_mode, development=development, as_mcp=as_mcp
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

    def interrupt_parser_model(self) -> Optional[LanguageModelLike]:
        """Chat model used to parse a user's natural-language HITL interrupt
        response (approve/reject/edit) into structured decisions.

        Resolution order (first match wins), so the parser matches the
        deployment's model rather than a hardcoded (possibly deprecated)
        endpoint:

          1. explicit ``orchestration.interrupt_model``
          2. supervisor router model (``orchestration.supervisor.model``)
          3. swarm default agent's model
          4. first declared agent's model
          5. None → ``handle_interrupt_response`` falls back to its own GA default

        The resolved endpoint and its source are logged so operators can see
        which model is parsing interrupts and why.
        """
        orch = self.app.orchestration if self.app else None

        # 1. Explicit override.
        if orch is not None and orch.interrupt_model is not None:
            logger.info(
                "HITL interrupt parser model resolved",
                source="orchestration.interrupt_model",
                endpoint=orch.interrupt_model.name,
            )
            return orch.interrupt_model.as_chat_model()

        # 2. Supervisor router model.
        if orch is not None and orch.supervisor is not None:
            logger.info(
                "HITL interrupt parser model resolved",
                source="orchestration.supervisor.model",
                endpoint=orch.supervisor.model.name,
            )
            return orch.supervisor.model.as_chat_model()

        agents: list[AgentModel] = list(self.agents.values())
        if not agents:
            logger.info(
                "HITL interrupt parser model unresolved — no agents declared; "
                "handle_interrupt_response will use its GA fallback",
                source="fallback",
            )
            return None

        # 3. Swarm default agent, else 4. first declared agent.
        default_agent: AgentModel = agents[0]
        source: str = "agents[0].model"
        if (
            orch is not None
            and orch.swarm
            and isinstance(orch.swarm.default_agent, AgentModel)
        ):
            default_agent = orch.swarm.default_agent
            source = "orchestration.swarm.default_agent.model"
        # ``model`` is InferenceEndpointModel | GenieAgentModel; only the former
        # exposes ``name``. Genie-backed agents have no endpoint name.
        endpoint = getattr(default_agent.model, "name", "<genie-agent>")
        logger.info(
            "HITL interrupt parser model resolved",
            source=source,
            agent=default_agent.name,
            endpoint=endpoint,
        )
        return default_agent.model.as_chat_model()

    def as_responses_agent(self) -> ResponsesAgent:
        from dao_ai.models import create_responses_agent

        graph: CompiledStateGraph = self.as_graph()
        tool_models: list[ToolModel] = [
            tool for agent in self.agents.values() for tool in agent.tools
        ]
        app: ResponsesAgent = create_responses_agent(
            graph,
            tool_models=tool_models,
            interrupt_model=self.interrupt_parser_model(),
        )

        background = self.app.background if self.app else None
        if background is not None:
            from dao_ai.background import (
                BackgroundResponsesAgent,
                BackgroundStore,
            )

            store = BackgroundStore(
                database=background.database,
                responses_table_name=background.responses_table_name,
                messages_table_name=background.messages_table_name,
            )
            app = BackgroundResponsesAgent(
                inner=app,
                store=store,
                max_duration_seconds=background.max_duration_seconds,
                poll_interval_seconds=background.poll_interval_seconds,
                default_enabled=background.default_enabled,
            )

        return app
