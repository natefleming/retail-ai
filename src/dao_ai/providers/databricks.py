import base64
import re
import time
from pathlib import Path
from typing import Any, Callable, Final, Optional, Sequence

import mlflow
import pandas as pd
import sqlparse
import yaml
from databricks import agents
from databricks.agents import PermissionLevel, set_permissions
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import (
    BadRequest,
    InvalidParameterValue,
    NotFound,
    PermissionDenied,
)
from databricks.sdk.service.catalog import (
    CatalogInfo,
    ColumnInfo,
    FunctionInfo,
    PrimaryKeyConstraint,
    SchemaInfo,
    TableConstraint,
    TableInfo,
    VolumeInfo,
    VolumeType,
)
from databricks.sdk.service.iam import User
from databricks.sdk.service.serving import EndpointStateConfigUpdate
from databricks.sdk.service.workspace import GetSecretResponse, ImportFormat
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import Experiment
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.genai.prompts import load_prompt
from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy
from mlflow.models.model import ModelInfo
from mlflow.models.resources import (
    DatabricksResource,
)
from unitycatalog.ai.core.base import FunctionExecutionResult
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

import dao_ai
from dao_ai.config import (
    AppConfig,
    ConnectionModel,
    DatabaseModel,
    DatabricksAppModel,
    DatasetModel,
    DeploymentTarget,
    FunctionModel,
    GenieEntitlement,
    GenieRoomModel,
    HasFullName,
    IndexModel,
    InferenceEndpointModel,
    IsDatabricksResource,
    PromptModel,
    SchemaModel,
    TableModel,
    UnityCatalogFunctionSqlModel,
    VectorStoreModel,
    VolumeModel,
    VolumePathModel,
    WarehouseModel,
    value_of,
)
from dao_ai.models import get_latest_model_version
from dao_ai.providers.base import ServiceProvider
from dao_ai.utils import (
    dao_ai_version,
    find_dev_wheel,
    get_installed_packages,
    is_lib_provided,
    is_published,
    is_source_layout,
    normalize_host,
    normalize_name,
)
from dao_ai.vector_search import endpoint_exists, index_exists

MAX_NUM_INDEXES: Final[int] = 50

_UUID_RE: Final = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _looks_like_uuid(value: str) -> bool:
    """True if ``value`` is shaped like a service principal application ID (UUID)."""
    return bool(_UUID_RE.match(value))


def with_available_indexes(endpoint: dict[str, Any]) -> bool:
    return endpoint["num_indexes"] < 50


def _workspace_client(
    pat: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    workspace_host: str | None = None,
) -> WorkspaceClient:
    """
    Create a WorkspaceClient instance with the provided parameters.
    If no parameters are provided, it will use the default configuration.
    """
    # Normalize the workspace host to ensure it has https:// scheme
    normalized_host = normalize_host(workspace_host)

    if client_id and client_secret and normalized_host:
        return WorkspaceClient(
            host=normalized_host,
            client_id=client_id,
            client_secret=client_secret,
            auth_type="oauth-m2m",
        )
    elif pat:
        return WorkspaceClient(host=normalized_host, token=pat, auth_type="pat")
    else:
        return WorkspaceClient()


def _vector_search_client(
    pat: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    workspace_host: str | None = None,
) -> VectorSearchClient:
    """
    Create a VectorSearchClient instance with the provided parameters.
    If no parameters are provided, it will use the default configuration.
    """
    # Normalize the workspace host to ensure it has https:// scheme
    normalized_host = normalize_host(workspace_host)

    if client_id and client_secret and normalized_host:
        return VectorSearchClient(
            workspace_url=normalized_host,
            service_principal_client_id=client_id,
            service_principal_client_secret=client_secret,
        )
    elif pat and normalized_host:
        return VectorSearchClient(
            workspace_url=normalized_host,
            personal_access_token=pat,
        )
    else:
        return VectorSearchClient()


def _function_client(w: WorkspaceClient | None = None) -> DatabricksFunctionClient:
    return DatabricksFunctionClient(w=w)


def _collect_resources_with_obo_flag(
    config: AppConfig,
) -> Sequence[IsDatabricksResource]:
    """Flatten every declared resource in ``config.resources`` into a single
    list while preserving the ``on_behalf_of_user`` flag on each entry.

    Tables on Genie rooms that aren't separately declared under
    ``config.resources.tables`` are pulled in so they appear in the auth
    policy too (the ``update_genie_tables`` validator may have run before
    Genie spaces were resolved).
    """
    if config.resources is None:
        return ()

    llms: Sequence[InferenceEndpointModel] = list(config.resources.models.values())
    vector_indexes: Sequence[VectorStoreModel] = list(
        config.resources.vector_stores.values()
    )
    warehouses: Sequence[WarehouseModel] = list(config.resources.warehouses.values())
    genie_rooms: Sequence[GenieRoomModel] = list(config.resources.genie_rooms.values())
    functions: Sequence[FunctionModel] = list(config.resources.functions.values())
    connections: Sequence[ConnectionModel] = list(config.resources.connections.values())
    databases: Sequence[DatabaseModel] = list(config.resources.databases.values())
    volumes: Sequence[VolumeModel] = list(config.resources.volumes.values())
    apps: Sequence[DatabricksAppModel] = list(config.resources.apps.values())

    tables_list: list[TableModel] = list(config.resources.tables.values())
    existing_table_names: set[str] = {t.full_name for t in tables_list}
    for genie_room in genie_rooms:
        for table in genie_room.tables:
            if table.full_name not in existing_table_names:
                tables_list.append(table)
                existing_table_names.add(table.full_name)

    return (
        list(llms)
        + list(vector_indexes)
        + list(warehouses)
        + list(genie_rooms)
        + list(functions)
        + tables_list
        + list(connections)
        + list(databases)
        + list(volumes)
        + list(apps)
    )


def build_auth_policy(config: AppConfig) -> AuthPolicy:
    """Build the MLflow ``AuthPolicy`` for a Model Serving deploy from an AppConfig.

    Partitions every resource by ``on_behalf_of_user``:

    - SP-backed (``on_behalf_of_user=False``) resources are flattened into
      the :class:`SystemAuthPolicy` so the Model Serving endpoint SP gets
      auto-granted the required permissions on each one at deploy time.
    - OBO (``on_behalf_of_user=True``) resources contribute their
      ``api_scopes`` to the :class:`UserAuthPolicy` instead, so the user's
      forwarded token has the right OAuth scopes for runtime calls.

    A resource never appears in *both* outputs.

    Trace tables from ``config.app.trace_location`` are intentionally NOT
    included — MLflow's tracing writer on Model Serving endpoints uses
    a separate authentication path that ``agents.deploy(resources=…)``
    auto-auth does not cover. See :meth:`TraceLocationModel.as_resources`
    for the empirical finding. On Databricks Apps, tracing works via an
    explicit post-deploy grant against the App's own runtime SP.

    Pure function: no I/O, no mutation of ``config``. Safe to unit-test.
    """
    all_models: Sequence[IsDatabricksResource] = _collect_resources_with_obo_flag(
        config
    )

    system_resources: list[DatabricksResource] = [
        resource
        for r in all_models
        for resource in r.as_resources()
        if not r.on_behalf_of_user
    ]

    if config.app and config.app.trace_location:
        # Currently a no-op (returns []). Kept as a call site so future
        # non-trace-table resources can attach to trace_location without
        # touching this function.
        system_resources.extend(config.app.trace_location.as_resources())

    # Translate resource-level api_scopes to canonical OBO user scopes — the
    # same translation used by the Databricks Apps path — so the deployed
    # model's forwarded user token carries the strings the Apps platform
    # recognizes (e.g. ``sql``, ``genie``, ``files``, ``vector-search``).
    # Without this, the user token would claim dao-ai's internal
    # ``sql.warehouses`` / ``dashboards.genie`` strings, which the platform
    # rejects.
    from dao_ai.apps.resources import generate_user_api_scopes

    api_scopes: list[str] = generate_user_api_scopes(config)

    return AuthPolicy(
        system_auth_policy=SystemAuthPolicy(resources=system_resources),
        user_auth_policy=UserAuthPolicy(api_scopes=api_scopes),
    )


def _link_experiment_trace_location(config: AppConfig, experiment_id: str) -> None:
    """Link an MLflow experiment to its UC trace location.

    Wraps ``mlflow.set_experiment(experiment_id=..., trace_location=
    UnityCatalog(...))`` — the post-MLflow-3.11 blessed API. Replaces the
    deprecated combination of ``mlflow.tracing.set_destination(
    UCSchemaLocation(...))`` + ``mlflow.tracing.enablement.
    set_experiment_trace_location(...)`` which emit deprecation warnings.

    This call (when reaching a runnable warehouse) creates the OTEL trace
    Delta tables in the configured UC schema and registers the experiment
    as a writer of those tables. It auto-starts a STOPPED warehouse and
    waits up to 1200s for RUNNING.

    The behavior is identical for Model Serving and Apps callers — both
    paths need the experiment linked to its UC trace destination before
    any code path that reads the model's auth_policy resources (Model
    Serving register_model / agents.deploy) or before the App's runtime
    handlers.py reaches its first request.

    Args:
        config: The AppConfig. No-op if ``config.app.trace_location`` is None.
        experiment_id: The MLflow experiment ID to link.
    """
    if not (config.app and config.app.trace_location):
        return

    from mlflow.entities import UnityCatalog

    loc = config.app.trace_location
    uc_kwargs: dict[str, Any] = {
        "catalog_name": loc.catalog_name,
        "schema_name": loc.schema_name,
    }
    table_prefix = loc.resolved_table_prefix
    if table_prefix:
        uc_kwargs["table_prefix"] = table_prefix

    try:
        mlflow.set_experiment(
            experiment_id=experiment_id,
            trace_location=UnityCatalog(**uc_kwargs),
        )
        logger.info(
            "Linked experiment to UC trace location",
            catalog=loc.catalog_name,
            schema=loc.schema_name,
            table_prefix=table_prefix,
        )
    except mlflow.exceptions.RestException as e:
        # The link is idempotent for our purposes — re-linking a schema
        # that the experiment already writes to is a no-op. The platform
        # rejects re-linking when the experiment already has traces with
        # the "already contains traces" error, which we tolerate. Other
        # RestExceptions (warehouse timeouts, schema permission errors)
        # surface so the caller can fail loudly.
        if "already contains traces" in str(e):
            logger.warning(
                "UC trace destination already linked or experiment has "
                "existing traces, skipping",
                experiment_id=experiment_id,
            )
        else:
            raise


def _grant_experiment_permissions_to_principal(
    principal: str,
    experiment_id: str,
) -> None:
    """Grant a service principal ``CAN_EDIT`` on an MLflow experiment.

    MLflow's tracing writer refuses ``set_experiment(experiment_id=...)``
    — and thus any subsequent trace write — unless the calling identity
    has at least ``CAN_READ`` on the experiment. Traces are writes, so
    we grant ``CAN_EDIT``. Without this the boot-time
    ``mlflow.set_experiment`` call in ``apps/model_serving.py`` crashes
    with ``PERMISSION_DENIED: User <sp> does not have permission to
    'View' experiment with id <id>``.

    Independent of Unity Catalog — required for any tracing writes,
    UC-backed or workspace-default alike.

    Uses ``PATCH /api/2.0/permissions/experiments/<id>`` which appends
    an ACL entry rather than replacing (verified against fevm). Prefers
    the typed SDK ``w.experiments.update_permissions`` when available,
    falling back to the raw REST call otherwise. Idempotent — repeated
    calls with the same principal + permission_level are a no-op.

    WARN-and-continue on failure so a deployer without ``CAN_MANAGE``
    on the experiment can still ship — the runtime will surface the
    misconfiguration at boot if the pre-provisioned grants are missing.

    Args:
        principal: Service principal client id (UUID) to grant.
        experiment_id: MLflow experiment id (numeric string).
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    try:
        try:
            from databricks.sdk.service.ml import (
                ExperimentAccessControlRequest,
                ExperimentPermissionLevel,
            )

            w.experiments.update_permissions(
                experiment_id=experiment_id,
                access_control_list=[
                    ExperimentAccessControlRequest(
                        service_principal_name=principal,
                        permission_level=ExperimentPermissionLevel.CAN_EDIT,
                    )
                ],
            )
        except (ImportError, AttributeError):
            w.api_client.do(
                "PATCH",
                f"/api/2.0/permissions/experiments/{experiment_id}",
                body={
                    "access_control_list": [
                        {
                            "service_principal_name": principal,
                            "permission_level": "CAN_EDIT",
                        }
                    ]
                },
            )
        logger.debug(
            "Granted CAN_EDIT on experiment",
            principal=principal,
            experiment_id=experiment_id,
        )
    except Exception as e:
        logger.warning(
            "Failed to grant experiment ACL — verify the calling identity has "
            "CAN_MANAGE on the experiment, or set app.manage_permissions=false "
            "when the admin has already provisioned this grant",
            principal=principal,
            experiment_id=experiment_id,
            error=str(e),
        )


def _grant_uc_trace_table_permissions_to_principal(
    principal: str,
    catalog_name: str,
    schema_name: str,
    table_prefix: str,
) -> None:
    """Grant the UC privileges MLflow tracing needs on OTEL trace tables.

    Per `Store OpenTelemetry traces in Unity Catalog`_, the writer of
    MLflow traces to a UC-backed experiment needs:

    * ``USE_CATALOG`` on the catalog,
    * ``USE_SCHEMA`` on the schema,
    * ``MODIFY`` + ``SELECT`` on each of the four OTEL Delta tables
      (``<prefix>_otel_{spans,logs,metrics,annotations}``).

    The docs explicitly note that ``ALL_PRIVILEGES`` at the schema level
    is NOT sufficient for UC trace tables: ``MODIFY`` and ``SELECT``
    must be granted explicitly at the table level.

    dao-ai calls this helper after ``agents.deploy`` (Model Serving)
    and after ``apps.create_and_wait`` (Databricks Apps), passing the
    endpoint or app service principal identifier. Idempotent — repeated
    calls with the same grants no-op on the UC side.

    Complementary to :func:`_grant_experiment_permissions_to_principal`
    — the experiment ACL grant is required for any tracing writes; this
    UC helper is only needed when the experiment's trace destination is
    a UC catalog.schema (``trace_location`` is set).

    Args:
        principal: The UC principal identifier to grant to. For a
            Databricks-managed service principal, this is the
            application_id (UUID). For a user, the email.
        catalog_name: UC catalog containing the trace tables.
        schema_name: UC schema containing the trace tables.
        table_prefix: Prefix used by ``mlflow.set_experiment(
            trace_location=UnityCatalog(table_prefix=...))``. When
            ``table_prefix`` was omitted by the caller of MLflow, this
            is the experiment_id (MLflow's default).

    .. _Store OpenTelemetry traces in Unity Catalog:
       https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog
    """
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    # The typed `grants.update()` SDK method serializes ``SecurableType`` as
    # ``SECURABLETYPE.TABLE`` on some SDK versions and the REST API rejects
    # it (``Invalid input: SECURABLETYPE.TABLE is not a valid securable
    # type``). Call the raw REST endpoint directly with lowercase-string
    # securable_type — that path works uniformly across SDK versions.
    def _update(securable_type: str, full_name: str, privileges: list[str]) -> None:
        try:
            w.api_client.do(
                "PATCH",
                f"/api/2.1/unity-catalog/permissions/{securable_type}/{full_name}",
                body={
                    "changes": [
                        {"principal": principal, "add": list(privileges)}
                    ]
                },
            )
            logger.debug(
                "Granted UC privileges for trace persistence",
                principal=principal,
                securable_type=securable_type,
                full_name=full_name,
                privileges=privileges,
            )
        except Exception as e:
            logger.warning(
                "Failed to grant UC privilege for trace persistence — "
                "verify the calling identity has GRANT rights on the resource",
                principal=principal,
                securable_type=securable_type,
                full_name=full_name,
                error=str(e),
            )

    _update("catalog", catalog_name, ["USE_CATALOG"])
    _update("schema", f"{catalog_name}.{schema_name}", ["USE_SCHEMA"])
    for suffix in ("spans", "logs", "metrics", "annotations"):
        _update(
            "table",
            f"{catalog_name}.{schema_name}.{table_prefix}_otel_{suffix}",
            ["SELECT", "MODIFY"],
        )
    logger.info(
        "Granted trace-write privileges to principal",
        principal=principal,
        catalog=catalog_name,
        schema=schema_name,
        table_prefix=table_prefix,
    )


def _resolve_trace_table_prefix(
    config: AppConfig, experiment_id: Optional[str]
) -> str:
    """Return the table prefix MLflow uses for OTEL trace tables.

    Mirrors ``mlflow.set_experiment(trace_location=UnityCatalog(
    table_prefix=<resolved_or_None>))``: an explicit ``table_prefix`` on
    ``TraceLocationModel`` wins; otherwise MLflow uses the experiment
    id. See :meth:`TraceLocationModel.resolved_table_prefix`.

    Raises when neither source is available — callers must supply an
    ``experiment_id`` fallback whenever ``trace_location.table_prefix``
    is unset.
    """
    if config.app and config.app.trace_location:
        prefix = config.app.trace_location.resolved_table_prefix
        if prefix:
            return prefix
    if not experiment_id:
        raise ValueError(
            "cannot resolve OTEL trace-table prefix: neither "
            "trace_location.table_prefix nor experiment_id is set"
        )
    return experiment_id


def _build_wheel(project_root: Path) -> Path:
    """Build a dao-ai wheel from source using uv."""
    import subprocess

    dist_dir = project_root / "dist"
    dist_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        ["uv", "build", "--wheel"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Wheel build failed: {result.stderr.strip()}")

    wheels = sorted(dist_dir.glob("dao_ai-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise RuntimeError(f"No wheel found in {dist_dir} after build")
    return wheels[-1]


class DatabricksProvider(ServiceProvider):
    def __init__(
        self,
        w: WorkspaceClient | None = None,
        vsc: VectorSearchClient | None = None,
        dfs: DatabricksFunctionClient | None = None,
        pat: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        workspace_host: str | None = None,
    ) -> None:
        # Store credentials for lazy initialization
        self._pat = pat
        self._client_id = client_id
        self._client_secret = client_secret
        self._workspace_host = workspace_host

        # Lazy initialization for WorkspaceClient
        self._w: WorkspaceClient | None = w
        self._w_initialized = w is not None

        # Lazy initialization for VectorSearchClient - only create when needed
        # This avoids authentication errors in Databricks Apps where VSC
        # requires explicit credentials but the platform uses ambient auth
        self._vsc: VectorSearchClient | None = vsc
        self._vsc_initialized = vsc is not None

        # Lazy initialization for DatabricksFunctionClient
        self._dfs: DatabricksFunctionClient | None = dfs
        self._dfs_initialized = dfs is not None

    @property
    def w(self) -> WorkspaceClient:
        """Lazy initialization of WorkspaceClient."""
        if not self._w_initialized:
            self._w = _workspace_client(
                pat=self._pat,
                client_id=self._client_id,
                client_secret=self._client_secret,
                workspace_host=self._workspace_host,
            )
            self._w_initialized = True
        return self._w  # type: ignore[return-value]

    @w.setter
    def w(self, value: WorkspaceClient) -> None:
        """Set WorkspaceClient and mark as initialized."""
        self._w = value
        self._w_initialized = True

    @property
    def vsc(self) -> VectorSearchClient:
        """Lazy initialization of VectorSearchClient."""
        if not self._vsc_initialized:
            self._vsc = _vector_search_client(
                pat=self._pat,
                client_id=self._client_id,
                client_secret=self._client_secret,
                workspace_host=self._workspace_host,
            )
            self._vsc_initialized = True
        return self._vsc  # type: ignore[return-value]

    @vsc.setter
    def vsc(self, value: VectorSearchClient) -> None:
        """Set VectorSearchClient and mark as initialized."""
        self._vsc = value
        self._vsc_initialized = True

    @property
    def dfs(self) -> DatabricksFunctionClient:
        """Lazy initialization of DatabricksFunctionClient."""
        if not self._dfs_initialized:
            self._dfs = _function_client(w=self.w)
            self._dfs_initialized = True
        return self._dfs  # type: ignore[return-value]

    @dfs.setter
    def dfs(self, value: DatabricksFunctionClient) -> None:
        """Set DatabricksFunctionClient and mark as initialized."""
        self._dfs = value
        self._dfs_initialized = True

    def experiment_name(self, config: AppConfig) -> str:
        """Resolve the experiment path from ``app.experiment.name``, or
        fall back to ``/Users/<deployer_email>/<app.name>`` when not set.

        The id-based ``app.experiment.id`` branch never lands here — it
        short-circuits ``get_or_create_experiment`` directly.
        """
        if config.app.experiment is not None and config.app.experiment.resolved_name:
            return config.app.experiment.resolved_name
        current_user: User = self.w.current_user.me()
        name: str = config.app.name
        return f"/Users/{current_user.user_name}/{name}"

    def get_or_create_experiment(self, config: AppConfig) -> Experiment:
        """Resolve to an MLflow ``Experiment``.

        * ``app.experiment`` set → delegate to ``self.create_experiment``
          (id takes precedence; name resolved lazily with optional create,
          per the dao-ai ``Model.create(w)`` convention).
        * otherwise → fall back to the historical default
          ``/Users/<deployer_email>/<app.name>`` with restore-if-deleted,
          create-if-missing.
        """
        if config.app.experiment is not None:
            return self.create_experiment(config.app.experiment)

        experiment_name: str = self.experiment_name(config)
        experiment: Experiment | None = mlflow.get_experiment_by_name(experiment_name)

        if experiment is not None and experiment.lifecycle_stage == "deleted":
            client: MlflowClient = MlflowClient()
            client.restore_experiment(experiment.experiment_id)
            logger.info(
                "Restored deleted experiment",
                experiment_id=experiment.experiment_id,
            )
            experiment = mlflow.get_experiment(experiment.experiment_id)
        elif experiment is None:
            experiment_id: str = mlflow.create_experiment(name=experiment_name)
            logger.success(
                "Created new MLflow experiment",
                experiment_name=experiment_name,
                experiment_id=experiment_id,
            )
            experiment = mlflow.get_experiment(experiment_id)

        return experiment

    def create_token(self) -> str:
        current_user: User = self.w.current_user.me()
        logger.debug("Authenticated to Databricks", user=str(current_user))
        headers: dict[str, str] = self.w.config.authenticate()
        token: str = headers["Authorization"].replace("Bearer ", "")
        return token

    def get_secret(
        self, secret_scope: str, secret_key: str, default_value: str | None = None
    ) -> str:
        try:
            secret_response: GetSecretResponse = self.w.secrets.get_secret(
                secret_scope, secret_key
            )
            logger.trace(
                "Retrieved secret", secret_key=secret_key, secret_scope=secret_scope
            )
            encoded_secret: str = secret_response.value
            decoded_secret: str = base64.b64decode(encoded_secret).decode("utf-8")
            return decoded_secret
        except NotFound:
            logger.warning(
                "Secret not found, using default value",
                secret_key=secret_key,
                secret_scope=secret_scope,
            )
        except Exception as e:
            logger.error(
                "Error retrieving secret",
                secret_key=secret_key,
                secret_scope=secret_scope,
                error=str(e),
            )

        return default_value

    def create_agent(
        self,
        config: AppConfig,
    ) -> ModelInfo:
        logger.info("Creating agent")
        mlflow.set_registry_uri("databricks-uc")

        # Set up experiment for proper tracking
        experiment: Experiment = self.get_or_create_experiment(config)
        mlflow.set_experiment(experiment_id=experiment.experiment_id)
        logger.debug(
            "Using MLflow experiment",
            experiment_name=experiment.name,
            experiment_id=experiment.experiment_id,
        )

        # Link experiment to UC trace location BEFORE log_model + register_model.
        # When the model is logged with auth_policy listing the 3 OTEL trace
        # tables (via TraceLocationModel.as_resources()), MLflow's
        # mlflow.register_model validates those resources by calling
        # generate-temporary-credentials on the model version. If the OTEL
        # tables don't exist yet, that call returns 404 / TABLE_DOES_NOT_EXIST
        # and the registration fails. mlflow.set_experiment(trace_location=...)
        # is what creates the tables — it also auto-starts a STOPPED warehouse
        # and waits for it. Calling it here, before log_model, guarantees the
        # tables exist when the model is registered. The same step is repeated
        # idempotently in deploy_model_serving_agent so that re-deploys (which
        # skip create_agent) still set up the link.
        _link_experiment_trace_location(config, experiment.experiment_id)

        auth_policy: AuthPolicy = build_auth_policy(config)
        logger.debug(
            "Auth policy created",
            system_resource_count=len(auth_policy.system_auth_policy.resources),
            user_api_scopes=list(auth_policy.user_auth_policy.api_scopes),
        )

        code_paths: list[str] = config.app.code_paths
        for path in code_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Code path does not exist: {path}")

        model_root_path: Path = Path(dao_ai.__file__).parent
        model_path: Path = model_root_path / "apps" / "model_serving.py"

        pip_requirements: Sequence[str] = config.app.pip_requirements

        if is_published():
            if not is_lib_provided("dao-ai", pip_requirements):
                pip_requirements += [
                    f"dao-ai=={dao_ai_version()}",
                ]
            logger.info(
                "dao-ai source: PyPI package",
                version=dao_ai_version(),
            )
        else:
            dev_wheel: Path | None = find_dev_wheel()
            if dev_wheel:
                code_paths.append(dev_wheel.as_posix())
                pip_requirements += [f"code/{dev_wheel.name}"]
                logger.info(
                    "dao-ai source: local wheel bundled via code_paths",
                    wheel=dev_wheel.name,
                    path=str(dev_wheel),
                )
            else:
                src_path: Path = model_root_path.parent
                if is_source_layout(model_root_path):
                    directories: list[Path] = [
                        d for d in src_path.iterdir() if d.is_dir()
                    ]
                    for directory in directories:
                        code_paths.append(directory.as_posix())
                    logger.info(
                        "dao-ai source: local source directories via code_paths",
                        source_root=str(src_path),
                        directories=[d.name for d in directories],
                    )
                else:
                    logger.warning(
                        "No dev wheel found and dao-ai is in site-packages. "
                        "Build a wheel with 'uv build --wheel' for reliable deployment.",
                    )

            pip_requirements += get_installed_packages()

        from dao_ai.skills import collect_skills_code_paths

        code_paths.extend(collect_skills_code_paths(config))

        code_paths = list(dict.fromkeys(code_paths))

        logger.trace("Pip requirements prepared", count=len(pip_requirements))
        logger.trace("Code paths prepared", count=len(code_paths))

        run_name: str = normalize_name(config.app.name)
        logger.debug(
            "Agent run configuration",
            run_name=run_name,
            model_path=model_path.as_posix(),
        )

        input_example: dict[str, Any] = None
        if config.app.input_example:
            input_example = config.app.input_example.model_dump()

        logger.trace("Input example configured", has_example=input_example is not None)

        # Create conda environment with configured Python version
        # This allows deploying from environments with different Python versions
        # (e.g., Databricks Apps with Python 3.11 can deploy to Model Serving with 3.12)
        target_python_version: str = config.app.python_version
        logger.debug("Target Python version configured", version=target_python_version)

        conda_env: dict[str, Any] = {
            "name": "mlflow-env",
            "channels": ["conda-forge"],
            "dependencies": [
                f"python={target_python_version}",
                "pip",
                {"pip": list(pip_requirements)},
            ],
        }
        logger.trace(
            "Conda environment configured",
            python_version=target_python_version,
            pip_packages_count=len(pip_requirements),
        )

        # End any stale runs before starting to ensure clean state on retry
        if mlflow.active_run():
            logger.warning(
                "Ending stale MLflow run before creating new agent",
                run_id=mlflow.active_run().info.run_id,
            )
            mlflow.end_run()

        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("type", "agent")
                mlflow.set_tag("dao_ai", dao_ai_version())
                logged_agent_info: ModelInfo = mlflow.pyfunc.log_model(
                    python_model=model_path.as_posix(),
                    code_paths=code_paths,
                    # exclude_none=True keeps the serialized config compatible
                    # with older dao-ai versions that may be installed in the
                    # Model Serving container — the registry can lag the
                    # development branch by several versions, and any new
                    # ``Optional`` field on a Pydantic model with ``extra=
                    # "forbid"`` becomes a load-time error if it's persisted
                    # as null and the deployed code doesn't know the field
                    # name yet. Dropping null values is forward-compatible.
                    model_config=config.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    ),
                    name="agent",
                    conda_env=conda_env,
                    input_example=input_example,
                    # resources=all_resources,
                    auth_policy=auth_policy,
                )
        except Exception as e:
            # Ensure run is ended on failure to prevent stale state on retry
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
            logger.error(
                "Failed to log model",
                run_name=run_name,
                error=str(e),
            )
            raise

        if config.app.registered_model is None:
            raise ValueError(
                "registered_model is required in app config for model registration. "
                "Please add a registered_model section to your config."
            )

        registered_model_name: str = config.app.registered_model.full_name

        model_version: ModelVersion = mlflow.register_model(
            name=registered_model_name, model_uri=logged_agent_info.model_uri
        )
        logger.success(
            "Model registered",
            model_name=registered_model_name,
            version=model_version.version,
        )

        client: MlflowClient = MlflowClient()

        # Set tags on the model version
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version.version,
            key="dao_ai",
            value=dao_ai_version(),
        )
        logger.trace("Set dao_ai tag on model version", version=model_version.version)

        client.set_registered_model_alias(
            name=registered_model_name,
            alias="Champion",
            version=model_version.version,
        )

        if config.app.alias:
            client.set_registered_model_alias(
                name=registered_model_name,
                alias=config.app.alias,
                version=model_version.version,
            )
            aliased_model: ModelVersion = client.get_model_version_by_alias(
                registered_model_name, config.app.alias
            )
            logger.info(
                "Model aliased",
                model_name=registered_model_name,
                alias=config.app.alias,
                version=aliased_model.version,
            )

    def _serving_endpoint_exists(self, endpoint_name: str) -> bool:
        try:
            self.w.serving_endpoints.get(endpoint_name)
            return True
        except NotFound:
            return False

    def _wait_serving_endpoint_config_idle(
        self,
        endpoint_name: str,
        *,
        timeout_seconds: float = 600.0,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        """Block until serving endpoint has no IN_PROGRESS config update, or endpoint is missing."""
        deadline: float = time.monotonic() + timeout_seconds
        while True:
            try:
                ep = self.w.serving_endpoints.get(endpoint_name)
            except NotFound:
                return
            state = ep.state
            config_update = state.config_update if state else None
            if config_update != EndpointStateConfigUpdate.IN_PROGRESS:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Serving endpoint {endpoint_name!r} config update still IN_PROGRESS "
                    f"after {timeout_seconds}s"
                )
            logger.info(
                "Waiting for serving endpoint config update to finish",
                endpoint_name=endpoint_name,
            )
            time.sleep(poll_interval_seconds)

    def deploy_model_serving_agent(self, config: AppConfig) -> None:
        """
        Deploy agent to Databricks Model Serving endpoint.

        This is the original deployment method that creates/updates a Model Serving
        endpoint with the registered model.

        Args:
            config: The AppConfig containing deployment configuration
        """
        logger.info(
            "Deploying agent to Model Serving", endpoint_name=config.app.endpoint_name
        )

        # Warn about Lakebase autoscaling on Model Serving. MLflow's
        # DatabricksLakebase resource doesn't support autoscaling projects
        # (https://github.com/mlflow/mlflow/issues/22452), so dao-ai
        # intentionally skips that resource emission. The deployed endpoint
        # has no auto-bound Lakebase grant -- the agent must manage auth
        # itself via OAuth M2M (client_id / client_secret on the
        # DatabaseModel).
        if config.resources and config.resources.databases:
            for db_key, db in config.resources.databases.items():
                if db.is_lakebase and db.project and not db.client_id:
                    logger.warning(
                        "Lakebase autoscaling on Model Serving requires manual "
                        "auth -- MLflow's DatabricksLakebase resource doesn't "
                        "support autoscaling projects. Set `client_id` and "
                        "`client_secret` on the database, or deploy to "
                        "Databricks Apps (which supports autoscaling Lakebase "
                        "natively via the postgres resource binding). See "
                        "https://github.com/mlflow/mlflow/issues/22452.",
                        database=db_key,
                        project=db.project,
                    )

        mlflow.set_registry_uri("databricks-uc")

        endpoint_name: str = config.app.endpoint_name
        if config.app.registered_model is None:
            raise ValueError(
                "registered_model is required in app config for deployment. "
                "Please add a registered_model section to your config."
            )
        registered_model_name: str = config.app.registered_model.full_name
        scale_to_zero: bool = config.app.scale_to_zero
        # Model Serving expects string env-var values. SecretVariableModel
        # instances stringify to ``{{secrets/scope/key}}`` — the platform
        # resolves that literal at endpoint boot. PrimitiveVariable and raw
        # strings pass through. Without this coercion, ``agents.deploy``
        # sees Pydantic-model values and silently drops them; the ensuing
        # empty ``DATABRICKS_CLIENT_ID/SECRET`` at runtime was one of the
        # observations behind the "stripped by platform" claim in 1b4290c.
        environment_vars: dict[str, str] = {
            k: str(v) if v is not None else ""
            for k, v in (config.app.environment_vars or {}).items()
        }
        workload_size: str = config.app.workload_size
        tags: dict[str, str] = config.app.tags.copy() if config.app.tags else {}

        # Add dao_ai framework tag
        tags["dao_ai"] = dao_ai_version()

        latest_version: int = get_latest_model_version(registered_model_name)

        try:
            chain_deployments = agents.get_deployments(registered_model_name)
            logger.debug(
                "Agent chain deployments for registered model",
                model_name=registered_model_name,
                count=len(chain_deployments),
            )
        except Exception as e:
            logger.debug(
                "get_deployments failed (non-fatal)",
                model_name=registered_model_name,
                error=str(e),
            )

        serving_exists: bool = self._serving_endpoint_exists(endpoint_name)
        if serving_exists:
            logger.debug(
                "Serving endpoint exists; skipping user tags on deploy to reduce patch+update races",
                endpoint_name=endpoint_name,
            )
            self._wait_serving_endpoint_config_idle(endpoint_name)

        tags_kw: dict[str, str] | None = None if serving_exists else tags

        # Resolve the MLflow experiment ONCE — used for:
        #   1. injecting ``MLFLOW_EXPERIMENT_ID`` into env_vars (symmetric
        #      with ``DATABRICKS_CLIENT_ID/SECRET`` — always set at deploy
        #      time so the boot code in apps/model_serving.py can find it)
        #   2. linking to UC trace_location (if configured)
        #   3. granting experiment CAN_EDIT + UC table perms to the SP
        experiment: Experiment = self.get_or_create_experiment(config)
        experiment_id: str = str(experiment.experiment_id)
        environment_vars.setdefault("MLFLOW_EXPERIMENT_ID", experiment_id)

        # Link experiment to UC trace location BEFORE agents.deploy — the
        # tracing writer expects the four OTEL tables to exist at endpoint
        # boot. Idempotent; no-op if the experiment already has a UC trace
        # destination linked.
        if config.app.trace_location:
            _link_experiment_trace_location(config, experiment_id)

        max_attempts: int = 6
        for attempt in range(1, max_attempts + 1):
            try:
                agents.deploy(
                    endpoint_name=endpoint_name,
                    model_name=registered_model_name,
                    model_version=latest_version,
                    scale_to_zero=scale_to_zero,
                    environment_vars=environment_vars,
                    workload_size=workload_size,
                    tags=tags_kw,
                )
                break
            except ValueError as e:
                err_msg: str = str(e)
                if "currently updating" in err_msg.lower() and attempt < max_attempts:
                    wait_s: float = min(30.0, 5.0 * attempt)
                    logger.warning(
                        "Serving endpoint busy, retrying agents.deploy",
                        endpoint_name=endpoint_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        wait_seconds=wait_s,
                        error=err_msg,
                    )
                    time.sleep(wait_s)
                    self._wait_serving_endpoint_config_idle(
                        endpoint_name, timeout_seconds=120.0
                    )
                    continue
                raise

        # Stage-1 diagnostic (MS-trace-persistence investigation):
        # reflect what env vars actually landed on the endpoint's
        # served-entity config after ``agents.deploy``. Compare with
        # what dao-ai sent (``environment_vars`` above) to answer:
        # did DATABRICKS_CLIENT_ID/SECRET survive, or did the platform
        # strip them? Logs the redacted view — secret values are
        # elided but presence + length round-trip is visible.
        try:
            from dao_ai.diagnostics import redacted_env_var_map

            endpoint_after = self.w.serving_endpoints.get(name=endpoint_name)
            landed_env: dict[str, str] = {}
            if endpoint_after.config and endpoint_after.config.served_entities:
                landed_env = (
                    endpoint_after.config.served_entities[0].environment_vars or {}
                )
            logger.info(
                "dao_ai.diagnostic.deploy_env_reflection",
                endpoint_name=endpoint_name,
                sent=redacted_env_var_map(environment_vars),
                landed=redacted_env_var_map(landed_env),
                sent_only=sorted(set(environment_vars) - set(landed_env)),
                landed_only=sorted(set(landed_env) - set(environment_vars)),
            )
        except Exception as e:
            logger.debug(
                "Post-deploy env-var reflection failed (non-fatal diagnostic)",
                endpoint_name=endpoint_name,
                error=str(e),
            )

        # Grant the runtime SP the permissions MLflow tracing needs.
        # Same two-path pattern as deploy_apps_agent — experiment ACL
        # is gated on SP alone (needed for any tracing writes), UC table
        # grants are additionally gated on trace_location. Both are
        # gated on ``manage_permissions`` so a deployer without GRANT
        # rights can opt out and rely on admin pre-provisioning.
        #
        # For Model Serving the SP is the caller-declared one from
        # ``config.app.service_principal.client_id`` (resolved into
        # ``DATABRICKS_CLIENT_ID`` on the endpoint). Contrast with Apps,
        # where the SP is auto-generated by the platform.
        if config.app.manage_permissions and config.app.service_principal is not None:
            try:
                sp_id: str = str(value_of(config.app.service_principal.client_id))
                _grant_experiment_permissions_to_principal(
                    principal=sp_id,
                    experiment_id=experiment_id,
                )
                if config.app.trace_location:
                    table_prefix: str = _resolve_trace_table_prefix(
                        config,
                        None
                        if config.app.trace_location.resolved_table_prefix
                        else experiment_id,
                    )
                    _grant_uc_trace_table_permissions_to_principal(
                        principal=sp_id,
                        catalog_name=config.app.trace_location.catalog_name,
                        schema_name=config.app.trace_location.schema_name,
                        table_prefix=table_prefix,
                    )
            except Exception as e:
                logger.warning(
                    "Failed to grant trace-persistence privileges to Model "
                    "Serving SP",
                    endpoint_name=endpoint_name,
                    error=str(e),
                )

        permissions: Sequence[dict[str, Any]] = config.app.permissions

        logger.debug(
            "Configuring model permissions",
            model_name=registered_model_name,
            permissions_count=len(permissions),
        )

        for permission in permissions:
            principals: Sequence[str] = permission.principals
            entitlements: Sequence[str] = permission.entitlements

            if not principals or not entitlements:
                continue
            for entitlement in entitlements:
                set_permissions(
                    model_name=registered_model_name,
                    users=principals,
                    permission_level=PermissionLevel[entitlement],
                )

        # Register production monitoring scorers if configured
        if config.app.monitoring:
            from dao_ai.evaluation import register_monitoring_scorers

            if not config.app.trace_location:
                experiment = self.get_or_create_experiment(config)

            sql_warehouse_id: str | None = (
                config.app.trace_location.warehouse_id
                if config.app.trace_location
                else None
            )
            registered_scorers = register_monitoring_scorers(
                monitoring_config=config.app.monitoring,
                experiment_id=experiment.experiment_id,
                sql_warehouse_id=sql_warehouse_id,
            )
            logger.info(
                "Production monitoring scorers registered for Model Serving",
                scorer_count=len(registered_scorers),
            )

    def deploy_apps_agent(self, config: AppConfig) -> None:
        """
        Deploy agent as a Databricks App.

        This method creates or updates a Databricks App that serves the agent
        using the app_server module.

        The deployment process:
        1. Determine the workspace source path for the app
        2. Upload the configuration file to the workspace
        3. Create the app if it doesn't exist
        4. Deploy the app

        Args:
            config: The AppConfig containing deployment configuration

        Note:
            The config file must be loaded via AppConfig.from_file() so that
            the source_config_path is available for upload.
        """
        import io

        from databricks.sdk.service.apps import (
            App,
            AppDeployment,
            AppDeploymentMode,
            AppDeploymentState,
            ApplicationState,
        )

        # Normalize app name: lowercase, replace underscores with dashes
        raw_name: str = config.app.name
        app_name: str = raw_name.lower().replace("_", "-")
        if app_name != raw_name:
            logger.info(
                "Normalized app name for Databricks Apps",
                original=raw_name,
                normalized=app_name,
            )

        logger.info("Deploying agent to Databricks Apps", app_name=app_name)

        # Use convention-based workspace path: /Workspace/Users/{user}/apps/{app_name}
        current_user: User = self.w.current_user.me()
        user_name: str = current_user.user_name or "default"
        source_path: str = f"/Workspace/Users/{user_name}/apps/{app_name}"

        logger.info("Using workspace source path", source_path=source_path)

        # Get or create experiment for this app (for tracing and tracking)
        from mlflow.entities import Experiment

        experiment: Experiment = self.get_or_create_experiment(config)
        logger.info(
            "Using MLflow experiment for app",
            experiment_name=experiment.name,
            experiment_id=experiment.experiment_id,
        )

        # Link experiment to UC trace location BEFORE App creation. Apps'
        # auto-created SP needs the experiment-trace destination configured
        # ahead of the first request, otherwise control-plane export
        # (unreachable from Apps) is what handlers.py would pick up.
        _link_experiment_trace_location(config, experiment.experiment_id)

        # Register production monitoring scorers if configured
        if config.app.monitoring:
            from dao_ai.evaluation import register_monitoring_scorers

            sql_warehouse_id: str | None = (
                config.app.trace_location.warehouse_id
                if config.app.trace_location
                else None
            )
            registered_scorers = register_monitoring_scorers(
                monitoring_config=config.app.monitoring,
                experiment_id=experiment.experiment_id,
                sql_warehouse_id=sql_warehouse_id,
            )
            logger.info(
                "Production monitoring scorers registered for app",
                scorer_count=len(registered_scorers),
            )

        # Upload the configuration file to the workspace.
        #
        # Three input shapes feed this step:
        #   1. AppConfig.from_file(path, params={...}) — substituted text
        #      is on config.rendered_yaml. Prefer that.
        #   2. Legacy AppConfig.from_file callers without params= — fall
        #      back to reading the raw source file from disk.
        #   3. AppConfig built in pure Python — neither rendered_yaml nor
        #      a source file exists, so serialize the in-memory model
        #      back to YAML and ship that.
        source_config_path: str | None = config.source_config_path
        config_file_name: str = "dao_ai.yaml"
        workspace_config_path: str = f"{source_path}/{config_file_name}"

        rendered: str | None = config.rendered_yaml
        config_content: bytes
        config_origin: str
        if rendered is not None:
            config_content = rendered.encode("utf-8")
            config_origin = "rendered_yaml (parameter-substituted)"
        elif source_config_path:
            with open(source_config_path, "rb") as f:
                config_content = f.read()
            config_origin = f"source file {source_config_path}"
        else:
            # Python-built AppConfig: serialize the in-memory object.
            config_dict: dict[str, Any] = config.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
            config_content = yaml.safe_dump(
                config_dict, sort_keys=False, default_flow_style=False
            ).encode("utf-8")
            config_origin = "in-memory AppConfig (programmatic)"

        logger.info(
            "Uploading config file to workspace",
            source=config_origin,
            destination=workspace_config_path,
        )

        # Clean the workspace directory to remove stale artifacts from
        # previous deployments (old wheels, leftover src/, etc.).
        try:
            self.w.workspace.delete(source_path, recursive=True)
        except Exception:
            pass  # Directory may not exist yet
        try:
            self.w.workspace.mkdirs(source_path)
        except Exception as e:
            logger.debug(f"Directory may already exist: {e}")

        # Upload the config file
        self.w.workspace.upload(
            path=workspace_config_path,
            content=io.BytesIO(config_content),
            format=ImportFormat.AUTO,
            overwrite=True,
        )
        logger.info("Config file uploaded", path=workspace_config_path)

        # Determine install command based on dev vs published mode.
        # Respect config.app.enable_chat_proxy (default True) so the deployed
        # app spawns the chat UI alongside the agent backend, matching the
        # behavior of `dao-ai generate-bundle`.
        enable_chat_proxy: bool = (
            config.app.enable_chat_proxy
            if config.app and config.app.enable_chat_proxy is not None
            else True
        )
        entrypoint_module: str = (
            "dao_ai.apps.start_app" if enable_chat_proxy else "dao_ai.apps.server"
        )
        if is_published():
            # Ship a ``requirements.txt`` (pinned) + ``pyproject.toml``
            # (pinned) so Databricks Apps' build phase installs the dao-ai
            # version the bundle declares. Apps' build step recognizes
            # ``requirements.txt`` directly (``pip install -r requirements.txt``)
            # and emits ``Updated file: python/source_code/requirements.txt``
            # in the deployment log; ``pyproject.toml`` alone (without
            # ``uv.lock``) is NOT recognized — the build step logs ``No
            # dependencies file found. Skipping installation.`` and the
            # venv persists from prior deploys to the same app slot.
            #
            # Without this pin, the previous default startup command
            # (``uv pip install dao-ai`` with no ``--upgrade``) only audited
            # the cached venv and never upgraded — meaning the deployed app
            # silently drifted behind the local dao-ai. Surfaced by the
            # workshop verification on 2026-06-23: Lab 15 introduced
            # ``app.background:``, but the cached venv was a pre-rename
            # dao-ai that rejected the new field as ``extra_forbidden`` and
            # crashed the app at startup.
            from dao_ai.apps.bundle import _PYPROJECT_TEMPLATE

            app_name_normalized = raw_name.lower().replace("_", "-")
            package_name = app_name_normalized.replace("-", "_")
            pyproject_content = _PYPROJECT_TEMPLATE.format(
                name=app_name_normalized,
                package_name=package_name,
                dao_ai_version=dao_ai_version(),
            )
            self.w.workspace.upload(
                path=f"{source_path}/pyproject.toml",
                content=io.BytesIO(pyproject_content.encode("utf-8")),
                format=ImportFormat.AUTO,
                overwrite=True,
            )

            requirements_content = f"dao-ai>={dao_ai_version()}\n"
            self.w.workspace.upload(
                path=f"{source_path}/requirements.txt",
                content=io.BytesIO(requirements_content.encode("utf-8")),
                format=ImportFormat.AUTO,
                overwrite=True,
            )

            # Stub package __init__.py so the hatch build target ``packages =
            # ["src/<package_name>"]`` declared in ``_PYPROJECT_TEMPLATE``
            # resolves during ``uv sync``.
            try:
                self.w.workspace.mkdirs(f"{source_path}/src/{package_name}")
            except Exception:
                pass
            self.w.workspace.upload(
                path=f"{source_path}/src/{package_name}/__init__.py",
                content=io.BytesIO(b""),
                format=ImportFormat.AUTO,
                overwrite=True,
            )

            app_command = ["python", "-m", entrypoint_module]
            logger.info(
                "dao-ai source for app: PyPI package",
                version=dao_ai_version(),
                entrypoint=entrypoint_module,
                chat_proxy=enable_chat_proxy,
            )
        else:
            dev_wheel: Path | None = find_dev_wheel()

            if not dev_wheel:
                # No pre-built wheel -- build from source
                project_root = Path(__file__).parents[3]
                if not (project_root / "pyproject.toml").exists():
                    raise RuntimeError(
                        "No dao-ai wheel found and project source not available. "
                        "Build a wheel first with: uv build --wheel"
                    )
                logger.info(
                    "No dev wheel found, building from source",
                    project_root=str(project_root),
                )
                dev_wheel = _build_wheel(project_root)

            wheel_path: Path = dev_wheel
            logger.info("Using dev wheel for app deployment", wheel=wheel_path.name)

            # Upload wheel to workspace source path under dist/
            try:
                self.w.workspace.mkdirs(f"{source_path}/dist")
            except Exception:
                pass
            workspace_wheel_path = f"{source_path}/dist/{wheel_path.name}"
            with open(wheel_path, "rb") as f:
                self.w.workspace.upload(
                    path=workspace_wheel_path,
                    content=io.BytesIO(f.read()),
                    format=ImportFormat.AUTO,
                    overwrite=True,
                )
            logger.info("Dev wheel uploaded", path=workspace_wheel_path)

            # Upload pyproject.toml (metadata + hatch build target for any
            # user code under src/<package_name>/). Deps install from
            # requirements.txt at Apps build time.
            from dao_ai.apps.bundle import (
                _PYPROJECT_DEV_TEMPLATE,
                _make_requirements_txt,
            )

            app_name_normalized = raw_name.lower().replace("_", "-")
            package_name = app_name_normalized.replace("-", "_")
            pyproject_content = _PYPROJECT_DEV_TEMPLATE.format(
                name=app_name_normalized,
                package_name=package_name,
            )
            self.w.workspace.upload(
                path=f"{source_path}/pyproject.toml",
                content=io.BytesIO(pyproject_content.encode("utf-8")),
                format=ImportFormat.AUTO,
                overwrite=True,
            )

            # Create stub package __init__.py
            try:
                self.w.workspace.mkdirs(f"{source_path}/src/{package_name}")
            except Exception:
                pass
            self.w.workspace.upload(
                path=f"{source_path}/src/{package_name}/__init__.py",
                content=io.BytesIO(b""),
                format=ImportFormat.AUTO,
                overwrite=True,
            )

            # Ship a requirements.txt that points at the bundled wheel.
            # Apps' build phase recognizes requirements.txt directly and
            # runs `pip install -r requirements.txt` — pip installs the
            # wheel and resolves transitive deps from public PyPI via the
            # wheel's metadata. Replaces the prior uv.lock-based path
            # which baked Databricks-internal pypi-proxy URLs into the
            # lock and timed out on Apps containers.
            requirements_content: str = _make_requirements_txt(
                development=True, wheel_filename=wheel_path.name
            )
            self.w.workspace.upload(
                path=f"{source_path}/requirements.txt",
                content=io.BytesIO(requirements_content.encode("utf-8")),
                format=ImportFormat.AUTO,
                overwrite=True,
            )
            logger.info(
                "dao-ai source for app: dev wheel (requirements.txt)",
                wheel=wheel_path.name,
                entrypoint=entrypoint_module,
                chat_proxy=enable_chat_proxy,
            )
            app_command = ["python", "-m", entrypoint_module]

        # The chat UI (e2e-chatbot-app-next) is cloned and built at runtime
        # by start_app.py, matching the official Databricks agent template
        # pattern.  No pre-build or archive upload is needed here.

        # Generate and upload app.yaml with dynamically discovered resources
        from dao_ai.apps.resources import generate_app_yaml

        app_yaml_content: str = generate_app_yaml(
            config,
            command=app_command,
            include_resources=True,
        )

        app_yaml_path: str = f"{source_path}/app.yaml"
        self.w.workspace.upload(
            path=app_yaml_path,
            content=io.BytesIO(app_yaml_content.encode("utf-8")),
            format=ImportFormat.AUTO,
            overwrite=True,
        )
        logger.info("app.yaml with resources uploaded", path=app_yaml_path)

        # Generate deployment resources as raw dicts for the REST API.
        # This includes all resource types, even those not yet supported
        # by the SDK enum (e.g. VECTOR_SEARCH_INDEX).
        from dao_ai.apps.resources import (
            generate_deployment_resources,
            generate_user_api_scopes,
        )

        deployment_resources = generate_deployment_resources(
            config, experiment_id=experiment.experiment_id
        )
        if deployment_resources:
            logger.info(
                "Discovered app resources from config",
                resource_count=len(deployment_resources),
                resources=[r.get("name") for r in deployment_resources],
            )

        # Generate user API scopes for on-behalf-of-user resources
        user_api_scopes = generate_user_api_scopes(config)
        if user_api_scopes:
            logger.info(
                "Discovered user API scopes for OBO resources",
                scopes=user_api_scopes,
            )

        # Check if app exists
        app_exists: bool = False
        try:
            existing_app: App = self.w.apps.get(name=app_name)
            app_exists = True
            logger.debug("App already exists, updating", app_name=app_name)
        except NotFound:
            logger.debug("Creating new app", app_name=app_name)

        # Create or update the app using the REST API to support all
        # resource types including those not yet in the SDK enum.
        app_body: dict = {
            "name": app_name,
            "description": config.app.description or f"DAO AI Agent: {app_name}",
        }
        if deployment_resources:
            app_body["resources"] = deployment_resources
        if user_api_scopes:
            app_body["user_api_scopes"] = user_api_scopes

        def _set_app_resources(method: str, path: str, body: dict) -> None:
            """Create or update app via REST API.

            If the workspace doesn't support ``VECTOR_SEARCH_INDEX`` as a UC
            securable type, retry once without those resources (capability
            gap, not a permissions issue).

            Permission-related failures are raised immediately with an
            actionable message — the deployer must hold ``MANAGE`` on every
            resource the app needs, otherwise the deployed app will start
            without the grants it needs and crash at request time.
            """
            _vector_search_unsupported = (
                "VECTOR_SEARCH_INDEX",
                "vector_search_index is not a supported",
            )
            _permission_patterns = (
                "permission to grant",
                "MANAGE permission on the resource",
                "Can View",
            )

            def _describe_resources(rs: list) -> str:
                return ", ".join(r.get("name", "?") for r in rs[:20]) or "(none)"

            try:
                self.w.api_client.do(method, path, body=body)
            except (BadRequest, InvalidParameterValue, PermissionDenied) as e:
                err_msg = str(e)

                if any(p in err_msg for p in _vector_search_unsupported):
                    kept = [
                        r
                        for r in body.get("resources", [])
                        if r.get("uc_securable", {}).get("securable_type")
                        != "VECTOR_SEARCH_INDEX"
                    ]
                    logger.warning(
                        "Workspace does not support VECTOR_SEARCH_INDEX "
                        "resources; retrying without them. Grant vector "
                        "search access manually after deploy.",
                        error=err_msg,
                    )
                    body["resources"] = kept
                    self.w.api_client.do(method, path, body=body)
                    return

                if any(p in err_msg for p in _permission_patterns):
                    requested = body.get("resources", [])
                    logger.error(
                        "App deploy requires the deployer to hold MANAGE on "
                        "every declared resource so the platform can grant "
                        "the app service principal access. The deploy will "
                        "be aborted to avoid leaving the app in a broken "
                        "state. Resolve by either (a) running the deploy as "
                        "a principal that owns/manages the underlying "
                        "catalog, schema, functions, and serving endpoints, "
                        "or (b) granting MANAGE on each of those securables "
                        "to the current deployer. Requested resources: "
                        f"{_describe_resources(requested)}",
                        error=err_msg,
                    )

                raise

        if not app_exists:
            logger.info("Creating Databricks App", app_name=app_name)
            _set_app_resources("POST", "/api/2.0/apps", app_body)
            # Wait for app to be ready
            app = self.w.apps.wait_get_app_active(name=app_name)
            logger.info("App created", app_name=app.name, app_url=app.url)

            # Ensure user_api_scopes are set — the fallback retry may
            # have created the app without them if resource granting
            # failed on the first attempt.
            if user_api_scopes:
                current_app = self.w.api_client.do("GET", f"/api/2.0/apps/{app_name}")
                if current_app.get("user_api_scopes") != user_api_scopes:
                    self.w.api_client.do(
                        "PATCH",
                        f"/api/2.0/apps/{app_name}",
                        body={
                            "name": app_name,
                            "user_api_scopes": user_api_scopes,
                        },
                    )
                    logger.info(
                        "User API scopes set on app",
                        scopes=user_api_scopes,
                    )
        else:
            app = existing_app
            # Update resources and scopes on existing app
            if deployment_resources or user_api_scopes:
                logger.info("Updating app resources and scopes", app_name=app_name)
                if not deployment_resources:
                    app_body.pop("resources", None)
                _set_app_resources("PATCH", f"/api/2.0/apps/{app_name}", app_body)
                app = self.w.apps.get(name=app_name)
                logger.info("App resources and scopes updated", app_name=app_name)

        # Deploy requires the app's compute to be active. The Apps API rejects
        # `start` when compute is already ACTIVE, so only call start when
        # compute is actually stopped — otherwise let the subsequent deploy
        # recover any unhealthy app process.
        fresh_app: App = self.w.apps.get(name=app_name)
        app_status = fresh_app.app_status
        compute_status = fresh_app.compute_status
        compute_state = (
            compute_status.state.value
            if compute_status is not None and compute_status.state is not None
            else None
        )
        if compute_state in (None, "STOPPED"):
            logger.info(
                "Databricks App compute not active; starting before deploy",
                app_name=app_name,
                compute_state=compute_state,
                app_state=(
                    app_status.state.value
                    if app_status is not None and app_status.state is not None
                    else None
                ),
            )
            self.w.apps.start_and_wait(name=app_name)
            fresh_app = self.w.apps.get(name=app_name)
        elif app_status is None or app_status.state != ApplicationState.RUNNING:
            logger.info(
                "Databricks App compute active but app not RUNNING; deploy will redeploy",
                app_name=app_name,
                compute_state=compute_state,
                app_state=(
                    app_status.state.value
                    if app_status is not None and app_status.state is not None
                    else None
                ),
            )
        app = fresh_app

        # Deploy the app with source code
        # The app will use the dao_ai.apps.server module as the entry point
        logger.info("Deploying app", app_name=app_name)

        # Create deployment configuration
        app_deployment = AppDeployment(
            mode=AppDeploymentMode.SNAPSHOT,
            source_code_path=source_path,
        )

        # Deploy the app
        deployment: AppDeployment = self.w.apps.deploy_and_wait(
            app_name=app_name,
            app_deployment=app_deployment,
        )

        if (
            deployment.status
            and deployment.status.state == AppDeploymentState.SUCCEEDED
        ):
            logger.info(
                "App deployed successfully",
                app_name=app_name,
                deployment_id=deployment.deployment_id,
                app_url=app.url if app else None,
            )
        else:
            status_message: str = (
                deployment.status.message if deployment.status else "Unknown error"
            )
            logger.error(
                "App deployment failed",
                app_name=app_name,
                status=status_message,
            )
            raise RuntimeError(f"App deployment failed: {status_message}")

        # Grant the App's runtime SP the permissions MLflow tracing needs.
        # Two independent grant paths, gated separately:
        #   1. Experiment CAN_EDIT — required for any tracing writes
        #      (workspace-default OR UC-backed). Gate: SP known.
        #   2. UC OTEL-table SELECT+MODIFY — only needed when the
        #      experiment's trace destination is a UC schema. Gate:
        #      trace_location set.
        # Both are additionally gated on ``config.app.manage_permissions``
        # so a deployer without GRANT rights can skip the attempt when
        # an admin has pre-provisioned everything.
        if config.app.manage_permissions:
            try:
                # Refresh App to pick up the assigned SP identifier.
                fresh_app: App = self.w.apps.get(name=app_name)
                sp_id: Optional[str] = (
                    fresh_app.service_principal_client_id
                    or fresh_app.service_principal_id
                )
                if not sp_id:
                    logger.warning(
                        "Could not resolve App SP identity; "
                        "trace-persistence grants skipped.",
                        app_name=app_name,
                    )
                else:
                    experiment: Experiment = self.get_or_create_experiment(config)
                    experiment_id: str = str(experiment.experiment_id)

                    _grant_experiment_permissions_to_principal(
                        principal=str(sp_id),
                        experiment_id=experiment_id,
                    )

                    if config.app.trace_location:
                        # Resolve the trace-table prefix — prefer explicit
                        # ``trace_location.table_prefix``, else fall back
                        # to experiment_id (MLflow's default when
                        # ``UnityCatalog(table_prefix=None)``).
                        table_prefix: str = _resolve_trace_table_prefix(
                            config,
                            None
                            if config.app.trace_location.resolved_table_prefix
                            else experiment_id,
                        )
                        _grant_uc_trace_table_permissions_to_principal(
                            principal=str(sp_id),
                            catalog_name=config.app.trace_location.catalog_name,
                            schema_name=config.app.trace_location.schema_name,
                            table_prefix=table_prefix,
                        )
            except Exception as e:
                logger.warning(
                    "Failed to grant trace-persistence privileges to App SP",
                    app_name=app_name,
                    error=str(e),
                )

    def deploy_agent(
        self,
        config: AppConfig,
        target: DeploymentTarget = DeploymentTarget.MODEL_SERVING,
    ) -> None:
        """
        Deploy agent to the specified target.

        This is the main deployment method that routes to the appropriate
        deployment implementation based on the target.

        Args:
            config: The AppConfig containing deployment configuration
            target: The deployment target (MODEL_SERVING or APPS)
        """
        if target == DeploymentTarget.BOTH:
            self.deploy_model_serving_agent(config)
            self.deploy_apps_agent(config)
        elif target == DeploymentTarget.MODEL_SERVING:
            self.deploy_model_serving_agent(config)
        elif target == DeploymentTarget.APPS:
            self.deploy_apps_agent(config)
        else:
            raise ValueError(f"Unknown deployment target: {target}")

    def create_catalog(self, schema: SchemaModel) -> CatalogInfo:
        catalog_info: CatalogInfo
        try:
            catalog_info = self.w.catalogs.get(name=schema.catalog_name)
        except NotFound:
            logger.info("Creating catalog", catalog_name=schema.catalog_name)
            catalog_info = self.w.catalogs.create(name=schema.catalog_name)
        return catalog_info

    def create_experiment(
        self, experiment: "ExperimentModel"
    ) -> Experiment:
        """Resolve an ``ExperimentModel`` to a live MLflow ``Experiment``,
        creating the experiment if only ``name`` was provided and it does
        not exist yet.

        Precedence:

        * ``experiment.id`` set → fetch by id via
          ``mlflow.get_experiment(id)``. No create attempted; errors if
          the id is invalid.
        * ``experiment.id`` unset, ``experiment.name`` set →
          ``get_experiment_by_name(name)``. When missing, create iff
          ``experiment.create_if_not_exists`` is True (default), else
          raise so the deployer knows to ask an admin.

        Idempotent — caches on ``experiment._resolved``. Populates
        ``experiment.id`` when it was inferred from ``name`` so
        subsequent calls short-circuit through the id-branch.
        """
        if experiment._resolved and experiment.id is not None:
            return mlflow.get_experiment(str(value_of(experiment.id)))

        if experiment.id is not None:
            experiment_id = str(value_of(experiment.id))
            experiment._resolved = True
            return mlflow.get_experiment(experiment_id)

        # id is None → look up by name, optionally create
        assert experiment.name is not None  # invariant of require_name_or_id
        name = value_of(experiment.name)
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            if not experiment.create_if_not_exists:
                raise ValueError(
                    f"MLflow experiment '{name}' does not exist and "
                    "create_if_not_exists=False. Ask an admin to provision "
                    "the experiment, or set create_if_not_exists=True."
                )
            experiment_id = mlflow.create_experiment(name=name)
            logger.info(
                "Created MLflow experiment",
                experiment_name=name,
                experiment_id=experiment_id,
            )
            exp = mlflow.get_experiment(experiment_id)
        experiment.id = str(exp.experiment_id)
        experiment._resolved = True
        return exp

    def create_schema(self, schema: SchemaModel) -> SchemaInfo:
        catalog_info: CatalogInfo = self.create_catalog(schema)
        schema_info: SchemaInfo
        try:
            schema_info = self.w.schemas.get(full_name=schema.full_name)
        except NotFound:
            logger.info("Creating schema", schema_name=schema.full_name)
            schema_info = self.w.schemas.create(
                name=schema.schema_name, catalog_name=catalog_info.name
            )
        return schema_info

    def create_volume(self, volume: VolumeModel) -> VolumeInfo:
        schema_info: SchemaInfo = self.create_schema(volume.schema_model)
        volume_info: VolumeInfo
        try:
            volume_info = self.w.volumes.read(name=volume.full_name)
        except NotFound:
            logger.info("Creating volume", volume_name=volume.full_name)
            volume_info = self.w.volumes.create(
                catalog_name=schema_info.catalog_name,
                schema_name=schema_info.name,
                name=volume.name,
                volume_type=VolumeType.MANAGED,
            )
        return volume_info

    def create_path(self, volume_path: VolumePathModel) -> Path:
        path: Path = volume_path.full_name
        logger.info("Creating volume path", path=str(path))
        self.w.files.create_directory(path)
        return path

    def create_dataset(self, dataset: DatasetModel) -> None:
        from pyspark.sql import SparkSession

        spark: SparkSession = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError(
                "No active Spark session found. This method requires Spark to be available."
            )

        table: str = dataset.table.full_name

        ddl: str | HasFullName = dataset.ddl
        if isinstance(ddl, HasFullName):
            ddl = ddl.full_name

        data: str | HasFullName = dataset.data
        if isinstance(data, HasFullName):
            data = data.full_name

        format: str = dataset.format
        read_options: dict[str, Any] = dataset.read_options or {}

        args: dict[str, Any] = {}
        for key, value in dataset.parameters.items():
            if isinstance(value, dict):
                schema_model: SchemaModel = SchemaModel(**value)
                value = schema_model.full_name
            args[key] = value

        if not args:
            args = {
                "database": dataset.table.schema_model.full_name,
            }

        if ddl:
            ddl_path: Path = Path(ddl)
            logger.debug("Executing DDL", ddl_path=str(ddl_path))
            statements: Sequence[str] = sqlparse.parse(ddl_path.read_text())
            for statement in statements:
                logger.trace(
                    "Executing DDL statement", statement=str(statement)[:100], args=args
                )
                spark.sql(
                    str(statement),
                    args=args,
                )

        if data:
            data_path: Path = Path(data)
            if format == "sql":
                logger.debug("Executing SQL from file", data_path=str(data_path))
                data_statements: Sequence[str] = sqlparse.parse(data_path.read_text())
                for statement in data_statements:
                    logger.trace(
                        "Executing SQL statement",
                        statement=str(statement)[:100],
                        args=args,
                    )
                    spark.sql(
                        str(statement),
                        args=args,
                    )
            else:
                logger.debug("Writing dataset to table", table=table)
                if not data_path.is_absolute():
                    data_path = Path.cwd() / data_path
                data_path = data_path.resolve()
                logger.trace("Data path resolved", path=str(data_path))

                # Parquet intentionally routed to Spark's distributed reader
                # below — pd.read_parquet on the driver OOMs on serverless for
                # text-heavy datasets (e.g. hardware_store products.parquet:
                # 17 MB on disk → 45 MB in pandas, dominated by the description
                # column).
                pandas_readers: dict[str, Callable[..., pd.DataFrame]] = {
                    "csv": lambda p, **kw: pd.read_csv(p, **kw),
                    "json": lambda p, **kw: pd.read_json(p, **kw),
                    "excel": lambda p, **kw: pd.read_excel(p, **kw),
                }

                reader: Callable[..., pd.DataFrame] | None = pandas_readers.get(format)
                if reader is not None:
                    pdf: pd.DataFrame = reader(str(data_path), **read_options)
                    schema = dataset.table_schema
                    if ddl:
                        target_schema = spark.table(table).schema
                        schema = target_schema
                    df = spark.createDataFrame(pdf, schema=schema)
                else:
                    df = (
                        spark.read.format(format)
                        .options(**read_options)
                        .load(
                            str(data_path),
                            schema=dataset.table_schema,
                        )
                    )

                if ddl:
                    target_cols: list[str] = [
                        f.name for f in spark.table(table).schema.fields
                    ]
                    df = df.select(*target_cols)
                    df.write.insertInto(table, overwrite=True)
                else:
                    df.write.mode("overwrite").saveAsTable(table)

    def create_vector_store(self, vector_store: VectorStoreModel) -> None:
        """
        Create a vector search index from a source table.

        This method expects a VectorStoreModel in provisioning mode with all
        required fields validated. Use VectorStoreModel.create() which handles
        mode detection and validation.

        Args:
            vector_store: VectorStoreModel configured for provisioning
        """
        # Ensure endpoint exists
        if not endpoint_exists(self.vsc, vector_store.endpoint.name):
            create_kwargs: dict[str, Any] = {
                "name": vector_store.endpoint.name,
                "endpoint_type": vector_store.endpoint.type,
                "verbose": True,
            }
            if vector_store.endpoint.target_qps is not None:
                create_kwargs["target_qps"] = vector_store.endpoint.target_qps
            self.vsc.create_endpoint_and_wait(**create_kwargs)
        elif vector_store.endpoint.target_qps is not None:
            logger.debug(
                "endpoint already exists; target_qps not reconciled",
                endpoint_name=vector_store.endpoint.name,
                configured_target_qps=vector_store.endpoint.target_qps,
            )

        logger.success(
            "Vector search endpoint ready", endpoint_name=vector_store.endpoint.name
        )

        if not index_exists(
            self.vsc, vector_store.endpoint.name, vector_store.index.full_name
        ):
            logger.info(
                "Creating vector search index",
                index_name=vector_store.index.full_name,
                endpoint_name=vector_store.endpoint.name,
            )
            self.vsc.create_delta_sync_index_and_wait(
                endpoint_name=vector_store.endpoint.name,
                index_name=vector_store.index.full_name,
                source_table_name=vector_store.source_table.full_name,
                pipeline_type="TRIGGERED",
                primary_key=vector_store.primary_key,
                embedding_source_column=vector_store.embedding_source_column,
                embedding_model_endpoint_name=vector_store.embedding_model.name,
                columns_to_sync=vector_store.columns,
            )
            index = self.vsc.get_index(
                vector_store.endpoint.name, vector_store.index.full_name
            )
        else:
            logger.debug(
                "Vector search index already exists, triggering sync",
                index_name=vector_store.index.full_name,
            )
            index = self.vsc.get_index(
                vector_store.endpoint.name, vector_store.index.full_name
            )
            # Wait for the index to be queryable before issuing sync, so we
            # don't race against a still-provisioning index.
            index.wait_until_ready(verbose=True, wait_for_updates=False)
            index.sync()

        # create_delta_sync_index_and_wait and index.sync() return before the
        # underlying data sync completes. wait_for_updates=True blocks until
        # the index is fully populated (ONLINE_NO_PENDING_UPDATE).
        index.wait_until_ready(verbose=True, wait_for_updates=True)

        logger.success(
            "Vector search index ready",
            index_name=vector_store.index.full_name,
            source_table=vector_store.source_table.full_name,
        )

    def get_vector_index(self, vector_store: VectorStoreModel) -> None:
        index: VectorSearchIndex = self.vsc.get_index(
            vector_store.endpoint.name, vector_store.index.full_name
        )
        return index

    def create_genie_space(self, room: GenieRoomModel) -> Any:
        """Create or update a Databricks Genie space from a ``GenieRoomModel``.

        Uses the Databricks SDK's ``WorkspaceClient.genie`` API exclusively:

        - ``create_space`` when ``room.space_id`` is unset.
        - ``update_space`` when ``room.space_id`` is set and the locally-built
          ``serialized_space`` differs from the one stored on the workspace.
          The current ``etag`` is supplied so concurrent edits are rejected
          rather than silently lost.
        - ``permissions.set`` to apply any configured entitlements.

        Returns the resulting ``GenieSpace`` SDK object.
        """
        import json

        if room.warehouse is None:
            raise ValueError(
                "GenieRoomModel.warehouse must be set to provision a Genie space. "
                "Provide a WarehouseModel with a name or warehouse_id."
            )

        # Defer to ensure_resolved so the warehouse_id is resolved from name
        # if needed before we commit it to the API call.
        room.warehouse.ensure_resolved()
        warehouse_id: str = value_of(room.warehouse.warehouse_id)
        if not warehouse_id:
            raise ValueError(
                f"Could not resolve warehouse_id for warehouse '{room.warehouse.name}'."
            )

        title: str = room.name or "Untitled Genie Space"
        description: str | None = room.description
        parent_path: str | None = (
            value_of(room.parent_path) if room.parent_path else None
        )
        serialized_payload: dict[str, Any] = room._build_serialized_space()
        serialized_str: str = json.dumps(serialized_payload, sort_keys=True)

        space_id: str | None = value_of(room.space_id) if room.space_id else None

        space: Any
        if space_id:
            existing = self.w.genie.get_space(
                space_id=space_id, include_serialized_space=True
            )
            existing_serialized: dict[str, Any] = {}
            if getattr(existing, "serialized_space", None):
                try:
                    existing_serialized = json.loads(existing.serialized_space)
                except json.JSONDecodeError:
                    existing_serialized = {}

            needs_update: bool = (
                existing_serialized != serialized_payload
                or (existing.title or None) != title
                or (existing.description or None) != description
                or (getattr(existing, "warehouse_id", None) or None) != warehouse_id
            )
            if needs_update:
                logger.info(
                    "Updating Genie space",
                    space_id=space_id,
                    title=title,
                )
                space = self.w.genie.update_space(
                    space_id=space_id,
                    title=title,
                    description=description,
                    serialized_space=serialized_str,
                    warehouse_id=warehouse_id,
                    etag=getattr(existing, "etag", None),
                )
            else:
                logger.info(
                    "Genie space already up to date; skipping update",
                    space_id=space_id,
                )
                space = existing
        else:
            logger.info(
                "Creating Genie space",
                title=title,
                warehouse_id=warehouse_id,
                parent_path=parent_path,
            )
            space = self.w.genie.create_space(
                warehouse_id=warehouse_id,
                serialized_space=serialized_str,
                title=title,
                description=description,
                parent_path=parent_path,
            )
            new_id: str | None = getattr(space, "space_id", None)
            if not new_id:
                raise RuntimeError(
                    "Genie space create_space did not return a space_id."
                )
            room.space_id = new_id
            logger.success("Genie space created", space_id=new_id, title=title)

        if room.entitlements:
            self._apply_genie_entitlements(value_of(room.space_id), room.entitlements)

        # Invalidate cached space details so subsequent property reads see the
        # post-write state.
        room._space_details = None
        return space

    def _apply_genie_entitlements(
        self, space_id: str, entitlements: Sequence[GenieEntitlement]
    ) -> None:
        """Apply workspace permission grants on a Genie space.

        Each entitlement maps to one or more ``AccessControlRequest`` entries
        applied via ``WorkspaceClient.permissions.set`` against object type
        ``genie``. Email-shaped principals are sent as ``user_name``,
        application-id-shaped principals as ``service_principal_name``, and
        anything else as ``group_name``.
        """
        from databricks.sdk.service.iam import (
            AccessControlRequest,
            PermissionLevel,
        )

        access_control: list[AccessControlRequest] = []
        for entitlement in entitlements:
            level_str: str = (
                entitlement.permission_level.value
                if hasattr(entitlement.permission_level, "value")
                else str(entitlement.permission_level)
            )
            try:
                level = PermissionLevel(level_str)
            except ValueError:
                logger.warning(
                    "Unknown permission level for Genie space; skipping",
                    level=level_str,
                )
                continue
            for principal in entitlement.principals:
                principal_str: str = value_of(principal)
                kwargs: dict[str, str] = {"permission_level": level}
                if "@" in principal_str and "." in principal_str:
                    kwargs["user_name"] = principal_str
                elif _looks_like_uuid(principal_str):
                    kwargs["service_principal_name"] = principal_str
                else:
                    kwargs["group_name"] = principal_str
                access_control.append(AccessControlRequest(**kwargs))

        if not access_control:
            return

        logger.info(
            "Applying Genie space entitlements",
            space_id=space_id,
            count=len(access_control),
        )
        self.w.permissions.set(
            request_object_type="genie",
            request_object_id=space_id,
            access_control_list=access_control,
        )

    def create_sql_function(
        self, unity_catalog_function: UnityCatalogFunctionSqlModel
    ) -> None:
        function: FunctionModel = unity_catalog_function.function
        schema: SchemaModel = function.schema_model
        ddl_path: Path = Path(unity_catalog_function.ddl)
        parameters: dict[str, Any] = unity_catalog_function.parameters

        statements: Sequence[str] = [
            str(s) for s in sqlparse.parse(ddl_path.read_text())
        ]

        if not parameters:
            parameters = {
                "catalog_name": schema.catalog_name,
                "schema_name": schema.schema_name,
            }

        for sql in statements:
            for key, value in parameters.items():
                if isinstance(value, HasFullName):
                    value = value.full_name
                sql = sql.replace(f"{{{key}}}", value)

            # sql = sql.replace("{catalog_name}", schema.catalog_name)
            # sql = sql.replace("{schema_name}", schema.schema_name)

            logger.info("Creating SQL function", function_name=function.name)
            logger.trace("SQL function body", sql=sql[:200])
            _: FunctionInfo = self.dfs.create_function(sql_function_body=sql)

            if unity_catalog_function.test:
                logger.debug(
                    "Testing function",
                    function_name=function.full_name,
                    parameters=unity_catalog_function.test.parameters,
                )

                result: FunctionExecutionResult = self.dfs.execute_function(
                    function_name=function.full_name,
                    parameters=unity_catalog_function.test.parameters,
                )

                if result.error:
                    logger.error(
                        "Function test failed",
                        function_name=function.full_name,
                        error=result.error,
                    )
                else:
                    logger.success(
                        "Function test passed", function_name=function.full_name
                    )
                    logger.debug("Function test result", result=str(result))

    def find_columns(self, table_model: TableModel) -> Sequence[str]:
        logger.trace("Finding columns for table", table=table_model.full_name)
        table_info: TableInfo = self.w.tables.get(full_name=table_model.full_name)
        columns: Sequence[ColumnInfo] = table_info.columns
        column_names: Sequence[str] = [c.name for c in columns]
        logger.debug(
            "Columns found",
            table=table_model.full_name,
            columns_count=len(column_names),
        )
        return column_names

    def find_primary_key(self, table_model: TableModel) -> Sequence[str] | None:
        logger.trace("Finding primary key for table", table=table_model.full_name)
        primary_keys: Sequence[str] | None = None
        table_info: TableInfo = self.w.tables.get(full_name=table_model.full_name)
        constraints: Sequence[TableConstraint] = table_info.table_constraints
        primary_key_constraint: PrimaryKeyConstraint | None = next(
            (c.primary_key_constraint for c in constraints if c.primary_key_constraint),
            None,
        )
        if primary_key_constraint:
            primary_keys = primary_key_constraint.child_columns

        logger.debug(
            "Primary key found", table=table_model.full_name, primary_keys=primary_keys
        )
        return primary_keys

    def find_vector_search_endpoint(
        self, predicate: Callable[[dict[str, Any]], bool]
    ) -> str | None:
        logger.trace("Finding vector search endpoint")
        endpoint_name: str | None = None
        vector_search_endpoints: Sequence[dict[str, Any]] = (
            self.vsc.list_endpoints().get("endpoints", [])
        )
        for endpoint in vector_search_endpoints:
            if predicate(endpoint):
                endpoint_name = endpoint["name"]
                break
        logger.debug("Vector search endpoint found", endpoint_name=endpoint_name)
        return endpoint_name

    def find_endpoint_for_index(self, index_model: IndexModel) -> str | None:
        logger.trace(
            "Finding endpoint for vector search index", index_name=index_model.full_name
        )
        all_endpoints: Sequence[dict[str, Any]] = self.vsc.list_endpoints().get(
            "endpoints", []
        )
        index_name: str = index_model.full_name
        found_endpoint_name: str | None = None
        for endpoint in all_endpoints:
            endpoint_name: str = endpoint["name"]
            indexes = self.vsc.list_indexes(name=endpoint_name)
            vector_indexes: Sequence[dict[str, Any]] = indexes.get("vector_indexes", [])
            logger.trace(
                "Checking endpoint for indexes",
                endpoint_name=endpoint_name,
                indexes_count=len(vector_indexes),
            )
            index_names = [vector_index["name"] for vector_index in vector_indexes]
            if index_name in index_names:
                found_endpoint_name = endpoint_name
                break
        logger.debug(
            "Vector search index endpoint found",
            index_name=index_model.full_name,
            endpoint_name=found_endpoint_name,
        )
        return found_endpoint_name

    def create_lakebase_autoscaling(self, database: DatabaseModel) -> None:
        """
        Create a Lakebase Autoscaling project using the Postgres API.

        Handles idempotent project creation, gracefully handling cases where
        the project already exists.
        """
        import time

        from databricks.sdk.service.postgres import (
            Project,
            ProjectDefaultEndpointSettings,
            ProjectSpec,
        )

        workspace_client: WorkspaceClient = database.workspace_client
        project_name = f"projects/{database.project}"

        try:
            existing_project = workspace_client.postgres.get_project(project_name)
            if existing_project:
                logger.info(
                    "Autoscaling Lakebase project already exists",
                    project=database.project,
                )

                # Check endpoint status
                try:
                    branches = list(
                        workspace_client.postgres.list_branches(project_name)
                    )
                    if branches:
                        default_branch = next(
                            (b for b in branches if b.status and b.status.default),
                            branches[0],
                        )
                        endpoints = list(
                            workspace_client.postgres.list_endpoints(
                                default_branch.name
                            )
                        )
                        for ep in endpoints:
                            if ep.status and ep.status.current_state == "ACTIVE":
                                logger.info(
                                    "Autoscaling Lakebase endpoint is ACTIVE",
                                    project=database.project,
                                )
                                return
                            elif ep.status and ep.status.current_state == "INIT":
                                logger.info(
                                    "Autoscaling Lakebase endpoint initializing, waiting",
                                    project=database.project,
                                )
                                max_wait = 300
                                elapsed = 0
                                while elapsed < max_wait:
                                    time.sleep(10)
                                    elapsed += 10
                                    eps = list(
                                        workspace_client.postgres.list_endpoints(
                                            default_branch.name
                                        )
                                    )
                                    if (
                                        eps
                                        and eps[0].status
                                        and eps[0].status.current_state == "ACTIVE"
                                    ):
                                        logger.success(
                                            "Autoscaling Lakebase endpoint is now ACTIVE",
                                            project=database.project,
                                        )
                                        return
                                logger.warning(
                                    "Timed out waiting for endpoint to become ACTIVE",
                                    project=database.project,
                                )
                                return
                except Exception as ep_err:
                    logger.warning(
                        "Could not check endpoint status",
                        project=database.project,
                        error=str(ep_err),
                    )
                return

        except NotFound:
            logger.info(
                "Creating new autoscaling Lakebase project",
                project=database.project,
            )

            try:
                min_cu = database.autoscaling_min_cu or 2
                max_cu = database.autoscaling_max_cu or 4

                endpoint_kwargs: dict[str, Any] = {
                    "autoscaling_limit_min_cu": min_cu,
                    "autoscaling_limit_max_cu": max_cu,
                }

                suspend_seconds = database.suspend_timeout_seconds
                if suspend_seconds is not None and suspend_seconds <= 0:
                    endpoint_kwargs["no_suspension"] = True
                elif suspend_seconds is not None and suspend_seconds >= 60:
                    from google.protobuf.duration_pb2 import (
                        Duration as ProtobufDuration,
                    )

                    dur = ProtobufDuration()
                    dur.FromJsonString(f"{suspend_seconds}s")
                    endpoint_kwargs["suspend_timeout_duration"] = dur

                project = Project(
                    spec=ProjectSpec(
                        default_endpoint_settings=ProjectDefaultEndpointSettings(
                            **endpoint_kwargs
                        ),
                    ),
                )

                workspace_client.postgres.create_project(
                    project=project,
                    project_id=database.project,
                )

                logger.success(
                    "Autoscaling Lakebase project created",
                    project=database.project,
                )

                # Wait for the endpoint to become ACTIVE
                max_wait = 300
                elapsed = 0
                while elapsed < max_wait:
                    time.sleep(10)
                    elapsed += 10
                    try:
                        branches = list(
                            workspace_client.postgres.list_branches(project_name)
                        )
                        if branches:
                            default_branch = next(
                                (b for b in branches if b.status and b.status.default),
                                branches[0],
                            )
                            endpoints = list(
                                workspace_client.postgres.list_endpoints(
                                    default_branch.name
                                )
                            )
                            if (
                                endpoints
                                and endpoints[0].status
                                and endpoints[0].status.current_state == "ACTIVE"
                            ):
                                logger.success(
                                    "Autoscaling Lakebase endpoint is now ACTIVE",
                                    project=database.project,
                                )
                                return
                    except Exception:
                        pass

                logger.warning(
                    "Timed out waiting for autoscaling Lakebase to become ACTIVE",
                    project=database.project,
                )
                return

            except Exception as create_error:
                error_msg = str(create_error)
                if (
                    "already exists" in error_msg.lower()
                    or "RESOURCE_ALREADY_EXISTS" in error_msg
                ):
                    logger.info(
                        "Autoscaling Lakebase project created concurrently",
                        project=database.project,
                    )
                    return
                logger.error(
                    "Error creating autoscaling Lakebase project",
                    project=database.project,
                    error=error_msg,
                )
                raise

        except Exception as e:
            error_msg = str(e)
            if (
                "already exists" in error_msg.lower()
                or "RESOURCE_ALREADY_EXISTS" in error_msg
            ):
                logger.info(
                    "Autoscaling Lakebase project already exists (detected via exception)",
                    project=database.project,
                )
                return
            logger.error(
                "Unexpected error while handling autoscaling Lakebase project",
                project=database.project,
                error=error_msg,
            )
            raise

    def _resolve_autoscaling_default_branch(
        self, workspace_client: WorkspaceClient, project: str
    ) -> str:
        """Resolve the default branch name for an autoscaling Lakebase project."""
        project_name = f"projects/{project}"
        branches = list(workspace_client.postgres.list_branches(project_name))
        if not branches:
            raise ValueError(
                f"No branches found for autoscaling Lakebase project '{project}'."
            )
        default_branch = next(
            (b for b in branches if b.status and b.status.default),
            branches[0],
        )
        return default_branch.name

    def create_lakebase_autoscaling_role(self, database: DatabaseModel) -> None:
        """
        Create a role for a service principal on an autoscaling Lakebase project.

        Roles are created at the branch level in the autoscaling Postgres API.
        """
        from databricks.sdk.service.postgres import (
            Role,
            RoleIdentityType,
            RoleMembershipRole,
            RoleRoleSpec,
        )

        from dao_ai.config import value_of

        if not database.client_id:
            logger.warning(
                "client_id required to create autoscaling role",
                project=database.project,
            )
            return

        client_id: str | None = value_of(database.client_id)
        if not client_id:
            logger.warning(
                "client_id resolved to None; skipping autoscaling role creation. "
                "Check that the configured source (secret scope, env var, etc.) "
                "is populated.",
                project=database.project,
                client_id_spec=database.client_id,
            )
            return

        workspace_client: WorkspaceClient = database.workspace_client

        # Roles are created on a branch, not on the project
        branch_name = self._resolve_autoscaling_default_branch(
            workspace_client, database.project
        )

        # role_id must match ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ so sanitize the client_id
        # Must start with a lowercase letter, so prefix with 'sp-' for service principal
        import re

        sanitized = re.sub(r"[^a-z0-9-]", "-", client_id.lower()).strip("-")
        sanitized_role_id = f"sp-{sanitized}"[:63]

        logger.debug(
            "Creating autoscaling Lakebase role",
            role_name=client_id,
            role_id=sanitized_role_id,
            project=database.project,
            branch=branch_name,
        )

        try:
            # Check if role already exists
            try:
                role_resource_name = f"{branch_name}/roles/{sanitized_role_id}"
                _ = workspace_client.postgres.get_role(name=role_resource_name)
                logger.info(
                    "Autoscaling Lakebase role already exists",
                    role_name=client_id,
                    project=database.project,
                )
                return
            except NotFound:
                logger.debug(
                    "Autoscaling role not found, creating",
                    role_name=client_id,
                )

            role = Role(
                spec=RoleRoleSpec(
                    postgres_role=client_id,
                    identity_type=RoleIdentityType.SERVICE_PRINCIPAL,
                    membership_roles=[RoleMembershipRole.DATABRICKS_SUPERUSER],
                ),
            )

            workspace_client.postgres.create_role(
                parent=branch_name,
                role=role,
                role_id=sanitized_role_id,
            )

            logger.success(
                "Autoscaling Lakebase role created",
                role_name=client_id,
                project=database.project,
            )

        except Exception as e:
            error_msg = str(e)
            if (
                "already exists" in error_msg.lower()
                or "RESOURCE_ALREADY_EXISTS" in error_msg
            ):
                logger.info(
                    "Autoscaling Lakebase role created concurrently",
                    role_name=client_id,
                    project=database.project,
                )
                return
            logger.error(
                "Error creating autoscaling Lakebase role",
                role_name=client_id,
                project=database.project,
                error=error_msg,
            )
            raise

    def get_prompt(self, prompt_model: PromptModel) -> PromptVersion:
        """
        Load prompt from MLflow Prompt Registry with fallback logic.

        If an explicit version or alias is specified in the prompt_model, uses that directly.
        Otherwise, tries to load prompts in this order:
        1. champion alias
        2. latest alias
        3. default alias
        4. Register default_template if provided (only if register_to_registry=True)
        5. Use default_template directly (fallback)

        The auto_register field controls whether the default_template is automatically
        synced to the prompt registry:
        - If True (default): Auto-registers/updates the default_template in the registry
        - If False: Never registers, but can still load existing prompts from registry
                   or use default_template directly as a local-only prompt

        Args:
            prompt_model: The prompt model configuration

        Returns:
            PromptVersion: The loaded prompt version

        Raises:
            ValueError: If no prompt can be loaded from any source
        """

        prompt_name: str = prompt_model.full_name

        # If explicit version or alias is specified, use it directly
        if prompt_model.version or prompt_model.alias:
            try:
                prompt_version: PromptVersion = prompt_model.as_prompt()
                version_or_alias = (
                    f"version {prompt_model.version}"
                    if prompt_model.version
                    else f"alias {prompt_model.alias}"
                )
                logger.debug(
                    "Loaded prompt with explicit version/alias",
                    prompt_name=prompt_name,
                    version_or_alias=version_or_alias,
                )
                return prompt_version
            except Exception as e:
                version_or_alias = (
                    f"version {prompt_model.version}"
                    if prompt_model.version
                    else f"alias {prompt_model.alias}"
                )
                logger.warning(
                    "Failed to load prompt with explicit version/alias",
                    prompt_name=prompt_name,
                    version_or_alias=version_or_alias,
                    error=str(e),
                )
                # Fall through to try other methods

        # Try to load in priority order: champion → default (with sync check)
        logger.trace(
            "Trying prompt fallback order",
            prompt_name=prompt_name,
            order="champion → default",
        )

        # First, sync default alias if template has changed (even if champion exists)
        # Only do this if auto_register is True
        if prompt_model.default_template and prompt_model.auto_register:
            try:
                # Try to load existing default
                existing_default = load_prompt(f"prompts:/{prompt_name}@default")

                # Check if champion exists and if it matches default
                champion_matches_default = False
                try:
                    existing_champion = load_prompt(f"prompts:/{prompt_name}@champion")
                    champion_matches_default = (
                        existing_champion.version == existing_default.version
                    )
                    status = (
                        "tracking" if champion_matches_default else "pinned separately"
                    )
                    logger.trace(
                        "Champion vs default version",
                        prompt_name=prompt_name,
                        champion_version=existing_champion.version,
                        default_version=existing_default.version,
                        status=status,
                    )
                except Exception:
                    # No champion exists
                    logger.trace("No champion alias found", prompt_name=prompt_name)

                # Check if default_template differs from existing default
                if (
                    existing_default.template.strip()
                    != prompt_model.default_template.strip()
                ):
                    logger.info(
                        "Default template changed, registering new version",
                        prompt_name=prompt_name,
                    )

                    # Only update champion if it was pointing to the old default
                    if champion_matches_default:
                        logger.info(
                            "Champion was tracking default, will update to new version",
                            prompt_name=prompt_name,
                            old_version=existing_default.version,
                        )
                        set_champion = True
                    else:
                        logger.info(
                            "Champion is pinned separately, preserving it",
                            prompt_name=prompt_name,
                        )
                        set_champion = False

                    self._register_default_template(
                        prompt_name,
                        prompt_model.default_template,
                        prompt_model.description,
                        set_champion=set_champion,
                    )
            except Exception as e:
                # No default exists yet, register it
                logger.trace(
                    "No default alias found", prompt_name=prompt_name, error=str(e)
                )
                logger.info(
                    "Registering default template as default alias",
                    prompt_name=prompt_name,
                )
                # First registration - set both default and champion
                self._register_default_template(
                    prompt_name,
                    prompt_model.default_template,
                    prompt_model.description,
                    set_champion=True,
                )
        elif prompt_model.default_template and not prompt_model.auto_register:
            logger.trace(
                "Prompt has auto_register=False, skipping registration",
                prompt_name=prompt_name,
            )

        # 1. Try champion alias (highest priority for execution)
        try:
            prompt_version = load_prompt(f"prompts:/{prompt_name}@champion")
            logger.info("Loaded prompt from champion alias", prompt_name=prompt_name)
            return prompt_version
        except Exception as e:
            logger.trace(
                "Champion alias not found", prompt_name=prompt_name, error=str(e)
            )

        # 2. Try default alias (already synced above)
        if prompt_model.default_template:
            try:
                prompt_version = load_prompt(f"prompts:/{prompt_name}@default")
                logger.info("Loaded prompt from default alias", prompt_name=prompt_name)
                return prompt_version
            except Exception as e:
                # Should not happen since we just registered it above, but handle anyway
                logger.trace(
                    "Default alias not found", prompt_name=prompt_name, error=str(e)
                )

        # 3. Try latest alias as final fallback
        try:
            prompt_version = load_prompt(f"prompts:/{prompt_name}@latest")
            logger.info("Loaded prompt from latest alias", prompt_name=prompt_name)
            return prompt_version
        except Exception as e:
            logger.trace(
                "Latest alias not found", prompt_name=prompt_name, error=str(e)
            )

        # 4. Final fallback: use default_template directly if available
        if prompt_model.default_template:
            logger.warning(
                "Could not load prompt from registry, using default_template directly",
                prompt_name=prompt_name,
            )
            return PromptVersion(
                name=prompt_name,
                version=1,
                template=prompt_model.default_template,
                tags={"dao_ai": dao_ai_version()},
            )

        raise ValueError(
            f"Prompt '{prompt_name}' not found in registry "
            "(tried champion, default, latest aliases) "
            "and no default_template provided"
        )

    def _register_default_template(
        self,
        prompt_name: str,
        default_template: str,
        description: str | None = None,
        set_champion: bool = True,
    ) -> PromptVersion:
        """Register default_template as a new prompt version.

        Registers the template and sets the 'default' alias.
        Optionally sets 'champion' alias if no champion exists.

        Args:
            prompt_name: Full name of the prompt
            default_template: The template content
            description: Optional description for commit message
            set_champion: Whether to also set champion alias (default: True)

        If registration fails (e.g., in Model Serving with restricted permissions),
        logs the error and raises.
        """
        logger.info(
            "Registering default template",
            prompt_name=prompt_name,
            set_champion=set_champion,
        )

        try:
            commit_message = description or "Auto-synced from default_template"
            prompt_version = mlflow.genai.register_prompt(
                name=prompt_name,
                template=default_template,
                commit_message=commit_message,
                tags={"dao_ai": dao_ai_version()},
            )

            # Always set default alias
            try:
                logger.debug(
                    "Setting default alias",
                    prompt_name=prompt_name,
                    version=prompt_version.version,
                )
                mlflow.genai.set_prompt_alias(
                    name=prompt_name, alias="default", version=prompt_version.version
                )
                logger.success(
                    "Set default alias for prompt",
                    prompt_name=prompt_name,
                    version=prompt_version.version,
                )
            except Exception as alias_error:
                logger.warning(
                    "Could not set default alias",
                    prompt_name=prompt_name,
                    error=str(alias_error),
                )

            # Optionally set champion alias (only if no champion exists or explicitly requested)
            if set_champion:
                try:
                    mlflow.genai.set_prompt_alias(
                        name=prompt_name,
                        alias="champion",
                        version=prompt_version.version,
                    )
                    logger.success(
                        "Set champion alias for prompt",
                        prompt_name=prompt_name,
                        version=prompt_version.version,
                    )
                except Exception as alias_error:
                    logger.warning(
                        "Could not set champion alias",
                        prompt_name=prompt_name,
                        error=str(alias_error),
                    )

            return prompt_version

        except Exception as reg_error:
            logger.error(
                "Failed to register prompt - please register from notebook with write permissions",
                prompt_name=prompt_name,
                error=str(reg_error),
            )
            return PromptVersion(
                name=prompt_name,
                version=1,
                template=default_template,
                tags={"dao_ai": dao_ai_version()},
            )
