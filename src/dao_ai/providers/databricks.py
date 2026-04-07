import base64
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Final, Sequence

import mlflow
import pandas as pd
import sqlparse
from databricks import agents
from databricks.agents import PermissionLevel, set_permissions
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import (
    BadRequest,
    InvalidParameterValue,
    NotFound,
    PermissionDenied,
)
from databricks.sdk.service.serving import EndpointStateConfigUpdate
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
from databricks.sdk.service.database import DatabaseCredential
from databricks.sdk.service.iam import User
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
from pyspark.sql import SparkSession
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
    GenieRoomModel,
    HasFullName,
    IndexModel,
    IsDatabricksResource,
    LLMModel,
    PromptModel,
    SchemaModel,
    TableModel,
    UnityCatalogFunctionSqlModel,
    VectorStoreModel,
    VolumeModel,
    VolumePathModel,
    WarehouseModel,
)
from dao_ai.models import get_latest_model_version
from dao_ai.providers.base import ServiceProvider
from dao_ai.utils import (
    dao_ai_version,
    get_installed_packages,
    is_installed,
    is_lib_provided,
    normalize_host,
    normalize_name,
)
from dao_ai.vector_search import endpoint_exists, index_exists

MAX_NUM_INDEXES: Final[int] = 50


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
        current_user: User = self.w.current_user.me()
        name: str = config.app.name
        return f"/Users/{current_user.user_name}/{name}"

    def get_or_create_experiment(self, config: AppConfig) -> Experiment:
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

        llms: Sequence[LLMModel] = list(config.resources.llms.values())
        vector_indexes: Sequence[IndexModel] = list(
            config.resources.vector_stores.values()
        )
        warehouses: Sequence[WarehouseModel] = list(
            config.resources.warehouses.values()
        )
        genie_rooms: Sequence[GenieRoomModel] = list(
            config.resources.genie_rooms.values()
        )
        tables: Sequence[TableModel] = list(config.resources.tables.values())
        functions: Sequence[FunctionModel] = list(config.resources.functions.values())
        connections: Sequence[ConnectionModel] = list(
            config.resources.connections.values()
        )
        databases: Sequence[DatabaseModel] = list(config.resources.databases.values())
        volumes: Sequence[VolumeModel] = list(config.resources.volumes.values())
        apps: Sequence[DatabricksAppModel] = list(config.resources.apps.values())

        resources: Sequence[IsDatabricksResource] = (
            llms
            + vector_indexes
            + warehouses
            + genie_rooms
            + functions
            + tables
            + connections
            + databases
            + volumes
            + apps
        )

        # Flatten all resources from all models into a single list
        all_resources: list[DatabricksResource] = []
        for r in resources:
            all_resources.extend(r.as_resources())

        system_resources: list[DatabricksResource] = [
            resource
            for r in resources
            for resource in r.as_resources()
            if not r.on_behalf_of_user
        ]

        if config.app and config.app.trace_location:
            trace_resources = config.app.trace_location.as_resources()
            system_resources.extend(trace_resources)
            all_resources.extend(trace_resources)
            logger.debug(
                "Added OTEL trace tables to system resources",
                count=len(trace_resources),
            )

        logger.trace(
            "System resources identified",
            count=len(system_resources),
            resources=[r.name for r in system_resources],
        )

        system_auth_policy: SystemAuthPolicy = SystemAuthPolicy(
            resources=system_resources
        )
        logger.trace("System auth policy created", policy=str(system_auth_policy))

        api_scopes: Sequence[str] = list(
            set(
                [
                    scope
                    for r in resources
                    if r.on_behalf_of_user
                    for scope in r.api_scopes
                ]
            )
        )
        logger.trace("API scopes identified", scopes=api_scopes)

        user_auth_policy: UserAuthPolicy = UserAuthPolicy(api_scopes=api_scopes)
        logger.trace("User auth policy created", policy=str(user_auth_policy))

        auth_policy: AuthPolicy = AuthPolicy(
            system_auth_policy=system_auth_policy, user_auth_policy=user_auth_policy
        )
        logger.debug(
            "Auth policy created",
            has_system_auth=system_auth_policy is not None,
            has_user_auth=user_auth_policy is not None,
        )

        code_paths: list[str] = config.app.code_paths
        for path in code_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Code path does not exist: {path}")

        model_root_path: Path = Path(dao_ai.__file__).parent
        model_path: Path = model_root_path / "apps" / "model_serving.py"

        pip_requirements: Sequence[str] = config.app.pip_requirements

        if is_installed():
            if not is_lib_provided("dao-ai", pip_requirements):
                pip_requirements += [
                    f"dao-ai=={dao_ai_version()}",
                ]
        else:
            src_path: Path = model_root_path.parent
            directories: Sequence[Path] = [d for d in src_path.iterdir() if d.is_dir()]
            for directory in directories:
                directory: Path
                code_paths.append(directory.as_posix())

            pip_requirements += get_installed_packages()

        from dao_ai.guardrails_hub import collect_hub_code_paths

        code_paths.extend(collect_hub_code_paths(config))

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
                    model_config=config.model_dump(mode="json", by_alias=True),
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
        mlflow.set_registry_uri("databricks-uc")

        endpoint_name: str = config.app.endpoint_name
        registered_model_name: str = config.app.registered_model.full_name
        scale_to_zero: bool = config.app.scale_to_zero
        environment_vars: dict[str, str] = config.app.environment_vars
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
                if (
                    "currently updating" in err_msg.lower()
                    and attempt < max_attempts
                ):
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

        # Link experiment to UC trace location if configured
        if config.app.trace_location:
            from mlflow.entities import UCSchemaLocation
            from mlflow.tracing.enablement import set_experiment_trace_location

            experiment: Experiment = self.get_or_create_experiment(config)
            loc = config.app.trace_location

            try:
                set_experiment_trace_location(
                    location=UCSchemaLocation(
                        catalog_name=loc.catalog_name,
                        schema_name=loc.schema_name,
                    ),
                    experiment_id=experiment.experiment_id,
                    sql_warehouse_id=loc.warehouse_id,
                )
                logger.info(
                    "Linked experiment to UC trace location for Model Serving",
                    catalog=loc.catalog_name,
                    schema=loc.schema_name,
                )
            except mlflow.exceptions.RestException as e:
                if "already contains traces" in str(e):
                    logger.warning(
                        "UC trace destination already linked or experiment has existing traces, skipping",
                        experiment_id=experiment.experiment_id,
                        catalog=loc.catalog_name,
                        schema=loc.schema_name,
                    )
                else:
                    raise

            self.grant_otel_table_permissions(config)

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

    def grant_otel_table_permissions(self, config: AppConfig) -> None:
        """Grant explicit MODIFY and SELECT on OTEL trace tables.

        Per Databricks docs, ALL_PRIVILEGES is not sufficient for OTEL trace
        table access. The ingestion path requires explicit MODIFY and SELECT
        grants on each table.
        """
        if not config.app.trace_location:
            return

        loc = config.app.trace_location
        schema_prefix: str = f"{loc.catalog_name}.{loc.schema_name}"

        # Determine grantee: service principal application ID or current user
        grantee: str | None = None
        if config.app.service_principal and config.app.service_principal.client_id:
            from dao_ai.config import value_of

            grantee = value_of(config.app.service_principal.client_id)

        if not grantee:
            logger.debug("No service principal configured, skipping OTEL table grants")
            return

        from databricks.sdk.service.catalog import (
            PermissionsChange,
            Privilege,
            SecurableType,
        )

        from dao_ai.config import TraceLocationModel

        for suffix in TraceLocationModel.OTEL_TABLE_SUFFIXES:
            table_name: str = f"{schema_prefix}.{suffix}"
            for privilege in ("MODIFY", "SELECT"):
                try:
                    self.w.grants.update(
                        full_name=table_name,
                        securable_type=SecurableType.TABLE,
                        changes=[
                            PermissionsChange(
                                add=[Privilege(privilege)],
                                principal=grantee,
                            )
                        ],
                    )
                except Exception as e:
                    error_msg: str = str(e)
                    if (
                        "already has" in error_msg.lower()
                        or "ALREADY_EXISTS" in error_msg
                    ):
                        logger.trace(
                            "Grant already exists",
                            table=table_name,
                            privilege=privilege,
                        )
                    else:
                        logger.warning(
                            "Could not grant privilege on OTEL table",
                            table=table_name,
                            privilege=privilege,
                            grantee=grantee,
                            error=error_msg,
                        )

        logger.info(
            "Granted MODIFY and SELECT on OTEL trace tables",
            grantee=grantee,
            table_count=len(TraceLocationModel.OTEL_TABLE_SUFFIXES),
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

        # Link experiment to UC trace location if configured
        if config.app.trace_location:
            from mlflow.entities import UCSchemaLocation
            from mlflow.tracing.enablement import set_experiment_trace_location

            loc = config.app.trace_location
            try:
                set_experiment_trace_location(
                    location=UCSchemaLocation(
                        catalog_name=loc.catalog_name,
                        schema_name=loc.schema_name,
                    ),
                    experiment_id=experiment.experiment_id,
                    sql_warehouse_id=loc.warehouse_id,
                )
                logger.info(
                    "Linked experiment to UC trace location",
                    catalog=loc.catalog_name,
                    schema=loc.schema_name,
                )
            except mlflow.exceptions.RestException as e:
                if "already contains traces" in str(e):
                    logger.warning(
                        "UC trace destination already linked or experiment has existing traces, skipping",
                        experiment_id=experiment.experiment_id,
                        catalog=loc.catalog_name,
                        schema=loc.schema_name,
                    )
                else:
                    raise

            self.grant_otel_table_permissions(config)

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

        # Upload the configuration file to the workspace
        source_config_path: str | None = config.source_config_path
        if source_config_path:
            # Read the config file and upload to workspace
            config_file_name: str = "dao_ai.yaml"
            workspace_config_path: str = f"{source_path}/{config_file_name}"

            logger.info(
                "Uploading config file to workspace",
                source=source_config_path,
                destination=workspace_config_path,
            )

            # Read the source config file
            with open(source_config_path, "rb") as f:
                config_content: bytes = f.read()

            # Create the directory if it doesn't exist and upload the file
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
        else:
            logger.warning(
                "No source config path available. "
                "Ensure DAO_AI_CONFIG_PATH is set in the app environment or "
                "dao_ai.yaml exists in the app source directory."
            )

        # Generate and upload app.yaml with dynamically discovered resources
        from dao_ai.apps.resources import generate_app_yaml

        app_yaml_content: str = generate_app_yaml(
            config,
            command=[
                "/bin/bash",
                "-c",
                "pip install dao-ai && python -m dao_ai.apps.server",
            ],
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
            """Create or update app via REST API with fallback.

            If the workspace doesn't support VECTOR_SEARCH_INDEX as a UC
            securable type, or if the resource limit is exceeded, retries
            without those resources. The source TABLE resources still
            provide catalog/schema permissions as a fallback.
            """
            try:
                self.w.api_client.do(method, path, body=body)
            except (BadRequest, InvalidParameterValue, PermissionDenied) as e:
                err_msg = str(e)
                if (
                    "permission to grant" in err_msg
                    or "less than or equal to" in err_msg
                    or "MANAGE permission on the resource" in err_msg
                ):
                    body["resources"] = [
                        r
                        for r in body.get("resources", [])
                        if r.get("uc_securable", {}).get("securable_type")
                        != "VECTOR_SEARCH_INDEX"
                    ]
                    logger.warning(
                        "Retrying without VECTOR_SEARCH_INDEX resources",
                        error=err_msg,
                        remaining_resources=len(body.get("resources", [])),
                    )
                    self.w.api_client.do(method, path, body=body)
                else:
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

        # Deploy requires app compute in RUNNING state (API error otherwise).
        fresh_app: App = self.w.apps.get(name=app_name)
        app_status = fresh_app.app_status
        if (
            app_status is None
            or app_status.state != ApplicationState.RUNNING
        ):
            logger.info(
                "Databricks App not in RUNNING state; starting before deploy",
                app_name=app_name,
                state=(
                    app_status.state.value
                    if app_status is not None and app_status.state is not None
                    else None
                ),
            )
            self.w.apps.start_and_wait(name=app_name)
            fresh_app = self.w.apps.get(name=app_name)
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

                pandas_readers: dict[str, Callable[..., pd.DataFrame]] = {
                    "parquet": lambda p, **kw: pd.read_parquet(p, **kw),
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
            self.vsc.create_endpoint_and_wait(
                name=vector_store.endpoint.name,
                endpoint_type=vector_store.endpoint.type,
                verbose=True,
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
        else:
            logger.debug(
                "Vector search index already exists, checking status",
                index_name=vector_store.index.full_name,
            )
            index = self.vsc.get_index(
                vector_store.endpoint.name, vector_store.index.full_name
            )

            # Wait for index to be in a syncable state
            import time

            max_wait_time = 600  # 10 minutes
            wait_interval = 10  # 10 seconds
            elapsed = 0

            while elapsed < max_wait_time:
                try:
                    index_status = index.describe()
                    pipeline_status = index_status.get("status", {}).get(
                        "detailed_state", "UNKNOWN"
                    )
                    logger.debug(f"Index pipeline status: {pipeline_status}")

                    if pipeline_status in [
                        "COMPLETED",
                        "ONLINE",
                        "FAILED",
                        "CANCELED",
                        "ONLINE_PIPELINE_FAILED",
                    ]:
                        logger.debug("Index ready to sync", status=pipeline_status)
                        break
                    elif pipeline_status in [
                        "WAITING_FOR_RESOURCES",
                        "PROVISIONING",
                        "INITIALIZING",
                        "INDEXING",
                    ]:
                        logger.trace(
                            "Index not ready, waiting",
                            status=pipeline_status,
                            wait_seconds=wait_interval,
                        )
                        time.sleep(wait_interval)
                        elapsed += wait_interval
                    else:
                        logger.warning(
                            "Unknown pipeline status, attempting sync",
                            status=pipeline_status,
                        )
                        break
                except Exception as status_error:
                    logger.warning(
                        "Could not check index status, attempting sync",
                        error=str(status_error),
                    )
                    break

            if elapsed >= max_wait_time:
                logger.warning(
                    "Timed out waiting for index to be ready",
                    max_wait_seconds=max_wait_time,
                )

            # Now attempt to sync
            try:
                index.sync()
                logger.success("Index sync completed")
            except Exception as sync_error:
                if "not ready to sync yet" in str(sync_error).lower():
                    logger.warning(
                        "Index still not ready to sync", error=str(sync_error)
                    )
                else:
                    raise sync_error

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

    def _wait_for_database_available(
        self,
        workspace_client: WorkspaceClient,
        instance_name: str,
        max_wait_time: int = 600,
        wait_interval: int = 10,
    ) -> None:
        """
        Wait for a database instance to become AVAILABLE.

        Args:
            workspace_client: The Databricks workspace client
            instance_name: Name of the database instance to wait for
            max_wait_time: Maximum time to wait in seconds (default: 600 = 10 minutes)
            wait_interval: Time between status checks in seconds (default: 10)

        Raises:
            TimeoutError: If the database doesn't become AVAILABLE within max_wait_time
            RuntimeError: If the database enters a failed or deleted state
        """
        import time
        from typing import Any

        logger.info(
            "Waiting for database instance to become AVAILABLE",
            instance_name=instance_name,
        )
        elapsed: int = 0

        while elapsed < max_wait_time:
            try:
                current_instance: Any = workspace_client.database.get_database_instance(
                    name=instance_name
                )
                current_state: str = current_instance.state
                logger.trace(
                    "Database instance state checked",
                    instance_name=instance_name,
                    state=current_state,
                )

                if current_state == "AVAILABLE":
                    logger.success(
                        "Database instance is now AVAILABLE",
                        instance_name=instance_name,
                    )
                    return
                elif current_state in ["STARTING", "UPDATING", "PROVISIONING"]:
                    logger.trace(
                        "Database instance not ready, waiting",
                        instance_name=instance_name,
                        state=current_state,
                        wait_seconds=wait_interval,
                    )
                    time.sleep(wait_interval)
                    elapsed += wait_interval
                elif current_state in ["STOPPED", "DELETING", "FAILED"]:
                    raise RuntimeError(
                        f"Database instance {instance_name} entered unexpected state: {current_state}"
                    )
                else:
                    logger.warning(
                        "Unknown database state, continuing to wait",
                        instance_name=instance_name,
                        state=current_state,
                    )
                    time.sleep(wait_interval)
                    elapsed += wait_interval
            except NotFound:
                raise RuntimeError(
                    f"Database instance {instance_name} was deleted while waiting for it to become AVAILABLE"
                )

        raise TimeoutError(
            f"Timed out waiting for database instance {instance_name} to become AVAILABLE after {max_wait_time} seconds"
        )

    def create_lakebase(self, database: DatabaseModel) -> None:
        """
        Create a Lakebase database instance using the Databricks workspace client.

        This method handles idempotent database creation, gracefully handling cases where:
        - The database instance already exists
        - The database is in an intermediate state (STARTING, UPDATING, etc.)

        Args:
            database: DatabaseModel containing the database configuration

        Returns:
            None

        Raises:
            Exception: If an unexpected error occurs during database creation
        """
        import time
        from typing import Any

        workspace_client: WorkspaceClient = database.workspace_client

        try:
            # First, check if the database instance already exists
            existing_instance: Any = workspace_client.database.get_database_instance(
                name=database.instance_name
            )

            if existing_instance:
                logger.debug(
                    "Database instance already exists",
                    instance_name=database.instance_name,
                    state=existing_instance.state,
                )

                # Check if database is in an intermediate state
                if existing_instance.state in ["STARTING", "UPDATING"]:
                    logger.info(
                        "Database instance in intermediate state, waiting",
                        instance_name=database.instance_name,
                        state=existing_instance.state,
                    )

                    # Wait for database to reach a stable state
                    max_wait_time: int = 600  # 10 minutes
                    wait_interval: int = 10  # 10 seconds
                    elapsed: int = 0

                    while elapsed < max_wait_time:
                        try:
                            current_instance: Any = (
                                workspace_client.database.get_database_instance(
                                    name=database.instance_name
                                )
                            )
                            current_state: str = current_instance.state
                            logger.trace(
                                "Checking database instance state",
                                instance_name=database.instance_name,
                                state=current_state,
                            )

                            if current_state == "AVAILABLE":
                                logger.success(
                                    "Database instance is now AVAILABLE",
                                    instance_name=database.instance_name,
                                )
                                break
                            elif current_state in ["STARTING", "UPDATING"]:
                                logger.trace(
                                    "Database instance not ready, waiting",
                                    instance_name=database.instance_name,
                                    state=current_state,
                                    wait_seconds=wait_interval,
                                )
                                time.sleep(wait_interval)
                                elapsed += wait_interval
                            elif current_state in ["STOPPED", "DELETING"]:
                                logger.warning(
                                    "Database instance in unexpected state",
                                    instance_name=database.instance_name,
                                    state=current_state,
                                )
                                break
                            else:
                                logger.warning(
                                    "Unknown database state, proceeding",
                                    instance_name=database.instance_name,
                                    state=current_state,
                                )
                                break
                        except NotFound:
                            logger.warning(
                                "Database instance no longer exists, will recreate",
                                instance_name=database.instance_name,
                            )
                            break
                        except Exception as state_error:
                            logger.warning(
                                "Could not check database state, proceeding",
                                instance_name=database.instance_name,
                                error=str(state_error),
                            )
                            break

                    if elapsed >= max_wait_time:
                        logger.warning(
                            "Timed out waiting for database to become AVAILABLE",
                            instance_name=database.instance_name,
                            max_wait_seconds=max_wait_time,
                        )

                elif existing_instance.state == "AVAILABLE":
                    logger.info(
                        "Database instance already exists and is AVAILABLE",
                        instance_name=database.instance_name,
                    )
                    return
                elif existing_instance.state in ["STOPPED", "DELETING"]:
                    logger.warning(
                        "Database instance in terminal state",
                        instance_name=database.instance_name,
                        state=existing_instance.state,
                    )
                    return
                else:
                    logger.info(
                        "Database instance already exists",
                        instance_name=database.instance_name,
                        state=existing_instance.state,
                    )
                    return

        except NotFound:
            # Database doesn't exist, proceed with creation
            logger.info(
                "Creating new database instance", instance_name=database.instance_name
            )

            try:
                # Resolve variable values for database parameters
                from databricks.sdk.service.database import DatabaseInstance

                capacity: str = database.capacity if database.capacity else "CU_2"

                # Create the database instance object
                database_instance: DatabaseInstance = DatabaseInstance(
                    name=database.instance_name,
                    capacity=capacity,
                    node_count=database.node_count,
                )

                # Create the database instance via API
                workspace_client.database.create_database_instance(
                    database_instance=database_instance
                )
                logger.success(
                    "Database instance created successfully",
                    instance_name=database.instance_name,
                )

                # Wait for the newly created database to become AVAILABLE
                self._wait_for_database_available(
                    workspace_client, database.instance_name
                )
                return

            except Exception as create_error:
                error_msg: str = str(create_error)

                # Handle case where database was created by another process concurrently
                if (
                    "already exists" in error_msg.lower()
                    or "not unique" in error_msg.lower()
                    or "RESOURCE_ALREADY_EXISTS" in error_msg
                ):
                    logger.info(
                        "Database instance was created concurrently",
                        instance_name=database.instance_name,
                    )
                    # Still need to wait for the database to become AVAILABLE
                    self._wait_for_database_available(
                        workspace_client, database.instance_name
                    )
                    return
                else:
                    # Re-raise unexpected errors
                    logger.error(
                        "Error creating database instance",
                        instance_name=database.instance_name,
                        error=str(create_error),
                    )
                    raise

        except Exception as e:
            # Handle other unexpected errors
            error_msg: str = str(e)

            # Check if this is actually a "resource already exists" type error
            if (
                "already exists" in error_msg.lower()
                or "RESOURCE_ALREADY_EXISTS" in error_msg
            ):
                logger.info(
                    "Database instance already exists (detected via exception)",
                    instance_name=database.instance_name,
                )
                return
            else:
                logger.error(
                    "Unexpected error while handling database",
                    instance_name=database.instance_name,
                    error=str(e),
                )
                raise

    def lakebase_password_provider(self, instance_name: str) -> str:
        """
        Ask Databricks to mint a fresh DB credential for this instance.
        """
        logger.trace(
            "Generating password for lakebase instance", instance_name=instance_name
        )
        w: WorkspaceClient = self.w
        cred: DatabaseCredential = w.database.generate_database_credential(
            request_id=str(uuid.uuid4()),
            instance_names=[instance_name],
        )
        return cred.token

    def create_lakebase_instance_role(self, database: DatabaseModel) -> None:
        """
        Create a database instance role for a Lakebase instance.

        This method creates a role with DATABRICKS_SUPERUSER membership for the
        service principal specified in the database configuration.

        Args:
            database: DatabaseModel containing the database and service principal configuration

        Returns:
            None

        Raises:
            ValueError: If client_id is not provided in the database configuration
            Exception: If an unexpected error occurs during role creation
        """
        from databricks.sdk.service.database import (
            DatabaseInstanceRole,
            DatabaseInstanceRoleIdentityType,
            DatabaseInstanceRoleMembershipRole,
        )

        from dao_ai.config import value_of

        # Validate that client_id is provided
        if not database.client_id:
            logger.warning(
                "client_id required to create instance role",
                instance_name=database.instance_name,
            )
            return

        # Resolve the client_id value
        client_id: str = value_of(database.client_id)
        role_name: str = client_id
        instance_name: str = database.instance_name

        logger.debug(
            "Creating instance role",
            role_name=role_name,
            instance_name=instance_name,
            principal=client_id,
        )

        try:
            # Check if role already exists
            try:
                _ = self.w.database.get_database_instance_role(
                    instance_name=instance_name,
                    name=role_name,
                )
                logger.info(
                    "Instance role already exists",
                    role_name=role_name,
                    instance_name=instance_name,
                )
                return
            except NotFound:
                # Role doesn't exist, proceed with creation
                logger.debug(
                    "Instance role not found, creating new role", role_name=role_name
                )

            # Create the database instance role
            role: DatabaseInstanceRole = DatabaseInstanceRole(
                name=role_name,
                identity_type=DatabaseInstanceRoleIdentityType.SERVICE_PRINCIPAL,
                membership_role=DatabaseInstanceRoleMembershipRole.DATABRICKS_SUPERUSER,
            )

            # Create the role using the API
            self.w.database.create_database_instance_role(
                instance_name=instance_name,
                database_instance_role=role,
            )

            logger.success(
                "Instance role created successfully",
                role_name=role_name,
                instance_name=instance_name,
            )

        except Exception as e:
            error_msg: str = str(e)

            # Handle case where role was created concurrently
            if (
                "already exists" in error_msg.lower()
                or "RESOURCE_ALREADY_EXISTS" in error_msg
            ):
                logger.info(
                    "Instance role was created concurrently",
                    role_name=role_name,
                    instance_name=instance_name,
                )
                return

            # Re-raise unexpected errors
            logger.error(
                "Error creating instance role",
                role_name=role_name,
                instance_name=instance_name,
                error=str(e),
            )
            raise

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

        client_id: str = value_of(database.client_id)
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
