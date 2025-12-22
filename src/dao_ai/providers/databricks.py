import base64
import uuid
from pathlib import Path
from typing import Any, Callable, Final, Sequence

import mlflow
import pandas as pd
import sqlparse
from databricks import agents
from databricks.agents import PermissionLevel, set_permissions
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound
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
from databricks.sdk.service.workspace import GetSecretResponse
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
    DatasetModel,
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
        if w is None:
            w = _workspace_client(
                pat=pat,
                client_id=client_id,
                client_secret=client_secret,
                workspace_host=workspace_host,
            )
        if vsc is None:
            vsc = _vector_search_client(
                pat=pat,
                client_id=client_id,
                client_secret=client_secret,
                workspace_host=workspace_host,
            )
        if dfs is None:
            dfs = _function_client(w=w)
        self.w = w
        self.vsc = vsc
        self.dfs = dfs

    def experiment_name(self, config: AppConfig) -> str:
        current_user: User = self.w.current_user.me()
        name: str = config.app.name
        return f"/Users/{current_user.user_name}/{name}"

    def get_or_create_experiment(self, config: AppConfig) -> Experiment:
        experiment_name: str = self.experiment_name(config)
        experiment: Experiment | None = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id: str = mlflow.create_experiment(name=experiment_name)
            logger.info(
                f"Created new experiment: {experiment_name} (ID: {experiment_id})"
            )
            experiment = mlflow.get_experiment(experiment_id)
        return experiment

    def create_token(self) -> str:
        current_user: User = self.w.current_user.me()
        logger.debug(f"Authenticated to Databricks as {current_user}")
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
            logger.debug(f"Retrieved secret {secret_key} from scope {secret_scope}")
            encoded_secret: str = secret_response.value
            decoded_secret: str = base64.b64decode(encoded_secret).decode("utf-8")
            return decoded_secret
        except NotFound:
            logger.warning(
                f"Secret {secret_key} not found in scope {secret_scope}, using default value"
            )
        except Exception as e:
            logger.error(
                f"Error retrieving secret {secret_key} from scope {secret_scope}: {e}"
            )

        return default_value

    def create_agent(
        self,
        config: AppConfig,
    ) -> ModelInfo:
        logger.debug("Creating agent...")
        mlflow.set_registry_uri("databricks-uc")

        # Set up experiment for proper tracking
        experiment: Experiment = self.get_or_create_experiment(config)
        mlflow.set_experiment(experiment_id=experiment.experiment_id)
        logger.debug(
            f"Using experiment: {experiment.name} (ID: {experiment.experiment_id})"
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
        )

        # Flatten all resources from all models into a single list
        all_resources: list[DatabricksResource] = []
        for r in resources:
            all_resources.extend(r.as_resources())

        system_resources: Sequence[DatabricksResource] = [
            resource
            for r in resources
            for resource in r.as_resources()
            if not r.on_behalf_of_user
        ]
        logger.debug(f"system_resources: {[r.name for r in system_resources]}")

        system_auth_policy: SystemAuthPolicy = SystemAuthPolicy(
            resources=system_resources
        )
        logger.debug(f"system_auth_policy: {system_auth_policy}")

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
        logger.debug(f"api_scopes: {api_scopes}")

        user_auth_policy: UserAuthPolicy = UserAuthPolicy(api_scopes=api_scopes)
        logger.debug(f"user_auth_policy: {user_auth_policy}")

        auth_policy: AuthPolicy = AuthPolicy(
            system_auth_policy=system_auth_policy, user_auth_policy=user_auth_policy
        )
        logger.debug(f"auth_policy: {auth_policy}")

        code_paths: list[str] = config.app.code_paths
        for path in code_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Code path does not exist: {path}")

        model_root_path: Path = Path(dao_ai.__file__).parent
        model_path: Path = model_root_path / "agent_as_code.py"

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

        logger.debug(f"pip_requirements: {pip_requirements}")
        logger.debug(f"code_paths: {code_paths}")

        run_name: str = normalize_name(config.app.name)
        logger.debug(f"run_name: {run_name}")
        logger.debug(f"model_path: {model_path.as_posix()}")

        input_example: dict[str, Any] = None
        if config.app.input_example:
            input_example = config.app.input_example.model_dump()

        logger.debug(f"input_example: {input_example}")

        # Create conda environment with configured Python version
        # This allows deploying from environments with different Python versions
        # (e.g., Databricks Apps with Python 3.11 can deploy to Model Serving with 3.12)
        target_python_version: str = config.app.python_version
        logger.debug(f"target_python_version: {target_python_version}")

        conda_env: dict[str, Any] = {
            "name": "mlflow-env",
            "channels": ["conda-forge"],
            "dependencies": [
                f"python={target_python_version}",
                "pip",
                {"pip": list(pip_requirements)},
            ],
        }
        logger.debug(f"conda_env: {conda_env}")

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

        registered_model_name: str = config.app.registered_model.full_name

        model_version: ModelVersion = mlflow.register_model(
            name=registered_model_name, model_uri=logged_agent_info.model_uri
        )
        logger.debug(
            f"Registered model: {registered_model_name} with version: {model_version.version}"
        )

        client: MlflowClient = MlflowClient()

        # Set tags on the model version
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version.version,
            key="dao_ai",
            value=dao_ai_version(),
        )
        logger.debug(f"Set dao_ai tag on model version {model_version.version}")

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
            logger.debug(
                f"Model {registered_model_name} aliased to {config.app.alias} with version: {aliased_model.version}"
            )

    def deploy_agent(self, config: AppConfig) -> None:
        logger.debug("Deploying agent...")
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

        # Check if endpoint exists to determine deployment strategy
        endpoint_exists: bool = False
        try:
            agents.get_deployments(endpoint_name)
            endpoint_exists = True
            logger.debug(
                f"Endpoint {endpoint_name} already exists, updating without tags to avoid conflicts..."
            )
        except Exception:
            logger.debug(
                f"Endpoint {endpoint_name} doesn't exist, creating new with tags..."
            )

        # Deploy - skip tags for existing endpoints to avoid conflicts
        agents.deploy(
            endpoint_name=endpoint_name,
            model_name=registered_model_name,
            model_version=latest_version,
            scale_to_zero=scale_to_zero,
            environment_vars=environment_vars,
            workload_size=workload_size,
            tags=tags if not endpoint_exists else None,
        )

        registered_model_name: str = config.app.registered_model.full_name
        permissions: Sequence[dict[str, Any]] = config.app.permissions

        logger.debug(registered_model_name)
        logger.debug(permissions)

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

    def create_catalog(self, schema: SchemaModel) -> CatalogInfo:
        catalog_info: CatalogInfo
        try:
            catalog_info = self.w.catalogs.get(name=schema.catalog_name)
        except NotFound:
            logger.debug(f"Creating catalog: {schema.catalog_name}")
            catalog_info = self.w.catalogs.create(name=schema.catalog_name)
        return catalog_info

    def create_schema(self, schema: SchemaModel) -> SchemaInfo:
        catalog_info: CatalogInfo = self.create_catalog(schema)
        schema_info: SchemaInfo
        try:
            schema_info = self.w.schemas.get(full_name=schema.full_name)
        except NotFound:
            logger.debug(f"Creating schema: {schema.full_name}")
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
            logger.debug(f"Creating volume: {volume.full_name}")
            volume_info = self.w.volumes.create(
                catalog_name=schema_info.catalog_name,
                schema_name=schema_info.name,
                name=volume.name,
                volume_type=VolumeType.MANAGED,
            )
        return volume_info

    def create_path(self, volume_path: VolumePathModel) -> Path:
        path: Path = volume_path.full_name
        logger.info(f"Creating volume path: {path}")
        self.w.files.create_directory(path)
        return path

    def create_dataset(self, dataset: DatasetModel) -> None:
        current_dir: Path = "file:///" / Path.cwd().relative_to("/")

        # Get or create Spark session
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
            logger.debug(f"Executing DDL from: {ddl_path}")
            statements: Sequence[str] = sqlparse.parse(ddl_path.read_text())
            for statement in statements:
                logger.debug(statement)
                logger.debug(f"args: {args}")
                spark.sql(
                    str(statement),
                    args=args,
                )

        if data:
            data_path: Path = Path(data)
            if format == "sql":
                logger.debug(f"Executing SQL from: {data_path}")
                data_statements: Sequence[str] = sqlparse.parse(data_path.read_text())
                for statement in data_statements:
                    logger.debug(statement)
                    logger.debug(f"args: {args}")
                    spark.sql(
                        str(statement),
                        args=args,
                    )
            else:
                logger.debug(f"Writing to: {table}")
                if not data_path.is_absolute():
                    data_path = current_dir / data_path
                logger.debug(f"Data path: {data_path.as_posix()}")
                if format == "excel":
                    pdf = pd.read_excel(data_path.as_posix())
                    df = spark.createDataFrame(pdf, schema=dataset.table_schema)
                else:
                    df = (
                        spark.read.format(format)
                        .options(**read_options)
                        .load(
                            data_path.as_posix(),
                            schema=dataset.table_schema,
                        )
                    )

                df.write.mode("overwrite").saveAsTable(table)

    def create_vector_store(self, vector_store: VectorStoreModel) -> None:
        if not endpoint_exists(self.vsc, vector_store.endpoint.name):
            self.vsc.create_endpoint_and_wait(
                name=vector_store.endpoint.name,
                endpoint_type=vector_store.endpoint.type,
                verbose=True,
            )

        logger.debug(f"Endpoint named {vector_store.endpoint.name} is ready.")

        if not index_exists(
            self.vsc, vector_store.endpoint.name, vector_store.index.full_name
        ):
            logger.debug(
                f"Creating index {vector_store.index.full_name} on endpoint {vector_store.endpoint.name}..."
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
                f"Index {vector_store.index.full_name} already exists, checking status and syncing..."
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
                        "FAILED",
                        "CANCELED",
                        "ONLINE_PIPELINE_FAILED",
                    ]:
                        logger.debug(
                            f"Index is ready to sync (status: {pipeline_status})"
                        )
                        break
                    elif pipeline_status in [
                        "WAITING_FOR_RESOURCES",
                        "PROVISIONING",
                        "INITIALIZING",
                        "INDEXING",
                        "ONLINE",
                    ]:
                        logger.debug(
                            f"Index not ready yet (status: {pipeline_status}), waiting {wait_interval} seconds..."
                        )
                        time.sleep(wait_interval)
                        elapsed += wait_interval
                    else:
                        logger.warning(
                            f"Unknown pipeline status: {pipeline_status}, attempting sync anyway"
                        )
                        break
                except Exception as status_error:
                    logger.warning(
                        f"Could not check index status: {status_error}, attempting sync anyway"
                    )
                    break

            if elapsed >= max_wait_time:
                logger.warning(
                    f"Timed out waiting for index to be ready after {max_wait_time} seconds"
                )

            # Now attempt to sync
            try:
                index.sync()
                logger.debug("Index sync completed successfully")
            except Exception as sync_error:
                if "not ready to sync yet" in str(sync_error).lower():
                    logger.warning(f"Index still not ready to sync: {sync_error}")
                else:
                    raise sync_error

        logger.debug(
            f"index {vector_store.index.full_name} on table {vector_store.source_table.full_name} is ready"
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

            logger.info(function.name)
            logger.info(sql)
            _: FunctionInfo = self.dfs.create_function(sql_function_body=sql)

            if unity_catalog_function.test:
                logger.info(unity_catalog_function.test.parameters)

                result: FunctionExecutionResult = self.dfs.execute_function(
                    function_name=function.full_name,
                    parameters=unity_catalog_function.test.parameters,
                )

                if result.error:
                    logger.error(result.error)
                else:
                    logger.info(f"Function {function.full_name} executed successfully.")
                    logger.info(f"Result: {result}")

    def find_columns(self, table_model: TableModel) -> Sequence[str]:
        logger.debug(f"Finding columns for table: {table_model.full_name}")
        table_info: TableInfo = self.w.tables.get(full_name=table_model.full_name)
        columns: Sequence[ColumnInfo] = table_info.columns
        column_names: Sequence[str] = [c.name for c in columns]
        logger.debug(f"Columns found: {column_names}")
        return column_names

    def find_primary_key(self, table_model: TableModel) -> Sequence[str] | None:
        logger.debug(f"Finding primary key for table: {table_model.full_name}")
        primary_keys: Sequence[str] | None = None
        table_info: TableInfo = self.w.tables.get(full_name=table_model.full_name)
        constraints: Sequence[TableConstraint] = table_info.table_constraints
        primary_key_constraint: PrimaryKeyConstraint | None = next(
            c.primary_key_constraint for c in constraints if c.primary_key_constraint
        )
        if primary_key_constraint:
            primary_keys = primary_key_constraint.child_columns

        logger.debug(f"Primary key for table {table_model.full_name}: {primary_keys}")
        return primary_keys

    def find_vector_search_endpoint(
        self, predicate: Callable[[dict[str, Any]], bool]
    ) -> str | None:
        logger.debug("Finding vector search endpoint...")
        endpoint_name: str | None = None
        vector_search_endpoints: Sequence[dict[str, Any]] = (
            self.vsc.list_endpoints().get("endpoints", [])
        )
        for endpoint in vector_search_endpoints:
            if predicate(endpoint):
                endpoint_name = endpoint["name"]
                break
        logger.debug(f"Vector search endpoint found: {endpoint_name}")
        return endpoint_name

    def find_endpoint_for_index(self, index_model: IndexModel) -> str | None:
        logger.debug(f"Finding vector search index: {index_model.full_name}")
        all_endpoints: Sequence[dict[str, Any]] = self.vsc.list_endpoints().get(
            "endpoints", []
        )
        index_name: str = index_model.full_name
        found_endpoint_name: str | None = None
        for endpoint in all_endpoints:
            endpoint_name: str = endpoint["name"]
            indexes = self.vsc.list_indexes(name=endpoint_name)
            vector_indexes: Sequence[dict[str, Any]] = indexes.get("vector_indexes", [])
            logger.trace(f"Endpoint: {endpoint_name}, vector_indexes: {vector_indexes}")
            index_names = [vector_index["name"] for vector_index in vector_indexes]
            if index_name in index_names:
                found_endpoint_name = endpoint_name
                break
        logger.debug(f"Vector search index found: {found_endpoint_name}")
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
            f"Waiting for database instance {instance_name} to become AVAILABLE..."
        )
        elapsed: int = 0

        while elapsed < max_wait_time:
            try:
                current_instance: Any = workspace_client.database.get_database_instance(
                    name=instance_name
                )
                current_state: str = current_instance.state
                logger.debug(
                    f"Database instance {instance_name} state: {current_state}"
                )

                if current_state == "AVAILABLE":
                    logger.info(f"Database instance {instance_name} is now AVAILABLE")
                    return
                elif current_state in ["STARTING", "UPDATING", "PROVISIONING"]:
                    logger.debug(
                        f"Database instance still in {current_state} state, waiting {wait_interval} seconds..."
                    )
                    time.sleep(wait_interval)
                    elapsed += wait_interval
                elif current_state in ["STOPPED", "DELETING", "FAILED"]:
                    raise RuntimeError(
                        f"Database instance {instance_name} entered unexpected state: {current_state}"
                    )
                else:
                    logger.warning(
                        f"Unknown database state: {current_state}, continuing to wait..."
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
                    f"Database instance {database.instance_name} already exists with state: {existing_instance.state}"
                )

                # Check if database is in an intermediate state
                if existing_instance.state in ["STARTING", "UPDATING"]:
                    logger.info(
                        f"Database instance {database.instance_name} is in {existing_instance.state} state, waiting for it to become AVAILABLE..."
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
                            logger.debug(f"Database instance state: {current_state}")

                            if current_state == "AVAILABLE":
                                logger.info(
                                    f"Database instance {database.instance_name} is now AVAILABLE"
                                )
                                break
                            elif current_state in ["STARTING", "UPDATING"]:
                                logger.debug(
                                    f"Database instance still in {current_state} state, waiting {wait_interval} seconds..."
                                )
                                time.sleep(wait_interval)
                                elapsed += wait_interval
                            elif current_state in ["STOPPED", "DELETING"]:
                                logger.warning(
                                    f"Database instance {database.instance_name} is in unexpected state: {current_state}"
                                )
                                break
                            else:
                                logger.warning(
                                    f"Unknown database state: {current_state}, proceeding anyway"
                                )
                                break
                        except NotFound:
                            logger.warning(
                                f"Database instance {database.instance_name} no longer exists, will attempt to recreate"
                            )
                            break
                        except Exception as state_error:
                            logger.warning(
                                f"Could not check database state: {state_error}, proceeding anyway"
                            )
                            break

                    if elapsed >= max_wait_time:
                        logger.warning(
                            f"Timed out waiting for database instance {database.instance_name} to become AVAILABLE after {max_wait_time} seconds"
                        )

                elif existing_instance.state == "AVAILABLE":
                    logger.info(
                        f"Database instance {database.instance_name} already exists and is AVAILABLE"
                    )
                    return
                elif existing_instance.state in ["STOPPED", "DELETING"]:
                    logger.warning(
                        f"Database instance {database.instance_name} is in {existing_instance.state} state"
                    )
                    return
                else:
                    logger.info(
                        f"Database instance {database.instance_name} already exists with state: {existing_instance.state}"
                    )
                    return

        except NotFound:
            # Database doesn't exist, proceed with creation
            logger.debug(
                f"Database instance {database.instance_name} not found, creating new instance..."
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
                logger.info(
                    f"Successfully created database instance: {database.instance_name}"
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
                    or "RESOURCE_ALREADY_EXISTS" in error_msg
                ):
                    logger.info(
                        f"Database instance {database.instance_name} was created concurrently by another process"
                    )
                    # Still need to wait for the database to become AVAILABLE
                    self._wait_for_database_available(
                        workspace_client, database.instance_name
                    )
                    return
                else:
                    # Re-raise unexpected errors
                    logger.error(
                        f"Error creating database instance {database.instance_name}: {create_error}"
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
                    f"Database instance {database.instance_name} already exists (detected via exception)"
                )
                return
            else:
                logger.error(
                    f"Unexpected error while handling database {database.instance_name}: {e}"
                )
                raise

    def lakebase_password_provider(self, instance_name: str) -> str:
        """
        Ask Databricks to mint a fresh DB credential for this instance.
        """
        logger.debug(f"Generating password for lakebase instance: {instance_name}")
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
                f"client_id is required to create instance role for database {database.instance_name}"
            )
            return

        # Resolve the client_id value
        client_id: str = value_of(database.client_id)
        role_name: str = client_id
        instance_name: str = database.instance_name

        logger.debug(
            f"Creating instance role '{role_name}' for database {instance_name} with principal {client_id}"
        )

        try:
            # Check if role already exists
            try:
                _ = self.w.database.get_database_instance_role(
                    instance_name=instance_name,
                    name=role_name,
                )
                logger.info(
                    f"Instance role '{role_name}' already exists for database {instance_name}"
                )
                return
            except NotFound:
                # Role doesn't exist, proceed with creation
                logger.debug(
                    f"Instance role '{role_name}' not found, creating new role..."
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

            logger.info(
                f"Successfully created instance role '{role_name}' for database {instance_name}"
            )

        except Exception as e:
            error_msg: str = str(e)

            # Handle case where role was created concurrently
            if (
                "already exists" in error_msg.lower()
                or "RESOURCE_ALREADY_EXISTS" in error_msg
            ):
                logger.info(
                    f"Instance role '{role_name}' was created concurrently for database {instance_name}"
                )
                return

            # Re-raise unexpected errors
            logger.error(
                f"Error creating instance role '{role_name}' for database {instance_name}: {e}"
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
        4. Register default_template if provided

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
                logger.debug(
                    f"Loaded prompt '{prompt_name}' with explicit "
                    f"{'version ' + str(prompt_model.version) if prompt_model.version else 'alias ' + prompt_model.alias}"
                )
                return prompt_version
            except Exception as e:
                logger.warning(
                    f"Failed to load prompt '{prompt_name}' with explicit "
                    f"{'version ' + str(prompt_model.version) if prompt_model.version else 'alias ' + prompt_model.alias}: {e}"
                )
                # Fall through to try other methods

        # Try to load in priority order: champion → latest → default
        logger.debug(
            f"Trying fallback order for '{prompt_name}': champion → latest → default"
        )

        # 1. Try champion alias
        try:
            prompt_version = load_prompt(f"prompts:/{prompt_name}@champion")
            logger.info(f"Loaded prompt '{prompt_name}' from champion alias")
            return prompt_version
        except Exception as e:
            logger.debug(f"Champion alias not found for '{prompt_name}': {e}")

        # 2. Try latest alias
        try:
            prompt_version = load_prompt(f"prompts:/{prompt_name}@latest")
            logger.info(f"Loaded prompt '{prompt_name}' from latest alias")
            return prompt_version
        except Exception as e:
            logger.debug(f"Latest alias not found for '{prompt_name}': {e}")

        # 3. Try default alias
        try:
            prompt_version = load_prompt(f"prompts:/{prompt_name}@default")
            logger.info(f"Loaded prompt '{prompt_name}' from default alias")
            return prompt_version
        except Exception as e:
            logger.debug(f"Default alias not found for '{prompt_name}': {e}")

        # 4. Try to register default_template if provided
        if prompt_model.default_template:
            logger.info(
                f"No existing prompt found for '{prompt_name}', "
                "attempting to register default_template"
            )
            return self._register_default_template(
                prompt_name, prompt_model.default_template, prompt_model.description
            )

        raise ValueError(
            f"Prompt '{prompt_name}' not found in registry "
            "(tried champion alias, latest version, default alias) "
            "and no default_template provided"
        )

    def _register_default_template(
        self, prompt_name: str, default_template: str, description: str | None = None
    ) -> PromptVersion:
        """Register default_template as a new prompt version.

        Called when no existing prompt version is found (champion, latest, default all failed).
        Registers the template and sets both 'default' and 'champion' aliases.

        If registration fails (e.g., in Model Serving with restricted permissions),
        logs the error and raises.
        """
        logger.info(
            f"No existing prompt found for '{prompt_name}', attempting to register default_template"
        )

        try:
            commit_message = description or "Auto-synced from default_template"
            prompt_version = mlflow.genai.register_prompt(
                name=prompt_name,
                template=default_template,
                commit_message=commit_message,
                tags={"dao_ai": dao_ai_version()},
            )

            # Try to set aliases (may fail in restricted environments)
            try:
                mlflow.genai.set_prompt_alias(
                    name=prompt_name, alias="default", version=prompt_version.version
                )
                mlflow.genai.set_prompt_alias(
                    name=prompt_name, alias="champion", version=prompt_version.version
                )
                logger.info(
                    f"Registered prompt '{prompt_name}' v{prompt_version.version} with aliases"
                )
            except Exception as alias_error:
                logger.warning(
                    f"Registered prompt '{prompt_name}' v{prompt_version.version} "
                    f"but failed to set aliases: {alias_error}"
                )

            return prompt_version

        except Exception as reg_error:
            logger.error(
                f"Failed to register prompt '{prompt_name}': {reg_error}. "
                f"Please register the prompt from a notebook with write permissions before deployment."
            )
            return PromptVersion(
                name=prompt_name,
                version=1,
                template=default_template,
                tags={"dao_ai": dao_ai_version()},
            )
