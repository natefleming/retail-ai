import base64
from pathlib import Path
from typing import Any, Callable, Final, Sequence

import mlflow
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
from databricks.sdk.service.iam import User
from databricks.sdk.service.workspace import GetSecretResponse
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import Experiment
from mlflow.entities.model_registry.model_version import ModelVersion
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
    DatasetModel,
    FunctionModel,
    GenieRoomModel,
    HasFullName,
    IndexModel,
    IsDatabricksResource,
    LLMModel,
    SchemaModel,
    TableModel,
    UnityCatalogFunctionSqlModel,
    VectorStoreModel,
    VolumeModel,
    WarehouseModel,
)
from dao_ai.models import get_latest_model_version
from dao_ai.providers.base import ServiceProvider
from dao_ai.utils import get_installed_packages, is_installed, normalize_name
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
    if client_id and client_secret and workspace_host:
        return WorkspaceClient(
            host=workspace_host,
            client_id=client_id,
            client_secret=client_secret,
            auth_type="oauth-m2m",
        )
    elif pat:
        return WorkspaceClient(host=workspace_host, token=pat, auth_type="pat")
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
    if client_id and client_secret and workspace_host:
        return VectorSearchClient(
            workspace_url=workspace_host,
            service_principal_client_id=client_id,
            service_principal_client_secret=client_secret,
        )
    elif pat and workspace_host:
        return VectorSearchClient(
            workspace_url=workspace_host,
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

        resources: Sequence[IsDatabricksResource] = (
            llms
            + vector_indexes
            + warehouses
            + genie_rooms
            + functions
            + tables
            + connections
        )

        # all_resources: Sequence[DatabricksResource] = [
        #     r.as_resource() for r in resources
        # ]

        system_resources: Sequence[DatabricksResource] = [
            r.as_resource() for r in resources if not r.on_behalf_of_user
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

        pip_requirements: Sequence[str] = (
            get_installed_packages() + config.app.pip_requirements
        )

        code_paths: list[str] = config.app.code_paths
        for path in code_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Code path does not exist: {path}")

        model_root_path: Path = Path(dao_ai.__file__).parent
        model_path: Path = model_root_path / "agent_as_code.py"

        if is_installed():
            pip_requirements += [
                f"dao-ai=={dao_ai.__version__}",
            ]
        else:
            src_path: Path = model_root_path.parent
            directories: Sequence[Path] = [d for d in src_path.iterdir() if d.is_dir()]
            for directory in directories:
                directory: Path
                code_paths.append(directory.as_posix())

        logger.debug(f"pip_requirements: {pip_requirements}")
        logger.debug(f"code_paths: {code_paths}")

        run_name: str = normalize_name(config.app.name)
        logger.debug(f"run_name: {run_name}")
        logger.debug(f"model_path: {model_path.as_posix()}")

        input_example: dict[str, Any] = None
        if config.app.input_example:
            input_example = config.app.input_example.model_dump()

        logger.debug(f"input_example: {input_example}")

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("type", "agent")
            logged_agent_info: ModelInfo = mlflow.pyfunc.log_model(
                python_model=model_path.as_posix(),
                code_paths=code_paths,
                model_config=config.model_dump(by_alias=True),
                name="agent",
                pip_requirements=pip_requirements,
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

        client.set_registered_model_alias(
            name=registered_model_name,
            alias="Current",
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
        tags: dict[str, str] = config.app.tags

        latest_version: int = get_latest_model_version(registered_model_name)

        agents.deploy(
            endpoint_name=endpoint_name,
            model_name=registered_model_name,
            model_version=latest_version,
            scale_to_zero=scale_to_zero,
            environment_vars=environment_vars,
            workload_size=workload_size,
            tags=tags,
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
                spark.read.format(format).options(**read_options).load(
                    data_path.as_posix(),
                    schema=dataset.table_schema,
                ).write.mode("overwrite").saveAsTable(table)

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
            self.vsc.get_index(
                vector_store.endpoint.name, vector_store.index.full_name
            ).sync()

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
