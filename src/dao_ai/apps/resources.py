"""
App resources module for generating Databricks App resource configurations.

This module provides utilities to dynamically discover and generate Databricks App
resource configurations from dao-ai AppConfig. Resources are extracted from the
config and converted to the format expected by Databricks Apps.

Databricks Apps resource documentation:
https://learn.microsoft.com/en-us/azure/databricks/dev-tools/databricks-apps/resources

Supported resource types and their mappings:
- LLMModel → serving-endpoint (Model Serving Endpoint)
- VectorStoreModel/IndexModel → vector-search-index (via UC Securable - not yet supported)
- WarehouseModel → sql-warehouse
- GenieRoomModel → genie-space
- VolumeModel → volume (via UC Securable)
- FunctionModel → function (via UC Securable - not yet supported)
- ConnectionModel → connection (not yet supported in SDK)
- DatabaseModel → database (Lakebase)
- DatabricksAppModel → app (not yet supported in SDK)

Usage:
    from dao_ai.apps.resources import generate_app_resources, generate_sdk_resources
    from dao_ai.config import AppConfig

    config = AppConfig.from_file("model_config.yaml")

    # For SDK-based deployment (recommended)
    sdk_resources = generate_sdk_resources(config)

    # For YAML-based documentation
    resources = generate_app_resources(config)
"""

from typing import Any

from databricks.sdk.service.apps import (
    AppResource,
    AppResourceDatabase,
    AppResourceDatabaseDatabasePermission,
    AppResourceExperiment,
    AppResourceExperimentExperimentPermission,
    AppResourceGenieSpace,
    AppResourceGenieSpaceGenieSpacePermission,
    AppResourceSecret,
    AppResourceSecretSecretPermission,
    AppResourceServingEndpoint,
    AppResourceServingEndpointServingEndpointPermission,
    AppResourceSqlWarehouse,
    AppResourceSqlWarehouseSqlWarehousePermission,
    AppResourceUcSecurable,
    AppResourceUcSecurableUcSecurablePermission,
    AppResourceUcSecurableUcSecurableType,
)
from loguru import logger

from dao_ai.config import (
    AppConfig,
    CompositeVariableModel,
    ConnectionModel,
    DatabaseModel,
    DatabricksAppModel,
    EnvironmentVariableModel,
    FunctionModel,
    GenieRoomModel,
    IsDatabricksResource,
    LLMModel,
    SecretVariableModel,
    TableModel,
    VectorStoreModel,
    VolumeModel,
    WarehouseModel,
    value_of,
)

# Resource type mappings from dao-ai to Databricks Apps
RESOURCE_TYPE_MAPPING: dict[type, str] = {
    LLMModel: "serving-endpoint",
    VectorStoreModel: "vector-search-index",
    WarehouseModel: "sql-warehouse",
    GenieRoomModel: "genie-space",
    VolumeModel: "volume",
    FunctionModel: "function",
    ConnectionModel: "connection",
    DatabaseModel: "database",
    DatabricksAppModel: "app",
}

# Default permissions for each resource type
DEFAULT_PERMISSIONS: dict[str, list[str]] = {
    "serving-endpoint": ["CAN_QUERY"],
    "vector-search-index": ["CAN_SELECT"],
    "sql-warehouse": ["CAN_USE"],
    "genie-space": ["CAN_RUN"],
    "volume": ["CAN_READ"],
    "function": ["CAN_EXECUTE"],
    "connection": ["USE_CONNECTION"],
    "database": ["CAN_CONNECT_AND_CREATE"],
    "app": ["CAN_VIEW"],
}

# Valid user API scopes for Databricks Apps
# These are the only scopes that can be requested for on-behalf-of-user access
VALID_USER_API_SCOPES: set[str] = {
    "sql",
    "serving.serving-endpoints",
    "vectorsearch.vector-search-indexes",
    "files.files",
    "dashboards.genie",
    "catalog.connections",
    "catalog.catalogs:read",
    "catalog.schemas:read",
    "catalog.tables:read",
}

# Mapping from resource api_scopes to valid user_api_scopes
# Some resource scopes map directly, others need translation
API_SCOPE_TO_USER_SCOPE: dict[str, str] = {
    # Direct mappings
    "serving.serving-endpoints": "serving.serving-endpoints",
    "vectorsearch.vector-search-indexes": "vectorsearch.vector-search-indexes",
    "files.files": "files.files",
    "dashboards.genie": "dashboards.genie",
    "catalog.connections": "catalog.connections",
    # SQL-related scopes map to "sql"
    "sql.warehouses": "sql",
    "sql.statement-execution": "sql",
    # Vector search endpoints also need serving
    "vectorsearch.vector-search-endpoints": "serving.serving-endpoints",
    # Catalog scopes
    "catalog.volumes": "files.files",
}


def _extract_llm_resources(
    llms: dict[str, LLMModel],
) -> list[dict[str, Any]]:
    """Extract model serving endpoint resources from LLMModels."""
    resources: list[dict[str, Any]] = []
    for idx, (key, llm) in enumerate(llms.items()):
        resource: dict[str, Any] = {
            "name": key,
            "type": "serving-endpoint",
            "serving_endpoint_name": llm.name,
            "permissions": [
                {"level": p} for p in DEFAULT_PERMISSIONS["serving-endpoint"]
            ],
        }
        resources.append(resource)
        logger.debug(f"Extracted serving endpoint resource: {key} -> {llm.name}")
    return resources


def _extract_vector_search_resources(
    vector_stores: dict[str, VectorStoreModel],
) -> list[dict[str, Any]]:
    """Extract vector search index resources from VectorStoreModels."""
    resources: list[dict[str, Any]] = []
    for key, vs in vector_stores.items():
        if vs.index is None:
            continue
        resource: dict[str, Any] = {
            "name": key,
            "type": "vector-search-index",
            "vector_search_index_name": vs.index.full_name,
            "permissions": [
                {"level": p} for p in DEFAULT_PERMISSIONS["vector-search-index"]
            ],
        }
        resources.append(resource)
        logger.debug(f"Extracted vector search resource: {key} -> {vs.index.full_name}")
    return resources


def _extract_warehouse_resources(
    warehouses: dict[str, WarehouseModel],
) -> list[dict[str, Any]]:
    """Extract SQL warehouse resources from WarehouseModels."""
    resources: list[dict[str, Any]] = []
    for key, warehouse in warehouses.items():
        warehouse_id = value_of(warehouse.warehouse_id)
        resource: dict[str, Any] = {
            "name": key,
            "type": "sql-warehouse",
            "sql_warehouse_id": warehouse_id,
            "permissions": [{"level": p} for p in DEFAULT_PERMISSIONS["sql-warehouse"]],
        }
        resources.append(resource)
        logger.debug(f"Extracted SQL warehouse resource: {key} -> {warehouse_id}")
    return resources


def _extract_genie_resources(
    genie_rooms: dict[str, GenieRoomModel],
) -> list[dict[str, Any]]:
    """Extract Genie space resources from GenieRoomModels."""
    resources: list[dict[str, Any]] = []
    for key, genie in genie_rooms.items():
        space_id = value_of(genie.space_id)
        resource: dict[str, Any] = {
            "name": key,
            "type": "genie-space",
            "genie_space_id": space_id,
            "permissions": [{"level": p} for p in DEFAULT_PERMISSIONS["genie-space"]],
        }
        resources.append(resource)
        logger.debug(f"Extracted Genie space resource: {key} -> {space_id}")
    return resources


def _extract_volume_resources(
    volumes: dict[str, VolumeModel],
) -> list[dict[str, Any]]:
    """Extract UC Volume resources from VolumeModels."""
    resources: list[dict[str, Any]] = []
    for key, volume in volumes.items():
        resource: dict[str, Any] = {
            "name": key,
            "type": "volume",
            "volume_name": volume.full_name,
            "permissions": [{"level": p} for p in DEFAULT_PERMISSIONS["volume"]],
        }
        resources.append(resource)
        logger.debug(f"Extracted volume resource: {key} -> {volume.full_name}")
    return resources


def _extract_function_resources(
    functions: dict[str, FunctionModel],
) -> list[dict[str, Any]]:
    """Extract UC Function resources from FunctionModels."""
    resources: list[dict[str, Any]] = []
    for key, func in functions.items():
        resource: dict[str, Any] = {
            "name": key,
            "type": "function",
            "function_name": func.full_name,
            "permissions": [{"level": p} for p in DEFAULT_PERMISSIONS["function"]],
        }
        resources.append(resource)
        logger.debug(f"Extracted function resource: {key} -> {func.full_name}")
    return resources


def _extract_connection_resources(
    connections: dict[str, ConnectionModel],
) -> list[dict[str, Any]]:
    """Extract UC Connection resources from ConnectionModels."""
    resources: list[dict[str, Any]] = []
    for key, conn in connections.items():
        resource: dict[str, Any] = {
            "name": key,
            "type": "connection",
            "connection_name": conn.name,
            "permissions": [{"level": p} for p in DEFAULT_PERMISSIONS["connection"]],
        }
        resources.append(resource)
        logger.debug(f"Extracted connection resource: {key} -> {conn.name}")
    return resources


def _extract_database_resources(
    databases: dict[str, DatabaseModel],
) -> list[dict[str, Any]]:
    """Extract Lakebase database resources from DatabaseModels."""
    resources: list[dict[str, Any]] = []
    for key, db in databases.items():
        # Only include Lakebase databases (those with instance_name)
        if not db.is_lakebase:
            continue
        resource: dict[str, Any] = {
            "name": key,
            "type": "database",
            "database_instance_name": db.instance_name,
            "permissions": [{"level": p} for p in DEFAULT_PERMISSIONS["database"]],
        }
        resources.append(resource)
        logger.debug(f"Extracted database resource: {key} -> {db.instance_name}")
    return resources


def _extract_app_resources(
    apps: dict[str, DatabricksAppModel],
) -> list[dict[str, Any]]:
    """Extract Databricks App resources from DatabricksAppModels."""
    resources: list[dict[str, Any]] = []
    for key, app in apps.items():
        resource: dict[str, Any] = {
            "name": key,
            "type": "app",
            "app_name": app.name,
            "permissions": [{"level": p} for p in DEFAULT_PERMISSIONS["app"]],
        }
        resources.append(resource)
        logger.debug(f"Extracted app resource: {key} -> {app.name}")
    return resources


def _extract_secrets_from_config(config: AppConfig) -> list[dict[str, Any]]:
    """
    Extract all secrets referenced in the config as resources.

    This function walks through the entire config object to find all
    SecretVariableModel instances and extracts their scope and key.

    Args:
        config: The AppConfig containing secret references

    Returns:
        A list of secret resource dictionaries with unique scope/key pairs
    """
    secrets: dict[tuple[str, str], dict[str, Any]] = {}
    used_names: set[str] = set()

    def get_unique_resource_name(base_name: str) -> str:
        """Generate a unique resource name, adding suffix if needed."""
        sanitized = _sanitize_resource_name(base_name)
        if sanitized not in used_names:
            used_names.add(sanitized)
            return sanitized
        # Name collision - add numeric suffix
        counter = 1
        while True:
            # Leave room for suffix (e.g., "_1", "_2", etc.)
            suffix = f"_{counter}"
            max_base_len = 30 - len(suffix)
            candidate = sanitized[:max_base_len] + suffix
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
            counter += 1

    def extract_from_value(value: Any, path: str = "") -> None:
        """Recursively extract secrets from any value."""
        if isinstance(value, SecretVariableModel):
            secret_key = (value.scope, value.secret)
            if secret_key not in secrets:
                # Create a unique name for the secret resource
                base_name = f"{value.scope}_{value.secret}".replace("-", "_").replace(
                    "/", "_"
                )
                resource_name = get_unique_resource_name(base_name)
                secrets[secret_key] = {
                    "name": resource_name,
                    "type": "secret",
                    "scope": value.scope,
                    "key": value.secret,
                    "permissions": [{"level": "READ"}],
                }
                logger.debug(
                    f"Found secret: {value.scope}/{value.secret} at {path} -> resource: {resource_name}"
                )
        elif isinstance(value, dict):
            for k, v in value.items():
                extract_from_value(v, f"{path}.{k}" if path else k)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                extract_from_value(v, f"{path}[{i}]")
        elif hasattr(value, "__dict__"):
            # Handle Pydantic models and other objects with __dict__
            for k, v in value.__dict__.items():
                if not k.startswith("_"):  # Skip private attributes
                    extract_from_value(v, f"{path}.{k}" if path else k)

    # Walk through the entire config
    extract_from_value(config)

    resources = list(secrets.values())
    logger.info(f"Extracted {len(resources)} secret resources from config")
    return resources


def generate_app_resources(config: AppConfig) -> list[dict[str, Any]]:
    """
    Generate Databricks App resource configurations from an AppConfig.

    This function extracts all resources defined in the AppConfig and converts
    them to the format expected by Databricks Apps. Resources are used to
    grant the app's service principal access to Databricks platform features.

    Args:
        config: The AppConfig containing resource definitions

    Returns:
        A list of resource dictionaries in Databricks Apps format

    Example:
        >>> config = AppConfig.from_file("model_config.yaml")
        >>> resources = generate_app_resources(config)
        >>> print(resources)
        [
            {
                "name": "default_llm",
                "type": "serving-endpoint",
                "serving_endpoint_name": "databricks-claude-sonnet-4",
                "permissions": [{"level": "CAN_QUERY"}]
            },
            ...
        ]
    """
    resources: list[dict[str, Any]] = []

    if config.resources is None:
        logger.debug("No resources defined in config")
        return resources

    # Extract resources from each category
    resources.extend(_extract_llm_resources(config.resources.llms))
    resources.extend(_extract_vector_search_resources(config.resources.vector_stores))
    resources.extend(_extract_warehouse_resources(config.resources.warehouses))
    resources.extend(_extract_genie_resources(config.resources.genie_rooms))
    resources.extend(_extract_volume_resources(config.resources.volumes))
    resources.extend(_extract_function_resources(config.resources.functions))
    resources.extend(_extract_connection_resources(config.resources.connections))
    resources.extend(_extract_database_resources(config.resources.databases))
    resources.extend(_extract_app_resources(config.resources.apps))

    # Extract secrets from the entire config
    resources.extend(_extract_secrets_from_config(config))

    logger.info(f"Generated {len(resources)} app resources from config")
    return resources


def generate_user_api_scopes(config: AppConfig) -> list[str]:
    """
    Generate user API scopes from resources with on_behalf_of_user=True.

    This function examines all resources in the config and collects the
    API scopes needed for on-behalf-of-user authentication. Only valid
    user API scopes are returned.

    Args:
        config: The AppConfig containing resource definitions

    Returns:
        A list of unique user API scopes needed for OBO authentication

    Example:
        >>> config = AppConfig.from_file("model_config.yaml")
        >>> scopes = generate_user_api_scopes(config)
        >>> print(scopes)
        ['sql', 'serving.serving-endpoints', 'dashboards.genie']
    """
    scopes: set[str] = set()

    if config.resources is None:
        return []

    # Collect all resources that have on_behalf_of_user=True
    obo_resources: list[IsDatabricksResource] = []

    # Check each resource category
    for llm in config.resources.llms.values():
        if llm.on_behalf_of_user:
            obo_resources.append(llm)

    for vs in config.resources.vector_stores.values():
        if vs.on_behalf_of_user:
            obo_resources.append(vs)

    for warehouse in config.resources.warehouses.values():
        if warehouse.on_behalf_of_user:
            obo_resources.append(warehouse)

    for genie in config.resources.genie_rooms.values():
        if genie.on_behalf_of_user:
            obo_resources.append(genie)

    for volume in config.resources.volumes.values():
        if volume.on_behalf_of_user:
            obo_resources.append(volume)

    for func in config.resources.functions.values():
        if func.on_behalf_of_user:
            obo_resources.append(func)

    for conn in config.resources.connections.values():
        if conn.on_behalf_of_user:
            obo_resources.append(conn)

    for db in config.resources.databases.values():
        if db.on_behalf_of_user:
            obo_resources.append(db)

    for table in config.resources.tables.values():
        if table.on_behalf_of_user:
            obo_resources.append(table)

    # Collect api_scopes from all OBO resources and map to user_api_scopes
    for resource in obo_resources:
        for api_scope in resource.api_scopes:
            # Map the api_scope to a valid user_api_scope
            if api_scope in API_SCOPE_TO_USER_SCOPE:
                user_scope = API_SCOPE_TO_USER_SCOPE[api_scope]
                if user_scope in VALID_USER_API_SCOPES:
                    scopes.add(user_scope)
            elif api_scope in VALID_USER_API_SCOPES:
                # Direct match
                scopes.add(api_scope)

    # Always add catalog read scopes if we have any table or function access
    if any(isinstance(r, (TableModel, FunctionModel)) for r in obo_resources):
        scopes.add("catalog.catalogs:read")
        scopes.add("catalog.schemas:read")
        scopes.add("catalog.tables:read")

    # Sort for consistent ordering
    result = sorted(scopes)
    logger.info(f"Generated {len(result)} user API scopes for OBO resources: {result}")
    return result


def _sanitize_resource_name(name: str) -> str:
    """
    Sanitize a resource name to meet Databricks Apps requirements.

    Resource names must be:
    - Between 2 and 30 characters
    - Only contain alphanumeric characters, hyphens, and underscores

    Args:
        name: The original resource name

    Returns:
        A sanitized name that meets the requirements
    """
    # Replace dots and special characters with underscores
    sanitized = name.replace(".", "_").replace("-", "_")

    # Remove any characters that aren't alphanumeric or underscore
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")

    # Ensure minimum length of 2
    if len(sanitized) < 2:
        sanitized = sanitized + "_r"

    # Truncate to maximum length of 30
    if len(sanitized) > 30:
        sanitized = sanitized[:30]

    return sanitized


def generate_sdk_resources(
    config: AppConfig,
    experiment_id: str | None = None,
) -> list[AppResource]:
    """
    Generate Databricks SDK AppResource objects from an AppConfig.

    This function extracts all resources defined in the AppConfig and converts
    them to SDK AppResource objects that can be passed to the Apps API when
    creating or updating an app.

    Args:
        config: The AppConfig containing resource definitions
        experiment_id: Optional MLflow experiment ID to add as a resource.
            When provided, the experiment is added with CAN_EDIT permission,
            allowing the app to log traces and runs.

    Returns:
        A list of AppResource objects for the Databricks SDK

    Example:
        >>> from databricks.sdk import WorkspaceClient
        >>> from databricks.sdk.service.apps import App
        >>> config = AppConfig.from_file("model_config.yaml")
        >>> resources = generate_sdk_resources(config, experiment_id="12345")
        >>> w = WorkspaceClient()
        >>> app = App(name="my-app", resources=resources)
        >>> w.apps.create_and_wait(app=app)
    """
    resources: list[AppResource] = []

    # Add experiment resource if provided
    if experiment_id:
        resources.append(_extract_sdk_experiment_resource(experiment_id))

    if config.resources is None:
        logger.debug("No resources defined in config")
        return resources

    # Extract SDK resources from each category
    resources.extend(_extract_sdk_llm_resources(config.resources.llms))
    resources.extend(_extract_sdk_warehouse_resources(config.resources.warehouses))
    resources.extend(_extract_sdk_genie_resources(config.resources.genie_rooms))
    resources.extend(_extract_sdk_database_resources(config.resources.databases))
    resources.extend(_extract_sdk_volume_resources(config.resources.volumes))

    # Extract secrets from the entire config
    resources.extend(_extract_sdk_secrets_from_config(config))

    # Note: Vector search indexes, functions, and connections are not yet
    # supported as app resources in the SDK

    logger.info(f"Generated {len(resources)} SDK app resources from config")
    return resources


def _extract_sdk_llm_resources(
    llms: dict[str, LLMModel],
) -> list[AppResource]:
    """Extract SDK AppResource objects for model serving endpoints."""
    resources: list[AppResource] = []
    for key, llm in llms.items():
        sanitized_name = _sanitize_resource_name(key)
        resource = AppResource(
            name=sanitized_name,
            description=llm.description,
            serving_endpoint=AppResourceServingEndpoint(
                name=llm.name,
                permission=AppResourceServingEndpointServingEndpointPermission.CAN_QUERY,
            ),
        )
        resources.append(resource)
        logger.debug(
            f"Extracted SDK serving endpoint resource: {sanitized_name} -> {llm.name}"
        )
    return resources


def _extract_sdk_warehouse_resources(
    warehouses: dict[str, WarehouseModel],
) -> list[AppResource]:
    """Extract SDK AppResource objects for SQL warehouses."""
    resources: list[AppResource] = []
    for key, warehouse in warehouses.items():
        warehouse_id = value_of(warehouse.warehouse_id)
        sanitized_name = _sanitize_resource_name(key)
        resource = AppResource(
            name=sanitized_name,
            description=warehouse.description,
            sql_warehouse=AppResourceSqlWarehouse(
                id=warehouse_id,
                permission=AppResourceSqlWarehouseSqlWarehousePermission.CAN_USE,
            ),
        )
        resources.append(resource)
        logger.debug(
            f"Extracted SDK SQL warehouse resource: {sanitized_name} -> {warehouse_id}"
        )
    return resources


def _extract_sdk_genie_resources(
    genie_rooms: dict[str, GenieRoomModel],
) -> list[AppResource]:
    """Extract SDK AppResource objects for Genie spaces."""
    resources: list[AppResource] = []
    for key, genie in genie_rooms.items():
        space_id = value_of(genie.space_id)
        sanitized_name = _sanitize_resource_name(key)
        resource = AppResource(
            name=sanitized_name,
            description=genie.description,
            genie_space=AppResourceGenieSpace(
                name=genie.name or key,
                space_id=space_id,
                permission=AppResourceGenieSpaceGenieSpacePermission.CAN_RUN,
            ),
        )
        resources.append(resource)
        logger.debug(
            f"Extracted SDK Genie space resource: {sanitized_name} -> {space_id}"
        )
    return resources


def _extract_sdk_database_resources(
    databases: dict[str, DatabaseModel],
) -> list[AppResource]:
    """Extract SDK AppResource objects for Lakebase databases."""
    resources: list[AppResource] = []
    for key, db in databases.items():
        # Only include Lakebase databases (those with instance_name)
        if not db.is_lakebase:
            continue
        sanitized_name = _sanitize_resource_name(key)
        # Use db.database for the actual database name (defaults to "databricks_postgres")
        # db.name is just the config key/description, not the actual database name
        database_name = value_of(db.database) if db.database else "databricks_postgres"
        resource = AppResource(
            name=sanitized_name,
            description=db.description,
            database=AppResourceDatabase(
                instance_name=db.instance_name,
                database_name=database_name,
                permission=AppResourceDatabaseDatabasePermission.CAN_CONNECT_AND_CREATE,
            ),
        )
        resources.append(resource)
        logger.debug(
            f"Extracted SDK database resource: {sanitized_name} -> "
            f"{db.instance_name}/{database_name}"
        )
    return resources


def _extract_sdk_volume_resources(
    volumes: dict[str, VolumeModel],
) -> list[AppResource]:
    """Extract SDK AppResource objects for Unity Catalog volumes."""
    resources: list[AppResource] = []
    for key, volume in volumes.items():
        sanitized_name = _sanitize_resource_name(key)
        resource = AppResource(
            name=sanitized_name,
            uc_securable=AppResourceUcSecurable(
                securable_full_name=volume.full_name,
                securable_type=AppResourceUcSecurableUcSecurableType.VOLUME,
                permission=AppResourceUcSecurableUcSecurablePermission.READ_VOLUME,
            ),
        )
        resources.append(resource)
        logger.debug(
            f"Extracted SDK volume resource: {sanitized_name} -> {volume.full_name}"
        )
    return resources


def _extract_sdk_experiment_resource(
    experiment_id: str,
    resource_name: str = "experiment",
) -> AppResource:
    """Create SDK AppResource for MLflow experiment.

    This allows the Databricks App to log traces and runs to the specified
    MLflow experiment. The experiment ID is exposed via the MLFLOW_EXPERIMENT_ID
    environment variable using valueFrom: experiment in app.yaml.

    Args:
        experiment_id: The MLflow experiment ID
        resource_name: The resource key name (default: "experiment")

    Returns:
        An AppResource for the MLflow experiment
    """
    resource = AppResource(
        name=resource_name,
        experiment=AppResourceExperiment(
            experiment_id=experiment_id,
            permission=AppResourceExperimentExperimentPermission.CAN_EDIT,
        ),
    )
    logger.debug(
        f"Extracted SDK experiment resource: {resource_name} -> {experiment_id}"
    )
    return resource


def _extract_sdk_secrets_from_config(config: AppConfig) -> list[AppResource]:
    """
    Extract SDK AppResource objects for all secrets referenced in the config.

    This function walks through the entire config object to find all
    SecretVariableModel instances and creates AppResource objects with
    READ permission for each unique scope/key pair.

    Args:
        config: The AppConfig containing secret references

    Returns:
        A list of AppResource objects for secrets
    """
    secrets: dict[tuple[str, str], AppResource] = {}
    used_names: set[str] = set()

    def get_unique_resource_name(base_name: str) -> str:
        """Generate a unique resource name, adding suffix if needed."""
        sanitized = _sanitize_resource_name(base_name)
        if sanitized not in used_names:
            used_names.add(sanitized)
            return sanitized
        # Name collision - add numeric suffix
        counter = 1
        while True:
            # Leave room for suffix (e.g., "_1", "_2", etc.)
            suffix = f"_{counter}"
            max_base_len = 30 - len(suffix)
            candidate = sanitized[:max_base_len] + suffix
            if candidate not in used_names:
                used_names.add(candidate)
                return candidate
            counter += 1

    def extract_from_value(value: Any) -> None:
        """Recursively extract secrets from any value."""
        if isinstance(value, SecretVariableModel):
            secret_key = (value.scope, value.secret)
            if secret_key not in secrets:
                # Create a unique name for the secret resource
                base_name = f"{value.scope}_{value.secret}".replace("-", "_").replace(
                    "/", "_"
                )
                resource_name = get_unique_resource_name(base_name)

                resource = AppResource(
                    name=resource_name,
                    secret=AppResourceSecret(
                        scope=value.scope,
                        key=value.secret,
                        permission=AppResourceSecretSecretPermission.READ,
                    ),
                )
                secrets[secret_key] = resource
                logger.debug(
                    f"Found secret for SDK resource: {value.scope}/{value.secret} -> resource: {resource_name}"
                )
        elif isinstance(value, dict):
            for v in value.values():
                extract_from_value(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                extract_from_value(v)
        elif hasattr(value, "__dict__"):
            # Handle Pydantic models and other objects with __dict__
            for k, v in value.__dict__.items():
                if not k.startswith("_"):  # Skip private attributes
                    extract_from_value(v)

    # Walk through the entire config
    extract_from_value(config)

    resources = list(secrets.values())
    logger.info(f"Extracted {len(resources)} SDK secret resources from config")
    return resources


def generate_resources_yaml(config: AppConfig) -> str:
    """
    Generate the resources section of app.yaml as a YAML string.

    Args:
        config: The AppConfig containing resource definitions

    Returns:
        A YAML-formatted string for the resources section
    """
    import yaml

    resources = generate_app_resources(config)
    if not resources:
        return ""

    return yaml.dump(
        {"resources": resources}, default_flow_style=False, sort_keys=False
    )


def _extract_env_vars_from_config(config: AppConfig) -> list[dict[str, str]]:
    """
    Extract environment variables from config.app.environment_vars for app.yaml.

    This function converts the environment_vars dict from AppConfig into the
    format expected by Databricks Apps. For each variable:
    - EnvironmentVariableModel: Creates env var with "value" (the env var name)
    - SecretVariableModel: Creates env var with "valueFrom" referencing the secret resource
    - CompositeVariableModel: Uses the first option in the list to determine the type
    - Plain strings: Creates env var with "value"

    Args:
        config: The AppConfig containing environment variable definitions

    Returns:
        A list of environment variable dictionaries for app.yaml

    Example:
        >>> config = AppConfig.from_file("model_config.yaml")
        >>> env_vars = _extract_env_vars_from_config(config)
        >>> # Returns:
        >>> # [
        >>> #     {"name": "API_KEY", "valueFrom": "my_scope_api_key"},
        >>> #     {"name": "LOG_LEVEL", "value": "INFO"},
        >>> # ]
    """
    env_vars: list[dict[str, str]] = []

    if config.app is None:
        return env_vars

    environment_vars = config.app.environment_vars
    if not environment_vars:
        return env_vars

    for var_name, var_value in environment_vars.items():
        env_entry: dict[str, str] = {"name": var_name}

        # Determine the type of the variable and create appropriate entry
        resolved_type = _resolve_variable_type(var_value)

        if resolved_type is None:
            # Plain value - use as-is
            if isinstance(var_value, str):
                env_entry["value"] = var_value
            else:
                env_entry["value"] = str(var_value)
        elif isinstance(resolved_type, SecretVariableModel):
            # Secret reference - use valueFrom with sanitized resource name
            resource_name = f"{resolved_type.scope}_{resolved_type.secret}".replace(
                "-", "_"
            ).replace("/", "_")
            resource_name = _sanitize_resource_name(resource_name)
            env_entry["valueFrom"] = resource_name
            logger.debug(
                f"Environment variable {var_name} references secret: "
                f"{resolved_type.scope}/{resolved_type.secret}"
            )
        elif isinstance(resolved_type, EnvironmentVariableModel):
            # Environment variable - resolve the value
            resolved_value = value_of(resolved_type)
            if resolved_value is not None:
                env_entry["value"] = str(resolved_value)
            elif resolved_type.default_value is not None:
                env_entry["value"] = str(resolved_type.default_value)
            else:
                # Skip if no value can be resolved
                logger.warning(
                    f"Environment variable {var_name} has no value "
                    f"(env: {resolved_type.env})"
                )
                continue
        else:
            # Other types - convert to string
            env_entry["value"] = str(var_value)

        env_vars.append(env_entry)
        logger.debug(f"Extracted environment variable: {var_name}")

    logger.info(f"Extracted {len(env_vars)} environment variables from config")
    return env_vars


def _resolve_variable_type(
    value: Any,
) -> SecretVariableModel | EnvironmentVariableModel | None:
    """
    Resolve the type of a variable for environment variable extraction.

    For CompositeVariableModel, returns the first option in the list to
    determine whether to use value or valueFrom in the app.yaml.

    Args:
        value: The variable value to analyze

    Returns:
        The resolved variable model (SecretVariableModel or EnvironmentVariableModel),
        or None if it's a plain value
    """
    if isinstance(value, SecretVariableModel):
        return value
    elif isinstance(value, EnvironmentVariableModel):
        return value
    elif isinstance(value, CompositeVariableModel):
        # Use the first option to determine the type
        if value.options:
            first_option = value.options[0]
            return _resolve_variable_type(first_option)
        return None
    else:
        # Plain value (str, int, etc.) or PrimitiveVariableModel
        return None


def generate_app_yaml(
    config: AppConfig,
    command: str | list[str] | None = None,
    include_resources: bool = True,
) -> str:
    """
    Generate a complete app.yaml for Databricks Apps deployment.

    This function creates a complete app.yaml configuration file that includes:
    - Command to run the app
    - Environment variables for MLflow and dao-ai
    - Resources extracted from the AppConfig (if include_resources is True)

    Args:
        config: The AppConfig containing deployment configuration
        command: Optional custom command. If not provided, uses default dao-ai app_server
        include_resources: Whether to include the resources section (default: True)

    Returns:
        A complete app.yaml as a string

    Example:
        >>> config = AppConfig.from_file("model_config.yaml")
        >>> app_yaml = generate_app_yaml(config)
        >>> print(app_yaml)
    """
    import yaml

    # Build the app.yaml structure
    app_config: dict[str, Any] = {}

    # Command section
    if command is None:
        app_config["command"] = [
            "/bin/bash",
            "-c",
            "pip install dao-ai && python -m dao_ai.apps.server",
        ]
    elif isinstance(command, str):
        app_config["command"] = [command]
    else:
        app_config["command"] = command

    # Base environment variables for MLflow and dao-ai
    env_vars: list[dict[str, str]] = [
        {"name": "MLFLOW_TRACKING_URI", "value": "databricks"},
        {"name": "MLFLOW_REGISTRY_URI", "value": "databricks-uc"},
        {"name": "MLFLOW_EXPERIMENT_ID", "valueFrom": "experiment"},
        {"name": "DAO_AI_CONFIG_PATH", "value": "dao_ai.yaml"},
    ]

    # Extract environment variables from config.app.environment_vars
    config_env_vars = _extract_env_vars_from_config(config)

    # Environment variables that are automatically provided by Databricks Apps
    # and should not be included in app.yaml
    platform_provided_env_vars = {"DATABRICKS_HOST"}

    # Filter out platform-provided env vars from config
    config_env_vars = [
        e for e in config_env_vars if e["name"] not in platform_provided_env_vars
    ]

    # Merge config env vars, avoiding duplicates (config takes precedence)
    base_env_names = {e["name"] for e in env_vars}
    for config_env in config_env_vars:
        if config_env["name"] not in base_env_names:
            env_vars.append(config_env)
        else:
            # Config env var takes precedence - replace the base one
            env_vars = [e for e in env_vars if e["name"] != config_env["name"]]
            env_vars.append(config_env)

    app_config["env"] = env_vars

    # Resources section (if requested)
    if include_resources:
        resources = generate_app_resources(config)
        if resources:
            app_config["resources"] = resources

    return yaml.dump(app_config, default_flow_style=False, sort_keys=False)


def get_resource_env_mappings(config: AppConfig) -> list[dict[str, Any]]:
    """
    Generate environment variable mappings that reference app resources.

    This creates environment variables that use `valueFrom` to reference
    configured resources, allowing the app to access resource values at runtime.

    Args:
        config: The AppConfig containing resource definitions

    Returns:
        A list of environment variable definitions with valueFrom references

    Example:
        >>> env_vars = get_resource_env_mappings(config)
        >>> # Returns:
        >>> # [
        >>> #     {"name": "SQL_WAREHOUSE_ID", "valueFrom": "default_warehouse"},
        >>> #     ...
        >>> # ]
    """
    env_mappings: list[dict[str, Any]] = []

    if config.resources is None:
        return env_mappings

    # Map warehouse IDs
    for key, warehouse in config.resources.warehouses.items():
        env_mappings.append(
            {
                "name": f"{key.upper()}_WAREHOUSE_ID",
                "valueFrom": key,
            }
        )

    # Map serving endpoint names
    for key, llm in config.resources.llms.items():
        env_mappings.append(
            {
                "name": f"{key.upper()}_ENDPOINT",
                "valueFrom": key,
            }
        )

    # Map Genie space IDs
    for key, genie in config.resources.genie_rooms.items():
        env_mappings.append(
            {
                "name": f"{key.upper()}_SPACE_ID",
                "valueFrom": key,
            }
        )

    # Map vector search indexes
    for key, vs in config.resources.vector_stores.items():
        if vs.index:
            env_mappings.append(
                {
                    "name": f"{key.upper()}_INDEX",
                    "valueFrom": key,
                }
            )

    # Map database instances
    for key, db in config.resources.databases.items():
        if db.is_lakebase:
            env_mappings.append(
                {
                    "name": f"{key.upper()}_DATABASE",
                    "valueFrom": key,
                }
            )

    return env_mappings
