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
    TraceLocationModel,
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
    "table": ["CAN_SELECT"],
    "volume": ["CAN_READ"],
    "function": ["CAN_EXECUTE"],
    "connection": ["USE_CONNECTION"],
    "database": ["CAN_CONNECT_AND_CREATE"],   # deprecated provisioned Lakebase
    "postgres": ["CAN_CONNECT_AND_CREATE"],    # autoscaling Lakebase project
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
# Some resource scopes map directly, others need translation.
#
# Resource-level api_scopes that have *no* corresponding user_api_scope
# (Databricks Apps does not expose an OBO scope for them today) are
# deliberately omitted and fall through to the no-op branch in
# generate_user_api_scopes:
#   - "apps.apps"                      (DatabricksAppModel — no cross-app OBO)
#   - "postgres" / "database.database-instances"  (Lakebase — no OBO scope;
#     non-OBO SP access is granted via the AppResourceDatabase resource
#     binding emitted by _extract_database_resources)
#   - "mcp.genie" / "mcp.functions" / "mcp.vectorsearch" / "mcp.external"
#     (MCP resource scopes; OBO falls back to the underlying Databricks
#      scopes on the same resource, e.g. catalog.connections +
#      serving.serving-endpoints on a ConnectionModel)
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
    # MCP resource scopes: an OBO MCP server is reached via serving
    # endpoints / UC connections, so surface the matching platform scopes.
    "mcp.genie": "dashboards.genie",
    "mcp.functions": "sql",
    "mcp.vectorsearch": "vectorsearch.vector-search-indexes",
    "mcp.external": "serving.serving-endpoints",
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


def _extract_table_resources(
    tables: dict[str, TableModel],
) -> list[dict[str, Any]]:
    """Extract UC Table resources from TableModels.

    Produces flat app.yaml-format dicts with type 'table'. When converted to
    bundle format, these become uc_securable resources with TABLE type and
    SELECT permission, which automatically grants USE CATALOG and USE SCHEMA.
    """
    resources: list[dict[str, Any]] = []
    for key, table in tables.items():
        if table.on_behalf_of_user:
            continue
        resource: dict[str, Any] = {
            "name": key,
            "type": "table",
            "table_name": table.full_name,
            "permissions": [{"level": p} for p in DEFAULT_PERMISSIONS["table"]],
        }
        resources.append(resource)
        logger.debug(f"Extracted table resource: {key} -> {table.full_name}")
    return resources


def _extract_genie_warehouse_resources(
    genie_rooms: dict[str, GenieRoomModel],
    existing_warehouse_ids: set[str],
) -> list[dict[str, Any]]:
    """Extract SQL warehouse resources from Genie rooms.

    Each Genie space is backed by a SQL warehouse. The app's service principal
    needs CAN_USE on that warehouse to execute queries through Genie.
    Skips warehouses already declared in resources.warehouses.
    """
    resources: list[dict[str, Any]] = []
    seen_ids: set[str] = set(existing_warehouse_ids)
    for key, genie in genie_rooms.items():
        warehouse = genie.warehouse
        if warehouse is None:
            continue
        wh_id = value_of(warehouse.warehouse_id)
        if wh_id in seen_ids:
            continue
        seen_ids.add(wh_id)
        resource: dict[str, Any] = {
            "name": f"{key}_warehouse",
            "type": "sql-warehouse",
            "sql_warehouse_id": wh_id,
            "permissions": [{"level": p} for p in DEFAULT_PERMISSIONS["sql-warehouse"]],
        }
        resources.append(resource)
        logger.debug(f"Extracted Genie warehouse resource: {key}_warehouse -> {wh_id}")
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
    """Extract Lakebase autoscaling-project resources from DatabaseModels.

    When a Lakebase project is registered as a Databricks App resource
    of type ``postgres``, the platform grants the app's auto-generated
    service principal ``CAN_CONNECT_AND_CREATE`` on that project -- no
    pre-created SP or secret-scope wiring needed.

    Resource shape: Databricks Apps exposes two database-shaped
    resource types -- ``database`` (deprecated, provisioned-instance
    Lakebase) and ``postgres`` (autoscaling Lakebase projects). Lakebase
    provisioned instances are deprecated; only the autoscaling project
    flow is supported going forward, so this emits the ``postgres``
    shape and points it at the project name.

    Standalone PostgreSQL connections (``host:`` set, no ``project:``)
    have no Databricks-managed resource binding and are skipped.
    OBO databases are skipped too -- the user identity handles
    permissions via ``user_api_scopes``.
    """
    resources: list[dict[str, Any]] = []
    for key, db in databases.items():
        if not db.is_lakebase:
            continue
        if db.on_behalf_of_user:
            continue
        sanitized_name: str = _sanitize_resource_name(key)
        # Apps platform requires both `branch` and `database` as FULL
        # resource paths on a postgres resource. The platform validates each
        # by calling get_branch / get_database server-side, which reject
        # bare IDs with INVALID_PARAMETER_VALUE.
        branch_path: str = _resolve_lakebase_branch_path(db)
        database_path: str = _resolve_lakebase_database_path(db, branch_path)
        resource: dict[str, Any] = {
            "name": sanitized_name,
            "type": "postgres",
            "database": database_path,
            "branch": branch_path,
            "permissions": [{"level": p} for p in DEFAULT_PERMISSIONS["database"]],
        }
        resources.append(resource)
        logger.debug(
            f"Extracted Lakebase postgres resource: {sanitized_name} -> "
            f"database={database_path} branch={branch_path}"
        )
    return resources


def _resolve_lakebase_branch_path(db: DatabaseModel) -> str:
    """Return the Apps-platform branch resource path for a Lakebase database.

    The Apps platform's ``postgres`` resource requires the ``branch`` field
    in full resource form: ``projects/{project_id}/branches/{branch_id}``.
    The platform validates by calling its own ``get_branch(name=...)`` API
    server-side, which rejects bare branch IDs with
    ``INVALID_PARAMETER_VALUE``.

    When ``db.branch`` is pinned in config we treat it as the branch ID and
    wrap it. Otherwise we resolve the project's default branch via
    ``DatabaseModel.resolve_default_branch()``. Falls back to ``"main"`` as
    the branch ID if the API call fails (offline bundle generation, missing
    creds, etc.) -- most Lakebase projects use ``main`` as the default.
    """
    branch_id: str
    if db.branch:
        branch_id = db.branch
    else:
        try:
            branch_id = db.resolve_default_branch()
        except Exception as e:  # pragma: no cover -- defensive fallback
            logger.debug(
                f"Could not resolve default branch for project '{db.project}': {e}. "
                f"Falling back to 'main'."
            )
            branch_id = "main"
    return f"projects/{db.project}/branches/{branch_id}"


def _resolve_lakebase_database_path(db: DatabaseModel, branch_path: str) -> str:
    """Return the Apps-platform database resource path for a Lakebase database.

    The Apps platform's ``postgres`` resource ``database`` field requires
    full resource form:
    ``projects/{project_id}/branches/{branch_id}/databases/{database_id}``.

    The Lakebase ``Database`` resource carries two distinct names:

    1. ``Database.name`` -- the platform resource path (random ID suffix,
       e.g. ``db-pmke-laez4ing20``). This is what the Apps platform binds to.
    2. ``Database.status.postgres_database`` -- the Postgres-level database
       name used by the connection (e.g. ``databricks_postgres``). This is
       what ``DatabaseModel.database`` carries in dao-ai configs.

    Resolution rules:

    * If ``db.database`` already looks like a full resource path
      (``projects/.../databases/...``), return it as-is.
    * Otherwise, list databases under the branch and find the one whose
      ``status.postgres_database`` matches ``db.database``; return its
      ``name``.
    * If no match is found, fall back to the first database under the
      branch (most Lakebase projects have exactly one).
    * If the API call fails entirely, fall back to constructing a path
      from ``db.database`` -- the deploy will surface a clear platform
      error if the constructed path doesn't exist.
    """
    from dao_ai.config import value_of

    database_value = value_of(db.database) if db.database is not None else None
    database_id: str = str(database_value) if database_value is not None else ""

    if database_id.startswith("projects/") and "/databases/" in database_id:
        return database_id

    try:
        w = db.workspace_client
        databases = list(w.postgres.list_databases(branch_path))
    except Exception as e:  # pragma: no cover -- defensive fallback
        logger.debug(
            f"Could not list databases under '{branch_path}': {e}. "
            f"Falling back to constructed path with database_id='{database_id}'."
        )
        return f"{branch_path}/databases/{database_id}"

    if database_id:
        for d in databases:
            pg_name = d.status.postgres_database if d.status else None
            if pg_name == database_id and d.name:
                return d.name

    if databases and databases[0].name:
        if database_id:
            logger.debug(
                f"No Lakebase database matched postgres name '{database_id}' under "
                f"'{branch_path}'; falling back to first database "
                f"'{databases[0].name}'."
            )
        return databases[0].name

    return f"{branch_path}/databases/{database_id}"


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
    resources.extend(_extract_connection_resources(config.resources.connections))
    resources.extend(_extract_database_resources(config.resources.databases))
    resources.extend(_extract_app_resources(config.resources.apps))

    # Genie room dependencies: tables, functions, and warehouses.
    # The update_genie_tables/functions validators run before resolution,
    # so we also pull directly from resolved Genie rooms as a fallback.
    all_tables: dict[str, TableModel] = dict(config.resources.tables)
    all_functions: dict[str, FunctionModel] = dict(config.resources.functions)
    for _room_key, genie_room in config.resources.genie_rooms.items():
        for table in genie_room.tables:
            short_name: str = table.full_name.rsplit(".", 1)[-1]
            table_key: str = f"tbl_{short_name}"
            if table.full_name not in {t.full_name for t in all_tables.values()}:
                all_tables[table_key] = table
        for func in genie_room.functions:
            short_name = func.full_name.rsplit(".", 1)[-1]
            func_key: str = f"fn_{short_name}"
            if func.full_name not in {f.full_name for f in all_functions.values()}:
                all_functions[func_key] = func
    resources.extend(_extract_table_resources(all_tables))
    resources.extend(_extract_function_resources(all_functions))

    existing_wh_ids: set[str] = {
        value_of(w.warehouse_id) for w in config.resources.warehouses.values()
    }
    resources.extend(
        _extract_genie_warehouse_resources(
            config.resources.genie_rooms, existing_wh_ids
        )
    )

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

    # Vector search indexes, functions, and connections use uc_securable
    # but some types (e.g. VECTOR_SEARCH_INDEX) are not yet in the SDK enum.
    # These are added as raw dicts and must be merged separately.
    # See generate_deployment_resources() for the combined output.

    logger.info(f"Generated {len(resources)} SDK app resources from config")
    return resources


def _extract_sdk_llm_resources(
    llms: dict[str, LLMModel],
) -> list[AppResource]:
    """Extract SDK AppResource objects for model serving endpoints.
    Skips OBO resources — user identity handles permissions via user_api_scopes."""
    resources: list[AppResource] = []
    for key, llm in llms.items():
        if llm.on_behalf_of_user:
            continue
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
    """Extract SDK AppResource objects for SQL warehouses.
    Skips OBO resources."""
    resources: list[AppResource] = []
    for key, warehouse in warehouses.items():
        if warehouse.on_behalf_of_user:
            continue
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
    """Extract SDK AppResource objects for Genie spaces.
    Skips OBO resources."""
    resources: list[AppResource] = []
    for key, genie in genie_rooms.items():
        if genie.on_behalf_of_user:
            continue
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
    """Extract SDK AppResource objects for Lakebase autoscaling projects.

    Mirror of ``_extract_database_resources`` for SDK-format deploys.
    Uses ``AppResourcePostgres`` (autoscaling) rather than
    ``AppResourceDatabase`` (deprecated provisioned-instance shape).
    Standalone PostgreSQL and OBO databases are skipped.

    The Apps platform requires both ``database`` (project name) and
    ``branch`` (full resource path ``projects/<p>/branches/<b>``) to be
    defined; ``branch`` is resolved from the project's default if the user
    didn't pin one in the config.
    """
    from databricks.sdk.service.apps import (
        AppResourcePostgres,
        AppResourcePostgresPostgresPermission,
    )

    resources: list[AppResource] = []
    for key, db in databases.items():
        if not db.is_lakebase:
            continue
        if db.on_behalf_of_user:
            continue
        sanitized_name: str = _sanitize_resource_name(key)
        branch_path: str = _resolve_lakebase_branch_path(db)
        database_path: str = _resolve_lakebase_database_path(db, branch_path)
        resource = AppResource(
            name=sanitized_name,
            postgres=AppResourcePostgres(
                database=database_path,
                branch=branch_path,
                permission=AppResourcePostgresPostgresPermission.CAN_CONNECT_AND_CREATE,
            ),
        )
        resources.append(resource)
        logger.debug(
            f"Extracted SDK Lakebase postgres resource: {sanitized_name} -> "
            f"database={database_path} branch={branch_path}"
        )
    return resources


def _extract_sdk_volume_resources(
    volumes: dict[str, VolumeModel],
) -> list[AppResource]:
    """Extract SDK AppResource objects for Unity Catalog volumes.
    Skips OBO resources."""
    resources: list[AppResource] = []
    for key, volume in volumes.items():
        if volume.on_behalf_of_user:
            continue
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


def _extract_raw_vector_search_resources(
    vector_stores: dict[str, VectorStoreModel],
) -> list[dict[str, Any]]:
    """
    Extract vector search index resources as raw dicts for the REST API.

    Adds a VECTOR_SEARCH_INDEX UC securable for each index that uses
    service principal auth (not OBO). When on_behalf_of_user=True, the
    calling user's identity provides all permissions via user_api_scopes,
    so no app resource is needed.

    Not all workspaces support VECTOR_SEARCH_INDEX as a UC securable
    type yet — the deployment code retries without it if rejected.
    """
    resources: list[dict[str, Any]] = []
    for key, vs in vector_stores.items():
        if vs.index is None:
            continue

        # OBO vector stores don't need app resources — the user's
        # identity provides permissions via user_api_scopes.
        if vs.on_behalf_of_user:
            logger.debug(f"Skipping vector search resource for OBO store: {key}")
            continue

        sanitized_name = _sanitize_resource_name(key)
        resources.append(
            {
                "name": sanitized_name,
                "uc_securable": {
                    "securable_full_name": vs.index.full_name,
                    "securable_type": "VECTOR_SEARCH_INDEX",
                    "permission": "SELECT",
                },
            }
        )
        logger.debug(
            f"Extracted vector search index resource: "
            f"{sanitized_name} -> {vs.index.full_name}"
        )
    return resources


def _extract_raw_function_resources(
    functions: dict[str, FunctionModel],
) -> list[dict[str, Any]]:
    """
    Extract UC function resources as raw dicts for the REST API.

    Uses the uc_securable format with FUNCTION type and EXECUTE permission.
    Skips OBO resources — user identity handles permissions via user_api_scopes.
    """
    resources: list[dict[str, Any]] = []
    for key, func in functions.items():
        if func.on_behalf_of_user:
            continue
        sanitized_name = _sanitize_resource_name(key)
        resource: dict[str, Any] = {
            "name": sanitized_name,
            "uc_securable": {
                "securable_full_name": func.full_name,
                "securable_type": "FUNCTION",
                "permission": "EXECUTE",
            },
        }
        resources.append(resource)
        logger.debug(
            f"Extracted raw function resource: {sanitized_name} -> {func.full_name}"
        )
    return resources


def _extract_raw_table_resources(
    tables: dict[str, TableModel],
) -> list[dict[str, Any]]:
    """
    Extract UC table resources as raw dicts for the REST API.

    Uses the uc_securable format with TABLE type and SELECT permission.
    Skips OBO resources — user identity handles permissions via user_api_scopes.
    """
    resources: list[dict[str, Any]] = []
    for key, table in tables.items():
        if table.on_behalf_of_user:
            continue
        sanitized_name = _sanitize_resource_name(key)
        resource: dict[str, Any] = {
            "name": sanitized_name,
            "uc_securable": {
                "securable_full_name": table.full_name,
                "securable_type": "TABLE",
                "permission": "SELECT",
            },
        }
        resources.append(resource)
        logger.debug(
            f"Extracted raw table resource: {sanitized_name} -> {table.full_name}"
        )
    return resources


def _extract_raw_connection_resources(
    connections: dict[str, ConnectionModel],
) -> list[dict[str, Any]]:
    """
    Extract UC connection resources as raw dicts for the REST API.

    Uses the uc_securable format with CONNECTION type and USE_CONNECTION
    permission. Skips OBO resources.
    """
    resources: list[dict[str, Any]] = []
    for key, conn in connections.items():
        if conn.on_behalf_of_user:
            continue
        sanitized_name = _sanitize_resource_name(key)
        resource: dict[str, Any] = {
            "name": sanitized_name,
            "uc_securable": {
                "securable_full_name": conn.full_name,
                "securable_type": "CONNECTION",
                "permission": "USE_CONNECTION",
            },
        }
        resources.append(resource)
        logger.debug(
            f"Extracted raw connection resource: {sanitized_name} -> {conn.full_name}"
        )
    return resources


def _extract_raw_trace_location_resources(
    trace_location: TraceLocationModel,
    existing_warehouse_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Extract trace location resources as raw dicts for the REST API.

    Generates:
    - A SQL warehouse resource for the trace warehouse (if not already
      present in existing_warehouse_ids)
    - TABLE UC securables for the 5 OTEL trace tables with SELECT permission,
      granting the app SP USE CATALOG + USE SCHEMA + SELECT

    Args:
        trace_location: The TraceLocationModel from AppModel
        existing_warehouse_ids: Set of warehouse IDs already added as resources,
            to avoid duplicates
    """
    resources: list[dict[str, Any]] = []

    # Add the trace warehouse
    try:
        wh_id = trace_location.warehouse_id
    except Exception:
        wh_id = None

    if wh_id and (
        existing_warehouse_ids is None or wh_id not in existing_warehouse_ids
    ):
        resource: dict[str, Any] = {
            "name": _sanitize_resource_name("trace_warehouse"),
            "sql_warehouse": {
                "id": wh_id,
                "permission": "CAN_USE",
            },
        }
        resources.append(resource)
        logger.debug(f"Extracted trace warehouse resource: trace_warehouse -> {wh_id}")

    # Add OTEL trace tables as TABLE UC securables
    # Use short names to stay within the 30-char resource name limit
    SHORT_SUFFIX_NAMES: dict[str, str] = {
        "mlflow_experiment_trace_otel_spans": "trace_otel_spans",
        "mlflow_experiment_trace_otel_logs": "trace_otel_logs",
        "mlflow_experiment_trace_otel_metrics": "trace_otel_metrics",
    }
    schema_prefix = f"{trace_location.catalog_name}.{trace_location.schema_name}"
    for suffix in TraceLocationModel.OTEL_TABLE_SUFFIXES:
        table_full_name = f"{schema_prefix}.{suffix}"
        short_name = SHORT_SUFFIX_NAMES.get(suffix, suffix)
        sanitized_name = _sanitize_resource_name(short_name)
        table_resource: dict[str, Any] = {
            "name": sanitized_name,
            "uc_securable": {
                "securable_full_name": table_full_name,
                "securable_type": "TABLE",
                "permission": "SELECT",
            },
        }
        resources.append(table_resource)
        logger.debug(
            f"Extracted trace table resource: {sanitized_name} -> {table_full_name}"
        )

    return resources


def generate_deployment_resources(
    config: AppConfig,
    experiment_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Generate ALL app resources as raw dicts for REST API deployment.

    This combines SDK-serializable resources with resources that require
    types not yet supported by the SDK enum (e.g. VECTOR_SEARCH_INDEX).
    The output is suitable for direct use with the Databricks REST API.

    Args:
        config: The AppConfig containing resource definitions
        experiment_id: Optional MLflow experiment ID to add as a resource

    Returns:
        A list of resource dicts for the Databricks REST API
    """
    resources: list[dict[str, Any]] = []

    # Serialize SDK-supported resources to dicts
    sdk_resources = generate_sdk_resources(config, experiment_id=experiment_id)
    for r in sdk_resources:
        resources.append(r.as_dict())

    if config.resources is not None:
        # Add resources not yet in the SDK's generate_sdk_resources()
        resources.extend(
            _extract_raw_vector_search_resources(config.resources.vector_stores)
        )
        resources.extend(_extract_raw_function_resources(config.resources.functions))
        # Include all tables as UC securable app resources.  Genie space
        # resources only grant CAN_RUN on the space — not SELECT on the
        # underlying tables.  Per-table uc_securable resources are the
        # sanctioned way to grant the auto-created Apps SP access.
        # The progressive fallback in _set_app_resources handles the
        # 20-resource limit.
        #
        # Note: config.resources.tables may be empty if the
        # update_genie_tables validator ran before Genie rooms were
        # resolved.  Collect tables directly from resolved rooms as a
        # fallback.
        all_tables: dict[str, TableModel] = dict(config.resources.tables)
        for room_key, genie_room in config.resources.genie_rooms.items():
            for table in genie_room.tables:
                table_key = f"{room_key}_{table.full_name}".replace(".", "_")
                if table.full_name not in {t.full_name for t in all_tables.values()}:
                    all_tables[table_key] = table
        if all_tables:
            resources.extend(_extract_raw_table_resources(all_tables))
        resources.extend(
            _extract_raw_connection_resources(config.resources.connections)
        )

    # Add trace location resources (warehouse + OTEL tables)
    if config.app and config.app.trace_location:
        # Collect existing warehouse IDs to avoid duplicates
        existing_wh_ids: set[str] = set()
        if config.resources:
            for wh in config.resources.warehouses.values():
                try:
                    existing_wh_ids.add(value_of(wh.warehouse_id))
                except Exception:
                    pass
        resources.extend(
            _extract_raw_trace_location_resources(
                config.app.trace_location,
                existing_warehouse_ids=existing_wh_ids,
            )
        )

    # Deduplicate resources by name. SDK resources (added first) take
    # priority over raw resources. For uc_securable resources with the same
    # securable_full_name, keep only the first occurrence.
    seen_names: set[str] = set()
    seen_securables: set[str] = set()
    deduplicated: list[dict[str, Any]] = []
    for r in resources:
        name = r.get("name", "")
        securable_fn = (
            r.get("uc_securable", {}).get("securable_full_name")
            if "uc_securable" in r
            else None
        )

        # Skip duplicate securable_full_name (same UC entity)
        if securable_fn and securable_fn in seen_securables:
            logger.debug(f"Skipping duplicate UC securable: {securable_fn}")
            continue

        # Ensure unique resource names by appending a counter
        unique_name = name
        if unique_name in seen_names:
            counter = 1
            while True:
                suffix = f"_{counter}"
                candidate = name[: 30 - len(suffix)] + suffix
                if candidate not in seen_names:
                    unique_name = candidate
                    r["name"] = unique_name
                    break
                counter += 1

        seen_names.add(unique_name)
        if securable_fn:
            seen_securables.add(securable_fn)
        deduplicated.append(r)

    if len(deduplicated) > 20:
        logger.warning(
            f"App resource limit is 20 but {len(deduplicated)} resources were "
            f"generated. The deployment may fail — consider reducing the number "
            f"of resources in your config.",
        )

    logger.info(
        f"Generated {len(deduplicated)} deployment resources "
        f"(from {len(resources)} before dedup, {len(sdk_resources)} SDK + "
        f"{len(resources) - len(sdk_resources)} raw)"
    )
    return deduplicated


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
                if "{{secrets/" in var_value:
                    logger.info(
                        f"Skipping environment variable {var_name} - contains Model "
                        f"Serving secret reference that is not supported in Databricks Apps"
                    )
                    continue
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

    # Add SQL warehouse ID for UC trace location if configured
    if config.app and config.app.trace_location:
        env_vars.append(
            {
                "name": "MLFLOW_TRACING_SQL_WAREHOUSE_ID",
                "value": config.app.trace_location.warehouse_id,
            }
        )

    # Add chat proxy env vars when enabled so the AgentServer can proxy
    # static asset requests to the frontend running on a separate port.
    enable_chat_proxy: bool = (
        config.app.enable_chat_proxy
        if config.app and config.app.enable_chat_proxy is not None
        else True
    )
    if enable_chat_proxy:
        from dao_ai.apps.chat_ui import chat_ui_env_vars

        env_vars.extend(chat_ui_env_vars())

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
