"""Service-principal lifecycle helpers for the ``dao-ai service-principal`` CLI.

Three operations, all workspace-level (no AccountClient needed):

* :func:`create` — create (or reuse) a workspace service principal and mint an
  OAuth secret. Returns the ``application_id`` (client id) + the one-time secret.
* :func:`store` — write client id / secret into a Databricks secret scope.
* :func:`grant` — walk an :class:`~dao_ai.config.AppConfig` and grant the service
  principal the read/execute privileges an agent runtime needs on every declared
  resource (catalog, schema, table, function, vector index, volume, connection,
  warehouse, genie room, experiment, serving endpoint).

The grant path reuses the same idempotent, warn-and-continue Unity Catalog
permissions REST call dao-ai already uses at deploy time
(``PATCH /api/2.1/unity-catalog/permissions/{securable_type}/{full_name}``).

Lakebase autoscaling projects are a separate plane: SP access there is a Postgres
role (created via the Postgres API), not a UC grant. :func:`grant` delegates to
:meth:`~dao_ai.providers.databricks.DatabricksProvider.create_lakebase_autoscaling_role`,
but only when the SP being granted matches the ``DatabaseModel``'s ``client_id`` —
otherwise the role would belong to a different identity than the one the agent
connects as, so the step is reported and skipped.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

from loguru import logger

from dao_ai.config import value_of

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

    from dao_ai.config import (
        AiSearchVectorStoreModel,
        AppConfig,
        ConnectionModel,
        DatabaseModel,
        FunctionModel,
        GenieRoomModel,
        SchemaModel,
        ServicePrincipalModel,
        TableModel,
        VolumeModel,
        WarehouseModel,
    )


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _looks_like_uuid(value: str) -> bool:
    """Return True if ``value`` is a UUID (a service-principal application id)."""
    return bool(_UUID_RE.match(value.strip()))


# =============================================================================
# create
# =============================================================================


@dataclass
class CreatedServicePrincipal:
    """Result of :func:`create`. ``client_secret`` is shown only once."""

    display_name: str
    client_id: str  # application_id (UUID) — the grantee principal
    sp_id: str  # numeric id, used to mint OAuth secrets
    client_secret: Optional[str] = None  # None when an existing SP was reused
    reused: bool = False


def create(
    w: "WorkspaceClient",
    *,
    display_name: str,
    lifetime: Optional[str] = None,
) -> CreatedServicePrincipal:
    """Create (or reuse) a workspace service principal and mint an OAuth secret.

    Idempotent on ``display_name``: if a service principal with the same display
    name already exists it is reused (and a fresh secret is still minted, so the
    caller always gets usable credentials).

    Args:
        w: Workspace client (profile already applied by the caller).
        display_name: Display name for the service principal.
        lifetime: Optional OAuth secret lifetime (e.g. ``"7776000s"``). Defaults
            to the workspace maximum when omitted.

    Returns:
        The created/reused principal plus the one-time client secret.
    """
    existing = next(
        (
            sp
            for sp in w.service_principals.list(
                filter=f'displayName eq "{display_name}"'
            )
            if sp.display_name == display_name
        ),
        None,
    )

    if existing is not None:
        logger.info(
            "Reusing existing service principal",
            display_name=display_name,
            application_id=existing.application_id,
        )
        sp = existing
        reused = True
    else:
        sp = w.service_principals.create(display_name=display_name, active=True)
        logger.info(
            "Created service principal",
            display_name=display_name,
            application_id=sp.application_id,
        )
        reused = False

    secret_resp = w.service_principal_secrets_proxy.create(
        service_principal_id=str(sp.id),
        **({"lifetime": lifetime} if lifetime else {}),
    )

    return CreatedServicePrincipal(
        display_name=display_name,
        client_id=sp.application_id,
        sp_id=str(sp.id),
        client_secret=secret_resp.secret,
        reused=reused,
    )


# =============================================================================
# store
# =============================================================================


def store(
    w: "WorkspaceClient",
    *,
    scope: str,
    client_id_key: str,
    client_secret_key: str,
    client_id: str,
    client_secret: str,
) -> None:
    """Write the service-principal credentials into a Databricks secret scope.

    Creates the scope if it does not already exist (idempotent).
    """
    _ensure_scope(w, scope)
    w.secrets.put_secret(scope=scope, key=client_id_key, string_value=client_id)
    w.secrets.put_secret(scope=scope, key=client_secret_key, string_value=client_secret)
    logger.info(
        "Stored service-principal credentials",
        scope=scope,
        client_id_key=client_id_key,
        client_secret_key=client_secret_key,
    )


def _ensure_scope(w: "WorkspaceClient", scope: str) -> None:
    """Create a secret scope, ignoring the error if it already exists."""
    try:
        w.secrets.create_scope(scope=scope)
        logger.info("Created secret scope", scope=scope)
    except Exception as e:  # noqa: BLE001 — SDK raises a generic error on dup
        if "RESOURCE_ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
            logger.debug("Secret scope already exists", scope=scope)
        else:
            raise


# =============================================================================
# provision — one-shot create + store + grant
# =============================================================================


@dataclass
class ProvisionResult:
    """Outcome of :func:`provision`. The secret is deliberately NOT included."""

    display_name: str
    client_id: str
    reused: bool
    stored_scope: Optional[str] = None
    stored_client_id_key: Optional[str] = None
    stored_client_secret_key: Optional[str] = None
    stored: bool = False
    grant_plan: Optional["GrantPlan"] = None


def provision(
    w: "WorkspaceClient",
    *,
    config: "AppConfig",
    display_name: str,
    scope: Optional[str] = None,
    client_id_key: Optional[str] = None,
    client_secret_key: Optional[str] = None,
    lifetime: Optional[str] = None,
    do_store: bool = True,
    do_grant: bool = True,
) -> ProvisionResult:
    """Create an SP, store its secret, and grant it the config's resources — one shot.

    The freshly-minted client secret is written straight to the secret scope and is
    never returned or printed. This is the recommended path to make a config's
    declared service principal usable end-to-end.

    Args:
        w: Workspace client (profile already applied by the caller).
        config: The AppConfig being provisioned for.
        display_name: Service-principal display name.
        scope / client_id_key / client_secret_key: Secret target. Resolved from the
            config's service_principals block when omitted; scope falls back to a
            name derived from the config (see :func:`default_scope_from_config`).
        lifetime: Optional OAuth secret lifetime.
        do_store: Write the credentials to the secret scope (default True).
        do_grant: Grant the SP the config's resources (default True).

    Raises:
        ValueError: if ``do_store`` is set but the secret scope/keys cannot be
            resolved. Validated BEFORE creating the service principal so a
            misconfigured call never leaves an orphaned SP behind.
    """
    # Resolve + validate the store target up front — before we create anything —
    # so an unresolvable config fails fast without orphaning a service principal.
    resolved_scope: Optional[str] = None
    cid_key: Optional[str] = None
    csec_key: Optional[str] = None
    if do_store:
        resolved_scope, cid_key, csec_key = resolve_secret_target(
            config,
            scope_override=scope,
            client_id_key_override=client_id_key,
            client_secret_key_override=client_secret_key,
        )
        resolved_scope = resolved_scope or default_scope_from_config(config)
        if not resolved_scope:
            raise ValueError(
                "Cannot determine a secret scope to store credentials. "
                "Pass --scope, or add a service_principals block to the config."
            )
        if not (cid_key and csec_key):
            raise ValueError(
                "Cannot determine which secret keys to store the credentials under. "
                "The config has no service_principals block or client_id/client_secret "
                "variables to infer them from. Pass --client-id-key and "
                "--client-secret-key (the keys your config reads its credentials from)."
            )

    created = create(w, display_name=display_name, lifetime=lifetime)

    result = ProvisionResult(
        display_name=created.display_name,
        client_id=created.client_id,
        reused=created.reused,
    )

    if do_store:
        assert resolved_scope and cid_key and csec_key  # validated above
        store(
            w,
            scope=resolved_scope,
            client_id_key=cid_key,
            client_secret_key=csec_key,
            client_id=created.client_id,
            client_secret=created.client_secret,
        )
        result.stored_scope = resolved_scope
        result.stored_client_id_key = cid_key
        result.stored_client_secret_key = csec_key
        result.stored = True

    if do_grant:
        result.grant_plan = grant(w, principal=created.client_id, config=config)

    return result


# =============================================================================
# grant
# =============================================================================


@dataclass
class Grant:
    """A single intended permission grant (used for dry-run reporting)."""

    kind: str  # "uc" | "warehouse" | "genie" | "experiment" | "serving_endpoint"
    #          | "lakebase_role"
    target: str  # full name / id
    privileges: Sequence[str]
    securable_type: Optional[str] = None  # for kind == "uc"
    # Human-readable context surfaced in the plan (dry-run and apply). Used by
    # the ``lakebase_role`` kind to explain an intentional skip (e.g. the granted
    # SP does not match the DatabaseModel's ``client_id``). When set on a
    # ``lakebase_role`` grant, the Postgres role is NOT created.
    note: Optional[str] = None
    # Set during apply (not dry-run): True if applied, False if it errored,
    # None if not attempted (dry-run).
    applied: Optional[bool] = None
    error: Optional[str] = None


@dataclass
class GrantPlan:
    """The full set of grants a :func:`grant` call will (or did) apply."""

    principal: str
    grants: list[Grant] = field(default_factory=list)


def build_grant_plan(config: "AppConfig", principal: str) -> GrantPlan:
    """Walk an AppConfig and compute the read/execute grants for ``principal``.

    Pure (no side effects) so it can back both ``--dry-run`` and the real apply.
    De-dupes catalogs and schemas across every resource that references them.
    """
    plan = GrantPlan(principal=principal)
    catalogs: set[str] = set()
    schemas: set[str] = set()

    def _add_schema(catalog_name: str, schema_name: str) -> None:
        if catalog_name and catalog_name not in catalogs:
            catalogs.add(catalog_name)
            plan.grants.append(Grant("uc", catalog_name, ["USE_CATALOG"], "catalog"))
        full = f"{catalog_name}.{schema_name}"
        if catalog_name and schema_name and full not in schemas:
            schemas.add(full)
            plan.grants.append(
                Grant("uc", full, ["USE_SCHEMA", "SELECT", "EXECUTE"], "schema")
            )

    # Top-level schemas
    schema: "SchemaModel"
    for schema in config.schemas.values():
        _add_schema(schema.catalog_name, schema.schema_name)

    resources = config.resources
    if resources is not None:
        # Tables → SELECT (+ ensure their schema is granted)
        table: "TableModel"
        for table in resources.tables.values():
            if table.schema_model is not None:
                _add_schema(
                    table.schema_model.catalog_name, table.schema_model.schema_name
                )
            if table.full_name and table.full_name.count(".") == 2:
                plan.grants.append(Grant("uc", table.full_name, ["SELECT"], "table"))

        # UC functions → EXECUTE
        func: "FunctionModel"
        for func in resources.functions.values():
            if func.schema_model is not None:
                _add_schema(
                    func.schema_model.catalog_name, func.schema_model.schema_name
                )
            if func.full_name and func.full_name.count(".") == 2:
                plan.grants.append(Grant("uc", func.full_name, ["EXECUTE"], "function"))

        # Vector-search indexes → SELECT on the backing UC index (a table securable)
        store_model: "AiSearchVectorStoreModel"
        for store_model in resources.vector_stores.values():
            index = store_model.index
            index_name = value_of(index.full_name) if index is not None else None
            if index_name and str(index_name).count(".") == 2:
                plan.grants.append(Grant("uc", str(index_name), ["SELECT"], "table"))

        # Volumes → READ_VOLUME (+ ensure their schema is granted)
        volume: "VolumeModel"
        for volume in resources.volumes.values():
            if volume.schema_model is not None:
                _add_schema(
                    volume.schema_model.catalog_name, volume.schema_model.schema_name
                )
            if volume.full_name and volume.full_name.count(".") == 2:
                plan.grants.append(
                    Grant("uc", volume.full_name, ["READ_VOLUME"], "volume")
                )

        # Connections → USE_CONNECTION (connection names are top-level, unqualified)
        connection: "ConnectionModel"
        for connection in resources.connections.values():
            if connection.full_name:
                plan.grants.append(
                    Grant(
                        "uc", connection.full_name, ["USE_CONNECTION"], "connection"
                    )
                )

        # Warehouses → CAN_USE (workspace permission, not UC)
        warehouse: "WarehouseModel"
        for warehouse in resources.warehouses.values():
            wid = value_of(warehouse.warehouse_id) if warehouse.warehouse_id else None
            if wid:
                plan.grants.append(Grant("warehouse", str(wid), ["CAN_USE"]))

        # Genie rooms → CAN_RUN (workspace permission)
        room: "GenieRoomModel"
        for room in resources.genie_rooms.values():
            space_id = value_of(room.space_id) if room.space_id else None
            if space_id:
                plan.grants.append(Grant("genie", str(space_id), ["CAN_RUN"]))

        # Lakebase autoscaling projects → Postgres SUPERUSER role (created via the
        # Postgres API, NOT a UC PATCH — see DatabricksProvider.create_lakebase_
        # autoscaling_role). The Postgres role is keyed on the DatabaseModel's own
        # ``client_id``, so we only create it when the SP being granted matches;
        # otherwise the deployed agent would connect to Postgres as one identity
        # while the role belongs to another (silent runtime auth failure). Mismatch
        # / unresolved cases are planned with a ``note`` and skipped at apply time.
        database: "DatabaseModel"
        for database in resources.databases.values():
            if not database.is_lakebase or database.on_behalf_of_user:
                continue
            configured = value_of(database.client_id) if database.client_id else None
            project = str(database.project) if database.project else "<lakebase>"
            note: Optional[str] = None
            if not configured:
                note = (
                    "SKIP: DatabaseModel.client_id is unset or resolved to None "
                    "(secret scope populated?). A Postgres role can only be created "
                    "for a concrete service-principal client id — provision the SP "
                    "and populate the scope, then re-run."
                )
            elif configured != principal:
                note = (
                    f"SKIP: granting SP '{principal}' but this Lakebase project is "
                    f"configured for client_id '{configured}'. The Postgres role is "
                    f"created for the configured id, so '{principal}' would fail at "
                    f"connect time. Grant the configured SP (--app-sp) or align "
                    f"DatabaseModel.client_id."
                )
            plan.grants.append(
                Grant(
                    "lakebase_role",
                    project,
                    ["DATABRICKS_SUPERUSER"],
                    note=note,
                )
            )

    # Experiment + serving endpoint (only if declared on the app)
    app = config.app
    if app is not None:
        if app.experiment is not None and app.experiment.name:
            plan.grants.append(
                Grant("experiment", str(value_of(app.experiment.name)), ["CAN_EDIT"])
            )
        # AppModel always populates endpoint_name (defaulting from app.name), so
        # this grant is planned for every app. It's best-effort: _grant_serving_endpoint
        # resolves the endpoint by name and skips (no-op) if it isn't deployed, so
        # Apps-only configs don't error — they simply have nothing to grant here.
        if app.endpoint_name:
            plan.grants.append(
                Grant("serving_endpoint", app.endpoint_name, ["CAN_QUERY"])
            )

    return plan


def grant(
    w: "WorkspaceClient",
    *,
    principal: str,
    config: "AppConfig",
    dry_run: bool = False,
) -> GrantPlan:
    """Grant ``principal`` read/execute access to every resource in ``config``.

    Returns the :class:`GrantPlan`. When ``dry_run`` is True nothing is applied.
    Individual failures warn-and-continue (consistent with deploy-time granting).
    """
    plan = build_grant_plan(config, principal)

    if dry_run:
        return plan

    for g in plan.grants:
        try:
            if g.kind == "uc":
                _grant_uc(w, principal, g.securable_type, g.target, g.privileges)
            elif g.kind == "warehouse":
                _grant_warehouse(w, principal, g.target)
            elif g.kind == "genie":
                _grant_genie(w, principal, g.target)
            elif g.kind == "experiment":
                _grant_experiment(w, principal, g.target)
            elif g.kind == "serving_endpoint":
                _grant_serving_endpoint(w, principal, g.target)
            elif g.kind == "lakebase_role":
                if g.note:
                    # Intentional skip (identity mismatch or unresolved
                    # client_id) — surface the reason and leave ``applied`` None
                    # (not attempted) so it never reads as a success.
                    logger.warning(
                        "Lakebase Postgres role not created",
                        project=g.target,
                        reason=g.note,
                    )
                    print(f"  ⚠ Lakebase '{g.target}': {g.note}", file=sys.stderr)
                    continue
                _grant_lakebase_role(w, config, g.target)
            g.applied = True
        except Exception as e:  # noqa: BLE001 — warn-and-continue per resource
            g.applied = False
            g.error = str(e)
            logger.warning(
                "Grant failed — verify the calling identity has GRANT rights",
                kind=g.kind,
                target=g.target,
                error=str(e),
            )

    return plan


def _grant_lakebase_role(
    w: "WorkspaceClient", config: "AppConfig", project: str
) -> None:
    """Create the Postgres SUPERUSER role for a Lakebase project's service principal.

    Delegates to the existing, idempotent
    :meth:`DatabricksProvider.create_lakebase_autoscaling_role` rather than
    reinventing the Postgres-API role logic. The provider method reads the
    ``DatabaseModel``'s own ``client_id`` and connects with its configured
    credentials, so the caller has already verified (in :func:`build_grant_plan`)
    that the SP being granted matches that ``client_id``.
    """
    from dao_ai.providers.databricks import DatabricksProvider

    databases = config.resources.databases if config.resources else {}
    database = next(
        (db for db in databases.values() if str(db.project) == project), None
    )
    if database is None:
        raise ValueError(
            f"No Lakebase DatabaseModel with project '{project}' found in config"
        )
    DatabricksProvider(w=w).create_lakebase_autoscaling_role(database)


def _grant_uc(
    w: "WorkspaceClient",
    principal: str,
    securable_type: str,
    full_name: str,
    privileges: Sequence[str],
) -> None:
    """Grant UC privileges via the raw REST permissions endpoint (idempotent).

    Mirrors ``_grant_uc_trace_table_permissions_to_principal`` in
    ``providers/databricks.py`` — lowercase securable type works across SDK
    versions where the typed ``grants.update`` serializes the enum incorrectly.
    """
    w.api_client.do(
        "PATCH",
        f"/api/2.1/unity-catalog/permissions/{securable_type}/{full_name}",
        body={"changes": [{"principal": principal, "add": list(privileges)}]},
    )
    logger.info(
        "Granted UC privileges",
        principal=principal,
        securable_type=securable_type,
        full_name=full_name,
        privileges=list(privileges),
    )


def _grant_warehouse(w: "WorkspaceClient", principal: str, warehouse_id: str) -> None:
    """Grant CAN_USE on a SQL warehouse to the service principal.

    Uses ``update_permissions`` (additive) — NOT ``set_permissions``, which
    replaces the entire ACL and would strip every other principal's access.
    """
    from databricks.sdk.service.sql import (
        WarehouseAccessControlRequest,
        WarehousePermissionLevel,
    )

    w.warehouses.update_permissions(
        warehouse_id=warehouse_id,
        access_control_list=[
            WarehouseAccessControlRequest(
                service_principal_name=principal,
                permission_level=WarehousePermissionLevel.CAN_USE,
            )
        ],
    )
    logger.info(
        "Granted warehouse CAN_USE", principal=principal, warehouse_id=warehouse_id
    )


def _grant_genie(w: "WorkspaceClient", principal: str, space_id: str) -> None:
    """Grant CAN_RUN on a Genie space to the service principal.

    Uses ``permissions.update`` (additive), not ``permissions.set`` (which
    replaces the whole ACL).
    """
    from databricks.sdk.service.iam import (
        AccessControlRequest,
        PermissionLevel,
    )

    kwargs = {"permission_level": PermissionLevel.CAN_RUN}
    if _looks_like_uuid(principal):
        kwargs["service_principal_name"] = principal
    elif "@" in principal:
        kwargs["user_name"] = principal
    else:
        kwargs["group_name"] = principal

    w.permissions.update(
        request_object_type="genie",
        request_object_id=space_id,
        access_control_list=[AccessControlRequest(**kwargs)],
    )
    logger.info("Granted genie CAN_RUN", principal=principal, space_id=space_id)


def _grant_experiment(
    w: "WorkspaceClient", principal: str, experiment_name: str
) -> None:
    """Grant CAN_EDIT on an MLflow experiment (reuses the provider helper)."""
    from dao_ai.providers.databricks import (
        _grant_experiment_permissions_to_principal,
    )

    experiment = w.experiments.get_by_name(experiment_name)
    exp_id = experiment.experiment.experiment_id if experiment.experiment else None
    if exp_id:
        _grant_experiment_permissions_to_principal(principal, exp_id)


def _grant_serving_endpoint(
    w: "WorkspaceClient", principal: str, endpoint_name: str
) -> None:
    """Grant CAN_QUERY on a Model Serving endpoint (best-effort; skip if absent).

    Uses ``update_permissions`` (additive), and resolves the endpoint's id from
    its name (``set/update_permissions`` key on the id, not the name).
    """
    from databricks.sdk.service.serving import (
        ServingEndpointAccessControlRequest,
        ServingEndpointPermissionLevel,
    )

    try:
        endpoint = w.serving_endpoints.get(name=endpoint_name)
    except Exception:  # noqa: BLE001 — endpoint not deployed yet; skip quietly
        logger.debug(
            "Serving endpoint not found; skipping grant", endpoint=endpoint_name
        )
        return

    endpoint_id = endpoint.id or endpoint_name
    w.serving_endpoints.update_permissions(
        serving_endpoint_id=endpoint_id,
        access_control_list=[
            ServingEndpointAccessControlRequest(
                service_principal_name=principal,
                permission_level=ServingEndpointPermissionLevel.CAN_QUERY,
            )
        ],
    )
    logger.info(
        "Granted serving endpoint CAN_QUERY",
        principal=principal,
        endpoint=endpoint_name,
    )


# =============================================================================
# config extraction helpers
# =============================================================================


def resolve_principal_from_config(
    config: "AppConfig", override: Optional[str] = None
) -> Optional[str]:
    """Resolve the grantee client id: explicit override, else config service principal."""
    if override:
        return override
    sp: "ServicePrincipalModel"
    for sp in config.service_principals.values():
        if sp.client_id is not None:
            client_id = value_of(sp.client_id)
            if client_id:
                return str(client_id)
    return None


def resolve_secret_target(
    config: "AppConfig",
    *,
    scope_override: Optional[str] = None,
    client_id_key_override: Optional[str] = None,
    client_secret_key_override: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve (scope, client_id_key, client_secret_key) for ``store``.

    Prefers explicit overrides, then discovers the secret scope + key names the
    config actually reads its credentials from, checking two *structural* sources
    (where the credential's role is unambiguous) in order:

    1. ``service_principals`` block — its ``client_id`` / ``client_secret`` vars.
    2. top-level ``variables`` named ``client_id`` / ``client_secret``.

    We deliberately do NOT try to infer keys from ``app.environment_vars`` by
    string-matching names like ``*_CLIENT_ID`` — that's a guess, not a fact, and a
    wrong guess would store the secret under keys the agent never reads. When
    neither structural source resolves a key, ``None`` is returned for it and the
    caller must supply ``--client-id-key`` / ``--client-secret-key`` / ``--scope``.

    Returns ``None`` for any component that could not be resolved (no fallbacks).
    """
    scope = scope_override
    client_id_key = client_id_key_override
    client_secret_key = client_secret_key_override

    def _merge(cid_ref: object, csec_ref: object) -> None:
        nonlocal scope, client_id_key, client_secret_key
        cid_scope, cid_key = _secret_ref(cid_ref)
        csec_scope, csec_key = _secret_ref(csec_ref)
        scope = scope or cid_scope or csec_scope
        client_id_key = client_id_key or cid_key
        client_secret_key = client_secret_key or csec_key

    def _done() -> bool:
        return bool(scope and client_id_key and client_secret_key)

    # 1. service_principals block (structural — role known by binding)
    if not _done():
        sp: "ServicePrincipalModel"
        for sp in config.service_principals.values():
            _merge(sp.client_id, sp.client_secret)
            if _done():
                break

    # 2. top-level variables named client_id / client_secret (structural — role known by name)
    if not _done():
        _merge(config.variables.get("client_id"), config.variables.get("client_secret"))

    return scope, client_id_key, client_secret_key


def _secret_ref(value: object) -> tuple[Optional[str], Optional[str]]:
    """Extract (scope, key) from a secret-backed variable, if it is one.

    ``value`` is an ``AnyVariable`` union: a literal, a ``SecretVariableModel``
    (has ``scope`` + ``secret``), or a ``CompositeVariableModel`` (has ``options``
    listing candidate resolutions). Narrow with isinstance rather than duck-typing.
    """
    from dao_ai.config import CompositeVariableModel, SecretVariableModel

    if isinstance(value, SecretVariableModel):
        return value.scope, value.secret
    if isinstance(value, CompositeVariableModel):
        for option in value.options or []:
            if isinstance(option, SecretVariableModel):
                return option.scope, option.secret
    return None, None


def default_scope_from_config(config: "AppConfig") -> Optional[str]:
    """Derive a fallback secret scope when the config has no service_principals block.

    Prefers the app name, then the first schema's catalog — so ``provision`` works
    on configs that never declared a service principal.
    """
    if config.app is not None and config.app.name:
        return config.app.name
    for schema in config.schemas.values():
        if schema.catalog_name:
            return str(schema.catalog_name)
    return None
