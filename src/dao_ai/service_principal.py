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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Optional, Sequence

from loguru import logger

from dao_ai.config import value_of

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.iam import ServicePrincipal

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


# Why a grant did not apply. ``absent`` means the target isn't in this
# workspace at all (a config/workspace mismatch — usually a wrong ``--var``
# override or profile); ``denied`` means it exists but the calling identity
# lacks GRANT/MANAGE on it. The distinction matters because the remedies are
# completely different, and reporting every failure as a permissions problem
# sends users chasing ACLs for a resource that was never there.
GRANT_FAILURE_ABSENT: Final[str] = "absent"
GRANT_FAILURE_DENIED: Final[str] = "denied"
GRANT_FAILURE_ERROR: Final[str] = "error"


def classify_grant_error(exc: BaseException) -> str:
    """Bucket a grant failure into ``absent`` / ``denied`` / ``error``.

    Prefers the SDK's typed exceptions, which are exact: a UC permissions PATCH
    against a missing catalog raises :class:`NotFound` with
    ``error_code='CATALOG_DOES_NOT_EXIST'``. Falls back to substring matching
    only for non-SDK errors raised by provider helpers.
    """
    from databricks.sdk.errors import DatabricksError, NotFound, PermissionDenied

    if isinstance(exc, NotFound):  # ResourceDoesNotExist subclasses NotFound
        return GRANT_FAILURE_ABSENT
    if isinstance(exc, PermissionDenied):
        return GRANT_FAILURE_DENIED

    error_code: str = ""
    if isinstance(exc, DatabricksError):
        error_code = str(getattr(exc, "error_code", "") or "")
    if error_code.endswith("_DOES_NOT_EXIST") or error_code == "NOT_FOUND":
        return GRANT_FAILURE_ABSENT
    if error_code == "PERMISSION_DENIED":
        return GRANT_FAILURE_DENIED

    text: str = str(exc).lower()
    if "does not exist" in text or "not found" in text:
        return GRANT_FAILURE_ABSENT
    if "permission_denied" in text or "not authorized" in text:
        return GRANT_FAILURE_DENIED
    return GRANT_FAILURE_ERROR


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


def find_service_principal(
    w: "WorkspaceClient", *, display_name: str
) -> Optional["ServicePrincipal"]:
    """Look up a workspace service principal by display name.

    Read-only — mints nothing and creates nothing. Split out of :func:`create`
    so the provisioning flow can report "would create" vs "would reuse (id …)"
    during a dry run, and decide whether a secret needs minting at all, without
    any side effect.

    Returns:
        The matching service principal, or None when no SP has that name.
    """
    return next(
        (
            sp
            for sp in w.service_principals.list(
                filter=f'displayName eq "{display_name}"'
            )
            if sp.display_name == display_name
        ),
        None,
    )


def create(
    w: "WorkspaceClient",
    *,
    display_name: str,
    lifetime: Optional[str] = None,
    dry_run: bool = False,
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
        dry_run: Report whether the SP exists without creating it or minting a
            secret. Note that minting IS a mutation — it registers a new OAuth
            secret on the principal — so a dry run must not reach it. Returns
            ``client_secret=None`` and an empty ``client_id`` when the SP does
            not exist yet.

    Returns:
        The created/reused principal plus the one-time client secret.
    """
    existing = find_service_principal(w, display_name=display_name)

    if dry_run:
        return CreatedServicePrincipal(
            display_name=display_name,
            client_id=(existing.application_id or "") if existing else "",
            sp_id=str(existing.id) if existing else "",
            client_secret=None,
            reused=existing is not None,
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
    overwrite: bool = True,
    dry_run: bool = False,
) -> "StoreResult":
    """Write the service-principal credentials into a Databricks secret scope.

    Creates the scope if it does not already exist (idempotent).

    Args:
        overwrite: When False, a key that already holds a value is left alone and
            reported in :attr:`StoreResult.skipped`. Defaults to True, preserving
            the original unconditional-write behaviour for existing callers.
        dry_run: Report what would be written without writing anything.

    Returns:
        Which keys were (or would be) written, and which were left untouched.
    """
    existing = (
        secret_keys_present(w, scope) if (not overwrite or dry_run) else frozenset()
    )
    pairs = ((client_id_key, client_id), (client_secret_key, client_secret))
    written: list[str] = []
    skipped: list[str] = []
    for key, secret_value in pairs:
        if not overwrite and key in existing:
            skipped.append(key)
            continue
        written.append(key)
        if not dry_run:
            if len(written) == 1:
                # Only touch the scope once we know we have something to write.
                _ensure_scope(w, scope)
            w.secrets.put_secret(scope=scope, key=key, string_value=secret_value)

    if not dry_run and written:
        logger.info(
            "Stored service-principal credentials",
            scope=scope,
            keys=written,
            skipped=skipped or None,
        )
    return StoreResult(
        scope=scope,
        written=written,
        skipped=skipped,
        scope_existed=scope in secret_scopes(w) if dry_run else None,
    )


@dataclass
class StoreResult:
    """Which secret keys :func:`store` wrote, and which it left alone."""

    scope: str
    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    # Only populated on a dry run, where "the scope will be created" is worth
    # reporting; None on a real run (the scope is ensured as a side effect).
    scope_existed: Optional[bool] = None


def secret_keys_present(w: "WorkspaceClient", scope: str) -> frozenset[str]:
    """Keys that already hold a value in ``scope`` (empty if the scope is absent).

    Uses ``list_secrets``, which returns key names and timestamps only — never a
    value. The SDK has no exists-check helper, and ``get_secret`` would
    materialize the credential just to learn whether it is there.
    """
    from databricks.sdk.errors import NotFound

    try:
        return frozenset(
            s.key for s in w.secrets.list_secrets(scope=scope) if s.key is not None
        )
    except NotFound:
        return frozenset()


def secret_scopes(w: "WorkspaceClient") -> frozenset[str]:
    """Names of every secret scope in the workspace."""
    return frozenset(s.name for s in w.secrets.list_scopes() if s.name is not None)


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
# ownership — which declared service principal owns which resource
# =============================================================================


@dataclass(frozen=True)
class SecretRef:
    """A ``(scope, key)`` pointer into a Databricks secret scope."""

    scope: str
    key: str


@dataclass
class ServicePrincipalTarget:
    """One service principal to provision/grant, plus the keys identifying it.

    ``client_id_ref`` is the *primary* identity key rather than
    ``configured_client_id``: a config's ``client_id`` variable resolves to
    whatever is currently in the secret scope, which on a first provision is
    nothing and after a rotation is the *previous* SP's id. The ``(scope, key)``
    pair is a property of the config itself, so it is stable across both.
    """

    name: str
    display_name: str
    model: Optional["ServicePrincipalModel"] = None
    scope: Optional[str] = None
    client_id_key: Optional[str] = None
    client_secret_key: Optional[str] = None
    client_id_ref: Optional[SecretRef] = None
    client_secret_ref: Optional[SecretRef] = None
    # Value of the config's client_id variable at load time. May be stale (a
    # previous SP) or None (scope not populated yet) — never authoritative.
    configured_client_id: Optional[str] = None
    # The real client id, filled in once the SP is looked up or created.
    resolved_client_id: Optional[str] = None
    # Whether a workspace SP with ``display_name`` already exists (set by the
    # dry-run-safe probe; None until probed).
    exists: Optional[bool] = None


@dataclass
class OwnershipMap:
    """Which named service principal owns which declared resource.

    Keys are ``(collection, config_key)`` — e.g. ``("databases",
    "retail_database")``. Plain string tuples, deliberately: the variable models
    a resource's ``client_id`` holds are unhashable (``CompositeVariableModel``
    wraps a list), so they cannot be used as keys.

    A resource absent from :attr:`owners` is **shared** — no declared SP claims
    it, so every SP is granted it.
    """

    owners: dict[tuple[str, str], str] = field(default_factory=dict)

    def owner_of(self, collection: str, key: str) -> Optional[str]:
        """Name of the SP owning ``collection[key]``, or None when shared."""
        return self.owners.get((collection, key))

    def owns(self, collection: str, key: str, sp_name: Optional[str]) -> bool:
        """True if ``sp_name`` should be granted ``collection[key]``.

        Shared resources (no owner) return True for every SP.
        """
        owner = self.owners.get((collection, key))
        return owner is None or owner == sp_name


# The ``IsDatabricksResource`` collections on ``ResourcesModel`` that can carry
# per-resource credentials, and so can be owned by a specific SP. ``schemas``
# is intentionally absent — ``SchemaModel`` is not an ``IsDatabricksResource``
# and catalogs/schemas are shared infrastructure every SP needs USE_CATALOG on.
_OWNABLE_COLLECTIONS: Final[tuple[str, ...]] = (
    "tables",
    "functions",
    "volumes",
    "connections",
    "vector_stores",
    "warehouses",
    "genie_rooms",
    "databases",
)


def placeholder_principal(sp_name: str) -> str:
    """A stand-in client id for an SP that does not exist yet (dry run only).

    A dry run reports what *would* happen, so it must not create the service
    principal — which means its client id is unknown. Ownership is resolved from
    the config's secret refs rather than from ids (see :func:`resource_owner`),
    so the plan is still correct; only the displayed principal is a stand-in.
    """
    return f"<new-sp:{sp_name}>"


def _is_placeholder_principal(principal: str) -> bool:
    """True if ``principal`` is a :func:`placeholder_principal` sentinel."""
    return principal.startswith("<new-sp:")


def resource_owner(
    resource: object, targets: Sequence[ServicePrincipalTarget]
) -> Optional[str]:
    """Return the name of the target owning ``resource``, or None if shared.

    Matching, cheapest and most exact first:

    0. ``resource.service_principal`` equals the target's model. Pydantic
       compares by value, so a shared YAML anchor matches even though each
       occurrence re-validates into a distinct object.
    1. The resource's ``client_id`` points at the same secret ``(scope, key)``
       as the target. **This is the load-bearing case** — it needs no API call
       and works before the SP exists, which is what makes single-pass
       provisioning of a brand-new SP possible.
    2. The resource's ``client_id`` resolves to the target's client id. Only
       reached when step 1 found no secret ref (a literal or env-var
       ``client_id``), because ``value_of`` on a secret-backed variable is a
       live, uncached ``secrets.get_secret`` call — running it per
       resource × target would multiply one API call into N×M.
    """
    model = getattr(resource, "service_principal", None)
    if model is not None:
        for target in targets:
            if target.model is not None and model == target.model:
                return target.name

    client_id = getattr(resource, "client_id", None)
    if client_id is None:
        return None

    scope, key = _secret_ref(client_id)
    if scope is not None and key is not None:
        for target in targets:
            ref = target.client_id_ref
            if ref is not None and ref.scope == scope and ref.key == key:
                return target.name
        # A secret-backed client_id that matches no target is genuinely
        # unowned; do NOT fall through to value_of() — that would read the
        # secret for nothing.
        return None

    resolved = value_of(client_id)
    if not resolved:
        return None
    for target in targets:
        if target.resolved_client_id and str(resolved) == target.resolved_client_id:
            return target.name
    for target in targets:
        if target.configured_client_id and str(resolved) == target.configured_client_id:
            return target.name
    return None


def build_ownership_map(
    config: "AppConfig", targets: Sequence[ServicePrincipalTarget]
) -> OwnershipMap:
    """Map each declared resource to the service principal that owns it.

    Only records matches — anything omitted is shared and granted to every SP.
    ``app.experiment`` / ``app.endpoint_name`` are deliberately never owned:
    ``AppModel.service_principal`` is the deploy/tracing identity, a different
    concept from "which SP does this resource authenticate as", and every SP
    should be able to query the endpoint.

    Raises:
        ValueError: if two targets share a secret ``(scope, client_id_key)``.
            They would overwrite each other's credential and ownership matching
            could not tell them apart. Raised before any workspace mutation.
    """
    seen_refs: dict[tuple[str, str], str] = {}
    for target in targets:
        ref = target.client_id_ref
        if ref is None:
            continue
        conflict = seen_refs.get((ref.scope, ref.key))
        if conflict is not None:
            raise ValueError(
                f"Service principals '{conflict}' and '{target.name}' both read "
                f"client_id from secret '{ref.scope}/{ref.key}'. Give each its "
                "own secret keys — otherwise they overwrite each other's "
                "credentials and their resources cannot be told apart."
            )
        seen_refs[(ref.scope, ref.key)] = target.name

    ownership = OwnershipMap()
    resources = config.resources
    if resources is None:
        return ownership

    for collection in _OWNABLE_COLLECTIONS:
        for key, resource in (getattr(resources, collection, None) or {}).items():
            owner = resource_owner(resource, targets)
            if owner is not None:
                ownership.owners[(collection, key)] = owner

    return ownership


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
    # The config dict key of the resource this grant came from. Set for the
    # ``lakebase_role`` kind so the apply step re-resolves the exact
    # ``DatabaseModel`` that passed the identity check in ``build_grant_plan``
    # (matching on ``project`` alone could pick a different model when two
    # DatabaseModels share a project but pin different ``client_id``s).
    resource_key: Optional[str] = None
    # Human-readable context surfaced in the plan (dry-run and apply). Used by
    # the ``lakebase_role`` kind to explain an intentional skip (e.g. the granted
    # SP does not match the DatabaseModel's ``client_id``). When set on a
    # ``lakebase_role`` grant, the Postgres role is NOT created.
    note: Optional[str] = None
    # Set during apply (not dry-run): True if applied, False if it errored,
    # None if not attempted (dry-run).
    applied: Optional[bool] = None
    error: Optional[str] = None
    # Why the grant failed, when ``applied is False`` — one of the
    # ``GRANT_FAILURE_*`` constants. Lets the report separate "this resource
    # isn't in the workspace" from "you lack GRANT on it", which need different
    # fixes. See :func:`classify_grant_error`.
    failure_kind: Optional[str] = None
    # For ``lakebase_role``: the client id to create the Postgres role for, when
    # it is known here but not yet readable from the config's secret scope (a
    # just-minted SP). Overrides ``value_of(DatabaseModel.client_id)``, which is
    # what makes single-pass provisioning possible.
    principal_override: Optional[str] = None


@dataclass
class GrantPlan:
    """The full set of grants a :func:`grant` call will (or did) apply."""

    principal: str
    grants: list[Grant] = field(default_factory=list)


def build_grant_plan(
    config: "AppConfig",
    principal: str,
    *,
    ownership: Optional[OwnershipMap] = None,
    sp_name: Optional[str] = None,
    targets: Optional[Sequence[ServicePrincipalTarget]] = None,
) -> GrantPlan:
    """Walk an AppConfig and compute the read/execute grants for ``principal``.

    Pure (no side effects) so it can back both ``--dry-run`` and the real apply.
    De-dupes catalogs and schemas across every resource that references them.

    Args:
        config: The config whose declared resources are walked.
        principal: Client id being granted. May be a ``<new-sp:NAME>`` sentinel
            during a dry run of a service principal that does not exist yet.
        ownership: When given, restrict the walk to the resources ``sp_name``
            owns plus every shared (unowned) resource. Omit it — the default —
            to grant the whole config to ``principal``, which is exactly the
            single-SP behaviour and keeps existing callers unchanged.
        sp_name: Which named service principal this plan is for. Only meaningful
            alongside ``ownership``.
        targets: All declared targets, used to resolve the Lakebase role's
            subject to the owning SP's client id.
    """
    plan = GrantPlan(principal=principal)
    catalogs: set[str] = set()
    schemas: set[str] = set()

    def _owns(collection: str, key: str) -> bool:
        """True when this plan should include ``collection[key]``."""
        return ownership is None or ownership.owns(collection, key, sp_name)

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
        for table_key, table in resources.tables.items():
            if not _owns("tables", table_key):
                continue
            if table.schema_model is not None:
                _add_schema(
                    table.schema_model.catalog_name, table.schema_model.schema_name
                )
            if table.full_name and table.full_name.count(".") == 2:
                plan.grants.append(Grant("uc", table.full_name, ["SELECT"], "table"))

        # UC functions → EXECUTE
        func: "FunctionModel"
        for func_key, func in resources.functions.items():
            if not _owns("functions", func_key):
                continue
            if func.schema_model is not None:
                _add_schema(
                    func.schema_model.catalog_name, func.schema_model.schema_name
                )
            if func.full_name and func.full_name.count(".") == 2:
                plan.grants.append(Grant("uc", func.full_name, ["EXECUTE"], "function"))

        # Vector-search indexes → SELECT on the backing UC index (a table securable)
        store_model: "AiSearchVectorStoreModel"
        for store_key, store_model in resources.vector_stores.items():
            if not _owns("vector_stores", store_key):
                continue
            index = store_model.index
            index_name = value_of(index.full_name) if index is not None else None
            if index_name and str(index_name).count(".") == 2:
                plan.grants.append(Grant("uc", str(index_name), ["SELECT"], "table"))

        # Volumes → READ_VOLUME (+ ensure their schema is granted)
        volume: "VolumeModel"
        for volume_key, volume in resources.volumes.items():
            if not _owns("volumes", volume_key):
                continue
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
        for conn_key, connection in resources.connections.items():
            if not _owns("connections", conn_key):
                continue
            if connection.full_name:
                plan.grants.append(
                    Grant("uc", connection.full_name, ["USE_CONNECTION"], "connection")
                )

        # Warehouses → CAN_USE (workspace permission, not UC)
        warehouse: "WarehouseModel"
        for wh_key, warehouse in resources.warehouses.items():
            if not _owns("warehouses", wh_key):
                continue
            wid = value_of(warehouse.warehouse_id) if warehouse.warehouse_id else None
            if wid:
                plan.grants.append(Grant("warehouse", str(wid), ["CAN_USE"]))

        # Genie rooms → CAN_RUN (workspace permission)
        room: "GenieRoomModel"
        for room_key, room in resources.genie_rooms.items():
            if not _owns("genie_rooms", room_key):
                continue
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
        db_key: str
        database: "DatabaseModel"
        for db_key, database in resources.databases.items():
            if not database.is_lakebase or database.on_behalf_of_user:
                continue
            # ``is_lakebase`` is defined as ``project is not None``, so project
            # is always set here.
            project = str(database.project)
            note: Optional[str] = None
            principal_override: Optional[str] = None

            if ownership is not None:
                owner = ownership.owner_of("databases", db_key)
                if owner is not None and owner != sp_name:
                    # Another declared SP owns this project. Emit nothing: with N
                    # service principals a per-SP mismatch note would print N-1
                    # alarming SKIPs per project per run, and the owner's own
                    # grant already tells the whole story.
                    continue
                # Owned by us, or shared. Either way the role subject is this
                # SP's real client id — which, for a freshly minted SP, is only
                # known here and NOT yet readable from the secret scope. Passing
                # it explicitly is what lets provisioning finish in one pass.
                if targets is not None:
                    principal_override = next(
                        (
                            t.resolved_client_id
                            for t in targets
                            if t.name == sp_name and t.resolved_client_id
                        ),
                        None,
                    )
                if principal_override is None and not _is_placeholder_principal(
                    principal
                ):
                    principal_override = principal
                if principal_override is None:
                    note = (
                        f"SKIP: no client id resolved yet for service principal "
                        f"'{sp_name}', so the Postgres role has no subject. This is "
                        f"expected in a dry run of an SP that does not exist yet."
                    )
            else:
                # Single-SP path (no ownership map): the role is keyed on the
                # DatabaseModel's own client_id, so only create it when the SP
                # being granted matches. Otherwise the agent would connect to
                # Postgres as one identity while the role belongs to another.
                configured = (
                    value_of(database.client_id) if database.client_id else None
                )
                if not configured:
                    note = (
                        "SKIP: DatabaseModel.client_id is unset or resolved to "
                        "None (secret scope populated?). A Postgres role can "
                        "only be created for a concrete service-principal "
                        "client id — provision the SP and populate the scope, "
                        "then re-run."
                    )
                elif configured != principal:
                    note = (
                        f"SKIP: granting SP '{principal}' but this Lakebase "
                        f"project is configured for client_id '{configured}'. "
                        f"The Postgres role is created for the configured id, "
                        f"so '{principal}' would fail at connect time. Grant "
                        f"the configured SP (sp grant --principal <client-id>) "
                        f"or align DatabaseModel.client_id."
                    )

            plan.grants.append(
                Grant(
                    "lakebase_role",
                    project,
                    ["DATABRICKS_SUPERUSER"],
                    resource_key=db_key,
                    note=note,
                    principal_override=principal_override,
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
    ownership: Optional[OwnershipMap] = None,
    sp_name: Optional[str] = None,
    targets: Optional[Sequence[ServicePrincipalTarget]] = None,
) -> GrantPlan:
    """Grant ``principal`` read/execute access to every resource in ``config``.

    Returns the :class:`GrantPlan`. When ``dry_run`` is True nothing is applied.
    Individual failures warn-and-continue (consistent with deploy-time granting).

    ``ownership`` / ``sp_name`` / ``targets`` restrict the grant to the resources
    one named service principal owns; see :func:`build_grant_plan`. Omitting them
    grants the whole config, which is the single-SP behaviour.
    """
    plan = build_grant_plan(
        config, principal, ownership=ownership, sp_name=sp_name, targets=targets
    )

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
                    # (not attempted) so it never reads as a success. The CLI
                    # layer (``_print_grants``) renders the note for the user.
                    logger.warning(
                        "Lakebase Postgres role not created",
                        project=g.target,
                        reason=g.note,
                    )
                    continue
                _grant_lakebase_role(
                    w, config, g.resource_key, client_id=g.principal_override
                )
            g.applied = True
        except Exception as e:  # noqa: BLE001 — warn-and-continue per resource
            g.applied = False
            g.error = str(e)
            g.failure_kind = classify_grant_error(e)
            # Say what actually went wrong. Blaming GRANT rights unconditionally
            # sends users auditing ACLs for a resource that isn't in the
            # workspace at all — usually a wrong ``--var`` override or profile.
            if g.failure_kind == GRANT_FAILURE_ABSENT:
                message = "Grant target does not exist in this workspace"
            elif g.failure_kind == GRANT_FAILURE_DENIED:
                message = "Grant denied — the calling identity lacks GRANT rights"
            else:
                message = "Grant failed"
            logger.warning(
                message,
                kind=g.kind,
                target=g.target,
                error=str(e),
            )

    return plan


def _grant_lakebase_role(
    w: "WorkspaceClient",
    config: "AppConfig",
    resource_key: Optional[str],
    *,
    client_id: Optional[str] = None,
) -> None:
    """Create the Postgres SUPERUSER role for a Lakebase project's service principal.

    Delegates to the existing, idempotent
    :meth:`DatabricksProvider.create_lakebase_autoscaling_role` rather than
    reinventing the Postgres-API role logic. The Postgres control-plane calls run
    as ``w`` — the caller's identity — since a service principal cannot create
    its own role.

    Re-resolves the model by its config dict key (not by ``project``) so we act
    on the exact ``DatabaseModel`` the plan selected — two models can share a
    ``project`` while pinning different ``client_id``s.

    Args:
        client_id: Role subject. Passed when the plan already knows the owning
            SP's client id (e.g. just minted, so not yet readable from the
            config's secret scope). Falls back to ``DatabaseModel.client_id``.
    """
    from dao_ai.providers.databricks import DatabricksProvider

    databases = config.resources.databases if config.resources else {}
    database = databases.get(resource_key) if resource_key is not None else None
    if database is None:
        raise ValueError(
            f"No Lakebase DatabaseModel with key '{resource_key}' found in config"
        )
    DatabricksProvider(w=w).create_lakebase_autoscaling_role(
        database, client_id=client_id
    )


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
