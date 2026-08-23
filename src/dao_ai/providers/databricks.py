import base64
import re
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Final, Optional, Sequence

import mlflow
import pandas as pd
import sqlparse
import yaml
from databricks import agents
from databricks.agents import PermissionLevel, set_permissions
from databricks.ai_search.client import VectorSearchClient
from databricks.ai_search.exceptions import BadRequest as AISearchBadRequest
from databricks.ai_search.index import VectorSearchIndex
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
    ConnectionType,
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
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import Experiment
from mlflow.entities.model_registry.model_version import ModelVersion
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
    ExperimentModel,
    FunctionModel,
    GenieEntitlement,
    GenieRoomModel,
    HasFullName,
    IndexModel,
    InferenceEndpointModel,
    IsDatabricksResource,
    SchemaModel,
    ServingMode,
    TableModel,
    UnityCatalogFunctionSqlModel,
    VectorStoreModel,
    VolumeModel,
    VolumePathModel,
    WarehouseModel,
    app_name_for,
    connection_name_for,
    mcp_service_name_for,
    resolve_connection_registration,
    value_of,
)
from dao_ai.models import get_latest_model_version
from dao_ai.providers.base import ServiceProvider
from dao_ai.utils import (
    dao_ai_version,
    find_dev_wheel,
    get_installed_packages,
    is_lib_provided,
    is_source_layout,
    normalize_host,
    normalize_name,
    resolve_use_local_source,
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


_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _app_can_use_acl_entry(principal: str) -> "AppAccessControlRequest":
    """Build a CAN_USE app-permission ACL entry for a principal, inferring its
    type: an email → user, a UUID → service principal (OAuth client id), anything
    else (e.g. ``account users``) → group. Used for U2M MCP connections, where the
    forwarded end users — not the app SP — must be authorized to invoke the app.
    """
    from databricks.sdk.service.apps import (
        AppAccessControlRequest,
        AppPermissionLevel,
    )

    p = principal.strip()
    lvl = AppPermissionLevel.CAN_USE
    if "@" in p:
        return AppAccessControlRequest(user_name=p, permission_level=lvl)
    if _UUID_RE.fullmatch(p):
        return AppAccessControlRequest(service_principal_name=p, permission_level=lvl)
    return AppAccessControlRequest(group_name=p, permission_level=lvl)


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
    # Each entry is either AiSearchVectorStoreModel (IsDatabricksResource
    # directly) or LakebaseVectorStoreModel (delegates auth + as_resources
    # to its nested DatabaseModel). Both quack as IsDatabricksResource
    # via delegation for the deploy iteration below.
    vector_indexes: Sequence[Any] = list(config.resources.vector_stores.values())
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

    # A UC-securable model contributes no resource (nothing in MLflow can
    # declare one), and Model Serving's automatic authentication hands the
    # container a token downscoped to exactly the declared resources. So the
    # model is simply invisible at runtime: verified on a live endpoint, the
    # gateway answers ``404 NOT_FOUND: '<name>' does not exist`` for a name that
    # returns 200 under a full-scope token. Say so at deploy time — the runtime
    # 404 names a model that demonstrably exists, which sends people looking in
    # the wrong place.
    # Deduped: two keys can address one model (a qualified name and a schema
    # anchor resolve alike), and this reads as a list of things to go fix.
    unreachable: list[str] = sorted(
        {
            m.full_name
            for m in config.resources.models.values()
            if not m.on_behalf_of_user and m.is_uc_securable
        }
        if config.resources is not None
        else set()
    )
    if unreachable:
        logger.warning(
            "UC-securable models are not reachable from a Model Serving "
            "endpoint under automatic authentication",
            models=unreachable,
            note=(
                "Model Serving downscopes the endpoint's token to the declared "
                "resources, and no MLflow resource type can declare a "
                "UC-securable model, so these will fail at request time with a "
                "404 that claims the model does not exist. Use the serving "
                "endpoint spelling (e.g. databricks-claude-sonnet-4-5) for a "
                "Model Serving deploy, or set on_behalf_of_user: true so the "
                "user's forwarded token is used instead."
            ),
        )

    if config.app and config.app.trace_location:
        # Currently a no-op (returns []). Kept as a call site so future
        # non-trace-table resources can attach to trace_location without
        # touching this function.
        system_resources.extend(config.app.trace_location.as_resources())

    # Translate resource-level api_scopes to canonical OBO user scopes — the
    # same translation used by the Databricks Apps path — so the deployed
    # model's forwarded user token carries the strings the platform recognizes
    # (e.g. ``sql``, ``genie``, ``files``, ``vector-search``). Without this, the
    # user token would claim dao-ai's internal ``sql.warehouses`` /
    # ``dashboards.genie`` strings, which the platform rejects.
    #
    # Then adapt for Model Serving, whose OBO allowlist differs from the Apps
    # platform's: it takes the single coarse ``unity-catalog`` scope where Apps
    # takes the per-securable ``catalog.*:read`` triple. Passing the Apps set
    # through unchanged fails the deploy outright with InvalidParameterValue.
    from dao_ai.apps.resources import (
        adapt_user_api_scopes_for_model_serving,
        generate_user_api_scopes,
    )

    api_scopes: list[str] = adapt_user_api_scopes_for_model_serving(
        generate_user_api_scopes(config)
    )

    return AuthPolicy(
        system_auth_policy=SystemAuthPolicy(resources=system_resources),
        user_auth_policy=UserAuthPolicy(api_scopes=api_scopes),
    )


def _use_local_source(development: bool | None) -> bool:
    """Decide whether a deploy should ship local dao-ai source/wheel vs PyPI.

    Thin wrapper over ``resolve_use_local_source`` (the shared tri-state resolver
    in ``dao_ai.utils``) so the provider, CLI handlers, and deploy notebook all
    agree on the ``--development`` semantics.
    """
    return resolve_use_local_source(development)


def _app_config_content(config: AppConfig) -> tuple[bytes, str]:
    """Pick the config bytes an Apps deploy uploads, and say where they came from.

    Three input shapes feed this:

    1. ``AppConfig.from_file(path, params={...})`` — the substituted text is on
       ``config.rendered_yaml``. Preferred.
    2. Legacy ``from_file`` callers without ``params=`` — read the raw source file
       off disk.
    3. An ``AppConfig`` built in pure Python — neither exists, so serialize the
       in-memory model back to YAML.

    The two *text* shapes carry bytes that predate resolution, so they get
    ``_bake_genie_room_details`` back-filling each Genie room's
    ``name``/``description``/``sample_questions`` — the same bake the DABs App
    bundle does. Without it a ``workflow up --mode apps`` deploy ships an unbaked
    config and the room loses its ``sample_questions``: ``name`` and
    ``description`` survive because the container's ``ensure_resolved`` back-fills
    them under ``CAN_RUN``, but the questions come out of the serialized space
    payload, which needs ``CAN_EDIT`` — a permission a deployed app SP does not
    hold. Two consumers need them there: a tool with
    ``include_example_questions: true``, whose routing hints would silently go
    missing, and space *provisioning*, which writes the declared questions back
    into the payload. Shape 3 needs nothing — it dumps objects ``initialize()``
    already resolved with the deployer's credentials.

    Best-effort, like the bake itself: any failure returns the unbaked bytes, so
    a deploy is never blocked by discovery.
    """
    if config.rendered_yaml is not None:
        content = config.rendered_yaml.encode("utf-8")
        origin = "rendered_yaml (parameter-substituted)"
    elif config.source_config_path:
        with open(config.source_config_path, "rb") as f:
            content = f.read()
        origin = f"source file {config.source_config_path}"
    else:
        config_dict: dict[str, Any] = config.model_dump(
            mode="json", by_alias=True, exclude_none=True
        )
        return (
            yaml.safe_dump(
                config_dict, sort_keys=False, default_flow_style=False
            ).encode("utf-8"),
            "in-memory AppConfig (programmatic)",
        )

    from dao_ai.apps.bundle import _bake_genie_room_details

    try:
        content = _bake_genie_room_details(content.decode("utf-8")).encode("utf-8")
    except Exception as e:  # noqa: BLE001 - discovery must never block a deploy
        logger.debug(f"Could not bake Genie room details into the config: {e}")

    return content, origin


def _warn_if_stale_dev_wheel(dev_wheel: Path | None) -> None:
    """Log which local wheel a deploy is about to ship, and warn if it looks
    stale relative to the working-tree source.

    ``dao-ai deploy`` in local-source mode ships ``find_dev_wheel()`` — a
    pre-built wheel under ``dist/``. If that wheel predates recent source
    edits, the deploy silently ships old code. Surface the wheel name + mtime,
    and warn when any tracked ``.py`` under the package source is newer than
    the wheel so the operator knows to rebuild (``uv build --wheel``).
    """
    if dev_wheel is None:
        return
    try:
        import dao_ai

        wheel_mtime: float = dev_wheel.stat().st_mtime
        pkg_root: Path = Path(dao_ai.__file__).parent
        newest_src: float = max(
            (p.stat().st_mtime for p in pkg_root.rglob("*.py")),
            default=0.0,
        )
        if newest_src > wheel_mtime:
            logger.warning(
                "Shipping a dev wheel that is OLDER than the working-tree "
                "source — the deploy may ship stale code. Rebuild with "
                "'uv build --wheel' before deploying.",
                wheel=dev_wheel.name,
                wheel_mtime=wheel_mtime,
                newest_source_mtime=newest_src,
            )
        else:
            logger.info("Shipping dev wheel", wheel=dev_wheel.name)
    except Exception as exc:  # noqa: BLE001 — diagnostics must never break deploy
        logger.debug("Dev-wheel staleness check skipped", error=str(exc))


# Bounded wait for a Delta-Sync index to reach ONLINE, so a stale checkpoint
# can't hang provisioning indefinitely (the failure mode was a ~2h stall).
_VS_INDEX_READY_TIMEOUT_SECONDS: int = 1200  # 20 min

# How long a full initial snapshot may run without visible progress before we
# give up. A recreate re-embeds the entire source table, which for a large table
# legitimately exceeds the plain readiness timeout above — a 38k-row table was
# observed taking ~22 min. The bound that matters there is "no rows indexed for
# this long", not total elapsed, so a big table gets the time it needs while a
# genuinely stuck index still fails fast.
_VS_SNAPSHOT_STALL_TIMEOUT_SECONDS: int = 900  # 15 min with no new rows

# Absolute ceiling on an initial snapshot, however much progress it is making.
_VS_SNAPSHOT_MAX_TIMEOUT_SECONDS: int = 7200  # 2h

# Index states that mean "a full initial snapshot is in flight".
_VS_SNAPSHOT_STATES: tuple[str, ...] = (
    "PROVISIONING_INITIAL_SNAPSHOT",
    "PROVISIONING",
)


def _describe_index_safe(index: VectorSearchIndex) -> dict[str, Any]:
    """``index.describe()`` wrapped so callers never crash on transient errors."""
    try:
        details = index.describe()
        return details if isinstance(details, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Vector index describe failed", error=str(exc))
        return {}


def _source_table_delta_uuid(source_table_full_name: str) -> str | None:
    """Return the source Delta table's stable GUID via ``DESCRIBE DETAIL``.

    The GUID changes when the table is dropped/recreated (``CREATE OR REPLACE``,
    overwrite reload, retried ingest) — the trigger that strands a Delta-Sync
    index's streaming checkpoint. Returns None if Spark is unavailable or the
    lookup fails (callers must treat None as "unknown", never as a mismatch).
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark is None:
            return None
        row = spark.sql(f"DESCRIBE DETAIL {source_table_full_name}").select("id").head()
        return row["id"] if row is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Could not read source table Delta UUID",
            source_table=source_table_full_name,
            error=str(exc),
        )
        return None


def _index_is_delta_sync(details: dict[str, Any]) -> bool:
    """True only for Delta-Sync indexes. Direct-Access indexes (no source
    table, no streaming checkpoint) can never go stale this way, so the
    self-heal must never touch them."""
    return bool(details.get("delta_sync_index_spec"))


def _index_is_stale(
    index: VectorSearchIndex,
    details: dict[str, Any],
    source_table_full_name: str | None,
) -> bool:
    """Decide whether an existing Delta-Sync index's checkpoint is stale and the
    index must be dropped + recreated rather than synced.

    Two OR'd detectors, both defensive (any error → not-stale, fall through to a
    normal sync so the heuristic can never block a healthy deploy):

    1. **State**: ``status.detailed_state`` contains ``FAILED`` → the sync
       pipeline is in the stuck-failing loop (the observed hang).
    2. **UUID**: the live source table's Delta GUID differs from the GUID the
       index recorded at creation (read back from the index's stored metadata).
       Authoritative for the drop/recreate/overwrite trigger.
    """
    # Only Delta-Sync indexes have a checkpoint to go stale.
    if not _index_is_delta_sync(details):
        return False

    # Detector 1 — failed sync state.
    state = str((details.get("status") or {}).get("detailed_state", "")).upper()
    if "FAILED" in state:
        logger.warning(
            "Vector index in a FAILED state — treating checkpoint as stale",
            detailed_state=state,
        )
        return True

    # Detector 2 — source table GUID drift vs. the GUID recorded at create time.
    if source_table_full_name:
        recorded = _read_index_source_uuid(details)
        live = _source_table_delta_uuid(source_table_full_name)
        if recorded and live and recorded != live:
            logger.warning(
                "Vector index source table GUID changed since index creation "
                "— checkpoint is stale",
                recorded_uuid=recorded,
                live_uuid=live,
                source_table=source_table_full_name,
            )
            return True

    return False


def _index_source_uuid_key() -> str:
    """Custom-tag key under which we stamp the source table's Delta GUID."""
    return "dao_ai_source_delta_uuid"


def _read_index_source_uuid(details: dict[str, Any]) -> str | None:
    """Read the source Delta GUID we stamped on the index at create time.

    Stored under the Delta-Sync spec (``columns_to_sync`` sibling) is not
    available, so we persist it in the index's custom tags. Returns None when
    absent (e.g. an index created before this feature shipped) — detector 2
    then no-ops and we rely on detector 1 + the bounded timeout.
    """
    spec = details.get("delta_sync_index_spec") or {}
    # Prefer an explicit recorded value if the platform ever surfaces one.
    for container in (details, spec):
        tags = container.get("custom_tags") or container.get("tags") or {}
        if isinstance(tags, dict) and tags.get(_index_source_uuid_key()):
            return str(tags[_index_source_uuid_key()])
        # tags can also be a list of {key, value} dicts.
        if isinstance(tags, list):
            for t in tags:
                if isinstance(t, dict) and t.get("key") == _index_source_uuid_key():
                    return str(t.get("value"))
    return None


def _wait_until_index_absent(
    vsc: VectorSearchClient,
    endpoint_name: str,
    index_full_name: str,
    timeout_seconds: int = 300,
) -> None:
    """Block until a just-deleted index no longer resolves, so a recreate can't
    race the delete. Bounded; logs and returns on timeout rather than hanging."""
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not index_exists(vsc, endpoint_name, index_full_name):
            return
        time.sleep(5)
    logger.warning(
        "Index still present after delete wait — proceeding anyway",
        index_name=index_full_name,
    )


def _wait_for_initial_snapshot(
    index: VectorSearchIndex,
    index_name: str,
    *,
    stall_timeout_seconds: int = _VS_SNAPSHOT_STALL_TIMEOUT_SECONDS,
    max_timeout_seconds: int = _VS_SNAPSHOT_MAX_TIMEOUT_SECONDS,
    poll_seconds: int = 15,
) -> str:
    """Wait out a full initial snapshot, bounding on *stalled progress* not elapsed.

    A freshly recreated Delta-Sync index re-embeds the whole source table, and for
    a large table that legitimately runs longer than
    ``_VS_INDEX_READY_TIMEOUT_SECONDS``. Timing out on total elapsed there fails a
    task whose recovery is still working — and tells the user to delete and
    recreate the index, which is exactly what just happened, so following the
    advice loops.

    Watching ``indexed_row_count`` instead separates the two cases: rising means
    the snapshot is progressing and deserves more time; flat for
    ``stall_timeout_seconds`` means it really is stuck.

    Returns:
        The terminal ``detailed_state`` observed — the caller decides whether it
        is acceptable.
    """
    deadline = time.monotonic() + max_timeout_seconds
    last_rows = -1
    last_progress_at = time.monotonic()
    state = "UNKNOWN"

    while time.monotonic() < deadline:
        status = _describe_index_safe(index).get("status") or {}
        state = str(status.get("detailed_state", "UNKNOWN")).upper()
        rows = int(status.get("indexed_row_count") or 0)

        if not any(s in state for s in _VS_SNAPSHOT_STATES):
            # Snapshot finished (well or badly) — hand the state back.
            logger.info(
                "Initial snapshot settled",
                index_name=index_name,
                detailed_state=state,
                indexed_rows=rows,
            )
            return state

        if rows > last_rows:
            logger.info(
                "Initial snapshot progressing",
                index_name=index_name,
                indexed_rows=rows,
                detailed_state=state,
            )
            last_rows = rows
            last_progress_at = time.monotonic()
        elif time.monotonic() - last_progress_at >= stall_timeout_seconds:
            logger.warning(
                "Initial snapshot stalled — no new rows indexed",
                index_name=index_name,
                indexed_rows=rows,
                detailed_state=state,
                stall_timeout_seconds=stall_timeout_seconds,
            )
            return state

        time.sleep(poll_seconds)

    logger.warning(
        "Initial snapshot exceeded its absolute ceiling",
        index_name=index_name,
        detailed_state=state,
        max_timeout_seconds=max_timeout_seconds,
    )
    return state


def _sync_when_pipeline_idle(
    index: VectorSearchIndex,
    index_name: str,
    timeout_seconds: int = _VS_INDEX_READY_TIMEOUT_SECONDS,
) -> None:
    """Trigger an incremental ``index.sync()``, tolerating a still-running sync.

    ``index.sync()`` (Delta-Sync, ``pipeline_type=TRIGGERED``) only succeeds when
    the underlying pipeline is idle; the platform rejects it with a 400 (message
    contains ``Pipeline is in state RUNNING``) while a sync is in flight. That is
    benign — a pipeline already RUNNING is a sync already in progress, which
    reaches the same end state. Poll past that specific 400 until the pipeline is
    idle and the sync is accepted (or the pipeline finishes on its own). Any other
    ``BadRequest`` re-raises unchanged. Bounded so a genuinely stuck pipeline
    can't hang provisioning — the caller's ``wait_until_ready(wait_for_updates=
    True)`` asserts the index actually lands ``ONLINE_NO_PENDING_UPDATE``.
    """
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        try:
            index.sync()
            return
        except AISearchBadRequest as exc:
            attempt += 1
            if "pipeline is in state running" not in str(exc).lower():
                raise
            if time.monotonic() >= deadline:
                logger.warning(
                    "Delta-Sync pipeline still RUNNING at sync timeout — relying "
                    "on the in-flight sync; proceeding to readiness wait",
                    index_name=index_name,
                    timeout_seconds=timeout_seconds,
                    attempts=attempt,
                )
                return
            logger.info(
                "Delta-Sync pipeline still RUNNING — a sync is already in flight; "
                "waiting for it to settle before retrying",
                index_name=index_name,
                attempt=attempt,
            )
            time.sleep(30)


def link_experiment_trace_location(config: AppConfig, experiment_id: str) -> None:
    """Link an MLflow experiment to its UC trace location.

    Wraps ``mlflow.set_experiment(experiment_id=..., trace_location=
    UnityCatalog(...))`` — the post-MLflow-3.11 blessed API. Replaces the
    deprecated combination of ``mlflow.tracing.set_destination(
    UCSchemaLocation(...))`` + ``mlflow.tracing.enablement.
    set_experiment_trace_location(...)`` which emit deprecation warnings.

    Idempotent: reads the experiment's current UC trace-destination tags
    (via ``MlflowClient.get_experiment``) and skips the API call when the
    destination already matches the desired config. This avoids MLflow's
    "already contains traces" rejection on re-deploys of an experiment
    that was previously linked to the same UC schema.

    On a truly conflicting state (experiment has traces + no UC linkage,
    or linkage points somewhere else), the underlying RestException
    surfaces — no silent swallow. Callers that want to tolerate this
    (Apps runtime, where re-linking sometimes fails on a broken deploy
    state) wrap the call in their own try/except and log the diagnostic.

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

    if _experiment_already_linked(experiment_id, loc, table_prefix):
        logger.info(
            "Experiment already linked to matching UC trace destination, skipping",
            experiment_id=experiment_id,
            catalog=loc.catalog_name,
            schema=loc.schema_name,
            table_prefix=table_prefix,
        )
        return

    try:
        mlflow.set_experiment(
            experiment_id=experiment_id,
            trace_location=UnityCatalog(**uc_kwargs),
        )
    except mlflow.exceptions.RestException as e:
        # Fall-through safety net: if the tag-based idempotency check
        # above missed (transient MlflowClient error, e.g.) and the
        # actual link is already in place, MLflow rejects the re-link
        # with "already contains traces". Treat that as idempotent and
        # continue. Other RestExceptions (warehouse timeouts, schema
        # permission errors) surface so the caller can fail loudly.
        if "already contains traces" in str(e):
            logger.warning(
                "UC trace destination re-link rejected ('already contains "
                "traces') — assuming prior link is still in effect",
                experiment_id=experiment_id,
            )
            return
        raise
    logger.info(
        "Linked experiment to UC trace location",
        catalog=loc.catalog_name,
        schema=loc.schema_name,
        table_prefix=table_prefix,
    )


def apply_runtime_trace_destination(config: AppConfig) -> None:
    """Populate MLflow's client-side trace-destination ContextVar with the
    ``UnityCatalog`` matching ``config.app.trace_location``.

    ``mlflow.set_experiment(..., trace_location=UnityCatalog(...))`` records
    the server-side link, but the client-side OTEL span exporter reads
    ``mlflow.tracing.utils.get_active_spans_table_name()`` — which returns
    ``None`` unless the ``_MLFLOW_TRACE_USER_DESTINATION`` ContextVar was
    set via ``mlflow.tracing.set_destination(...)``. When the ContextVar
    is empty AND ``MLFLOW_TRACING_DESTINATION`` env is set to the 2-part
    ``catalog.schema`` string, MLflow parses that as the deprecated
    ``UCSchemaLocation`` whose ``full_otel_spans_table_name`` falls back
    to the hardcoded default ``mlflow_experiment_trace_otel_spans`` — the
    OTEL exporter then writes to a table that doesn't exist on the
    prefixed schema, silently dropping traces.

    Fix: after linking (or when the link is already in place), also set
    the ContextVar to the correct ``UnityCatalog(catalog, schema,
    table_prefix)``. Idempotent — safe to call multiple times.

    No-op when ``config.app.trace_location`` is None.
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
        # ``mlflow.tracing.set_destination(UnityCatalog(...))`` explicitly
        # rejects the table-prefix form ("not supported by set_destination").
        # The correct API is ``mlflow.set_experiment(trace_location=UC(...))``
        # which internally calls ``_sync_trace_destination_and_provider``
        # to populate the ``_MLFLOW_TRACE_USER_DESTINATION`` ContextVar
        # and reset the OTEL provider. We replicate that here so the
        # runtime picks the correct table even when ``set_experiment`` is
        # skipped (idempotent link-already-in-place path).
        from mlflow.tracing.provider import (
            _MLFLOW_TRACE_USER_DESTINATION,
            provider,
        )

        if provider.once._done:
            provider.reset()

        if table_prefix:
            # Prefixed case: set the ContextVar to a fully-qualified
            # UnityCatalog so the exporter picks the correct table (e.g.
            # ``<catalog>.<schema>.<prefix>_otel_spans``).
            destination = UnityCatalog(**uc_kwargs)
            # ``UnityCatalog.full_otel_*_table_name`` returns the private
            # ``_otel_*_table_name`` fields verbatim (unlike ``UCSchemaLocation``
            # which auto-qualifies with catalog/schema). Constructing with only
            # ``table_prefix`` leaves those fields ``None`` → the OTEL span
            # exporter reads ``get_active_spans_table_name() == None`` and
            # silently skips every write. Populate them with the FULLY-QUALIFIED
            # three-level names — the trace-server rejects a bare table name with
            # ``Invalid full table name`` (a failure otherwise hidden by the
            # Databricks SDK round-trip logger crashing on the BytesIO span
            # payload, ``object of type '_io.BytesIO' has no len()``).
            fq_prefix = f"{loc.catalog_name}.{loc.schema_name}.{table_prefix}"
            destination._otel_spans_table_name = f"{fq_prefix}_otel_spans"
            destination._otel_logs_table_name = f"{fq_prefix}_otel_logs"
            _MLFLOW_TRACE_USER_DESTINATION.set(destination)
        else:
            # No-prefix case: constructing ``UnityCatalog(catalog, schema)``
            # without ``table_prefix`` raises at ``full_table_prefix`` /
            # ``full_otel_spans_table_name`` access time — the exporter
            # then silently drops every span. Clear the ContextVar
            # instead so MLflow's own ``_resolve_experiment_uc_location``
            # kicks in and reads the experiment-linked ``UnityCatalog``
            # (with the backend-computed experiment-id prefix) from the
            # tracking store.
            _MLFLOW_TRACE_USER_DESTINATION.set(None)
        logger.info(
            "Set MLflow runtime trace destination",
            catalog=loc.catalog_name,
            schema=loc.schema_name,
            table_prefix=table_prefix,
            fallback_to_experiment_resolver=table_prefix is None,
        )
    except Exception as exc:
        # A failure here would silently drop traces — log loudly but do
        # not raise, so the app still boots. Operators can diagnose via
        # the log line + MLflow's own warnings on the first export.
        logger.warning(
            "Failed to set MLflow runtime trace destination",
            catalog=loc.catalog_name,
            schema=loc.schema_name,
            table_prefix=table_prefix,
            error=str(exc),
        )


# Kept as a private alias for internal call sites; new callers use the
# public name above. Remove once we're confident no external code depends
# on the private symbol.
_link_experiment_trace_location = link_experiment_trace_location


def _experiment_already_linked(
    experiment_id: str,
    loc: Any,
    table_prefix: Optional[str],
) -> bool:
    """True when the experiment's linked UC destination already matches.

    MLflow records the linked destination as a single dotted-string tag
    ``mlflow.experiment.databricksTraceDestinationPath`` (validated by
    reading MLflow 3.11 source — ``utils/mlflow_tags.py`` and
    ``tracking/fluent.py:329-336``). Format:
    ``<catalog>.<schema>.<table_prefix>``. When ``table_prefix`` was
    omitted at link time, MLflow substitutes the experiment id.

    We match by parsing the tag and comparing catalog + schema exactly.
    For ``table_prefix``: an explicit non-None config-side value must
    match the tag's third segment; a None config-side value matches any
    prefix (the user asked MLflow to auto-assign, so any existing
    linkage with the right catalog/schema is compatible).

    Any error looking up the experiment (permissions, transient API
    failure, etc.) returns ``False`` so the caller falls through to the
    normal link attempt — safest default.
    """
    try:
        from mlflow.tracking import MlflowClient

        exp = MlflowClient().get_experiment(experiment_id)
    except Exception:  # noqa: BLE001
        return False
    tags = exp.tags or {}
    destination_path: Optional[str] = tags.get(
        "mlflow.experiment.databricksTraceDestinationPath"
    )
    if not destination_path:
        return False
    parts = destination_path.split(".")
    if len(parts) < 2:
        return False
    tag_catalog, tag_schema = parts[0], parts[1]
    tag_prefix = parts[2] if len(parts) >= 3 else None
    if tag_catalog != loc.catalog_name:
        return False
    if tag_schema != loc.schema_name:
        return False
    if table_prefix is not None and tag_prefix != table_prefix:
        return False
    # table_prefix is None on the config side → any existing prefix is
    # compatible (user delegated naming to MLflow).
    return True


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
                body={"changes": [{"principal": principal, "add": list(privileges)}]},
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


def _otel_table_names(
    catalog_name: str, schema_name: str, table_prefix: str
) -> list[str]:
    """The four fully-qualified OTEL trace table names for a prefix.

    Mirrors what MLflow materializes for a UC trace location:
    ``<catalog>.<schema>.<prefix>_otel_{spans,logs,metrics,annotations}``.
    """
    return [
        f"{catalog_name}.{schema_name}.{table_prefix}_otel_{suffix}"
        for suffix in ("spans", "logs", "metrics", "annotations")
    ]


def _drop_uc_otel_tables(
    catalog_name: str,
    schema_name: str,
    table_prefix: str,
    profile: Optional[str] = None,
) -> None:
    """Permanently drop the four OTEL trace tables for ``table_prefix``.

    The inverse of :func:`_grant_uc_trace_table_permissions_to_principal` — used by
    ``down --purge`` to clean up the Delta tables MLflow lazily materialized at
    ``<catalog>.<schema>.<prefix>_otel_{spans,logs,metrics,annotations}``. Deletes
    each table via the UC Tables SDK (``w.tables.delete(full_name=...)``) rather
    than a SQL ``DROP`` — no warehouse needed, and it sidesteps the identifier-
    quoting trap where a fully-qualified dotted name inside a single backtick pair
    is parsed as one literal identifier (so ``DROP TABLE IF EXISTS`` silently
    no-ops instead of dropping). A table that was never materialized (traces never
    exported) raises ``NotFound`` and is treated as a harmless no-op.

    CALLER CONTRACT: only call this with a prefix that uniquely identifies ONE
    experiment (i.e. the experiment_id, when ``trace_location.table_prefix`` is
    unset). An explicitly-configured ``table_prefix`` may be SHARED across agents,
    so purge must not drop those tables — see :func:`_purge_experiment`.

    Best-effort: each delete is independent; any failure is logged and swallowed so
    ``down`` still completes.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound

    w = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
    for full_name in _otel_table_names(catalog_name, schema_name, table_prefix):
        try:
            w.tables.delete(full_name=full_name)
            logger.info(f"Purged (dropped) OTEL trace table '{full_name}'.")
        except NotFound:
            logger.debug(
                f"OTEL trace table '{full_name}' not found (never materialized) "
                "— nothing to drop."
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to drop OTEL trace table during purge",
                full_name=full_name,
                error=str(e),
            )


def _resolve_trace_table_prefix(config: AppConfig, experiment_id: Optional[str]) -> str:
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
    """Build a dao-ai wheel from source using uv.

    Stamps a unique PEP 440 local version so the dev wheel always out-ranks the
    same-base-version published package in an Apps container (pip skips a
    same-version reinstall otherwise). See ``dev_local_version``.
    """
    import subprocess

    from dao_ai.utils import dev_local_version

    dist_dir = project_root / "dist"
    dist_dir.mkdir(exist_ok=True)

    with dev_local_version(project_root / "pyproject.toml"):
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

    def experiment_name(self, config: AppConfig, *, as_mcp: bool = False) -> str:
        """Resolve the experiment path from ``app.experiment.name``, or
        fall back to ``/Users/<deployer_email>/<app.name>`` when not set.

        The id-based ``app.experiment.id`` branch never lands here — it
        short-circuits ``get_or_create_experiment`` directly.

        ``as_mcp`` prefixes the fallback name with ``mcp-``, matching what the
        MCP bundle declares (``_build_app_block`` derives the experiment path
        from the same prefixed app name). Without this the SDK ``--direct`` path
        and the bundle path would bind different experiments for one deployment.
        """
        if config.app.experiment is not None and config.app.experiment.resolved_name:
            return config.app.experiment.resolved_name
        current_user: User = self.w.current_user.me()
        name: str = f"mcp-{config.app.name}" if as_mcp else config.app.name
        return f"/Users/{current_user.user_name}/{name}"

    def get_or_create_experiment(
        self, config: AppConfig, *, as_mcp: bool = False
    ) -> Experiment:
        """Resolve to an MLflow ``Experiment``.

        * ``app.experiment`` set → delegate to ``self.create_experiment``
          (id takes precedence; name resolved lazily with optional create,
          per the dao-ai ``Model.create(w)`` convention).
        * otherwise → fall back to the historical default
          ``/Users/<deployer_email>/<app.name>`` with restore-if-deleted,
          create-if-missing (``mcp-`` prefixed when ``as_mcp``).
        """
        if config.app.experiment is not None:
            return self.create_experiment(config.app.experiment)

        experiment_name: str = self.experiment_name(config, as_mcp=as_mcp)
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
        development: bool | None = None,
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

        # Resolve the model's custom code: explicit ``config.app.code_paths``
        # (out-of-tree, fail-loud on a missing entry) UNION the colocated ``src/``
        # packages (the zero-config convention), deduped. Each is passed to
        # ``log_model(code_paths=...)`` so MLflow copies it to ``code/<pkg>`` and
        # it imports prefix-free (``src/foo`` -> ``foo.bar``). ``from_file`` already
        # put these on ``sys.path`` (best-effort) for the in-process validation load.
        from dao_ai.code_paths import collect_serving_code_paths

        code_paths: list[str] = collect_serving_code_paths(config)

        model_root_path: Path = Path(dao_ai.__file__).parent
        model_path: Path = model_root_path / "apps" / "model_serving.py"

        # A *copy*: everything below appends to this list, and the serving-only
        # additions must not leak back onto the config. ``config.app`` is a live
        # pydantic model, so ``+=`` on the field itself is an in-place extend —
        # a `deploy_agent(mode=BOTH)` would then hand the Apps bundle
        # ``code/dao_ai-<ver>.whl`` (an MLflow-relative wheel path, not a PEP 508
        # requirement) plus the whole frozen environment, and its ``uv lock``
        # would fail.
        pip_requirements: list[str] = list(config.app.pip_requirements)

        # Resolve which optional-feature extras this config exercises so the
        # deployed model pins exactly the packages its features need — no more
        # (keeps the image small), no less (missing one crashes at serving load).
        from dao_ai._extras import (
            expand_all,
            format_extras_suffix,
            resolve_required_extras,
        )

        # Model Serving does not mount A2A routes, so the always-on a2a routes
        # must NOT bloat the serving image — only an explicit A2A tool pulls it.
        # Use the PRECISE resolver (not ``_or_all``): a deployed artifact always
        # wants the minimal config-specific extras, even though the deploy runs
        # inside a notebook (where ``_or_all`` would short-circuit to every extra).
        required_extras: set[str] = resolve_required_extras(
            config, target="model_serving"
        )
        extras_suffix: str = format_extras_suffix(required_extras)

        if not _use_local_source(development):
            if not is_lib_provided("dao-ai", pip_requirements):
                pip_requirements += [
                    f"dao-ai{extras_suffix}=={dao_ai_version()}",
                ]
            logger.info(
                "dao-ai source: PyPI package",
                version=dao_ai_version(),
                extras=sorted(required_extras),
            )
        else:
            dev_wheel: Path | None = find_dev_wheel()
            if dev_wheel:
                _warn_if_stale_dev_wheel(dev_wheel)
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
                    # Local-source mode was requested (explicit --development or
                    # auto-detect said "not published"), but there is neither a
                    # pre-built wheel nor a source tree to ship — dao-ai is
                    # installed into site-packages. Continuing would log a model
                    # with NO dao-ai (only its transitive deps), which fails to
                    # import at serving load. Fail loud, matching the Apps and
                    # generate-agent paths, instead of shipping a broken model.
                    raise RuntimeError(
                        "No dao-ai wheel found and project source not available "
                        "(dao-ai is installed from a package index). Build a "
                        "wheel first with 'uv build --wheel', or pass "
                        "--no-development to deploy the published PyPI package."
                    )

            pip_requirements += get_installed_packages(expand_all(required_extras))

        from dao_ai.skills import (
            assert_skill_assets_resolvable,
            collect_instruction_file_code_paths,
            collect_skills_code_paths,
        )

        # Stop before logging a model whose skills cannot load. At serve time a
        # missing skill only warns (raising there would turn a degraded agent into
        # a dead endpoint), so this is the last point with a human watching.
        assert_skill_assets_resolvable(config, target="Model Serving deploy")
        code_paths.extend(collect_skills_code_paths(config))
        code_paths.extend(collect_instruction_file_code_paths(config))

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

        # Create conda environment with configured Python version. This lets the
        # Model Serving container's Python be pinned independently of the environment
        # running the deploy (a local machine, CI, or a job may be on a different version).
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
        workload_size: str = config.app.serving_workload_size()
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
                    "Failed to grant trace-persistence privileges to Model Serving SP",
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

    def _upload_code_paths(self, config: AppConfig, source_path: str) -> None:
        """Upload ``config.app.code_paths`` files under the app ``source_path``.

        Each entry is placed preserving its config-relative layout (or under
        ``code/`` for absolute / ``../``-climbing entries), so the app's
        ``add_code_paths_to_sys_path`` validator finds it on ``sys.path`` at
        runtime. Directories are walked file-by-file (``workspace.upload`` has no
        recursive form); parent workspace dirs are created as needed.
        """
        from dao_ai.code_paths import iter_code_path_stagings

        stagings = iter_code_path_stagings(config)
        if not stagings:
            return

        uploaded = 0
        for src, dest in stagings:
            uploaded += self._upload_dir_files(src, dest, source_path)
        logger.info(
            "Uploaded custom code (app.code_paths) to app source",
            file_count=uploaded,
            source_path=source_path,
        )

    def _upload_skill_dirs(self, config: AppConfig, source_path: str) -> None:
        """Upload local skill directories under the app ``source_path``.

        The direct-deploy path is the fourth place skill content has to be
        staged, alongside the three bundle generators. It is easy to miss because
        nothing fails without it: the uploaded config still *names* its skills,
        the app comes up healthy, and the agent simply runs without them.

        The config-relative layout (``skills/<vertical>/<skill>``) is what lets
        the relative source in the uploaded config resolve against the app's CWD,
        which is ``source_path``.
        """
        from dao_ai.skills import iter_skill_stagings

        stagings = iter_skill_stagings(config)
        if not stagings:
            return

        uploaded = 0
        for src, dest in stagings:
            uploaded += self._upload_dir_files(src, dest, source_path)
        logger.info(
            "Uploaded skill directories to app source",
            file_count=uploaded,
            skill_count=len(stagings),
            source_path=source_path,
        )

    def _upload_instruction_files(self, config: AppConfig, source_path: str) -> None:
        """Upload ``deep_agent.instruction_files`` under the app ``source_path``.

        Files already inside an uploaded skill directory are not re-uploaded —
        :func:`~dao_ai.skills.iter_instruction_file_stagings` drops them, because
        the natural home for an ``AGENTS.md`` is the skill it documents and
        :meth:`_upload_skill_dirs` has already put it there.
        """
        from dao_ai.skills import iter_instruction_file_stagings

        stagings = iter_instruction_file_stagings(config)
        if not stagings:
            return

        uploaded = 0
        for src, dest in stagings:
            uploaded += self._upload_dir_files(src, dest, source_path)
        logger.info(
            "Uploaded instruction files to app source",
            file_count=uploaded,
            source_path=source_path,
        )

    def _upload_src_packages(self, config: AppConfig, source_path: str) -> None:
        """Upload colocated ``src/<pkg>`` packages under ``<source_path>/src``.

        The ``src/`` convention: every top-level package under the config's
        ``src/`` is uploaded so the app's ``uv sync`` (with the generated
        ``packages=["src"]`` pyproject) builds it into the app wheel — importing
        prefix-free (``src/foo`` -> ``foo.bar``). No config declaration needed.
        """
        from dao_ai.code_paths import _SRC_DIRNAME, discover_src_packages

        uploaded = 0
        for pkg_dir in discover_src_packages(config):
            uploaded += self._upload_dir_files(
                pkg_dir, f"{_SRC_DIRNAME}/{pkg_dir.name}", source_path
            )
        if uploaded:
            logger.info(
                "Uploaded src/ packages to app source",
                file_count=uploaded,
                source_path=source_path,
            )

    def _upload_dir_files(self, src: "Path", dest: str, source_path: str) -> int:
        """Upload one staging pair's files under ``source_path`` (workspace).

        ``workspace.upload`` has no recursive form, so directories are walked
        file-by-file with parent workspace dirs created as needed. Returns the
        number of files uploaded.
        """
        import io

        from dao_ai.code_paths import walk_code_path_files

        count = 0
        for file_src, file_dest in walk_code_path_files(src, dest):
            ws_path = f"{source_path}/{file_dest}"
            parent = ws_path.rsplit("/", 1)[0]
            try:
                self.w.workspace.mkdirs(parent)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"workspace parent dir may exist: {e}")
            with open(file_src, "rb") as f:
                self.w.workspace.upload(
                    path=ws_path,
                    content=io.BytesIO(f.read()),
                    format=ImportFormat.AUTO,
                    overwrite=True,
                )
            count += 1
        return count

    def _deploy_app(
        self,
        config: AppConfig,
        *,
        app_command: list[str],
        extras: set[str],
        include_chat_ui: bool,
        as_mcp: bool = False,
        development: bool | None = None,
    ) -> None:
        """
        Deploy a dao-ai app to Databricks Apps via the SDK-native path.

        Shared machinery for both protocols served by :meth:`deploy_apps_agent`
        — the chat UI and the MCP server. Creates or updates a
        Databricks App, uploading config + code + a portable pyproject/uv.lock,
        then deploys and waits, links the trace destination, and grants the
        App SP the trace-persistence privileges.

        The deployment process:
        1. Determine the workspace source path for the app
        2. Upload the configuration file to the workspace
        3. Create the app if it doesn't exist
        4. Deploy the app

        Args:
            config: The AppConfig containing deployment configuration.
            app_command: The container command to run (e.g. the chat proxy or
                the MCP server entrypoint).
            extras: The dao-ai optional-feature extras to install.
            include_chat_ui: Whether the generated app.yaml should inject the
                chat-UI proxy env vars.
            as_mcp: Whether this is an MCP-server deployment. Selects the
                deployed App name (``mcp-`` prefixed) so an MCP server and a
                chat App from the same config don't replace one another.
            development: When True, ship local dao-ai source/wheel; when False,
                the published PyPI package; when None, auto-detect.

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

        # Resolve the deployed App name: lowercased/hyphenated, and ``mcp-``
        # prefixed for MCP deployments so the two protocols don't collide.
        raw_name: str = config.app.name
        app_name: str = app_name_for(config.app.name, as_mcp=as_mcp)
        if app_name != raw_name:
            logger.info(
                "Normalized app name for Databricks Apps",
                original=raw_name,
                normalized=app_name,
            )

        logger.info("Deploying agent to Databricks Apps", app_name=app_name)

        # Format the caller-resolved optional-feature extras so the uploaded
        # requirements install exactly what this deploy needs (e.g.
        # dao-ai[a2a,rerank]).
        from dao_ai._extras import format_extras_suffix

        extras_suffix: str = format_extras_suffix(extras)
        # User-declared extra pip packages for their custom app code — appended
        # to the uploaded requirements.txt so Apps installs them (parity with
        # Model Serving, which bakes them into the model's conda_env).
        user_pip_requirements: list[str] = (
            list(config.app.pip_requirements) if config.app else []
        )
        from dao_ai.apps.bundle import _format_extra_deps

        # User pip_requirements folded into the generated pyproject deps so the
        # uploaded uv.lock captures them (parity with the bundle generators).
        extra_deps: str = _format_extra_deps(user_pip_requirements)

        # Use convention-based workspace path: /Workspace/Users/{user}/apps/{app_name}
        current_user: User = self.w.current_user.me()
        user_name: str = current_user.user_name or "default"
        source_path: str = f"/Workspace/Users/{user_name}/apps/{app_name}"

        logger.info("Using workspace source path", source_path=source_path)

        # Get or create experiment for this app (for tracing and tracking)
        from mlflow.entities import Experiment

        experiment: Experiment = self.get_or_create_experiment(config, as_mcp=as_mcp)
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

        # Fail before touching the workspace: the upload below deletes
        # ``source_path`` recursively, so a config with an unresolvable skill
        # would otherwise take out a working deployment on its way to a
        # silently skill-less one.
        from dao_ai.skills import assert_skill_assets_resolvable

        assert_skill_assets_resolvable(config, target="Apps deploy")

        # Upload the configuration file to the workspace. See
        # ``_app_config_content`` for the three input shapes and which of them
        # get the Genie bake.
        config_file_name: str = "dao_ai.yaml"
        workspace_config_path: str = f"{source_path}/{config_file_name}"

        config_content: bytes
        config_origin: str
        config_content, config_origin = _app_config_content(config)

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

        # Upload the config's custom code (app.code_paths) next to the config so
        # it is importable in the app container: the app CWD is source_path and
        # AppConfig.from_file's add_code_paths_to_sys_path validator inserts each
        # entry's parent onto sys.path. Same declaration used by Model Serving
        # (log_model code_paths) and the bundle generators.
        self._upload_code_paths(config, source_path)

        # Upload the config's skill directories for the same reason, on the same
        # anchor: the app CWD is source_path, so a relative skills source in the
        # uploaded config resolves only if the content sits beside it.
        self._upload_skill_dirs(config, source_path)

        # And the config's instruction files (deepagents' ``memory=``), which are
        # resolved by declared path rather than discovered, so they have to land at
        # exactly the relative location the uploaded config names.
        self._upload_instruction_files(config, source_path)

        # Upload colocated src/<pkg> packages (the zero-config convention) so the
        # app's uv sync (packages=["src"]) builds them into the app wheel.
        self._upload_src_packages(config, source_path)

        # Determine install command based on dev vs published mode. The
        # container command (chat proxy vs MCP server) is chosen by the caller
        # and threaded in via ``app_command``.
        if not _use_local_source(development):
            # Ship ``pyproject.toml`` (sole ``dao-ai>={ver}`` dep) + a portable
            # ``uv.lock`` so Databricks Apps' build phase runs
            # ``uv sync --locked --no-dev`` and installs the exact pinned
            # closure. Both files are required for the uv path — ``pyproject.toml``
            # alone (no ``uv.lock``) logs ``No dependencies file found. Skipping
            # installation`` — and ``requirements.txt`` must be ABSENT (it would
            # take precedence and force the pip path).
            #
            # Pinning matters: without it the deployed app silently drifts behind
            # the local dao-ai (cached venv reused across deploys). Surfaced by the
            # workshop verification on 2026-06-23: Lab 15 introduced
            # ``app.background:``, but the cached venv was a pre-rename dao-ai that
            # rejected the new field as ``extra_forbidden`` and crashed at startup.
            # The lock is de-mirrored (public-CDN URLs) so it resolves in the Apps
            # container — see ``dao_ai._locking``.
            from dao_ai._locking import render_portable_lock
            from dao_ai.apps.bundle import _PYPROJECT_TEMPLATE

            app_name_normalized = raw_name.lower().replace("_", "-")
            package_name = app_name_normalized.replace("-", "_")
            pyproject_content = _PYPROJECT_TEMPLATE.format(
                name=app_name_normalized,
                package_name=package_name,
                dao_ai_version=dao_ai_version(),
                extras=extras_suffix,
                extra_deps=extra_deps,
            )
            self.w.workspace.upload(
                path=f"{source_path}/pyproject.toml",
                content=io.BytesIO(pyproject_content.encode("utf-8")),
                format=ImportFormat.AUTO,
                overwrite=True,
            )

            # Ship a portable uv.lock (public-CDN pinned) alongside pyproject.
            # Apps' build phase runs ``uv sync --locked --no-dev`` from
            # pyproject.toml + uv.lock; requirements.txt is intentionally NOT
            # shipped (it would take precedence over the uv path).
            lock_content = render_portable_lock(pyproject_content)
            self.w.workspace.upload(
                path=f"{source_path}/uv.lock",
                content=io.BytesIO(lock_content.encode("utf-8")),
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

            logger.info(
                "dao-ai source for app: PyPI package",
                version=dao_ai_version(),
                command=app_command,
            )
        else:
            dev_wheel: Path | None = find_dev_wheel()
            _warn_if_stale_dev_wheel(dev_wheel)

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

            # Upload pyproject.toml (dao-ai redirected to the bundled wheel via
            # ``[tool.uv.sources]``) + a portable uv.lock. Apps' build phase runs
            # ``uv sync --locked --no-dev`` and installs THIS wheel.
            from dao_ai._locking import render_portable_lock
            from dao_ai.apps.bundle import _PYPROJECT_DEV_TEMPLATE

            app_name_normalized = raw_name.lower().replace("_", "-")
            package_name = app_name_normalized.replace("-", "_")
            pyproject_content = _PYPROJECT_DEV_TEMPLATE.format(
                name=app_name_normalized,
                package_name=package_name,
                wheel_filename=wheel_path.name,
                extras=extras_suffix,
                extra_deps=extra_deps,
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

            # Portable uv.lock — locked against the local wheel (via
            # ``[tool.uv.sources]``) plus the public-PyPI closure, with any
            # internal-mirror host rewritten to the public CDN. No
            # requirements.txt (it would take precedence over the uv path).
            lock_content: str = render_portable_lock(
                pyproject_content, wheel_path=wheel_path
            )
            self.w.workspace.upload(
                path=f"{source_path}/uv.lock",
                content=io.BytesIO(lock_content.encode("utf-8")),
                format=ImportFormat.AUTO,
                overwrite=True,
            )
            logger.info(
                "dao-ai source for app: dev wheel (uv.lock)",
                wheel=wheel_path.name,
                command=app_command,
            )

        # Pin the Apps build/runtime interpreter to Python 3.12, matching the
        # bundle path (``dao_ai.apps.bundle`` writes the same ``.python-version``).
        # Without it Apps selects its default interpreter (currently 3.14), for
        # which some pinned transitive deps ship no wheel — e.g. ``whenever`` (via
        # ``databricks-agents``) has cp312/cp313 wheels only — so ``uv sync`` falls
        # back to a source build that needs a Rust/C toolchain absent from the Apps
        # builder and fails with "Error installing packages".
        self.w.workspace.upload(
            path=f"{source_path}/.python-version",
            content=io.BytesIO(b"3.12\n"),
            format=ImportFormat.AUTO,
            overwrite=True,
        )

        # The chat UI (e2e-chatbot-app-next) is cloned and built at runtime
        # by start_app.py, matching the official Databricks agent template
        # pattern.  No pre-build or archive upload is needed here.

        # Generate and upload app.yaml with dynamically discovered resources
        from dao_ai.apps.resources import generate_app_yaml

        app_yaml_content: str = generate_app_yaml(
            config,
            command=app_command,
            include_resources=True,
            include_chat_ui=include_chat_ui,
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
        # Coerce workload_size → Apps compute_size. None (Small/Medium) leaves
        # the platform default (MEDIUM); Large/XLarge set the tier explicitly.
        # Sent as a raw string so XLARGE flows through even though the installed
        # SDK ComputeSize enum may not include it. Applied on CREATE only — the
        # PATCH update API rejects compute_size ("Compute size updates are not
        # supported in this update API"), so an existing app's size is left as
        # is on redeploy via this SDK path (change it in the UI, or tear down
        # and recreate).
        apps_compute_size: str | None = config.app.apps_compute_size()

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
            if apps_compute_size:
                app_body["compute_size"] = apps_compute_size
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
            # compute_size can't be changed on an existing app via the update
            # API (it rejects the field), so warn if the configured size no
            # longer matches — otherwise the resize would silently no-op.
            if apps_compute_size:
                current_size = getattr(existing_app.compute_size, "value", None) or (
                    existing_app.compute_size
                )
                if current_size and str(current_size) != apps_compute_size:
                    logger.warning(
                        "App compute_size cannot be changed on an existing app "
                        "via the update API; the app keeps its current size. To "
                        "resize, change it in the Databricks UI or tear down and "
                        "recreate the app.",
                        app_name=app_name,
                        current_size=str(current_size),
                        requested_size=apps_compute_size,
                    )
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
                    experiment: Experiment = self.get_or_create_experiment(
                        config, as_mcp=as_mcp
                    )
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

    def deploy_apps_agent(
        self,
        config: AppConfig,
        *,
        as_mcp: bool = False,
        development: bool | None = None,
        with_connection: bool = False,
    ) -> None:
        """Deploy the agent as a Databricks App via the SDK path.

        Serves the chat UI by default, or the dao-ai MCP server when
        ``as_mcp``. Both are the same Apps runtime and share ``_deploy_app``;
        only the container command, the extras, the chat-UI env vars, and the
        deployed App name differ.

        When ``with_connection`` (only valid with ``as_mcp``), a UC HTTP/MCP
        connection is created and registered with the Unity AI Gateway after the
        MCP app is up — see :meth:`register_mcp_connection`.
        """
        # Enforce the requires-as_mcp contract here too (not only in
        # ``deploy_agent``), so a direct caller of this method can't silently
        # drop a ``with_connection`` request against a chat app.
        if with_connection and not as_mcp:
            raise ValueError(
                "with_connection requires as_mcp (the UC connection targets the "
                "app's /mcp surface, served only by the MCP deployment)."
            )
        # Fail fast on a missing target schema BEFORE deploying, so a config
        # error surfaces immediately rather than after the app is already up.
        if with_connection:
            resolve_connection_registration(config.app)

        # Use the PRECISE resolver (not ``_or_all``): a deployed App pins the
        # minimal config-specific extras, even though the deploy runs inside a
        # notebook (where ``_or_all`` would short-circuit to every extra).
        from dao_ai._extras import expand_all, resolve_required_extras

        if as_mcp:
            self._deploy_app(
                config,
                app_command=["python", "-m", "dao_ai.mcp.server"],
                extras={
                    "mcp",
                    *expand_all(resolve_required_extras(config, target="mcp")),
                },
                include_chat_ui=False,
                as_mcp=True,
                development=development,
            )
            if with_connection:
                self.register_mcp_connection(config)
            return

        enable_chat_proxy: bool = (
            config.app.enable_chat_proxy
            if config.app and config.app.enable_chat_proxy is not None
            else True
        )
        entrypoint: str = (
            "dao_ai.apps.start_app" if enable_chat_proxy else "dao_ai.apps.server"
        )
        self._deploy_app(
            config,
            app_command=["python", "-m", entrypoint],
            extras=set(resolve_required_extras(config, target="apps")),
            include_chat_ui=enable_chat_proxy,
            as_mcp=False,
            development=development,
        )

    def register_mcp_connection(self, config: AppConfig) -> None:
        """Create a UC HTTP/MCP connection for a deployed MCP app and register it
        with the Unity AI Gateway as an MCP service.

        Runs after the ``mcp-<app>`` App is up (``deploy_apps_agent`` calls it on
        the SDK path; the CLI bundle path calls it after ``deploy_app_bundle``).
        The connection's auth mode is set by ``app.connection.on_behalf_of_user``:

        - **M2M (default, ``on_behalf_of_user: false``):** grant the app's own SP
          ``CAN_USE`` on the app (the connection authenticates as that SP), mint a
          fresh SP OAuth secret, and create an ``OAUTH_M2M`` HTTP connection at
          ``<app_url>/mcp``. Every caller reaches the app as the SP.
        - **U2M (``on_behalf_of_user: true``):** grant the forwarding users
          (``grant_principals``) ``CAN_USE`` on the app (they invoke as
          themselves), mint NO app-SP secret, and create an ``OAUTH_U2M_MAPPING``
          connection with ``authorization_endpoint`` + the ``oauth_client_id``
          (and ``oauth_client_secret``) of a DEDICATED account-level custom OAuth
          app integration. The app's own ``oauth2_app_client_id`` cannot be used
          — its redirect allowlist is pinned to the app URL and rejects the
          connection callback ``/login/oauth/http.html`` (GAIA-435). Each user
          completes a one-time OAuth consent before the connection resolves their
          token.

        Both modes then register the MCP service under the target schema and grant
        ``USE_CONNECTION`` + ``EXECUTE``.

        Idempotent. Error handling is split by intent: creating the connection
        and registering the service are the explicit deliverable of
        ``--with-connection``, so they FAIL LOUD — a silent success would leave
        Genie One unable to connect with no signal why. Only the final
        ``USE_CONNECTION`` / ``EXECUTE`` grants are best-effort (a deployer may
        lack GRANT rights), matching the trace-permission grant tail.
        """
        from databricks.sdk.service.apps import (
            App,
            AppAccessControlRequest,
            AppPermissionLevel,
        )

        reg = resolve_connection_registration(config.app)
        app_name: str = app_name_for(config.app.name, as_mcp=True)
        conn: str = reg.name or connection_name_for(config.app.name)
        svc: str = reg.service_name or mcp_service_name_for(config.app.name)
        catalog: str = str(reg.schema_model.catalog_name)
        schema: str = str(reg.schema_model.schema_name)
        service_fqn: str = f"{catalog}.{schema}.{svc}"

        logger.info(
            "Registering UC MCP connection",
            app_name=app_name,
            connection=conn,
            mcp_service=service_fqn,
        )

        app: App = self.w.apps.get(name=app_name)
        client_id: Optional[str] = app.service_principal_client_id
        sp_id: Optional[int] = app.service_principal_id
        app_url: str = (app.url or "").rstrip("/")
        # The connection needs the app's URL and service principal. If the App
        # resource hasn't yielded them yet, FAIL rather than create a connection
        # with an empty host — idempotency would then skip the fix on re-deploy,
        # leaving a permanently broken connection.
        if not client_id or sp_id is None or not app_url:
            raise RuntimeError(
                f"App '{app_name}' is not ready for connection registration "
                f"(url/service-principal unresolved); re-run once it is ACTIVE."
            )
        host: str = self.w.config.host.rstrip("/")

        # 1. Authorize invocation of the app. Who needs CAN_USE depends on the
        # connection's auth mode:
        #   - M2M (default): the connection authenticates as the app's OWN
        #     service principal, so that SP needs CAN_USE (without it the app
        #     rejects the connection and Genie One fails to connect).
        #   - U2M (on_behalf_of_user): the connection forwards the *calling
        #     user's* identity, so the END USERS (grant_principals) — not the
        #     app SP — must have CAN_USE. Each user also completes a one-time
        #     OAuth consent before the connection resolves their token.
        # Merges, idempotent.
        if reg.on_behalf_of_user:
            acl: list[AppAccessControlRequest] = [
                _app_can_use_acl_entry(p) for p in reg.grant_principals
            ]
            if acl:
                self.w.apps.update_permissions(app_name, access_control_list=acl)
            else:
                # U2M forwards the caller's identity, so users need CAN_USE on the
                # app — an empty grant_principals leaves the connection unusable by
                # everyone. Warn rather than silently create a dead connection.
                logger.warning(
                    "U2M connection has no grant_principals to grant CAN_USE on "
                    "the app — no user will be able to invoke it until granted.",
                    app_name=app_name,
                    connection=conn,
                )
        else:
            self.w.apps.update_permissions(
                app_name,
                access_control_list=[
                    AppAccessControlRequest(
                        service_principal_name=client_id,
                        permission_level=AppPermissionLevel.CAN_USE,
                    )
                ],
            )

        # 2. Create the HTTP (MCP) connection if absent; RECONCILE it to the
        # current deploy config if present (instead of leaving it stale — these
        # connections are dao-ai-managed). Existence is an O(1) ``get`` (NotFound
        # == absent), not a full-metastore ``list``. Mint a fresh SP secret for the
        # M2M OAuth flow only when actually writing (create, or update on drift).
        from databricks.sdk.errors import NotFound

        try:
            existing_conn: Optional[Any] = self.w.connections.get(name=conn)
        except NotFound:
            existing_conn = None
        if existing_conn is not None:
            self._reconcile_mcp_connection(
                existing=existing_conn,
                reg=reg,
                conn=conn,
                app_url=app_url,
                host=host,
                client_id=client_id,
                sp_id=sp_id,
            )
        elif reg.on_behalf_of_user:
            # U2M (OAUTH_U2M_MAPPING): the connection forwards the *calling
            # user's* Databricks identity via a DEDICATED custom OAuth app, so the
            # app's on-behalf-of-user tools run as the end user (not the app SP).
            # No app-SP secret is minted — the per-user token comes from a one-time
            # OAuth consent. `authorization_endpoint` is what makes UC classify
            # the connection as U2M. Each end user must (a) have CAN_USE on the
            # app and (b) complete the connection's OAuth consent once.
            #
            # The client MUST be a dedicated account-level custom OAuth app
            # integration whose redirect_urls include the connection callback
            # `<host>/login/oauth/http.html`. The app's own auto-generated
            # `oauth2_app_client_id` cannot be used — its redirect allowlist is
            # pinned to the app URL, so the consent redirect is rejected with
            # "redirect_uri ... not registered for OAuth application" (GAIA-435).
            #
            # Validate the RESOLVED value, not just that the field is set: an
            # `oauth_client_id: {env: …}` AnyVariable passes the model validator
            # (field not None) but can resolve to None/"" when the env/secret is
            # missing — which would otherwise create a permanently broken
            # connection with client_id="None" (and the idempotent get→create
            # guard would skip fixing it on re-deploy).
            resolved = value_of(reg.oauth_client_id)
            oauth_client_id: str = str(resolved).strip() if resolved else ""
            if not oauth_client_id or oauth_client_id == "None":
                raise RuntimeError(
                    "app.connection.on_behalf_of_user is true but oauth_client_id "
                    f"resolved to empty for connection '{conn}'. Provide the "
                    "dedicated custom OAuth app's client id (or ensure the env var "
                    "/ secret it references is set at deploy time). The app's own "
                    "oauth2_app_client_id cannot be used for a U2M connection."
                )
            # U2M needs a refresh token so a user's consent outlives the access
            # token (~1h) — otherwise the connection stops working an hour after
            # each consent. Ensure `offline_access` is requested (it's a no-op for
            # M2M, so it lives only on this U2M path, not the shared default).
            u2m_scope: str = reg.oauth_scope
            if "offline_access" not in u2m_scope.split():
                u2m_scope = f"{u2m_scope} offline_access".strip()
            options: dict[str, str] = {
                "host": app_url,
                "port": "443",
                "base_path": "/mcp",
                "is_mcp_connection": "true",
                "oauth_scope": u2m_scope,
                "authorization_endpoint": f"{host}/oidc/v1/authorize",
                "token_endpoint": f"{host}/oidc/v1/token",
                "client_id": oauth_client_id,
            }
            # Confidential dedicated OAuth apps also carry a secret, which the
            # connection needs to complete the authorization-code exchange.
            resolved_secret = value_of(reg.oauth_client_secret)
            oauth_client_secret: str = (
                str(resolved_secret).strip() if resolved_secret else ""
            )
            if oauth_client_secret and oauth_client_secret != "None":
                options["client_secret"] = oauth_client_secret
            try:
                self.w.connections.create(
                    name=conn,
                    connection_type=ConnectionType.HTTP,
                    options=options,
                )
            except Exception as e:
                # ``options`` may carry the dedicated OAuth app's client_secret, and
                # the SDK exception can echo the request body — so REDACT the secret
                # value but keep the message, so a real cause (e.g. UC rejecting an
                # invalid oauth_scope at token-exchange) still surfaces instead of an
                # opaque failure.
                detail = str(e)
                if oauth_client_secret and oauth_client_secret != "None":
                    detail = detail.replace(oauth_client_secret, "***")
                raise RuntimeError(
                    f"Failed to create UC MCP connection '{conn}': {detail}"
                ) from None
            logger.info(
                "Created UC MCP connection (U2M / on-behalf-of-user)",
                name=conn,
                oauth_client_id=oauth_client_id,
            )
        else:
            minted = self.w.service_principal_secrets_proxy.create(
                service_principal_id=sp_id
            )
            try:
                self.w.connections.create(
                    name=conn,
                    connection_type=ConnectionType.HTTP,
                    options={
                        "host": app_url,
                        "port": "443",
                        "base_path": "/mcp",
                        "is_mcp_connection": "true",
                        "oauth_scope": reg.oauth_scope,
                        "token_endpoint": f"{host}/oidc/v1/token",
                        "client_id": client_id,
                        "client_secret": minted.secret,
                    },
                )
            except Exception as e:
                # Don't orphan the secret we just minted on the app SP.
                try:
                    self.w.service_principal_secrets_proxy.delete(
                        service_principal_id=sp_id, secret_id=str(minted.id)
                    )
                except Exception:
                    pass
                # The payload carries the freshly minted client_secret; REDACT its
                # value from the error (the SDK may echo the request body) but keep
                # the message so a real cause (e.g. invalid oauth_scope) surfaces.
                detail = (
                    str(e).replace(minted.secret, "***") if minted.secret else str(e)
                )
                raise RuntimeError(
                    f"Failed to create UC MCP connection '{conn}': {detail}"
                ) from None
            logger.info("Created UC MCP connection", name=conn)

        # 3. Register the MCP service (parent + id are query params; the
        # connection ref needs the ``connections/`` prefix). Idempotent — the
        # live API returns ``name`` as ``mcp-services/<fqn>``, so match the fqn
        # at a path boundary (``== fqn`` or ``.../<fqn>``), NOT a bare
        # ``endswith(fqn)`` which would false-match a catalog whose name is a
        # suffix of ours (``xmain.genie.svc`` vs ``main.genie.svc``).
        services: dict[str, Any] = (
            self.w.api_client.do(
                "GET",
                "/api/2.1/unity-catalog/mcp-services",
                query={"parent": f"schemas/{catalog}.{schema}"},
            )
            or {}
        )
        exists: bool = any(
            (name := str(s.get("name", ""))) == service_fqn
            or name.endswith(f"/{service_fqn}")
            for s in services.get("mcp_services", [])
        )
        if exists:
            logger.info("MCP service already exists; skipping", name=service_fqn)
        else:
            try:
                self.w.api_client.do(
                    "POST",
                    "/api/2.1/unity-catalog/mcp-services",
                    query={
                        "parent": f"schemas/{catalog}.{schema}",
                        "mcp_service_id": svc,
                    },
                    body={
                        "config": {
                            "source_connection": {"name": f"connections/{conn}"},
                            "include_tool_selectors": ["*"],
                        }
                    },
                )
                logger.info("Registered MCP service", name=service_fqn)
            except Exception as e:
                # Belt-and-suspenders on the idempotency check above: if the
                # GET-based check ever misses an already-registered service (API
                # shape drift), treat an "already exists" conflict as success
                # rather than failing the re-deploy. Match "already exists"
                # SPECIFICALLY — a bare "exist" substring would also swallow a
                # real "... does not exist" failure and report a false success.
                msg = str(e).lower()
                if "already exists" in msg or "already_exists" in msg:
                    logger.info(
                        "MCP service already exists; skipping", name=service_fqn
                    )
                else:
                    raise

        # 4. Grant access: USE_CONNECTION on the connection, EXECUTE on the MCP
        # service, for each configured principal. Best-effort and PER-GRANT — a
        # deployer without GRANT rights degrades to a warning (the connection +
        # service are already created), and one principal's failure must not
        # skip the rest. Collect failures so the warning names exactly what
        # didn't land rather than implying every grant failed.
        failed: list[str] = []
        for principal in reg.grant_principals:
            for securable, path, priv in (
                ("connection", f"connection/{conn}", "USE_CONNECTION"),
                ("mcp_service", f"mcp_service/{service_fqn}", "EXECUTE"),
            ):
                try:
                    self.w.api_client.do(
                        "PATCH",
                        f"/api/2.1/unity-catalog/permissions/{path}",
                        body={"changes": [{"principal": principal, "add": [priv]}]},
                    )
                except Exception as e:
                    failed.append(f"{priv} for '{principal}' on {securable} ({e})")
        if failed:
            logger.warning(
                "Created the UC MCP connection + service, but some grants failed "
                "— apply them manually.",
                connection=conn,
                mcp_service=service_fqn,
                failed_grants=failed,
            )
        else:
            logger.info(
                "Granted UC MCP connection access",
                principals=reg.grant_principals,
                connection=conn,
                mcp_service=service_fqn,
            )

    def _reconcile_mcp_connection(
        self,
        *,
        existing: Any,
        reg: Any,
        conn: str,
        app_url: str,
        host: str,
        client_id: Optional[str],
        sp_id: Optional[int],
    ) -> None:
        """Reconcile an EXISTING dao-ai UC MCP connection to the current deploy
        config, instead of the historical skip-if-exists (these connections are
        dao-ai-managed, so a redeploy should keep them in sync).

        - ``credential_type`` is fixed at creation, so an auth-mode change
          (M2M<->U2M) CANNOT be applied in place — fail loud with guidance to drop
          + recreate (the only safe path; it also resets the Genie One binding and
          any per-user U2M consents, so it isn't done automatically).
        - Otherwise compare the managed, config-driven option fields (host,
          base_path, oauth_scope, is_mcp_connection). On drift, update the
          connection. A UC connection update may REPLACE the options map, so the
          full set is sent, including a secret: M2M mints a fresh SP secret (only
          when actually writing — no churn on a no-drift redeploy); U2M reuses the
          dedicated OAuth app secret from config. The secret is scrubbed from any
          error and a minted-but-unused secret is not orphaned.
        """
        intended_ct: str = (
            "OAUTH_U2M_MAPPING" if reg.on_behalf_of_user else "OAUTH_M2M"
        )
        live_ct_raw = getattr(existing, "credential_type", None)
        live_ct: str = str(getattr(live_ct_raw, "value", live_ct_raw) or "")
        if live_ct and live_ct != intended_ct:
            raise RuntimeError(
                f"UC MCP connection '{conn}' exists as {live_ct}, but the deploy "
                f"config requires {intended_ct} "
                f"(app.connection.on_behalf_of_user={reg.on_behalf_of_user}). A "
                "connection's credential type cannot be changed in place — drop "
                "the connection and its MCP service, then redeploy with "
                "--with-connection to recreate it in the new mode (recreating also "
                "resets the Genie One binding and any per-user consents)."
            )

        # U2M requests offline_access (refresh token) so a consent outlives the
        # ~1h access token; it's a no-op for M2M.
        if reg.on_behalf_of_user:
            desired_scope: str = reg.oauth_scope
            if "offline_access" not in desired_scope.split():
                desired_scope = f"{desired_scope} offline_access".strip()
        else:
            desired_scope = reg.oauth_scope

        managed: dict[str, str] = {
            "host": app_url,
            "base_path": "/mcp",
            "is_mcp_connection": "true",
            "oauth_scope": desired_scope,
        }
        live_opts: dict[str, Any] = getattr(existing, "options", None) or {}
        drift: list[str] = sorted(
            k for k, v in managed.items() if str(live_opts.get(k)) != str(v)
        )
        if not drift:
            logger.info("UC connection matches config; leaving as-is", name=conn)
            return

        # Rebuild the FULL options (update may replace, not merge) + a secret.
        if reg.on_behalf_of_user:
            resolved = value_of(reg.oauth_client_id)
            oauth_client_id: str = str(resolved).strip() if resolved else ""
            if not oauth_client_id or oauth_client_id == "None":
                raise RuntimeError(
                    "app.connection.on_behalf_of_user is true but oauth_client_id "
                    f"resolved to empty for connection '{conn}'. Provide the "
                    "dedicated custom OAuth app's client id (or ensure the env var "
                    "/ secret it references is set at deploy time)."
                )
            options: dict[str, str] = {
                "host": app_url,
                "port": "443",
                "base_path": "/mcp",
                "is_mcp_connection": "true",
                "oauth_scope": desired_scope,
                "authorization_endpoint": f"{host}/oidc/v1/authorize",
                "token_endpoint": f"{host}/oidc/v1/token",
                "client_id": oauth_client_id,
            }
            resolved_secret = value_of(reg.oauth_client_secret)
            oauth_client_secret: str = (
                str(resolved_secret).strip() if resolved_secret else ""
            )
            if oauth_client_secret and oauth_client_secret != "None":
                options["client_secret"] = oauth_client_secret
            try:
                self.w.connections.update(name=conn, options=options)
            except Exception as e:
                # REDACT the dedicated OAuth app secret (if any) but keep the
                # message so real causes (e.g. invalid oauth_scope) surface.
                detail = str(e)
                if oauth_client_secret and oauth_client_secret != "None":
                    detail = detail.replace(oauth_client_secret, "***")
                raise RuntimeError(
                    f"Failed to update UC MCP connection '{conn}': {detail}"
                ) from None
        else:
            minted = self.w.service_principal_secrets_proxy.create(
                service_principal_id=sp_id
            )
            options = {
                "host": app_url,
                "port": "443",
                "base_path": "/mcp",
                "is_mcp_connection": "true",
                "oauth_scope": desired_scope,
                "token_endpoint": f"{host}/oidc/v1/token",
                "client_id": client_id,
                "client_secret": minted.secret,
            }
            try:
                self.w.connections.update(name=conn, options=options)
            except Exception as e:
                # Don't orphan the freshly minted secret. REDACT its value from the
                # error but keep the message so real causes (e.g. UC rejecting an
                # invalid oauth_scope at token-exchange) surface.
                try:
                    self.w.service_principal_secrets_proxy.delete(
                        service_principal_id=sp_id, secret_id=str(minted.id)
                    )
                except Exception:
                    pass
                detail = (
                    str(e).replace(minted.secret, "***") if minted.secret else str(e)
                )
                raise RuntimeError(
                    f"Failed to update UC MCP connection '{conn}': {detail}"
                ) from None
        logger.info(
            "Reconciled UC MCP connection to current config",
            name=conn,
            changed=drift,
        )

    def unregister_mcp_connection(self, config: AppConfig) -> None:
        """Teardown counterpart to :meth:`register_mcp_connection`: delete the
        Unity AI Gateway MCP service and the UC connection created for an MCP app.

        Neither is a DAB resource, so ``bundle destroy`` / ``apps.delete`` leave
        them stranded — this removes them so ``down`` fully tears the deployment
        down. Best-effort throughout: a missing service/connection is a no-op
        (so it is safe to call on any ``--as-mcp`` down, even when
        ``--with-connection`` was never used), and it works after the app is
        already gone. It deletes the connection ONLY when it is the MCP
        connection dao-ai created for THIS app (``is_mcp_connection`` + a host
        bound to the app name), never a hand-made connection sharing the name.
        """
        try:
            reg = resolve_connection_registration(config.app)
        except ValueError:
            # No derivable schema -> this feature could not have registered
            # anything for this config; nothing to tear down.
            return
        app_name: str = app_name_for(config.app.name, as_mcp=True)
        conn: str = reg.name or connection_name_for(config.app.name)
        svc: str = reg.service_name or mcp_service_name_for(config.app.name)
        catalog: str = str(reg.schema_model.catalog_name)
        schema: str = str(reg.schema_model.schema_name)
        service_fqn: str = f"{catalog}.{schema}.{svc}"

        # 1. Delete the MCP service first (it references the connection).
        try:
            self.w.api_client.do(
                "DELETE", f"/api/2.1/unity-catalog/mcp-services/{service_fqn}"
            )
            logger.info("Deleted MCP service", name=service_fqn)
        except Exception as e:
            if "exist" in str(e).lower() or "not found" in str(e).lower():
                logger.info(
                    "MCP service not found; nothing to delete", name=service_fqn
                )
            else:
                logger.warning(
                    "Could not delete MCP service; remove it manually with "
                    "`databricks api delete /api/2.1/unity-catalog/mcp-services/"
                    f"{service_fqn}`.",
                    error=str(e),
                )

        # 2. Delete the connection — only if it is the MCP connection THIS app
        # created, so a hand-made connection sharing the name is never nuked.
        from databricks.sdk.errors import NotFound

        try:
            try:
                existing = self.w.connections.get(name=conn)
            except NotFound:
                logger.info("UC connection not found; nothing to delete", name=conn)
                return
            options: dict[str, str] = dict(existing.options or {})
            host: str = options.get("host", "")
            # Ownership check. Authoritative when the app still exists (compare
            # the connection host to the live app URL); if the app is already
            # gone (the agent path deletes it before this runs), fall back to the
            # ``is_mcp_connection`` marker — the connection name is already
            # app-derived, so this only guards against a hand-made non-MCP
            # connection that happens to share the name.
            try:
                app_url: str = (self.w.apps.get(name=app_name).url or "").rstrip("/")
                is_ours: bool = bool(app_url) and host.rstrip("/") == app_url
            except NotFound:
                is_ours = options.get("is_mcp_connection") == "true"
            if is_ours:
                self.w.connections.delete(name=conn)
                logger.info("Deleted UC MCP connection", name=conn)
            else:
                logger.warning(
                    "Leaving UC connection in place — it is not the MCP "
                    "connection dao-ai created for this app; delete manually if "
                    "intended.",
                    name=conn,
                )
        except Exception as e:
            logger.warning(
                "Could not delete UC connection; remove it manually with "
                f"`databricks connections delete {conn}`.",
                error=str(e),
            )

    def deploy_agent(
        self,
        config: AppConfig,
        mode: ServingMode = ServingMode.MODEL_SERVING,
        development: bool | None = None,
        as_mcp: bool = False,
        with_connection: bool = False,
    ) -> None:
        """
        Deploy agent using the specified serving platform.

        This is the main deployment method that routes to the appropriate
        deployment implementation based on the serving platform.

        Args:
            config: The AppConfig containing deployment configuration
            mode: The serving platform (MODEL_SERVING or APPS)
            development: When True, ship local dao-ai source/wheel; when False,
                the published PyPI package; when None, auto-detect from the
                install type. Only the Apps path consumes this today.
            as_mcp: Serve the agent over MCP instead of the chat UI. Requires
                ``mode=APPS`` — MCP runs on the Apps runtime, so there is no
                Model Serving equivalent.
            with_connection: After deploying the MCP server, create a UC HTTP/MCP
                connection and register it with the Unity AI Gateway. Requires
                ``mode=APPS`` and ``as_mcp`` (the connection targets ``/mcp``).
        """
        if with_connection and not as_mcp:
            # A UC MCP connection points at the ``/mcp`` surface, which only the
            # ``mcp-<app>`` deployment serves. Refuse rather than register a
            # connection that would 404.
            raise ValueError(
                "with_connection requires as_mcp (the UC connection targets the "
                "app's /mcp surface, served only by the MCP deployment)."
            )
        if mode == ServingMode.MODEL_SERVING:
            if as_mcp:
                raise ValueError(
                    "as_mcp requires mode=APPS (MCP is served on the Databricks "
                    "Apps runtime); got mode=MODEL_SERVING"
                )
            self.deploy_model_serving_agent(config)
        elif mode == ServingMode.APPS:
            self.deploy_apps_agent(
                config,
                as_mcp=as_mcp,
                development=development,
                with_connection=with_connection,
            )
        else:
            raise ValueError(f"Unknown serving mode: {mode}")

    def create_catalog(self, schema: SchemaModel) -> CatalogInfo:
        catalog_info: CatalogInfo
        try:
            catalog_info = self.w.catalogs.get(name=schema.catalog_name)
        except NotFound:
            logger.info("Creating catalog", catalog_name=schema.catalog_name)
            catalog_info = self.w.catalogs.create(name=schema.catalog_name)
        return catalog_info

    def create_experiment(self, experiment: ExperimentModel) -> Experiment:
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
            ddl_path: Path = dataset.resolve_asset_path(ddl)
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
            data_path: Path = dataset.resolve_asset_path(data)
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
                data_path = data_path.resolve()
                logger.trace("Data path resolved", path=str(data_path))

                # csv/parquet/orc/delta go to Spark's distributed reader below.
                # Parquet was moved off pandas because pd.read_parquet on the
                # driver OOMs on serverless for text-heavy datasets (e.g.
                # hardware_store products.parquet: 17 MB on disk → 45 MB in
                # pandas). csv carries the same OOM exposure, and its
                # read_options are authored in Spark's vocabulary (``header:
                # true``) — which pd.read_csv rejects — so Spark is also the
                # correct reader. json/excel stay on pandas: Spark's json
                # reader defaults to JSON Lines (not the records array
                # pd.read_json expects) and there is no serverless Spark excel
                # reader.
                pandas_readers: dict[str, Callable[..., pd.DataFrame]] = {
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
                    # Spark executors cannot read workspace files (/Workspace/...)
                    # on serverless compute, so a config-relative asset staged
                    # into the bundle is unreadable here. Stage it to a UC volume
                    # — FUSE-writable on the driver, readable by executors — and
                    # point Spark at the /Volumes path. A path already on a
                    # volume passes through untouched.
                    load_path: str = self._resolve_spark_read_path(dataset, data_path)
                    df = (
                        spark.read.format(format)
                        .options(**read_options)
                        .load(
                            load_path,
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

    def _resolve_spark_read_path(self, dataset: DatasetModel, data_path: Path) -> str:
        """Return a path Spark executors can read on serverless compute.

        Serverless Spark cannot read workspace files (``/Workspace/...``), so a
        config-relative asset staged into the deployed bundle is unreadable via
        ``spark.read``. A ``data:`` value already on a UC volume
        (``/Volumes/...``) is executor-readable and passes through unchanged.
        Anything else is a driver-local/workspace file: copy it into a managed
        UC volume in the dataset's own target schema and return the ``/Volumes``
        path. (Cloud/``dbfs:`` URIs are not a case here — a string ``data:``
        value is wrapped in ``Path`` by ``resolve_asset_path``, which only
        preserves ``/Volumes`` and local absolute paths.)

        The copy uses the volume's FUSE mount (writable on the driver) and the
        volume is idempotent — the same destination is overwritten on each run.
        """
        raw: str = str(data_path)
        if raw.startswith("/Volumes/"):
            return raw

        schema: SchemaModel = self._dataset_staging_schema(dataset)
        volume: VolumeModel = VolumeModel(schema=schema, name="dao_ai_staging")
        self.create_volume(volume)

        dest: str = f"/Volumes/{volume.full_name.replace('.', '/')}/{data_path.name}"
        logger.info("Staging asset to volume for Spark read", src=raw, dest=dest)
        shutil.copy2(raw, dest)
        return dest

    def _dataset_staging_schema(self, dataset: DatasetModel) -> SchemaModel:
        """Derive the catalog+schema to stage a dataset's asset into.

        Prefers the target table's ``schema_model``. Falls back to parsing a
        fully-qualified ``catalog.schema.table`` name when no schema reference
        is set (both forms are permitted by ``TableModel``).
        """
        if dataset.table is not None and dataset.table.schema_model is not None:
            return dataset.table.schema_model

        full_name: str = dataset.table.full_name if dataset.table else ""
        parts: list[str] = full_name.split(".")
        if len(parts) < 3:
            raise ValueError(
                "Cannot derive a staging schema for dataset asset: table must "
                "have a schema reference or a fully-qualified "
                f"catalog.schema.table name (got {full_name!r})."
            )
        return SchemaModel(catalog_name=parts[0], schema_name=parts[1])

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

        endpoint_name: str = vector_store.endpoint.name
        index_name: str = vector_store.index.full_name
        source_table: str = vector_store.source_table.full_name

        if not index_exists(self.vsc, endpoint_name, index_name):
            logger.info(
                "Creating vector search index",
                index_name=index_name,
                endpoint_name=endpoint_name,
            )
            index = self._create_delta_sync_index(vector_store)
        else:
            index = self.vsc.get_index(endpoint_name, index_name)
            details = _describe_index_safe(index)
            current_state = str(
                (details.get("status") or {}).get("detailed_state", "UNKNOWN")
            )
            is_delta_sync = _index_is_delta_sync(details)
            logger.info(
                "Vector search index already exists — evaluating for stale checkpoint",
                index_name=index_name,
                detailed_state=current_state,
                index_kind="delta_sync" if is_delta_sync else "direct_access",
            )
            if _index_is_stale(index, details, source_table):
                # Stale checkpoint: the source table was recreated / its history
                # aged out, so the index's streaming checkpoint no longer
                # resolves and a plain sync would fail-loop forever. Drop and
                # recreate — the manual recovery, automated.
                logger.warning(
                    "Dropping + recreating vector search index to clear a stale "
                    "Delta-Sync checkpoint (re-embeds the full source table)",
                    index_name=index_name,
                    source_table=source_table,
                    detailed_state=current_state,
                )
                self.vsc.delete_index(endpoint_name, index_name)
                _wait_until_index_absent(self.vsc, endpoint_name, index_name)
                logger.info(
                    "Stale index deleted — recreating from source",
                    index_name=index_name,
                    source_table=source_table,
                )
                index = self._create_delta_sync_index(vector_store)
            else:
                # Healthy (or a Direct-Access index, which has no checkpoint to
                # go stale) — a normal incremental sync suffices.
                logger.info(
                    "Vector search index healthy — triggering incremental sync",
                    index_name=index_name,
                    detailed_state=current_state,
                )
                # Wait for the pipeline to be idle (not merely queryable) before
                # issuing sync. wait_for_updates=False returns while an update is
                # still in flight (detailed_state merely *contains* ONLINE), and
                # sync() 400s ("Pipeline is in state RUNNING") against a running
                # pipeline. wait_for_updates=True blocks until
                # ONLINE_NO_PENDING_UPDATE. Bounded like the final readiness wait.
                index.wait_until_ready(
                    verbose=True,
                    wait_for_updates=True,
                    timeout=timedelta(seconds=_VS_INDEX_READY_TIMEOUT_SECONDS),
                )
                # Defensive: tolerate the residual TOCTOU window if the pipeline
                # re-enters RUNNING between the wait and the sync.
                _sync_when_pipeline_idle(index, index_name)

        # create_delta_sync_index_and_wait and index.sync() return before the
        # underlying data sync completes. wait_for_updates=True blocks until the
        # index is fully populated (ONLINE_NO_PENDING_UPDATE). Bounded so a
        # still-stale index fails fast with a diagnostic instead of hanging.
        # The SDK raises a bare Exception on timeout OR on an OFFLINE/failed
        # state — both mean the index never came online (a still-stale
        # checkpoint or unrecoverable Delta history, e.g. VACUUM / retention
        # exceeded). Re-raise as an actionable error instead of a 24h hang.
        try:
            index.wait_until_ready(
                verbose=True,
                wait_for_updates=True,
                timeout=timedelta(seconds=_VS_INDEX_READY_TIMEOUT_SECONDS),
            )
        except Exception as exc:  # noqa: BLE001 — normalize SDK's bare Exception
            final_state = str(
                (_describe_index_safe(index).get("status") or {}).get(
                    "detailed_state", "UNKNOWN"
                )
            )
            # A full initial snapshot re-embeds the entire source table and can
            # outlast the timeout above while still making progress. Don't fail a
            # recovery that is working — switch to bounding on stalled progress.
            if any(s in final_state for s in _VS_SNAPSHOT_STATES):
                logger.info(
                    "Readiness timeout hit while an initial snapshot is still "
                    "progressing — waiting on row progress instead of elapsed time",
                    index_name=index_name,
                    detailed_state=final_state,
                )
                final_state = _wait_for_initial_snapshot(index, index_name)
                if "ONLINE" in final_state and "FAILED" not in final_state:
                    logger.success(
                        "Vector search index ready after initial snapshot",
                        index_name=index_name,
                        source_table=source_table,
                        detailed_state=final_state,
                    )
                    return
            logger.error(
                "Vector search index failed to reach ONLINE",
                index_name=index_name,
                source_table=source_table,
                final_state=final_state,
                timeout_seconds=_VS_INDEX_READY_TIMEOUT_SECONDS,
                error=str(exc),
            )
            raise RuntimeError(
                f"Vector search index {index_name} did not reach ONLINE "
                f"(last state: {final_state}). Provisioning already dropped and "
                f"recreated the index, so recreating it again will not help. The "
                f"source table's Delta history is likely unrecoverable — its "
                f"change data aged out of "
                f"delta.deletedFileRetentionDuration (default 168h), or the table "
                f"was replaced. Rewrite {source_table} to give it fresh history, "
                f"and raise its retention to stop this recurring:\n"
                f"  ALTER TABLE {source_table} SET TBLPROPERTIES "
                f"('delta.deletedFileRetentionDuration' = 'interval 30 days');"
            ) from exc

        logger.success(
            "Vector search index ready",
            index_name=index_name,
            source_table=source_table,
        )

    def _create_delta_sync_index(
        self, vector_store: VectorStoreModel
    ) -> VectorSearchIndex:
        """Create the Delta-Sync index and stamp the source table's Delta GUID
        onto it so a later run can detect a recreated source (stale checkpoint).

        Shared by the initial-create and stale-recreate paths so their index
        parameters can never drift apart.
        """
        endpoint_name: str = vector_store.endpoint.name
        index_name: str = vector_store.index.full_name
        source_table: str = vector_store.source_table.full_name

        source_uuid: str | None = _source_table_delta_uuid(source_table)
        create_kwargs: dict[str, Any] = dict(
            endpoint_name=endpoint_name,
            index_name=index_name,
            source_table_name=source_table,
            pipeline_type="TRIGGERED",
            primary_key=vector_store.primary_key,
            embedding_source_column=vector_store.embedding_source_column,
            embedding_model_endpoint_name=vector_store.embedding_model.name,
            columns_to_sync=vector_store.columns,
        )
        stamped_uuid = False
        if source_uuid:
            # Best-effort: only pass tags if the SDK accepts them, so a signature
            # change can't break index creation.
            try:
                import inspect

                if (
                    "custom_tags"
                    in inspect.signature(
                        self.vsc.create_delta_sync_index_and_wait
                    ).parameters
                ):
                    create_kwargs["custom_tags"] = {
                        _index_source_uuid_key(): source_uuid
                    }
                    stamped_uuid = True
            except Exception:  # noqa: BLE001
                pass

        logger.info(
            "Creating Delta-Sync vector index",
            index_name=index_name,
            source_table=source_table,
            embedding_model=vector_store.embedding_model.name,
            embedding_source_column=vector_store.embedding_source_column,
            source_delta_uuid=source_uuid or "unknown",
            uuid_stamped=stamped_uuid,
        )
        self.vsc.create_delta_sync_index_and_wait(**create_kwargs)
        return self.vsc.get_index(endpoint_name, index_name)

    def get_vector_index(self, vector_store: VectorStoreModel) -> VectorSearchIndex:
        # Endpoint discovery is deferred out of config parse (serving-safe). A
        # caller can reach here with an unresolved endpoint — e.g. a retriever's
        # deep-copied vector_store that never went through create(), which only
        # populates the resources.vector_stores instance. Resolve on demand and
        # stamp it so refresh()/subsequent calls are consistent, falling back to
        # an endpoint-less lookup (the SDK's get_index accepts endpoint_name=None
        # and resolves the index by full name alone).
        endpoint_name: str | None = (
            vector_store.endpoint.name if vector_store.endpoint is not None else None
        )
        if endpoint_name is None and vector_store.index is not None:
            # Best-effort discovery. If it fails (e.g. an unauthenticated
            # VectorSearchClient), fall through to the endpoint-less lookup
            # rather than propagating — get_index resolves the index by full
            # name alone.
            try:
                endpoint_name = self.find_endpoint_for_index(vector_store.index)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Endpoint discovery failed; using endpoint-less index lookup",
                    index_name=vector_store.index.full_name,
                    error=f"{type(e).__name__}: {e}",
                )
                endpoint_name = None
            if endpoint_name is not None:
                from dao_ai.config import VectorSearchEndpoint

                vector_store.endpoint = VectorSearchEndpoint(name=endpoint_name)

        index: VectorSearchIndex = self.vsc.get_index(
            endpoint_name, vector_store.index.full_name
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
        ddl_path: Path = unity_catalog_function.resolve_asset_path(
            unity_catalog_function.ddl
        )
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

        Runs as ``self.w`` — the caller's identity — for the same reason as
        :meth:`create_lakebase_autoscaling_role`: creating a Database project is
        a control-plane operation, and the database's own service principal has
        no standing to create the project it is about to be granted a role on.
        Runtime connections continue to authenticate as the configured SP via
        ``database.workspace_client``.
        """
        import time

        from databricks.sdk.service.postgres import (
            Project,
            ProjectDefaultEndpointSettings,
            ProjectSpec,
        )

        workspace_client: WorkspaceClient = self.w
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

    def create_lakebase_autoscaling_role(
        self,
        database: DatabaseModel,
        *,
        client_id: str | None = None,
    ) -> None:
        """
        Create a role for a service principal on an autoscaling Lakebase project.

        Roles are created at the branch level in the autoscaling Postgres API.

        Two distinct identities are in play here, and conflating them is a bug:

        * The **caller** — every Postgres control-plane call below
          (``list_branches`` / ``list_roles`` / ``create_role``) requires
          ``Can Manage`` on the Database project. A service principal cannot
          create its own role, so these run as ``self.w``: the identity the
          caller supplied (e.g. the admin profile behind ``dao-ai -p <profile>``).
        * The **role subject** — ``postgres_role`` is the service principal the
          role is created *for*, i.e. the identity the agent later connects to
          Postgres as. That remains ``client_id``.

        Args:
            database: The Lakebase ``DatabaseModel`` to create the role on.
            client_id: Service-principal client id to create the role for. When
                given it is authoritative — used by ``dao-ai sp provision`` to
                pass a freshly minted SP's id, so provisioning completes in one
                pass instead of requiring the secret scope to be populated
                first. When omitted, falls back to ``database.client_id``.
        """
        from databricks.sdk.service.postgres import (
            Role,
            RoleIdentityType,
            RoleMembershipRole,
            RoleRoleSpec,
        )

        from dao_ai.config import value_of

        if client_id is None:
            if not database.client_id:
                logger.warning(
                    "client_id required to create autoscaling role",
                    project=database.project,
                )
                return

            client_id = value_of(database.client_id)
            if not client_id:
                logger.warning(
                    "client_id resolved to None; skipping autoscaling role creation. "
                    "Check that the configured source (secret scope, env var, etc.) "
                    "is populated.",
                    project=database.project,
                    client_id_spec=database.client_id,
                )
                return

        # The caller's client — NOT ``database.workspace_client`` (the SP's own
        # oauth-m2m client), which cannot grant itself DATABRICKS_SUPERUSER.
        workspace_client: WorkspaceClient = self.w

        # Roles are created on a branch, not on the project
        branch_name = self._resolve_autoscaling_default_branch(
            workspace_client, database.project
        )

        # ``role_id`` is a client-supplied hint that must match
        # ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ — sanitize the client_id and prefix
        # with 'sp-'. NOTE: the server does NOT persist roles under this id; it
        # assigns its own (e.g. ``rol-<random>``) and dedupes on ``postgres_role``
        # (the client_id). So this hint is only used on the create call — the
        # existence check below must match on ``status.postgres_role`` instead.
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
            # Check if a role for this service principal already exists. Roles
            # are keyed server-side by ``status.postgres_role`` (the client_id),
            # NOT by the ``role_id`` we pass to create_role, so we must scan the
            # branch's roles rather than get_role(name=".../sp-<client-id>")
            # (which always 404s and used to force a create → RESOURCE_ALREADY_
            # EXISTS round-trip on every call).
            existing_role = next(
                (
                    r
                    for r in workspace_client.postgres.list_roles(branch_name)
                    if r.status and r.status.postgres_role == client_id
                ),
                None,
            )
            if existing_role is not None:
                logger.info(
                    "Autoscaling Lakebase role already exists",
                    role_name=client_id,
                    role_id=existing_role.role_id,
                    project=database.project,
                )
                return

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
