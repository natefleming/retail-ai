"""Tests for the Model Serving auth-policy partition built by
:func:`dao_ai.providers.databricks.build_auth_policy`.

The contract is symmetric with the Apps OBO partition in
``tests/dao_ai/test_apps_obo_partition.py``:

- An ``on_behalf_of_user=False`` (SP-backed) resource is flattened into
  the :class:`SystemAuthPolicy` so the deployed Model Serving endpoint's
  service principal is auto-granted the required permission on it at
  deploy time.
- An ``on_behalf_of_user=True`` (OBO) resource contributes its
  ``api_scopes`` to the :class:`UserAuthPolicy` instead, so the user's
  forwarded OAuth token has the right scopes at runtime.

A single resource is *never* in both outputs — listing an OBO resource
in SystemAuthPolicy would prompt the operator to authorize a permission
the endpoint SP never uses, and not listing it in UserAuthPolicy would
break user-scoped runtime calls.

These tests construct minimal :class:`AppConfig` graphs and assert the
partition for every resource type that supports the OBO flag.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from dao_ai.config import (
    AppConfig,
    ConnectionModel,
    DatabaseModel,
    FunctionModel,
    GenieRoomModel,
    IndexModel,
    InferenceEndpointModel,
    ResourcesModel,
    SchemaModel,
    TableModel,
    VectorStoreModel,
    VolumeModel,
    WarehouseModel,
)
from dao_ai.providers.databricks import (
    _collect_resources_with_obo_flag,
    build_auth_policy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SCHEMA = SchemaModel(catalog_name="cat", schema_name="sch")


def _config(**resources: dict) -> AppConfig:
    """Build an AppConfig with the given resources dict applied to ResourcesModel.

    Empty resources dicts are allowed; pass ``models={...}``, ``functions={...}``,
    etc. to populate specific categories.
    """
    return AppConfig(resources=ResourcesModel(**resources))


def _policy_resource_names(resources: Sequence) -> set[str]:
    """Extract the ``.name`` of every DatabricksResource (works across subclasses)."""
    out: set[str] = set()
    for r in resources:
        name = getattr(r, "name", None)
        if isinstance(name, str):
            out.add(name)
        # uc_securable types use function_name / index_name / etc.
        for attr in (
            "function_name",
            "index_name",
            "table_name",
            "warehouse_id",
            "genie_space_id",
            "endpoint_name",
            "connection_name",
            "database_name",
        ):
            v = getattr(r, attr, None)
            if isinstance(v, str):
                out.add(v)
    return out


# ---------------------------------------------------------------------------
# Per-resource-type OBO partition matrix
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerResourceTypeAuthPolicyPartition:
    """For each resource type that supports ``on_behalf_of_user``, the SP-backed
    instance lands in ``SystemAuthPolicy.resources`` and the OBO instance
    contributes its ``api_scopes`` to ``UserAuthPolicy.api_scopes`` instead.

    Concretely: build a config with one SP + one OBO resource of the same
    type, then assert (a) the SP resource's identifier is in the system
    policy, (b) the OBO resource's identifier is NOT in the system policy,
    (c) the OBO resource's scopes are in the user policy.
    """

    def test_llm_partition(self) -> None:
        config = _config(
            models={
                "sp_llm": InferenceEndpointModel(
                    name="databricks-claude-sonnet-4-5", on_behalf_of_user=False
                ),
                "obo_llm": InferenceEndpointModel(
                    name="databricks-gte-large-en", on_behalf_of_user=True
                ),
            }
        )
        policy = build_auth_policy(config)
        names = _policy_resource_names(policy.system_auth_policy.resources)
        assert "databricks-claude-sonnet-4-5" in names
        assert "databricks-gte-large-en" not in names
        assert "serving.serving-endpoints" in policy.user_auth_policy.api_scopes

    def test_vector_store_partition(self) -> None:
        config = _config(
            vector_stores={
                "sp_vs": VectorStoreModel(
                    index=IndexModel(schema=_SCHEMA, name="sp_idx"),
                    on_behalf_of_user=False,
                ),
                "obo_vs": VectorStoreModel(
                    index=IndexModel(schema=_SCHEMA, name="obo_idx"),
                    on_behalf_of_user=True,
                ),
            }
        )
        policy = build_auth_policy(config)
        names = _policy_resource_names(policy.system_auth_policy.resources)
        assert "cat.sch.sp_idx" in names
        assert "cat.sch.obo_idx" not in names
        # Canonical OBO scopes: vector-search + its mcp.vectorsearch companion
        # (vector-search-endpoints + vector-search-indexes both translate
        # through this pair). serving.serving-endpoints also comes from
        # VectorStoreModel's api_scopes for the embedding model invocation.
        scopes = set(policy.user_auth_policy.api_scopes)
        assert "vector-search" in scopes
        assert "mcp.vectorsearch" in scopes
        assert "serving.serving-endpoints" in scopes

    def test_warehouse_partition(self) -> None:
        config = _config(
            warehouses={
                "sp_wh": WarehouseModel(
                    name="sp", warehouse_id="wh-sp", on_behalf_of_user=False
                ),
                "obo_wh": WarehouseModel(
                    name="obo", warehouse_id="wh-obo", on_behalf_of_user=True
                ),
            }
        )
        policy = build_auth_policy(config)
        names = _policy_resource_names(policy.system_auth_policy.resources)
        assert "wh-sp" in names
        assert "wh-obo" not in names
        # Canonical OBO scopes: sql + mcp.functions companion.
        scopes = set(policy.user_auth_policy.api_scopes)
        assert "sql" in scopes
        assert "mcp.functions" in scopes

    def test_genie_room_partition(self) -> None:
        config = _config(
            genie_rooms={
                "sp_room": GenieRoomModel(
                    name="sp", space_id="01f0sp", on_behalf_of_user=False
                ),
                "obo_room": GenieRoomModel(
                    name="obo", space_id="01f0obo", on_behalf_of_user=True
                ),
            }
        )
        policy = build_auth_policy(config)
        names = _policy_resource_names(policy.system_auth_policy.resources)
        assert "01f0sp" in names
        assert "01f0obo" not in names
        # Canonical OBO scopes: genie + mcp.genie companion.
        scopes = set(policy.user_auth_policy.api_scopes)
        assert "genie" in scopes
        assert "mcp.genie" in scopes

    def test_function_partition(self) -> None:
        config = _config(
            functions={
                "sp_fn": FunctionModel(
                    schema=_SCHEMA, name="sp_fn", on_behalf_of_user=False
                ),
                "obo_fn": FunctionModel(
                    schema=_SCHEMA, name="obo_fn", on_behalf_of_user=True
                ),
            }
        )
        policy = build_auth_policy(config)
        names = _policy_resource_names(policy.system_auth_policy.resources)
        assert "cat.sch.sp_fn" in names
        assert "cat.sch.obo_fn" not in names
        # Canonical OBO scopes: sql + mcp.functions companion; functions also
        # pull in catalog.*:read auto-additions.
        scopes = set(policy.user_auth_policy.api_scopes)
        assert "sql" in scopes
        assert "mcp.functions" in scopes
        assert "catalog.catalogs:read" in scopes
        assert "catalog.schemas:read" in scopes
        assert "catalog.tables:read" in scopes

    def test_table_partition(self) -> None:
        config = _config(
            tables={
                "sp_t": TableModel(
                    schema=_SCHEMA, name="sp_t", on_behalf_of_user=False
                ),
                "obo_t": TableModel(
                    schema=_SCHEMA, name="obo_t", on_behalf_of_user=True
                ),
            }
        )
        policy = build_auth_policy(config)
        names = _policy_resource_names(policy.system_auth_policy.resources)
        assert "cat.sch.sp_t" in names
        assert "cat.sch.obo_t" not in names
        # Tables: sql + mcp.functions companion + catalog.*:read auto-add.
        scopes = set(policy.user_auth_policy.api_scopes)
        assert "sql" in scopes
        assert "mcp.functions" in scopes
        assert "catalog.catalogs:read" in scopes
        assert "catalog.schemas:read" in scopes
        assert "catalog.tables:read" in scopes

    def test_volume_partition(self) -> None:
        """VolumeModel.as_resources() returns ``[]`` by design (no MLflow Resource
        type for UC volumes today), so an SP-backed volume contributes nothing
        to SystemAuthPolicy. An OBO volume still contributes its scopes to
        UserAuthPolicy — that's the only side of the partition we can assert.
        """
        config = _config(
            volumes={
                "sp_v": VolumeModel(
                    schema=_SCHEMA, name="sp_v", on_behalf_of_user=False
                ),
                "obo_v": VolumeModel(
                    schema=_SCHEMA, name="obo_v", on_behalf_of_user=True
                ),
            }
        )
        policy = build_auth_policy(config)
        names = _policy_resource_names(policy.system_auth_policy.resources)
        # Neither volume produces a DatabricksResource — neither name should
        # appear in the system policy.
        assert "cat.sch.sp_v" not in names
        assert "cat.sch.obo_v" not in names
        # OBO volume pushes its canonical user-OBO scope.
        scopes = set(policy.user_auth_policy.api_scopes)
        assert "files" in scopes

    def test_connection_partition(self) -> None:
        config = _config(
            connections={
                "sp_conn": ConnectionModel(name="sp_conn", on_behalf_of_user=False),
                "obo_conn": ConnectionModel(name="obo_conn", on_behalf_of_user=True),
            }
        )
        policy = build_auth_policy(config)
        names = _policy_resource_names(policy.system_auth_policy.resources)
        assert "sp_conn" in names
        assert "obo_conn" not in names
        assert "catalog.connections" in policy.user_auth_policy.api_scopes


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAuthPolicyInvariants:
    """Properties that must hold across the whole partition."""

    def test_empty_config_produces_empty_policy(self) -> None:
        """An AppConfig with no resources produces empty system + user policies."""
        config = _config()
        policy = build_auth_policy(config)
        assert list(policy.system_auth_policy.resources) == []
        assert list(policy.user_auth_policy.api_scopes) == []

    def test_obo_resource_never_in_system_policy(self) -> None:
        """If a resource is OBO, none of its DatabricksResource instances
        leak into SystemAuthPolicy. Covers every resource type at once
        via a mixed config."""
        config = _config(
            models={
                "obo_llm": InferenceEndpointModel(
                    name="obo-llm", on_behalf_of_user=True
                )
            },
            functions={
                "obo_fn": FunctionModel(
                    schema=_SCHEMA, name="obo_fn", on_behalf_of_user=True
                )
            },
            warehouses={
                "obo_wh": WarehouseModel(
                    name="obo", warehouse_id="wh-obo", on_behalf_of_user=True
                )
            },
            genie_rooms={
                "obo_room": GenieRoomModel(
                    name="obo", space_id="01f0obo", on_behalf_of_user=True
                )
            },
            vector_stores={
                "obo_vs": VectorStoreModel(
                    index=IndexModel(schema=_SCHEMA, name="obo_idx"),
                    on_behalf_of_user=True,
                )
            },
            tables={
                "obo_t": TableModel(
                    schema=_SCHEMA, name="obo_t", on_behalf_of_user=True
                )
            },
            volumes={
                "obo_v": VolumeModel(
                    schema=_SCHEMA, name="obo_v", on_behalf_of_user=True
                )
            },
            connections={
                "obo_conn": ConnectionModel(name="obo_conn", on_behalf_of_user=True)
            },
        )
        policy = build_auth_policy(config)
        # System policy is empty when every declared resource is OBO.
        assert list(policy.system_auth_policy.resources) == [], (
            "OBO resources leaked into SystemAuthPolicy: "
            + repr(policy.system_auth_policy.resources)
        )
        # User policy carries every OBO-resource canonical scope plus
        # MCP companions.
        scopes = set(policy.user_auth_policy.api_scopes)
        assert "serving.serving-endpoints" in scopes
        assert "sql" in scopes
        assert "mcp.functions" in scopes
        assert "genie" in scopes
        assert "mcp.genie" in scopes
        assert "vector-search" in scopes
        assert "mcp.vectorsearch" in scopes
        assert "files" in scopes
        assert "catalog.connections" in scopes
        assert "mcp.external" in scopes
        # Table + function presence triggers catalog.*:read auto-add.
        assert "catalog.catalogs:read" in scopes
        assert "catalog.schemas:read" in scopes
        assert "catalog.tables:read" in scopes

    def test_sp_resource_never_in_user_policy_scopes(self) -> None:
        """If every declared resource is SP-backed, UserAuthPolicy is empty."""
        config = _config(
            models={
                "sp_llm": InferenceEndpointModel(name="sp-llm", on_behalf_of_user=False)
            },
            functions={
                "sp_fn": FunctionModel(
                    schema=_SCHEMA, name="sp_fn", on_behalf_of_user=False
                )
            },
        )
        policy = build_auth_policy(config)
        assert list(policy.user_auth_policy.api_scopes) == []

    def test_mixed_obo_and_sp_partitioned_correctly(self) -> None:
        """SP and OBO of the same resource type land on opposite sides."""
        config = _config(
            models={
                "sp_llm": InferenceEndpointModel(
                    name="sp-llm", on_behalf_of_user=False
                ),
                "obo_llm": InferenceEndpointModel(
                    name="obo-llm", on_behalf_of_user=True
                ),
            },
            functions={
                "sp_fn": FunctionModel(
                    schema=_SCHEMA, name="sp_fn", on_behalf_of_user=False
                ),
                "obo_fn": FunctionModel(
                    schema=_SCHEMA, name="obo_fn", on_behalf_of_user=True
                ),
            },
        )
        policy = build_auth_policy(config)
        names = _policy_resource_names(policy.system_auth_policy.resources)
        assert "sp-llm" in names
        assert "cat.sch.sp_fn" in names
        assert "obo-llm" not in names
        assert "cat.sch.obo_fn" not in names
        # Both SP and OBO LLMs share the same `serving.serving-endpoints`
        # scope on the model — the partition is by resource OBO flag, not
        # by scope.
        assert "serving.serving-endpoints" in policy.user_auth_policy.api_scopes

    def test_user_api_scopes_are_deduplicated(self) -> None:
        """Two OBO LLMs share one scope — it appears exactly once in the user policy."""
        config = _config(
            models={
                "obo_llm_a": InferenceEndpointModel(name="a", on_behalf_of_user=True),
                "obo_llm_b": InferenceEndpointModel(name="b", on_behalf_of_user=True),
            }
        )
        policy = build_auth_policy(config)
        scopes = list(policy.user_auth_policy.api_scopes)
        assert scopes.count("serving.serving-endpoints") == 1

    def test_genie_tables_propagate_to_system_when_sp(self) -> None:
        """Tables hung off a Genie room (via ``table_sources``) but not
        declared under ``resources.tables`` must still appear in the system
        policy when the Genie room's own ``on_behalf_of_user=False``."""
        from dao_ai.config import GenieTableSource

        room = GenieRoomModel(
            name="sp_room",
            space_id="01f0sp",
            on_behalf_of_user=False,
            table_sources=[
                GenieTableSource(
                    table=TableModel(
                        schema=_SCHEMA, name="genie_tbl", on_behalf_of_user=False
                    )
                )
            ],
        )
        config = _config(genie_rooms={"sp_room": room})
        names = _policy_resource_names(
            build_auth_policy(config).system_auth_policy.resources
        )
        # Genie space itself
        assert "01f0sp" in names
        # The Genie-sourced table is pulled in via _collect_resources_with_obo_flag
        assert "cat.sch.genie_tbl" in names


# ---------------------------------------------------------------------------
# Sanity check on the pure resource-collection helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollectResourcesWithObOFlag:
    """The flattener preserves the OBO flag on each entry so downstream
    policy-building can partition them. Tested separately so a flattener
    bug shows up before a policy bug."""

    def test_obo_flag_preserved(self) -> None:
        config = _config(
            models={
                "sp_llm": InferenceEndpointModel(
                    name="sp-llm", on_behalf_of_user=False
                ),
                "obo_llm": InferenceEndpointModel(
                    name="obo-llm", on_behalf_of_user=True
                ),
            }
        )
        collected = _collect_resources_with_obo_flag(config)
        flags = {r.name: r.on_behalf_of_user for r in collected}
        assert flags == {"sp-llm": False, "obo-llm": True}

    def test_resources_none_returns_empty(self) -> None:
        config = AppConfig()
        assert _collect_resources_with_obo_flag(config) == ()


# ---------------------------------------------------------------------------
# Canonical-string + AI Gateway gating + MCP companion pairing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCanonicalUserApiScopes:
    """The Apps OBO platform expects canonical strings (``sql``, ``genie``,
    ``files``, ``vector-search``, ``ai-gateway``, …). dao-ai's resource models
    declare older dotted resource-scope names (``sql.warehouses``,
    ``dashboards.genie``, …) internally — these MUST be translated before
    they leave the dao-ai boundary, on both Apps and Model Serving surfaces.
    """

    def test_canonical_strings_emitted_not_dao_internal_names(self) -> None:
        """A mixed OBO config produces canonical strings, never dao-ai's
        internal resource-scope names."""
        config = _config(
            warehouses={
                "wh": WarehouseModel(
                    name="w", warehouse_id="wh-1", on_behalf_of_user=True
                )
            },
            genie_rooms={
                "g": GenieRoomModel(name="g", space_id="01f0", on_behalf_of_user=True)
            },
            volumes={
                "v": VolumeModel(schema=_SCHEMA, name="v", on_behalf_of_user=True)
            },
            vector_stores={
                "vs": VectorStoreModel(
                    index=IndexModel(schema=_SCHEMA, name="idx"),
                    on_behalf_of_user=True,
                )
            },
        )
        scopes = set(build_auth_policy(config).user_auth_policy.api_scopes)
        # Canonical strings present:
        assert {"sql", "genie", "files", "vector-search"} <= scopes
        # dao-ai internal resource-scope names absent:
        leaked_internal = {
            "sql.warehouses",
            "sql.statement-execution",
            "dashboards.genie",
            "files.files",
            "catalog.volumes",
            "vectorsearch.vector-search-indexes",
            "vectorsearch.vector-search-endpoints",
        } & scopes
        assert leaked_internal == set(), (
            f"Internal resource-scope names leaked to UserAuthPolicy: {leaked_internal}"
        )


@pytest.mark.unit
class TestMcpCompanionPairing:
    """Each native OBO scope emits its MCP companion automatically.
    Pairings (user-confirmed): sql ↔ mcp.functions, genie ↔ mcp.genie,
    vector-search ↔ mcp.vectorsearch, catalog.connections ↔ mcp.external.
    """

    def test_sql_pairs_with_mcp_functions(self) -> None:
        config = _config(
            warehouses={
                "wh": WarehouseModel(
                    name="w", warehouse_id="wh-1", on_behalf_of_user=True
                )
            }
        )
        scopes = set(build_auth_policy(config).user_auth_policy.api_scopes)
        assert "sql" in scopes
        assert "mcp.functions" in scopes

    def test_genie_pairs_with_mcp_genie(self) -> None:
        config = _config(
            genie_rooms={
                "g": GenieRoomModel(name="g", space_id="01f0", on_behalf_of_user=True)
            }
        )
        scopes = set(build_auth_policy(config).user_auth_policy.api_scopes)
        assert "genie" in scopes
        assert "mcp.genie" in scopes

    def test_vector_search_pairs_with_mcp_vectorsearch(self) -> None:
        config = _config(
            vector_stores={
                "vs": VectorStoreModel(
                    index=IndexModel(schema=_SCHEMA, name="idx"),
                    on_behalf_of_user=True,
                )
            }
        )
        scopes = set(build_auth_policy(config).user_auth_policy.api_scopes)
        assert "vector-search" in scopes
        assert "mcp.vectorsearch" in scopes

    def test_connection_pairs_with_mcp_external(self) -> None:
        config = _config(
            connections={"c": ConnectionModel(name="c", on_behalf_of_user=True)}
        )
        scopes = set(build_auth_policy(config).user_auth_policy.api_scopes)
        assert "catalog.connections" in scopes
        assert "mcp.external" in scopes

    def test_mcp_external_scoped_to_connection_only(self) -> None:
        """``mcp.external`` is the UC Connection's companion — it MUST NOT be
        emitted for SQL/Vector/Genie OBO resources."""
        config = _config(
            warehouses={
                "wh": WarehouseModel(
                    name="w", warehouse_id="wh-1", on_behalf_of_user=True
                )
            },
            genie_rooms={
                "g": GenieRoomModel(name="g", space_id="01f0", on_behalf_of_user=True)
            },
            vector_stores={
                "vs": VectorStoreModel(
                    index=IndexModel(schema=_SCHEMA, name="idx"),
                    on_behalf_of_user=True,
                )
            },
        )
        scopes = set(build_auth_policy(config).user_auth_policy.api_scopes)
        assert "mcp.external" not in scopes


@pytest.mark.unit
class TestAiGatewayGating:
    """``ai-gateway`` is emitted only when an ``InferenceEndpointModel`` has
    BOTH ``on_behalf_of_user=True`` AND ``ai_gateway=True``."""

    def test_both_flags_true_emits_ai_gateway(self) -> None:
        config = _config(
            models={
                "obo_gw": InferenceEndpointModel(
                    name="claude-via-gw",
                    on_behalf_of_user=True,
                    ai_gateway=True,
                ),
            }
        )
        scopes = set(build_auth_policy(config).user_auth_policy.api_scopes)
        assert "ai-gateway" in scopes
        assert "serving.serving-endpoints" in scopes

    def test_obo_only_no_gateway_omits_ai_gateway(self) -> None:
        """OBO LLM without ai_gateway flag must NOT emit ai-gateway."""
        config = _config(
            models={
                "obo": InferenceEndpointModel(
                    name="claude",
                    on_behalf_of_user=True,
                    ai_gateway=False,
                ),
            }
        )
        scopes = set(build_auth_policy(config).user_auth_policy.api_scopes)
        assert "ai-gateway" not in scopes
        assert "serving.serving-endpoints" in scopes

    def test_gateway_only_no_obo_omits_ai_gateway(self) -> None:
        """SP-backed LLM with ai_gateway flag must NOT emit ai-gateway as a
        user scope — there's no user token to scope it to. The LLM stays in
        the system policy and no user_api_scope is emitted for it."""
        config = _config(
            models={
                "sp_gw": InferenceEndpointModel(
                    name="claude-sp-gw",
                    on_behalf_of_user=False,
                    ai_gateway=True,
                ),
            }
        )
        policy = build_auth_policy(config)
        assert "ai-gateway" not in policy.user_auth_policy.api_scopes
        # The model still lands in system policy as an SP resource.
        names = _policy_resource_names(policy.system_auth_policy.resources)
        assert "claude-sp-gw" in names

    def test_neither_flag_no_ai_gateway(self) -> None:
        config = _config(
            models={
                "sp": InferenceEndpointModel(
                    name="claude-sp",
                    on_behalf_of_user=False,
                    ai_gateway=False,
                ),
            }
        )
        scopes = set(build_auth_policy(config).user_auth_policy.api_scopes)
        assert "ai-gateway" not in scopes


@pytest.mark.unit
class TestPostgresObO:
    """``postgres`` is now a first-class OBO scope. A Lakebase database with
    ``on_behalf_of_user=True`` must contribute ``postgres`` to the user
    policy, and the database resource itself must NOT leak into the system
    policy."""

    def test_lakebase_obo_emits_postgres_scope(self) -> None:
        config = _config(
            databases={
                "lb": DatabaseModel(
                    project="lb-project",
                    on_behalf_of_user=True,
                ),
            }
        )
        policy = build_auth_policy(config)
        assert "postgres" in policy.user_auth_policy.api_scopes

    def test_lakebase_sp_does_not_emit_postgres_user_scope(self) -> None:
        config = _config(
            databases={
                "lb": DatabaseModel(
                    project="lb-project",
                    on_behalf_of_user=False,
                ),
            }
        )
        policy = build_auth_policy(config)
        assert "postgres" not in policy.user_auth_policy.api_scopes
