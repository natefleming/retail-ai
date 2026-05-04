"""Tests for Genie space provisioning via :meth:`GenieRoomModel.create`.

Covers:
- ``_build_serialized_space`` payload construction across all sub-models.
- ``DatabricksProvider.create_genie_space`` create / update / no-op /
  entitlement application logic.
- YAML alias/anchor support so warehouses and tables can be reused via
  ``*anchor`` references.
- Benchmark round-tripping (declared benchmarks land in the payload).
"""

from __future__ import annotations

import json
import textwrap
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml

from dao_ai.config import (
    AppConfig,
    FunctionModel,
    GenieBenchmarkQuestion,
    GenieColumnConfig,
    GenieEntitlement,
    GenieEntitlementLevel,
    GenieExampleSql,
    GenieJoinSpec,
    GenieMetricViewSource,
    GenieRelationshipType,
    GenieRoomModel,
    GenieSqlFunctionSource,
    GenieSqlSnippet,
    GenieTableSource,
    SchemaModel,
    TableModel,
    WarehouseModel,
)


@pytest.fixture
def schema() -> SchemaModel:
    return SchemaModel(catalog_name="cat", schema_name="sch")


@pytest.fixture
def warehouse() -> WarehouseModel:
    return WarehouseModel(name="retail_wh", warehouse_id="wh-abc-123")


@pytest.fixture
def fully_configured_room(
    schema: SchemaModel, warehouse: WarehouseModel
) -> GenieRoomModel:
    products = TableModel(schema=schema, name="products")
    orders = TableModel(schema=schema, name="orders")
    daily_sales = TableModel(schema=schema, name="daily_sales_mv")
    find_product = FunctionModel(schema=schema, name="find_product")

    return GenieRoomModel(
        name="Retail Genie",
        description="Retail Genie space for natural-language SQL.",
        warehouse=warehouse,
        parent_path="/Users/me@example.com/genie",
        table_sources=[
            GenieTableSource(
                table=products,
                description="Product catalog.",
                column_configs=[
                    GenieColumnConfig(
                        name="product_id",
                        description="Primary key.",
                        synonyms=["sku", "item id"],
                        build_value_dictionary=True,
                    )
                ],
            ),
            GenieTableSource(table=orders),
        ],
        metric_view_sources=[
            GenieMetricViewSource(table=daily_sales, description="Daily sales.")
        ],
        function_sources=[GenieSqlFunctionSource(function=find_product)],
        text_instructions=["Always join orders to products via product_id."],
        example_sqls=[
            GenieExampleSql(
                question="Top selling products last month",
                sql="SELECT * FROM cat.sch.products LIMIT 10",
                usage_guidance="Use when asked about ranking.",
            )
        ],
        join_specs=[
            GenieJoinSpec(
                left=orders,
                right=products,
                sql="orders.product_id = products.product_id",
                relationship_type=GenieRelationshipType.MANY_TO_ONE,
                comment="Orders join to product catalog.",
            )
        ],
        sql_filters=[
            GenieSqlSnippet(
                display_name="Active products",
                sql="status = 'ACTIVE'",
                instruction="Apply when user filters to active items.",
                synonyms=["live", "available"],
            )
        ],
        sql_measures=[
            GenieSqlSnippet(display_name="Total revenue", sql="SUM(amount)")
        ],
        sample_questions=["What were the top selling products last month?"],
        benchmarks=[
            GenieBenchmarkQuestion(
                question="How many orders were placed yesterday?",
                expected_sql="SELECT COUNT(*) FROM cat.sch.orders WHERE order_date = current_date - 1",
            )
        ],
        entitlements=[
            GenieEntitlement(
                principals=["user@example.com", "data-team"],
                permission_level=GenieEntitlementLevel.CAN_RUN,
            )
        ],
    )


@pytest.fixture
def mock_workspace_client() -> Mock:
    mock = Mock()
    mock.genie = Mock()
    mock.warehouses = Mock()
    mock.permissions = Mock()
    return mock


@pytest.mark.unit
class TestBuildSerializedSpace:
    """``_build_serialized_space`` produces the documented Genie JSON shape."""

    def test_payload_has_all_top_level_sections(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        assert payload["version"] == 1
        assert "config" in payload and "sample_questions" in payload["config"]
        assert "data_sources" in payload
        assert "tables" in payload["data_sources"]
        assert "metric_views" in payload["data_sources"]
        assert "instructions" in payload
        assert {
            "text_instructions",
            "example_question_sqls",
            "sql_functions",
            "join_specs",
            "sql_snippets",
        }.issubset(payload["instructions"].keys())
        assert "benchmarks" in payload

    def test_table_with_column_configs_serializes_synonyms_and_dictionary(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        product_entry = payload["data_sources"]["tables"][0]
        assert product_entry["identifier"] == "cat.sch.products"
        assert product_entry["description"] == ["Product catalog."]
        col = product_entry["column_configs"][0]
        assert col["column_name"] == "product_id"
        assert col["synonyms"] == ["sku", "item id"]
        assert col["build_value_dictionary"] is True
        assert col["exclude"] is False
        assert col["get_example_values"] is True

    def test_metric_view_serializes_with_identifier_and_description(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        mv = payload["data_sources"]["metric_views"][0]
        assert mv["identifier"] == "cat.sch.daily_sales_mv"
        assert mv["description"] == ["Daily sales."]

    def test_join_spec_encodes_relationship_type_in_sql(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        join = payload["instructions"]["join_specs"][0]
        assert join["left"]["identifier"] == "cat.sch.orders"
        assert join["right"]["identifier"] == "cat.sch.products"
        assert "--rt=MANY_TO_ONE--" in join["sql"][0]

    def test_sql_function_uses_full_uc_identifier(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        fn = payload["instructions"]["sql_functions"][0]
        assert fn["identifier"] == "cat.sch.find_product"
        assert "id" in fn

    def test_sql_snippets_split_by_kind(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        snippets = payload["instructions"]["sql_snippets"]
        assert "filters" in snippets and "measures" in snippets
        assert "expressions" not in snippets  # not configured
        assert snippets["filters"][0]["display_name"] == "Active products"
        assert snippets["filters"][0]["synonyms"] == ["live", "available"]

    def test_benchmark_round_trips(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        question = payload["benchmarks"]["questions"][0]
        assert question["question"] == ["How many orders were placed yesterday?"]
        answer = question["answer"][0]
        assert answer["format"] == "SQL"
        assert "ORDER" not in answer["content"][0].upper().split()[0]  # SELECT, not ORDER

    def test_payload_is_stable_across_runs(
        self, fully_configured_room: GenieRoomModel
    ):
        """Stable IDs ensure idempotent diff checks against the live space."""
        first = fully_configured_room._build_serialized_space()
        second = fully_configured_room._build_serialized_space()
        assert first == second

    def test_empty_room_produces_minimal_payload(
        self, warehouse: WarehouseModel
    ):
        room = GenieRoomModel(name="Empty", warehouse=warehouse)
        payload = room._build_serialized_space()
        assert payload == {"version": 1}


@pytest.mark.unit
class TestCreateGenieSpace:
    """Provider-level create/update/no-op behavior."""

    def _patched_provider(self, w: Mock):
        from dao_ai.providers.databricks import DatabricksProvider

        provider = DatabricksProvider.__new__(DatabricksProvider)
        provider.w = w
        return provider

    def test_creates_new_space_when_no_space_id(
        self,
        fully_configured_room: GenieRoomModel,
        mock_workspace_client: Mock,
    ):
        created = MagicMock()
        created.space_id = "new-space-xyz"
        mock_workspace_client.genie.create_space.return_value = created

        provider = self._patched_provider(mock_workspace_client)
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            provider.create_genie_space(fully_configured_room)

        mock_workspace_client.genie.create_space.assert_called_once()
        call_kwargs = mock_workspace_client.genie.create_space.call_args.kwargs
        assert call_kwargs["title"] == "Retail Genie"
        assert call_kwargs["warehouse_id"] == "wh-abc-123"
        assert call_kwargs["parent_path"] == "/Users/me@example.com/genie"
        assert "data_sources" in json.loads(call_kwargs["serialized_space"])
        assert fully_configured_room.space_id == "new-space-xyz"

    def test_updates_existing_space_when_payload_differs(
        self,
        fully_configured_room: GenieRoomModel,
        mock_workspace_client: Mock,
    ):
        fully_configured_room.space_id = "existing-id"
        existing = MagicMock()
        existing.space_id = "existing-id"
        existing.title = "Old Title"
        existing.description = "old"
        existing.warehouse_id = "wh-abc-123"
        existing.serialized_space = json.dumps({"version": 1})
        existing.etag = "etag-1"
        mock_workspace_client.genie.get_space.return_value = existing
        mock_workspace_client.genie.update_space.return_value = existing

        provider = self._patched_provider(mock_workspace_client)
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            provider.create_genie_space(fully_configured_room)

        mock_workspace_client.genie.create_space.assert_not_called()
        mock_workspace_client.genie.update_space.assert_called_once()
        call_kwargs = mock_workspace_client.genie.update_space.call_args.kwargs
        assert call_kwargs["space_id"] == "existing-id"
        assert call_kwargs["etag"] == "etag-1"

    def test_skips_update_when_payload_unchanged(
        self,
        fully_configured_room: GenieRoomModel,
        mock_workspace_client: Mock,
    ):
        fully_configured_room.space_id = "existing-id"
        existing = MagicMock()
        existing.space_id = "existing-id"
        existing.title = fully_configured_room.name
        existing.description = fully_configured_room.description
        existing.warehouse_id = "wh-abc-123"
        existing.serialized_space = json.dumps(
            fully_configured_room._build_serialized_space()
        )
        existing.etag = "etag-1"
        mock_workspace_client.genie.get_space.return_value = existing

        provider = self._patched_provider(mock_workspace_client)
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            provider.create_genie_space(fully_configured_room)

        mock_workspace_client.genie.update_space.assert_not_called()
        mock_workspace_client.genie.create_space.assert_not_called()

    def test_applies_entitlements_after_create(
        self,
        fully_configured_room: GenieRoomModel,
        mock_workspace_client: Mock,
    ):
        created = MagicMock()
        created.space_id = "new-space-xyz"
        mock_workspace_client.genie.create_space.return_value = created

        provider = self._patched_provider(mock_workspace_client)
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            provider.create_genie_space(fully_configured_room)

        mock_workspace_client.permissions.set.assert_called_once()
        call_kwargs = mock_workspace_client.permissions.set.call_args.kwargs
        assert call_kwargs["request_object_type"] == "genie"
        assert call_kwargs["request_object_id"] == "new-space-xyz"
        acl = call_kwargs["access_control_list"]
        assert len(acl) == 2  # one user + one group
        principals = {ac.user_name or ac.group_name for ac in acl}
        assert principals == {"user@example.com", "data-team"}

    def test_raises_when_no_warehouse_configured(
        self, schema: SchemaModel, mock_workspace_client: Mock
    ):
        room = GenieRoomModel(
            name="No Warehouse",
            table_sources=[
                GenieTableSource(table=TableModel(schema=schema, name="t1"))
            ],
        )
        provider = self._patched_provider(mock_workspace_client)
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            with pytest.raises(ValueError, match="warehouse"):
                provider.create_genie_space(room)


@pytest.mark.unit
class TestYamlAnchors:
    """YAML aliases and anchors flow through to GenieRoomModel via AppConfig."""

    def test_warehouse_and_tables_via_anchor(self, tmp_path):
        yaml_text = textwrap.dedent(
            """
            schemas:
              retail: &retail_schema
                catalog_name: cat
                schema_name: sch

            resources:
              warehouses:
                retail_wh: &retail_wh
                  name: retail_wh
                  warehouse_id: wh-abc-123
              tables:
                products: &products
                  schema: *retail_schema
                  name: products
                orders: &orders
                  schema: *retail_schema
                  name: orders
              genie_rooms:
                retail_genie:
                  name: Retail Genie
                  warehouse: *retail_wh
                  table_sources:
                    - table: *products
                      description: Product catalog.
                    - table: *orders
                  text_instructions:
                    - Join orders to products on product_id.
            """
        )
        config_path = tmp_path / "genie_config.yaml"
        config_path.write_text(yaml_text)

        # Construct AppConfig directly without triggering the network-bound
        # initialize() path; the YAML round-trip is what we care about here.
        raw = yaml.safe_load(yaml_text)
        config = AppConfig(**raw)

        room = config.resources.genie_rooms["retail_genie"]
        assert room.warehouse is not None
        assert room.warehouse.warehouse_id == "wh-abc-123"
        assert len(room.table_sources) == 2
        assert room.table_sources[0].table.full_name == "cat.sch.products"
        assert room.table_sources[0].description == "Product catalog."
        assert room.table_sources[1].table.full_name == "cat.sch.orders"

        # The serialized payload should include both anchored tables.
        payload = room._build_serialized_space()
        identifiers = [t["identifier"] for t in payload["data_sources"]["tables"]]
        assert identifiers == ["cat.sch.products", "cat.sch.orders"]

    def test_full_provisioning_yaml_fixture_loads(self):
        """The shipped tests/config/test_genie_provisioning_config.yaml parses cleanly."""
        from pathlib import Path

        config_path = (
            Path(__file__).parents[1] / "config" / "test_genie_provisioning_config.yaml"
        )
        raw = yaml.safe_load(config_path.read_text())
        config = AppConfig(**raw)

        room = config.resources.genie_rooms["retail_genie"]
        assert room.warehouse.warehouse_id == "wh-abc-123"
        assert len(room.table_sources) == 2
        assert len(room.metric_view_sources) == 1
        assert room.metric_view_sources[0].table.full_name == "cat.sch.daily_sales_mv"
        assert len(room.function_sources) == 1
        assert room.function_sources[0].function.full_name == "cat.sch.find_product"
        assert len(room.join_specs) == 1
        assert room.join_specs[0].relationship_type == GenieRelationshipType.MANY_TO_ONE
        assert len(room.entitlements) == 2

        # The serialized payload should round-trip cleanly through json.
        payload = room._build_serialized_space()
        roundtripped = json.loads(json.dumps(payload, sort_keys=True))
        assert roundtripped == payload

    def test_warehouse_field_aliased_so_tables_property_falls_back(
        self, tmp_path, schema: SchemaModel
    ):
        """Pre-provisioning, ``room.tables`` returns table_sources as TableModels."""
        room = GenieRoomModel(
            name="X",
            warehouse=WarehouseModel(name="wh", warehouse_id="abc"),
            table_sources=[
                GenieTableSource(table=TableModel(schema=schema, name="t1")),
            ],
            function_sources=[
                GenieSqlFunctionSource(
                    function=FunctionModel(schema=schema, name="fn1")
                )
            ],
        )
        # Without a live serialized_space, tables/functions properties fall back
        # to the configured sources.
        assert [t.full_name for t in room.tables] == ["cat.sch.t1"]
        assert [f.full_name for f in room.functions] == ["cat.sch.fn1"]


@pytest.mark.unit
class TestRefresh:
    """Round-tripping ``serialized_space`` back into structured fields."""

    def test_refresh_round_trips_via_build(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()

        fresh = GenieRoomModel(
            name="Fresh",
            warehouse=fully_configured_room.warehouse,
        )
        result = fresh.refresh(payload=payload)
        assert result is fresh
        assert fresh._build_serialized_space() == payload

    def test_refresh_populates_each_section(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        fresh = GenieRoomModel(name="Fresh")
        fresh.refresh(payload=payload)

        assert fresh.sample_questions == fully_configured_room.sample_questions
        assert len(fresh.table_sources) == len(fully_configured_room.table_sources)
        assert fresh.table_sources[0].table.full_name == "cat.sch.products"
        assert fresh.table_sources[0].description == "Product catalog."
        assert fresh.table_sources[0].column_configs[0].synonyms == [
            "sku",
            "item id",
        ]
        assert len(fresh.metric_view_sources) == 1
        assert fresh.metric_view_sources[0].table.full_name == "cat.sch.daily_sales_mv"
        assert len(fresh.function_sources) == 1
        assert fresh.function_sources[0].function.full_name == "cat.sch.find_product"
        assert fresh.text_instructions == [
            "Always join orders to products via product_id."
        ]
        assert fresh.example_sqls[0].usage_guidance == "Use when asked about ranking."
        assert (
            fresh.join_specs[0].relationship_type == GenieRelationshipType.MANY_TO_ONE
        )
        # The --rt= suffix should be stripped from the parsed SQL
        assert "--rt=" not in fresh.join_specs[0].sql
        assert fresh.sql_filters[0].display_name == "Active products"
        assert fresh.sql_measures[0].display_name == "Total revenue"
        assert (
            fresh.benchmarks[0].question
            == "How many orders were placed yesterday?"
        )

    def test_refresh_is_idempotent(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        fresh = GenieRoomModel(name="Fresh")
        fresh.refresh(payload=payload)
        first_state = fresh._build_serialized_space()
        fresh.refresh(payload=payload)
        second_state = fresh._build_serialized_space()
        assert first_state == second_state

    def test_refresh_uses_cached_space_details(
        self,
        fully_configured_room: GenieRoomModel,
        mock_workspace_client: Mock,
    ):
        existing = MagicMock()
        existing.serialized_space = json.dumps(
            fully_configured_room._build_serialized_space()
        )
        mock_workspace_client.genie.get_space.return_value = existing

        with patch(
            "dao_ai.config.WorkspaceClient", return_value=mock_workspace_client
        ):
            room = GenieRoomModel(name="X", space_id="space-123")
            room.ensure_resolved()

            room.refresh()
            room.refresh()
            assert mock_workspace_client.genie.get_space.call_count == 1

            room.refresh(force=True)
            assert mock_workspace_client.genie.get_space.call_count == 2

    def test_refresh_tolerates_unknown_server_keys(self):
        # Sub-models use extra="allow", so refresh never crashes on a
        # serialized_space payload that contains keys we don't model.
        # (Full re-emission of those extras during _build_serialized_space
        # is a follow-up enhancement; this test only locks down the
        # crash-free behavior.)
        payload = {
            "version": 1,
            "data_sources": {
                "tables": [
                    {
                        "identifier": "cat.sch.products",
                        "future_server_table_field": "ignored",
                        "column_configs": [
                            {
                                "column_name": "product_id",
                                "future_server_column_field": "ignored",
                            }
                        ],
                    }
                ],
            },
            "instructions": {
                "future_server_instructions_field": "ignored",
                "text_instructions": [
                    {"id": "abc", "content": ["hi"], "future": "ignored"}
                ],
            },
        }
        room = GenieRoomModel(name="X")
        # Should not raise; known fields are populated, unknowns are skipped.
        room.refresh(payload=payload)
        assert room.table_sources[0].table.full_name == "cat.sch.products"
        assert room.text_instructions == ["hi"]

    def test_refresh_relationship_type_round_trip(self):
        # Build a join with each relationship type, refresh, assert the
        # suffix encoding decodes back to the enum value.
        for rt in GenieRelationshipType:
            schema = SchemaModel(catalog_name="cat", schema_name="sch")
            orig = GenieRoomModel(
                name="X",
                warehouse=WarehouseModel(name="wh", warehouse_id="abc"),
                join_specs=[
                    GenieJoinSpec(
                        left=TableModel(schema=schema, name="a"),
                        right=TableModel(schema=schema, name="b"),
                        sql="a.id = b.fk",
                        relationship_type=rt,
                    )
                ],
            )
            payload = orig._build_serialized_space()
            fresh = GenieRoomModel(name="X")
            fresh.refresh(payload=payload)
            assert fresh.join_specs[0].relationship_type == rt
            assert fresh.join_specs[0].sql == "a.id = b.fk"


@pytest.mark.unit
class TestFromSpace:
    """Factory ``GenieRoomModel.from_space`` returns a fully-hydrated model."""

    def test_from_space_constructs_and_hydrates(
        self,
        fully_configured_room: GenieRoomModel,
        mock_workspace_client: Mock,
    ):
        existing = MagicMock()
        existing.title = "Retail Genie"
        existing.description = "x"
        existing.warehouse_id = "wh-abc-123"
        existing.serialized_space = json.dumps(
            fully_configured_room._build_serialized_space()
        )
        mock_workspace_client.genie.get_space.return_value = existing
        mock_workspace_client.genie.list_spaces.return_value = MagicMock(
            spaces=[MagicMock(title="Retail Genie", space_id="space-123")],
            next_page_token=None,
        )

        with patch(
            "dao_ai.config.WorkspaceClient", return_value=mock_workspace_client
        ):
            room = GenieRoomModel.from_space("space-123")

        assert room.space_id == "space-123"
        assert room._resolved is True
        assert room.table_sources is not None
        assert len(room.table_sources) == 2


@pytest.mark.unit
class TestGenieEntitlement:
    def test_resolves_service_principal_to_client_id(self):
        from dao_ai.config import ServicePrincipalModel

        sp = ServicePrincipalModel(client_id="client-id-1", client_secret="secret")
        ent = GenieEntitlement(
            principals=[sp, "human@example.com"],
            permission_level=GenieEntitlementLevel.CAN_VIEW,
        )
        assert ent.principals == ["client-id-1", "human@example.com"]


def _has_genie_provision_env() -> bool:
    import os

    if not pytest.importorskip("conftest").has_databricks_env():
        return False
    return all(
        var in os.environ
        for var in ("DAO_AI_TEST_WAREHOUSE_NAME", "DAO_AI_TEST_PARENT_PATH")
    )


@pytest.mark.skipif(
    not _has_genie_provision_env(),
    reason="Set DAO_AI_TEST_WAREHOUSE_NAME and DAO_AI_TEST_PARENT_PATH (plus DATABRICKS_*) to run live provisioning tests.",
)
@pytest.mark.integration
class TestRealGenieProvisioning:
    """Integration tests that hit a live workspace.

    Skipped automatically unless DATABRICKS_TOKEN / DATABRICKS_HOST etc. are set.
    Requires a SQL warehouse named ``Serverless Starter Warehouse`` (or any
    warehouse identifiable by ``DAO_AI_TEST_WAREHOUSE_NAME``) and a parent
    workspace path identified by ``DAO_AI_TEST_PARENT_PATH``.
    """

    def _make_room(self) -> GenieRoomModel:
        import os

        warehouse_name = os.environ["DAO_AI_TEST_WAREHOUSE_NAME"]
        parent_path = os.environ["DAO_AI_TEST_PARENT_PATH"]
        return GenieRoomModel(
            name="dao-ai integration test space",
            description="Created by dao-ai integration test; safe to delete.",
            warehouse=WarehouseModel(name=warehouse_name),
            parent_path=parent_path,
            text_instructions=["This is an automated test."],
            sample_questions=["What is 1 + 1?"],
            benchmarks=[
                GenieBenchmarkQuestion(
                    question="echo", expected_sql="SELECT 1"
                )
            ],
        )

    def test_create_then_update_then_cleanup(self):
        room = self._make_room()
        room.create()
        space_id = room.space_id
        assert space_id

        # Re-running .create() with no changes should be a no-op (skip update).
        room.create()

        # Mutate text_instructions and re-run; should trigger an update.
        room.text_instructions = ["Updated by integration test."]
        room.create()

        # Cleanup: trash the space using the workspace client directly.
        try:
            room.workspace_client.genie.trash_space(space_id=space_id)
        except Exception:
            pass
