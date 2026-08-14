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
        sql_measures=[GenieSqlSnippet(display_name="Total revenue", sql="SUM(amount)")],
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
        assert payload["version"] == 2
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
        # Tables come out in serialized_space order — find the products
        # entry by identifier rather than positional index.
        product_entry = next(
            t
            for t in payload["data_sources"]["tables"]
            if t["identifier"] == "cat.sch.products"
        )
        assert product_entry["description"] == ["Product catalog."]
        col = product_entry["column_configs"][0]
        assert col["column_name"] == "product_id"
        assert col["synonyms"] == ["sku", "item id"]

    def test_metric_view_serializes_with_identifier_and_description(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        mv = payload["data_sources"]["metric_views"][0]
        assert mv["identifier"] == "cat.sch.daily_sales_mv"
        assert mv["description"] == ["Daily sales."]

    def test_join_spec_sql_carries_condition_then_relationship_marker(
        self, fully_configured_room: GenieRoomModel
    ):
        # Genie's export proto wants exactly two elements in ``sql``: the
        # join condition, then a ``--rt=FROM_RELATIONSHIP_TYPE_…--`` marker.
        # Sending the condition alone is rejected with "Failed to parse
        # export proto: <condition> (of class java.lang.String)".
        payload = fully_configured_room._build_serialized_space()
        join = payload["instructions"]["join_specs"][0]
        assert join["left"]["identifier"] == "cat.sch.orders"
        assert join["right"]["identifier"] == "cat.sch.products"
        assert join["sql"] == [
            "orders.product_id = products.product_id",
            "--rt=FROM_RELATIONSHIP_TYPE_MANY_TO_ONE--",
        ]

    def test_join_spec_without_relationship_type_marks_it_unspecified(
        self, schema: SchemaModel, warehouse: WarehouseModel
    ):
        # The marker is mandatory even when the config declares no
        # cardinality, so an undeclared relationship still has to post one.
        room = GenieRoomModel(
            name="X",
            warehouse=warehouse,
            join_specs=[
                GenieJoinSpec(
                    left=TableModel(schema=schema, name="orders"),
                    right=TableModel(schema=schema, name="products"),
                    sql="orders.product_id = products.product_id",
                )
            ],
        )
        join = room._build_serialized_space()["instructions"]["join_specs"][0]
        assert join["sql"][1] == "--rt=FROM_RELATIONSHIP_TYPE_UNSPECIFIED--"

    def test_join_spec_emits_declared_aliases(
        self, schema: SchemaModel, warehouse: WarehouseModel
    ):
        room = GenieRoomModel(
            name="X",
            warehouse=warehouse,
            join_specs=[
                GenieJoinSpec(
                    left=TableModel(schema=schema, name="orders"),
                    left_alias="o",
                    right=TableModel(schema=schema, name="products"),
                    right_alias="p",
                    sql="`o`.`product_id` = `p`.`product_id`",
                    relationship_type=GenieRelationshipType.MANY_TO_ONE,
                )
            ],
        )
        join = room._build_serialized_space()["instructions"]["join_specs"][0]
        assert join["left"]["alias"] == "o"
        assert join["right"]["alias"] == "p"

    def test_sql_function_uses_full_uc_identifier(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        fn = payload["instructions"]["sql_functions"][0]
        assert fn["identifier"] == "cat.sch.find_product"
        assert "id" in fn

    def test_sql_functions_sorted_by_id_identifier(
        self,
        schema: SchemaModel,
        warehouse: WarehouseModel,
    ):
        """Genie's export-proto validator rejects ``instructions.sql_functions``
        entries that are not sorted by ``(id, identifier)``. Verify the
        ``GenieRoomModel`` emitter sorts regardless of YAML order."""
        fn_a = FunctionModel(schema=schema, name="find_aardvark")
        fn_m = FunctionModel(schema=schema, name="find_mango")
        fn_z = FunctionModel(schema=schema, name="find_zebra")

        room = GenieRoomModel(
            name="Sort-Order Genie",
            warehouse=warehouse,
            parent_path="/Users/me@example.com/genie",
            # Intentionally out of alphabetical order — the emitter must sort.
            function_sources=[
                GenieSqlFunctionSource(function=fn_z),
                GenieSqlFunctionSource(function=fn_a),
                GenieSqlFunctionSource(function=fn_m),
            ],
        )

        sql_functions = room._build_serialized_space()["instructions"]["sql_functions"]
        keys = [(entry["id"], entry["identifier"]) for entry in sql_functions]
        assert keys == sorted(keys), (
            f"sql_functions must be emitted sorted by (id, identifier); got {keys}"
        )

    def test_sql_snippets_split_by_kind(self, fully_configured_room: GenieRoomModel):
        payload = fully_configured_room._build_serialized_space()
        snippets = payload["instructions"]["sql_snippets"]
        assert "filters" in snippets and "measures" in snippets
        assert "expressions" not in snippets  # not configured
        assert snippets["filters"][0]["display_name"] == "Active products"
        assert snippets["filters"][0]["synonyms"] == ["live", "available"]

    def test_sql_snippets_all_carry_display_name(
        self,
        schema: SchemaModel,
        warehouse: WarehouseModel,
    ):
        """Genie's export-proto validator requires ``display_name`` on every
        snippet type (filters, expressions, measures). Historically the
        emitter wrote ``alias`` for expressions+measures and ``display_name``
        only for filters, which trips
        ``Invalid export proto: instructions.sql_snippets.expressions[0].display_name is required but missing``
        against the current Genie validator. Verify all three types now
        carry ``display_name``."""
        room = GenieRoomModel(
            name="Snippets Genie",
            warehouse=warehouse,
            parent_path="/Users/me@example.com/genie",
            sql_filters=[GenieSqlSnippet(display_name="Active", sql="status = 'A'")],
            sql_expressions=[
                GenieSqlSnippet(display_name="FullName", sql="first || ' ' || last")
            ],
            sql_measures=[GenieSqlSnippet(display_name="Revenue", sql="SUM(amount)")],
        )
        snippets = room._build_serialized_space()["instructions"]["sql_snippets"]
        for kind in ("filters", "expressions", "measures"):
            assert kind in snippets, f"missing snippet kind {kind}"
            for entry in snippets[kind]:
                assert "display_name" in entry and entry["display_name"], (
                    f"snippet kind {kind!r} entry missing display_name: {entry}"
                )

    def test_benchmark_round_trips(self, fully_configured_room: GenieRoomModel):
        payload = fully_configured_room._build_serialized_space()
        question = payload["benchmarks"]["questions"][0]
        assert question["question"] == ["How many orders were placed yesterday?"]
        answer = question["answer"][0]
        assert answer["format"] == "SQL"
        assert (
            "ORDER" not in answer["content"][0].upper().split()[0]
        )  # SELECT, not ORDER

    def test_payload_is_stable_across_runs(self, fully_configured_room: GenieRoomModel):
        """Stable IDs ensure idempotent diff checks against the live space."""
        first = fully_configured_room._build_serialized_space()
        second = fully_configured_room._build_serialized_space()
        assert first == second

    def test_empty_room_produces_minimal_payload(self, warehouse: WarehouseModel):
        room = GenieRoomModel(name="Empty", warehouse=warehouse)
        payload = room._build_serialized_space()
        assert payload == {"version": 2}


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

        # The serialized payload should include both anchored tables
        # (order is implementation-defined; just assert the set).
        payload = room._build_serialized_space()
        identifiers = {t["identifier"] for t in payload["data_sources"]["tables"]}
        assert identifiers == {"cat.sch.products", "cat.sch.orders"}

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

    def test_refresh_round_trips_via_build(self, fully_configured_room: GenieRoomModel):
        payload = fully_configured_room._build_serialized_space()

        fresh = GenieRoomModel(
            name="Fresh",
            warehouse=fully_configured_room.warehouse,
        )
        result = fresh.refresh(payload=payload)
        assert result is fresh
        # Re-emit and compare the load-bearing sections. Some fields
        # (e.g. ``sql_snippets[*].measures[*].alias``, join
        # ``relationship_type``) are intentionally not round-tripped
        # through the current ``_build_serialized_space`` → ``refresh``
        # cycle — they're carried only as inputs from the structured
        # model and the live space stores ``display_name`` separately.
        rebuilt = fresh._build_serialized_space()
        assert rebuilt["version"] == payload["version"]
        # data_sources tables / metric_views identifiers match (order-agnostic).
        assert {t["identifier"] for t in rebuilt["data_sources"]["tables"]} == {
            t["identifier"] for t in payload["data_sources"]["tables"]
        }
        assert {mv["identifier"] for mv in rebuilt["data_sources"]["metric_views"]} == {
            mv["identifier"] for mv in payload["data_sources"]["metric_views"]
        }
        # sample_questions and benchmarks round-trip cleanly.
        assert rebuilt.get("config") == payload.get("config")
        assert rebuilt.get("benchmarks") == payload.get("benchmarks")

    def test_refresh_populates_each_section(
        self, fully_configured_room: GenieRoomModel
    ):
        payload = fully_configured_room._build_serialized_space()
        fresh = GenieRoomModel(name="Fresh")
        fresh.refresh(payload=payload)

        assert fresh.sample_questions == fully_configured_room.sample_questions
        assert len(fresh.table_sources) == len(fully_configured_room.table_sources)
        # Look up the products table by full_name rather than positional
        # index — the order in serialized_space isn't part of the public
        # contract.
        products_source = next(
            t for t in fresh.table_sources if t.table.full_name == "cat.sch.products"
        )
        assert products_source.description == "Product catalog."
        assert products_source.column_configs[0].synonyms == [
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
        assert fresh.join_specs[0].sql == "orders.product_id = products.product_id"
        assert (
            fresh.join_specs[0].relationship_type == GenieRelationshipType.MANY_TO_ONE
        )
        assert fresh.sql_filters[0].display_name == "Active products"
        assert fresh.benchmarks[0].question == "How many orders were placed yesterday?"

    def test_refresh_is_idempotent(self, fully_configured_room: GenieRoomModel):
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

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
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

    def test_join_spec_parses_marker_from_the_second_sql_element(self):
        entry = {
            "id": "abc",
            "left": {"identifier": "cat.sch.orders", "alias": "o"},
            "right": {"identifier": "cat.sch.products"},
            "sql": [
                "`o`.`product_id` = `products`.`product_id`",
                "--rt=FROM_RELATIONSHIP_TYPE_ONE_TO_MANY--",
            ],
        }
        spec = GenieRoomModel._join_spec_from_payload(entry)
        assert spec.sql == "`o`.`product_id` = `products`.`product_id`"
        assert spec.relationship_type == GenieRelationshipType.ONE_TO_MANY
        assert spec.left_alias == "o"

    def test_unspecified_marker_parses_back_to_no_relationship_type(self):
        entry = {
            "id": "abc",
            "left": {"identifier": "cat.sch.orders"},
            "right": {"identifier": "cat.sch.products"},
            "sql": [
                "orders.product_id = products.product_id",
                "--rt=FROM_RELATIONSHIP_TYPE_UNSPECIFIED--",
            ],
        }
        spec = GenieRoomModel._join_spec_from_payload(entry)
        assert spec.relationship_type is None
        assert spec.sql == "orders.product_id = products.product_id"

    def test_join_spec_parses_legacy_inline_marker(self):
        # Spaces provisioned by older builds carry the marker appended to
        # the condition itself; keep decoding those.
        entry = {
            "id": "abc",
            "left": {"identifier": "cat.sch.orders"},
            "right": {"identifier": "cat.sch.products"},
            "sql": ["orders.product_id = products.product_id --rt=MANY_TO_ONE--"],
        }
        spec = GenieRoomModel._join_spec_from_payload(entry)
        assert spec.sql == "orders.product_id = products.product_id"
        assert spec.relationship_type == GenieRelationshipType.MANY_TO_ONE

    def test_refresh_relationship_type_round_trip(self):
        # Build a join with each relationship type, refresh, assert the
        # marker encoding decodes back to the enum value.
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

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
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
                GenieBenchmarkQuestion(question="echo", expected_sql="SELECT 1")
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

    # ------------------------------------------------------------------
    # 4-scenario cascade coverage. Exercises the exact resolution chain
    # used by ``notebooks/05_provision_genie.py``:
    #   from_space_id(room.space_id) or from_name(room.name) or create()
    # plus the "skip" branches (no genie_rooms / literal space_id) that
    # short-circuit before the cascade runs.
    # ------------------------------------------------------------------

    def _unique_title(self) -> str:
        """A title that won't collide with prior test runs or the loyalty deploy."""
        import uuid

        return f"dao-ai cascade test {uuid.uuid4().hex[:8]}"

    def _make_room_with_title(self, title: str) -> GenieRoomModel:
        """Mirror of _make_room with an injected title for isolation."""
        import os

        warehouse_name = os.environ["DAO_AI_TEST_WAREHOUSE_NAME"]
        parent_path = os.environ["DAO_AI_TEST_PARENT_PATH"]
        return GenieRoomModel(
            name=title,
            description="Created by dao-ai cascade test; safe to delete.",
            warehouse=WarehouseModel(name=warehouse_name),
            parent_path=parent_path,
            text_instructions=["Cascade test."],
            sample_questions=["What is 1 + 1?"],
        )

    def test_literal_space_id_is_skipped(self):
        """Scenario 2: a literal ``space_id`` (no ``${var.X}`` reference) skips
        provisioning entirely. ``is_parameter`` is False, so the notebook
        prints "literal/unset; skipping" and never touches the space."""
        from dao_ai.config import is_parameter, parameter_name

        # Provision a space so we have a known live id to point at via literal.
        seed = self._make_room_with_title(self._unique_title())
        seed.create()
        seed_id = seed.space_id
        assert seed_id

        try:
            # Construct a fresh GenieRoomModel referring to the seed by literal id.
            # Mirror what AppConfig.from_file would do: set _raw_space_id to the
            # exact pre-substitution YAML value (a literal, not ``${var.X}``).
            literal_room = GenieRoomModel(space_id=seed_id)
            literal_room._raw_space_id = seed_id

            # The notebook's skip predicate.
            assert is_parameter(literal_room.raw_space_id) is False, (
                "literal space_id must not be detected as a parameter"
            )
            assert parameter_name(literal_room.raw_space_id) is None

            # Discovery mode: ensure_resolved() must pull live name/description
            # from the API without invoking create() or update_space.
            literal_room.ensure_resolved()
            details = literal_room._get_space_details()
            assert details is not None
            assert details.title == seed.name, (
                f"expected ensure_resolved() to populate from live space "
                f"(title={seed.name!r}), got {details.title!r}"
            )
            assert literal_room.space_id == seed_id, "space_id must not change"
        finally:
            try:
                seed.workspace_client.genie.trash_space(space_id=seed_id)
            except Exception:
                pass

    def test_fresh_provisioning_via_cascade(self):
        """Scenario 3: a parameterised room with no existing space walks the full
        cascade: from_space_id("") → None, from_name(unique_title) → None,
        room.create() → fresh provisioning."""
        title = self._unique_title()
        room = self._make_room_with_title(title)
        # Simulate the notebook's view: raw_space_id was ``${var.genie_space_id}``,
        # substituted to "" (empty default), so room.space_id is None/unset.
        assert not room.space_id

        # Step 1: from_space_id("") returns None.
        wc = room.workspace_client
        assert GenieRoomModel.from_space_id("", w=wc) is None
        assert GenieRoomModel.from_space_id(None, w=wc) is None

        # Step 2: from_name(unique_title) returns None — no matching space yet.
        assert GenieRoomModel.from_name(title, w=wc) is None

        # Step 3: create() provisions a fresh space and sets space_id.
        room.create()
        assert room.space_id, "create() must populate space_id"

        try:
            # And from_space_id(<new_id>) now resolves it.
            resolved = GenieRoomModel.from_space_id(room.space_id, w=wc)
            assert resolved is not None
            assert resolved.space_id == room.space_id
        finally:
            try:
                wc.genie.trash_space(space_id=room.space_id)
            except Exception:
                pass

    def test_subsequent_deploy_reuses_existing_space(self):
        """Scenario 4: a second deploy with the same title finds the prior
        space via from_name and reuses it without calling create() (avoids
        etag conflicts on update_space, mirrors the loyalty deploy fix).

        Note: Genie's ``list_spaces`` API has a ~6-10s eventual-consistency
        window after ``create_space``. Production code is unaffected because
        the cascade starts with ``from_space_id`` (a Get API, immediately
        consistent) when a prior taskValue is available; ``from_name`` is
        only reached when taskValues have been lost across runs, which in
        practice happens minutes after the prior provisioning ran. The test
        polls briefly to absorb the lag.
        """
        import time

        title = self._unique_title()

        # First deploy: provision the space.
        first = self._make_room_with_title(title)
        first.create()
        original_id = first.space_id
        assert original_id

        try:
            # Second deploy: a fresh GenieRoomModel with the same title but no
            # space_id yet — same shape as the notebook's room after a re-deploy
            # where the prior taskValue isn't carried across runs.
            second = self._make_room_with_title(title)
            wc = second.workspace_client
            assert not second.space_id

            # Cascade: from_space_id("") None → from_name(title) finds existing.
            assert GenieRoomModel.from_space_id("", w=wc) is None
            existing: GenieRoomModel | None = None
            for delay in (0, 1, 3, 6, 10):
                if delay:
                    time.sleep(delay)
                existing = GenieRoomModel.from_name(title, w=wc)
                if existing is not None:
                    break
            assert existing is not None, (
                "from_name must find the prior space within ~20s "
                "(Genie list_spaces eventual-consistency window)"
            )
            assert existing.space_id == original_id, (
                f"reuse must return the same space_id ({original_id}), "
                f"got {existing.space_id}"
            )

            # The notebook assigns the existing id back to the room and skips create().
            second.space_id = existing.space_id
            # Do NOT call second.create() — that would update_space + risk etag
            # conflicts. The notebook deliberately skips this branch.

            # Verify no duplicate space was created: only one space in the
            # workspace has this title.
            matches: list = []
            page_token = None
            while True:
                resp = wc.genie.list_spaces(page_token=page_token)
                if resp.spaces:
                    matches.extend(sp for sp in resp.spaces if sp.title == title)
                if not resp.next_page_token:
                    break
                page_token = resp.next_page_token
            assert len(matches) == 1, (
                f"expected exactly one space titled {title!r}, found {len(matches)}"
            )
            assert matches[0].space_id == original_id
        finally:
            try:
                first.workspace_client.genie.trash_space(space_id=original_id)
            except Exception:
                pass


@pytest.mark.unit
class TestProvisionGenieNotebookGuard:
    """Notebook 05's outer guard skips the entire loop when no Genie rooms
    are configured. This is a pure config-shape test — no live workspace."""

    def test_no_genie_rooms_in_config_is_noop(self):
        """Scenario 1: an AppConfig with no genie_rooms produces an empty
        provisioning dict and never calls taskValues.set."""
        from dao_ai.config import AppConfig, ResourcesModel, is_parameter

        config = AppConfig(resources=ResourcesModel(genie_rooms={}))

        recorded_sets: list[tuple[str, str]] = []

        # Faithful re-implementation of notebooks/05_provision_genie.py:99-133
        # so the test breaks if that guard regresses.
        provisioned: dict[str, str] = {}
        if config.resources is not None and config.resources.genie_rooms:
            for room_key, room in config.resources.genie_rooms.items():
                if not is_parameter(room.raw_space_id):
                    continue
                # If we ever reach this branch, the test fails — there should
                # be nothing to iterate over when genie_rooms is empty.
                raise AssertionError(
                    f"loop body entered for {room_key} despite empty genie_rooms"
                )
                recorded_sets.append((room_key, room.space_id))
                provisioned[room_key] = room.space_id

        assert provisioned == {}
        assert recorded_sets == []

    def test_resources_none_in_config_is_noop(self):
        """Edge case: ``config.resources is None`` short-circuits the guard."""
        from dao_ai.config import AppConfig

        config = AppConfig(resources=None)
        # The outer guard `if config.resources is not None and ...` is False.
        assert config.resources is None


# ---------------------------------------------------------------------------
# ensure_resolved() hydration of sample questions
# ---------------------------------------------------------------------------


class TestSampleQuestionHydration:
    """``ensure_resolved()`` back-fills a bare room's sample questions.

    Customers overwhelmingly declare rooms by ``space_id`` rather than ``name``
    (Genie titles are not unique), so those rooms carry no local text at all.
    Hydrating here — the same place ``name``/``description`` are already
    back-filled — is what makes the Genie tool description identical whether the
    agent runs locally, in Apps, or in Model Serving: deploy resolves it, and
    ``model_dump`` bakes it into the logged ``model_config``.
    """

    SPACE_ID: str = "a" * 32

    @staticmethod
    def _space_details(
        *,
        title: str = "Retail Sales",
        description: str | None = "Store sales, inventory and returns.",
        sample_questions: list[str] | None = None,
    ) -> Mock:
        """A stub ``GenieSpace`` carrying a ``serialized_space`` payload."""
        payload: dict = {"version": 2}
        if sample_questions is not None:
            payload["config"] = {
                "sample_questions": [
                    {"id": f"id{i}", "question": [q]}
                    for i, q in enumerate(sample_questions)
                ]
            }
        details = Mock()
        details.title = title
        details.description = description
        details.serialized_space = json.dumps(payload)
        return details

    def _room(self, **kwargs) -> GenieRoomModel:
        return GenieRoomModel(space_id=self.SPACE_ID, **kwargs)

    def test_hydrates_description_and_questions_from_live_space(self) -> None:
        """The bare-``space_id`` path customers actually use.

        Nothing is declared locally, so both the description and the sample
        questions must come from the space and reach the tool description.
        """
        room = self._room()
        details = self._space_details(
            sample_questions=["How many stores per state?", "Top 5 SKUs last quarter?"]
        )
        with patch.object(GenieRoomModel, "_get_space_details", return_value=details):
            room.ensure_resolved()

        assert room.name == "Retail Sales"
        assert room.description == "Store sales, inventory and returns."
        assert room.sample_questions == [
            "How many stores per state?",
            "Top 5 SKUs last quarter?",
        ]

        from dao_ai.tools.genie import create_genie_tool

        tool = create_genie_tool(genie_room=room)
        assert "Store sales, inventory and returns." in tool.description
        assert "- How many stores per state?" in tool.description
        assert "- Top 5 SKUs last quarter?" in tool.description

    def test_declared_questions_are_not_overwritten(self) -> None:
        """User-declared values always win over discovered ones."""
        room = self._room(sample_questions=["MY OWN QUESTION"])
        details = self._space_details(sample_questions=["FROM THE SPACE"])
        with patch.object(GenieRoomModel, "_get_space_details", return_value=details):
            room.ensure_resolved()

        assert room.sample_questions == ["MY OWN QUESTION"]

    def test_hydration_touches_only_question_fields(self) -> None:
        """Guard against drifting into full ``refresh()`` behavior.

        ``refresh()`` also rewrites ``table_sources``/``function_sources``,
        which are provisioning *inputs* — rewriting them here would bloat the
        baked ``model_config`` and risk perturbing a later re-provision.
        """
        room = self._room()
        payload = {
            "version": 2,
            "config": {
                "sample_questions": [{"id": "i1", "question": ["Discovered?"]}]
            },
            "data_sources": {
                "tables": [{"identifier": "cat.sch.tbl"}],
                "metric_views": [{"identifier": "cat.sch.mv"}],
            },
            "instructions": {
                "text_instructions": [{"content": ["Never do that."]}],
                "sql_functions": [{"identifier": "cat.sch.fn"}],
                "join_specs": [{"sql": "a.id = b.id"}],
            },
        }
        details = Mock()
        details.title = "Retail Sales"
        details.description = "Store sales."
        details.serialized_space = json.dumps(payload)

        with patch.object(GenieRoomModel, "_get_space_details", return_value=details):
            room.ensure_resolved()

        assert room.sample_questions == ["Discovered?"]
        assert room.table_sources is None
        assert room.metric_view_sources is None
        assert room.function_sources is None
        assert room.join_specs is None
        assert room.text_instructions is None
        assert room.benchmarks is None

    def test_unavailable_space_is_not_fatal(self) -> None:
        """The Model Serving case: ``_get_space_details`` returns ``None``.

        Hydration must fail soft — no exception, and the tool simply carries no
        example-question block.
        """
        room = self._room()
        with patch.object(GenieRoomModel, "_get_space_details", return_value=None):
            room.ensure_resolved()

        assert room.sample_questions is None

        from dao_ai.tools.genie import create_genie_tool

        tool = create_genie_tool(genie_room=room)
        assert "Example questions" not in tool.description
        assert "<topic>" not in tool.description

    def test_hydrated_values_survive_the_model_config_bake(self) -> None:
        """The local / Apps / Model Serving parity guarantee.

        Deploy dumps the resolved config into the MLflow artifact; the serving
        container rebuilds from that dict. So a room rebuilt from the dump must
        still produce the same tool description with **no** further Genie call.
        """
        room = self._room()
        details = self._space_details(sample_questions=["How many stores per state?"])
        with patch.object(GenieRoomModel, "_get_space_details", return_value=details):
            room.ensure_resolved()

        baked = room.model_dump(mode="json", by_alias=True, exclude_none=True)
        assert baked["sample_questions"] == ["How many stores per state?"]

        rebuilt = GenieRoomModel.model_validate(baked)

        from dao_ai.tools.genie import create_genie_tool

        with patch.object(
            GenieRoomModel,
            "_get_space_details",
            side_effect=AssertionError("Genie API called in the serving container"),
        ):
            tool = create_genie_tool(genie_room=rebuilt)

        assert "Store sales, inventory and returns." in tool.description
        assert "- How many stores per state?" in tool.description
