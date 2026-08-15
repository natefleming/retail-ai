"""
Tests for ResourcesModel integration with GenieRoomModel.

This test suite verifies that tables and functions from Genie rooms are
automatically populated into the ResourcesModel during validation.
"""

import json
from unittest.mock import Mock, patch

import pytest

from dao_ai.config import (
    GenieRoomModel,
    ResourcesModel,
    TableModel,
    WarehouseModel,
)


@pytest.fixture
def mock_workspace_client():
    """Create a mock WorkspaceClient for testing."""
    mock_client = Mock()
    mock_client.genie = Mock()
    return mock_client


@pytest.fixture
def mock_genie_space_with_resources():
    """Create a mock GenieSpace with tables and functions."""
    mock_space = Mock()
    mock_space.space_id = "test-space-123"
    mock_space.title = "Test Genie Space"
    mock_space.description = "Test space with resources"
    mock_space.warehouse_id = "test-warehouse"

    # Real Databricks structure with tables and functions
    serialized_data = {
        "version": "1.0",
        "data_sources": {
            "tables": [
                {"identifier": "catalog.schema.customers", "column_configs": []},
                {"identifier": "catalog.schema.orders", "column_configs": []},
                {"identifier": "catalog.schema.products", "column_configs": []},
            ],
            "functions": [
                {"identifier": "catalog.schema.get_customer"},
                {"identifier": "catalog.schema.calculate_total"},
            ],
        },
    }
    mock_space.serialized_space = json.dumps(serialized_data)
    return mock_space


@pytest.fixture
def mock_genie_space_no_functions():
    """Create a mock GenieSpace with only tables."""
    mock_space = Mock()
    mock_space.space_id = "test-space-456"
    mock_space.title = "Test Genie Space (Tables Only)"
    mock_space.description = "Test space with only tables"
    mock_space.warehouse_id = "test-warehouse"

    serialized_data = {
        "version": "1.0",
        "data_sources": {
            "tables": [
                {"identifier": "catalog.schema.inventory"},
                {"identifier": "catalog.schema.suppliers"},
            ]
        },
    }
    mock_space.serialized_space = json.dumps(serialized_data)
    return mock_space


@pytest.mark.unit
class TestResourcesModelGenieIntegration:
    """Test suite for ResourcesModel Genie integration."""

    def test_genie_tables_and_functions_auto_populated(
        self, mock_workspace_client, mock_genie_space_with_resources
    ):
        """Test that tables and functions from Genie rooms are automatically added."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_resources
            )

            # Mock warehouse response
            mock_warehouse_response = Mock()
            mock_warehouse_response.name = "Test Warehouse"
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            # Create GenieRoomModel and resolve before passing to ResourcesModel
            genie_model = GenieRoomModel(
                name="my-genie-room", space_id="test-space-123"
            )
            genie_model.ensure_resolved()

            # Create ResourcesModel with a genie room
            resources = ResourcesModel(genie_rooms={"my_genie": genie_model})

            # Verify tables were added
            assert len(resources.tables) == 3
            assert "my_genie_room_catalog_schema_customers" in resources.tables
            assert "my_genie_room_catalog_schema_orders" in resources.tables
            assert "my_genie_room_catalog_schema_products" in resources.tables

            # Verify table names are correct
            assert (
                resources.tables["my_genie_room_catalog_schema_customers"].name
                == "catalog.schema.customers"
            )
            assert (
                resources.tables["my_genie_room_catalog_schema_orders"].name
                == "catalog.schema.orders"
            )
            assert (
                resources.tables["my_genie_room_catalog_schema_products"].name
                == "catalog.schema.products"
            )

            # Verify functions were added
            assert len(resources.functions) == 2
            assert "my_genie_room_catalog_schema_get_customer" in resources.functions
            assert "my_genie_room_catalog_schema_calculate_total" in resources.functions

            # Verify function names are correct
            assert (
                resources.functions["my_genie_room_catalog_schema_get_customer"].name
                == "catalog.schema.get_customer"
            )
            assert (
                resources.functions["my_genie_room_catalog_schema_calculate_total"].name
                == "catalog.schema.calculate_total"
            )

    def test_genie_resources_with_existing_tables(
        self, mock_workspace_client, mock_genie_space_with_resources
    ):
        """Test that existing manually-defined tables are preserved."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_resources
            )

            # Mock warehouse response
            mock_warehouse_response = Mock()
            mock_warehouse_response.name = "Test Warehouse"
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            # Create GenieRoomModel and resolve before passing to ResourcesModel
            genie_model = GenieRoomModel(
                name="my-genie-room", space_id="test-space-123"
            )
            genie_model.ensure_resolved()

            # Create ResourcesModel with existing tables and a genie room
            resources = ResourcesModel(
                tables={"manual_table": TableModel(name="catalog.schema.manual_table")},
                genie_rooms={"my_genie": genie_model},
            )

            # Verify manual table is preserved
            assert "manual_table" in resources.tables
            assert (
                resources.tables["manual_table"].name == "catalog.schema.manual_table"
            )

            # Verify genie tables were added
            assert len(resources.tables) == 4  # 1 manual + 3 from genie
            assert "my_genie_room_catalog_schema_customers" in resources.tables

    def test_genie_resources_deduplication(
        self, mock_workspace_client, mock_genie_space_with_resources
    ):
        """Test that duplicate tables/functions are not added."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_resources
            )

            # Mock warehouse response
            mock_warehouse_response = Mock()
            mock_warehouse_response.name = "Test Warehouse"
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            # Create GenieRoomModel and resolve before passing to ResourcesModel
            genie_model = GenieRoomModel(
                name="my-genie-room", space_id="test-space-123"
            )
            genie_model.ensure_resolved()

            # Create ResourcesModel with a table that matches one from Genie
            resources = ResourcesModel(
                tables={
                    "existing_customers": TableModel(name="catalog.schema.customers")
                },
                genie_rooms={"my_genie": genie_model},
            )

            # Verify the manually-defined table is kept
            assert "existing_customers" in resources.tables

            # Verify the duplicate from Genie was not added
            # So we should have: existing_customers + 2 unique from genie (orders, products)
            assert len(resources.tables) == 3  # 1 manual + 2 unique from genie
            assert "my_genie_room_catalog_schema_orders" in resources.tables
            assert "my_genie_room_catalog_schema_products" in resources.tables

            # Verify the table names
            assert (
                resources.tables["my_genie_room_catalog_schema_orders"].name
                == "catalog.schema.orders"
            )
            assert (
                resources.tables["my_genie_room_catalog_schema_products"].name
                == "catalog.schema.products"
            )

    def test_multiple_genie_rooms(
        self,
        mock_workspace_client,
        mock_genie_space_with_resources,
        mock_genie_space_no_functions,
    ):
        """Test that resources from multiple Genie rooms are collected."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):

            def get_space_side_effect(space_id, **kwargs):
                if space_id == "test-space-123":
                    return mock_genie_space_with_resources
                elif space_id == "test-space-456":
                    return mock_genie_space_no_functions
                return None

            mock_workspace_client.genie.get_space.side_effect = get_space_side_effect

            # Mock warehouse response
            mock_warehouse_response = Mock()
            mock_warehouse_response.name = "Test Warehouse"
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            # Create GenieRoomModels and resolve before passing to ResourcesModel
            genie_model_1 = GenieRoomModel(
                name="genie-room-1", space_id="test-space-123"
            )
            genie_model_1.ensure_resolved()
            genie_model_2 = GenieRoomModel(
                name="genie-room-2", space_id="test-space-456"
            )
            genie_model_2.ensure_resolved()

            # Create ResourcesModel with multiple genie rooms
            resources = ResourcesModel(
                genie_rooms={
                    "genie_room_1": genie_model_1,
                    "genie_room_2": genie_model_2,
                }
            )

            # Verify tables from both rooms were added
            assert len(resources.tables) == 5  # 3 from room1 + 2 from room2

            # From room 1
            assert "genie_room_1_catalog_schema_customers" in resources.tables
            assert "genie_room_1_catalog_schema_orders" in resources.tables
            assert "genie_room_1_catalog_schema_products" in resources.tables
            assert (
                resources.tables["genie_room_1_catalog_schema_customers"].name
                == "catalog.schema.customers"
            )
            assert (
                resources.tables["genie_room_1_catalog_schema_orders"].name
                == "catalog.schema.orders"
            )
            assert (
                resources.tables["genie_room_1_catalog_schema_products"].name
                == "catalog.schema.products"
            )

            # From room 2
            assert "genie_room_2_catalog_schema_inventory" in resources.tables
            assert "genie_room_2_catalog_schema_suppliers" in resources.tables
            assert (
                resources.tables["genie_room_2_catalog_schema_inventory"].name
                == "catalog.schema.inventory"
            )
            assert (
                resources.tables["genie_room_2_catalog_schema_suppliers"].name
                == "catalog.schema.suppliers"
            )

            # Verify functions from room 1 (room 2 has none)
            assert len(resources.functions) == 2
            assert "genie_room_1_catalog_schema_get_customer" in resources.functions
            assert "genie_room_1_catalog_schema_calculate_total" in resources.functions
            assert (
                resources.functions["genie_room_1_catalog_schema_get_customer"].name
                == "catalog.schema.get_customer"
            )
            assert (
                resources.functions["genie_room_1_catalog_schema_calculate_total"].name
                == "catalog.schema.calculate_total"
            )

    def test_genie_resources_inherit_authentication(
        self, mock_workspace_client, mock_genie_space_with_resources
    ):
        """Test that tables/functions from Genie inherit authentication from the room."""
        from dao_ai.config import ServicePrincipalModel

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_resources
            )

            service_principal = ServicePrincipalModel(
                client_id="test-client-id", client_secret="test-client-secret"
            )

            # Create GenieRoomModel and resolve before passing to ResourcesModel
            genie_model = GenieRoomModel(
                name="my-genie-room",
                space_id="test-space-123",
                on_behalf_of_user=True,
                service_principal=service_principal,
                workspace_host="https://test.databricks.com",
            )
            genie_model.ensure_resolved()

            # Create ResourcesModel with authenticated genie room
            resources = ResourcesModel(genie_rooms={"my_genie": genie_model})

            # Verify tables inherit authentication
            for table_key, table in resources.tables.items():
                assert table.on_behalf_of_user
                assert table.service_principal == service_principal
                assert table.workspace_host == "https://test.databricks.com"

            # Verify functions inherit authentication including OBO (UC serverless uses user identity)
            for function_key, function in resources.functions.items():
                assert function.on_behalf_of_user
                assert function.service_principal == service_principal
                assert function.workspace_host == "https://test.databricks.com"

    def test_empty_genie_rooms(self):
        """Test that ResourcesModel works with no genie rooms."""
        resources = ResourcesModel(
            tables={"manual_table": TableModel(name="catalog.schema.test")}
        )

        # Should only have the manual table
        assert len(resources.tables) == 1
        assert "manual_table" in resources.tables
        assert len(resources.functions) == 0

    def test_genie_room_with_no_resources(self, mock_workspace_client):
        """Test handling of Genie room with no tables or functions."""
        mock_space = Mock()
        mock_space.space_id = "test-space-empty"
        mock_space.title = "Empty Space"
        mock_space.description = None
        mock_space.warehouse_id = "test-warehouse"
        mock_space.serialized_space = json.dumps({"data_sources": {}})

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = mock_space

            # Create GenieRoomModel and resolve before passing to ResourcesModel
            genie_model = GenieRoomModel(
                name="empty-genie-room", space_id="test-space-empty"
            )
            genie_model.ensure_resolved()

            resources = ResourcesModel(genie_rooms={"empty_genie": genie_model})

            # Should have no tables or functions
            assert len(resources.tables) == 0
            assert len(resources.functions) == 0

    def test_genie_warehouses_auto_populated(
        self, mock_workspace_client, mock_genie_space_with_resources
    ):
        """Test that warehouses from Genie rooms are automatically added."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_resources
            )

            # Mock warehouse response
            mock_warehouse_response = Mock()
            mock_warehouse_response.name = "Test Warehouse"
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            # Create GenieRoomModel and resolve before passing to ResourcesModel
            genie_model = GenieRoomModel(
                name="my-genie-room", space_id="test-space-123"
            )
            genie_model.ensure_resolved()

            # Create ResourcesModel with a genie room
            resources = ResourcesModel(genie_rooms={"my_genie": genie_model})

            # Verify warehouse was added, keyed off the genie_rooms mapping key
            # plus the ``_warehouse`` suffix that keeps it from colliding with the
            # room's own App resource name.
            assert len(resources.warehouses) == 1
            assert "my_genie_warehouse" in resources.warehouses

            # Verify warehouse properties
            warehouse = resources.warehouses["my_genie_warehouse"]
            assert warehouse.name == "Test Warehouse"
            assert warehouse.warehouse_id == "test-warehouse"

    def test_genie_warehouses_with_existing_warehouses(
        self, mock_workspace_client, mock_genie_space_with_resources
    ):
        """Test that existing manually-defined warehouses are preserved."""
        from dao_ai.config import WarehouseModel

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_resources
            )

            # Mock warehouse response
            mock_warehouse_response = Mock()
            mock_warehouse_response.name = "Genie Warehouse"
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            # Create GenieRoomModel and resolve before passing to ResourcesModel
            genie_model = GenieRoomModel(
                name="my-genie-room", space_id="test-space-123"
            )
            genie_model.ensure_resolved()

            # Create ResourcesModel with existing warehouse and a genie room
            resources = ResourcesModel(
                warehouses={
                    "manual_warehouse": WarehouseModel(
                        name="manual-warehouse", warehouse_id="manual-wh-123"
                    )
                },
                genie_rooms={"my_genie": genie_model},
            )

            # Verify manual warehouse is preserved
            assert "manual_warehouse" in resources.warehouses
            assert (
                resources.warehouses["manual_warehouse"].warehouse_id == "manual-wh-123"
            )

            # Verify genie warehouse was added
            assert len(resources.warehouses) == 2
            assert "my_genie_warehouse" in resources.warehouses

    def test_genie_warehouses_deduplication(
        self, mock_workspace_client, mock_genie_space_with_resources
    ):
        """Test that duplicate warehouses are not added."""
        from dao_ai.config import WarehouseModel

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_resources
            )

            # Mock warehouse response
            mock_warehouse_response = Mock()
            mock_warehouse_response.name = "Test Warehouse"
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            # Create GenieRoomModel and resolve before passing to ResourcesModel
            genie_model = GenieRoomModel(
                name="my-genie-room", space_id="test-space-123"
            )
            genie_model.ensure_resolved()

            # Create ResourcesModel with a warehouse that matches the Genie warehouse_id
            resources = ResourcesModel(
                warehouses={
                    "existing_warehouse": WarehouseModel(
                        name="existing-warehouse", warehouse_id="test-warehouse"
                    )
                },
                genie_rooms={"my_genie": genie_model},
            )

            # Verify the manually-defined warehouse is kept
            assert "existing_warehouse" in resources.warehouses

            # Verify the duplicate from Genie was not added
            assert len(resources.warehouses) == 1
            assert (
                resources.warehouses["existing_warehouse"].warehouse_id
                == "test-warehouse"
            )

    def test_multiple_genie_rooms_with_warehouses(
        self,
        mock_workspace_client,
        mock_genie_space_with_resources,
        mock_genie_space_no_functions,
    ):
        """Test that warehouses from multiple Genie rooms are collected."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):

            def get_space_side_effect(space_id, **kwargs):
                if space_id == "test-space-123":
                    return mock_genie_space_with_resources
                elif space_id == "test-space-456":
                    return mock_genie_space_no_functions
                return None

            mock_workspace_client.genie.get_space.side_effect = get_space_side_effect

            # Mock warehouse responses for different warehouse IDs
            def get_warehouse_side_effect(warehouse_id):
                mock_response = Mock()
                if warehouse_id == "test-warehouse":
                    mock_response.name = "Warehouse 1"
                    mock_response.description = "First warehouse"
                return mock_response

            mock_workspace_client.warehouses.get.side_effect = get_warehouse_side_effect

            # Create GenieRoomModels and resolve before passing to ResourcesModel
            genie_model_1 = GenieRoomModel(
                name="genie-room-1", space_id="test-space-123"
            )
            genie_model_1.ensure_resolved()
            genie_model_2 = GenieRoomModel(
                name="genie-room-2", space_id="test-space-456"
            )
            genie_model_2.ensure_resolved()

            # Create ResourcesModel with multiple genie rooms
            resources = ResourcesModel(
                genie_rooms={
                    "genie_room_1": genie_model_1,
                    "genie_room_2": genie_model_2,
                }
            )

            # Both rooms share the same warehouse, so only one should be added
            assert len(resources.warehouses) == 1
            assert "genie_room_1_warehouse" in resources.warehouses

    def test_genie_warehouses_inherit_authentication(
        self, mock_workspace_client, mock_genie_space_with_resources
    ):
        """Test that warehouses from Genie inherit authentication from the room."""
        from dao_ai.config import ServicePrincipalModel

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_resources
            )

            # Mock warehouse response
            mock_warehouse_response = Mock()
            mock_warehouse_response.name = "Test Warehouse"
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            service_principal = ServicePrincipalModel(
                client_id="test-client-id", client_secret="test-client-secret"
            )

            # Create GenieRoomModel and resolve before passing to ResourcesModel.
            # Deliberately NOT on_behalf_of_user: those rooms query as the calling
            # user and contribute no warehouse at all (the app SP is never granted
            # CAN_USE on their behalf), so an OBO room here would leave the loop
            # below with nothing to assert. That case is covered by
            # TestGenieWarehouseDiscoveryAfterResolution's OBO test.
            genie_model = GenieRoomModel(
                name="my-genie-room",
                space_id="test-space-123",
                service_principal=service_principal,
                workspace_host="https://test.databricks.com",
            )
            genie_model.ensure_resolved()

            # Create ResourcesModel with authenticated genie room
            resources = ResourcesModel(genie_rooms={"my_genie": genie_model})

            # Verify warehouses inherit authentication
            assert resources.warehouses
            for warehouse_key, warehouse in resources.warehouses.items():
                assert not warehouse.on_behalf_of_user
                assert warehouse.service_principal == service_principal
                assert warehouse.workspace_host == "https://test.databricks.com"

    def test_genie_room_with_no_warehouse(self, mock_workspace_client):
        """Test handling of Genie room with no warehouse_id."""
        mock_space = Mock()
        mock_space.space_id = "test-space-no-wh"
        mock_space.title = "No Warehouse Space"
        mock_space.description = None
        mock_space.warehouse_id = None
        mock_space.serialized_space = json.dumps({"data_sources": {}})

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = mock_space

            # Create GenieRoomModel and resolve before passing to ResourcesModel
            genie_model = GenieRoomModel(
                name="no-warehouse-room", space_id="test-space-no-wh"
            )
            genie_model.ensure_resolved()

            resources = ResourcesModel(genie_rooms={"no_warehouse_genie": genie_model})

            # Should have no warehouses
            assert len(resources.warehouses) == 0

    def test_genie_warehouse_api_error_handling(
        self, mock_workspace_client, mock_genie_space_with_resources
    ):
        """Test that warehouse API errors are handled gracefully."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_resources
            )

            # Mock warehouse API to raise an error
            mock_workspace_client.warehouses.get.side_effect = Exception(
                "Warehouse API error"
            )

            # Create GenieRoomModel and resolve before passing to ResourcesModel
            genie_model = GenieRoomModel(
                name="my-genie-room", space_id="test-space-123"
            )
            genie_model.ensure_resolved()

            # Should not raise an exception
            resources = ResourcesModel(genie_rooms={"my_genie": genie_model})

            # No warehouses should be added due to API error
            assert len(resources.warehouses) == 0


@pytest.mark.unit
class TestGenieWarehouseDiscoveryAfterResolution:
    """Discovery has to run *after* the Genie rooms are resolved.

    The tests above construct a ``GenieRoomModel``, call ``ensure_resolved()`` by
    hand, and only then build the ``ResourcesModel`` — the inverse of what
    ``AppConfig.from_file`` does, and the reason this whole class of bug went
    unnoticed. In real loading order the ``update_genie_warehouses`` validator
    fires while the room is still unresolved, so ``discover_warehouse`` goes
    through ``_get_space_details``, which returns ``None`` when
    ``not self._resolved`` — and a room referencing an existing space without an
    inline ``warehouse:`` contributed nothing. The app's service principal never
    got ``CAN_USE``, and the first Genie question 500'd.

    So these load through ``AppConfig.from_file(initialize=True)``, where
    ``_resolve_all_resources()`` back-fills again once resolution has run — the
    choke point every deploy path funnels through, ``initialize()`` included.
    """

    def _config(self, tmp_path, body: str):
        """Load a config through the real ``from_file`` ordering."""
        from dao_ai.config import AppConfig

        path = tmp_path / "config.yaml"
        path.write_text(body, encoding="utf-8")
        return AppConfig.from_file(str(path))

    @pytest.fixture
    def live_space(self, mock_workspace_client, mock_genie_space_with_resources):
        """A readable space bound to ``test-warehouse``."""
        mock_workspace_client.genie.get_space.return_value = (
            mock_genie_space_with_resources
        )
        warehouse_response = Mock()
        warehouse_response.name = "Genie Warehouse"
        mock_workspace_client.warehouses.get.return_value = warehouse_response
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            yield mock_workspace_client

    def test_bare_space_id_room_contributes_a_warehouse(self, tmp_path, live_space):
        """The common config shape: an existing space by id, no ``warehouse:``,
        no ``name:``. Before the back-fill in ``initialize()`` this produced zero
        warehouses.
        """
        config = self._config(
            tmp_path,
            "resources:\n  genie_rooms:\n    orders:\n      space_id: test-space-123\n",
        )

        assert "orders_warehouse" in config.resources.warehouses
        assert (
            config.resources.warehouses["orders_warehouse"].warehouse_id
            == "test-warehouse"
        )

    def test_inline_warehouse_without_a_room_name_does_not_raise(self, tmp_path):
        """Keying off the mapping key rather than ``genie_room.name`` also fixes a
        parse-time crash: the old ``"_".join([genie_room.name, ...])`` raised
        TypeError for a room whose ``name`` is None until resolution runs.
        """
        from dao_ai.config import AppConfig

        body = (
            "resources:\n"
            "  genie_rooms:\n"
            "    orders:\n"
            "      space_id: test-space-123\n"
            "      warehouse:\n"
            "        name: inline-warehouse\n"
            "        warehouse_id: inline-wh-1\n"
        )
        path = tmp_path / "config.yaml"
        path.write_text(body, encoding="utf-8")

        # No workspace client at all — an inline warehouse needs no network, and
        # this must not even try.
        config = AppConfig.from_file(str(path), initialize=False)

        assert (
            config.resources.warehouses["orders_warehouse"].warehouse_id == "inline-wh-1"
        )

    def test_discovered_warehouse_reaches_the_grant_plan(self, tmp_path, live_space):
        """``build_grant_plan`` is what actually issues the ``CAN_USE`` the failing
        Genie query needed, and it reads ``resources.warehouses`` — not the room.
        """
        from dao_ai.service_principal import build_grant_plan

        config = self._config(
            tmp_path,
            "resources:\n  genie_rooms:\n    orders:\n      space_id: test-space-123\n",
        )
        plan = build_grant_plan(config, principal="sp-client-id")

        warehouse_grants = [
            g
            for g in plan.grants
            if g.kind == "warehouse" and g.target == "test-warehouse"
        ]
        assert warehouse_grants, "no CAN_USE grant for the Genie warehouse"
        assert warehouse_grants[0].privileges == ["CAN_USE"]

    def test_discovered_warehouse_reaches_both_resource_generators(
        self, tmp_path, live_space
    ):
        """``generate_deployment_resources`` is the SDK Apps path (``workflow up``),
        which had no Genie warehouse fallback of its own;
        ``generate_app_resources`` is the DABs path.
        """
        from dao_ai.apps.resources import (
            generate_app_resources,
            generate_deployment_resources,
        )

        config = self._config(
            tmp_path,
            "resources:\n  genie_rooms:\n    orders:\n      space_id: test-space-123\n",
        )

        app_warehouses = [
            r
            for r in generate_app_resources(config)
            if r.get("type") == "sql-warehouse"
        ]
        assert [r["sql_warehouse_id"] for r in app_warehouses] == ["test-warehouse"]

        deployment_warehouses = [
            r
            for r in generate_deployment_resources(config)
            if "sql_warehouse" in r or r.get("type") == "sql-warehouse"
        ]
        assert deployment_warehouses, "SDK Apps deploy carries no warehouse resource"

    def test_obo_room_discovers_nothing(self, tmp_path, live_space):
        """An on-behalf-of-user room queries as the caller, whose own warehouse
        access applies; the app SP is never granted CAN_USE for them, so there is
        nothing to *discover* a warehouse for.
        """
        config = self._config(
            tmp_path,
            "resources:\n"
            "  genie_rooms:\n"
            "    orders:\n"
            "      space_id: test-space-123\n"
            "      on_behalf_of_user: true\n",
        )

        assert config.resources.warehouses == {}

    def test_obo_room_keeps_its_inline_warehouse(self, tmp_path, live_space):
        """Skipping *discovery* for OBO rooms must not also drop a warehouse the
        author declared inline.

        An inline warehouse on an OBO room is how that room's ``sql`` user API
        scope gets emitted: ``generate_user_api_scopes`` and ``build_auth_policy``
        both read OBO warehouses out of ``resources.warehouses``, never off the
        room. Filtering it out here silently breaks OBO Genie at runtime — the
        forwarded user token arrives without the scope needed to run the query.
        """
        from dao_ai.apps.resources import (
            generate_app_resources,
            generate_user_api_scopes,
        )

        config = self._config(
            tmp_path,
            "resources:\n"
            "  genie_rooms:\n"
            "    orders:\n"
            "      space_id: test-space-123\n"
            "      on_behalf_of_user: true\n"
            "      warehouse:\n"
            "        name: obo-warehouse\n"
            "        warehouse_id: obo-wh-1\n"
            "        on_behalf_of_user: true\n",
        )

        warehouse = config.resources.warehouses["orders_warehouse"]
        assert warehouse.warehouse_id == "obo-wh-1"
        assert warehouse.on_behalf_of_user

        assert "sql" in generate_user_api_scopes(config)
        # ...and it is still NOT an App resource: OBO means no SP grant.
        assert not [
            r for r in generate_app_resources(config) if r["type"] == "sql-warehouse"
        ]

    def test_second_room_inline_warehouse_survives_a_shared_space_id(
        self, tmp_path, live_space
    ):
        """The per-space dedupe is a *discovery* optimization, so it must not gate
        a room that declares its warehouse inline and needs no lookup at all.

        Two rooms over one space where only the second names a warehouse: gating
        on ``space_id`` first would skip the second room entirely and lose its
        explicit declaration to the first room's discovered value.
        """
        config = self._config(
            tmp_path,
            "resources:\n"
            "  genie_rooms:\n"
            "    orders:\n"
            "      space_id: test-space-123\n"
            "    orders_alt:\n"
            "      space_id: test-space-123\n"
            "      warehouse:\n"
            "        name: explicit-warehouse\n"
            "        warehouse_id: explicit-wh-1\n",
        )

        ids = {w.warehouse_id for w in config.resources.warehouses.values()}
        assert ids == {"test-warehouse", "explicit-wh-1"}

    def test_derived_key_cannot_collide_with_the_room_resource_name(
        self, tmp_path, live_space
    ):
        """A warehouse key and a Genie room key both become an App resource
        *name*, and the DABs bundle has no uniquify pass — so keying the derived
        warehouse on the bare room key emitted two resources with one name, and
        the ``value_from`` env bindings became ambiguous about which they meant.
        """
        from dao_ai.apps.bundle import _convert_to_bundle_resources
        from dao_ai.apps.resources import generate_app_resources

        config = self._config(
            tmp_path,
            "resources:\n  genie_rooms:\n    orders:\n      space_id: test-space-123\n",
        )

        app_resources = generate_app_resources(config)
        names = [r["name"] for r in app_resources]
        assert len(names) == len(set(names)), f"duplicate App resource names: {names}"
        assert {"orders", "orders_warehouse"} <= set(names)

        bundle_names = [r["name"] for r in _convert_to_bundle_resources(app_resources)]
        assert len(bundle_names) == len(set(bundle_names))

    def test_a_taken_derived_key_is_suffixed_not_clobbered(self, tmp_path, live_space):
        """A hand-declared warehouse may already own ``<room>_warehouse``. Pointing
        that key at the discovered warehouse instead would silently retarget it.
        """
        config = self._config(
            tmp_path,
            "resources:\n"
            "  warehouses:\n"
            "    orders_warehouse:\n"
            "      name: hand-declared\n"
            "      warehouse_id: hand-wh-1\n"
            "  genie_rooms:\n"
            "    orders:\n"
            "      space_id: test-space-123\n",
        )

        assert config.resources.warehouses["orders_warehouse"].warehouse_id == "hand-wh-1"
        assert {w.warehouse_id for w in config.resources.warehouses.values()} == {
            "hand-wh-1",
            "test-warehouse",
        }

    def test_backfill_runs_on_every_deploy_path_not_just_initialize(
        self, tmp_path, live_space
    ):
        """``initialize()`` is not the only entry point.

        ``agent up --mode apps``, ``--direct``, ``agent build`` and
        ``service-principal grant -c`` all load with ``initialize=False`` and call
        ``_resolve_all_resources()`` themselves, so hooking discovery into
        ``initialize()`` alone left exactly the deploy paths that need the
        ``CAN_USE`` grant without it.
        """
        from dao_ai.config import AppConfig

        path = tmp_path / "config.yaml"
        path.write_text(
            "resources:\n  genie_rooms:\n    orders:\n      space_id: test-space-123\n",
            encoding="utf-8",
        )
        config = AppConfig.from_file(str(path), initialize=False)
        assert config.resources.warehouses == {}, "resolution has not run yet"

        config._resolve_all_resources()

        assert (
            config.resources.warehouses["orders_warehouse"].warehouse_id
            == "test-warehouse"
        )

    def test_discovery_is_deduped_per_space_and_idempotent(self, tmp_path, live_space):
        """Two rooms over one space (different tool descriptions, same data) must
        not double-look-up, and re-running the back-fill must add nothing.
        """
        config = self._config(
            tmp_path,
            "resources:\n"
            "  genie_rooms:\n"
            "    orders:\n"
            "      space_id: test-space-123\n"
            "    orders_alt:\n"
            "      space_id: test-space-123\n",
        )

        assert len(config.resources.warehouses) == 1
        calls_before = live_space.warehouses.get.call_count

        config.resources.backfill_genie_warehouses()

        assert len(config.resources.warehouses) == 1
        assert live_space.warehouses.get.call_count == calls_before

    def test_unreadable_space_still_loads(self, tmp_path, mock_workspace_client):
        """A space the deploying identity cannot inspect leaves the config exactly
        as written. Discovery is best-effort — it must never fail a load.
        """
        mock_workspace_client.genie.get_space.side_effect = Exception("403 Forbidden")

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            config = self._config(
                tmp_path,
                "resources:\n"
                "  genie_rooms:\n"
                "    orders:\n"
                "      space_id: test-space-123\n",
            )

        assert config.resources.warehouses == {}

    def test_declared_warehouse_is_not_duplicated(self, tmp_path, live_space):
        """A hand-declared warehouse for the same id wins; the back-fill adds no
        second entry pointing at it.
        """
        config = self._config(
            tmp_path,
            "resources:\n"
            "  warehouses:\n"
            "    shared:\n"
            "      name: shared-warehouse\n"
            "      warehouse_id: test-warehouse\n"
            "  genie_rooms:\n"
            "    orders:\n"
            "      space_id: test-space-123\n",
        )

        assert list(config.resources.warehouses) == ["shared"]


class TestBackfillGenieWarehousesReviewFindings:
    """Regression tests for the six review findings on Genie warehouse backfill.

    Each stubs ``discover_warehouse`` rather than the workspace client, so the
    subject under test is the backfill's own bookkeeping: what it caches, what it
    writes back, and how it names the key.
    """

    @staticmethod
    def _room(warehouse_id: str | None = None, **kwargs):
        """A room whose discovery is deterministic and counted."""
        calls: list[str] = []

        class _Room(GenieRoomModel):
            def discover_warehouse(self):
                calls.append(str(self.space_id))
                if warehouse_id is None:
                    raise PermissionError("identity cannot read this space")
                return WarehouseModel(name="discovered", warehouse_id=warehouse_id)

        room = _Room(**kwargs)
        return room, calls

    def test_discovered_warehouse_is_written_back_to_the_room(self) -> None:
        """Finding 1: without the write-back, every reload re-issues ``get_space``.

        The result was added to ``warehouses`` but never to the room, so a fresh
        parse rediscovered it and then dropped the answer at the dedupe check —
        paying cold-start latency for something the baked config already knew.
        """
        room, calls = self._room("disc-id", space_id="sp-1")
        resources = ResourcesModel(genie_rooms={"orders": room})
        resources.backfill_genie_warehouses()

        assert resources.genie_rooms["orders"].warehouse is not None
        assert resources.genie_rooms["orders"].warehouse.warehouse_id == "disc-id"

        # A second pass takes the inline branch and costs no further lookup.
        before = len(calls)
        resources.backfill_genie_warehouses()
        assert len(calls) == before

    def test_one_identitys_failure_does_not_suppress_another(self) -> None:
        """Finding 3: the cache keyed on ``space_id`` alone, and was written
        *before* the lookup, so the first room to fail claimed the space and every
        later room over it was skipped — including one holding credentials that
        could read it.
        """
        blocked, blocked_calls = self._room(None, space_id="shared")
        allowed, allowed_calls = self._room(
            "via-pat", space_id="shared", pat="scope/key"
        )

        resources = ResourcesModel(
            genie_rooms={"a_no_permission": blocked, "b_has_pat": allowed}
        )
        resources.backfill_genie_warehouses()

        assert allowed_calls, "the second identity was never given a turn"
        assert list(resources.warehouses) == ["b_has_pat_warehouse"]

    def test_successful_space_is_not_looked_up_twice(self) -> None:
        """The caching this finding tightened must still hold per identity."""
        room, calls = self._room("disc-id", space_id="sp-2")
        resources = ResourcesModel(genie_rooms={"orders": room})
        resources.backfill_genie_warehouses()
        resources.backfill_genie_warehouses()

        assert len(calls) == 1

    def test_differently_cased_existing_key_counts_as_taken(self) -> None:
        """Finding 6: ``warehouse_key`` is normalized but the collision guard
        compared it against config keys as written, so ``Orders_Warehouse`` did
        not register as taken and a colliding sibling was added beside it.
        """
        room, _ = self._room("zzz", space_id="sp-3")
        resources = ResourcesModel(
            genie_rooms={"orders": room},
            warehouses={
                "Orders_Warehouse": WarehouseModel(name="hand", warehouse_id="hand-id")
            },
        )
        resources.backfill_genie_warehouses()

        assert "orders_warehouse" not in resources.warehouses
        assert resources.warehouses["Orders_Warehouse"].warehouse_id == "hand-id"
        assert "orders_zzz" in resources.warehouses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
