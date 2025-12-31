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

            # Create ResourcesModel with a genie room
            resources = ResourcesModel(
                genie_rooms={
                    "my_genie": GenieRoomModel(
                        name="my-genie-room", space_id="test-space-123"
                    )
                }
            )

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

            # Create ResourcesModel with existing tables and a genie room
            resources = ResourcesModel(
                tables={"manual_table": TableModel(name="catalog.schema.manual_table")},
                genie_rooms={
                    "my_genie": GenieRoomModel(
                        name="my-genie-room", space_id="test-space-123"
                    )
                },
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

            # Create ResourcesModel with a table that matches one from Genie
            resources = ResourcesModel(
                tables={
                    "existing_customers": TableModel(name="catalog.schema.customers")
                },
                genie_rooms={
                    "my_genie": GenieRoomModel(
                        name="my-genie-room", space_id="test-space-123"
                    )
                },
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

            # Create ResourcesModel with multiple genie rooms
            resources = ResourcesModel(
                genie_rooms={
                    "genie_room_1": GenieRoomModel(
                        name="genie-room-1", space_id="test-space-123"
                    ),
                    "genie_room_2": GenieRoomModel(
                        name="genie-room-2", space_id="test-space-456"
                    ),
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

            # Create ResourcesModel with authenticated genie room
            resources = ResourcesModel(
                genie_rooms={
                    "my_genie": GenieRoomModel(
                        name="my-genie-room",
                        space_id="test-space-123",
                        on_behalf_of_user=True,
                        service_principal=service_principal,
                        workspace_host="https://test.databricks.com",
                    )
                }
            )

            # Verify tables inherit authentication
            for table_key, table in resources.tables.items():
                assert table.on_behalf_of_user
                assert table.service_principal == service_principal
                assert table.workspace_host == "https://test.databricks.com"

            # Verify functions inherit authentication
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

            resources = ResourcesModel(
                genie_rooms={
                    "empty_genie": GenieRoomModel(
                        name="empty-genie-room", space_id="test-space-empty"
                    )
                }
            )

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

            # Create ResourcesModel with a genie room
            resources = ResourcesModel(
                genie_rooms={
                    "my_genie": GenieRoomModel(
                        name="my-genie-room", space_id="test-space-123"
                    )
                }
            )

            # Verify warehouse was added
            assert len(resources.warehouses) == 1
            assert "my_genie_room_test_warehouse" in resources.warehouses

            # Verify warehouse properties
            warehouse = resources.warehouses["my_genie_room_test_warehouse"]
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

            # Create ResourcesModel with existing warehouse and a genie room
            resources = ResourcesModel(
                warehouses={
                    "manual_warehouse": WarehouseModel(
                        name="manual-warehouse", warehouse_id="manual-wh-123"
                    )
                },
                genie_rooms={
                    "my_genie": GenieRoomModel(
                        name="my-genie-room", space_id="test-space-123"
                    )
                },
            )

            # Verify manual warehouse is preserved
            assert "manual_warehouse" in resources.warehouses
            assert (
                resources.warehouses["manual_warehouse"].warehouse_id == "manual-wh-123"
            )

            # Verify genie warehouse was added
            assert len(resources.warehouses) == 2
            assert "my_genie_room_test_warehouse" in resources.warehouses

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

            # Create ResourcesModel with a warehouse that matches the Genie warehouse_id
            resources = ResourcesModel(
                warehouses={
                    "existing_warehouse": WarehouseModel(
                        name="existing-warehouse", warehouse_id="test-warehouse"
                    )
                },
                genie_rooms={
                    "my_genie": GenieRoomModel(
                        name="my-genie-room", space_id="test-space-123"
                    )
                },
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

            # Create ResourcesModel with multiple genie rooms
            resources = ResourcesModel(
                genie_rooms={
                    "genie_room_1": GenieRoomModel(
                        name="genie-room-1", space_id="test-space-123"
                    ),
                    "genie_room_2": GenieRoomModel(
                        name="genie-room-2", space_id="test-space-456"
                    ),
                }
            )

            # Both rooms share the same warehouse, so only one should be added
            assert len(resources.warehouses) == 1
            assert "genie_room_1_test_warehouse" in resources.warehouses

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

            # Create ResourcesModel with authenticated genie room
            resources = ResourcesModel(
                genie_rooms={
                    "my_genie": GenieRoomModel(
                        name="my-genie-room",
                        space_id="test-space-123",
                        on_behalf_of_user=True,
                        service_principal=service_principal,
                        workspace_host="https://test.databricks.com",
                    )
                }
            )

            # Verify warehouses inherit authentication
            for warehouse_key, warehouse in resources.warehouses.items():
                assert warehouse.on_behalf_of_user
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

            resources = ResourcesModel(
                genie_rooms={
                    "no_warehouse_genie": GenieRoomModel(
                        name="no-warehouse-room", space_id="test-space-no-wh"
                    )
                }
            )

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

            # Should not raise an exception
            resources = ResourcesModel(
                genie_rooms={
                    "my_genie": GenieRoomModel(
                        name="my-genie-room", space_id="test-space-123"
                    )
                }
            )

            # No warehouses should be added due to API error
            assert len(resources.warehouses) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
