"""
Integration tests for GenieRoomModel parsing serialized Genie spaces.

This test suite verifies that GenieRoomModel correctly parses serialized_space
JSON strings from Databricks Genie and extracts TableModel and FunctionModel instances.
"""

import json
from unittest.mock import Mock, patch

import pytest

from dao_ai.config import FunctionModel, GenieRoomModel, TableModel


@pytest.fixture
def mock_workspace_client():
    """Create a mock WorkspaceClient for testing.

    By default, all tables and functions are configured to exist.
    Use the tables.get.side_effect or functions.get.side_effect
    to customize which resources exist.
    """

    mock_client = Mock()
    mock_client.genie = Mock()

    # By default, all tables and functions exist (return a mock for any name)
    mock_client.tables = Mock()
    mock_client.tables.get = Mock(return_value=Mock())

    mock_client.functions = Mock()
    mock_client.functions.get = Mock(return_value=Mock())

    return mock_client


@pytest.fixture
def mock_genie_space_with_serialized_data():
    """Create a mock GenieSpace with serialized_space data matching Databricks structure."""
    mock_space = Mock()
    mock_space.space_id = "test-space-123"
    mock_space.title = "Test Genie Space"
    mock_space.description = "A test Genie space"
    mock_space.warehouse_id = "test-warehouse"

    # Real Databricks structure: data_sources.tables with 'identifier' field
    serialized_data = {
        "version": "1.0",
        "data_sources": {
            "tables": [
                {"identifier": "catalog.schema.table1", "column_configs": []},
                {"identifier": "catalog.schema.table2", "column_configs": []},
                {"identifier": "catalog.schema.table3", "column_configs": []},
            ],
            "functions": [
                {"identifier": "catalog.schema.function1"},
                {"identifier": "catalog.schema.function2"},
            ],
        },
    }
    mock_space.serialized_space = json.dumps(serialized_data)
    return mock_space


@pytest.fixture
def mock_genie_space_with_name_fallback():
    """Create a mock GenieSpace testing backward compatibility with 'name' field."""
    mock_space = Mock()
    mock_space.space_id = "test-space-456"
    mock_space.title = "Test Genie Space (Name Fallback)"
    mock_space.description = "A test Genie space with name field fallback"
    mock_space.warehouse_id = "test-warehouse"

    # Test fallback to 'name' field when 'identifier' is not present
    serialized_data = {
        "data_sources": {
            "tables": [
                {"name": "catalog.schema.customers", "type": "table"},
                {"name": "catalog.schema.orders", "type": "table"},
            ],
            "functions": [{"name": "catalog.schema.get_customer", "type": "function"}],
        }
    }
    mock_space.serialized_space = json.dumps(serialized_data)
    return mock_space


@pytest.fixture
def mock_genie_space_with_string_arrays():
    """Create a mock GenieSpace with string arrays in data_sources."""
    mock_space = Mock()
    mock_space.space_id = "test-space-789"
    mock_space.title = "Test Genie Space (String Arrays)"
    mock_space.description = "A test Genie space with string arrays"
    mock_space.warehouse_id = "test-warehouse"

    # String arrays in data_sources
    serialized_data = {
        "data_sources": {
            "tables": ["catalog.schema.products", "catalog.schema.inventory"],
            "functions": [
                "catalog.schema.find_product",
                "catalog.schema.check_inventory",
            ],
        }
    }
    mock_space.serialized_space = json.dumps(serialized_data)
    return mock_space


@pytest.fixture
def mock_genie_space_empty():
    """Create a mock GenieSpace with no tables or functions."""
    mock_space = Mock()
    mock_space.space_id = "test-space-empty"
    mock_space.title = "Empty Genie Space"
    mock_space.description = "A Genie space with no tables or functions"
    mock_space.warehouse_id = "test-warehouse"
    mock_space.serialized_space = json.dumps({})
    return mock_space


@pytest.fixture
def mock_genie_space_no_serialized():
    """Create a mock GenieSpace without serialized_space."""
    mock_space = Mock()
    mock_space.space_id = "test-space-no-data"
    mock_space.title = "No Data Genie Space"
    mock_space.description = "A Genie space without serialized data"
    mock_space.warehouse_id = "test-warehouse"
    mock_space.serialized_space = None
    return mock_space


@pytest.fixture
def mock_genie_space_with_sql_functions():
    """Create a mock GenieSpace with SQL functions in instructions.sql_functions."""
    mock_space = Mock()
    mock_space.space_id = "test-space-sql-funcs"
    mock_space.title = "Test Genie Space (SQL Functions)"
    mock_space.description = "A test Genie space with SQL functions"
    mock_space.warehouse_id = "test-warehouse"

    # Real structure with instructions.sql_functions
    serialized_data = {
        "version": 1,
        "data_sources": {
            "tables": [
                {"identifier": "catalog.schema.orders", "column_configs": []},
                {"identifier": "catalog.schema.products", "column_configs": []},
            ]
        },
        "instructions": {
            "sql_functions": [
                {
                    "id": "01f05c14e85d11379c62e45a73f72adb",
                    "identifier": "catalog.schema.lookup_items_by_descriptions",
                },
                {
                    "id": "01f05c14fa561b879fbcbde149f3f015",
                    "identifier": "catalog.schema.match_historical_item_order_by_date",
                },
                {
                    "id": "01f05c1505c01037b3726f5a33bc0d5e",
                    "identifier": "catalog.schema.match_item_by_description_and_price",
                },
            ]
        },
    }
    mock_space.serialized_space = json.dumps(serialized_data)
    return mock_space


@pytest.mark.unit
class TestGenieRoomModelSerialization:
    """Test suite for GenieRoomModel serialized_space parsing."""

    def test_parse_serialized_space_with_identifier_field(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test parsing serialized_space with standard Databricks structure (identifier field)."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            # Test tables extraction
            tables = genie_room.tables
            assert len(tables) == 3
            assert all(isinstance(table, TableModel) for table in tables)
            assert tables[0].name == "catalog.schema.table1"
            assert tables[1].name == "catalog.schema.table2"
            assert tables[2].name == "catalog.schema.table3"

            # Test functions extraction
            functions = genie_room.functions
            assert len(functions) == 2
            assert all(isinstance(func, FunctionModel) for func in functions)
            assert functions[0].name == "catalog.schema.function1"
            assert functions[1].name == "catalog.schema.function2"

    def test_parse_serialized_space_with_name_fallback(
        self, mock_workspace_client, mock_genie_space_with_name_fallback
    ):
        """Test parsing serialized_space with fallback to 'name' field."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_name_fallback
            )

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-456"
            )

            # Test tables extraction with name fallback
            tables = genie_room.tables
            assert len(tables) == 2
            assert tables[0].name == "catalog.schema.customers"
            assert tables[1].name == "catalog.schema.orders"

            # Test functions extraction with name fallback
            functions = genie_room.functions
            assert len(functions) == 1
            assert functions[0].name == "catalog.schema.get_customer"

    def test_parse_serialized_space_with_string_arrays(
        self, mock_workspace_client, mock_genie_space_with_string_arrays
    ):
        """Test parsing serialized_space with simple string arrays."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_string_arrays
            )

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-789"
            )

            # Test tables extraction
            tables = genie_room.tables
            assert len(tables) == 2
            assert tables[0].name == "catalog.schema.products"
            assert tables[1].name == "catalog.schema.inventory"

            # Test functions extraction
            functions = genie_room.functions
            assert len(functions) == 2
            assert functions[0].name == "catalog.schema.find_product"
            assert functions[1].name == "catalog.schema.check_inventory"

    def test_parse_serialized_space_empty(
        self, mock_workspace_client, mock_genie_space_empty
    ):
        """Test parsing empty serialized_space."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = mock_genie_space_empty

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-empty"
            )

            # Should return empty lists
            tables = genie_room.tables
            assert len(tables) == 0
            assert tables == []

            functions = genie_room.functions
            assert len(functions) == 0
            assert functions == []

    def test_parse_serialized_space_none(
        self, mock_workspace_client, mock_genie_space_no_serialized
    ):
        """Test parsing when serialized_space is None."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_no_serialized
            )

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-no-data"
            )

            # Should return empty lists without errors
            tables = genie_room.tables
            assert len(tables) == 0

            functions = genie_room.functions
            assert len(functions) == 0

    def test_parse_serialized_space_invalid_json(self, mock_workspace_client):
        """Test handling of invalid JSON in serialized_space."""
        mock_space = Mock()
        mock_space.space_id = "test-space-invalid"
        mock_space.title = "Invalid JSON Space"
        mock_space.description = "Space with invalid JSON"
        mock_space.warehouse_id = "test-warehouse"
        mock_space.serialized_space = "not valid json {{{["

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = mock_space

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-invalid"
            )

            # Should handle gracefully and return empty lists
            tables = genie_room.tables
            assert len(tables) == 0

            functions = genie_room.functions
            assert len(functions) == 0

    def test_parse_serialized_space_with_sql_functions(
        self, mock_workspace_client, mock_genie_space_with_sql_functions
    ):
        """Test parsing serialized_space with SQL functions in instructions.sql_functions."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_sql_functions
            )

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-sql-funcs"
            )

            # Test tables extraction
            tables = genie_room.tables
            assert len(tables) == 2
            assert tables[0].name == "catalog.schema.orders"
            assert tables[1].name == "catalog.schema.products"

            # Test SQL functions extraction from instructions.sql_functions
            functions = genie_room.functions
            assert len(functions) == 3
            assert all(isinstance(func, FunctionModel) for func in functions)
            assert functions[0].name == "catalog.schema.lookup_items_by_descriptions"
            assert (
                functions[1].name
                == "catalog.schema.match_historical_item_order_by_date"
            )
            assert (
                functions[2].name
                == "catalog.schema.match_item_by_description_and_price"
            )

    def test_genie_space_details_caching(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that _get_space_details caches the space details."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            # First call should fetch from API
            tables1 = genie_room.tables
            assert mock_workspace_client.genie.get_space.call_count == 1

            # Second call should use cached data
            tables2 = genie_room.tables
            assert mock_workspace_client.genie.get_space.call_count == 1

            # Results should be the same
            assert len(tables1) == len(tables2)
            assert tables1[0].name == tables2[0].name

    def test_genie_room_model_api_scopes(self, mock_workspace_client):
        """Test that GenieRoomModel returns correct API scopes."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            api_scopes = genie_room.api_scopes
            assert "dashboards.genie" in api_scopes

    def test_genie_room_model_as_resources(self, mock_workspace_client):
        """Test that GenieRoomModel returns correct resources."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            resources = genie_room.as_resources()
            assert len(resources) == 1

            from mlflow.models.resources import DatabricksGenieSpace

            assert isinstance(resources[0], DatabricksGenieSpace)
            # The genie_space_id is stored as the 'name' attribute
            assert resources[0].name == "test-space-123"

    def test_name_populated_from_space(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that name is automatically populated from GenieSpace.title if not provided."""
        # Set a title on the mock space
        mock_genie_space_with_serialized_data.title = "My Retail Analytics Space"

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Create without name
            genie_room = GenieRoomModel(space_id="test-space-123")

            # Name should be populated from the space title
            assert genie_room.name == "My Retail Analytics Space"

    def test_name_not_overridden_if_provided(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that provided name is not overridden by GenieSpace title."""
        # Set a title on the mock space
        mock_genie_space_with_serialized_data.title = "Space Title from API"

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Create with explicit name
            genie_room = GenieRoomModel(
                name="My Custom Name", space_id="test-space-123"
            )

            # Custom name should be preserved
            assert genie_room.name == "My Custom Name"

    def test_name_handles_none_from_space(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that None title from space is handled gracefully."""
        # Set title to None
        mock_genie_space_with_serialized_data.title = None

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Create without name
            genie_room = GenieRoomModel(space_id="test-space-123")

            # Name should remain None
            assert genie_room.name is None

    def test_description_populated_from_space(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that description is automatically populated from GenieSpace if not provided."""
        # Set a description on the mock space
        mock_genie_space_with_serialized_data.description = (
            "This is a test Genie space description"
        )

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Create without description
            genie_room = GenieRoomModel(space_id="test-space-123")

            # Description should be populated from the space
            assert genie_room.description == "This is a test Genie space description"

    def test_name_and_description_populated_together(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that both name and description are populated from GenieSpace when not provided."""
        # Set both title and description on the mock space
        mock_genie_space_with_serialized_data.title = "Production Analytics"
        mock_genie_space_with_serialized_data.description = (
            "Production space for analytics queries"
        )

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Create without name or description
            genie_room = GenieRoomModel(space_id="test-space-123")

            # Both should be populated from the space
            assert genie_room.name == "Production Analytics"
            assert genie_room.description == "Production space for analytics queries"

    def test_description_not_overridden_if_provided(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that provided description is not overridden by GenieSpace description."""
        # Set a description on the mock space
        mock_genie_space_with_serialized_data.description = "Space description"

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Create with explicit description
            genie_room = GenieRoomModel(
                name="test-genie-room",
                space_id="test-space-123",
                description="My custom description",
            )

            # Custom description should be preserved
            assert genie_room.description == "My custom description"

    def test_description_handles_none_from_space(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that None description from space is handled gracefully."""
        # Set description to None
        mock_genie_space_with_serialized_data.description = None

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Create without description
            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            # Description should remain None
            assert genie_room.description is None

    def test_partial_population_name_provided_description_auto(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that description is still populated when name is provided."""
        # Set both title and description on the mock space
        mock_genie_space_with_serialized_data.title = "API Space Title"
        mock_genie_space_with_serialized_data.description = "API Space Description"

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Create with name but without description
            genie_room = GenieRoomModel(name="Custom Name", space_id="test-space-123")

            # Custom name should be preserved, description should be auto-populated
            assert genie_room.name == "Custom Name"
            assert genie_room.description == "API Space Description"

    def test_partial_population_description_provided_name_auto(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that name is still populated when description is provided."""
        # Set both title and description on the mock space
        mock_genie_space_with_serialized_data.title = "API Space Title"
        mock_genie_space_with_serialized_data.description = "API Space Description"

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Create with description but without name
            genie_room = GenieRoomModel(
                description="Custom Description", space_id="test-space-123"
            )

            # Custom description should be preserved, name should be auto-populated
            assert genie_room.name == "API Space Title"
            assert genie_room.description == "Custom Description"

    def test_populate_name_and_description_handles_api_error(
        self, mock_workspace_client
    ):
        """Test that populate_name_and_description validator handles API errors gracefully."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            # Mock API to raise an error
            mock_workspace_client.genie.get_space.side_effect = Exception(
                "API Error: Connection timeout"
            )

            # Create without name or description - should not raise exception
            genie_room = GenieRoomModel(space_id="test-space-123")

            # Name and description should remain None (not populated due to API error)
            assert genie_room.name is None
            assert genie_room.description is None

    def test_populate_name_and_description_not_called_when_both_provided(
        self, mock_workspace_client
    ):
        """Test that API is not called when both name and description are provided."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            # Create with both name and description
            genie_room = GenieRoomModel(
                name="Custom Name",
                description="Custom Description",
                space_id="test-space-123",
            )

            # API should not have been called
            mock_workspace_client.genie.get_space.assert_not_called()

            # Values should be preserved
            assert genie_room.name == "Custom Name"
            assert genie_room.description == "Custom Description"

    def test_tables_and_functions_inherit_authentication(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that TableModel and FunctionModel inherit authentication from GenieRoomModel."""
        from dao_ai.config import ServicePrincipalModel

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Create GenieRoomModel with specific authentication
            service_principal = ServicePrincipalModel(
                client_id="test-client-id", client_secret="test-client-secret"
            )

            genie_room = GenieRoomModel(
                name="test-genie-room",
                space_id="test-space-123",
                on_behalf_of_user=True,
                service_principal=service_principal,
                workspace_host="https://test.databricks.com",
            )

            # Get tables and functions
            tables = genie_room.tables
            functions = genie_room.functions

            # Verify tables inherit authentication
            assert len(tables) > 0
            for table in tables:
                assert table.on_behalf_of_user
                assert table.service_principal == service_principal
                assert table.workspace_host == "https://test.databricks.com"

            # Verify functions inherit authentication
            assert len(functions) > 0
            for function in functions:
                assert function.on_behalf_of_user
                assert function.service_principal == service_principal
                assert function.workspace_host == "https://test.databricks.com"

    def test_warehouse_extraction(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that warehouse is correctly extracted from GenieSpace."""
        from dao_ai.config import WarehouseModel

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Mock warehouse response
            mock_warehouse_response = Mock()
            mock_warehouse_response.name = "Test Warehouse"
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            # Get warehouse
            warehouse = genie_room.warehouse

            # Verify warehouse was extracted
            assert warehouse is not None
            assert isinstance(warehouse, WarehouseModel)
            assert warehouse.name == "Test Warehouse"
            assert warehouse.warehouse_id == "test-warehouse"

            # Verify warehouse API was called with correct ID
            mock_workspace_client.warehouses.get.assert_called_once_with(
                "test-warehouse"
            )

    def test_warehouse_inherits_authentication(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that WarehouseModel inherits authentication from GenieRoomModel."""
        from dao_ai.config import ServicePrincipalModel

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Mock warehouse response
            mock_warehouse_response = Mock()
            mock_warehouse_response.name = "Test Warehouse"
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            # Create GenieRoomModel with specific authentication
            service_principal = ServicePrincipalModel(
                client_id="test-client-id", client_secret="test-client-secret"
            )

            genie_room = GenieRoomModel(
                name="test-genie-room",
                space_id="test-space-123",
                on_behalf_of_user=True,
                service_principal=service_principal,
                workspace_host="https://test.databricks.com",
            )

            # Get warehouse
            warehouse = genie_room.warehouse

            # Verify warehouse inherits authentication
            assert warehouse is not None
            assert warehouse.on_behalf_of_user
            assert warehouse.service_principal == service_principal
            assert warehouse.workspace_host == "https://test.databricks.com"

    def test_warehouse_handles_missing_warehouse_id(self, mock_workspace_client):
        """Test that warehouse property handles missing warehouse_id gracefully."""
        mock_space = Mock()
        mock_space.space_id = "test-space-123"
        mock_space.title = "Test Space"
        mock_space.description = "Test"
        mock_space.warehouse_id = None
        mock_space.serialized_space = json.dumps({"data_sources": {}})

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = mock_space

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            # Get warehouse - should return None
            warehouse = genie_room.warehouse
            assert warehouse is None

            # Verify warehouses.get was not called
            mock_workspace_client.warehouses.get.assert_not_called()

    def test_warehouse_handles_api_error(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that warehouse property handles API errors gracefully."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Mock warehouse API to raise an error
            mock_workspace_client.warehouses.get.side_effect = Exception(
                "API Error: Warehouse not found"
            )

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            # Get warehouse - should return None on error
            warehouse = genie_room.warehouse
            assert warehouse is None

    def test_warehouse_uses_warehouse_id_as_fallback_name(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that warehouse uses warehouse_id as name if response.name is None."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Mock warehouse response with no name
            mock_warehouse_response = Mock()
            mock_warehouse_response.name = None
            mock_warehouse_response.description = "A test warehouse"
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            # Get warehouse
            warehouse = genie_room.warehouse

            # Verify warehouse uses warehouse_id as name
            assert warehouse is not None
            assert warehouse.name == "test-warehouse"
            assert warehouse.warehouse_id == "test-warehouse"

    def test_tables_filters_nonexistent_tables(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that tables property filters out tables that don't exist in Unity Catalog."""
        from databricks.sdk.errors.platform import NotFound

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Configure tables.get to raise NotFound for one table
            def table_exists_side_effect(full_name: str):
                if full_name == "catalog.schema.table2":
                    raise NotFound("Table not found")
                return Mock()

            mock_workspace_client.tables.get.side_effect = table_exists_side_effect

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            # Test tables extraction - should exclude the non-existent table
            tables = genie_room.tables
            assert len(tables) == 2  # Only 2 of 3 tables should be returned
            table_names = [t.name for t in tables]
            assert "catalog.schema.table1" in table_names
            assert "catalog.schema.table2" not in table_names  # This one doesn't exist
            assert "catalog.schema.table3" in table_names

    def test_functions_filters_nonexistent_functions(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that functions property filters out functions that don't exist in Unity Catalog."""
        from databricks.sdk.errors.platform import NotFound

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # Configure functions.get to raise NotFound for one function
            def function_exists_side_effect(name: str):
                if name == "catalog.schema.function1":
                    raise NotFound("Function not found")
                return Mock()

            mock_workspace_client.functions.get.side_effect = (
                function_exists_side_effect
            )

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            # Test functions extraction - should exclude the non-existent function
            functions = genie_room.functions
            assert len(functions) == 1  # Only 1 of 2 functions should be returned
            function_names = [f.name for f in functions]
            assert (
                "catalog.schema.function1" not in function_names
            )  # This one doesn't exist
            assert "catalog.schema.function2" in function_names

    def test_tables_all_nonexistent_returns_empty(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that tables property returns empty list if no tables exist."""
        from databricks.sdk.errors.platform import NotFound

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # All tables don't exist
            mock_workspace_client.tables.get.side_effect = NotFound("Table not found")

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            tables = genie_room.tables
            assert len(tables) == 0

    def test_functions_all_nonexistent_returns_empty(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that functions property returns empty list if no functions exist."""
        from databricks.sdk.errors.platform import NotFound

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # All functions don't exist
            mock_workspace_client.functions.get.side_effect = NotFound(
                "Function not found"
            )

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            functions = genie_room.functions
            assert len(functions) == 0

    def test_tables_handles_api_errors_gracefully(
        self, mock_workspace_client, mock_genie_space_with_serialized_data
    ):
        """Test that tables property handles generic API errors gracefully."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.genie.get_space.return_value = (
                mock_genie_space_with_serialized_data
            )

            # One table has an API error (not NotFound)
            def table_exists_side_effect(full_name: str):
                if full_name == "catalog.schema.table2":
                    raise Exception("API Error: Connection timeout")
                return Mock()

            mock_workspace_client.tables.get.side_effect = table_exists_side_effect

            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            # Should exclude the table with API error
            tables = genie_room.tables
            assert len(tables) == 2
            table_names = [t.name for t in tables]
            assert "catalog.schema.table2" not in table_names


@pytest.mark.skipif(
    not pytest.importorskip("conftest").has_databricks_env(),
    reason="Databricks environment variables not set",
)
@pytest.mark.integration
class TestGenieRoomModelRealAPI:
    """Integration tests using a real Genie space from Databricks."""

    def test_real_genie_space_parsing(self):
        """Test parsing a real Genie space with ID: 01f01c91f1f414d59daaefd2b7ec82ea"""
        from dao_ai.config import GenieRoomModel

        # Create a GenieRoomModel with the real space ID
        genie_room = GenieRoomModel(
            name="test-real-genie-room", space_id="01f01c91f1f414d59daaefd2b7ec82ea"
        )

        # Test that we can fetch space details
        space_details = genie_room._get_space_details()
        assert space_details is not None
        assert space_details.space_id == "01f01c91f1f414d59daaefd2b7ec82ea"

        # Test that serialized_space is present
        assert space_details.serialized_space is not None
        print(f"\nSpace Title: {space_details.title}")
        print(f"Space Description: {space_details.description}")

        # Test parsing the serialized space
        parsed_space = genie_room._parse_serialized_space()
        assert isinstance(parsed_space, dict)
        print(f"\nParsed space keys: {list(parsed_space.keys())}")

        # Test extracting tables
        tables = genie_room.tables
        print(f"\nExtracted {len(tables)} tables:")
        for i, table in enumerate(tables[:5], 1):  # Print first 5
            print(f"  {i}. {table.name}")
        if len(tables) > 5:
            print(f"  ... and {len(tables) - 5} more")

        # Test extracting functions
        functions = genie_room.functions
        print(f"\nExtracted {len(functions)} functions:")
        for i, func in enumerate(functions[:5], 1):  # Print first 5
            print(f"  {i}. {func.name}")
        if len(functions) > 5:
            print(f"  ... and {len(functions) - 5} more")

        # Verify that we got some data
        assert len(tables) >= 0, "Should return a list of tables (may be empty)"
        assert len(functions) >= 0, "Should return a list of functions (may be empty)"

        # Verify all extracted tables are TableModel instances
        from dao_ai.config import TableModel

        for table in tables:
            assert isinstance(table, TableModel)
            assert table.name is not None
            assert isinstance(table.name, str)

        # Verify all extracted functions are FunctionModel instances
        from dao_ai.config import FunctionModel

        for func in functions:
            assert isinstance(func, FunctionModel)
            assert func.name is not None
            assert isinstance(func.name, str)

    def test_real_genie_space_caching(self):
        """Test that real API calls are cached properly."""
        from dao_ai.config import GenieRoomModel

        genie_room = GenieRoomModel(
            name="test-real-genie-room", space_id="01f01c91f1f414d59daaefd2b7ec82ea"
        )

        # First access - fetches from API and caches
        tables1 = genie_room.tables
        assert len(tables1) > 0  # Should have tables

        # Verify _space_details is cached
        assert genie_room._space_details is not None

        # Second access - should use cached data
        tables2 = genie_room.tables
        assert len(tables1) == len(tables2)
        assert tables1[0].name == tables2[0].name

        # Third access with functions - should still use cached space details
        functions = genie_room.functions
        assert len(functions) >= 0  # May or may not have functions

        # Verify the cached space details are still the same object
        assert genie_room._space_details is not None

    def test_real_genie_space_resources(self):
        """Test that resources are correctly generated for a real Genie space."""
        from mlflow.models.resources import DatabricksGenieSpace

        from dao_ai.config import GenieRoomModel

        genie_room = GenieRoomModel(
            name="test-real-genie-room", space_id="01f01c91f1f414d59daaefd2b7ec82ea"
        )

        # Test as_resources
        resources = genie_room.as_resources()
        assert len(resources) == 1
        assert isinstance(resources[0], DatabricksGenieSpace)
        assert resources[0].name == "01f01c91f1f414d59daaefd2b7ec82ea"

        # Test api_scopes
        api_scopes = genie_room.api_scopes
        assert "dashboards.genie" in api_scopes

    def test_real_genie_space_warehouse(self):
        """Test that warehouse is correctly extracted from a real Genie space."""
        from dao_ai.config import GenieRoomModel, WarehouseModel

        genie_room = GenieRoomModel(
            name="test-real-genie-room", space_id="01f01c91f1f414d59daaefd2b7ec82ea"
        )

        # Test warehouse extraction
        warehouse = genie_room.warehouse

        if warehouse is not None:
            # If warehouse exists, verify it's a WarehouseModel
            assert isinstance(warehouse, WarehouseModel)
            assert warehouse.warehouse_id is not None
            assert warehouse.name is not None
            print(f"\nWarehouse ID: {warehouse.warehouse_id}")
            print(f"Warehouse Name: {warehouse.name}")

            # Verify warehouse has valid API scopes
            assert "sql.warehouses" in warehouse.api_scopes
            assert "sql.statement-execution" in warehouse.api_scopes
        else:
            # If no warehouse, that's okay - just log it
            print("\nNo warehouse associated with this Genie space")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
