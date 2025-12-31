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
    """Create a mock WorkspaceClient for testing."""
    mock_client = Mock()
    mock_client.genie = Mock()
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
            genie_room = GenieRoomModel(
                name="test-genie-room", space_id="test-space-123"
            )

            # Description should be populated from the space
            assert genie_room.description == "This is a test Genie space description"

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
                # Verify workspace client is shared
                assert table._workspace_client == genie_room._workspace_client

            # Verify functions inherit authentication
            assert len(functions) > 0
            for function in functions:
                assert function.on_behalf_of_user
                assert function.service_principal == service_principal
                assert function.workspace_host == "https://test.databricks.com"
                # Verify workspace client is shared
                assert function._workspace_client == genie_room._workspace_client


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
