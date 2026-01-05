"""
Unit tests for WarehouseModel auto-population of name from warehouse API.
"""

from unittest.mock import Mock, patch

import pytest

from dao_ai.config import WarehouseModel


@pytest.fixture
def mock_workspace_client():
    """Create a mock WorkspaceClient for testing."""
    mock_client = Mock()
    mock_client.warehouses = Mock()
    return mock_client


@pytest.fixture
def mock_warehouse_response():
    """Create a mock GetWarehouseResponse."""
    mock_response = Mock()
    mock_response.name = "Production Warehouse"
    mock_response.id = "abc123"
    mock_response.cluster_size = "Medium"
    mock_response.state = Mock(value="RUNNING")
    return mock_response


@pytest.mark.unit
class TestWarehouseModelNamePopulation:
    """Test suite for WarehouseModel name auto-population."""

    def test_name_populated_from_warehouse(
        self, mock_workspace_client, mock_warehouse_response
    ):
        """Test that name is automatically populated from warehouse API if not provided."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            # Create without name
            warehouse = WarehouseModel(warehouse_id="abc123")

            # Name should be populated from the warehouse API
            assert warehouse.name == "Production Warehouse"
            mock_workspace_client.warehouses.get.assert_called_once_with(id="abc123")

    def test_name_not_overridden_if_provided(
        self, mock_workspace_client, mock_warehouse_response
    ):
        """Test that provided name is not overridden by warehouse API."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            # Create with explicit name
            warehouse = WarehouseModel(name="Custom Name", warehouse_id="abc123")

            # Custom name should be preserved
            assert warehouse.name == "Custom Name"
            # API should not have been called since name was provided
            mock_workspace_client.warehouses.get.assert_not_called()

    def test_name_handles_none_from_warehouse(self, mock_workspace_client):
        """Test that None name from warehouse is handled gracefully."""
        mock_response = Mock()
        mock_response.name = None

        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.warehouses.get.return_value = mock_response

            # Create without name
            warehouse = WarehouseModel(warehouse_id="abc123")

            # Name should remain None
            assert warehouse.name is None

    def test_populate_name_handles_api_error(self, mock_workspace_client):
        """Test that populate_name validator handles API errors gracefully."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            # Mock API to raise an error
            mock_workspace_client.warehouses.get.side_effect = Exception(
                "API Error: Connection timeout"
            )

            # Create without name - should not raise exception
            warehouse = WarehouseModel(warehouse_id="abc123")

            # Name should remain None (not populated due to API error)
            assert warehouse.name is None

    def test_warehouse_id_is_resolved(
        self, mock_workspace_client, mock_warehouse_response
    ):
        """Test that warehouse_id is resolved via value_of."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            warehouse = WarehouseModel(warehouse_id="xyz789")

            # warehouse_id should be the resolved value
            assert warehouse.warehouse_id == "xyz789"

    def test_warehouse_details_caching(
        self, mock_workspace_client, mock_warehouse_response
    ):
        """Test that _get_warehouse_details caches the warehouse details."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            warehouse = WarehouseModel(warehouse_id="abc123")

            # First call already happened in the validator
            assert mock_workspace_client.warehouses.get.call_count == 1

            # Call _get_warehouse_details again
            details = warehouse._get_warehouse_details()
            assert details == mock_warehouse_response

            # Should still be only 1 call (cached)
            assert mock_workspace_client.warehouses.get.call_count == 1

    def test_warehouse_model_api_scopes(self, mock_workspace_client):
        """Test that WarehouseModel returns correct API scopes."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            warehouse = WarehouseModel(name="Test", warehouse_id="abc123")

            api_scopes = warehouse.api_scopes
            assert "sql.warehouses" in api_scopes
            assert "sql.statement-execution" in api_scopes

    def test_warehouse_model_as_resources(
        self, mock_workspace_client, mock_warehouse_response
    ):
        """Test that WarehouseModel returns correct resources."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            warehouse = WarehouseModel(warehouse_id="abc123")

            resources = warehouse.as_resources()
            assert len(resources) == 1

            from mlflow.models.resources import DatabricksSQLWarehouse

            assert isinstance(resources[0], DatabricksSQLWarehouse)

    def test_warehouse_model_with_description(
        self, mock_workspace_client, mock_warehouse_response
    ):
        """Test that description can be provided (not auto-populated from API)."""
        with patch("dao_ai.config.WorkspaceClient", return_value=mock_workspace_client):
            mock_workspace_client.warehouses.get.return_value = mock_warehouse_response

            # Create with description
            warehouse = WarehouseModel(
                warehouse_id="abc123", description="My custom description"
            )

            # Description should be preserved, name auto-populated
            assert warehouse.name == "Production Warehouse"
            assert warehouse.description == "My custom description"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
