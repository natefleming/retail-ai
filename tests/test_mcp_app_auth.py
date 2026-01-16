"""
Tests for MCP Databricks App authentication.

These tests verify that when credentials are set on McpFunctionModel,
they are correctly used for both URL resolution and authentication,
even when the nested DatabricksAppModel doesn't have credentials.
"""

from unittest.mock import MagicMock, patch

import pytest

from dao_ai.config import DatabricksAppModel, McpFunctionModel
from dao_ai.tools.mcp import _get_auth_resource, _has_auth_configured


class TestHasAuthConfigured:
    """Tests for _has_auth_configured helper."""

    def test_no_auth_configured(self):
        """Resource without auth returns False."""
        app = DatabricksAppModel(name="test-app")
        assert not _has_auth_configured(app)

    def test_obo_configured(self):
        """Resource with OBO returns True."""
        app = DatabricksAppModel(name="test-app", on_behalf_of_user=True)
        assert _has_auth_configured(app)

    def test_client_id_configured(self):
        """Resource with client_id returns True."""
        app = DatabricksAppModel(name="test-app", client_id="test-client-id")
        assert _has_auth_configured(app)


class TestGetAuthResource:
    """Tests for _get_auth_resource function."""

    def test_app_without_auth_falls_back_to_function(self):
        """When app has no auth, McpFunctionModel is used."""
        function = McpFunctionModel(
            app=DatabricksAppModel(name="mcp-gremlin-server"),
            client_id="my-client-id",
            client_secret="my-client-secret",
            workspace_host="https://my-workspace.databricks.com",
        )
        auth_resource = _get_auth_resource(function)
        # Should return the function itself, not the app
        assert auth_resource is function
        assert auth_resource.client_id == "my-client-id"

    def test_app_with_auth_uses_app(self):
        """When app has auth configured, app is used."""
        function = McpFunctionModel(
            app=DatabricksAppModel(
                name="mcp-gremlin-server",
                on_behalf_of_user=True,
            ),
            client_id="function-client-id",
        )
        auth_resource = _get_auth_resource(function)
        # Should return the app since it has OBO configured
        assert auth_resource is function.app

    def test_no_app_uses_function(self):
        """When no app is set, function is used."""
        function = McpFunctionModel(
            url="https://example.com/mcp",
            client_id="my-client-id",
        )
        auth_resource = _get_auth_resource(function)
        assert auth_resource is function


class TestMcpUrlResolution:
    """Tests for mcp_url property with app configuration."""

    @patch("dao_ai.config.WorkspaceClient")
    def test_mcp_url_uses_function_credentials(self, mock_ws_class):
        """mcp_url should use McpFunctionModel's credentials to resolve app URL."""
        # Setup mock
        mock_ws = MagicMock()
        mock_ws_class.return_value = mock_ws
        mock_app = MagicMock()
        mock_app.url = (
            "https://mcp-gremlin-server-984752964297111.11.azure.databricksapps.com"
        )
        mock_ws.apps.get.return_value = mock_app

        # Create function with credentials on McpFunctionModel, not on DatabricksAppModel
        function = McpFunctionModel(
            app=DatabricksAppModel(name="mcp-gremlin-server"),
            client_id="my-client-id",
            client_secret="my-client-secret",
            workspace_host="https://my-workspace.databricks.com",
        )

        # Get the URL - this should use McpFunctionModel's workspace_client
        url = function.mcp_url

        # Verify the URL includes /mcp suffix
        assert (
            url
            == "https://mcp-gremlin-server-984752964297111.11.azure.databricksapps.com/mcp"
        )

        # Verify the workspace client was created with the function's credentials
        mock_ws_class.assert_called()
        # The app.get should have been called with the app name
        mock_ws.apps.get.assert_called_with("mcp-gremlin-server")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
