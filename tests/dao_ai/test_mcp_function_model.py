"""Unit tests for McpFunctionModel convenience objects and validation."""

import pytest
from pydantic import ValidationError

from dao_ai.config import (
    ConnectionModel,
    FunctionType,
    GenieRoomModel,
    IndexModel,
    McpFunctionModel,
    SchemaModel,
    TableModel,
    TransportType,
    VectorStoreModel,
)


class TestMcpFunctionModelValidation:
    """Test mutual exclusivity validation for MCP function URL sources."""

    def test_no_url_source_raises_error(self):
        """Test that missing URL source raises validation error."""
        with pytest.raises(
            ValidationError, match="exactly one of the following must be provided"
        ):
            McpFunctionModel(
                name="test_mcp",
                transport=TransportType.STREAMABLE_HTTP,
            )

    def test_multiple_url_sources_raises_error(self):
        """Test that multiple URL sources raise validation error."""
        with pytest.raises(
            ValidationError, match="only one URL source can be provided"
        ):
            McpFunctionModel(
                name="test_mcp",
                transport=TransportType.STREAMABLE_HTTP,
                url="https://example.com/mcp",
                genie_room=GenieRoomModel(
                    name="test_genie",
                    space_id="test_space_id",
                ),
                workspace_host="https://workspace.com",
            )

    def test_url_and_connection_mutually_exclusive(self):
        """Test that url and connection cannot be provided together."""
        with pytest.raises(
            ValidationError, match="only one URL source can be provided"
        ):
            McpFunctionModel(
                name="test_mcp",
                transport=TransportType.STREAMABLE_HTTP,
                url="https://example.com/mcp",
                connection=ConnectionModel(name="test_connection"),
            )

    def test_genie_room_and_sql_mutually_exclusive(self):
        """Test that genie_room and sql cannot be provided together."""
        with pytest.raises(
            ValidationError, match="only one URL source can be provided"
        ):
            McpFunctionModel(
                name="test_mcp",
                transport=TransportType.STREAMABLE_HTTP,
                genie_room=GenieRoomModel(name="test_genie", space_id="space_123"),
                sql=True,
                workspace_host="https://workspace.com",
            )

    def test_vector_search_and_functions_mutually_exclusive(self):
        """Test that vector_search and functions cannot be provided together."""
        schema = SchemaModel(catalog_name="catalog", schema_name="schema")
        table = TableModel(schema=schema, name="test_table")
        index = IndexModel(schema=schema, name="test_index")

        with pytest.raises(
            ValidationError, match="only one URL source can be provided"
        ):
            McpFunctionModel(
                name="test_mcp",
                transport=TransportType.STREAMABLE_HTTP,
                vector_search=VectorStoreModel(
                    source_table=table,
                    embedding_source_column="text",
                    index=index,
                    primary_key="id",  # Provide primary key to avoid DB lookup
                ),
                functions=schema,
                workspace_host="https://workspace.com",
            )

    def test_stdio_requires_command_and_args(self):
        """Test that STDIO transport requires command and args."""
        with pytest.raises(ValidationError, match="command must be provided"):
            McpFunctionModel(
                name="test_mcp",
                transport=TransportType.STDIO,
                command=None,
            )

        with pytest.raises(ValidationError, match="args must be provided"):
            McpFunctionModel(
                name="test_mcp",
                transport=TransportType.STDIO,
                command="python",
                args=[],
            )


class TestMcpFunctionModelUrlGeneration:
    """Test URL generation for different convenience objects."""

    def test_direct_url_passthrough(self):
        """Test that direct URL is returned as-is."""
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com/api/mcp",
        )
        assert model.mcp_url == "https://example.com/api/mcp"

    def test_genie_room_url_generation(self):
        """Test URL generation for Genie room."""
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            genie_room=GenieRoomModel(
                name="test_genie",
                space_id="01f01c91f1f414d59daaefd2b7ec82ea",
            ),
            workspace_host="https://adb-123.azuredatabricks.net",
        )
        expected = "https://adb-123.azuredatabricks.net/api/2.0/mcp/genie/01f01c91f1f414d59daaefd2b7ec82ea"
        assert model.mcp_url == expected

    def test_sql_url_generation(self):
        """Test URL generation for DBSQL MCP server (serverless)."""
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            sql=True,
            workspace_host="https://adb-123.azuredatabricks.net",
        )
        # Per Databricks MCP documentation, the DBSQL MCP server is serverless and workspace-level
        expected = "https://adb-123.azuredatabricks.net/api/2.0/mcp/sql"
        assert model.mcp_url == expected

    def test_vector_search_url_generation(self):
        """Test URL generation for Vector Search."""
        schema = SchemaModel(catalog_name="nfleming", schema_name="retail_ai")
        table = TableModel(schema=schema, name="test_table")
        index = IndexModel(schema=schema, name="test_index")

        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            vector_search=VectorStoreModel(
                source_table=table,
                embedding_source_column="text",
                index=index,
                primary_key="id",  # Provide primary key to avoid DB lookup
            ),
            workspace_host="https://adb-123.azuredatabricks.net",
        )
        expected = "https://adb-123.azuredatabricks.net/api/2.0/mcp/vector-search/nfleming/retail_ai"
        assert model.mcp_url == expected

    def test_functions_url_generation(self):
        """Test URL generation for UC Functions MCP server."""
        schema = SchemaModel(catalog_name="nfleming", schema_name="retail_ai")
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            functions=schema,
            workspace_host="https://adb-123.azuredatabricks.net",
        )
        expected = "https://adb-123.azuredatabricks.net/api/2.0/mcp/functions/nfleming/retail_ai"
        assert model.mcp_url == expected

    def test_connection_url_generation(self):
        """Test URL generation for UC Connection."""
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            connection=ConnectionModel(name="my_connection"),
            workspace_host="https://adb-123.azuredatabricks.net",
        )
        expected = (
            "https://adb-123.azuredatabricks.net/api/2.0/mcp/external/my_connection"
        )
        assert model.mcp_url == expected

    def test_trailing_slash_removed_from_workspace_host(self):
        """Test that trailing slash is removed from workspace host."""
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            genie_room=GenieRoomModel(
                name="test_genie",
                space_id="space_123",
            ),
            workspace_host="https://adb-123.azuredatabricks.net/",  # Note trailing slash
        )
        # Should not have double slash before /api
        assert "//" not in model.mcp_url.replace("https://", "")
        assert model.mcp_url.endswith("genie/space_123")

    def test_workspace_host_can_be_omitted(self):
        """Test that workspace_host can be omitted and will be derived from workspace client."""
        # This test validates that the model can be created without workspace_host
        # The actual URL generation will use the default WorkspaceClient in runtime
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            genie_room=GenieRoomModel(
                name="test_genie",
                space_id="space_123",
            ),
            # workspace_host not provided - will be derived from workspace client at runtime
        )

        # Model should be created successfully
        assert model.name == "test_mcp"
        assert model.genie_room is not None
        # Note: Actual URL generation requires a configured WorkspaceClient

    def test_vector_search_without_schema_raises_error(self):
        """Test that vector_search without schema (no schema_model on index) raises error."""
        # Create table without schema
        table = TableModel(name="test_table")  # No schema provided
        # Create index without schema
        index = IndexModel(name="test_index")  # No schema provided

        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            vector_search=VectorStoreModel(
                source_table=table,
                embedding_source_column="text",
                index=index,
                primary_key="id",  # Provide primary key to avoid DB lookup
            ),
            workspace_host="https://adb-123.azuredatabricks.net",
        )

        with pytest.raises(
            ValueError, match="vector_search must have an index with a schema"
        ):
            _ = model.mcp_url


class TestMcpFunctionModelProperties:
    """Test basic properties and serialization."""

    def test_full_name_property(self):
        """Test full_name property returns name."""
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com/mcp",
        )
        assert model.full_name == "test_mcp"

    def test_function_type_is_mcp(self):
        """Test that type is always MCP."""
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com/mcp",
        )
        assert model.type == FunctionType.MCP

    def test_transport_serialization(self):
        """Test that transport enum is serialized to string."""
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com/mcp",
        )
        assert (
            model.serialize_transport(TransportType.STREAMABLE_HTTP)
            == "streamable_http"
        )
        assert model.serialize_transport(TransportType.STDIO) == "stdio"


class TestMcpFunctionModelAuthValidation:
    """Test authentication validation."""

    def test_oauth_and_pat_mutually_exclusive(self):
        """Test that OAuth and PAT cannot be used together."""
        with pytest.raises(
            ValidationError, match="Cannot use both OAuth and user authentication"
        ):
            McpFunctionModel(
                name="test_mcp",
                transport=TransportType.STREAMABLE_HTTP,
                url="https://example.com/mcp",
                client_id="client_id",
                client_secret="client_secret",
                pat="personal_access_token",
                workspace_host="https://workspace.com",
            )

    def test_workspace_host_optional_for_auth(self):
        """Test that workspace_host is optional (will be derived from workspace client)."""
        # Should not raise validation error - workspace_host will be derived from workspace client
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com/mcp",
            client_id="client_id",
            client_secret="client_secret",
            # workspace_host not provided - will be derived from workspace client
        )
        assert model.client_id == "client_id"
        assert model.client_secret == "client_secret"

    def test_oauth_requires_both_credentials(self):
        """Test that OAuth requires both client_id and client_secret."""
        # Only client_id provided - should not trigger OAuth validation
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com/mcp",
            client_id="client_id",
            # client_secret missing
        )
        # Should succeed because OAuth is not considered "has_oauth" without both
        assert model.name == "test_mcp"

    def test_valid_oauth_configuration(self):
        """Test valid OAuth configuration."""
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com/mcp",
            client_id="client_id",
            client_secret="client_secret",
            workspace_host="https://workspace.com",
        )
        assert model.client_id == "client_id"
        assert model.client_secret == "client_secret"

    def test_valid_pat_configuration(self):
        """Test valid PAT configuration."""
        model = McpFunctionModel(
            name="test_mcp",
            transport=TransportType.STREAMABLE_HTTP,
            url="https://example.com/mcp",
            pat="personal_access_token",
            workspace_host="https://workspace.com",
        )
        assert model.pat == "personal_access_token"
