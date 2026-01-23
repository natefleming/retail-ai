"""Unit and integration tests for vector search with reranking functionality."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from conftest import add_databricks_resource_attrs

from dao_ai.config import (
    RerankParametersModel,
    RetrieverModel,
    SchemaModel,
    TableModel,
    VectorStoreModel,
)
from dao_ai.tools.vector_search import create_vector_search_tool


@pytest.mark.unit
class TestReRankParametersModel:
    """Unit tests for ReRankParametersModel configuration."""

    def test_default_values(self) -> None:
        """Test that RerankParametersModel has sensible defaults."""
        rerank = RerankParametersModel()

        assert (
            rerank.model is None
        )  # No FlashRank by default (use columns for Databricks)
        assert rerank.top_n is None
        assert rerank.cache_dir == "~/.dao_ai/cache/flashrank"

    def test_custom_values(self) -> None:
        """Test ReRankParametersModel with custom values."""
        rerank = RerankParametersModel(
            model="ms-marco-TinyBERT-L-2-v2", top_n=10, cache_dir="/custom/cache"
        )

        assert rerank.model == "ms-marco-TinyBERT-L-2-v2"
        assert rerank.top_n == 10
        assert rerank.cache_dir == "/custom/cache"

    def test_serialization(self) -> None:
        """Test that ReRankParametersModel can be serialized."""
        rerank = RerankParametersModel(model="rank-T5-flan", top_n=5)
        dumped = rerank.model_dump()

        assert dumped["model"] == "rank-T5-flan"
        assert dumped["top_n"] == 5
        assert "cache_dir" in dumped


def create_mock_vector_store() -> Mock:
    """Create a mock VectorStoreModel with IsDatabricksResource attrs."""
    vector_store = Mock(spec=VectorStoreModel)
    vector_store.columns = ["text", "metadata"]
    vector_store.embedding_model = None
    vector_store.primary_key = "id"
    vector_store.index = Mock()
    vector_store.index.full_name = "catalog.schema.test_index"
    vector_store.endpoint = Mock()
    # New optional fields for VectorStoreModel
    vector_store.source_table = None  # Use existing index mode
    vector_store.embedding_source_column = None
    add_databricks_resource_attrs(vector_store)
    return vector_store


@pytest.mark.unit
class TestRetrieverModelWithReranker:
    """Unit tests for RetrieverModel with reranking configuration."""

    def test_rerank_as_bool_true(self) -> None:
        """Test that rerank=True is converted to RerankParametersModel with FlashRank default."""
        vector_store = create_mock_vector_store()

        retriever = RetrieverModel(vector_store=vector_store, rerank=True)

        assert isinstance(retriever.rerank, RerankParametersModel)
        # When rerank=True, the default FlashRank model is set
        assert retriever.rerank.model == "ms-marco-MiniLM-L-12-v2"
        assert retriever.rerank.top_n is None

    def test_rerank_as_bool_false(self) -> None:
        """Test that rerank=False remains False."""
        vector_store = create_mock_vector_store()

        retriever = RetrieverModel(vector_store=vector_store, rerank=False)

        assert retriever.rerank is False

    def test_rerank_as_model(self) -> None:
        """Test that ReRankParametersModel is preserved."""
        vector_store = create_mock_vector_store()

        rerank_config = RerankParametersModel(model="ms-marco-MiniLM-L-6-v2", top_n=3)
        retriever = RetrieverModel(vector_store=vector_store, rerank=rerank_config)

        assert isinstance(retriever.rerank, RerankParametersModel)
        assert retriever.rerank.model == "ms-marco-MiniLM-L-6-v2"
        assert retriever.rerank.top_n == 3

    def test_rerank_none(self) -> None:
        """Test that rerank=None remains None."""
        vector_store = create_mock_vector_store()

        retriever = RetrieverModel(vector_store=vector_store, rerank=None)

        assert retriever.rerank is None


@pytest.mark.unit
class TestVectorSearchToolCreation:
    """Unit tests for vector search tool creation using @tool decorator pattern."""

    @patch("dao_ai.tools.vector_search.DatabricksVectorSearch")
    def test_creates_tool_without_reranker(self, mock_vector_search: MagicMock) -> None:
        """Test that tool is created without reranker when not configured."""
        # Create mock retriever config without reranking
        retriever_config = Mock(spec=RetrieverModel)
        retriever_config.rerank = None
        retriever_config.instructed = None
        retriever_config.router = None
        retriever_config.verifier = None
        retriever_config.columns = ["text"]
        retriever_config.search_parameters = Mock()
        retriever_config.search_parameters.model_dump.return_value = {"num_results": 10}

        vector_store = Mock(spec=VectorStoreModel)
        vector_store.index = Mock()
        vector_store.index.full_name = "catalog.schema.index"
        vector_store.primary_key = "id"
        vector_store.doc_uri = "https://docs.example.com"
        vector_store.embedding_source_column = "text"
        vector_store.workspace_client = None
        add_databricks_resource_attrs(vector_store)
        retriever_config.vector_store = vector_store

        # Create tool
        tool = create_vector_search_tool(
            retriever=retriever_config,
            name="test_tool",
            description="Test description",
        )

        # Verify tool was created (StructuredTool in langchain 1.x is not directly callable,
        # but has .invoke() method)
        assert hasattr(tool, "invoke")
        assert tool.name == "test_tool"
        # Description includes filter columns when columns are specified
        assert tool.description.startswith("Test description")

    @patch("dao_ai.providers.databricks.DatabricksProvider")
    @patch("dao_ai.tools.vector_search.DatabricksVectorSearch")
    def test_creates_tool_from_vector_store_directly(
        self,
        mock_vector_search: MagicMock,
        mock_provider_class: MagicMock,
    ) -> None:
        """Test that tool can be created from VectorStoreModel directly with defaults."""
        # Mock the provider to avoid actual Databricks calls
        mock_provider = MagicMock()
        mock_provider.find_primary_key.return_value = ["id"]
        mock_provider.find_endpoint_for_index.return_value = "test_endpoint"
        mock_provider_class.return_value = mock_provider

        # Mock DatabricksVectorSearch to return documents
        mock_vs_instance = MagicMock()
        mock_vs_instance.similarity_search.return_value = []
        mock_vector_search.return_value = mock_vs_instance

        # Create a real VectorStoreModel (not a mock)
        schema = SchemaModel(catalog_name="test_catalog", schema_name="test_schema")
        table = TableModel(schema=schema, name="test_table")
        vector_store = VectorStoreModel(
            source_table=table,
            embedding_source_column="text",
            columns=["text", "metadata"],
            doc_uri="https://docs.example.com",
        )

        # Create tool directly from vector store using vector_store parameter
        tool = create_vector_search_tool(
            vector_store=vector_store,
            name="test_tool",
            description="Test description",
        )

        # Verify tool was created
        assert hasattr(tool, "invoke")
        assert tool.name == "test_tool"
        # Description includes filter columns when columns are specified
        assert tool.description.startswith("Test description")
        assert "Available filter columns: text, metadata" in tool.description

        # DatabricksVectorSearch uses lazy initialization - it's only created when
        # the tool is invoked. Verify the tool structure is correct without invoking.
        # The actual DatabricksVectorSearch instantiation happens at runtime for OBO support.

    def test_validation_requires_one_parameter(self) -> None:
        """Test that validation fails when neither retriever nor vector_store is provided."""
        with pytest.raises(
            ValueError, match="Must provide either 'retriever' or 'vector_store'"
        ):
            create_vector_search_tool()

    @patch("dao_ai.tools.vector_search.DatabricksVectorSearch")
    @patch("mlflow.models.set_retriever_schema")
    def test_validation_rejects_both_parameters(
        self, mock_set_schema: MagicMock, mock_vector_search: MagicMock
    ) -> None:
        """Test that validation fails when both retriever and vector_store are provided."""
        # Create mock retriever and vector store
        retriever_config = Mock(spec=RetrieverModel)
        vector_store_config = Mock(spec=VectorStoreModel)

        with pytest.raises(
            ValueError, match="Cannot provide both 'retriever' and 'vector_store'"
        ):
            create_vector_search_tool(
                retriever=retriever_config, vector_store=vector_store_config
            )

    @patch("dao_ai.tools.vector_search.DatabricksVectorSearch")
    def test_creates_tool_with_reranker(self, mock_vector_search: MagicMock) -> None:
        """Test that tool is created with reranker when configured."""
        # Create mock retriever config with reranking
        reranker_config = RerankParametersModel(
            model="ms-marco-MiniLM-L-6-v2", top_n=5, cache_dir="/tmp/test"
        )

        retriever_config = Mock(spec=RetrieverModel)
        retriever_config.rerank = reranker_config
        retriever_config.instructed = None
        retriever_config.router = None
        retriever_config.verifier = None
        retriever_config.columns = ["text"]
        retriever_config.search_parameters = Mock()
        retriever_config.search_parameters.model_dump.return_value = {"num_results": 20}

        vector_store = Mock(spec=VectorStoreModel)
        vector_store.index = Mock()
        vector_store.index.full_name = "catalog.schema.index"
        vector_store.primary_key = "id"
        vector_store.doc_uri = "https://docs.example.com"
        vector_store.embedding_source_column = "text"
        vector_store.workspace_client = None
        add_databricks_resource_attrs(vector_store)
        retriever_config.vector_store = vector_store

        # Create tool
        tool = create_vector_search_tool(
            retriever=retriever_config,
            name="reranking_tool",
            description="Reranking test",
        )

        # Verify tool was created (StructuredTool in langchain 1.x is not directly callable,
        # but has .invoke() method)
        assert hasattr(tool, "invoke")
        assert tool.name == "reranking_tool"
        # Description includes filter columns when columns are specified
        assert tool.description.startswith("Reranking test")


@pytest.mark.integration
@pytest.mark.skipif(
    True, reason="Requires Databricks workspace and vector search index"
)
class TestRerankingIntegration:
    """Integration tests for reranking with real Databricks vector search."""

    def test_reranking_with_real_index(self) -> None:
        """
        Integration test with real Databricks vector search index.

        This test requires:
        - Valid Databricks workspace credentials
        - An existing vector search index
        - FlashRank models downloaded

        To enable: Set ENABLE_INTEGRATION_TESTS=true and configure workspace
        """
        # This would be a real integration test
        # Skipped by default as it requires real Databricks resources
        pass

    def test_reranking_improves_results(self) -> None:
        """
        Test that reranking improves result quality.

        This integration test would:
        1. Query without reranking
        2. Query with reranking
        3. Verify that reranked results are more relevant
        """
        pass


@pytest.mark.unit
class TestRerankingE2E:
    """End-to-end unit tests with mocked components."""

    def test_reranker_parameters_validation(self) -> None:
        """Test that reranker parameters are valid."""
        # Test that any model name is accepted
        reranker = RerankParametersModel(model="invalid-model-name")
        assert reranker.model == "invalid-model-name"  # Should accept any string

        # Test that valid configurations work
        reranker2 = RerankParametersModel(
            model="ms-marco-MiniLM-L-12-v2", top_n=5, cache_dir="/tmp/test"
        )
        assert reranker2.top_n == 5
        assert reranker2.cache_dir == "/tmp/test"

    @patch("dao_ai.tools.vector_search.DatabricksVectorSearch")
    @patch("mlflow.models.set_retriever_schema")
    def test_tool_creation_flow(
        self, mock_set_schema: MagicMock, mock_vector_search: MagicMock
    ) -> None:
        """Test that create_vector_search_tool returns a callable tool."""
        # Create mock retriever config
        retriever_config = Mock(spec=RetrieverModel)
        retriever_config.rerank = RerankParametersModel()
        retriever_config.instructed = None
        retriever_config.router = None
        retriever_config.verifier = None
        retriever_config.columns = ["text"]
        retriever_config.search_parameters = Mock()
        retriever_config.search_parameters.model_dump.return_value = {"num_results": 10}

        vector_store = Mock(spec=VectorStoreModel)
        vector_store.index = Mock()
        vector_store.index.full_name = "catalog.schema.index"
        vector_store.primary_key = "id"
        vector_store.doc_uri = "https://docs.example.com"
        vector_store.embedding_source_column = "text"
        vector_store.workspace_client = None
        add_databricks_resource_attrs(vector_store)
        retriever_config.vector_store = vector_store

        # Create tool
        tool = create_vector_search_tool(
            retriever=retriever_config, name="test_tool", description="Test"
        )

        # Verify tool has expected attributes (StructuredTool in langchain 1.x
        # is not directly callable, but has .invoke() method)
        assert hasattr(tool, "invoke")
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")


@pytest.mark.unit
class TestRerankingDocumentation:
    """Tests to ensure reranking is well-documented."""

    def test_rerank_model_has_docstring(self) -> None:
        """Verify ReRankParametersModel has comprehensive docstring."""
        assert RerankParametersModel.__doc__ is not None
        assert "FlashRank" in RerankParametersModel.__doc__
        assert "reranking" in RerankParametersModel.__doc__.lower()

    def test_vector_search_tool_factory_has_docstring(self) -> None:
        """Verify create_vector_search_tool has comprehensive docstring."""
        assert create_vector_search_tool.__doc__ is not None
        assert "rerank" in create_vector_search_tool.__doc__.lower()

    def test_field_descriptions_present(self) -> None:
        """Verify all reranking fields have descriptions."""
        fields = RerankParametersModel.model_fields

        assert fields["model"].description is not None
        assert fields["top_n"].description is not None
        assert fields["cache_dir"].description is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
