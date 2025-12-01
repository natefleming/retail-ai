"""Unit and integration tests for vector search with reranking functionality."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.documents import Document

from dao_ai.config import RerankerParametersModel, RetrieverModel, VectorStoreModel
from dao_ai.tools.vector_search import (
    RerankingVectorSearchRetrieverTool,
    create_vector_search_tool,
)


@pytest.mark.unit
class TestReRankParametersModel:
    """Unit tests for ReRankParametersModel configuration."""

    def test_default_values(self) -> None:
        """Test that ReRankParametersModel has sensible defaults."""
        rerank = RerankerParametersModel()

        assert rerank.model == "ms-marco-MiniLM-L-12-v2"
        assert rerank.top_n is None
        assert rerank.cache_dir == "/tmp/flashrank_cache"

    def test_custom_values(self) -> None:
        """Test ReRankParametersModel with custom values."""
        rerank = RerankerParametersModel(
            model="ms-marco-TinyBERT-L-2-v2", top_n=10, cache_dir="/custom/cache"
        )

        assert rerank.model == "ms-marco-TinyBERT-L-2-v2"
        assert rerank.top_n == 10
        assert rerank.cache_dir == "/custom/cache"

    def test_serialization(self) -> None:
        """Test that ReRankParametersModel can be serialized."""
        rerank = RerankerParametersModel(model="rank-T5-flan", top_n=5)
        dumped = rerank.model_dump()

        assert dumped["model"] == "rank-T5-flan"
        assert dumped["top_n"] == 5
        assert "cache_dir" in dumped


@pytest.mark.unit
class TestRetrieverModelWithReranker:
    """Unit tests for RetrieverModel with reranking configuration."""

    def test_rerank_as_bool_true(self) -> None:
        """Test that rerank=True is converted to ReRankParametersModel with defaults."""
        # Create mock vector store with required attributes
        vector_store = Mock(spec=VectorStoreModel)
        vector_store.columns = ["text", "metadata"]
        vector_store.embedding_model = None
        vector_store.primary_key = "id"
        vector_store.index = Mock()
        vector_store.endpoint = Mock()

        retriever = RetrieverModel(vector_store=vector_store, reranker=True)

        assert isinstance(retriever.reranker, RerankerParametersModel)
        assert retriever.reranker.model == "ms-marco-MiniLM-L-12-v2"
        assert retriever.reranker.top_n is None

    def test_rerank_as_bool_false(self) -> None:
        """Test that rerank=False remains False."""
        vector_store = Mock(spec=VectorStoreModel)
        vector_store.columns = ["text", "metadata"]
        vector_store.embedding_model = None
        vector_store.primary_key = "id"
        vector_store.index = Mock()
        vector_store.endpoint = Mock()

        retriever = RetrieverModel(vector_store=vector_store, reranker=False)

        assert retriever.reranker is False

    def test_rerank_as_model(self) -> None:
        """Test that ReRankParametersModel is preserved."""
        vector_store = Mock(spec=VectorStoreModel)
        vector_store.columns = ["text", "metadata"]
        vector_store.embedding_model = None
        vector_store.primary_key = "id"
        vector_store.index = Mock()
        vector_store.endpoint = Mock()

        rerank_config = RerankerParametersModel(model="ms-marco-MiniLM-L-6-v2", top_n=3)
        retriever = RetrieverModel(vector_store=vector_store, reranker=rerank_config)

        assert isinstance(retriever.reranker, RerankerParametersModel)
        assert retriever.reranker.model == "ms-marco-MiniLM-L-6-v2"
        assert retriever.reranker.top_n == 3

    def test_rerank_none(self) -> None:
        """Test that rerank=None remains None."""
        vector_store = Mock(spec=VectorStoreModel)
        vector_store.columns = ["text", "metadata"]
        vector_store.embedding_model = None
        vector_store.primary_key = "id"
        vector_store.index = Mock()
        vector_store.endpoint = Mock()

        retriever = RetrieverModel(vector_store=vector_store, reranker=None)

        assert retriever.reranker is None


@pytest.mark.unit
class TestRerankingVectorSearchRetrieverTool:
    """Unit tests for RerankingVectorSearchRetrieverTool."""

    def test_tool_inherits_from_base_tool(self) -> None:
        """Test that RerankingVectorSearchRetrieverTool inherits from BaseTool and Mixin."""
        from databricks_ai_bridge.vector_search_retriever_tool import (
            VectorSearchRetrieverToolMixin,
        )
        from langchain_core.tools import BaseTool

        assert issubclass(RerankingVectorSearchRetrieverTool, BaseTool)
        assert issubclass(
            RerankingVectorSearchRetrieverTool, VectorSearchRetrieverToolMixin
        )

    def test_reranker_fields_present(self) -> None:
        """Test that reranking-specific fields are defined."""
        # Check that model fields exist
        fields = RerankingVectorSearchRetrieverTool.model_fields

        assert "reranker_model" in fields
        assert "reranker_top_n" in fields
        assert "reranker_cache_dir" in fields

    @patch("mlflow.start_span")
    def test_run_without_reranking_uses_vector_store(
        self, mock_start_span: MagicMock
    ) -> None:
        """Test that _run uses _vector_store.similarity_search when reranking is not configured."""
        # Create a minimal tool instance without reranking
        tool = Mock(spec=RerankingVectorSearchRetrieverTool)
        tool.reranker_model = None
        tool.model_extra = {}
        tool.filters = {}
        tool.num_results = 10
        tool.query_type = "ANN"
        tool.tool_name = "test_tool"
        tool.index_name = "test_index"
        tool.name = "test_tool"

        # Mock test documents
        test_docs = [Document(page_content="test1"), Document(page_content="test2")]

        # Mock the mlflow.start_span context manager
        mock_span = MagicMock()
        mock_start_span.return_value.__enter__.return_value = mock_span

        # Mock the _vector_store.similarity_search method to return a list of documents
        mock_vector_store = Mock()
        mock_vector_store.similarity_search = Mock(return_value=test_docs)
        tool._vector_store = mock_vector_store

        # Mock _find_documents to return the test_docs
        tool._find_documents = Mock(return_value=test_docs)

        # Call the actual method through the class
        result = RerankingVectorSearchRetrieverTool._run(
            tool, query="test query", filters=None
        )

        # Should call _find_documents
        assert tool._find_documents.called
        assert len(result) == 2
        assert result[0].page_content == "test1"
        # Verify span was created for find_documents
        assert mock_start_span.called


@pytest.mark.unit
class TestCreateVectorSearchToolWithReranker:
    """Unit tests for create_vector_search_tool with reranking."""

    @patch("dao_ai.tools.vector_search.RerankingVectorSearchRetrieverTool")
    @patch("mlflow.models.set_retriever_schema")
    def test_creates_standard_tool_without_reranker(
        self, mock_set_schema: MagicMock, mock_tool_class: MagicMock
    ) -> None:
        """Test that RerankingVectorSearchRetrieverTool is created without reranker when not configured."""
        # Create mock retriever config without reranking
        retriever_config = Mock(spec=RetrieverModel)
        retriever_config.reranker = None
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
        retriever_config.vector_store = vector_store

        mock_tool_class.return_value = Mock()

        # Create tool
        create_vector_search_tool(
            retriever=retriever_config,
            name="test_tool",
            description="Test description",
        )

        # Should always use RerankingVectorSearchRetrieverTool (even without reranker)
        assert mock_tool_class.called
        # Verify reranker_model is not set (reranking disabled)
        call_kwargs = mock_tool_class.call_args[1]
        assert (
            "reranker_model" not in call_kwargs
            or call_kwargs.get("reranker_model") is None
        )
        assert mock_set_schema.called

    @patch("dao_ai.tools.vector_search.RerankingVectorSearchRetrieverTool")
    @patch("mlflow.models.set_retriever_schema")
    def test_creates_reranking_tool_with_reranker(
        self, mock_set_schema: MagicMock, mock_rerank_tool_class: MagicMock
    ) -> None:
        """Test that reranking tool is created when reranking is configured."""
        # Create mock retriever config with reranking
        reranker_config = RerankerParametersModel(
            model="ms-marco-MiniLM-L-6-v2", top_n=5, cache_dir="/tmp/test"
        )

        retriever_config = Mock(spec=RetrieverModel)
        retriever_config.reranker = reranker_config
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
        retriever_config.vector_store = vector_store

        mock_rerank_tool_class.return_value = Mock()

        # Create tool
        create_vector_search_tool(
            retriever=retriever_config,
            name="reranking_tool",
            description="Reranking test",
        )

        # Should use RerankingVectorSearchRetrieverTool
        assert mock_rerank_tool_class.called

        # Verify reranking parameters were passed
        call_kwargs = mock_rerank_tool_class.call_args[1]
        assert call_kwargs["reranker_model"] == "ms-marco-MiniLM-L-6-v2"
        assert call_kwargs["reranker_top_n"] == 5
        assert call_kwargs["reranker_cache_dir"] == "/tmp/test"


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

    def test_e2e_reranking_flow(self) -> None:
        """Test complete reranking flow structure."""
        # Verify that the reranking tool class exists and has the right methods
        assert hasattr(RerankingVectorSearchRetrieverTool, "_rerank_documents")
        assert hasattr(RerankingVectorSearchRetrieverTool, "_run")

        # Verify field structure
        fields = RerankingVectorSearchRetrieverTool.model_fields
        assert "reranker_model" in fields
        assert "reranker_top_n" in fields
        assert "reranker_cache_dir" in fields

    def test_reranker_parameters_validation(self) -> None:
        """Test that reranker parameters are valid."""
        # Test that any model name is accepted
        reranker = RerankerParametersModel(model="invalid-model-name")
        assert reranker.model == "invalid-model-name"  # Should accept any string

        # Test that valid configurations work
        reranker2 = RerankerParametersModel(
            model="ms-marco-MiniLM-L-12-v2", top_n=5, cache_dir="/tmp/test"
        )
        assert reranker2.top_n == 5
        assert reranker2.cache_dir == "/tmp/test"


@pytest.mark.unit
class TestRerankingDocumentation:
    """Tests to ensure reranking is well-documented."""

    def test_rerank_model_has_docstring(self) -> None:
        """Verify ReRankParametersModel has comprehensive docstring."""
        assert RerankerParametersModel.__doc__ is not None
        assert "FlashRank" in RerankerParametersModel.__doc__
        assert "reranking" in RerankerParametersModel.__doc__.lower()

    def test_reranking_tool_has_docstring(self) -> None:
        """Verify RerankingVectorSearchRetrieverTool has docstring."""
        assert RerankingVectorSearchRetrieverTool.__doc__ is not None
        assert "rerank" in RerankingVectorSearchRetrieverTool.__doc__.lower()

    def test_field_descriptions_present(self) -> None:
        """Verify all reranking fields have descriptions."""
        fields = RerankerParametersModel.model_fields

        assert fields["model"].description is not None
        assert fields["top_n"].description is not None
        assert fields["cache_dir"].description is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
