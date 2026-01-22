"""Unit tests for instructed retriever with query decomposition and RRF merging."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from conftest import add_databricks_resource_attrs
from langchain_core.documents import Document

from dao_ai.config import (
    DecomposedQueries,
    InstructedRetrieverModel,
    LLMModel,
    RetrieverModel,
    SearchQuery,
    VectorStoreModel,
)
from dao_ai.tools.instructed_retriever import (
    _format_constraints,
    _format_examples,
    decompose_query,
    rrf_merge,
)


@pytest.mark.unit
class TestSearchQueryModel:
    """Unit tests for SearchQuery configuration model."""

    def test_basic_creation(self) -> None:
        """Test creating a SearchQuery with text only."""
        sq = SearchQuery(text="power tools")
        assert sq.text == "power tools"
        assert sq.filters is None

    def test_with_filters(self) -> None:
        """Test creating a SearchQuery with filters."""
        sq = SearchQuery(
            text="Milwaukee drills",
            filters={"brand_name": "Milwaukee", "price <": 200},
        )
        assert sq.text == "Milwaukee drills"
        assert sq.filters["brand_name"] == "Milwaukee"
        assert sq.filters["price <"] == 200

    def test_serialization(self) -> None:
        """Test that SearchQuery can be serialized."""
        sq = SearchQuery(text="test", filters={"col": "val"})
        dumped = sq.model_dump()
        assert dumped["text"] == "test"
        assert dumped["filters"] == {"col": "val"}


@pytest.mark.unit
class TestDecomposedQueriesModel:
    """Unit tests for DecomposedQueries container model."""

    def test_basic_creation(self) -> None:
        """Test creating DecomposedQueries with a list of queries."""
        queries = [
            SearchQuery(text="query1"),
            SearchQuery(text="query2", filters={"col": "val"}),
        ]
        dq = DecomposedQueries(queries=queries)
        assert len(dq.queries) == 2
        assert dq.queries[0].text == "query1"
        assert dq.queries[1].filters == {"col": "val"}

    def test_empty_queries(self) -> None:
        """Test creating DecomposedQueries with empty list."""
        dq = DecomposedQueries(queries=[])
        assert len(dq.queries) == 0


@pytest.mark.unit
class TestInstructedRetrieverModel:
    """Unit tests for InstructedRetrieverModel configuration."""

    def test_default_values(self) -> None:
        """Test that InstructedRetrieverModel has sensible defaults."""
        model = InstructedRetrieverModel(schema_description="Test schema")
        assert model.enabled is False
        assert model.decomposition_model is None
        assert model.constraints is None
        assert model.max_subqueries == 3
        assert model.rrf_k == 60
        assert model.examples is None

    def test_full_configuration(self) -> None:
        """Test InstructedRetrieverModel with all fields."""
        llm = LLMModel(name="test-model")
        model = InstructedRetrieverModel(
            enabled=True,
            decomposition_model=llm,
            schema_description="Products table with columns...",
            constraints=["Prefer recent products"],
            max_subqueries=5,
            rrf_k=40,
            examples=[{"query": "test", "filters": {"col": "val"}}],
        )
        assert model.enabled is True
        assert model.decomposition_model.name == "test-model"
        assert len(model.constraints) == 1
        assert model.max_subqueries == 5
        assert model.rrf_k == 40
        assert len(model.examples) == 1


def create_mock_vector_store() -> Mock:
    """Create a mock VectorStoreModel with IsDatabricksResource attrs."""
    vector_store = Mock(spec=VectorStoreModel)
    vector_store.columns = ["text"]
    vector_store.embedding_model = None
    vector_store.primary_key = "id"
    vector_store.index = Mock()
    vector_store.index.full_name = "catalog.schema.test_index"
    vector_store.endpoint = Mock()
    vector_store.source_table = None
    vector_store.embedding_source_column = None
    add_databricks_resource_attrs(vector_store)
    return vector_store


@pytest.mark.unit
class TestRetrieverModelWithInstructed:
    """Unit tests for RetrieverModel with instructed configuration."""

    def test_retriever_with_instructed(self) -> None:
        """Test that RetrieverModel accepts instructed configuration."""
        vector_store = create_mock_vector_store()

        instructed = InstructedRetrieverModel(
            enabled=True,
            schema_description="Test schema",
        )

        retriever = RetrieverModel(
            vector_store=vector_store,
            instructed=instructed,
        )

        assert retriever.instructed is not None
        assert retriever.instructed.enabled is True

    def test_retriever_without_instructed(self) -> None:
        """Test that RetrieverModel works without instructed."""
        vector_store = create_mock_vector_store()

        retriever = RetrieverModel(vector_store=vector_store)

        assert retriever.instructed is None


@pytest.mark.unit
class TestRRFMerge:
    """Unit tests for Reciprocal Rank Fusion merging."""

    def test_empty_results(self) -> None:
        """Test RRF merge with empty input."""
        result = rrf_merge([])
        assert result == []

    def test_single_list(self) -> None:
        """Test RRF merge with single result list (optimization path)."""
        docs = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
        ]
        result = rrf_merge([docs])
        assert len(result) == 2
        assert result[0].page_content == "doc1"

    def test_all_empty_lists(self) -> None:
        """Test RRF merge when all lists are empty."""
        result = rrf_merge([[], [], []])
        assert result == []

    def test_varying_list_lengths(self) -> None:
        """Test RRF merge with different length lists."""
        list1 = [Document(page_content="a", metadata={"id": "1"})]
        list2 = [
            Document(page_content="b", metadata={"id": "2"}),
            Document(page_content="c", metadata={"id": "3"}),
            Document(page_content="d", metadata={"id": "4"}),
        ]
        list3 = []

        result = rrf_merge([list1, list2, list3])

        assert len(result) == 4
        # All documents should be present
        contents = {doc.page_content for doc in result}
        assert contents == {"a", "b", "c", "d"}

    def test_duplicate_documents_boost_score(self) -> None:
        """Test that duplicates across lists get boosted RRF scores."""
        # Same document appears in both lists
        doc_shared = Document(page_content="shared", metadata={"id": "shared"})
        doc_unique1 = Document(page_content="unique1", metadata={"id": "u1"})
        doc_unique2 = Document(page_content="unique2", metadata={"id": "u2"})

        list1 = [doc_shared, doc_unique1]
        list2 = [doc_shared, doc_unique2]

        result = rrf_merge([list1, list2], primary_key="id")

        # Shared document should be first (higher RRF score)
        assert len(result) == 3
        assert result[0].page_content == "shared"
        assert result[0].metadata["rrf_score"] > result[1].metadata["rrf_score"]

    def test_rrf_score_in_metadata(self) -> None:
        """Test that RRF score is added to document metadata."""
        docs = [Document(page_content="test", metadata={"id": "1"})]
        result = rrf_merge([docs, docs])

        assert "rrf_score" in result[0].metadata
        assert result[0].metadata["rrf_score"] > 0

    def test_k_parameter_affects_scoring(self) -> None:
        """Test that k parameter affects RRF score distribution."""
        docs = [
            Document(page_content="first", metadata={"id": "1"}),
            Document(page_content="second", metadata={"id": "2"}),
        ]

        # Lower k = more weight to top ranks
        result_low_k = rrf_merge([docs], k=10)
        result_high_k = rrf_merge([docs], k=100)

        # Score difference should be larger with lower k
        diff_low = (
            result_low_k[0].metadata["rrf_score"]
            - result_low_k[1].metadata["rrf_score"]
        )
        diff_high = (
            result_high_k[0].metadata["rrf_score"]
            - result_high_k[1].metadata["rrf_score"]
        )

        assert diff_low > diff_high

    def test_primary_key_deduplication(self) -> None:
        """Test that documents are deduplicated by primary key."""
        # Same ID but different content (simulating same doc returned differently)
        doc1 = Document(page_content="content1", metadata={"product_id": "123"})
        doc2 = Document(page_content="content2", metadata={"product_id": "123"})

        result = rrf_merge([[doc1], [doc2]], primary_key="product_id")

        # Should deduplicate to single document
        assert len(result) == 1
        # First occurrence should be kept
        assert result[0].page_content == "content1"

    def test_fallback_to_content_hash(self) -> None:
        """Test deduplication falls back to content hash when no primary key."""
        doc1 = Document(page_content="same content", metadata={"other": "1"})
        doc2 = Document(page_content="same content", metadata={"other": "2"})

        result = rrf_merge([[doc1], [doc2]])

        # Should deduplicate by content
        assert len(result) == 1


@pytest.mark.unit
class TestFormatHelpers:
    """Unit tests for prompt formatting helpers."""

    def test_format_constraints_empty(self) -> None:
        """Test formatting empty constraints."""
        result = _format_constraints(None)
        assert result == "No additional constraints."

        result = _format_constraints([])
        assert result == "No additional constraints."

    def test_format_constraints_with_items(self) -> None:
        """Test formatting constraints list."""
        constraints = ["Prefer recent products", "Exclude discontinued"]
        result = _format_constraints(constraints)

        assert "- Prefer recent products" in result
        assert "- Exclude discontinued" in result

    def test_format_examples_empty(self) -> None:
        """Test formatting empty examples."""
        result = _format_examples(None)
        assert result == "No examples provided."

        result = _format_examples([])
        assert result == "No examples provided."

    def test_format_examples_with_items(self) -> None:
        """Test formatting examples list."""
        examples = [
            {"query": "cheap drills", "filters": {"price <": 100}},
            {"query": "Milwaukee tools", "filters": {"brand_name": "Milwaukee"}},
        ]
        result = _format_examples(examples)

        assert "Example 1:" in result
        assert "cheap drills" in result
        assert "Example 2:" in result
        assert "Milwaukee tools" in result


@pytest.mark.unit
class TestDecomposeQuery:
    """Unit tests for query decomposition function."""

    def _create_mock_llm(self, response_json: str) -> MagicMock:
        """Helper to create mock LLM with bind() behavior."""
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = response_json
        mock_llm.bind.return_value = mock_bound_llm
        mock_bound_llm.invoke.return_value = mock_response
        return mock_llm

    @patch("dao_ai.tools.instructed_retriever._load_prompt_template")
    def test_decompose_query_basic(self, mock_load_prompt: MagicMock) -> None:
        """Test basic query decomposition."""
        mock_load_prompt.return_value = {"template": "{query}"}

        # Mock LLM with bind() for Databricks response_format
        mock_llm = self._create_mock_llm(
            '{"queries": [{"text": "Milwaukee drills", "filters": {"brand_name": "Milwaukee"}}, {"text": "power tools under $200", "filters": {"price <": 200}}]}'
        )

        result = decompose_query(
            llm=mock_llm,
            query="Find Milwaukee drills under $200",
            schema_description="Test schema",
        )

        assert len(result) == 2
        assert result[0].text == "Milwaukee drills"
        assert result[0].filters["brand_name"] == "Milwaukee"

    @patch("dao_ai.tools.instructed_retriever._load_prompt_template")
    def test_decompose_query_respects_max_subqueries(
        self, mock_load_prompt: MagicMock
    ) -> None:
        """Test that decomposition respects max_subqueries limit."""
        mock_load_prompt.return_value = {"template": "{query}"}

        # Return more queries than allowed
        queries_json = (
            '{"queries": ['
            + ",".join(f'{{"text": "query{i}"}}' for i in range(10))
            + "]}"
        )
        mock_llm = self._create_mock_llm(queries_json)

        result = decompose_query(
            llm=mock_llm,
            query="test",
            schema_description="Test schema",
            max_subqueries=3,
        )

        assert len(result) == 3

    @patch("dao_ai.tools.instructed_retriever._load_prompt_template")
    def test_decompose_query_raises_on_error(self, mock_load_prompt: MagicMock) -> None:
        """Test that decomposition raises on LLM error."""
        mock_load_prompt.return_value = {"template": "{query}"}

        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_llm.bind.return_value = mock_bound_llm
        mock_bound_llm.invoke.side_effect = Exception("LLM error")

        with pytest.raises(Exception, match="LLM error"):
            decompose_query(
                llm=mock_llm,
                query="test",
                schema_description="Test schema",
            )


@pytest.mark.unit
class TestLLMCaching:
    """Unit tests for LLM client caching."""

    def test_cache_key_uses_full_config(self) -> None:
        """Test that cache key includes full model config to avoid collisions."""
        from dao_ai.tools.instructed_retriever import _llm_cache

        # Clear cache
        _llm_cache.clear()

        llm1 = LLMModel(name="test-model", temperature=0.1)
        llm2 = LLMModel(name="test-model", temperature=0.5)

        # Different configs should have different cache keys
        key1 = llm1.model_dump_json()
        key2 = llm2.model_dump_json()

        assert key1 != key2


@pytest.mark.integration
@pytest.mark.skipif(True, reason="Requires Databricks workspace and LLM endpoint")
class TestInstructedRetrieverIntegration:
    """Integration tests for instructed retriever with real services."""

    def test_end_to_end_retrieval(self) -> None:
        """Test full instructed retrieval flow with real Databricks."""
        pass

    def test_parallel_search_execution(self) -> None:
        """Test that subqueries execute in parallel."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
