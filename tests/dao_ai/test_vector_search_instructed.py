"""Integration tests for instructed retrieval control flow in vector_search.py.

Tests the retry loop, empty-result fallback, and verification flow at the
create_vector_search_tool / _vector_search_tool level with mocked dependencies.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from conftest import add_databricks_resource_attrs
from langchain_core.documents import Document

from dao_ai.config import (
    AiSearchRetrieverModel,
    ColumnInfo,
    DecompositionModel,
    FilterItem,
    InstructedRetrieverModel,
    LLMModel,
    RouterModel,
    SearchQuery,
    VectorStoreModel,
    VerificationResult,
    VerifierModel,
)

# Shared test column definitions
_TEST_COLUMNS = [
    ColumnInfo(name="brand", type="string"),
    ColumnInfo(name="text", type="string"),
]


def _create_mock_vector_store() -> Mock:
    """Create a mock VectorStoreModel for tool creation."""
    vector_store = Mock(spec=VectorStoreModel)
    vector_store.columns = ["text"]
    vector_store.embedding_model = None
    vector_store.primary_key = "id"
    vector_store.index = Mock()
    vector_store.index.full_name = "catalog.schema.test_index"
    vector_store.index.name = "test_index"
    vector_store.index.columns = ["text"]
    vector_store.endpoint = Mock()
    vector_store.source_table = None
    vector_store.embedding_source_column = None
    vector_store.doc_uri = None
    add_databricks_resource_attrs(vector_store)

    # workspace_client_from returns a mock WorkspaceClient
    mock_ws = MagicMock()
    vector_store.workspace_client_from.return_value = mock_ws

    return vector_store


def _make_instructed(
    llm_model: LLMModel,
    verifier: VerifierModel | None = None,
    router: RouterModel | None = None,
    max_subqueries: int = 3,
) -> InstructedRetrieverModel:
    """Build an InstructedRetrieverModel with the new nested config structure."""
    return InstructedRetrieverModel(
        columns=_TEST_COLUMNS,
        decomposition=DecompositionModel(
            model=llm_model,
            max_subqueries=max_subqueries,
        ),
        verifier=verifier,
        router=router,
    )


@pytest.mark.unit
class TestEmptyResultFallback:
    """Tests that empty instructed retrieval results trigger standard search fallback."""

    @patch("dao_ai.tools.instructed_pipeline.rrf_merge")
    @patch("dao_ai.tools.instructed_pipeline.decompose_query")
    @patch("dao_ai.tools.instructed_pipeline._get_cached_llm")
    @patch("dao_ai.tools.vector_search.DatabricksVectorSearch")
    def test_empty_merge_falls_back_to_standard_search(
        self,
        MockDVS,
        mock_get_llm,
        mock_decompose,
        mock_rrf,
    ):
        """When rrf_merge returns empty, _execute_instructed_retrieval should
        fall back to vs.similarity_search with base_filters only."""
        vector_store = _create_mock_vector_store()
        llm_model = LLMModel(name="test-decomp-model")
        instructed = _make_instructed(llm_model, max_subqueries=2)
        retriever = AiSearchRetrieverModel(
            vector_store=vector_store,
            instructed=instructed,
        )

        # Setup mocks
        mock_vs = MagicMock()
        MockDVS.return_value = mock_vs
        mock_get_llm.return_value = MagicMock()

        # decompose returns subqueries, but rrf_merge returns empty
        mock_decompose.return_value = [
            SearchQuery(
                text="subquery1", filters=[FilterItem(key="brand", value="ACME")]
            ),
        ]
        mock_rrf.return_value = []  # Empty merge result

        # The fallback similarity_search should return something
        fallback_docs = [Document(page_content="fallback result", metadata={"id": "1"})]
        mock_vs.similarity_search.return_value = fallback_docs

        from dao_ai.tools.vector_search import create_vector_search_tool

        tool = create_vector_search_tool(retriever=retriever)
        result_json = tool.invoke({"query": "test query"})
        result = json.loads(result_json)

        assert len(result) == 1
        assert result[0]["page_content"] == "fallback result"

        # similarity_search should have been called at least twice:
        # once for the subquery execution, once for the fallback
        # The fallback call should NOT include decomposed filters
        calls = mock_vs.similarity_search.call_args_list
        assert len(calls) >= 2
        # Last call is the fallback — should use base_filters (empty), not decomposed filters
        fallback_call = calls[-1]
        # filter should be None or empty dict (no decomposed filters)
        fallback_filter = fallback_call.kwargs.get("filter") or fallback_call[1].get(
            "filter"
        )
        assert fallback_filter is None or fallback_filter == {}

    @patch("dao_ai.tools.instructed_pipeline.rrf_merge")
    @patch("dao_ai.tools.instructed_pipeline.decompose_query")
    @patch("dao_ai.tools.instructed_pipeline._get_cached_llm")
    @patch("dao_ai.tools.vector_search.DatabricksVectorSearch")
    def test_nonempty_merge_does_not_fallback(
        self,
        MockDVS,
        mock_get_llm,
        mock_decompose,
        mock_rrf,
    ):
        """When rrf_merge returns results, no fallback should occur."""
        vector_store = _create_mock_vector_store()
        llm_model = LLMModel(name="test-decomp-model")
        instructed = _make_instructed(llm_model)
        retriever = AiSearchRetrieverModel(
            vector_store=vector_store,
            instructed=instructed,
        )

        mock_vs = MagicMock()
        MockDVS.return_value = mock_vs
        mock_get_llm.return_value = MagicMock()

        mock_decompose.return_value = [SearchQuery(text="subquery1")]
        merged_docs = [
            Document(
                page_content="merged result", metadata={"id": "1", "rrf_score": 0.5}
            )
        ]
        mock_rrf.return_value = merged_docs

        # This should NOT be called as fallback
        mock_vs.similarity_search.return_value = [
            Document(page_content="standard", metadata={})
        ]

        from dao_ai.tools.vector_search import create_vector_search_tool

        tool = create_vector_search_tool(retriever=retriever)
        result_json = tool.invoke({"query": "test query"})
        result = json.loads(result_json)

        assert len(result) == 1
        assert result[0]["page_content"] == "merged result"


@pytest.mark.unit
class TestVerificationRetryLoop:
    """Tests that verification failure re-executes instructed search with feedback."""

    @patch("dao_ai.tools.instructed_pipeline.verify_results")
    @patch("dao_ai.tools.instructed_pipeline.rrf_merge")
    @patch("dao_ai.tools.instructed_pipeline.decompose_query")
    @patch("dao_ai.tools.instructed_pipeline._get_cached_llm")
    @patch("dao_ai.tools.vector_search.DatabricksVectorSearch")
    def test_retry_re_executes_search_with_feedback(
        self,
        MockDVS,
        mock_get_llm,
        mock_decompose,
        mock_rrf,
        mock_verify,
    ):
        """When verification fails and on_failure=retry, the tool should
        re-execute instructed retrieval with previous_feedback from the verifier."""
        vector_store = _create_mock_vector_store()
        llm_model = LLMModel(name="test-model")
        verifier = VerifierModel(
            model=llm_model,
            on_failure="retry",
            max_retries=1,
        )
        instructed = _make_instructed(llm_model, verifier=verifier)
        retriever = AiSearchRetrieverModel(
            vector_store=vector_store,
            instructed=instructed,
        )

        mock_vs = MagicMock()
        MockDVS.return_value = mock_vs
        mock_get_llm.return_value = MagicMock()

        # First decomposition returns initial results
        first_docs = [Document(page_content="first attempt", metadata={"id": "1"})]
        second_docs = [Document(page_content="second attempt", metadata={"id": "2"})]

        mock_decompose.side_effect = [
            [SearchQuery(text="initial query")],
            [SearchQuery(text="adjusted query")],
        ]
        mock_rrf.side_effect = [first_docs, second_docs]
        mock_vs.similarity_search.return_value = []

        # First verification fails with feedback, second passes
        mock_verify.side_effect = [
            VerificationResult(
                passed=False,
                confidence=0.3,
                feedback="Brand filter was too restrictive",
            ),
            VerificationResult(
                passed=True,
                confidence=0.9,
            ),
        ]

        from dao_ai.tools.vector_search import create_vector_search_tool

        tool = create_vector_search_tool(retriever=retriever)
        result_json = tool.invoke({"query": "test query"})
        result = json.loads(result_json)

        # Should get second attempt results (after retry)
        assert len(result) == 1
        assert result[0]["page_content"] == "second attempt"

        # decompose_query should have been called twice
        assert mock_decompose.call_count == 2

        # Second decompose call should include previous_feedback
        second_call_kwargs = mock_decompose.call_args_list[1]
        assert (
            second_call_kwargs.kwargs.get("previous_feedback")
            == "Brand filter was too restrictive"
        )

        # verify_results should have been called twice
        assert mock_verify.call_count == 2

    @patch("dao_ai.tools.instructed_pipeline.verify_results")
    @patch("dao_ai.tools.instructed_pipeline.rrf_merge")
    @patch("dao_ai.tools.instructed_pipeline.decompose_query")
    @patch("dao_ai.tools.instructed_pipeline._get_cached_llm")
    @patch("dao_ai.tools.vector_search.DatabricksVectorSearch")
    def test_warn_mode_does_not_retry(
        self,
        MockDVS,
        mock_get_llm,
        mock_decompose,
        mock_rrf,
        mock_verify,
    ):
        """When on_failure=warn, verification failure should annotate docs but not retry."""
        vector_store = _create_mock_vector_store()
        llm_model = LLMModel(name="test-model")
        verifier = VerifierModel(
            model=llm_model,
            on_failure="warn",
            max_retries=3,
        )
        instructed = _make_instructed(llm_model, verifier=verifier)
        retriever = AiSearchRetrieverModel(
            vector_store=vector_store,
            instructed=instructed,
        )

        mock_vs = MagicMock()
        MockDVS.return_value = mock_vs
        mock_get_llm.return_value = MagicMock()

        docs = [Document(page_content="result", metadata={"id": "1"})]
        mock_decompose.return_value = [SearchQuery(text="query")]
        mock_rrf.return_value = docs

        mock_verify.return_value = VerificationResult(
            passed=False,
            confidence=0.2,
            feedback="Results don't match constraints",
        )

        from dao_ai.tools.vector_search import create_vector_search_tool

        tool = create_vector_search_tool(retriever=retriever)
        result_json = tool.invoke({"query": "test query"})
        result = json.loads(result_json)

        # decompose should only be called once (no retry)
        assert mock_decompose.call_count == 1

        # verify should only be called once
        assert mock_verify.call_count == 1

        # Result should contain verification metadata
        assert result[0]["metadata"]["_verification_status"] == "failed"

    @patch("dao_ai.tools.instructed_pipeline.verify_results")
    @patch("dao_ai.tools.instructed_pipeline.rrf_merge")
    @patch("dao_ai.tools.instructed_pipeline.decompose_query")
    @patch("dao_ai.tools.instructed_pipeline._get_cached_llm")
    @patch("dao_ai.tools.vector_search.DatabricksVectorSearch")
    def test_retries_exhausted_annotates_docs(
        self,
        MockDVS,
        mock_get_llm,
        mock_decompose,
        mock_rrf,
        mock_verify,
    ):
        """When max_retries is exhausted, docs should be annotated with exhausted status."""
        vector_store = _create_mock_vector_store()
        llm_model = LLMModel(name="test-model")
        verifier = VerifierModel(
            model=llm_model,
            on_failure="retry",
            max_retries=1,
        )
        instructed = _make_instructed(llm_model, verifier=verifier)
        retriever = AiSearchRetrieverModel(
            vector_store=vector_store,
            instructed=instructed,
        )

        mock_vs = MagicMock()
        MockDVS.return_value = mock_vs
        mock_get_llm.return_value = MagicMock()

        docs = [Document(page_content="result", metadata={"id": "1"})]
        mock_decompose.return_value = [SearchQuery(text="query")]
        mock_rrf.return_value = docs

        # Verification always fails
        mock_verify.return_value = VerificationResult(
            passed=False,
            confidence=0.1,
            feedback="Still not matching",
        )

        from dao_ai.tools.vector_search import create_vector_search_tool

        tool = create_vector_search_tool(retriever=retriever)
        result_json = tool.invoke({"query": "test query"})
        result = json.loads(result_json)

        # decompose called twice: initial + 1 retry
        assert mock_decompose.call_count == 2

        # verify called twice: initial + after retry
        assert mock_verify.call_count == 2

        # Result should have exhausted verification status
        assert result[0]["metadata"]["_verification_status"] == "exhausted"


@pytest.mark.unit
class TestStandardModeNoRetry:
    """Tests that standard mode never retries verification, even when retry is configured."""

    @patch("dao_ai.tools.instructed_pipeline.verify_results")
    @patch("dao_ai.tools.instructed_pipeline._get_cached_llm")
    @patch("dao_ai.tools.vector_search.DatabricksVectorSearch")
    def test_standard_mode_does_not_retry_on_verification_failure(
        self,
        MockDVS,
        mock_get_llm,
        mock_verify,
    ):
        """When mode is standard, verification failure should annotate docs
        but never retry — there is no decomposition to adjust."""
        vector_store = _create_mock_vector_store()
        llm_model = LLMModel(name="test-model")
        verifier = VerifierModel(
            model=llm_model,
            on_failure="retry",
            max_retries=3,
        )
        # Router with auto_bypass=False so verification actually runs in standard mode
        router = RouterModel(auto_bypass=False)
        instructed = _make_instructed(llm_model, verifier=verifier, router=router)
        retriever = AiSearchRetrieverModel(
            vector_store=vector_store,
            instructed=instructed,
        )

        mock_vs = MagicMock()
        MockDVS.return_value = mock_vs
        mock_get_llm.return_value = MagicMock()

        docs = [Document(page_content="standard result", metadata={"id": "1"})]
        mock_vs.similarity_search.return_value = docs

        # Verification fails
        mock_verify.return_value = VerificationResult(
            passed=False,
            confidence=0.2,
            feedback="Results don't match constraints",
        )

        from dao_ai.tools.vector_search import create_vector_search_tool

        tool = create_vector_search_tool(retriever=retriever)
        result_json = tool.invoke({"query": "test query"})
        result = json.loads(result_json)

        # similarity_search should be called exactly once — no retry for standard mode
        assert mock_vs.similarity_search.call_count == 1

        # verify should be called exactly once
        assert mock_verify.call_count == 1

        # Result should be annotated as failed (not exhausted)
        assert result[0]["metadata"]["_verification_status"] == "failed"
