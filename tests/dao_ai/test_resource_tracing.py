"""Unit tests for resource tracing utilities.

Tests the ``ResourceInfo`` dataclass and ``set_resource_attributes`` helper,
as well as verifying that resource attributes are correctly set in each
instrumented tool and middleware.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from dao_ai.tools.tracing import (
    ATTR_RESOURCE_NAME,
    ATTR_RESOURCE_OBO,
    ATTR_RESOURCE_TYPE,
    ResourceInfo,
    set_resource_attributes,
)

# ---------------------------------------------------------------------------
# ResourceInfo dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResourceInfo:
    """Tests for the ResourceInfo frozen dataclass."""

    def test_basic_creation(self) -> None:
        info = ResourceInfo("vector_search", True, "my_index")
        assert info.resource_type == "vector_search"
        assert info.on_behalf_of_user is True
        assert info.name == "my_index"

    def test_name_defaults_to_none(self) -> None:
        info = ResourceInfo("model_serving", False)
        assert info.name is None

    def test_frozen(self) -> None:
        info = ResourceInfo("genie", True, "space_123")
        with pytest.raises(AttributeError):
            info.resource_type = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ResourceInfo("sql_warehouse", False, "wh1")
        b = ResourceInfo("sql_warehouse", False, "wh1")
        assert a == b

    def test_inequality(self) -> None:
        a = ResourceInfo("sql_warehouse", False, "wh1")
        b = ResourceInfo("sql_warehouse", True, "wh1")
        assert a != b


# ---------------------------------------------------------------------------
# set_resource_attributes helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSetResourceAttributes:
    """Tests for the set_resource_attributes helper function."""

    @patch("dao_ai.tools.tracing.mlflow")
    def test_sets_all_attributes_with_name(self, mock_mlflow: MagicMock) -> None:
        mock_span = MagicMock()
        mock_mlflow.get_current_active_span.return_value = mock_span

        info = ResourceInfo("vector_search", True, "catalog.schema.index")
        set_resource_attributes(info)

        mock_span.set_attribute.assert_any_call(ATTR_RESOURCE_TYPE, "vector_search")
        mock_span.set_attribute.assert_any_call(ATTR_RESOURCE_OBO, True)
        mock_span.set_attribute.assert_any_call(
            ATTR_RESOURCE_NAME, "catalog.schema.index"
        )

    @patch("dao_ai.tools.tracing.mlflow")
    def test_sets_attributes_without_name(self, mock_mlflow: MagicMock) -> None:
        mock_span = MagicMock()
        mock_mlflow.get_current_active_span.return_value = mock_span

        info = ResourceInfo("model_serving", False)
        set_resource_attributes(info)

        mock_span.set_attribute.assert_any_call(ATTR_RESOURCE_TYPE, "model_serving")
        mock_span.set_attribute.assert_any_call(ATTR_RESOURCE_OBO, False)
        assert call(ATTR_RESOURCE_NAME) not in [
            c for c in mock_span.set_attribute.call_args_list if len(c.args) == 1
        ]
        # Verify name was NOT set (only 2 calls total)
        assert mock_span.set_attribute.call_count == 2

    @patch("dao_ai.tools.tracing.mlflow")
    def test_obo_false_uses_native_bool(self, mock_mlflow: MagicMock) -> None:
        mock_span = MagicMock()
        mock_mlflow.get_current_active_span.return_value = mock_span

        set_resource_attributes(ResourceInfo("genie", False, "space_1"))

        mock_span.set_attribute.assert_any_call(ATTR_RESOURCE_OBO, False)

    @patch("dao_ai.tools.tracing.mlflow")
    def test_obo_true_uses_native_bool(self, mock_mlflow: MagicMock) -> None:
        mock_span = MagicMock()
        mock_mlflow.get_current_active_span.return_value = mock_span

        set_resource_attributes(ResourceInfo("genie", True, "space_1"))

        mock_span.set_attribute.assert_any_call(ATTR_RESOURCE_OBO, True)

    @patch("dao_ai.tools.tracing.mlflow")
    def test_noop_when_no_active_span(self, mock_mlflow: MagicMock) -> None:
        mock_mlflow.get_current_active_span.return_value = None

        # Should not raise
        set_resource_attributes(ResourceInfo("mcp", True, "my_tool"))


# ---------------------------------------------------------------------------
# Integration: router passes resource_info through
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRouterResourceInfo:
    """Verify route_query sets resource attributes when resource_info is provided."""

    @patch("dao_ai.tools.router._load_prompt_template")
    @patch("dao_ai.tools.router.mlflow")
    @patch("dao_ai.tools.router.set_resource_attributes")
    def test_resource_info_forwarded(
        self,
        mock_set: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_prompt: MagicMock,
    ) -> None:
        from dao_ai.tools.router import RouterDecision, route_query

        mock_load_prompt.return_value = {"template": "{schema_description} {query}"}
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RouterDecision(mode="standard")

        info = ResourceInfo("model_serving", True, "my_model")
        route_query(llm=mock_llm, query="test", columns=[], resource_info=info)

        mock_set.assert_called_once_with(info)

    @patch("dao_ai.tools.router._load_prompt_template")
    @patch("dao_ai.tools.router.mlflow")
    @patch("dao_ai.tools.router.set_resource_attributes")
    def test_no_resource_info_skips(
        self,
        mock_set: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_prompt: MagicMock,
    ) -> None:
        from dao_ai.tools.router import RouterDecision, route_query

        mock_load_prompt.return_value = {"template": "{schema_description} {query}"}
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RouterDecision(mode="standard")

        route_query(llm=mock_llm, query="test", columns=[])

        mock_set.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: decompose_query passes resource_info through
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDecomposeQueryResourceInfo:
    """Verify decompose_query sets resource attributes when resource_info is provided."""

    @patch("dao_ai.tools.instructed_retriever._load_prompt_template")
    @patch("dao_ai.tools.instructed_retriever.mlflow")
    @patch("dao_ai.tools.instructed_retriever.set_resource_attributes")
    def test_resource_info_forwarded(
        self,
        mock_set: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_prompt: MagicMock,
    ) -> None:
        from dao_ai.config import DecomposedQueries
        from dao_ai.tools.instructed_retriever import decompose_query

        mock_load_prompt.return_value = {"template": "{query}"}

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = DecomposedQueries(queries=[])

        info = ResourceInfo("model_serving", False, "decomp_model")
        decompose_query(llm=mock_llm, query="test", columns=[], resource_info=info)

        mock_set.assert_called_once_with(info)


# ---------------------------------------------------------------------------
# Integration: instruction_aware_rerank passes resource_info through
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRerankerResourceInfo:
    """Verify instruction_aware_rerank sets resource attributes when resource_info is provided."""

    @patch("dao_ai.tools.instruction_reranker._load_prompt_template")
    @patch("dao_ai.tools.instruction_reranker.mlflow")
    @patch("dao_ai.tools.instruction_reranker.set_resource_attributes")
    def test_resource_info_forwarded(
        self,
        mock_set: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_prompt: MagicMock,
    ) -> None:
        from langchain_core.documents import Document

        from dao_ai.config import RankingResult
        from dao_ai.tools.instruction_reranker import instruction_aware_rerank

        mock_load_prompt.return_value = {"template": "{query}"}

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = RankingResult(rankings=[])

        info = ResourceInfo("model_serving", True, "rerank_model")
        instruction_aware_rerank(
            llm=mock_llm,
            query="test",
            documents=[Document(page_content="doc1")],
            resource_info=info,
        )

        mock_set.assert_called_once_with(info)


# ---------------------------------------------------------------------------
# Integration: verify_results passes resource_info through
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVerifierResourceInfo:
    """Verify verify_results sets resource attributes when resource_info is provided."""

    @patch("dao_ai.tools.verifier._load_prompt_template")
    @patch("dao_ai.tools.verifier.mlflow")
    @patch("dao_ai.tools.verifier.set_resource_attributes")
    def test_resource_info_forwarded(
        self,
        mock_set: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_prompt: MagicMock,
    ) -> None:
        from langchain_core.documents import Document

        from dao_ai.config import VerificationResult
        from dao_ai.tools.verifier import verify_results

        mock_load_prompt.return_value = {
            "template": "{query} {results_summary} {constraints}",
        }
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = VerificationResult(
            passed=True, confidence=0.95, unmet_constraints=[], feedback=""
        )

        info = ResourceInfo("model_serving", False, "verifier_model")
        verify_results(
            llm=mock_llm,
            query="test",
            documents=[Document(page_content="doc1")],
            columns=[],
            resource_info=info,
        )

        mock_set.assert_called_once_with(info)


# ---------------------------------------------------------------------------
# Integration: OBOModelMiddleware sets resource attributes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOBOMiddlewareResourceInfo:
    """Verify OBOModelMiddleware sets resource attributes on model calls."""

    @patch("dao_ai.middleware.obo.set_resource_attributes")
    @patch("dao_ai.middleware.obo.ChatDatabricks")
    def test_sync_wrap_sets_attributes(
        self,
        mock_chat: MagicMock,
        mock_set: MagicMock,
    ) -> None:
        from dao_ai.config import LLMModel
        from dao_ai.middleware.obo import OBOModelMiddleware

        llm_model = MagicMock(spec=LLMModel)
        llm_model.name = "my-obo-model"
        llm_model.on_behalf_of_user = True
        llm_model.temperature = 0.0
        llm_model.max_tokens = None
        llm_model.use_responses_api = False

        middleware = OBOModelMiddleware(llm_model)

        request = MagicMock()
        request.runtime.context = None
        handler = MagicMock(return_value="response")

        middleware.wrap_model_call(request, handler)

        mock_set.assert_called_once()
        info: ResourceInfo = mock_set.call_args[0][0]
        assert info.resource_type == "model_serving"
        assert info.on_behalf_of_user is True
        assert info.name == "my-obo-model"

    @patch("dao_ai.middleware.obo.set_resource_attributes")
    @patch("dao_ai.middleware.obo.ChatDatabricks")
    def test_async_wrap_sets_attributes(
        self,
        mock_chat: MagicMock,
        mock_set: MagicMock,
    ) -> None:
        import asyncio

        from dao_ai.config import LLMModel
        from dao_ai.middleware.obo import OBOModelMiddleware

        llm_model = MagicMock(spec=LLMModel)
        llm_model.name = "my-obo-model"
        llm_model.on_behalf_of_user = True
        llm_model.temperature = 0.0
        llm_model.max_tokens = None
        llm_model.use_responses_api = False

        middleware = OBOModelMiddleware(llm_model)

        request = MagicMock()
        request.runtime.context = None

        async def async_handler(req: MagicMock) -> str:
            return "response"

        asyncio.run(middleware.awrap_model_call(request, async_handler))

        mock_set.assert_called_once()
        info: ResourceInfo = mock_set.call_args[0][0]
        assert info.resource_type == "model_serving"
        assert info.on_behalf_of_user is True
        assert info.name == "my-obo-model"
