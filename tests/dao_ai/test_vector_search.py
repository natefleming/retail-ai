from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain.tools import ToolRuntime, tool

from dao_ai.config import FilterItem
from dao_ai.tools.vector_search import VectorSearchInput
from dao_ai.vector_search import endpoint_exists, index_exists


@pytest.mark.unit
def test_endpoint_exists_with_matching_endpoint() -> None:
    """Test endpoint_exists when the endpoint exists in the list."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.return_value = {
        "endpoints": [
            {"name": "endpoint1"},
            {"name": "target_endpoint"},
            {"name": "endpoint3"},
        ]
    }

    result = endpoint_exists(mock_vsc, "target_endpoint")

    assert result is True
    mock_vsc.list_endpoints.assert_called_once()


@pytest.mark.unit
def test_endpoint_exists_with_no_matching_endpoint() -> None:
    """Test endpoint_exists when the endpoint doesn't exist in the list."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.return_value = {
        "endpoints": [
            {"name": "endpoint1"},
            {"name": "endpoint2"},
            {"name": "endpoint3"},
        ]
    }

    result = endpoint_exists(mock_vsc, "missing_endpoint")

    assert result is False
    mock_vsc.list_endpoints.assert_called_once()


@pytest.mark.unit
def test_endpoint_exists_with_empty_endpoints() -> None:
    """Test endpoint_exists when no endpoints are returned."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.return_value = {"endpoints": []}

    result = endpoint_exists(mock_vsc, "any_endpoint")

    assert result is False
    mock_vsc.list_endpoints.assert_called_once()


@pytest.mark.unit
def test_endpoint_exists_with_rate_limit_error() -> None:
    """Test endpoint_exists handles rate limit errors gracefully."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.side_effect = Exception(
        "REQUEST_LIMIT_EXCEEDED: Too many requests"
    )

    with patch("builtins.print") as mock_print:
        result = endpoint_exists(mock_vsc, "any_endpoint")

    assert result is True  # Should assume endpoint exists during rate limit
    mock_print.assert_called_once_with(
        "WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error."
    )


@pytest.mark.unit
def test_endpoint_exists_with_other_exception() -> None:
    """Test endpoint_exists re-raises non-rate-limit exceptions."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.side_effect = Exception("Some other error")

    with pytest.raises(Exception, match="Some other error"):
        endpoint_exists(mock_vsc, "any_endpoint")


@pytest.mark.unit
def test_endpoint_exists_with_missing_endpoints_key() -> None:
    """Test endpoint_exists when the response doesn't have endpoints key."""
    mock_vsc = Mock()
    mock_vsc.list_endpoints.return_value = {}

    result = endpoint_exists(mock_vsc, "any_endpoint")

    assert result is False


@pytest.mark.unit
def test_index_exists_when_index_exists() -> None:
    """Test index_exists when the index exists and describe succeeds."""
    mock_vsc = Mock()
    mock_index = Mock()
    mock_index.describe.return_value = {"status": "READY"}
    mock_vsc.get_index.return_value = mock_index

    result = index_exists(mock_vsc, "test_endpoint", "catalog.schema.table")

    assert result is True
    mock_vsc.get_index.assert_called_once_with("test_endpoint", "catalog.schema.table")
    mock_index.describe.assert_called_once()


@pytest.mark.unit
def test_index_exists_when_index_does_not_exist() -> None:
    """Test index_exists when the index doesn't exist."""
    mock_vsc = Mock()
    mock_index = Mock()
    mock_index.describe.side_effect = Exception(
        "RESOURCE_DOES_NOT_EXIST: Index not found"
    )
    mock_vsc.get_index.return_value = mock_index

    result = index_exists(mock_vsc, "test_endpoint", "catalog.schema.missing_table")

    assert result is False
    mock_vsc.get_index.assert_called_once_with(
        "test_endpoint", "catalog.schema.missing_table"
    )
    mock_index.describe.assert_called_once()


@pytest.mark.unit
def test_index_exists_with_permission_error() -> None:
    """Test index_exists with permission errors."""
    mock_vsc = Mock()
    mock_index = Mock()
    mock_index.describe.side_effect = Exception("PERMISSION_DENIED: Access denied")
    mock_vsc.get_index.return_value = mock_index

    with patch("builtins.print") as mock_print:
        with pytest.raises(Exception, match="PERMISSION_DENIED: Access denied"):
            index_exists(mock_vsc, "test_endpoint", "catalog.schema.table")

    mock_print.assert_called_once_with(
        "Unexpected error describing the index. This could be a permission issue."
    )


@pytest.mark.unit
def test_index_exists_with_other_unexpected_error() -> None:
    """Test index_exists with other unexpected errors."""
    mock_vsc = Mock()
    mock_index = Mock()
    mock_index.describe.side_effect = Exception("UNKNOWN_ERROR: Something went wrong")
    mock_vsc.get_index.return_value = mock_index

    with patch("builtins.print") as mock_print:
        with pytest.raises(Exception, match="UNKNOWN_ERROR: Something went wrong"):
            index_exists(mock_vsc, "test_endpoint", "catalog.schema.table")

    mock_print.assert_called_once_with(
        "Unexpected error describing the index. This could be a permission issue."
    )


@pytest.mark.unit
def test_index_exists_get_index_failure() -> None:
    """Test index_exists when get_index itself fails."""
    mock_vsc = Mock()
    mock_vsc.get_index.side_effect = Exception(
        "ENDPOINT_NOT_FOUND: Endpoint doesn't exist"
    )

    with pytest.raises(Exception, match="ENDPOINT_NOT_FOUND: Endpoint doesn't exist"):
        index_exists(mock_vsc, "missing_endpoint", "catalog.schema.table")


# ---------------------------------------------------------------------------
# VectorSearchInput — the args_schema we attach to the vector_search @tool
# so the LLM receives a structural JSON schema (type=array, items=FilterItem)
# for `filters` instead of a bare description string.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vector_search_input_accepts_list_of_filter_items() -> None:
    """Canonical happy path — LLM emits a list of {key, value} filter objects."""
    payload = {
        "query": "B2B return policy",
        "filters": [
            {"key": "category", "value": "B2B"},
            {"key": "price <=", "value": 150},
        ],
    }
    validated = VectorSearchInput.model_validate(payload)
    assert validated.query == "B2B return policy"
    assert validated.filters is not None and len(validated.filters) == 2
    assert all(isinstance(f, FilterItem) for f in validated.filters)
    assert validated.filters[0].key == "category"
    assert validated.filters[0].value == "B2B"


@pytest.mark.unit
def test_vector_search_input_accepts_null_filters() -> None:
    """LLM may omit filters or pass null when no constraint applies."""
    assert VectorSearchInput.model_validate({"query": "no filters"}).filters is None
    assert (
        VectorSearchInput.model_validate({"query": "explicit null", "filters": None}).filters
        is None
    )


@pytest.mark.unit
def test_vector_search_input_rejects_flat_dict_for_filters() -> None:
    """The pre-fix LLM-mistake shape (flat dict instead of list of objects)
    must be rejected by Pydantic so the tool returns a clear error rather
    than silently semi-running on a broken filter."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        VectorSearchInput.model_validate({"query": "x", "filters": {"category": "B2B"}})
    assert "filters" in str(exc.value).lower()


@pytest.mark.unit
def test_vector_search_input_ignores_injected_runtime_kwarg() -> None:
    """LangGraph's ToolNode injects `runtime` as a kwarg AFTER args_schema
    validation runs, so VectorSearchInput must tolerate the extra. Without
    this (ConfigDict(extra='forbid')) every tool call fails with
    `validation error … runtime: Extra inputs are not permitted`."""
    sentinel = object()
    payload = {
        "query": "x",
        "filters": None,
        "runtime": sentinel,  # impersonates the injected ToolRuntime
        "state": sentinel,  # impersonates injected graph state
    }
    validated = VectorSearchInput.model_validate(payload)
    # extras are silently dropped from the validated model …
    assert not hasattr(validated, "runtime")
    assert not hasattr(validated, "state")
    # … but model_dump only contains the LLM-facing fields, so downstream
    # callers can't accidentally pass forged injection through.
    assert set(validated.model_dump().keys()) == {"query", "filters"}


@pytest.mark.unit
def test_vector_search_input_json_schema_is_structural() -> None:
    """The whole point of args_schema= is that the LLM sees a *structural*
    type for `filters` (not just a description string). Pin the shape so
    a future refactor that drops the schema, or downgrades it back to an
    Annotated hint, fails this test loudly."""
    schema: dict[str, Any] = VectorSearchInput.model_json_schema()
    filters_schema = schema["properties"]["filters"]
    # anyOf with [array-of-FilterItem, null] — i.e. the LLM must pass a
    # JSON array, not a flat dict.
    any_of = filters_schema.get("anyOf") or []
    array_variant = next((v for v in any_of if v.get("type") == "array"), None)
    assert array_variant is not None, f"filters not declared as array: {filters_schema}"
    assert array_variant["items"].get("$ref", "").endswith("/FilterItem"), array_variant
    # FilterItem itself is a proper sub-schema with key+value, not free-form.
    filter_item = schema["$defs"]["FilterItem"]
    assert set(filter_item["required"]) == {"key", "value"}


@pytest.mark.unit
def test_vector_search_tool_decorator_passes_runtime_through() -> None:
    """End-to-end of the args_schema integration: a @tool-decorated function
    using VectorSearchInput as args_schema must still receive `runtime` at
    call time (LangGraph injects it; Pydantic validation tolerates it)."""
    received: dict[str, Any] = {}

    @tool(
        name_or_callable="probe_search",
        description="probe",
        args_schema=VectorSearchInput,
    )
    def probe(
        query: str,
        filters: list[FilterItem] | None = None,
        # Annotation must be the bare ToolRuntime (not Optional[ToolRuntime])
        # so LangChain's _is_injected_arg_type recognises it. = None is just
        # a default; the annotation is what gates injection.
        runtime: ToolRuntime = None,  # type: ignore[assignment]
    ) -> str:
        received["query"] = query
        received["filters"] = filters
        received["runtime"] = runtime
        return "ok"

    # LangGraph's ToolNode strips LLM-supplied injected keys and adds back the
    # trusted runtime. We simulate the final tool-call shape it produces.
    sentinel_runtime = Mock(name="ToolRuntime")
    result = probe.invoke(
        {
            "query": "test",
            "filters": [{"key": "k", "value": "v"}],
            "runtime": sentinel_runtime,
        }
    )

    assert result == "ok"
    assert received["query"] == "test"
    assert received["filters"] and received["filters"][0].key == "k"
    assert received["runtime"] is sentinel_runtime, (
        "runtime injection did not reach the function — extra='ignore' may be wrong"
    )


# ---------------------------------------------------------------------------
# OBO end-to-end through the args_schema + ToolRuntime chain:
#
# runtime.context.headers -> _get_vector_search(context) ->
# vector_store.workspace_client_from(context) -> DatabricksVectorSearch
# (workspace_client=user-scoped WC)
#
# These tests pin that the args_schema fix does NOT regress OBO: the LLM-facing
# schema only exposes {query, filters}, but the function still receives
# `runtime` and uses `runtime.context` to build an OBO-scoped vector search
# client. If LangChain ever stops re-injecting runtime after args_schema
# validation, OBO would silently degrade to SP auth — that's the future
# regression these tests protect against.
# ---------------------------------------------------------------------------


def _make_obo_vector_store_mock() -> Mock:
    """Build a minimal VectorStoreModel mock with on_behalf_of_user=True."""
    from dao_ai.config import VectorStoreModel

    vs = Mock(spec=VectorStoreModel)
    vs.columns = ["text"]
    vs.embedding_model = None
    vs.primary_key = "id"
    vs.index = Mock()
    vs.index.full_name = "catalog.schema.obo_index"
    vs.index.name = "obo_index"
    vs.index.columns = ["text"]
    vs.endpoint = Mock()
    vs.source_table = None
    vs.embedding_source_column = None
    vs.doc_uri = None
    vs.on_behalf_of_user = True
    # workspace_client_from must return a mock WC; we capture the context
    # it's called with via .call_args
    vs.workspace_client_from.return_value = MagicMock(name="obo_wc")
    # databricks-resource attrs needed by the factory
    try:
        from conftest import add_databricks_resource_attrs

        add_databricks_resource_attrs(vs)
    except ImportError:
        pass
    return vs


@pytest.mark.unit
def test_create_vector_search_tool_obo_passes_context_to_workspace_client_from() -> None:
    """When on_behalf_of_user=True and the tool is invoked with a ToolRuntime
    carrying user headers, the factory must call
    vector_store.workspace_client_from(context) with THAT same Context — so
    the user's x-forwarded-access-token flows through to the VS query."""
    from langgraph.runtime import Runtime

    from dao_ai.config import RetrieverModel
    from dao_ai.state import Context
    from dao_ai.tools.vector_search import create_vector_search_tool

    vs_model = _make_obo_vector_store_mock()
    retriever = RetrieverModel(vector_store=vs_model)

    with patch("dao_ai.tools.vector_search.DatabricksVectorSearch") as MockDVS:
        mock_vs_client = MagicMock()
        mock_vs_client.similarity_search.return_value = []
        MockDVS.return_value = mock_vs_client

        tool = create_vector_search_tool(retriever=retriever)

        # Simulate the exact shape LangGraph's ToolNode produces: it strips
        # any LLM-supplied 'runtime' and re-adds the trusted ToolRuntime.
        user_headers = {"x-forwarded-access-token": "user-dapi-token-abc"}
        ctx = Context(headers=user_headers)
        # ToolRuntime is normally constructed by ToolNode; for a unit test
        # langgraph.runtime.Runtime is the concrete carrier of `.context`.
        rt = Runtime(context=ctx)

        tool.invoke({"query": "obo probe", "filters": None, "runtime": rt})

    # The OBO branch in _get_vector_search must have asked the model for an
    # OBO-scoped WC by passing the runtime's Context through verbatim.
    assert vs_model.workspace_client_from.called, (
        "OBO regressed: workspace_client_from was never called — runtime did "
        "not reach the function or _get_vector_search was bypassed."
    )
    received_ctx = vs_model.workspace_client_from.call_args.args[0]
    assert received_ctx is ctx, (
        f"OBO context not forwarded verbatim: got {received_ctx!r}, expected {ctx!r}"
    )
    assert received_ctx.headers == user_headers, (
        f"OBO headers lost in transit: got {received_ctx.headers!r}"
    )


@pytest.mark.unit
def test_create_vector_search_tool_obo_off_does_not_use_context_headers() -> None:
    """Symmetric guard: with on_behalf_of_user=False, workspace_client_from is
    still called (to build a non-OBO WC), but the runtime injection path must
    not crash even when headers are absent."""
    from langgraph.runtime import Runtime

    from dao_ai.config import RetrieverModel
    from dao_ai.state import Context
    from dao_ai.tools.vector_search import create_vector_search_tool

    vs_model = _make_obo_vector_store_mock()
    vs_model.on_behalf_of_user = False  # OBO off
    retriever = RetrieverModel(vector_store=vs_model)

    with patch("dao_ai.tools.vector_search.DatabricksVectorSearch") as MockDVS:
        MockDVS.return_value.similarity_search.return_value = []
        tool = create_vector_search_tool(retriever=retriever)
        rt = Runtime(context=Context(headers={}))
        # Must not raise
        tool.invoke({"query": "non-obo probe", "filters": None, "runtime": rt})

    # Still called (to mint the SP-scoped WC), but headers were empty.
    assert vs_model.workspace_client_from.called
