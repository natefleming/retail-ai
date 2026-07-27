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
        VectorSearchInput.model_validate(
            {"query": "explicit null", "filters": None}
        ).filters
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
def test_create_vector_search_tool_obo_passes_context_to_workspace_client_from() -> (
    None
):
    """When on_behalf_of_user=True and the tool is invoked with a ToolRuntime
    carrying user headers, the factory must call
    vector_store.workspace_client_from(context) with THAT same Context — so
    the user's x-forwarded-access-token flows through to the VS query."""
    from langgraph.runtime import Runtime

    from dao_ai.config import AiSearchRetrieverModel
    from dao_ai.state import Context
    from dao_ai.tools.vector_search import create_vector_search_tool

    vs_model = _make_obo_vector_store_mock()
    retriever = AiSearchRetrieverModel(vector_store=vs_model)

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

    from dao_ai.config import AiSearchRetrieverModel
    from dao_ai.state import Context
    from dao_ai.tools.vector_search import create_vector_search_tool

    vs_model = _make_obo_vector_store_mock()
    vs_model.on_behalf_of_user = False  # OBO off
    retriever = AiSearchRetrieverModel(vector_store=vs_model)

    with patch("dao_ai.tools.vector_search.DatabricksVectorSearch") as MockDVS:
        MockDVS.return_value.similarity_search.return_value = []
        tool = create_vector_search_tool(retriever=retriever)
        rt = Runtime(context=Context(headers={}))
        # Must not raise
        tool.invoke({"query": "non-obo probe", "filters": None, "runtime": rt})

    # Still called (to mint the SP-scoped WC), but headers were empty.
    assert vs_model.workspace_client_from.called


# ---------------------------------------------------------------------------
# Auth-mode matrix for _client_args_from_ambient_wc + effective_client_args
# in create_vector_search_tool. All four modes must resolve on Serverless v5:
#
#   #1  ambient App SP  → auth_type=oauth-m2m       → client_args=None (library-native)
#   #2  ambient user    → auth_type=databricks-cli  → client_args={ws_url, PAT} (dao-ai fills in)
#   #3  OBO             → on_behalf_of_user=true    → client_args=None (library uses WC)
#   #4  explicit PAT/SP → YAML vector_store.pat|.client_id/.client_secret → client_args populated
#
# The existing OBO tests above cover mode #3 for the runtime-context path.
# The tests below pin the four modes at the DatabricksVectorSearch(...) boundary:
# they capture the client_args kwarg dao-ai passes and assert the exact shape
# each mode should produce. Without the ambient-user fallback, mode #2 would
# leave client_args={} and VectorSearchClient({disable_notice:True}) would
# raise InvalidInputException. All four modes are the regression watchdog.
# ---------------------------------------------------------------------------


def _make_ambient_vector_store_mock(*, on_behalf_of_user: bool = False) -> Mock:
    """VectorStoreModel mock with all four auth-mode fields defaulted to None
    (no explicit PAT/SP). Tests override individual fields as needed.
    """
    from dao_ai.config import VectorStoreModel

    vs = Mock(spec=VectorStoreModel)
    vs.columns = ["text"]
    vs.embedding_model = None
    vs.primary_key = "id"
    vs.index = Mock()
    vs.index.full_name = "catalog.schema.some_index"
    vs.index.name = "some_index"
    vs.index.columns = ["text"]
    vs.endpoint = Mock()
    vs.source_table = None
    vs.embedding_source_column = None
    vs.doc_uri = None
    vs.on_behalf_of_user = on_behalf_of_user
    # No explicit PAT / SP in YAML
    vs.pat = None
    vs.client_id = None
    vs.client_secret = None
    vs.workspace_host = None
    try:
        from conftest import add_databricks_resource_attrs

        add_databricks_resource_attrs(vs)
    except ImportError:
        pass
    # add_databricks_resource_attrs re-sets on_behalf_of_user=False; restore intent.
    vs.on_behalf_of_user = on_behalf_of_user
    return vs


def _fake_wc(
    *,
    auth_type: str,
    host: str = "https://fevm.example.com",
    client_id: Any = None,
    client_secret: Any = None,
    bearer: str | None = None,
) -> MagicMock:
    """Build a WorkspaceClient mock whose .config exposes the auth_type shape
    the databricks-langchain library and _client_args_from_ambient_wc read."""
    wc = MagicMock(name=f"wc[{auth_type}]")
    wc.config.auth_type = auth_type
    wc.config.host = host
    wc.config.client_id = client_id
    wc.config.client_secret = client_secret
    if bearer is not None:
        wc.config.authenticate.return_value = {"Authorization": f"Bearer {bearer}"}
    else:
        wc.config.authenticate.return_value = {}
    return wc


def _capture_client_args_and_invoke(
    vs_model: Mock,
    wc: MagicMock,
    *,
    monkeypatch: Any | None = None,
    query: str = "probe",
) -> dict[str, Any]:
    """Wire the vector_store mock to return the given fake WC, patch
    DatabricksVectorSearch to capture the kwargs it's constructed with, invoke
    the tool once, and return the captured kwargs dict.
    """
    from langgraph.runtime import Runtime

    from dao_ai.config import AiSearchRetrieverModel
    from dao_ai.state import Context
    from dao_ai.tools.vector_search import create_vector_search_tool

    vs_model.workspace_client_from.return_value = wc
    retriever = AiSearchRetrieverModel(vector_store=vs_model)

    captured: dict[str, Any] = {}
    with (
        patch("dao_ai.tools.vector_search.DatabricksVectorSearch") as MockDVS,
        patch("dao_ai.tools.vector_search._vsc_for_refresh", return_value=None),
        patch("dao_ai.tools.vector_search._fetch_index_columns", return_value=None),
    ):
        # Build-time refresh is stubbed out; these tests exercise the
        # query-time auth-mode behavior, not schema hydration.
        vs_model.refresh = MagicMock(return_value=None)

        def _capture(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            m = MagicMock()
            m.similarity_search.return_value = []
            return m

        MockDVS.side_effect = _capture

        tool = create_vector_search_tool(retriever=retriever)
        rt = Runtime(context=Context(headers={}))
        tool.invoke({"query": query, "filters": None, "runtime": rt})

    return captured


@pytest.mark.unit
def test_vs_mode1_ambient_app_sp_oauth_m2m(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mode #1: deployed Databricks App on Serverless v5.
    WorkspaceClient.config.auth_type == 'oauth-m2m' (owning SP). The library
    handles this branch natively, so dao-ai must pass client_args=None (or
    empty/falsy) — the library reads config.client_id / .client_secret from
    the WorkspaceClient we hand it.
    """
    for var in (
        "DATABRICKS_TOKEN",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_HOST",
    ):
        monkeypatch.delenv(var, raising=False)

    vs_model = _make_ambient_vector_store_mock()
    wc = _fake_wc(
        auth_type="oauth-m2m",
        client_id="sp-client-id",
        client_secret="sp-client-secret",
    )
    kwargs = _capture_client_args_and_invoke(vs_model, wc)

    assert kwargs["workspace_client"] is wc
    # None or empty are both acceptable here — both let the library run its
    # native oauth-m2m extraction. The important part is we did NOT override
    # with a bogus PAT from a bearer that this WC doesn't actually emit.
    assert kwargs.get("client_args") in (None, {}, {"disable_notice": True}), (
        f"mode #1 must not synthesize client_args for oauth-m2m; got {kwargs.get('client_args')!r}"
    )


@pytest.mark.unit
def test_vs_mode2_ambient_serverless_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mode #2: Serverless v5 notebook / job with no explicit PAT or SP.
    WorkspaceClient.config.auth_type is one of {databricks-cli, oauth-u2m,
    default}. The databricks-langchain library does NOT extract creds from
    these, so dao-ai must fill client_args from the runtime bearer.
    """
    for var in (
        "DATABRICKS_TOKEN",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_HOST",
    ):
        monkeypatch.delenv(var, raising=False)

    vs_model = _make_ambient_vector_store_mock()
    wc = _fake_wc(
        auth_type="databricks-cli",
        host="https://fevm.example.com",
        bearer="ambient-oauth-token",
    )
    kwargs = _capture_client_args_and_invoke(vs_model, wc)

    ca = kwargs.get("client_args") or {}
    assert ca.get("personal_access_token") == "ambient-oauth-token", (
        f"mode #2 must synthesize a PAT from the ambient bearer; got {ca!r}"
    )
    assert ca.get("workspace_url", "").startswith("https://fevm.example.com"), (
        f"mode #2 must include workspace_url; got {ca!r}"
    )


@pytest.mark.unit
def test_vs_mode2_other_ambient_auth_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mode #2 variants: oauth-u2m and 'default' behave the same as
    databricks-cli — dao-ai must synthesize client_args from the bearer.
    """
    for var in (
        "DATABRICKS_TOKEN",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_HOST",
    ):
        monkeypatch.delenv(var, raising=False)

    for auth_type in ("oauth-u2m", "default"):
        vs_model = _make_ambient_vector_store_mock()
        wc = _fake_wc(
            auth_type=auth_type,
            bearer=f"tok-{auth_type}",
        )
        kwargs = _capture_client_args_and_invoke(vs_model, wc)
        ca = kwargs.get("client_args") or {}
        assert ca.get("personal_access_token") == f"tok-{auth_type}", (
            f"auth_type={auth_type} did not produce a PAT: {ca!r}"
        )


@pytest.mark.unit
def test_vs_mode3_obo_passes_none_client_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mode #3: OBO with vector_store.on_behalf_of_user=True. dao-ai must pass
    client_args=None so the library uses the forwarded-bearer WorkspaceClient
    directly instead of building a separate VectorSearchClient.
    """
    for var in (
        "DATABRICKS_TOKEN",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_HOST",
    ):
        monkeypatch.delenv(var, raising=False)

    vs_model = _make_ambient_vector_store_mock(on_behalf_of_user=True)
    # OBO WC — auth_type shouldn't matter; the library uses the WC directly.
    wc = _fake_wc(auth_type="databricks-cli", bearer="user-forwarded-bearer")
    kwargs = _capture_client_args_and_invoke(vs_model, wc)

    assert kwargs.get("client_args") is None, (
        f"mode #3 (OBO) must pass client_args=None; got {kwargs.get('client_args')!r}"
    )
    assert kwargs["workspace_client"] is wc


@pytest.mark.unit
def test_vs_mode4_explicit_pat_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mode #4a: DATABRICKS_TOKEN env var set. dao-ai must populate
    client_args["personal_access_token"] from the env var, not synthesize
    from the ambient bearer.
    """
    monkeypatch.setenv("DATABRICKS_TOKEN", "explicit-pat-from-env")
    monkeypatch.setenv("DATABRICKS_HOST", "https://fevm.example.com")
    for var in ("DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)

    vs_model = _make_ambient_vector_store_mock()
    wc = _fake_wc(auth_type="databricks-cli", bearer="ambient-should-be-ignored")
    kwargs = _capture_client_args_and_invoke(vs_model, wc)

    ca = kwargs.get("client_args") or {}
    assert ca.get("personal_access_token") == "explicit-pat-from-env", (
        f"mode #4a must use env DATABRICKS_TOKEN, not ambient bearer; got {ca!r}"
    )


@pytest.mark.unit
def test_vs_mode4_explicit_sp_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mode #4b: DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET env vars set.
    dao-ai must populate client_args with the SP creds, not the ambient bearer.
    """
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "sp-id")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "sp-secret")
    monkeypatch.setenv("DATABRICKS_HOST", "https://fevm.example.com")
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    vs_model = _make_ambient_vector_store_mock()
    wc = _fake_wc(
        auth_type="oauth-m2m",
        client_id="wc-id",
        client_secret="wc-secret",
        bearer="ambient-should-be-ignored",
    )
    kwargs = _capture_client_args_and_invoke(vs_model, wc)

    ca = kwargs.get("client_args") or {}
    assert ca.get("service_principal_client_id") == "sp-id", (
        f"mode #4b must use DATABRICKS_CLIENT_ID env; got {ca!r}"
    )
    assert ca.get("service_principal_client_secret") == "sp-secret", (
        f"mode #4b must use DATABRICKS_CLIENT_SECRET env; got {ca!r}"
    )


# Direct unit tests of the helper — cheap and precise.


@pytest.mark.unit
def test_client_args_from_ambient_wc_returns_none_for_library_native() -> None:
    """The helper must decline to touch auth_types the library handles itself,
    so we don't accidentally override the library's built-in extraction.
    """
    from dao_ai.tools.vector_search import _client_args_from_ambient_wc

    for auth_type in ("pat", "oauth-m2m", "model_serving_user_credentials"):
        wc = _fake_wc(auth_type=auth_type, bearer="does-not-matter")
        assert _client_args_from_ambient_wc(wc) is None, (
            f"helper must return None for library-native auth_type={auth_type}"
        )


@pytest.mark.unit
def test_client_args_from_ambient_wc_returns_none_when_no_bearer() -> None:
    """If no bearer is resolvable, the helper returns None so the caller can
    pass client_args=None to the library — which will then raise its native
    error rather than dao-ai producing a broken PAT-shaped payload.
    """
    from dao_ai.tools.vector_search import _client_args_from_ambient_wc

    wc = _fake_wc(auth_type="databricks-cli")  # authenticate() -> {}
    assert _client_args_from_ambient_wc(wc) is None


# ---------------------------------------------------------------------------
# F1 — regression guard: workspace_url is mandatory when PAT is passed.
# VectorSearchClient.validate() (databricks/ai_search/client.py:186-189)
# raises when personal_access_token is supplied without workspace_url. If the
# helper synthesized a partial dict it would degrade the error UX, so we
# require it to return None whenever wc.config.host is missing.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_client_args_from_ambient_wc_returns_none_when_no_host() -> None:
    """host absent → helper returns None so the caller can pass client_args=None."""
    from dao_ai.tools.vector_search import _client_args_from_ambient_wc

    for host_val in (None, ""):
        wc = _fake_wc(auth_type="databricks-cli", host=host_val, bearer="tok")
        assert _client_args_from_ambient_wc(wc) is None, (
            f"helper must return None when host={host_val!r}"
        )


# ---------------------------------------------------------------------------
# F3-a — M3b Model Serving OBO. When on_behalf_of_user=True and no forwarded
# bearer is in Context.headers, workspace_client_from() falls through to
# self.workspace_client (config.py:439-445), which builds
# WorkspaceClient(credentials_strategy=ModelServingUserCredentials()). That
# WC's config.auth_type is "model_serving_user_credentials"; the library's
# native branch then adds credential_strategy=MODEL_SERVING_USER_CREDENTIALS
# to client_args and VectorSearchClient reads invoker creds from the MS env.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vs_mode3b_model_serving_obo(monkeypatch: pytest.MonkeyPatch) -> None:
    """OBO on Model Serving path: WC auth_type is model_serving_user_credentials,
    dao-ai must pass client_args=None so the library's native branch fires.
    """
    for var in (
        "DATABRICKS_TOKEN",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_HOST",
    ):
        monkeypatch.delenv(var, raising=False)

    vs_model = _make_ambient_vector_store_mock(on_behalf_of_user=True)
    wc = _fake_wc(auth_type="model_serving_user_credentials", bearer="ms-invoker-tok")
    kwargs = _capture_client_args_and_invoke(vs_model, wc)

    assert kwargs.get("client_args") is None, (
        "mode #3b (Model Serving OBO) must pass client_args=None so the library "
        "forwards to VectorSearchClient with "
        "credential_strategy=MODEL_SERVING_USER_CREDENTIALS; "
        f"got {kwargs.get('client_args')!r}"
    )
    assert kwargs["workspace_client"] is wc


# ---------------------------------------------------------------------------
# F3-b — defense in depth: a WorkspaceClient whose authenticate() returns a
# non-Bearer Authorization header (e.g. Basic) must NOT be turned into a
# spurious PAT. The helper punts (returns None), the library errors natively.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_client_args_from_ambient_wc_returns_none_for_basic_auth() -> None:
    from dao_ai.tools.vector_search import _client_args_from_ambient_wc

    wc = MagicMock(name="wc[basic]")
    wc.config.auth_type = "basic"
    wc.config.host = "https://ws.example.com"
    wc.config.authenticate.return_value = {"Authorization": "Basic dXNlcjpwYXNz"}
    assert _client_args_from_ambient_wc(wc) is None


# ---------------------------------------------------------------------------
# F5 — auth_type="default" edge. DefaultCredentials.auth_type() returns
# "default" (credentials_provider.py:1426), and ModelServingUserCredentials
# outside a Model Serving container ALSO returns "default"
# (credentials_provider.py:1503). Either case must produce a workable
# client_args from the ambient bearer.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_vs_default_auth_type_falls_back_to_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (
        "DATABRICKS_TOKEN",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
        "DATABRICKS_HOST",
    ):
        monkeypatch.delenv(var, raising=False)

    vs_model = _make_ambient_vector_store_mock()
    wc = _fake_wc(auth_type="default", bearer="fallback-bearer")
    kwargs = _capture_client_args_and_invoke(vs_model, wc)

    ca = kwargs.get("client_args") or {}
    assert ca.get("personal_access_token") == "fallback-bearer", (
        f"auth_type='default' must fall through to bearer extraction; got {ca!r}"
    )
    assert ca.get("workspace_url"), "workspace_url must be present alongside PAT"
