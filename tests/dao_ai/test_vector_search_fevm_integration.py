"""fevm integration tests for the vector-search tool auth-mode matrix.

Runs the real dao-ai VS tool factory against a real Databricks Vector Search
index on the fevm workspace. These tests are the regression watchdog for
the four auth modes documented in ``create_vector_search_tool``:

    #1  ambient App SP        (auth_type=oauth-m2m) — library-native path
    #2  ambient Serverless v5 (auth_type=databricks-cli / oauth-u2m / default)
                              — dao-ai fills client_args from runtime bearer
    #3  OBO (user auth)       (vector_store.on_behalf_of_user=True)
    #4  explicit PAT/SP       (DATABRICKS_TOKEN / DATABRICKS_CLIENT_ID env or YAML)

Local-laptop runs of these tests exercise **mode #2 exclusively** because
``WorkspaceClient(profile="fevm")`` picks up ``auth_type="databricks-cli"``
on ``~/.databrickscfg`` profiles. Modes #1 and #3 are only meaningful inside
a deployed Databricks App and can't be reproduced from a laptop; those are
covered by the mocked unit tests in ``test_vector_search.py``.

To run:
    pytest tests/dao_ai/test_vector_search_fevm_integration.py -v -m integration

Requires the ``fevm`` CLI profile in ``~/.databrickscfg`` OR
``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN`` env vars. Uses the healthy
phase2 replacement indexes provisioned during workshop validation:
    - retail_consumer_goods.dao_ai_phase2.products_index
    - retail_consumer_goods.dao_ai_phase2.kb_articles_index
Both live on VS endpoint ``dao_ai_workshop_vs``.
"""

from __future__ import annotations

import json
import os

import pytest
from databricks.sdk import WorkspaceClient
from langchain_core.messages import ToolCall as LCToolCall
from langchain_core.messages import ToolMessage

from dao_ai.config import (
    AiSearchRetrieverModel,
    IndexModel,
    SchemaModel,
    SearchParametersModel,
    VectorSearchEndpoint,
    VectorStoreModel,
)
from dao_ai.tools.vector_search import create_vector_search_tool

TEST_PROFILE = os.getenv("DAO_AI_TEST_PROFILE", "fevm")


def _has_fevm_profile() -> bool:
    """Cheap check that fevm profile is usable without actually calling the API."""
    if os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN"):
        return True
    cfg = os.path.expanduser("~/.databrickscfg")
    if not os.path.exists(cfg):
        return False
    try:
        with open(cfg) as f:
            return f"[{TEST_PROFILE}]" in f.read()
    except Exception:
        return False


SKIP_MSG = (
    f"Requires DATABRICKS_CONFIG_PROFILE={TEST_PROFILE} in ~/.databrickscfg "
    "or DATABRICKS_HOST + DATABRICKS_TOKEN env vars."
)


def _extract_documents(result):
    """Unwrap the tool result into a list. Copied from test_reranking_integration."""
    if isinstance(result, ToolMessage):
        content = result.content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                import ast

                return ast.literal_eval(content)
        return content
    return result


def _products_retriever() -> AiSearchRetrieverModel:
    schema = SchemaModel(
        catalog_name="retail_consumer_goods",
        schema_name="dao_ai_phase2",
    )
    return AiSearchRetrieverModel(
        vector_store=VectorStoreModel(
            index=IndexModel(name="products_index", schema=schema),
            endpoint=VectorSearchEndpoint(name="dao_ai_workshop_vs"),
            primary_key="sku",
            embedding_source_column="description",
            columns=["sku", "product_name", "category", "description"],
        ),
        search_parameters=SearchParametersModel(num_results=3),
    )


def _kb_retriever() -> AiSearchRetrieverModel:
    schema = SchemaModel(
        catalog_name="retail_consumer_goods",
        schema_name="dao_ai_phase2",
    )
    return AiSearchRetrieverModel(
        vector_store=VectorStoreModel(
            index=IndexModel(name="kb_articles_index", schema=schema),
            endpoint=VectorSearchEndpoint(name="dao_ai_workshop_vs"),
            primary_key="article_id",
            embedding_source_column="body",
            columns=["article_id", "title", "topic", "body"],
        ),
        search_parameters=SearchParametersModel(num_results=3),
    )


def _invoke(tool, query: str):
    tool_call = LCToolCall(
        name=tool.name,
        args={"query": query},
        id="test-tc-1",
        type="tool_call",
    )
    result = tool.invoke(tool_call)
    return _extract_documents(result)


@pytest.mark.integration
@pytest.mark.skipif(not _has_fevm_profile(), reason=SKIP_MSG)
class TestVectorSearchFevmAmbientMode2:
    """Mode #2: ambient auth via ``fevm`` profile (auth_type=databricks-cli).

    Without the ``_client_args_from_ambient_wc`` fix, every method below
    would raise ``InvalidInputException: Please specify either personal
    access token or service principal client ID and secret`` at
    ``create_vector_search_tool``-invocation time.
    """

    @pytest.fixture(autouse=True)
    def _clear_env_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure the ambient path is what actually gets exercised — clear
        any DATABRICKS_TOKEN / SP env vars that would flip us into mode #4.
        """
        for var in (
            "DATABRICKS_TOKEN",
            "DATABRICKS_CLIENT_ID",
            "DATABRICKS_CLIENT_SECRET",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", TEST_PROFILE)

    def test_workspace_client_auth_type_is_ambient(self) -> None:
        """Guardrail: confirm the profile really produces one of the
        ambient auth_types the fix targets. If this ever flips to 'pat'
        or 'oauth-m2m', these tests would silently exercise the wrong
        code path.
        """
        wc = WorkspaceClient(profile=TEST_PROFILE)
        assert wc.config.auth_type in ("databricks-cli", "oauth-u2m", "default"), (
            f"expected ambient auth_type, got {wc.config.auth_type!r}"
        )

    def test_products_index_search_returns_results(self) -> None:
        tool = create_vector_search_tool(
            retriever=_products_retriever(),
            name="products_search",
            description="Search products by description",
        )
        docs = _invoke(tool, "cordless drill")
        assert isinstance(docs, list) and len(docs) > 0, (
            f"products search returned no docs (or wrong shape): {docs!r}"
        )

    def test_kb_articles_index_search_returns_results(self) -> None:
        tool = create_vector_search_tool(
            retriever=_kb_retriever(),
            name="kb_search",
            description="Search support KB articles",
        )
        docs = _invoke(tool, "reset password")
        assert isinstance(docs, list) and len(docs) > 0, (
            f"kb search returned no docs (or wrong shape): {docs!r}"
        )


@pytest.mark.integration
@pytest.mark.skipif(not _has_fevm_profile(), reason=SKIP_MSG)
class TestVectorSearchFevmExplicitAuthMode4:
    """Mode #4: explicit PAT via ``DATABRICKS_TOKEN`` env var. Exercises the
    path that was working *before* the fix, so we don't regress it.
    """

    @pytest.fixture(autouse=True)
    def _mint_pat_from_ambient(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Pull an OAuth bearer from the ambient CLI profile and set it as
        DATABRICKS_TOKEN so the mode #4 code path runs. This mimics what a
        legacy notebook that hard-set the env var would look like.
        """
        wc = WorkspaceClient(profile=TEST_PROFILE)
        headers = wc.config.authenticate() or {}
        bearer = headers.get("Authorization", "")
        assert bearer.startswith("Bearer "), f"can't get bearer: {bearer!r}"
        token = bearer[len("Bearer ") :]
        monkeypatch.setenv("DATABRICKS_TOKEN", token)
        monkeypatch.setenv("DATABRICKS_HOST", wc.config.host)
        return token

    def test_products_index_search_with_explicit_pat(self) -> None:
        tool = create_vector_search_tool(
            retriever=_products_retriever(),
            name="products_search",
            description="Search products by description",
        )
        docs = _invoke(tool, "circular saw")
        assert isinstance(docs, list) and len(docs) > 0, (
            f"mode #4 explicit-PAT search returned no docs: {docs!r}"
        )


# ---------------------------------------------------------------------------
# Dynamic schema hydration — the LLM-facing args_schema is built from the
# live index metadata (columns via ``index.describe()``, types via UC Tables
# API) rather than YAML alone. The regression this closes is MLflow trace
# ``fc785d795b77675ac0e42fe5296b523a`` on ``agent-commerce-super-dao``:
# the LLM emitted ``{"key": "name NOT LIKE"}`` against
# ``retail_consumer_goods.commerce_swarm.products_description_index`` which
# has ``product_name`` (not ``name``), the vector search API rejected the
# call with ``Columns referenced in filters are not present in index: name``,
# and the whole trace bubbled to ``state=ERROR``.
# ---------------------------------------------------------------------------

COMMERCE_SCHEMA = SchemaModel(
    catalog_name="retail_consumer_goods",
    schema_name="commerce_swarm",
)

COMMERCE_VS_ENDPOINT = VectorSearchEndpoint(name="dbdemos_vs_endpoint")


def _bare_vector_store(index_name: str) -> VectorStoreModel:
    """VectorStoreModel with ONLY an index reference — no columns declared.

    Forces the factory down the auto-discovery path
    (``refresh()`` + ``_fetch_index_columns``).
    """
    return VectorStoreModel(
        index=IndexModel(schema=COMMERCE_SCHEMA, name=index_name),
        endpoint=COMMERCE_VS_ENDPOINT,
    )


def _tool_enum_keys(tool) -> list[str]:
    """Pull the ``filters[].key`` enum out of the tool's args_schema JSON."""
    schema = tool.args_schema.model_json_schema()
    defs = schema.get("$defs", {})
    item = defs.get("DynamicFilterItem") or defs.get("FilterItem") or {}
    return item.get("properties", {}).get("key", {}).get("enum", []) or []


@pytest.mark.integration
@pytest.mark.skipif(not _has_fevm_profile(), reason=SKIP_MSG)
class TestDynamicSchemaHydration:
    """Live-hydration tests against the three commerce_swarm indexes on FEVM:

      * ``products_description_index``  (10-col products table)
      * ``faqs_index``                  (FAQ knowledge base)
      * ``policies_index``              (policies knowledge base)

    Each test builds a tool from just an index reference (no ``columns:``
    in YAML) and asserts (a) the args_schema exposes the live column set
    with all 8 operator suffixes per column, (b) invalid keys are rejected
    by pydantic without hitting the retriever, (c) valid keys pass through
    and the live search returns.
    """

    def test_products_columns_discovered_from_live_index(self) -> None:
        vs = _bare_vector_store("products_description_index")
        tool = create_vector_search_tool(
            retriever=AiSearchRetrieverModel(vector_store=vs),
            name="product_search",
        )
        enum = _tool_enum_keys(tool)
        # Live products index columns (see commerce_swarm data DDL):
        # product_id, sku, product_name, brand, category, subcategory,
        # description, price, is_b2b_only.
        for c in (
            "product_id",
            "product_name",
            "brand",
            "category",
            "price",
            "is_b2b_only",
        ):
            assert c in enum, f"missing column {c!r} in {enum}"
        # Every column gets every operator suffix — no type-aware
        # narrowing. Databricks VS rejects invalid combinations at query
        # time (matches upstream databricks-langchain).
        assert "product_name LIKE" in enum
        assert "product_name >" in enum  # ordering allowed on any col
        assert "price <=" in enum
        assert "price LIKE" in enum  # LIKE allowed on any col
        assert "is_b2b_only NOT" in enum
        assert "is_b2b_only <" in enum  # ordering allowed on any col
        # Regression key still absent (LLM cannot emit ``name`` — real
        # column is ``product_name``).
        assert not any(k.startswith("name ") or k == "name" for k in enum), enum

    def test_regression_products_bad_key_rejected_before_vs_call(self) -> None:
        """The exact regression from trace fc785d795b... — must fail at
        pydantic validation, NOT surface to the retriever."""
        vs = _bare_vector_store("products_description_index")
        tool = create_vector_search_tool(
            retriever=AiSearchRetrieverModel(vector_store=vs),
            name="product_search",
        )
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            tool.invoke(
                {
                    "query": "dessert",
                    "filters": [{"key": "name NOT LIKE", "value": "peanut"}],
                }
            )

    def test_products_valid_filter_reaches_vs_and_returns(self) -> None:
        vs = _bare_vector_store("products_description_index")
        tool = create_vector_search_tool(
            retriever=AiSearchRetrieverModel(vector_store=vs),
            name="product_search",
        )
        tool_call = LCToolCall(
            name=tool.name,
            args={
                "query": "chocolate dessert",
                "filters": [
                    {"key": "product_name NOT LIKE", "value": "peanut"},
                    {"key": "price <=", "value": 50},
                    {"key": "is_b2b_only", "value": False},
                ],
            },
            id="dyn-schema-tc-1",
            type="tool_call",
        )
        docs = _extract_documents(tool.invoke(tool_call))
        assert isinstance(docs, list) and len(docs) > 0, (
            f"live products search with valid dyn-schema filters returned nothing: {docs!r}"
        )

    def test_faqs_columns_discovered_from_live_index(self) -> None:
        vs = _bare_vector_store("faqs_index")
        tool = create_vector_search_tool(
            retriever=AiSearchRetrieverModel(vector_store=vs),
            name="faq_search",
        )
        enum = _tool_enum_keys(tool)
        for c in ("faq_id", "category", "question", "answer"):
            assert c in enum, f"missing column {c!r} in {enum}"

    def test_policies_columns_discovered_from_live_index(self) -> None:
        vs = _bare_vector_store("policies_index")
        tool = create_vector_search_tool(
            retriever=AiSearchRetrieverModel(vector_store=vs),
            name="policy_search",
        )
        enum = _tool_enum_keys(tool)
        for c in ("policy_id", "category", "title", "body", "effective_date"):
            assert c in enum, f"missing column {c!r} in {enum}"

    def test_multiple_valid_filters_across_types(self) -> None:
        vs = _bare_vector_store("products_description_index")
        tool = create_vector_search_tool(
            retriever=AiSearchRetrieverModel(vector_store=vs),
            name="product_search",
        )
        tool_call = LCToolCall(
            name=tool.name,
            args={
                "query": "frozen bakery product",
                "filters": [
                    {"key": "category", "value": "Frozen Desserts"},
                    {"key": "price >", "value": 5},
                    {"key": "is_b2b_only", "value": False},
                    {"key": "description NOT LIKE", "value": "peanut"},
                ],
            },
            id="dyn-schema-tc-multi",
            type="tool_call",
        )
        docs = _extract_documents(tool.invoke(tool_call))
        assert isinstance(docs, list), f"unexpected shape: {docs!r}"
        # An empty result is OK — what we care about is that all four
        # different type-flavoured operators (equality/string, ordering/
        # numeric, equality/boolean, NOT LIKE/string) all validated and
        # reached the retriever without pydantic rejecting any of them.
