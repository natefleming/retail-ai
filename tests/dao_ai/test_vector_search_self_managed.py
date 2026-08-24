"""Unit tests for self-managed / precomputed-embeddings support in the
``ai_search`` tool.

A managed-embeddings index embeds the query server-side, so
``DatabricksVectorSearch`` is built with ``text_column=None`` and no
``embedding``. A self-managed index (Direct Access, or Delta Sync with
``embedding_vector_columns``) does not embed queries server-side, so the
factory must pass the config-declared ``text_column`` plus an ``embedding``
built from ``embedding_model`` — the model that produced the stored vectors.

The embedding mode is auto-detected from the index ``describe()`` response the
factory already fetches via ``refresh()`` (``_index_details``); config only
supplies ``text_column`` + ``embedding_model``, which cannot be derived from
metadata.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from dao_ai.config import (
    AiSearchRetrieverModel,
    AiSearchVectorStoreModel,
    IndexModel,
    InferenceEndpointModel,
    SchemaModel,
    VectorSearchEndpoint,
    VectorStoreModel,
)
from dao_ai.tools.vector_search import (
    _index_is_managed_embeddings,
    _index_is_self_managed_embeddings,
    create_vector_search_tool,
)


def _managed_describe() -> dict[str, Any]:
    return {
        "name": "cat.sch.managed_index",
        "index_type": "DELTA_SYNC",
        "primary_key": "id",
        "delta_sync_index_spec": {
            "source_table": "cat.sch.docs",
            "embedding_source_columns": [
                {
                    "name": "content",
                    "embedding_model_endpoint_name": "databricks-gte-large-en",
                }
            ],
            "columns_to_sync": ["id", "content"],
        },
    }


def _self_managed_describe() -> dict[str, Any]:
    return {
        "name": "cat.sch.self_managed_index",
        "index_type": "DELTA_SYNC",
        "primary_key": "id",
        "delta_sync_index_spec": {
            "source_table": "cat.sch.docs",
            "embedding_vector_columns": [{"name": "content_embedding"}],
            "columns_to_sync": ["id", "content", "content_embedding"],
        },
    }


def _direct_access_describe() -> dict[str, Any]:
    return {
        "name": "cat.sch.direct_index",
        "index_type": "DIRECT_ACCESS",
        "primary_key": "id",
        "direct_access_index_spec": {
            "embedding_vector_columns": [{"name": "content_embedding"}],
        },
    }


def _make_vs(
    *,
    text_column: str | None = None,
    embedding_model: str | None = None,
) -> AiSearchVectorStoreModel:
    schema = SchemaModel(catalog_name="cat", schema_name="sch")
    return AiSearchVectorStoreModel(
        index=IndexModel(schema=schema, name="idx"),
        endpoint=VectorSearchEndpoint(name="ep"),
        columns=["id", "content"],
        text_column=text_column,
        embedding_model=(
            InferenceEndpointModel(name=embedding_model) if embedding_model else None
        ),
    )


def _build_tool_with_describe(
    vs: AiSearchVectorStoreModel,
    payload: dict[str, Any] | None,
    *,
    invoke: bool = False,
):
    """Create the tool with ``payload`` as the index describe() response.

    ``payload`` is stashed on ``_index_details`` so the factory's detection
    reads it and the real ``refresh()`` skips the provider call. ``payload is
    None`` simulates a describe() that never ran (permission denied): refresh
    is patched to raise, leaving ``_index_details`` None. When ``invoke`` is
    True the tool is invoked so the lazy ``DatabricksVectorSearch`` build runs.
    """
    ctx: list[Any] = [
        patch("dao_ai.tools.vector_search._vsc_for_refresh", return_value=None),
        patch("dao_ai.tools.vector_search._fetch_index_columns", return_value=None),
        patch(
            "dao_ai.config.InferenceEndpointModel.as_embeddings_model",
            return_value=MagicMock(spec=Embeddings, name="query_embedder"),
        ),
        patch("dao_ai.config.WorkspaceClient"),
        patch("dao_ai.tools.vector_search.DatabricksVectorSearch"),
    ]
    if payload is None:
        ctx.append(
            patch.object(
                VectorStoreModel,
                "refresh",
                autospec=True,
                side_effect=RuntimeError("describe() unavailable (permission denied)"),
            )
        )
    else:
        vs._index_details = payload

    retriever = AiSearchRetrieverModel(vector_store=vs)
    with ExitStack() as stack:
        entered = [stack.enter_context(cm) for cm in ctx]
        mock_dvs = entered[4]
        mock_dvs.return_value.similarity_search.return_value = []
        tool = create_vector_search_tool(retriever=retriever, name="t")
        if invoke:
            tool.invoke({"query": "hello", "filters": None})
    return tool, mock_dvs


@pytest.mark.unit
class TestManagedEmbeddingsDetection:
    def test_managed_delta_sync_is_managed(self) -> None:
        assert _index_is_managed_embeddings(_managed_describe()) is True

    def test_self_managed_delta_sync_is_not_managed(self) -> None:
        assert _index_is_managed_embeddings(_self_managed_describe()) is False

    def test_direct_access_is_not_managed(self) -> None:
        assert _index_is_managed_embeddings(_direct_access_describe()) is False

    def test_none_is_not_managed(self) -> None:
        assert _index_is_managed_embeddings(None) is False

    def test_non_dict_is_not_managed(self) -> None:
        assert _index_is_managed_embeddings(MagicMock()) is False

    def test_delta_sync_without_source_columns_is_not_managed(self) -> None:
        details = {"index_type": "DELTA_SYNC", "delta_sync_index_spec": {}}
        assert _index_is_managed_embeddings(details) is False


@pytest.mark.unit
class TestSelfManagedDetection:
    def test_managed_is_not_self_managed(self) -> None:
        assert _index_is_self_managed_embeddings(_managed_describe()) is False

    def test_self_managed_delta_sync_is_self_managed(self) -> None:
        assert _index_is_self_managed_embeddings(_self_managed_describe()) is True

    def test_direct_access_is_self_managed(self) -> None:
        assert _index_is_self_managed_embeddings(_direct_access_describe()) is True

    def test_none_is_not_self_managed(self) -> None:
        assert _index_is_self_managed_embeddings(None) is False

    def test_unknown_shape_is_neither(self) -> None:
        # An unrecognized shape is positively neither — the factory falls back
        # to config intent rather than hard-failing a managed index.
        details = {"index_type": "MYSTERY", "delta_sync_index_spec": {}}
        assert _index_is_managed_embeddings(details) is False
        assert _index_is_self_managed_embeddings(details) is False


@pytest.mark.unit
class TestSelfManagedQueryPath:
    def test_self_managed_passes_text_column_and_embedding(self) -> None:
        vs = _make_vs(text_column="content", embedding_model="databricks-bge-large-en")
        _, mock_dvs = _build_tool_with_describe(
            vs, _self_managed_describe(), invoke=True
        )
        kwargs = mock_dvs.call_args.kwargs
        assert kwargs["text_column"] == "content"
        assert kwargs["embedding"] is not None
        assert isinstance(kwargs["embedding"], Embeddings)

    def test_managed_passes_no_text_column_or_embedding(self) -> None:
        vs = _make_vs()
        _, mock_dvs = _build_tool_with_describe(vs, _managed_describe(), invoke=True)
        kwargs = mock_dvs.call_args.kwargs
        assert kwargs["text_column"] is None
        assert kwargs["embedding"] is None

    def test_describe_unavailable_with_text_column_uses_self_managed(self) -> None:
        vs = _make_vs(text_column="content", embedding_model="databricks-bge-large-en")
        _, mock_dvs = _build_tool_with_describe(vs, None, invoke=True)
        kwargs = mock_dvs.call_args.kwargs
        assert kwargs["text_column"] == "content"
        assert kwargs["embedding"] is not None


@pytest.mark.unit
class TestConfigErrors:
    def test_self_managed_without_text_column_raises(self) -> None:
        vs = _make_vs()  # no text_column, no embedding_model
        with pytest.raises(ValueError, match="self-managed"):
            _build_tool_with_describe(vs, _self_managed_describe())

    def test_managed_with_text_column_raises(self) -> None:
        vs = _make_vs(text_column="content", embedding_model="databricks-bge-large-en")
        with pytest.raises(ValueError, match="managed embeddings"):
            _build_tool_with_describe(vs, _managed_describe())


@pytest.mark.unit
class TestModelValidator:
    def test_text_column_without_embedding_model_rejected(self) -> None:
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        with pytest.raises(ValidationError, match="embedding_model"):
            AiSearchVectorStoreModel(
                index=IndexModel(schema=schema, name="idx"),
                text_column="content",
            )

    def test_text_column_with_embedding_model_accepted(self) -> None:
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        vs = AiSearchVectorStoreModel(
            index=IndexModel(schema=schema, name="idx"),
            text_column="content",
            embedding_model=InferenceEndpointModel(name="databricks-bge-large-en"),
        )
        assert vs.text_column == "content"
        assert vs.embedding_model is not None

    def test_embedding_model_alone_still_valid_for_provisioning(self) -> None:
        # embedding_model without text_column must remain valid (provisioning).
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        vs = AiSearchVectorStoreModel(
            index=IndexModel(schema=schema, name="idx"),
            embedding_model=InferenceEndpointModel(name="databricks-gte-large-en"),
        )
        assert vs.text_column is None


@pytest.mark.unit
class TestReviewFixes:
    def test_text_column_not_appended_to_return_columns(self) -> None:
        # #4: text_column must not be injected into the columns dao-ai builds
        # the filter enum from — the library merges it for page_content itself.
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        vs = AiSearchVectorStoreModel(
            index=IndexModel(schema=schema, name="idx"),
            columns=["id", "content"],  # text column 'body' intentionally absent
            text_column="body",
            embedding_model=InferenceEndpointModel(name="databricks-gte-large-en"),
        )
        _, mock_dvs = _build_tool_with_describe(
            vs, _self_managed_describe(), invoke=True
        )
        assert "body" not in mock_dvs.call_args.kwargs["columns"]

    def test_unknown_describe_shape_without_text_column_stays_managed(self) -> None:
        # #5: an unrecognized describe() shape must not hard-fail a managed
        # index; with no text_column it takes the managed path (text_column=None).
        vs = _make_vs()  # no text_column
        _, mock_dvs = _build_tool_with_describe(
            vs, {"index_type": "MYSTERY", "delta_sync_index_spec": {}}, invoke=True
        )
        kwargs = mock_dvs.call_args.kwargs
        assert kwargs["text_column"] is None
        assert kwargs["embedding"] is None

    def test_as_resources_includes_embedding_endpoint_for_self_managed(self) -> None:
        # #1: deployed principal needs a grant on the query-embedding endpoint.
        vs = _make_vs(text_column="body", embedding_model="my-custom-emb")
        kinds = [type(r).__name__ for r in vs.as_resources()]
        assert any("ServingEndpoint" in k for k in kinds)

    def test_as_resources_excludes_embedding_endpoint_for_managed(self) -> None:
        # Managed index (no text_column) embeds server-side — no extra grant.
        vs = _make_vs()  # no text_column, no embedding_model
        kinds = [type(r).__name__ for r in vs.as_resources()]
        assert not any("ServingEndpoint" in k for k in kinds)
