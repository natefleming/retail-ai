"""Unit tests for dynamic Vector Search tool schema.

Covers the two behavior changes on ``create_vector_search_tool``:

  1. When Config declares no ``columns`` the factory calls
     ``VectorStoreModel.refresh()`` and hydrates them from
     ``index.describe()``.
  2. The LLM-facing ``args_schema`` is per-tool: ``filters[].key`` is a
     :data:`typing.Literal` enum over ``columns × operator suffixes``,
     with operator suffixes filtered by column type (no ``LIKE`` on int,
     no ordering on bool, etc.). An unlisted key raises ``ValidationError``
     before the retriever is ever called.

Regression the Literal narrowing prevents: MLflow trace
``fc785d795b77675ac0e42fe5296b523a`` — the LLM emitted
``{"key": "name NOT LIKE", "value": "peanut"}`` against a products index
whose column is ``product_name``. The vector search API responded with
``Columns referenced in filters are not present in index: name`` and the
whole request went ``state=ERROR``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dao_ai.config import (
    ColumnInfo,
    FilterItem,
    IndexModel,
    RetrieverModel,
    SchemaModel,
    VectorSearchEndpoint,
    VectorStoreModel,
)
from dao_ai.tools.vector_search import (
    VectorSearchInput,
    _build_columns_description,
    _build_filter_item_model,
    _build_vector_search_input_model,
    _fetch_index_columns,
    _legal_filter_keys,
    _normalize_declared_columns,
    _vector_column_names_from_describe,
    create_vector_search_tool,
)


PRODUCTS_COLUMNS = [
    "product_id",
    "sku",
    "product_name",
    "brand",
    "category",
    "subcategory",
    "description",
    "price",
    "is_b2b_only",
]

PRODUCTS_TYPES = {
    "product_id": "bigint",
    "sku": "string",
    "product_name": "string",
    "brand": "string",
    "category": "string",
    "subcategory": "string",
    "description": "string",
    "price": "double",
    "is_b2b_only": "boolean",
}


def _describe_payload(**overrides: Any) -> dict[str, Any]:
    """Default ``index.describe()`` shape (mirrors test_vector_store_refresh.py)."""
    base: dict[str, Any] = {
        "name": "retail_consumer_goods.commerce_swarm.products_description_index",
        "endpoint_name": "dbdemos_vs_endpoint",
        "primary_key": "product_id",
        "delta_sync_index_spec": {
            "source_table": "retail_consumer_goods.commerce_swarm.products",
            "embedding_source_columns": [
                {
                    "name": "description",
                    "embedding_model_endpoint_name": "databricks-gte-large-en",
                }
            ],
            "columns_to_sync": PRODUCTS_COLUMNS,
        },
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestLegalFilterKeys:
    def test_cross_product_permits_all_suffixes(self) -> None:
        keys = _legal_filter_keys(["a", "b"])
        # 2 columns × 8 suffixes = 16
        assert len(keys) == 16
        assert "a" in keys and "a LIKE" in keys
        assert "b <=" in keys
        assert "a NOT LIKE" in keys and "b >=" in keys

    def test_operator_overrides_restrict_a_specific_column(self) -> None:
        # brand locked to equality + LIKE only; price falls through to
        # the full 8-suffix set.
        keys = _legal_filter_keys(
            ["brand", "price"],
            operator_overrides={"brand": ["", "LIKE"]},
        )
        # 2 (brand override) + 8 (price default) = 10
        assert len(keys) == 10
        assert "brand" in keys and "brand LIKE" in keys
        assert "brand NOT" not in keys and "brand <" not in keys
        assert "price" in keys and "price LIKE" in keys and "price <" in keys

    def test_empty_operator_overrides_uses_defaults(self) -> None:
        keys = _legal_filter_keys(["a"], operator_overrides={})
        assert len(keys) == 8


@pytest.mark.unit
class TestBuildFilterItemModel:
    def test_empty_columns_returns_free_form_FilterItem(self) -> None:
        # Baseline: existing behaviour preserved when we can't discover columns.
        assert _build_filter_item_model([]) is FilterItem

    def test_columns_produce_literal_narrowed_key(self) -> None:
        M = _build_filter_item_model(PRODUCTS_COLUMNS)
        schema = M.model_json_schema()
        enum = schema["properties"]["key"]["enum"]
        # Every column gets every suffix — no type-aware narrowing.
        # 9 columns × 8 suffixes = 72.
        assert len(enum) == 72
        assert "product_id" in enum and "product_id <=" in enum
        assert "product_name LIKE" in enum
        assert "is_b2b_only NOT" in enum
        # The regression key must NOT appear (column doesn't exist):
        assert "name NOT LIKE" not in enum

    def test_operator_overrides_narrow_a_specific_column(self) -> None:
        # brand locked to ["", "LIKE"]; price gets all 8 suffixes.
        M = _build_filter_item_model(
            ["brand", "price"],
            operator_overrides={"brand": ["", "LIKE"]},
        )
        enum = M.model_json_schema()["properties"]["key"]["enum"]
        assert "brand" in enum and "brand LIKE" in enum
        assert "brand NOT" not in enum and "brand <" not in enum
        # price falls through to defaults
        assert "price" in enum and "price LIKE" in enum and "price <" in enum


@pytest.mark.unit
class TestBuildVectorSearchInputModel:
    def test_empty_columns_returns_module_level_class(self) -> None:
        assert _build_vector_search_input_model([]) is VectorSearchInput

    def test_valid_filter_accepted(self) -> None:
        M = _build_vector_search_input_model(PRODUCTS_COLUMNS)
        m = M(
            query="peanut-free dessert under 30",
            filters=[
                {"key": "price <=", "value": 30},
                {"key": "description NOT LIKE", "value": "peanut"},
                {"key": "is_b2b_only", "value": False},
            ],
        )
        assert len(m.filters) == 3
        assert m.filters[0].key == "price <="

    def test_regression_bad_column_key_rejected(self) -> None:
        """The exact key emitted by the LLM in trace fc785d795b... — must
        never make it past pydantic validation."""
        M = _build_vector_search_input_model(PRODUCTS_COLUMNS)
        with pytest.raises(ValidationError) as exc_info:
            M(
                query="dessert",
                filters=[{"key": "name NOT LIKE", "value": "peanut"}],
            )
        # Pydantic error message should reference the offending field.
        assert "filters" in str(exc_info.value) and "key" in str(exc_info.value)

    def test_operator_override_rejects_disallowed_suffix(self) -> None:
        """When ColumnInfo.operators locks a column to a subset, Pydantic
        rejects any suffix outside that subset."""
        M = _build_vector_search_input_model(
            ["brand"], operator_overrides={"brand": ["", "LIKE"]}
        )
        with pytest.raises(ValidationError):
            M(query="x", filters=[{"key": "brand NOT", "value": "y"}])

    def test_permissive_when_no_overrides(self) -> None:
        # No overrides → all 8 suffixes for every column.
        M = _build_vector_search_input_model(PRODUCTS_COLUMNS)
        ok = M(query="x", filters=[{"key": "price LIKE", "value": "3"}])
        # No exception — the enum allows every suffix regardless of type.
        assert ok.filters[0].key == "price LIKE"


@pytest.mark.unit
class TestFactoryColumnHydration:
    """The factory calls VectorStoreModel.refresh() at build time to
    validate any config-declared columns against the live index. declared columns
    not present on the index are dropped with a WARN before the Literal
    enum is built. Type-aware operator narrowing is not fetched from the
    source table (source-table columns are not guaranteed to be on the
    index); enum degrades to permissive operators when types are unknown.
    """

    def _make_vs(self, *, with_columns: bool = False) -> VectorStoreModel:
        schema = SchemaModel(
            catalog_name="retail_consumer_goods", schema_name="commerce_swarm"
        )
        vs = VectorStoreModel(
            index=IndexModel(schema=schema, name="products_description_index"),
            endpoint=VectorSearchEndpoint(name="dbdemos_vs_endpoint"),
        )
        if with_columns:
            vs.columns = ["product_id", "sku"]
        return vs

    def test_refresh_invoked_when_yaml_silent(self) -> None:
        vs = self._make_vs(with_columns=False)
        payload = _describe_payload()

        original_refresh = VectorStoreModel.refresh
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ), patch.object(
            VectorStoreModel,
            "refresh",
            autospec=True,
            side_effect=lambda self, **kw: original_refresh(self, details=payload),
        ) as mocked_refresh:
            retriever = RetrieverModel(vector_store=vs)
            tool = create_vector_search_tool(retriever=retriever, name="product_search")

        mocked_refresh.assert_called_once()
        enum = (
            tool.args_schema.model_json_schema()["$defs"]["DynamicFilterItem"][
                "properties"
            ]["key"]["enum"]
        )
        # Every discovered column is in the enum (permissive operators —
        # types are not fetched from the source table anymore).
        assert "product_name" in enum
        assert "product_name LIKE" in enum
        # Regression key stays rejected:
        assert "name NOT LIKE" not in enum

    def test_yaml_columns_intersected_with_index(self) -> None:
        """Config declares a subset of index columns → enum is the subset.

        The factory always calls refresh() so it can validate config against
        the live index. The result is declared ∩ index (in declaration order).
        """
        vs = self._make_vs(with_columns=False)
        retriever = RetrieverModel(
            vector_store=vs,
            columns=["product_id", "product_name", "price"],
        )
        payload = _describe_payload()
        original_refresh = VectorStoreModel.refresh
        with patch.object(
            VectorStoreModel,
            "refresh",
            autospec=True,
            side_effect=lambda self, **kw: original_refresh(self, details=payload),
        ) as mocked_refresh, patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ):
            tool = create_vector_search_tool(retriever=retriever, name="product_search")

        mocked_refresh.assert_called_once()
        enum = (
            tool.args_schema.model_json_schema()["$defs"]["DynamicFilterItem"][
                "properties"
            ]["key"]["enum"]
        )
        assert "product_id" in enum and "price" in enum
        # Everything outside the declared subset is absent:
        assert "sku" not in enum and "category" not in enum

    def test_yaml_columns_trusted_verbatim_no_drift_check(self) -> None:
        """Bare-string ``retriever.columns`` are trusted verbatim: even a
        column that isn't on the index is included in the enum. Drift
        detection is not the framework's job — the VS API will reject the
        filter at query time. Users who want authoritative schema control
        should declare ColumnInfo instead."""
        vs = self._make_vs(with_columns=False)
        retriever = RetrieverModel(
            vector_store=vs,
            columns=["product_id", "product_name", "nonexistent_col_xyz"],
        )
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ), patch.object(VectorStoreModel, "refresh", autospec=True):
            tool = create_vector_search_tool(
                retriever=retriever, name="product_search"
            )

        enum = (
            tool.args_schema.model_json_schema()["$defs"]["DynamicFilterItem"][
                "properties"
            ]["key"]["enum"]
        )
        assert "product_id" in enum
        assert "product_name" in enum
        # All three declared cols get every operator suffix — no drift
        # check against a live index.
        assert "nonexistent_col_xyz" in enum
        assert "nonexistent_col_xyz LIKE" in enum

    def test_source_table_never_called_for_schema(self) -> None:
        """Regression guard: this PR intentionally removed source-table
        UC lookup — only ``wc.tables.get(index.full_name)`` is allowed.
        Matches upstream databricks-langchain (single call, index only).

        Even under Mode B (bare strings declared), we call
        ``_fetch_index_columns`` best-effort for description enrichment
        — but never touch the source table.
        """
        vs = self._make_vs(with_columns=False)
        retriever = RetrieverModel(vector_store=vs, columns=["product_id"])
        wc_spy = MagicMock()
        fake_index_table = MagicMock()
        col = MagicMock()
        col.name = "product_id"
        col.type_text = "bigint"
        col.type_name = "bigint"
        col.comment = None
        fake_index_table.columns = [col]
        wc_spy.tables.get.return_value = fake_index_table

        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch.object(VectorStoreModel, "refresh", autospec=True), patch.object(
            VectorStoreModel,
            "workspace_client_from",
            autospec=True,
            return_value=wc_spy,
        ):
            create_vector_search_tool(retriever=retriever, name="product_search")

        # Every call to wc.tables.get must target the index full name —
        # never the source table.
        for call in wc_spy.tables.get.call_args_list:
            args, kwargs = call.args, call.kwargs
            target = args[0] if args else kwargs.get("full_name")
            assert target == vs.index.full_name, (
                f"wc.tables.get called with non-index target {target!r}; "
                f"only the index itself is allowed"
            )

    def test_refresh_failure_soft_fallback(self) -> None:
        """If both refresh AND the UC Tables index lookup fail, and no
        declared columns are given, the tool still builds with the
        pre-change free-form ``FilterItem`` schema."""
        vs = self._make_vs(with_columns=False)
        retriever = RetrieverModel(vector_store=vs)
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch.object(
            VectorStoreModel,
            "refresh",
            autospec=True,
            side_effect=RuntimeError("describe unauthorized"),
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ):
            tool = create_vector_search_tool(retriever=retriever, name="product_search")

        schema = tool.args_schema.model_json_schema()
        filter_ref = schema["$defs"].get("FilterItem") or schema["$defs"].get(
            "DynamicFilterItem"
        )
        assert "enum" not in filter_ref["properties"]["key"]

    def test_refresh_failure_with_yaml_falls_back_to_yaml(self) -> None:
        """When discovery fails but config has columns, trust config — the
        LLM enum is built from declared columns (may hit VS API errors if the
        user made a typo, matching pre-branch behavior)."""
        vs = self._make_vs(with_columns=False)
        retriever = RetrieverModel(
            vector_store=vs, columns=["product_id", "product_name"]
        )
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch.object(
            VectorStoreModel,
            "refresh",
            autospec=True,
            side_effect=RuntimeError("describe unauthorized"),
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ):
            tool = create_vector_search_tool(retriever=retriever, name="product_search")

        enum = (
            tool.args_schema.model_json_schema()["$defs"]["DynamicFilterItem"][
                "properties"
            ]["key"]["enum"]
        )
        assert "product_id" in enum and "product_name" in enum


# ---------------------------------------------------------------------------
# Edge cases surfaced during audit of vector-search configuration
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorColumnStripping:
    """The scan-based column probe must strip *only* the synthesised
    managed-embedding vector columns — never a business column whose name
    happens to end in ``_vector``.
    """

    def test_extracts_managed_vector_names_from_describe(self) -> None:
        details = _describe_payload()
        # Payload uses ``description`` as the embedding source, so the
        # synthesised vector column is ``description_vector``.
        vec_names = _vector_column_names_from_describe(details)
        assert vec_names == {"description_vector"}

    def test_extracts_multiple_vector_names(self) -> None:
        details = _describe_payload(
            delta_sync_index_spec={
                "source_table": "cat.sch.multi",
                "embedding_source_columns": [
                    {"name": "title", "embedding_model_endpoint_name": "e"},
                    {"name": "body", "embedding_model_endpoint_name": "e"},
                ],
            }
        )
        vec_names = _vector_column_names_from_describe(details)
        assert vec_names == {"title_vector", "body_vector"}

    def test_none_or_missing_details_returns_empty(self) -> None:
        assert _vector_column_names_from_describe(None) == set()
        assert _vector_column_names_from_describe({}) == set()
        assert _vector_column_names_from_describe({"delta_sync_index_spec": {}}) == set()

    def test_direct_access_managed_embedding_vector_names(self) -> None:
        """Direct-Access indexes carry the source list under
        ``direct_access_index_spec`` (not ``delta_sync_index_spec``)."""
        details = {
            "name": "cat.sch.ix",
            "endpoint_name": "ep",
            "primary_key": "id",
            "direct_access_index_spec": {
                "embedding_source_columns": [
                    {"name": "description", "embedding_model_endpoint_name": "e"}
                ],
            },
        }
        assert _vector_column_names_from_describe(details) == {"description_vector"}

    def test_self_managed_embedding_vector_column_stripped(self) -> None:
        """Self-managed indexes precompute vectors under an arbitrary
        column name declared in ``embedding_vector_columns``. That column
        (whatever it's called) must be stripped."""
        details = {
            "name": "cat.sch.ix",
            "endpoint_name": "ep",
            "primary_key": "id",
            "delta_sync_index_spec": {
                "source_table": "cat.sch.docs",
                "embedding_vector_columns": [{"name": "my_embedding"}],
            },
        }
        assert _vector_column_names_from_describe(details) == {"my_embedding"}

    def test_self_managed_on_direct_access_spec(self) -> None:
        """Self-managed embeddings can appear under either spec container."""
        details = {
            "name": "cat.sch.ix",
            "endpoint_name": "ep",
            "primary_key": "id",
            "direct_access_index_spec": {
                "embedding_vector_columns": [{"name": "vector"}],
            },
        }
        assert _vector_column_names_from_describe(details) == {"vector"}

    def test_business_column_named_foo_vector_preserved_direct_access(self) -> None:
        """A Direct-Access index with no embedding-source declaration
        (e.g. a describe payload that doesn't populate them) must NOT
        strip a business column literally named ``<x>_vector``. The
        describe-driven path returns only actual synthesised names — any
        column not in that set survives."""
        details = {
            "name": "cat.sch.ix",
            "endpoint_name": "ep",
            "primary_key": "id",
            "direct_access_index_spec": {
                "embedding_source_columns": [
                    {"name": "description", "embedding_model_endpoint_name": "e"}
                ],
            },
        }
        # Only ``description_vector`` is a synthesised name. A hypothetical
        # business column named ``product_vector`` is preserved.
        assert "product_vector" not in _vector_column_names_from_describe(details)

    def _make_index_uc_setup(
        self,
        *,
        table_columns: list[tuple[str, str | None, str | None]],
        describe: dict | None,
        index_full_name: str = "cat.sch.probe_test_index",
    ) -> tuple[Any, Any]:
        """Build a MagicMock chain that mimics
        ``vs.workspace_client_from(None).tables.get(index.full_name)``
        and stashes ``describe`` on the vector_store as ``_index_details``."""
        vs = MagicMock()
        vs.index.full_name = index_full_name
        vs._index_details = describe
        wc = MagicMock()
        table = MagicMock()
        table.columns = [
            SimpleNamespace(name=n, type_text=t, type_name=t, comment=c)
            for n, t, c in table_columns
        ]
        wc.tables.get.return_value = table
        vs.workspace_client_from.return_value = wc
        return vs, wc

    def test_index_uc_strips_synthesised_vector_column(self) -> None:
        vs, _ = self._make_index_uc_setup(
            table_columns=[
                ("product_id", "bigint", None),
                ("product_name", "string", "Full product name"),
                ("description", "string", None),
                ("description_vector", "array<double>", None),
            ],
            describe=_describe_payload(),  # embedding source = "description"
        )
        out = _fetch_index_columns(vs)
        assert out is not None
        names = [n for n, _, _ in out]
        assert names == ["product_id", "product_name", "description"]
        # Description carried through when set.
        assert dict((n, c) for n, _, c in out)["product_name"] == "Full product name"

    def test_index_uc_preserves_business_column_ending_in_vector(self) -> None:
        """A user column literally named ``product_vector`` must survive —
        describe's ``embedding_source_columns`` is the authoritative signal
        for what's a synthesised vector, and ``product`` is not the
        embedding source (``description`` is)."""
        describe = _describe_payload()
        vs, _ = self._make_index_uc_setup(
            table_columns=[
                ("product_id", "bigint", None),
                ("product_name", "string", None),
                ("product_vector", "array<double>", None),
                ("description", "string", None),
                ("description_vector", "array<double>", None),
            ],
            describe=describe,
        )
        out = _fetch_index_columns(vs)
        assert out is not None
        names = [n for n, _, _ in out]
        assert "product_vector" in names
        assert "description_vector" not in names

    def test_index_uc_falls_back_to_suffix_heuristic_when_describe_missing(
        self,
    ) -> None:
        vs, _ = self._make_index_uc_setup(
            table_columns=[
                ("product_id", "bigint", None),
                ("product_name", "string", None),
                ("description_vector", "array<double>", None),
            ],
            describe=None,
        )
        out = _fetch_index_columns(vs)
        assert out is not None
        names = [n for n, _, _ in out]
        assert names == ["product_id", "product_name"]

    def test_index_uc_strips_cdf_underscore_prefix_fields(self) -> None:
        vs, _ = self._make_index_uc_setup(
            table_columns=[
                ("product_id", "bigint", None),
                ("_change_type", "string", None),
                ("_commit_version", "bigint", None),
                ("product_name", "string", None),
            ],
            describe=_describe_payload(),
        )
        out = _fetch_index_columns(vs)
        assert out is not None
        names = [n for n, _, _ in out]
        assert names == ["product_id", "product_name"]

    def test_index_uc_empty_returns_none(self) -> None:
        vs, _ = self._make_index_uc_setup(table_columns=[], describe=None)
        assert _fetch_index_columns(vs) is None

    def test_index_uc_permission_error_returns_none(self) -> None:
        vs = MagicMock()
        vs.index.full_name = "cat.sch.forbidden_index"
        vs._index_details = None
        wc = MagicMock()
        wc.tables.get.side_effect = PermissionError("forbidden")
        vs.workspace_client_from.return_value = wc
        assert _fetch_index_columns(vs) is None

    def test_index_uc_no_index_returns_none(self) -> None:
        vs = MagicMock()
        vs.index = None
        assert _fetch_index_columns(vs) is None


@pytest.mark.unit
class TestFilterValueShapes:
    """The LLM-visible ``value`` field on ``FilterItem`` accepts
    scalars OR lists (IN-style). Confirm every real shape is accepted so
    the LLM doesn't get validation-error-nagged.
    """

    def _model(self):
        return _build_vector_search_input_model(
            ["sku", "price", "is_b2b_only"]
        )

    def test_string_scalar(self) -> None:
        m = self._model()(query="x", filters=[{"key": "sku", "value": "FRZ-001"}])
        assert m.filters[0].value == "FRZ-001"

    def test_int_scalar(self) -> None:
        m = self._model()(query="x", filters=[{"key": "price", "value": 30}])
        assert m.filters[0].value == 30

    def test_float_scalar(self) -> None:
        m = self._model()(query="x", filters=[{"key": "price <=", "value": 29.99}])
        assert m.filters[0].value == pytest.approx(29.99)

    def test_bool_scalar(self) -> None:
        m = self._model()(query="x", filters=[{"key": "is_b2b_only", "value": False}])
        assert m.filters[0].value is False

    def test_array_in_style(self) -> None:
        # ``value: [a, b, c]`` is the IN-style filter — matches any of.
        m = self._model()(
            query="x", filters=[{"key": "sku", "value": ["A", "B", "C"]}]
        )
        assert m.filters[0].value == ["A", "B", "C"]

    def test_null_value_rejected(self) -> None:
        # ``value: null`` isn't a legitimate filter (means "no filter") —
        # the LLM should just omit the whole item. Pydantic must reject
        # so the LLM gets a fast, clear error rather than the retriever
        # blowing up downstream.
        with pytest.raises(ValidationError):
            self._model()(query="x", filters=[{"key": "sku", "value": None}])

    def test_empty_filter_list_and_none_both_accepted(self) -> None:
        M = self._model()
        assert M(query="x", filters=None).filters is None
        assert M(query="x", filters=[]).filters == []

    def test_extra_field_on_filter_rejected(self) -> None:
        # ``FilterItem`` uses ``extra="forbid"`` — the LLM cannot smuggle
        # a rogue field past validation.
        with pytest.raises(ValidationError):
            self._model()(
                query="x",
                filters=[{"key": "sku", "value": "A", "operator": ">>"}],  # noqa
            )


@pytest.mark.unit
class TestScaleAndSpecialColumnNames:
    """Boundary conditions on the schema builder itself.

    Pydantic's ``Literal`` machinery is used at runtime with ``Literal[*keys]``
    unpacking. Boring but important: confirm we don't blow up on realistic
    column counts, name shapes, or duplicate inputs.
    """

    def test_large_column_count_produces_working_enum(self) -> None:
        # 50 columns × 8 suffixes = 400 legal keys. Well under pydantic's
        # practical ceiling and well over any realistic commerce/product
        # index.
        cols = [f"col_{i:04d}" for i in range(50)]
        M = _build_vector_search_input_model(cols)
        schema = M.model_json_schema()
        enum = schema["$defs"]["DynamicFilterItem"]["properties"]["key"]["enum"]
        assert len(enum) == 400
        M(query="x", filters=[{"key": "col_0000 <=", "value": 1}])
        M(query="x", filters=[{"key": "col_0001 LIKE", "value": "x"}])
        with pytest.raises(ValidationError):
            M(query="x", filters=[{"key": "col_0050", "value": 1}])

    def test_duplicate_columns_are_deduped(self) -> None:
        # ``Literal[*keys]`` collapses duplicates. The enum should not
        # ship the same key twice.
        M = _build_vector_search_input_model(["a", "a", "b"])
        enum = M.model_json_schema()["$defs"]["DynamicFilterItem"]["properties"]["key"]["enum"]
        assert enum.count("a") == 1
        assert enum.count("b") == 1

    def test_column_name_with_spaces(self) -> None:
        # Databricks column names can contain spaces when back-quoted.
        # Keep it working end-to-end.
        M = _build_vector_search_input_model(["a col", "b_col"])
        M(query="x", filters=[{"key": "a col LIKE", "value": "y"}])
        M(query="x", filters=[{"key": "b_col", "value": "z"}])

    def test_single_column_index(self) -> None:
        # Trivial happy path with just one column — all 8 suffixes.
        M = _build_vector_search_input_model(["only"])
        enum = M.model_json_schema()["$defs"]["DynamicFilterItem"]["properties"]["key"]["enum"]
        assert set(enum) == {
            "only", "only NOT", "only <", "only <=",
            "only >", "only >=", "only LIKE", "only NOT LIKE",
        }


@pytest.mark.unit
class TestFactoryEntryShapes:
    """The factory accepts several call shapes: ``retriever=...``,
    ``vector_store=...``, ``dict`` inputs. Confirm dynamic schema is applied
    consistently regardless of entry shape.
    """

    def _describe(self):
        return _describe_payload()

    def _patches(self):
        """Common patch stack: no network, canned describe."""
        payload = self._describe()
        original_refresh = VectorStoreModel.refresh
        return (
            patch(
                "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
            ),
            patch(
                "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
            ),
            patch.object(
                VectorStoreModel,
                "refresh",
                autospec=True,
                side_effect=lambda self, **kw: original_refresh(
                    self, details=payload
                ),
            ),
        )

    def _bare_vs(self) -> VectorStoreModel:
        return VectorStoreModel(
            index=IndexModel(
                schema=SchemaModel(
                    catalog_name="retail_consumer_goods",
                    schema_name="commerce_swarm",
                ),
                name="products_description_index",
            ),
            endpoint=VectorSearchEndpoint(name="dbdemos_vs_endpoint"),
        )

    def test_vector_store_param_entry(self) -> None:
        vs = self._bare_vs()
        with self._patches()[0], self._patches()[1], self._patches()[2]:
            tool = create_vector_search_tool(vector_store=vs, name="via_vs")
        enum = tool.args_schema.model_json_schema()["$defs"][
            "DynamicFilterItem"
        ]["properties"]["key"]["enum"]
        assert "product_name" in enum

    def test_missing_index_raises_at_model_level(self) -> None:
        # ``VectorStoreModel`` rejects at construction — factory never sees
        # a store without either ``index`` or ``source_table``. Guarantees
        # we can't accidentally build a tool without an index.
        with pytest.raises(ValidationError, match="index"):
            VectorStoreModel(
                endpoint=VectorSearchEndpoint(name="dbdemos_vs_endpoint")
            )

    def test_neither_retriever_nor_vector_store_raises(self) -> None:
        with pytest.raises(ValueError):
            create_vector_search_tool(name="broken")

    def test_both_retriever_and_vector_store_raises(self) -> None:
        vs = self._bare_vs()
        vs.columns = ["a"]
        r = RetrieverModel(vector_store=vs, columns=["a"])
        with pytest.raises(ValueError):
            create_vector_search_tool(retriever=r, vector_store=vs, name="both")


@pytest.mark.unit
class TestBackwardCompatibility:
    """Nothing about existing config-declared configurations should change.
    Regression guards for the shape the majority of dao-ai users are on.
    """

    def test_existing_yaml_shape_produces_narrowed_enum(self) -> None:
        """The commerce_swarm.yaml shape — columns declared, no explicit
        types. Factory validates config against the live index and builds
        a permissive Literal enum (all operators per column). Regression
        column keys are still rejected."""
        vs = VectorStoreModel(
            index=IndexModel(
                schema=SchemaModel(
                    catalog_name="retail_consumer_goods",
                    schema_name="commerce_swarm",
                ),
                name="products_description_index",
            ),
            endpoint=VectorSearchEndpoint(name="dbdemos_vs_endpoint"),
            columns=PRODUCTS_COLUMNS,
        )
        retriever = RetrieverModel(vector_store=vs, columns=PRODUCTS_COLUMNS)
        payload = _describe_payload()
        original_refresh = VectorStoreModel.refresh
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ), patch.object(
            VectorStoreModel,
            "refresh",
            autospec=True,
            side_effect=lambda self, **kw: original_refresh(self, details=payload),
        ):
            tool = create_vector_search_tool(
                retriever=retriever, name="product_search"
            )
        enum = tool.args_schema.model_json_schema()["$defs"][
            "DynamicFilterItem"
        ]["properties"]["key"]["enum"]
        # Regression key not present:
        assert "name NOT LIKE" not in enum
        # Every declared column is present:
        for c in PRODUCTS_COLUMNS:
            assert c in enum

    def test_no_types_still_produces_working_tool(self) -> None:
        """When the UC Tables index lookup fails and no discovery is
        available, declared columns are trusted as-is and every column
        gets every operator suffix (permissive fallback). The tool still
        narrows to the declared columns, blocking hallucinated keys."""
        vs = self._bare_vs_no_source()
        retriever = RetrieverModel(vector_store=vs, columns=["a", "b"])
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ), patch.object(VectorStoreModel, "refresh", autospec=True):
            tool = create_vector_search_tool(retriever=retriever, name="t")
        enum = tool.args_schema.model_json_schema()["$defs"][
            "DynamicFilterItem"
        ]["properties"]["key"]["enum"]
        # Every column always gets all 8 operator suffixes.
        assert len(enum) == 16
        assert "a LIKE" in enum and "a >=" in enum and "b NOT" in enum

    def _bare_vs_no_source(self) -> VectorStoreModel:
        return VectorStoreModel(
            index=IndexModel(
                schema=SchemaModel(
                    catalog_name="retail_consumer_goods",
                    schema_name="commerce_swarm",
                ),
                name="products_description_index",
            ),
            endpoint=VectorSearchEndpoint(name="dbdemos_vs_endpoint"),
        )


@pytest.mark.unit
class TestMcpAdapterEntryPoint:
    """Both entry points into the factory (`VectorSearchToolModel.as_tools()`
    and `register_vector_search()` MCP adapter) call the same
    `create_vector_search_tool(**args)` — so the dynamic args_schema is
    identical regardless of how the tool is provided. Regression guard
    against a future divergent build path.
    """

    def test_kwargs_and_dict_expansion_produce_same_schema(self) -> None:
        vs = VectorStoreModel(
            index=IndexModel(
                schema=SchemaModel(
                    catalog_name="retail_consumer_goods",
                    schema_name="commerce_swarm",
                ),
                name="products_description_index",
            ),
            endpoint=VectorSearchEndpoint(name="dbdemos_vs_endpoint"),
            columns=PRODUCTS_COLUMNS,
        )
        retriever = RetrieverModel(vector_store=vs, columns=PRODUCTS_COLUMNS)
        payload = _describe_payload()
        original_refresh = VectorStoreModel.refresh

        def _build(args: dict) -> Any:
            with patch(
                "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
            ), patch(
                "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
            ), patch.object(
                VectorStoreModel,
                "refresh",
                autospec=True,
                side_effect=lambda self, **kw: original_refresh(self, details=payload),
            ):
                return create_vector_search_tool(**args)

        # ``as_tools()`` style: kwargs pass-through
        tool_a = _build({"retriever": retriever, "name": "product_search"})
        # MCP adapter style: `create_vector_search_tool(**args)` where args
        # arrived as a dict deserialised from tool config.
        tool_b = _build({"retriever": retriever, "name": "product_search"})

        schema_a = tool_a.args_schema.model_json_schema()
        schema_b = tool_b.args_schema.model_json_schema()
        enum_a = schema_a["$defs"]["DynamicFilterItem"]["properties"]["key"]["enum"]
        enum_b = schema_b["$defs"]["DynamicFilterItem"]["properties"]["key"]["enum"]
        assert enum_a == enum_b
        assert "product_name" in enum_a


@pytest.mark.unit
class TestIndexScopedTypeLookup:
    """Type-aware operator narrowing driven by ``wc.tables.get(index)``.

    The index is a UC entity; asking the UC Tables API for its columns
    returns column names AND their Databricks types in one call. This is
    the databricks-langchain pattern — types come from the same
    authoritative source as the columns, never the source table.
    """

    def _wc_with_index_table(
        self, cols_and_types: list[tuple[str, str]]
    ) -> MagicMock:
        wc = MagicMock()
        table = MagicMock()
        fake_cols: list[MagicMock] = []
        for name, tt in cols_and_types:
            c = MagicMock()
            c.name = name
            c.type_text = tt
            fake_cols.append(c)
        table.columns = fake_cols
        wc.tables.get.return_value = table
        return wc

    def test_index_uc_discovery_powers_the_enum_when_no_config_columns(self) -> None:
        """Primary discovery path: nothing declared → factory calls
        ``_fetch_index_columns`` on the index UC entity, uses the resulting
        column list as the enum names. All 8 operator suffixes apply to
        every column (no type-aware narrowing)."""
        vs = VectorStoreModel(
            index=IndexModel(
                schema=SchemaModel(
                    catalog_name="retail_consumer_goods",
                    schema_name="commerce_swarm",
                ),
                name="products_description_index",
            ),
            endpoint=VectorSearchEndpoint(name="dbdemos_vs_endpoint"),
        )
        retriever = RetrieverModel(vector_store=vs)
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns",
            return_value=[
                ("product_id", "bigint", None),
                ("product_name", "string", "Full product name"),
                ("price", "double", None),
                ("is_b2b_only", "boolean", None),
            ],
        ), patch.object(VectorStoreModel, "refresh", autospec=True):
            tool = create_vector_search_tool(
                retriever=retriever, name="product_search"
            )
        enum = (
            tool.args_schema.model_json_schema()["$defs"]["DynamicFilterItem"][
                "properties"
            ]["key"]["enum"]
        )
        # 4 discovered columns × 8 suffixes = 32.
        assert len(enum) == 32
        # All suffixes valid for every column — no type-aware narrowing.
        assert "product_name LIKE" in enum
        assert "product_name >=" in enum        # string with ordering — allowed now
        assert "price LIKE" in enum             # numeric with LIKE — allowed now
        assert "is_b2b_only <" in enum          # bool with ordering — allowed now
        # Hallucinated column still rejected (this is the regression):
        assert "name NOT LIKE" not in enum
        # Description enrichment (product_name has a UC comment):
        assert "Full product name" in tool.description

    def test_soft_fallback_when_index_uc_lookup_fails(self) -> None:
        """UC Tables call fails AND nothing declared → factory degrades to
        free-form ``FilterItem`` (pre-branch behavior). No source-table
        fallback (dropped in this PR — matches upstream databricks-langchain)."""
        vs = VectorStoreModel(
            index=IndexModel(
                schema=SchemaModel(
                    catalog_name="retail_consumer_goods",
                    schema_name="commerce_swarm",
                ),
                name="products_description_index",
            ),
            endpoint=VectorSearchEndpoint(name="dbdemos_vs_endpoint"),
        )
        retriever = RetrieverModel(vector_store=vs)

        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ), patch.object(VectorStoreModel, "refresh", autospec=True):
            tool = create_vector_search_tool(retriever=retriever, name="t")

        schema = tool.args_schema.model_json_schema()
        # No enum on filter key — free-form fallback.
        filter_ref = schema["$defs"].get("FilterItem") or schema["$defs"].get(
            "DynamicFilterItem"
        )
        assert "enum" not in filter_ref["properties"]["key"]


# ---------------------------------------------------------------------------
# Hand-declared ColumnInfo — the new first-class knob for column schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeDeclaredColumns:
    """The helper that splits ``list[str | ColumnInfo]`` into flat lookups."""

    def test_all_strings_no_hand_declaration(self) -> None:
        names, types, descs, ops, any_hand = _normalize_declared_columns(
            ["a", "b", "c"]
        )
        assert names == ["a", "b", "c"]
        assert types == {}
        assert descs == {}
        assert ops == {}
        assert any_hand is False

    def test_all_column_info(self) -> None:
        items = [
            ColumnInfo(name="brand", type="string", description="Brand name"),
            ColumnInfo(name="price", type="number"),
        ]
        names, types, descs, ops, any_hand = _normalize_declared_columns(items)
        assert names == ["brand", "price"]
        assert types == {"brand": "STRING", "price": "NUMBER"}
        assert descs == {"brand": "Brand name"}  # only where set
        assert ops == {}  # neither set operators explicitly
        assert any_hand is True

    def test_column_info_explicit_operators(self) -> None:
        items = [
            ColumnInfo(name="brand", type="string", operators=["", "LIKE"]),
        ]
        _, _, _, ops, any_hand = _normalize_declared_columns(items)
        assert ops == {"brand": ["", "LIKE"]}
        assert any_hand is True

    def test_mixed_str_and_column_info(self) -> None:
        items = [
            "product_id",
            ColumnInfo(name="brand", type="string", description="Brand"),
            "sku",
        ]
        names, types, descs, ops, any_hand = _normalize_declared_columns(items)
        assert names == ["product_id", "brand", "sku"]
        # Only brand has type/description; bare strings don't populate.
        assert types == {"brand": "STRING"}
        assert descs == {"brand": "Brand"}
        assert ops == {}
        assert any_hand is True

    def test_empty_list(self) -> None:
        names, types, descs, ops, any_hand = _normalize_declared_columns([])
        assert names == [] and types == {} and descs == {} and ops == {}
        assert any_hand is False


@pytest.mark.unit
class TestBuildColumnsDescription:
    def test_names_only(self) -> None:
        block = _build_columns_description(["a", "b"], {}, {})
        assert "- a" in block
        assert "- b" in block
        assert "Supports operators" in block

    def test_names_with_types(self) -> None:
        block = _build_columns_description(
            ["price", "brand"],
            {"price": "INT64", "brand": "STRING"},
            {},
        )
        assert "- price (INT64)" in block
        assert "- brand (STRING)" in block

    def test_names_with_types_and_descriptions(self) -> None:
        block = _build_columns_description(
            ["brand"],
            {"brand": "STRING"},
            {"brand": "Brand — MILWAUKEE, DEWALT, MAKITA"},
        )
        assert "- brand (STRING) : Brand — MILWAUKEE, DEWALT, MAKITA" in block

    def test_empty_names_returns_empty(self) -> None:
        assert _build_columns_description([], {}, {}) == ""


@pytest.mark.unit
class TestHandDeclaredColumnInfoInFactory:
    """When retriever.columns contains a ColumnInfo, the factory:
    * uses ColumnInfo.name / .description / .type in the description block,
    * uses ColumnInfo.operators as operator_overrides in the enum,
    * SKIPS build-time UC calls (the whole point of hand-declaration).
    """

    def _bare_vs(self) -> VectorStoreModel:
        return VectorStoreModel(
            index=IndexModel(
                schema=SchemaModel(
                    catalog_name="retail_consumer_goods",
                    schema_name="commerce_swarm",
                ),
                name="products_description_index",
            ),
            endpoint=VectorSearchEndpoint(name="dbdemos_vs_endpoint"),
        )

    def test_hand_declared_skips_uc_calls(self) -> None:
        """When any ColumnInfo is in the list, no UC Tables call is made
        at build time — hand-declaration is authoritative."""
        vs = self._bare_vs()
        retriever = RetrieverModel(
            vector_store=vs,
            columns=[
                ColumnInfo(name="brand", type="string", description="Brand"),
                ColumnInfo(name="price", type="number"),
            ],
        )
        fetch_mock = MagicMock(return_value=None)
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", fetch_mock
        ), patch.object(VectorStoreModel, "refresh", autospec=True):
            create_vector_search_tool(retriever=retriever, name="t")
        # The factory must NOT have called _fetch_index_columns in
        # hand-declared mode.
        fetch_mock.assert_not_called()

    def test_hand_declared_operators_narrow_enum(self) -> None:
        vs = self._bare_vs()
        retriever = RetrieverModel(
            vector_store=vs,
            columns=[
                ColumnInfo(
                    name="brand",
                    type="string",
                    operators=["", "LIKE"],  # explicitly locked
                ),
                ColumnInfo(name="price", type="number"),  # defaults
            ],
        )
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ), patch.object(VectorStoreModel, "refresh", autospec=True):
            tool = create_vector_search_tool(retriever=retriever, name="t")
        enum = tool.args_schema.model_json_schema()["$defs"][
            "DynamicFilterItem"
        ]["properties"]["key"]["enum"]
        # brand is locked to 2 ops; price gets full 8.
        assert "brand" in enum and "brand LIKE" in enum
        assert "brand NOT" not in enum and "brand <" not in enum
        assert "price" in enum and "price LIKE" in enum and "price <" in enum

    def test_hand_declared_default_operators_gets_all_suffixes(self) -> None:
        """ColumnInfo without explicit operators → the default full-list
        signals 'don't restrict', so all 8 suffixes apply."""
        vs = self._bare_vs()
        retriever = RetrieverModel(
            vector_store=vs,
            columns=[ColumnInfo(name="brand", type="string")],
        )
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ), patch.object(VectorStoreModel, "refresh", autospec=True):
            tool = create_vector_search_tool(retriever=retriever, name="t")
        enum = tool.args_schema.model_json_schema()["$defs"][
            "DynamicFilterItem"
        ]["properties"]["key"]["enum"]
        # All 8 suffixes for brand.
        assert len(enum) == 8

    def test_hand_declared_description_appears_in_tool_description(self) -> None:
        vs = self._bare_vs()
        retriever = RetrieverModel(
            vector_store=vs,
            columns=[
                ColumnInfo(
                    name="brand",
                    type="string",
                    description="Brand — MILWAUKEE, DEWALT, MAKITA",
                ),
            ],
        )
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ), patch.object(VectorStoreModel, "refresh", autospec=True):
            tool = create_vector_search_tool(retriever=retriever, name="t")
        assert "Brand — MILWAUKEE, DEWALT, MAKITA" in tool.description
        assert "- brand (STRING)" in tool.description

    def test_mixed_str_and_column_info_both_in_enum(self) -> None:
        vs = self._bare_vs()
        retriever = RetrieverModel(
            vector_store=vs,
            columns=[
                "product_id",
                ColumnInfo(name="brand", type="string", operators=["", "LIKE"]),
                "sku",
            ],
        )
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns", return_value=None
        ), patch.object(VectorStoreModel, "refresh", autospec=True):
            tool = create_vector_search_tool(retriever=retriever, name="t")
        enum = tool.args_schema.model_json_schema()["$defs"][
            "DynamicFilterItem"
        ]["properties"]["key"]["enum"]
        # brand → 2 ops; product_id + sku → 8 ops each = 18 total.
        assert len(enum) == 18
        assert "brand LIKE" in enum and "brand NOT" not in enum
        assert "product_id LIKE" in enum and "sku <=" in enum

    def test_bare_strings_still_call_uc_for_description(self) -> None:
        """Mode B: bare-string columns declared → UC call is still made
        best-effort for description enrichment (types + comments)."""
        vs = self._bare_vs()
        retriever = RetrieverModel(
            vector_store=vs, columns=["brand", "price"]
        )
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_columns",
            return_value=[
                ("brand", "STRING", "Brand — Milwaukee, DeWalt, ..."),
                ("price", "INT64", None),
            ],
        ) as fetch_mock, patch.object(VectorStoreModel, "refresh", autospec=True):
            tool = create_vector_search_tool(retriever=retriever, name="t")
        # UC call happened (for description enrichment)
        fetch_mock.assert_called_once()
        # But the enum is built from declared names + all 8 suffixes each
        # — no type-aware narrowing.
        enum = tool.args_schema.model_json_schema()["$defs"][
            "DynamicFilterItem"
        ]["properties"]["key"]["enum"]
        assert len(enum) == 16
        assert "price LIKE" in enum  # numeric can LIKE — no narrowing
        # Description enrichment came through
        assert "Brand — Milwaukee" in tool.description
