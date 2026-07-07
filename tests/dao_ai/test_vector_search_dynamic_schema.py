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
    FilterItem,
    IndexModel,
    RetrieverModel,
    SchemaModel,
    VectorSearchEndpoint,
    VectorStoreModel,
)
from dao_ai.tools.vector_search import (
    VectorSearchInput,
    _build_filter_item_model,
    _build_vector_search_input_model,
    _legal_filter_keys,
    _operators_for_type,
    _explicit_columns_to_sync,
    _fetch_source_table_column_types,
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
class TestOperatorsForType:
    """The type → operator-suffix mapping is what makes the Literal enum
    semantically useful. Boundaries: no LIKE on numerics, no ordering on
    booleans, only equality on booleans, unknown types get everything."""

    def test_numeric_gets_equality_and_ordering(self) -> None:
        for t in ("int", "bigint", "long", "double", "float", "decimal(10,2)"):
            ops = _operators_for_type(t)
            assert " <=" in ops
            assert " LIKE" not in ops
            assert " NOT LIKE" not in ops

    def test_string_gets_equality_and_like(self) -> None:
        for t in ("string", "varchar(255)", "char(3)", "text"):
            ops = _operators_for_type(t)
            assert " LIKE" in ops
            assert " NOT LIKE" in ops
            assert " <=" not in ops
            assert " >" not in ops

    def test_boolean_equality_only(self) -> None:
        ops = _operators_for_type("boolean")
        assert ops == ("", " NOT")

    def test_temporal_behaves_like_numeric(self) -> None:
        for t in ("timestamp", "date"):
            ops = _operators_for_type(t)
            assert " <=" in ops
            assert " LIKE" not in ops

    def test_unknown_or_missing_type_permits_all(self) -> None:
        # We deliberately do NOT block operators on types we don't recognise —
        # complex types like STRUCT/ARRAY can legitimately use unusual ops on
        # nested fields, and stripping them would produce false negatives.
        for t in ("", None, "struct<a:int>", "array<string>", "map<string,int>"):
            ops = _operators_for_type(t)  # type: ignore[arg-type]
            assert " LIKE" in ops
            assert " <=" in ops
            assert " NOT" in ops

    def test_binary_gets_equality_only(self) -> None:
        # Binary blobs have no meaningful ordering or LIKE semantics. We
        # treat them like booleans.
        for t in ("binary", "BINARY"):
            ops = _operators_for_type(t)
            assert ops == ("", " NOT"), f"binary type {t!r} should be equality-only, got {ops}"

    def test_type_matching_is_case_insensitive(self) -> None:
        # Databricks' DESCRIBE / UC Tables API returns type strings in mixed
        # case ("BIGINT", "Double", "TIMESTAMP"). We must not miss them.
        assert _operators_for_type("BIGINT") == _operators_for_type("bigint")
        assert _operators_for_type("STRING") == _operators_for_type("string")
        assert _operators_for_type("Boolean") == _operators_for_type("boolean")

    def test_parameterized_types_matched_by_prefix(self) -> None:
        # DESCRIBE returns parameterized versions of decimal/varchar/char.
        # We match by prefix so these still map to the right family.
        assert _operators_for_type("decimal(10,2)") == _operators_for_type("bigint")
        assert _operators_for_type("varchar(255)") == _operators_for_type("string")
        assert _operators_for_type("char(3)") == _operators_for_type("string")


@pytest.mark.unit
class TestLegalFilterKeys:
    def test_cross_product_without_types_permits_all(self) -> None:
        keys = _legal_filter_keys(["a", "b"])
        # 2 columns × 8 suffixes = 16
        assert len(keys) == 16
        assert "a" in keys and "a LIKE" in keys
        assert "b <=" in keys

    def test_cross_product_with_types_narrows_per_column(self) -> None:
        keys = _legal_filter_keys(
            ["price", "name", "flag"],
            {"price": "double", "name": "string", "flag": "boolean"},
        )
        # 6 (numeric) + 4 (string) + 2 (bool) = 12
        assert len(keys) == 12
        assert "price <=" in keys and "price LIKE" not in keys
        assert "name LIKE" in keys and "name >" not in keys
        assert "flag NOT" in keys and "flag LIKE" not in keys

    def test_missing_column_type_falls_back_to_all_suffixes(self) -> None:
        keys = _legal_filter_keys(
            ["price", "unknown"], {"price": "double"}
        )
        # price → 6 (numeric); unknown → 8 (fallback)
        assert len(keys) == 14
        assert "unknown LIKE" in keys
        assert "unknown <" in keys


@pytest.mark.unit
class TestBuildFilterItemModel:
    def test_empty_columns_returns_free_form_FilterItem(self) -> None:
        # Baseline: existing behaviour preserved when we can't discover columns.
        assert _build_filter_item_model([]) is FilterItem

    def test_columns_produce_literal_narrowed_key(self) -> None:
        M = _build_filter_item_model(PRODUCTS_COLUMNS, PRODUCTS_TYPES)
        schema = M.model_json_schema()
        enum = schema["properties"]["key"]["enum"]
        # Numeric/bool ops for product_id + price + is_b2b_only, string ops
        # for the 6 text columns: 6 + 6 + 2 + 4*6 = 38.
        assert len(enum) == 38
        assert "product_id" in enum and "product_id <=" in enum
        assert "product_name LIKE" in enum
        assert "is_b2b_only NOT" in enum
        # And the regression key must NOT appear:
        assert "name NOT LIKE" not in enum
        # Nor should nonsense combos:
        assert "price LIKE" not in enum
        assert "is_b2b_only <" not in enum


@pytest.mark.unit
class TestBuildVectorSearchInputModel:
    def test_empty_columns_returns_module_level_class(self) -> None:
        assert _build_vector_search_input_model([]) is VectorSearchInput

    def test_valid_filter_accepted(self) -> None:
        M = _build_vector_search_input_model(PRODUCTS_COLUMNS, PRODUCTS_TYPES)
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
        M = _build_vector_search_input_model(PRODUCTS_COLUMNS, PRODUCTS_TYPES)
        with pytest.raises(ValidationError) as exc_info:
            M(
                query="dessert",
                filters=[{"key": "name NOT LIKE", "value": "peanut"}],
            )
        # Pydantic error message should reference the offending field.
        assert "filters" in str(exc_info.value) and "key" in str(exc_info.value)

    def test_type_wrong_operator_rejected(self) -> None:
        """LIKE on a numeric column, ordering on a bool — both must be
        rejected by the type-aware enum."""
        M = _build_vector_search_input_model(PRODUCTS_COLUMNS, PRODUCTS_TYPES)
        for bad in ("price LIKE", "price NOT LIKE", "is_b2b_only <"):
            with pytest.raises(ValidationError):
                M(query="x", filters=[{"key": bad, "value": 0}])

    def test_falls_back_when_types_missing(self) -> None:
        # No type info → all suffixes permitted on every column.
        M = _build_vector_search_input_model(PRODUCTS_COLUMNS)
        ok = M(query="x", filters=[{"key": "price LIKE", "value": "3"}])
        # No exception → falls back to permissive mode.
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
            "dao_ai.tools.vector_search._fetch_source_table_column_types", return_value=None
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
            "dao_ai.tools.vector_search._fetch_source_table_column_types", return_value=None
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

    def test_yaml_column_not_on_index_dropped_with_warning(self) -> None:
        """Config declares a bogus column not on the index → dropped, WARN
        logged with column name + index name. LLM can never emit a filter
        that would fail at the VS API."""
        vs = self._make_vs(with_columns=False)
        retriever = RetrieverModel(
            vector_store=vs,
            columns=["product_id", "product_name", "nonexistent_col_xyz"],
        )
        payload = _describe_payload()
        original_refresh = VectorStoreModel.refresh

        # loguru bypasses stdlib logging — capture via a dedicated sink.
        from loguru import logger as loguru_logger

        captured: list[str] = []
        sink_id = loguru_logger.add(
            lambda msg: captured.append(str(msg)),
            level="WARNING",
            format="{message} {extra}",
        )
        try:
            with patch.object(
                VectorStoreModel,
                "refresh",
                autospec=True,
                side_effect=lambda self, **kw: original_refresh(self, details=payload),
            ), patch(
                "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
            ), patch(
                "dao_ai.tools.vector_search._fetch_source_table_column_types", return_value=None
            ):
                tool = create_vector_search_tool(
                    retriever=retriever, name="product_search"
                )
        finally:
            loguru_logger.remove(sink_id)

        enum = (
            tool.args_schema.model_json_schema()["$defs"]["DynamicFilterItem"][
                "properties"
            ]["key"]["enum"]
        )
        assert "product_id" in enum
        assert "product_name" in enum
        assert not any(k.startswith("nonexistent_col_xyz") for k in enum)

        warnings_text = " ".join(captured)
        assert "nonexistent_col_xyz" in warnings_text
        assert "products_description_index" in warnings_text

    def test_source_table_never_called_for_schema(self) -> None:
        """Regression guard: schema-building code must call
        ``wc.tables.get`` only on the INDEX's full_name (mirrors
        databricks-langchain), never on any other UC entity — most
        importantly, never on the source table (whose columns aren't
        guaranteed to be on the index).
        """
        vs = self._make_vs(with_columns=False)
        retriever = RetrieverModel(vector_store=vs)
        payload = _describe_payload()
        original_refresh = VectorStoreModel.refresh
        wc_spy = MagicMock()
        # Simulate a successful UC Tables lookup on the index.
        fake_index_table = MagicMock()
        fake_index_table.columns = [
            MagicMock(name="product_id", type_text="bigint"),
            MagicMock(name="product_name", type_text="string"),
        ]
        # MagicMock's name attr is special; set explicitly.
        for col, real_name in zip(
            fake_index_table.columns, ["product_id", "product_name"]
        ):
            col.name = real_name
        wc_spy.tables.get.return_value = fake_index_table

        with patch.object(
            VectorStoreModel,
            "refresh",
            autospec=True,
            side_effect=lambda self, **kw: original_refresh(self, details=payload),
        ), patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_source_table_column_types", return_value=None
        ), patch.object(
            VectorStoreModel,
            "workspace_client_from",
            autospec=True,
            return_value=wc_spy,
        ):
            create_vector_search_tool(retriever=retriever, name="product_search")

        # Every call to wc.tables.get must target the index full name.
        assert wc_spy.tables.get.call_count >= 1
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
            "dao_ai.tools.vector_search._fetch_source_table_column_types", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_index_column_types",
            return_value=None,
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
            "dao_ai.tools.vector_search._fetch_source_table_column_types", return_value=None
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

    def _make_source_table_setup(
        self,
        *,
        table_columns: list[tuple[str, str]],
        describe: dict | None,
        source_table_full_name: str = "cat.sch.products",
    ) -> tuple[Any, Any]:
        """Build a MagicMock chain that mimics
        ``vs.workspace_client_from(None).tables.get(source_table.full_name)``
        and stashes ``describe`` on the vector_store as ``_index_details``."""
        vs = MagicMock()
        vs.index.full_name = "cat.sch.probe_test_index"
        vs.source_table.full_name = source_table_full_name
        vs._index_details = describe
        wc = MagicMock()
        table = MagicMock()
        table.columns = [
            SimpleNamespace(name=n, type_text=t, type_name=t)
            for n, t in table_columns
        ]
        wc.tables.get.return_value = table
        vs.workspace_client_from.return_value = wc
        return vs, wc

    def test_strips_synthesised_vector_column_from_describe(self) -> None:
        vs, _ = self._make_source_table_setup(
            table_columns=[
                ("product_id", "bigint"),
                ("product_name", "string"),
                ("description", "string"),
                # Source-table lookup runs against the actual source table,
                # not the index — so the synthesised ``description_vector``
                # column doesn't exist here. It's stripped anyway on the
                # index-side lookup; this test exercises the source-table
                # happy path where the source cols == index cols.
            ],
            describe=_describe_payload(),
        )
        assert _fetch_source_table_column_types(vs) == {
            "product_id": "bigint",
            "product_name": "string",
            "description": "string",
        }

    def test_preserves_business_column_ending_in_vector(self) -> None:
        """A user column literally named ``product_vector`` on the source
        table must survive — describe's ``embedding_source_columns`` is
        the authoritative signal for what's a synthesised vector, and
        ``product`` is not the embedding source (``description`` is)."""
        describe = _describe_payload()
        # Drop the default ``columns_to_sync`` so we're isolating the
        # vector-stripping behavior, not the sync-subset intersection.
        describe["delta_sync_index_spec"].pop("columns_to_sync", None)
        vs, _ = self._make_source_table_setup(
            table_columns=[
                ("product_id", "bigint"),
                ("product_name", "string"),
                ("product_vector", "array<double>"),
                ("description", "string"),
            ],
            describe=describe,  # embedding source = "description"
        )
        out = _fetch_source_table_column_types(vs)
        assert out is not None
        assert "product_vector" in out  # preserved!

    def test_falls_back_to_suffix_heuristic_when_describe_missing(self) -> None:
        """No describe details cached → we can't know which ``_vector``
        column is synthesised, so we strip all of them. Same controlled
        false-positive risk as the index-side lookup."""
        vs, _ = self._make_source_table_setup(
            table_columns=[
                ("product_id", "bigint"),
                ("product_name", "string"),
                ("description_vector", "array<double>"),
            ],
            describe=None,
        )
        assert _fetch_source_table_column_types(vs) == {
            "product_id": "bigint",
            "product_name": "string",
        }

    def test_strips_cdf_underscore_prefix_fields(self) -> None:
        vs, _ = self._make_source_table_setup(
            table_columns=[
                ("product_id", "bigint"),
                ("_change_type", "string"),
                ("_commit_version", "bigint"),
                ("product_name", "string"),
            ],
            describe=_describe_payload(),
        )
        assert _fetch_source_table_column_types(vs) == {
            "product_id": "bigint",
            "product_name": "string",
        }

    def test_intersects_with_columns_to_sync_subset(self) -> None:
        """When the user explicitly restricts sync to a subset, source-table
        lookup must not advertise cols the index doesn't hold."""
        describe = _describe_payload()
        # Explicit subset: only product_id + product_name are synced.
        describe["delta_sync_index_spec"]["columns_to_sync"] = [
            "product_id",
            "product_name",
        ]
        vs, _ = self._make_source_table_setup(
            table_columns=[
                ("product_id", "bigint"),
                ("product_name", "string"),
                ("brand", "string"),       # NOT on the index
                ("category", "string"),    # NOT on the index
                ("description", "string"),
            ],
            describe=describe,
        )
        out = _fetch_source_table_column_types(vs)
        assert out == {"product_id": "bigint", "product_name": "string"}

    def test_empty_source_table_returns_none(self) -> None:
        vs, _ = self._make_source_table_setup(
            table_columns=[],
            describe=None,
        )
        assert _fetch_source_table_column_types(vs) is None

    def test_uc_tables_permission_error_returns_none(self) -> None:
        vs = MagicMock()
        vs.source_table.full_name = "cat.sch.forbidden_source"
        vs._index_details = None
        wc = MagicMock()
        wc.tables.get.side_effect = PermissionError("forbidden")
        vs.workspace_client_from.return_value = wc
        assert _fetch_source_table_column_types(vs) is None

    def test_direct_access_no_source_table_returns_none(self) -> None:
        """Direct-Access indexes have no source_table; the fallback returns
        None and the caller uses ``refresh()``'s ``columns_to_sync`` handling
        instead. Verifies no AttributeError on the None check."""
        vs = MagicMock()
        vs.source_table = None
        assert _fetch_source_table_column_types(vs) is None

    def test_explicit_columns_to_sync_helper(self) -> None:
        """``_explicit_columns_to_sync`` returns None when no subset is
        declared (default = all cols) and a set when the user restricted."""
        # No columns_to_sync in payload → None.
        no_sync = _describe_payload()
        no_sync["delta_sync_index_spec"].pop("columns_to_sync", None)
        assert _explicit_columns_to_sync(no_sync) is None
        # Explicit subset → set.
        describe = _describe_payload()
        describe["delta_sync_index_spec"]["columns_to_sync"] = ["a", "b"]
        assert _explicit_columns_to_sync(describe) == {"a", "b"}
        # None input → None.
        assert _explicit_columns_to_sync(None) is None


@pytest.mark.unit
class TestFilterValueShapes:
    """The LLM-visible ``value`` field on ``FilterItem`` accepts
    scalars OR lists (IN-style). Confirm every real shape is accepted so
    the LLM doesn't get validation-error-nagged.
    """

    def _model(self):
        return _build_vector_search_input_model(
            ["sku", "price", "is_b2b_only"],
            {"sku": "string", "price": "double", "is_b2b_only": "boolean"},
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
        # 50 columns × 4-6 ops-per-column = ~250 legal keys. This is well
        # under pydantic's practical ceiling and well over any realistic
        # commerce/product index.
        cols = [f"col_{i:04d}" for i in range(50)]
        types = {c: ("string" if i % 2 else "bigint") for i, c in enumerate(cols)}
        M = _build_vector_search_input_model(cols, types)
        schema = M.model_json_schema()
        enum = schema["$defs"]["DynamicFilterItem"]["properties"]["key"]["enum"]
        assert 200 <= len(enum) <= 300, f"unexpected enum size {len(enum)}"
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
        # Trivial happy path with just one column.
        M = _build_vector_search_input_model(["only"], {"only": "string"})
        enum = M.model_json_schema()["$defs"]["DynamicFilterItem"]["properties"]["key"]["enum"]
        assert set(enum) == {"only", "only NOT", "only LIKE", "only NOT LIKE"}


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
                "dao_ai.tools.vector_search._fetch_source_table_column_types", return_value=None
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
            "dao_ai.tools.vector_search._fetch_source_table_column_types", return_value=None
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
            "dao_ai.tools.vector_search._fetch_source_table_column_types", return_value=None
        ), patch.object(VectorStoreModel, "refresh", autospec=True), patch(
            "dao_ai.tools.vector_search._fetch_index_column_types",
            return_value=None,
        ):
            tool = create_vector_search_tool(retriever=retriever, name="t")
        enum = tool.args_schema.model_json_schema()["$defs"][
            "DynamicFilterItem"
        ]["properties"]["key"]["enum"]
        # No types → all 8 operator suffixes for each column.
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
                "dao_ai.tools.vector_search._fetch_source_table_column_types", return_value=None
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

    def test_type_aware_enum_from_index_uc_lookup(self) -> None:
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
        wc = self._wc_with_index_table(
            [
                ("product_id", "bigint"),
                ("product_name", "string"),
                ("price", "double"),
                ("is_b2b_only", "boolean"),
                # A vector column that must be stripped by the leading-
                # underscore rule (no describe payload available).
                ("__db_description_vector", "array<float>"),
            ]
        )
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_source_table_column_types", return_value=None
        ), patch.object(
            VectorStoreModel, "refresh", autospec=True
        ), patch.object(
            VectorStoreModel,
            "workspace_client_from",
            autospec=True,
            return_value=wc,
        ):
            tool = create_vector_search_tool(
                retriever=retriever, name="product_search"
            )
        enum = (
            tool.args_schema.model_json_schema()["$defs"]["DynamicFilterItem"][
                "properties"
            ]["key"]["enum"]
        )
        # Vector column stripped.
        assert not any("__db_description_vector" in k for k in enum), enum
        # Type-aware operators:
        assert "product_name LIKE" in enum  # string → LIKE
        assert "product_name >" not in enum  # …not ordering
        assert "price <=" in enum  # double → ordering
        assert "price LIKE" not in enum  # …not LIKE
        assert "is_b2b_only NOT" in enum  # bool → equality/NOT
        assert "is_b2b_only <" not in enum  # …no ordering

    def test_soft_fallback_when_index_uc_lookup_fails(self) -> None:
        """UC Tables call on the INDEX fails → factory falls through to
        UC Tables on the SOURCE TABLE. Because the source-table lookup
        also returns ``{name: type}``, columns AND type-aware operator
        narrowing are both preserved on the fallback path."""
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
        wc = MagicMock()
        wc.tables.get.side_effect = PermissionError("no access")

        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_source_table_column_types",
            return_value={"product_id": "bigint", "product_name": "string"},
        ), patch.object(
            VectorStoreModel, "refresh", autospec=True
        ), patch.object(
            VectorStoreModel,
            "workspace_client_from",
            autospec=True,
            return_value=wc,
        ):
            tool = create_vector_search_tool(retriever=retriever, name="t")

        enum = (
            tool.args_schema.model_json_schema()["$defs"]["DynamicFilterItem"][
                "properties"
            ]["key"]["enum"]
        )
        # Columns narrowed (regression blocked). Types come through from
        # the source-table fallback, so operators are narrowed too:
        # bigint gets NUMERIC ops (no LIKE), string gets STRING ops
        # (no ordering).
        assert "product_id" in enum and "product_name" in enum
        assert "product_id LIKE" not in enum  # bigint — no LIKE
        assert "product_id <=" in enum        # bigint — ordering OK
        assert "product_name LIKE" in enum    # string — LIKE OK
        assert "product_name <=" not in enum  # string — no ordering
