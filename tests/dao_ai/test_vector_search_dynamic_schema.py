"""Unit tests for dynamic Vector Search tool schema.

Covers the two behavior changes on ``create_vector_search_tool``:

  1. When YAML declares no ``columns`` the factory calls
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
    _probe_index_columns,
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
    """The factory calls VectorStoreModel.refresh() at build time when YAML
    is silent about ``columns``, and calls _fetch_column_types when the
    source table is populated."""

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

        # Patch:
        #  * ``_vsc_for_refresh`` — don't build a real VectorSearchClient
        #  * ``refresh`` — hydrate via the standard path with our canned
        #    describe payload (populates source_table, columns_to_sync)
        #  * ``_fetch_column_types`` — the UC Tables API call for types
        # Unit tests must not touch the network.
        original_refresh = VectorStoreModel.refresh
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._probe_index_columns", return_value=None
        ), patch.object(
            VectorStoreModel,
            "refresh",
            autospec=True,
            side_effect=lambda self, **kw: original_refresh(self, details=payload),
        ) as mocked_refresh, patch(
            "dao_ai.tools.vector_search._fetch_column_types",
            return_value=PRODUCTS_TYPES,
        ):
            retriever = RetrieverModel(vector_store=vs)
            tool = create_vector_search_tool(retriever=retriever, name="product_search")

        mocked_refresh.assert_called_once()
        # The tool now advertises the columns via its args_schema enum.
        enum = (
            tool.args_schema.model_json_schema()["$defs"]["DynamicFilterItem"][
                "properties"
            ]["key"]["enum"]
        )
        assert "product_name LIKE" in enum
        assert "price <=" in enum
        # And still rejects the regression key:
        assert "name NOT LIKE" not in enum

    def test_yaml_columns_take_precedence_no_refresh(self) -> None:
        vs = self._make_vs(with_columns=False)
        retriever = RetrieverModel(
            vector_store=vs,
            columns=["product_id", "product_name", "price"],
        )
        with patch.object(
            VectorStoreModel, "refresh", autospec=True
        ) as mocked_refresh, patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._probe_index_columns", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_column_types",
            return_value={
                "product_id": "bigint",
                "product_name": "string",
                "price": "double",
            },
        ):
            tool = create_vector_search_tool(retriever=retriever, name="product_search")

        mocked_refresh.assert_not_called()
        enum = (
            tool.args_schema.model_json_schema()["$defs"]["DynamicFilterItem"][
                "properties"
            ]["key"]["enum"]
        )
        # Only the three declared columns are in the enum:
        assert "product_id" in enum and "price <=" in enum
        assert "sku" not in enum and "category" not in enum

    def test_refresh_failure_soft_fallback(self) -> None:
        """If refresh raises we must NOT propagate — the tool still builds,
        with the pre-change free-form ``FilterItem`` schema. That's a
        deliberate demo-resilience choice."""
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
            "dao_ai.tools.vector_search._probe_index_columns", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._fetch_column_types", return_value=None
        ):
            tool = create_vector_search_tool(retriever=retriever, name="product_search")

        # No columns → args_schema is the module-level VectorSearchInput
        # (free-form key). Sanity: the schema does NOT carry an enum.
        schema = tool.args_schema.model_json_schema()
        filter_ref = schema["$defs"].get("FilterItem") or schema["$defs"].get(
            "DynamicFilterItem"
        )
        assert "enum" not in filter_ref["properties"]["key"]


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

    def _make_probe_setup(
        self, *, scan_fields: list[str], describe: dict | None
    ) -> tuple[Any, Any]:
        """Build a MagicMock chain that mimics vsc.get_index().scan() and
        stashes ``describe`` on the vector_store as ``_index_details``."""
        vs = MagicMock()
        vs.index.full_name = "cat.sch.probe_test_index"
        vs._index_details = describe
        idx = MagicMock()
        idx.scan.return_value = {
            "data": [{"fields": [{"key": k, "value": {}} for k in scan_fields]}]
        }
        vsc = MagicMock()
        vsc.get_index.return_value = idx
        return vs, vsc

    def test_strips_synthesised_vector_column_from_describe(self) -> None:
        vs, vsc = self._make_probe_setup(
            scan_fields=["product_id", "product_name", "description", "description_vector"],
            describe=_describe_payload(),
        )
        assert _probe_index_columns(vs, vsc) == ["product_id", "product_name", "description"]

    def test_preserves_business_column_ending_in_vector(self) -> None:
        """A user column literally named ``product_vector`` must survive
        — the describe payload's ``embedding_source_columns`` is the
        authoritative signal for what's a synthesised vector."""
        vs, vsc = self._make_probe_setup(
            scan_fields=[
                "product_id",
                "product_name",
                "product_vector",  # a business column, NOT the embedding
                "description",
                "description_vector",  # the actual synthesised vector
            ],
            describe=_describe_payload(),  # embedding source = "description"
        )
        cols = _probe_index_columns(vs, vsc)
        assert cols is not None
        assert "product_vector" in cols  # preserved!
        assert "description_vector" not in cols  # actually synthesised, stripped

    def test_falls_back_to_suffix_heuristic_when_describe_missing(self) -> None:
        # No describe details cached → we can't know which _vector column
        # is synthesised, so we strip all of them. This is a controlled
        # false-positive risk documented in the docstring.
        vs, vsc = self._make_probe_setup(
            scan_fields=["product_id", "product_name", "description_vector"],
            describe=None,
        )
        assert _probe_index_columns(vs, vsc) == ["product_id", "product_name"]

    def test_strips_cdf_underscore_prefix_fields(self) -> None:
        vs, vsc = self._make_probe_setup(
            scan_fields=["product_id", "_change_type", "_commit_version", "product_name"],
            describe=_describe_payload(),
        )
        assert _probe_index_columns(vs, vsc) == ["product_id", "product_name"]

    def test_empty_scan_result_returns_none(self) -> None:
        vs = MagicMock()
        vs.index.full_name = "cat.sch.empty_index"
        vs._index_details = None
        idx = MagicMock()
        idx.scan.return_value = {"data": []}
        vsc = MagicMock()
        vsc.get_index.return_value = idx
        assert _probe_index_columns(vs, vsc) is None

    def test_scan_permission_error_returns_none(self) -> None:
        vs = MagicMock()
        vs.index.full_name = "cat.sch.forbidden_index"
        vs._index_details = None
        idx = MagicMock()
        idx.scan.side_effect = PermissionError("forbidden")
        vsc = MagicMock()
        vsc.get_index.return_value = idx
        assert _probe_index_columns(vs, vsc) is None

    def test_null_vsc_returns_none(self) -> None:
        # Guard against the caller not being able to mint a
        # VectorSearchClient (e.g. no ambient auth available).
        assert _probe_index_columns(MagicMock(), None) is None


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
                "dao_ai.tools.vector_search._probe_index_columns", return_value=None
            ),
            patch.object(
                VectorStoreModel,
                "refresh",
                autospec=True,
                side_effect=lambda self, **kw: original_refresh(
                    self, details=payload
                ),
            ),
            patch(
                "dao_ai.tools.vector_search._fetch_column_types",
                return_value=PRODUCTS_TYPES,
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
        with self._patches()[0], self._patches()[1], self._patches()[2], self._patches()[3]:
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
    """Nothing about existing YAML-declared configurations should change.
    Regression guards for the shape the majority of dao-ai users are on.
    """

    def test_existing_yaml_shape_produces_narrowed_enum_with_types(self) -> None:
        # This mirrors the shape used in commerce_swarm.yaml — columns
        # declared, types not. The factory should still narrow to the
        # declared columns and (if source_table is set + types available)
        # be type-aware.
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
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._probe_index_columns", return_value=None
        ), patch.object(VectorStoreModel, "refresh", autospec=True) as mocked_refresh, patch(
            "dao_ai.tools.vector_search._fetch_column_types",
            return_value=PRODUCTS_TYPES,
        ):
            tool = create_vector_search_tool(
                retriever=retriever, name="product_search"
            )
        # YAML gave us columns → refresh should NOT have been called.
        mocked_refresh.assert_not_called()
        enum = tool.args_schema.model_json_schema()["$defs"][
            "DynamicFilterItem"
        ]["properties"]["key"]["enum"]
        # Regression key not present:
        assert "name NOT LIKE" not in enum
        # Type awareness applied:
        assert "price LIKE" not in enum
        assert "is_b2b_only <" not in enum

    def test_no_types_fetched_still_produces_working_tool(self) -> None:
        """If source_table isn't set (or UC Tables API returns nothing),
        the enum should exist but with all operators — the tool still
        prevents unknown columns, it just doesn't narrow per-type."""
        vs = self._bare_vs_no_source()
        retriever = RetrieverModel(vector_store=vs, columns=["a", "b"])
        with patch(
            "dao_ai.tools.vector_search._vsc_for_refresh", return_value=None
        ), patch(
            "dao_ai.tools.vector_search._probe_index_columns", return_value=None
        ), patch.object(VectorStoreModel, "refresh", autospec=True), patch(
            "dao_ai.tools.vector_search._fetch_column_types", return_value=None
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
