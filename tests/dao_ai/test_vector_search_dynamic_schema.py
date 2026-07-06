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
