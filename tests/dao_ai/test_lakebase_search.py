"""Unit tests for the lakebase_search first-class tool + LakebaseRetriever.

These tests don't require a live Postgres/Lakebase. They cover:

- Config: Pydantic discriminator dispatch, embedding-model string coercion,
  BM25/HYBRID guard against missing tsvector, mutual-exclusivity guard.
- Retriever SQL builders: correct operator selection, filter translation
  (allowlist, IN / IS NULL / comparisons), unknown-column rejection.
- Retriever RRF: deterministic fusion of two ranked lists.
- Retriever end-to-end: ANN and BM25 paths against a mocked psycopg pool.
- Tool factory: mutual exclusivity, dict-to-model coercion, name/description.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dao_ai.config import (
    ColumnInfo,
    FilterItem,
    FunctionType,
    InferenceEndpointModel,
    LakebaseRetrieverModel,
    LakebaseSearchToolModel,
    LakebaseVectorStoreModel,
    SearchParametersModel,
    ToolModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _vs_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "database": {"project": "test-lakebase"},
        "table": "kb_articles",
        "content_column": "passage",
        "embedding_column": "embedding",
        "tsvector_column": "passage_tsv",
        "embedding_model": "databricks-gte-large-en",
        "metadata_columns": ["category", "source_url"],
    }
    base.update(overrides)
    return base


@pytest.fixture()
def vector_store() -> LakebaseVectorStoreModel:
    return LakebaseVectorStoreModel(**_vs_dict())


@pytest.fixture()
def retriever_model(
    vector_store: LakebaseVectorStoreModel,
) -> LakebaseRetrieverModel:
    return LakebaseRetrieverModel(
        vector_store=vector_store,
        search_parameters=SearchParametersModel(query_type="ANN", num_results=5),
    )


# ---------------------------------------------------------------------------
# Config-layer tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_embedding_model_string_coerced(self) -> None:
        vs = LakebaseVectorStoreModel(**_vs_dict())
        assert isinstance(vs.embedding_model, InferenceEndpointModel)
        assert vs.embedding_model.name == "databricks-gte-large-en"

    def test_embedding_model_object_passthrough(self) -> None:
        vs = LakebaseVectorStoreModel(
            **_vs_dict(
                embedding_model=InferenceEndpointModel(name="my-endpoint"),
            )
        )
        assert vs.embedding_model.name == "my-endpoint"

    def test_bm25_index_name_auto_derived(self) -> None:
        vs = LakebaseVectorStoreModel(**_vs_dict())
        assert vs.bm25_index_name == "public.kb_articles_passage_tsv_bm25"

    def test_bm25_index_name_explicit_respected(self) -> None:
        vs = LakebaseVectorStoreModel(**_vs_dict(bm25_index_name="my.custom_bm25"))
        assert vs.bm25_index_name == "my.custom_bm25"

    def test_discriminator_dispatch(self) -> None:
        m = ToolModel.model_validate(
            {
                "name": "kb",
                "function": {"type": "lakebase_search", "vector_store": _vs_dict()},
            }
        )
        assert isinstance(m.function, LakebaseSearchToolModel)
        assert m.function.type == FunctionType.LAKEBASE_SEARCH.value

    def test_mutual_exclusivity(self) -> None:
        with pytest.raises(ValidationError):
            LakebaseSearchToolModel.model_validate(
                {
                    "type": "lakebase_search",
                    "vector_store": _vs_dict(),
                    "retriever": {"vector_store": _vs_dict()},
                }
            )

    def test_at_least_one_required(self) -> None:
        with pytest.raises(ValidationError):
            LakebaseSearchToolModel.model_validate({"type": "lakebase_search"})

    @pytest.mark.parametrize("mode", ["BM25", "HYBRID"])
    def test_bm25_hybrid_require_tsvector(self, mode: str) -> None:
        vs = LakebaseVectorStoreModel(
            **_vs_dict(tsvector_column=None, bm25_index_name=None)
        )
        with pytest.raises(ValidationError):
            LakebaseRetrieverModel(
                vector_store=vs,
                search_parameters=SearchParametersModel(query_type=mode),
            )


# ---------------------------------------------------------------------------
# Retriever SQL builder tests
# ---------------------------------------------------------------------------


class TestSqlBuilders:
    def _make(
        self,
        vs: LakebaseVectorStoreModel,
        query_type: str = "ANN",
    ):
        from dao_ai.retrievers.lakebase import LakebaseRetriever

        return LakebaseRetriever(
            vector_store=vs,
            search_parameters=SearchParametersModel(
                query_type=query_type, num_results=5
            ),
        )

    @pytest.mark.parametrize(
        "metric,expected_op",
        [("cosine", "<=>"), ("l2", "<->"), ("ip", "<#>")],
    )
    def test_ann_sql_uses_correct_operator(
        self, metric: str, expected_op: str
    ) -> None:
        vs = LakebaseVectorStoreModel(**_vs_dict(distance_metric=metric))
        r = self._make(vs)
        stmt, params = r._build_ann_sql({}, k=5)
        rendered = stmt.as_string(None)
        # Distance op appears in SELECT and ORDER BY.
        assert rendered.count(expected_op) >= 2
        assert '"embedding"' in rendered
        assert '"kb_articles"' in rendered
        assert params == [5]  # only LIMIT param (vector injected at execute time)

    def test_bm25_sql_composes_to_bm25query(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store, query_type="BM25")
        stmt, params = r._build_bm25_sql({}, k=5)
        rendered = stmt.as_string(None)
        assert "to_bm25query" in rendered
        assert "<@>" in rendered
        assert '"passage_tsv"' in rendered
        assert params == [5]

    # ------------------------------------------------------------------
    # Suffix-based WHERE clause tests — ai_search operator convention
    # ------------------------------------------------------------------

    def test_parse_filter_key_no_suffix(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        assert r._parse_filter_key("category") == ("category", "")

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("category NOT", ("category", " NOT")),
            ("priority <", ("priority", " <")),
            ("priority <=", ("priority", " <=")),
            ("priority >", ("priority", " >")),
            ("priority >=", ("priority", " >=")),
            ("source_url LIKE", ("source_url", " LIKE")),
            ("source_url NOT LIKE", ("source_url", " NOT LIKE")),
        ],
    )
    def test_parse_filter_key_suffixes(
        self,
        vector_store: LakebaseVectorStoreModel,
        key: str,
        expected: tuple[str, str],
    ) -> None:
        r = self._make(vector_store)
        assert r._parse_filter_key(key) == expected

    def test_where_scalar_equality(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        clause, params = r._build_where({"category": "faq"})
        rendered = clause.as_string(None)
        assert '"category" = %s' in rendered
        assert params == ["faq"]

    def test_where_scalar_not(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        clause, params = r._build_where({"category NOT": "faq"})
        rendered = clause.as_string(None)
        assert '"category" <> %s' in rendered
        assert params == ["faq"]

    def test_where_in_via_list_value(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        clause, params = r._build_where({"category": ["faq", "howto"]})
        assert '= ANY(%s)' in clause.as_string(None)
        assert params == [["faq", "howto"]]

    def test_where_not_in_via_list_value(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        clause, params = r._build_where({"category NOT": ["faq"]})
        assert 'NOT ("category" = ANY(%s))' in clause.as_string(None)
        assert params == [["faq"]]

    @pytest.mark.parametrize(
        "suffix,expected_op",
        [(" <", "<"), (" <=", "<="), (" >", ">"), (" >=", ">=")],
    )
    def test_where_comparison_suffixes(
        self,
        vector_store: LakebaseVectorStoreModel,
        suffix: str,
        expected_op: str,
    ) -> None:
        r = self._make(vector_store)
        clause, params = r._build_where({f"source_url{suffix}": "https://a"})
        rendered = clause.as_string(None)
        assert f'"source_url" {expected_op} %s' in rendered
        assert params == ["https://a"]

    def test_where_like_maps_to_ilike(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        clause, params = r._build_where({"source_url LIKE": "%docs%"})
        rendered = clause.as_string(None)
        assert '"source_url" ILIKE %s' in rendered
        assert params == ["%docs%"]

    def test_where_not_like_maps_to_not_ilike(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        clause, params = r._build_where({"source_url NOT LIKE": "%tmp%"})
        rendered = clause.as_string(None)
        assert 'NOT ("source_url" ILIKE %s)' in rendered
        assert params == ["%tmp%"]

    @pytest.mark.parametrize(
        "key",
        ["source_url <=", "source_url LIKE", "source_url NOT LIKE"],
    )
    def test_where_list_value_rejected_on_non_equality(
        self, vector_store: LakebaseVectorStoreModel, key: str
    ) -> None:
        r = self._make(vector_store)
        with pytest.raises(ValueError, match="list values"):
            r._build_where({key: ["x", "y"]})

    def test_where_unknown_column_rejected(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        with pytest.raises(ValueError, match="Unknown filter column"):
            r._build_where({"totally_not_a_column": "x"})

    def test_where_unknown_column_rejected_with_suffix(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        with pytest.raises(ValueError, match="Unknown filter column 'bogus'"):
            r._build_where({"bogus >=": 1})

    def test_stage1_op_form_rejected_at_llm_boundary(self) -> None:
        """Regression guard at the correct layer — the FilterItem model on
        the LLM-facing input rejects Stage 1's ``{op, value}`` dict value.
        (``_build_where`` itself operates on already-normalized suffixed
        keys and can't tell dict-values from other scalars — the shape
        validation happens one level up.)"""
        from pydantic import ValidationError

        from dao_ai.tools.lakebase_search import _build_lakebase_search_input_model

        cls = _build_lakebase_search_input_model(["category"])
        with pytest.raises(ValidationError):
            cls.model_validate(
                {
                    "query": "hi",
                    "filters": [
                        {"key": "category", "value": {"op": "in", "values": ["faq"]}}
                    ],
                }
            )


# ---------------------------------------------------------------------------
# RRF fusion test
# ---------------------------------------------------------------------------


class TestRrf:
    def test_rrf_orders_by_fused_score(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.retrievers.lakebase import LakebaseRetriever

        r = LakebaseRetriever(
            vector_store=vector_store,
            search_parameters=SearchParametersModel(query_type="HYBRID"),
        )
        # doc "A" is #1 in ANN, #3 in BM25 → highest RRF score.
        ann = [
            {"id": "A", "passage": "alpha", "category": "faq", "source_url": None},
            {"id": "B", "passage": "beta", "category": "faq", "source_url": None},
            {"id": "C", "passage": "gamma", "category": "faq", "source_url": None},
        ]
        bm25 = [
            {"id": "C", "passage": "gamma", "category": "faq", "source_url": None},
            {"id": "B", "passage": "beta", "category": "faq", "source_url": None},
            {"id": "A", "passage": "alpha", "category": "faq", "source_url": None},
        ]
        docs = r._rrf(ann, bm25, k=3)
        # A and C tie at (1/61 + 1/63); B in the middle; A wins by insertion order.
        ids = [d.metadata["id"] for d in docs]
        assert ids[0] in {"A", "C"}
        assert set(ids) == {"A", "B", "C"}
        # RRF score must be populated
        assert all("_rrf_score" in d.metadata for d in docs)


# ---------------------------------------------------------------------------
# End-to-end with mocked pool
# ---------------------------------------------------------------------------


class _MockCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.executed: list[tuple[Any, Any]] = []

    def __enter__(self) -> "_MockCursor":
        return self

    def __exit__(self, *a: Any) -> None:
        pass

    def execute(self, stmt: Any, params: Any) -> None:
        self.executed.append((stmt, params))

    def fetchall(self) -> list[dict[str, Any]]:
        return self._rows


class _MockConn:
    def __init__(self, cursor: _MockCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> "_MockConn":
        return self

    def __exit__(self, *a: Any) -> None:
        pass

    def cursor(self, **_kw: Any) -> _MockCursor:
        return self._cursor


class _MockPool:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.cursor = _MockCursor(rows)

    @contextmanager
    def connection(self):
        yield _MockConn(self.cursor)


class TestEndToEnd:
    def test_ann_returns_documents(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.retrievers.lakebase import LakebaseRetriever

        rows = [
            {
                "id": "d1",
                "passage": "How to reset your password",
                "category": "faq",
                "source_url": "https://ex/1",
                "_distance": 0.1,
            },
            {
                "id": "d2",
                "passage": "Enabling MFA",
                "category": "faq",
                "source_url": "https://ex/2",
                "_distance": 0.3,
            },
        ]
        pool = _MockPool(rows)
        r = LakebaseRetriever(
            vector_store=vector_store,
            search_parameters=SearchParametersModel(query_type="ANN", num_results=5),
        )
        with (
            patch.object(r, "_sync_pool", return_value=pool),
            patch.object(r, "_embed", return_value=[0.0] * 1024),
        ):
            docs = r.invoke("reset password")

        assert len(docs) == 2
        assert docs[0].page_content == "How to reset your password"
        # Both Document.id (LangChain top-level) and metadata[id_column] populated.
        assert docs[0].id == "d1"
        assert docs[0].metadata["id"] == "d1"
        assert docs[0].metadata["_distance"] == 0.1
        # SQL was executed with expected LIMIT bind at end.
        _, params = pool.cursor.executed[0]
        assert params[-1] == 5

    def test_ann_with_filters_injects_where(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.retrievers.lakebase import LakebaseRetriever

        pool = _MockPool([])
        r = LakebaseRetriever(
            vector_store=vector_store,
            search_parameters=SearchParametersModel(
                query_type="ANN",
                num_results=5,
                filters={"category": "faq"},  # suffixed-key form (no suffix = equality)
            ),
        )
        with (
            patch.object(r, "_sync_pool", return_value=pool),
            patch.object(r, "_embed", return_value=[0.0] * 1024),
        ):
            r.invoke("x")
        stmt, params = pool.cursor.executed[0]
        rendered = stmt.as_string(None)
        assert '"category" = %s' in rendered
        # Params: [vector, "faq", vector, limit]
        assert params[1] == "faq"
        assert params[-1] == 5

    def test_ann_with_suffix_filter_injects_where(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """Static filter with operator suffix on the key ends up as the
        matching Postgres predicate."""
        from dao_ai.retrievers.lakebase import LakebaseRetriever

        pool = _MockPool([])
        r = LakebaseRetriever(
            vector_store=vector_store,
            search_parameters=SearchParametersModel(
                query_type="ANN",
                num_results=5,
                filters={"source_url LIKE": "%docs%"},
            ),
        )
        with (
            patch.object(r, "_sync_pool", return_value=pool),
            patch.object(r, "_embed", return_value=[0.0] * 1024),
        ):
            r.invoke("x")
        stmt, params = pool.cursor.executed[0]
        rendered = stmt.as_string(None)
        assert '"source_url" ILIKE %s' in rendered
        assert "%docs%" in params

    def test_bm25_returns_documents(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.retrievers.lakebase import LakebaseRetriever

        rows = [
            {
                "id": "d1",
                "passage": "password reset instructions",
                "category": "faq",
                "source_url": None,
                "_score": -8.2,
            },
        ]
        pool = _MockPool(rows)
        r = LakebaseRetriever(
            vector_store=vector_store,
            search_parameters=SearchParametersModel(query_type="BM25", num_results=5),
        )
        with patch.object(r, "_sync_pool", return_value=pool):
            docs = r.invoke("password reset")
        assert len(docs) == 1
        assert docs[0].metadata["_score"] == -8.2


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactory:
    def test_factory_requires_one(self) -> None:
        from dao_ai.tools import create_lakebase_search_tool

        with pytest.raises(ValueError):
            create_lakebase_search_tool()

    def test_factory_rejects_both(self) -> None:
        from dao_ai.tools import create_lakebase_search_tool

        with pytest.raises(ValueError):
            create_lakebase_search_tool(retriever={}, vector_store={})

    def test_factory_from_vector_store_dict(self) -> None:
        from dao_ai.tools import create_lakebase_search_tool

        tool = create_lakebase_search_tool(vector_store=_vs_dict())
        assert tool.name == "lakebase_search"
        assert "Lakebase" in (tool.description or "")

    def test_factory_custom_name_description(self) -> None:
        from dao_ai.tools import create_lakebase_search_tool

        tool = create_lakebase_search_tool(
            vector_store=_vs_dict(),
            name="policy_search",
            description="Search company policies.",
        )
        assert tool.name == "policy_search"
        assert tool.description == "Search company policies."

    def test_factory_output_is_json(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.retrievers.lakebase import LakebaseRetriever
        from dao_ai.tools import create_lakebase_search_tool

        rows = [
            {
                "id": "d1",
                "passage": "hello",
                "category": "faq",
                "source_url": None,
                "_distance": 0.1,
            }
        ]
        pool = _MockPool(rows)

        tool = create_lakebase_search_tool(vector_store=vector_store.model_dump())
        # Patch the retriever the tool closed over. Also stub the Postgres
        # column-discovery lookup so the factory's Mode B/C path doesn't try
        # a real connection.
        with (
            patch(
                "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
                return_value=None,
            ),
            patch.object(LakebaseRetriever, "_sync_pool", return_value=pool),
            patch.object(LakebaseRetriever, "_embed", return_value=[0.0] * 1024),
        ):
            tool = create_lakebase_search_tool(vector_store=vector_store.model_dump())
            output = tool.invoke({"query": "hi"})

        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert parsed[0]["page_content"] == "hello"
        assert parsed[0]["metadata"]["id"] == "d1"

    def test_factory_runtime_filter_item_list(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """LLM-facing FilterItem list is converted to a suffixed dict at the
        tool boundary and reaches _build_where correctly."""
        from dao_ai.retrievers.lakebase import LakebaseRetriever
        from dao_ai.tools import create_lakebase_search_tool

        pool = _MockPool([])

        with (
            patch(
                "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
                return_value=None,
            ),
            patch.object(LakebaseRetriever, "_sync_pool", return_value=pool),
            patch.object(LakebaseRetriever, "_embed", return_value=[0.0] * 1024),
        ):
            tool = create_lakebase_search_tool(vector_store=vector_store.model_dump())
            tool.invoke(
                {
                    "query": "hi",
                    "filters": [
                        {"key": "category", "value": "faq"},
                        {"key": "source_url LIKE", "value": "%docs%"},
                    ],
                }
            )

        stmt, params = pool.cursor.executed[0]
        rendered = stmt.as_string(None)
        assert '"category" = %s' in rendered
        assert '"source_url" ILIKE %s' in rendered
        assert "faq" in params
        assert "%docs%" in params

    def test_factory_dynamic_schema_rejects_unlisted_key(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """When we can enumerate columns, the tool's args_schema narrows
        `filters[].key` to a Literal — an unlisted key fails before we ever
        reach the retriever."""
        from dao_ai.tools import create_lakebase_search_tool

        with patch(
            "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
            # Simulate Postgres returning the metadata columns already known.
            return_value=[("category", "string", None), ("priority", "number", None)],
        ):
            tool = create_lakebase_search_tool(vector_store=vector_store.model_dump())

        # Invalid column key — the DynamicLakebaseSearchInput / DynamicFilterItem
        # narrowing must reject it at Pydantic validation.
        with pytest.raises(Exception):
            tool.invoke(
                {
                    "query": "hi",
                    "filters": [{"key": "does_not_exist", "value": "x"}],
                }
            )

    def test_factory_free_form_filters_when_no_columns(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """When metadata_columns is empty AND Postgres discovery fails, the
        factory falls back to the free-form LakebaseSearchInput schema."""
        from dao_ai.tools import create_lakebase_search_tool
        from dao_ai.tools.lakebase_search import LakebaseSearchInput

        vs = LakebaseVectorStoreModel(**_vs_dict(metadata_columns=[]))
        with patch(
            "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
            return_value=None,
        ):
            tool = create_lakebase_search_tool(vector_store=vs.model_dump())
        # args_schema is the base free-form model, not a Dynamic subclass.
        assert tool.args_schema is LakebaseSearchInput

    def test_factory_column_info_operator_narrowing(
        self, database_model_dict: dict[str, Any] | None = None,
    ) -> None:
        """Hand-declared ColumnInfo with a narrower operator list is honored
        end-to-end. Reaches the schema builder as an operator override so
        the LLM sees only the allowed suffixes."""
        from dao_ai.tools import create_lakebase_search_tool

        vs = LakebaseVectorStoreModel(
            **_vs_dict(),
        )
        # Hand-declared column, operators locked to equality + LIKE only.
        retriever = LakebaseRetrieverModel(
            vector_store=vs,
            columns=[
                ColumnInfo(name="category", type="string", operators=["", "LIKE"]),
            ],
        )
        with patch(
            "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
            return_value=None,  # would be skipped anyway in Mode A
        ):
            tool = create_lakebase_search_tool(retriever=retriever)

        # A comparison suffix on the hand-declared column must be rejected
        # by the args_schema Literal.
        with pytest.raises(Exception):
            tool.invoke(
                {
                    "query": "hi",
                    "filters": [{"key": "category >=", "value": "x"}],
                }
            )


# ---------------------------------------------------------------------------
# Dynamic schema builder
# ---------------------------------------------------------------------------


class TestDynamicSchemaBuilder:
    def test_empty_columns_returns_freeform(self) -> None:
        from dao_ai.tools.lakebase_search import (
            LakebaseSearchInput,
            _build_lakebase_search_input_model,
        )

        cls = _build_lakebase_search_input_model([])
        assert cls is LakebaseSearchInput

    def test_narrowed_schema_accepts_declared_key(self) -> None:
        from dao_ai.tools.lakebase_search import _build_lakebase_search_input_model

        cls = _build_lakebase_search_input_model(["category", "priority"])
        # Valid: bare column, and column-with-suffix.
        inst = cls.model_validate(
            {
                "query": "hi",
                "filters": [
                    {"key": "category", "value": "faq"},
                    {"key": "priority >=", "value": 2},
                ],
            }
        )
        assert len(inst.filters) == 2

    def test_narrowed_schema_rejects_undeclared_key(self) -> None:
        from pydantic import ValidationError

        from dao_ai.tools.lakebase_search import _build_lakebase_search_input_model

        cls = _build_lakebase_search_input_model(["category"])
        with pytest.raises(ValidationError):
            cls.model_validate(
                {
                    "query": "hi",
                    "filters": [{"key": "not_a_column", "value": "x"}],
                }
            )

    def test_operator_overrides_narrow_enum(self) -> None:
        from pydantic import ValidationError

        from dao_ai.tools.lakebase_search import _build_lakebase_search_input_model

        cls = _build_lakebase_search_input_model(
            ["category"], operator_overrides={"category": ["", "LIKE"]}
        )
        # Equality + LIKE allowed.
        cls.model_validate(
            {
                "query": "hi",
                "filters": [
                    {"key": "category", "value": "faq"},
                    {"key": "category LIKE", "value": "%faq%"},
                ],
            }
        )
        # Comparison NOT allowed under the override.
        with pytest.raises(ValidationError):
            cls.model_validate(
                {
                    "query": "hi",
                    "filters": [{"key": "category >=", "value": "x"}],
                }
            )


# ---------------------------------------------------------------------------
# _fetch_lakebase_columns + Mode A/B/C dispatch
# ---------------------------------------------------------------------------


class TestFetchLakebaseColumns:
    def test_returns_none_on_pool_error(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.tools.lakebase_search import _fetch_lakebase_columns

        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            side_effect=ConnectionError("nope"),
        ):
            assert _fetch_lakebase_columns(vector_store) is None

    def test_strips_reserved_columns(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """Reserved columns (id/content/embedding/tsvector) never appear on
        the LLM-facing filter enum."""
        from dao_ai.tools.lakebase_search import _fetch_lakebase_columns

        # information_schema returns everything; we should strip the 4 reserved.
        rows = [
            {"column_name": "id", "data_type": "text", "comment": None},
            {"column_name": "passage", "data_type": "text", "comment": None},
            {"column_name": "embedding", "data_type": "USER-DEFINED", "comment": None},
            {"column_name": "passage_tsv", "data_type": "tsvector", "comment": None},
            {"column_name": "category", "data_type": "text", "comment": "cat comment"},
            {"column_name": "priority", "data_type": "integer", "comment": None},
        ]
        pool = _MockPool(rows)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool", return_value=pool
        ):
            out = _fetch_lakebase_columns(vector_store)

        assert out is not None
        names = [n for n, _, _ in out]
        assert names == ["category", "priority"]
        # Type mapping applied.
        types = {n: t for n, t, _ in out}
        assert types["category"] == "string"
        assert types["priority"] == "number"
        # Column comment surfaced.
        comments = {n: c for n, _, c in out}
        assert comments["category"] == "cat comment"


class TestModeDispatch:
    """Cover the 3 discovery-mode branches in create_lakebase_search_tool."""

    def test_mode_a_hand_declared_skips_postgres(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.tools import create_lakebase_search_tool

        retriever = LakebaseRetrieverModel(
            vector_store=vector_store,
            columns=[ColumnInfo(name="category", type="string")],
        )
        with patch(
            "dao_ai.tools.lakebase_search._fetch_lakebase_columns"
        ) as mock_fetch:
            create_lakebase_search_tool(retriever=retriever)
            mock_fetch.assert_not_called()

    def test_mode_b_bare_strings_calls_postgres_for_enrichment(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.tools import create_lakebase_search_tool

        # metadata_columns declares bare strings; ColumnInfo count == 0.
        with patch(
            "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
            return_value=[
                ("category", "string", "cat comment"),
                ("priority", "number", None),
            ],
        ) as mock_fetch:
            create_lakebase_search_tool(vector_store=vector_store.model_dump())
            mock_fetch.assert_called_once()

    def test_mode_c_empty_metadata_columns_uses_discovery(self) -> None:
        from dao_ai.tools import create_lakebase_search_tool

        vs = LakebaseVectorStoreModel(**_vs_dict(metadata_columns=[]))
        with patch(
            "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
            return_value=[("discovered_col", "string", None)],
        ) as mock_fetch:
            tool = create_lakebase_search_tool(vector_store=vs.model_dump())
            mock_fetch.assert_called_once()

        # args_schema should be the dynamic subclass with `discovered_col`
        # accepted as a key.
        inst = tool.args_schema.model_validate(
            {
                "query": "hi",
                "filters": [{"key": "discovered_col", "value": "x"}],
            }
        )
        assert len(inst.filters) == 1

    def test_mode_c_falls_back_to_metadata_columns_on_pg_error(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.tools import create_lakebase_search_tool

        # metadata_columns is set; simulate Postgres discovery failing.
        with patch(
            "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
            return_value=None,
        ):
            tool = create_lakebase_search_tool(vector_store=vector_store.model_dump())

        # Enum built from declared metadata_columns.
        inst = tool.args_schema.model_validate(
            {
                "query": "hi",
                "filters": [{"key": "category", "value": "faq"}],
            }
        )
        assert len(inst.filters) == 1

    def test_array_column_narrowed_to_equality_only(self) -> None:
        """When Postgres reports an ARRAY column type, the factory adds an
        operator override so only equality is exposed on the LLM enum."""
        from pydantic import ValidationError

        from dao_ai.tools import create_lakebase_search_tool

        vs = LakebaseVectorStoreModel(**_vs_dict(metadata_columns=[]))
        with patch(
            "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
            return_value=[("tags", "array", None)],
        ):
            tool = create_lakebase_search_tool(vector_store=vs.model_dump())

        # Equality on the array column is fine.
        tool.args_schema.model_validate(
            {"query": "hi", "filters": [{"key": "tags", "value": ["a", "b"]}]}
        )
        # LIKE on an array column is NOT valid.
        with pytest.raises(ValidationError):
            tool.args_schema.model_validate(
                {"query": "hi", "filters": [{"key": "tags LIKE", "value": "%x%"}]}
            )
