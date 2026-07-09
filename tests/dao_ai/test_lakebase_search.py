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

    def test_where_scalar_equality(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        clause, params = r._build_where({"category": "faq"})
        rendered = clause.as_string(None)
        assert '"category" = %s' in rendered
        assert params == ["faq"]

    def test_where_in_operator(self, vector_store: LakebaseVectorStoreModel) -> None:
        r = self._make(vector_store)
        clause, params = r._build_where(
            {"category": {"op": "in", "values": ["faq", "howto"]}}
        )
        rendered = clause.as_string(None)
        assert '= ANY(%s)' in rendered
        assert params == [["faq", "howto"]]

    def test_where_comparison_operators(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        clause, params = r._build_where(
            {"source_url": {"op": ">=", "value": "https://a"}}
        )
        rendered = clause.as_string(None)
        assert '"source_url" >= %s' in rendered
        assert params == ["https://a"]

    def test_where_is_null(self, vector_store: LakebaseVectorStoreModel) -> None:
        r = self._make(vector_store)
        clause, _ = r._build_where({"category": {"op": "is_null"}})
        assert '"category" IS NULL' in clause.as_string(None)

        clause_neg, _ = r._build_where(
            {"category": {"op": "is_null", "value": False}}
        )
        assert '"category" IS NOT NULL' in clause_neg.as_string(None)

    def test_where_unknown_column_rejected(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        with pytest.raises(ValueError, match="Unknown filter column"):
            r._build_where({"totally_not_a_column": "x"})

    def test_where_unsupported_op_rejected(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = self._make(vector_store)
        with pytest.raises(ValueError, match="Unsupported filter op"):
            r._build_where({"category": {"op": "regex", "value": "foo"}})


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
                filters={"category": "faq"},
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
        # Patch the retriever the tool closed over.
        with (
            patch.object(LakebaseRetriever, "_sync_pool", return_value=pool),
            patch.object(LakebaseRetriever, "_embed", return_value=[0.0] * 1024),
        ):
            output = tool.invoke({"query": "hi"})

        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert parsed[0]["page_content"] == "hello"
        assert parsed[0]["metadata"]["id"] == "d1"
