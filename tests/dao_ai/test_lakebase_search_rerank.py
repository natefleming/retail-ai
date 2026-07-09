"""Unit tests for lakebase_search's FlashRank rerank integration.

Covers:
- Config: `rerank: true` coerces to default FlashRank model; explicit
  `RerankParametersModel` passes through; `rerank: false` and unset both
  disable reranking.
- Factory: `build_flashrank_ranker` invoked at factory init when rerank
  is set; skipped when unset.
- End-to-end (mocked): `rerank_documents` is called after retrieval;
  `reranker_score` lands in each Document.metadata; ordering matches
  the mocked ranker output; `top_n` truncation is honored.
- Regression: factory works unchanged when `rerank` is unset (Stage 1
  tests continue to pass — asserted by a null-rerank path here).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dao_ai.config import (
    LakebaseRetrieverModel,
    LakebaseVectorStoreModel,
    RerankParametersModel,
    SearchParametersModel,
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
        "metadata_columns": ["category"],
    }
    base.update(overrides)
    return base


@pytest.fixture()
def vector_store() -> LakebaseVectorStoreModel:
    return LakebaseVectorStoreModel(**_vs_dict())


# Mocked psycopg pool for the E2E rerank test path.
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


# ---------------------------------------------------------------------------
# Config coercion
# ---------------------------------------------------------------------------


class TestRerankConfig:
    def test_rerank_true_coerces_to_default_model(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = LakebaseRetrieverModel(vector_store=vector_store, rerank=True)
        assert isinstance(r.rerank, RerankParametersModel)
        assert r.rerank.model == "ms-marco-MiniLM-L-12-v2"

    def test_rerank_false_stays_false(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = LakebaseRetrieverModel(vector_store=vector_store, rerank=False)
        assert r.rerank is False

    def test_rerank_none_default(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        r = LakebaseRetrieverModel(vector_store=vector_store)
        assert r.rerank is None

    def test_explicit_rerank_params_pass_through(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        params = RerankParametersModel(
            model="ms-marco-TinyBERT-L-2-v2", top_n=3
        )
        r = LakebaseRetrieverModel(vector_store=vector_store, rerank=params)
        assert r.rerank is params
        assert r.rerank.model == "ms-marco-TinyBERT-L-2-v2"
        assert r.rerank.top_n == 3

    def test_yaml_dict_form_validates(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """YAML-friendly dict form on the rerank field."""
        r = LakebaseRetrieverModel.model_validate(
            {
                "vector_store": vector_store.model_dump(),
                "rerank": {"model": "ms-marco-MiniLM-L-12-v2", "top_n": 5},
            }
        )
        assert isinstance(r.rerank, RerankParametersModel)
        assert r.rerank.top_n == 5


# ---------------------------------------------------------------------------
# Factory: build_flashrank_ranker is invoked when configured
# ---------------------------------------------------------------------------


class TestFactoryRankerInit:
    def test_ranker_built_when_rerank_set(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.tools import create_lakebase_search_tool

        retriever = LakebaseRetrieverModel(vector_store=vector_store, rerank=True)
        with (
            patch(
                "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
                return_value=None,
            ),
            patch(
                "dao_ai.tools.lakebase_search.build_flashrank_ranker"
            ) as mock_builder,
        ):
            mock_builder.return_value = MagicMock(name="Ranker")
            create_lakebase_search_tool(retriever=retriever)
            mock_builder.assert_called_once()
            # Passed the coerced RerankParametersModel, not the raw bool.
            (arg,) = mock_builder.call_args.args
            assert isinstance(arg, RerankParametersModel)

    def test_ranker_skipped_when_rerank_unset(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.tools import create_lakebase_search_tool

        retriever = LakebaseRetrieverModel(vector_store=vector_store)  # no rerank
        with (
            patch(
                "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
                return_value=None,
            ),
            patch(
                "dao_ai.tools.lakebase_search.build_flashrank_ranker"
            ) as mock_builder,
        ):
            create_lakebase_search_tool(retriever=retriever)
            # Called with None → returns None internally; no ranker built.
            mock_builder.assert_called_once_with(None)

    def test_ranker_skipped_when_rerank_false(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        from dao_ai.tools import create_lakebase_search_tool

        retriever = LakebaseRetrieverModel(vector_store=vector_store, rerank=False)
        with (
            patch(
                "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
                return_value=None,
            ),
            patch(
                "dao_ai.tools.lakebase_search.build_flashrank_ranker"
            ) as mock_builder,
        ):
            create_lakebase_search_tool(retriever=retriever)
            mock_builder.assert_called_once_with(None)


# ---------------------------------------------------------------------------
# End-to-end: rerank actually reorders the retriever's output
# ---------------------------------------------------------------------------


class TestRerankEndToEnd:
    def test_rerank_reorders_and_annotates(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """Retriever returns 3 docs in ANN order; ranker returns a
        different order with scores; final output reflects the ranker
        order and every doc has reranker_score in its metadata."""
        from dao_ai.retrievers.lakebase import LakebaseRetriever
        from dao_ai.tools import create_lakebase_search_tool

        rows = [
            {"id": "d1", "passage": "First doc", "category": "a", "_distance": 0.1},
            {"id": "d2", "passage": "Second doc", "category": "b", "_distance": 0.2},
            {"id": "d3", "passage": "Third doc", "category": "c", "_distance": 0.3},
        ]
        pool = _MockPool(rows)

        # Fake ranker: reverses the input order + attaches scores.
        fake_ranker = MagicMock(name="Ranker")

        def fake_rerank(request):
            # Return passages in reverse order with descending scores.
            reversed_passages = list(reversed(request.passages))
            for i, p in enumerate(reversed_passages):
                p["score"] = 0.9 - (i * 0.1)
            return reversed_passages

        fake_ranker.rerank = fake_rerank

        retriever = LakebaseRetrieverModel(
            vector_store=vector_store,
            rerank=True,
            search_parameters=SearchParametersModel(query_type="ANN", num_results=3),
        )

        with (
            patch(
                "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
                return_value=None,
            ),
            patch(
                "dao_ai.tools.lakebase_search.build_flashrank_ranker",
                return_value=fake_ranker,
            ),
            patch.object(LakebaseRetriever, "_sync_pool", return_value=pool),
            patch.object(LakebaseRetriever, "_embed", return_value=[0.0] * 1024),
        ):
            tool = create_lakebase_search_tool(retriever=retriever)
            output = tool.invoke({"query": "anything"})

        parsed = json.loads(output)
        # 3 docs, reversed from retriever order.
        ids = [d["metadata"]["id"] for d in parsed]
        assert ids == ["d3", "d2", "d1"], f"expected reversed, got {ids}"
        # Every doc has reranker_score.
        for d in parsed:
            assert "reranker_score" in d["metadata"], d["metadata"]
        assert parsed[0]["metadata"]["reranker_score"] == pytest.approx(0.9)

    def test_rerank_top_n_truncates(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """rerank.top_n limits the returned doc count."""
        from dao_ai.retrievers.lakebase import LakebaseRetriever
        from dao_ai.tools import create_lakebase_search_tool

        rows = [
            {"id": f"d{i}", "passage": f"Doc {i}", "category": "x", "_distance": i / 10}
            for i in range(5)
        ]
        pool = _MockPool(rows)

        fake_ranker = MagicMock(name="Ranker")

        def fake_rerank(request):
            for i, p in enumerate(request.passages):
                p["score"] = 1.0 - (i * 0.1)
            return request.passages

        fake_ranker.rerank = fake_rerank

        retriever = LakebaseRetrieverModel(
            vector_store=vector_store,
            rerank=RerankParametersModel(model="ms-marco-MiniLM-L-12-v2", top_n=2),
            search_parameters=SearchParametersModel(query_type="ANN", num_results=5),
        )

        with (
            patch(
                "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
                return_value=None,
            ),
            patch(
                "dao_ai.tools.lakebase_search.build_flashrank_ranker",
                return_value=fake_ranker,
            ),
            patch.object(LakebaseRetriever, "_sync_pool", return_value=pool),
            patch.object(LakebaseRetriever, "_embed", return_value=[0.0] * 1024),
        ):
            tool = create_lakebase_search_tool(retriever=retriever)
            output = tool.invoke({"query": "x"})

        parsed = json.loads(output)
        assert len(parsed) == 2, f"top_n=2 should truncate, got {len(parsed)}"

    def test_no_rerank_when_docs_empty(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """When retriever returns no docs, the ranker is not invoked."""
        from dao_ai.retrievers.lakebase import LakebaseRetriever
        from dao_ai.tools import create_lakebase_search_tool

        pool = _MockPool([])  # empty result set
        fake_ranker = MagicMock(name="Ranker")

        retriever = LakebaseRetrieverModel(vector_store=vector_store, rerank=True)

        with (
            patch(
                "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
                return_value=None,
            ),
            patch(
                "dao_ai.tools.lakebase_search.build_flashrank_ranker",
                return_value=fake_ranker,
            ),
            patch.object(LakebaseRetriever, "_sync_pool", return_value=pool),
            patch.object(LakebaseRetriever, "_embed", return_value=[0.0] * 1024),
        ):
            tool = create_lakebase_search_tool(retriever=retriever)
            output = tool.invoke({"query": "x"})

        parsed = json.loads(output)
        assert parsed == []
        fake_ranker.rerank.assert_not_called()

    def test_no_rerank_when_ranker_init_failed(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """When build_flashrank_ranker returns None (init failed), the
        retriever still returns docs — just without rerank scores."""
        from dao_ai.retrievers.lakebase import LakebaseRetriever
        from dao_ai.tools import create_lakebase_search_tool

        rows = [
            {"id": "d1", "passage": "hello", "category": "a", "_distance": 0.1}
        ]
        pool = _MockPool(rows)

        retriever = LakebaseRetrieverModel(vector_store=vector_store, rerank=True)

        with (
            patch(
                "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
                return_value=None,
            ),
            patch(
                "dao_ai.tools.lakebase_search.build_flashrank_ranker",
                return_value=None,  # simulate init failure
            ),
            patch.object(LakebaseRetriever, "_sync_pool", return_value=pool),
            patch.object(LakebaseRetriever, "_embed", return_value=[0.0] * 1024),
        ):
            tool = create_lakebase_search_tool(retriever=retriever)
            output = tool.invoke({"query": "x"})

        parsed = json.loads(output)
        assert len(parsed) == 1
        assert "reranker_score" not in parsed[0]["metadata"]


# ---------------------------------------------------------------------------
# Regression: unset rerank behaves exactly like Stage 1
# ---------------------------------------------------------------------------


class TestNoRegressionWhenRerankUnset:
    def test_stage1_path_unchanged(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """When `rerank` is unset the factory + invocation path matches
        pre-PR-A behavior (docs returned as-is, no reranker_score)."""
        from dao_ai.retrievers.lakebase import LakebaseRetriever
        from dao_ai.tools import create_lakebase_search_tool

        rows = [
            {"id": "d1", "passage": "One", "category": "a", "_distance": 0.1},
            {"id": "d2", "passage": "Two", "category": "b", "_distance": 0.2},
        ]
        pool = _MockPool(rows)

        # No rerank param.
        retriever = LakebaseRetrieverModel(vector_store=vector_store)

        with (
            patch(
                "dao_ai.tools.lakebase_search._fetch_lakebase_columns",
                return_value=None,
            ),
            patch.object(LakebaseRetriever, "_sync_pool", return_value=pool),
            patch.object(LakebaseRetriever, "_embed", return_value=[0.0] * 1024),
        ):
            tool = create_lakebase_search_tool(retriever=retriever)
            output = tool.invoke({"query": "x"})

        parsed = json.loads(output)
        # Original retriever order, no reranker_score annotation.
        assert [d["metadata"]["id"] for d in parsed] == ["d1", "d2"]
        for d in parsed:
            assert "reranker_score" not in d["metadata"]
