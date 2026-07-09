"""Unit tests for `dao_ai.lakebase.backfill_embeddings`.

Mocks the DatabaseModel's execute_query + execute_many so tests exercise
the batching / no-op / SQL-shape behavior without a live database.
"""

from unittest.mock import MagicMock, patch

import pytest

from dao_ai.config import DatabaseModel, LakebaseVectorStoreModel
from dao_ai.lakebase import backfill_embeddings


def _vs(**overrides: object) -> LakebaseVectorStoreModel:
    """Build a minimal Lakebase vector store config."""
    base: dict[str, object] = dict(
        database=DatabaseModel(project="p", name="p_db"),
        table="kb_articles",
        content_column="passage",
        embedding_column="embedding",
        embedding_model="databricks-gte-large-en",
    )
    base.update(overrides)
    return LakebaseVectorStoreModel(**base)


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        # Deterministic 3-dim vectors for verification
        self.next_vec: int = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        vecs: list[list[float]] = []
        for _ in texts:
            self.next_vec += 1
            vecs.append([float(self.next_vec), 0.0, 0.0])
        return vecs


@pytest.mark.unit
class TestBackfillEmbeddings:
    def test_no_rows_returns_zero_and_no_writes(self) -> None:
        vs = _vs()
        with (
            patch.object(DatabaseModel, "execute_query", return_value=[]) as eq,
            patch.object(DatabaseModel, "execute_many") as em,
        ):
            count = backfill_embeddings(vs, embedder=_FakeEmbedder())
        assert count == 0
        eq.assert_called_once()
        em.assert_not_called()

    def test_single_batch(self) -> None:
        vs = _vs()
        rows = [{"id": "d01", "content": "hi"}, {"id": "d02", "content": "there"}]
        embedder = _FakeEmbedder()
        with (
            patch.object(DatabaseModel, "execute_query", return_value=rows),
            patch.object(DatabaseModel, "execute_many") as em,
        ):
            count = backfill_embeddings(vs, embedder=embedder, batch_size=10)
        assert count == 2
        assert embedder.calls == [["hi", "there"]]
        em.assert_called_once()
        sql, params = em.call_args.args
        assert sql == (
            "UPDATE public.kb_articles SET embedding = %s::vector WHERE id = %s"
        )
        assert list(params) == [([1.0, 0.0, 0.0], "d01"), ([2.0, 0.0, 0.0], "d02")]

    def test_batching_splits_by_batch_size(self) -> None:
        vs = _vs()
        rows = [{"id": f"d{i}", "content": f"txt{i}"} for i in range(5)]
        embedder = _FakeEmbedder()
        with (
            patch.object(DatabaseModel, "execute_query", return_value=rows),
            patch.object(DatabaseModel, "execute_many") as em,
        ):
            count = backfill_embeddings(vs, embedder=embedder, batch_size=2)
        assert count == 5
        assert [len(c) for c in embedder.calls] == [2, 2, 1]
        assert em.call_count == 3

    def test_reads_qualified_schema_and_columns_from_config(self) -> None:
        vs = _vs(schema_name="kb", table="docs", id_column="doc_id",
                 content_column="body", embedding_column="vec")
        with (
            patch.object(DatabaseModel, "execute_query", return_value=[]) as eq,
            patch.object(DatabaseModel, "execute_many"),
        ):
            backfill_embeddings(vs, embedder=_FakeEmbedder())
        (select_sql,) = eq.call_args.args
        assert select_sql == (
            "SELECT doc_id AS id, body AS content FROM kb.docs WHERE vec IS NULL"
        )
