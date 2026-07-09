"""Unit tests for `DatabaseModel.execute_sql` + `LakebaseVectorStoreModel.provision`.

Both helpers wrap `PostgresPoolManager.get_pool(db).connection()`.
We mock the pool + cursor and assert the exact SQL statements emitted.
Live Postgres testing is out of scope here — the DDL grammar is verified
by the Lakebase server at runtime.
"""

from unittest.mock import MagicMock, patch

import pytest

from dao_ai.config import DatabaseModel, LakebaseVectorStoreModel


def _mock_pool(cursor: MagicMock) -> MagicMock:
    """Build a context-manager pool whose .connection().cursor() → given mock."""
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    pool = MagicMock()
    pool.connection.return_value.__enter__.return_value = conn
    return pool, conn


def _db() -> DatabaseModel:
    """Build a minimal Lakebase DatabaseModel for the helper tests."""
    return DatabaseModel(project="my-project", name="my_project_db")


@pytest.mark.unit
class TestExecuteSql:
    def test_single_statement(self) -> None:
        cur = MagicMock()
        pool, conn = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            _db().execute_sql("SELECT 1")
        cur.execute.assert_called_once_with("SELECT 1")
        conn.commit.assert_called_once()
        conn.rollback.assert_not_called()

    def test_sequence_runs_all_and_commits_once(self) -> None:
        cur = MagicMock()
        pool, conn = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            _db().execute_sql(["CREATE EXTENSION x;", "CREATE TABLE y (id int);"])
        assert cur.execute.call_count == 2
        cur.execute.assert_any_call("CREATE EXTENSION x;")
        cur.execute.assert_any_call("CREATE TABLE y (id int);")
        conn.commit.assert_called_once()

    def test_rollback_on_error(self) -> None:
        cur = MagicMock()
        cur.execute.side_effect = RuntimeError("boom")
        pool, conn = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            with pytest.raises(RuntimeError, match="boom"):
                _db().execute_sql("bad sql")
        conn.commit.assert_not_called()
        conn.rollback.assert_called_once()

    def test_empty_sequence_is_no_op(self) -> None:
        cur = MagicMock()
        pool, _ = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            _db().execute_sql([])
        cur.execute.assert_not_called()
        # No pool acquisition either
        pool.connection.assert_not_called()

    def test_parameters_forwarded(self) -> None:
        cur = MagicMock()
        pool, _ = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            _db().execute_sql("SELECT %s", parameters=(42,))
        cur.execute.assert_called_once_with("SELECT %s", (42,))

    def test_parameters_with_sequence_rejected(self) -> None:
        with pytest.raises(ValueError, match="single string"):
            _db().execute_sql(["a", "b"], parameters=(1,))


def _vs(**overrides) -> LakebaseVectorStoreModel:
    """Minimal LakebaseVectorStoreModel with sensible defaults."""
    base = dict(
        database=_db(),
        table="kb_articles",
        content_column="passage",
        embedding_column="embedding",
        embedding_model="databricks-gte-large-en",
    )
    base.update(overrides)
    return LakebaseVectorStoreModel(**base)


@pytest.mark.unit
class TestProvision:
    def test_ann_only_minimal(self) -> None:
        cur = MagicMock()
        pool, _ = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            _vs().provision(dimension=1024)
        stmts = [c.args[0] for c in cur.execute.call_args_list]
        # Only lakebase_vector — no lakebase_text without tsvector_column
        assert any("CREATE EXTENSION IF NOT EXISTS lakebase_vector" in s for s in stmts)
        assert not any("lakebase_text" in s for s in stmts)
        assert any(
            "CREATE TABLE IF NOT EXISTS public.kb_articles" in s for s in stmts
        )
        assert any("id text PRIMARY KEY" in s for s in stmts)
        assert any("passage text NOT NULL" in s for s in stmts)
        assert any("embedding vector(1024)" in s for s in stmts)
        # ANN index only — no BM25 without tsvector
        ann_stmts = [s for s in stmts if "USING lakebase_ann" in s]
        bm25_stmts = [s for s in stmts if "USING lakebase_bm25" in s]
        assert len(ann_stmts) == 1
        assert len(bm25_stmts) == 0
        assert "vector_cosine_ops" in ann_stmts[0]
        assert "CREATE INDEX IF NOT EXISTS" in ann_stmts[0]

    def test_hybrid_adds_tsvector_extension_column_and_index(self) -> None:
        cur = MagicMock()
        pool, _ = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            _vs(tsvector_column="passage_tsv").provision(dimension=1024)
        stmts = [c.args[0] for c in cur.execute.call_args_list]
        assert any("lakebase_text" in s for s in stmts)
        assert any(
            "passage_tsv tsvector GENERATED ALWAYS AS "
            "(to_tsvector('english', passage)) STORED" in s
            for s in stmts
        )
        bm25 = [s for s in stmts if "USING lakebase_bm25" in s]
        assert len(bm25) == 1
        assert "passage_tsv" in bm25[0]

    def test_metadata_column_types_respected(self) -> None:
        cur = MagicMock()
        pool, _ = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            _vs(metadata_columns=["category", "priority"]).provision(
                dimension=1024,
                metadata_column_types={"priority": "int"},
            )
        stmts = [c.args[0] for c in cur.execute.call_args_list]
        create = next(s for s in stmts if "CREATE TABLE" in s)
        # category defaults to text; priority typed explicitly
        assert "category text" in create
        assert "priority int" in create

    def test_distance_metric_l2(self) -> None:
        cur = MagicMock()
        pool, _ = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            _vs(distance_metric="l2").provision(dimension=768)
        stmts = [c.args[0] for c in cur.execute.call_args_list]
        ann = next(s for s in stmts if "USING lakebase_ann" in s)
        assert "vector_l2_ops" in ann
        assert "vector(768)" in next(s for s in stmts if "CREATE TABLE" in s)

    def test_id_column_type_override(self) -> None:
        cur = MagicMock()
        pool, _ = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            _vs().provision(dimension=1024, id_column_type="bigint")
        stmts = [c.args[0] for c in cur.execute.call_args_list]
        create = next(s for s in stmts if "CREATE TABLE" in s)
        assert "id bigint PRIMARY KEY" in create

    def test_custom_schema_name_qualifies_object_names(self) -> None:
        cur = MagicMock()
        pool, _ = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            _vs(schema_name="dao_ai_kb").provision(dimension=1024)
        stmts = [c.args[0] for c in cur.execute.call_args_list]
        assert any("CREATE TABLE IF NOT EXISTS dao_ai_kb.kb_articles" in s for s in stmts)

    def test_bad_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="positive int"):
            _vs().provision(dimension=0)

    def test_idempotent_uses_if_not_exists_everywhere(self) -> None:
        """Every statement must include an IF NOT EXISTS guard so re-runs are safe."""
        cur = MagicMock()
        pool, _ = _mock_pool(cur)
        with patch(
            "dao_ai.memory.postgres.PostgresPoolManager.get_pool",
            return_value=pool,
        ):
            _vs(tsvector_column="passage_tsv").provision(dimension=1024)
        stmts = [c.args[0] for c in cur.execute.call_args_list]
        for stmt in stmts:
            assert "IF NOT EXISTS" in stmt, f"missing IF NOT EXISTS: {stmt!r}"
