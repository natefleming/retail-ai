"""Live end-to-end tests for LakebaseRetriever + create_lakebase_search_tool.

Runs against a real Databricks Lakebase project with the ``lakebase_vector``
and ``lakebase_text`` extensions enabled. Skipped unless:

    LAKEBASE_TEST_PROJECT=<project id>            (required)
    LAKEBASE_TEST_SECRET_SCOPE=<secret scope>     (required)
    LAKEBASE_TEST_CLIENT_ID_KEY=<key>             (required)
    LAKEBASE_TEST_CLIENT_SECRET_KEY=<key>         (required)
    LAKEBASE_TEST_HOST_KEY=<key>                  (required)
    LAKEBASE_TEST_SCHEMA=<schema>                 (default: dao_ai_lakebase_test)
    LAKEBASE_TEST_TABLE=<table>                   (default: kb_articles)
    LAKEBASE_TEST_EMBEDDING_ENDPOINT=<endpoint>   (default: databricks-gte-large-en)

Also requires ``DATABRICKS_CONFIG_PROFILE`` (or ambient auth) to be able to
read the secret scope.

The fixture provisions its own schema, table, and indexes and seeds a small
KB; the tests then exercise every query type × several filter shapes.

Marked ``integration`` so it's excluded from the default fast unit run:

    uv run pytest tests/dao_ai/test_lakebase_search_live.py -v -m integration
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from psycopg import sql

from dao_ai.config import (
    DatabaseModel,
    LakebaseRetrieverModel,
    LakebaseVectorStoreModel,
    SearchParametersModel,
)
from dao_ai.memory.postgres import PostgresPoolManager
from dao_ai.retrievers.lakebase import LakebaseRetriever
from dao_ai.tools import create_lakebase_search_tool

pytestmark = pytest.mark.integration

REQUIRED_ENV = (
    "LAKEBASE_TEST_PROJECT",
    "LAKEBASE_TEST_SECRET_SCOPE",
    "LAKEBASE_TEST_CLIENT_ID_KEY",
    "LAKEBASE_TEST_CLIENT_SECRET_KEY",
    "LAKEBASE_TEST_HOST_KEY",
)

if any(os.environ.get(k) is None for k in REQUIRED_ENV):
    pytest.skip(
        f"live Lakebase tests require env vars: {', '.join(REQUIRED_ENV)}",
        allow_module_level=True,
    )

SCHEMA = os.environ.get("LAKEBASE_TEST_SCHEMA", "dao_ai_lakebase_test")
TABLE = os.environ.get("LAKEBASE_TEST_TABLE", "kb_articles")
EMBEDDING_ENDPOINT = os.environ.get(
    "LAKEBASE_TEST_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
)
EMBEDDING_DIM = 1024  # matches databricks-gte-large-en


SEED_DOCS: list[dict[str, Any]] = [
    {
        "id": "d01",
        "category": "auth",
        "priority": 1,
        "passage": "To reset your password, go to Settings then Security and click 'Reset password'.",
    },
    {
        "id": "d02",
        "category": "auth",
        "priority": 2,
        "passage": "Enable multi-factor authentication (MFA) from your account security page.",
    },
    {
        "id": "d03",
        "category": "billing",
        "priority": 3,
        "passage": "Invoices are issued on the first of each month for the previous billing cycle.",
    },
    {
        "id": "d04",
        "category": "billing",
        "priority": 1,
        "passage": "You can update your payment method under Billing then Payment methods.",
    },
    {
        "id": "d05",
        "category": "shipping",
        "priority": 2,
        "passage": "Standard shipping takes 3-5 business days within the continental US.",
    },
    {
        "id": "d06",
        "category": "shipping",
        "priority": 3,
        "passage": "International orders may incur customs duties calculated at checkout.",
    },
    {
        "id": "d07",
        "category": "returns",
        "priority": 1,
        "passage": "Returns are accepted within 30 days of delivery with a valid receipt.",
    },
    {
        "id": "d08",
        "category": "returns",
        "priority": 2,
        "passage": "Refunds are processed to the original payment method within 7 business days.",
    },
    {
        "id": "d09",
        "category": "auth",
        "priority": 3,
        "passage": "If you forgot your username, contact support with the email on file.",
    },
    {
        "id": "d10",
        "category": "shipping",
        "priority": 1,
        "passage": "Track your package status from the Orders page in your account.",
    },
    {
        "id": "d11",
        "category": "billing",
        "priority": 2,
        "passage": "Late payments incur a 2% monthly finance charge after 30 days overdue.",
    },
    {
        "id": "d12",
        "category": "auth",
        "priority": 1,
        "passage": "Password reset links expire after 24 hours for security reasons.",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def database() -> DatabaseModel:
    return DatabaseModel(
        project=os.environ["LAKEBASE_TEST_PROJECT"],
        client_id={
            "scope": os.environ["LAKEBASE_TEST_SECRET_SCOPE"],
            "secret": os.environ["LAKEBASE_TEST_CLIENT_ID_KEY"],
        },
        client_secret={
            "scope": os.environ["LAKEBASE_TEST_SECRET_SCOPE"],
            "secret": os.environ["LAKEBASE_TEST_CLIENT_SECRET_KEY"],
        },
        workspace_host={
            "scope": os.environ["LAKEBASE_TEST_SECRET_SCOPE"],
            "secret": os.environ["LAKEBASE_TEST_HOST_KEY"],
        },
    )


@pytest.fixture(scope="module")
def seeded_table(database: DatabaseModel) -> None:
    """Provision schema/table/indexes and seed embeddings once per module."""
    from databricks_langchain import DatabricksEmbeddings

    pool = PostgresPoolManager.get_pool(database)
    with pool.connection() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE")
            cur.execute("CREATE EXTENSION IF NOT EXISTS lakebase_text")
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA))
            )
            cur.execute(
                sql.SQL(
                    "CREATE TABLE IF NOT EXISTS {}.{} ("
                    "id TEXT PRIMARY KEY, "
                    "category TEXT, "
                    "priority INTEGER, "
                    "passage TEXT NOT NULL, "
                    "embedding VECTOR({dim}), "
                    "passage_tsv TSVECTOR GENERATED ALWAYS AS "
                    "  (to_tsvector('english', passage)) STORED)"
                ).format(
                    sql.Identifier(SCHEMA),
                    sql.Identifier(TABLE),
                    dim=sql.SQL(str(EMBEDDING_DIM)),
                )
            )
            # Ensure priority column exists even if the table was created by
            # a prior schema (idempotent — safe to re-run).
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS priority INTEGER"
                ).format(sql.Identifier(SCHEMA), sql.Identifier(TABLE))
            )
            cur.execute(
                sql.SQL(
                    "SELECT COUNT(*) AS n FROM {}.{} WHERE priority IS NOT NULL"
                ).format(sql.Identifier(SCHEMA), sql.Identifier(TABLE))
            )
            existing = int(cur.fetchone()["n"])

    if existing < len(SEED_DOCS):
        embeddings = DatabricksEmbeddings(endpoint=EMBEDDING_ENDPOINT)
        vectors = embeddings.embed_documents([d["passage"] for d in SEED_DOCS])
        with pool.connection() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                for doc, vec in zip(SEED_DOCS, vectors):
                    cur.execute(
                        sql.SQL(
                            "INSERT INTO {}.{} (id, category, priority, passage, embedding) "
                            "VALUES (%s, %s, %s, %s, %s::vector) "
                            "ON CONFLICT (id) DO UPDATE SET "
                            "category = EXCLUDED.category, "
                            "priority = EXCLUDED.priority, "
                            "passage = EXCLUDED.passage, "
                            "embedding = EXCLUDED.embedding"
                        ).format(sql.Identifier(SCHEMA), sql.Identifier(TABLE)),
                        (
                            doc["id"],
                            doc["category"],
                            doc["priority"],
                            doc["passage"],
                            vec,
                        ),
                    )

    with pool.connection() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {}.{} "
                    "USING lakebase_ann (embedding vector_cosine_ops)"
                ).format(
                    sql.Identifier(f"{TABLE}_embedding_ann"),
                    sql.Identifier(SCHEMA),
                    sql.Identifier(TABLE),
                )
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {}.{} "
                    "USING lakebase_bm25 (passage_tsv)"
                ).format(
                    sql.Identifier(f"{TABLE}_passage_tsv_bm25"),
                    sql.Identifier(SCHEMA),
                    sql.Identifier(TABLE),
                )
            )


@pytest.fixture()
def vector_store(
    database: DatabaseModel, seeded_table: None
) -> LakebaseVectorStoreModel:
    return LakebaseVectorStoreModel(
        database=database,
        schema_name=SCHEMA,
        table=TABLE,
        id_column="id",
        content_column="passage",
        embedding_column="embedding",
        tsvector_column="passage_tsv",
        embedding_model=EMBEDDING_ENDPOINT,
        metadata_columns=["category", "priority"],
        distance_metric="cosine",
    )


def _retriever(
    vs: LakebaseVectorStoreModel,
    query_type: str,
    filters: dict[str, Any] | None = None,
    k: int = 5,
) -> LakebaseRetriever:
    """Build a retriever with static suffixed-key filters (config-time shape).

    ``filters`` uses the ai_search-aligned dict form: keys are column names
    optionally suffixed with an operator (``"col LIKE"``, ``"col >="``);
    values may be scalars or lists (list ⇒ IN semantics).
    """
    return LakebaseRetriever(
        vector_store=vs,
        search_parameters=SearchParametersModel(
            query_type=query_type, num_results=k, filters=filters or {}
        ),
    )


# ---------------------------------------------------------------------------
# Query type coverage
# ---------------------------------------------------------------------------


class TestQueryTypes:
    """Every query_type against real Lakebase — verifies retrieval correctness."""

    def test_ann_returns_semantically_relevant_docs(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        docs = _retriever(vector_store, "ANN", k=3).invoke(
            "how do I reset a forgotten password?"
        )
        assert len(docs) == 3
        # Top 3 should be dominated by auth-category docs (password-related).
        top_categories = [d.metadata["category"] for d in docs]
        assert top_categories.count("auth") >= 2
        assert all("_distance" in d.metadata for d in docs)

    def test_bm25_matches_literal_terms(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        docs = _retriever(vector_store, "BM25", k=5).invoke("password reset")
        assert len(docs) >= 1
        # Top-ranked doc must contain at least one query term literally.
        # (BM25 fills the LIMIT even with weak matches on a small corpus, so
        # we assert on the winner rather than every returned row.)
        top = docs[0].page_content.lower()
        assert "password" in top or "reset" in top
        assert all("_score" in d.metadata for d in docs)

    def test_bm25_scores_are_negative(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        docs = _retriever(vector_store, "BM25", k=3).invoke("shipping")
        assert docs, "expected at least one BM25 hit for 'shipping'"
        # Lakebase BM25 returns negative scores (more negative = better).
        assert all(d.metadata["_score"] <= 0 for d in docs)

    def test_hybrid_fuses_both_signals(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        docs = _retriever(vector_store, "HYBRID", k=5).invoke("password reset")
        assert len(docs) >= 1
        # RRF must have populated a fused score.
        assert all("_rrf_score" in d.metadata for d in docs)
        # Top result should be a password/auth doc.
        assert docs[0].metadata["category"] == "auth"

    def test_hybrid_covers_both_lexical_and_semantic_hits(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        # "forgot" is not in the corpus verbatim; ANN should still surface d09.
        docs = _retriever(vector_store, "HYBRID", k=5).invoke("I forgot my login")
        ids = {d.metadata["id"] for d in docs}
        assert "d09" in ids  # "If you forgot your username, contact support..."


# ---------------------------------------------------------------------------
# Filter coverage
# ---------------------------------------------------------------------------


class TestFilters:
    """Live coverage for the suffix-based operator convention in ``_build_where``.

    Uses ai_search's suffixed-key dict shape (``"col LIKE"``, ``"col >="``,
    plain ``"col"`` for equality, list value for IN). The LLM-facing
    FilterItem list variant is exercised in :class:`TestToolFactory`.
    """

    def test_scalar_equality(self, vector_store: LakebaseVectorStoreModel) -> None:
        docs = _retriever(
            vector_store, "ANN", filters={"category": "billing"}, k=10
        ).invoke("payment")
        assert docs
        assert all(d.metadata["category"] == "billing" for d in docs)

    def test_in_via_list_value(self, vector_store: LakebaseVectorStoreModel) -> None:
        docs = _retriever(
            vector_store,
            "ANN",
            filters={"category": ["returns", "shipping"]},
            k=10,
        ).invoke("order")
        assert docs
        assert all(d.metadata["category"] in {"returns", "shipping"} for d in docs)

    def test_not_in_via_list_value_and_not_suffix(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        docs = _retriever(
            vector_store,
            "ANN",
            filters={"category NOT": ["auth"]},
            k=10,
        ).invoke("account")
        assert docs
        assert all(d.metadata["category"] != "auth" for d in docs)

    def test_comparison_greater_or_equal(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        docs = _retriever(
            vector_store,
            "ANN",
            filters={"priority >=": 2},
            k=10,
        ).invoke("service")
        assert docs
        assert all(d.metadata["priority"] >= 2 for d in docs)

    def test_comparison_less_than(self, vector_store: LakebaseVectorStoreModel) -> None:
        docs = _retriever(
            vector_store,
            "ANN",
            filters={"priority <": 2},
            k=10,
        ).invoke("service")
        assert docs
        assert all(d.metadata["priority"] < 2 for d in docs)

    def test_not_equal_via_not_suffix(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        docs = _retriever(
            vector_store,
            "ANN",
            filters={"category NOT": "billing"},
            k=10,
        ).invoke("customer")
        assert docs
        assert all(d.metadata["category"] != "billing" for d in docs)

    def test_like_is_case_insensitive(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """LIKE translates to ILIKE at the SQL layer — mixed-case pattern
        still matches lower-case data."""
        docs = _retriever(
            vector_store,
            "ANN",
            filters={"category LIKE": "SHIP%"},
            k=10,
        ).invoke("delivery")
        assert docs
        assert all(d.metadata["category"] == "shipping" for d in docs)

    def test_not_like_excludes_matches(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        docs = _retriever(
            vector_store,
            "ANN",
            filters={"category NOT LIKE": "auth%"},
            k=10,
        ).invoke("service")
        assert docs
        assert all(d.metadata["category"] != "auth" for d in docs)

    def test_filters_work_with_bm25(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        docs = _retriever(
            vector_store,
            "BM25",
            filters={"category": "auth"},
            k=5,
        ).invoke("password")
        assert docs
        assert all(d.metadata["category"] == "auth" for d in docs)

    def test_filters_work_with_hybrid(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        docs = _retriever(
            vector_store,
            "HYBRID",
            filters={"category": ["returns", "billing"]},
            k=5,
        ).invoke("get a refund")
        assert docs
        assert all(d.metadata["category"] in {"returns", "billing"} for d in docs)

    def test_unknown_column_rejected(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        with pytest.raises(ValueError, match="Unknown filter column"):
            _retriever(
                vector_store, "ANN", filters={"nonexistent_col": "x"}, k=3
            ).invoke("anything")

    def test_stage1_op_form_no_longer_supported(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """Stage-1 ``{op, value}`` dict values are no longer supported at
        the retriever layer — the shape is treated as an unknown scalar
        (which the DB then rejects). Regression guard so nobody
        accidentally puts the Stage-1 form back on a static config and
        gets a silent success."""
        # Retriever accepts the shape as an "equality against a dict" and
        # Postgres rejects at execute time. Assert we surface *some* error.
        with pytest.raises(Exception):
            _retriever(
                vector_store,
                "ANN",
                filters={"category": {"op": "in", "values": ["faq"]}},
                k=3,
            ).invoke("anything")


# ---------------------------------------------------------------------------
# Tool factory — live invocation
# ---------------------------------------------------------------------------


class TestToolFactory:
    """Verify the StructuredTool wrapper works end-to-end and returns JSON."""

    def test_tool_json_shape(self, vector_store: LakebaseVectorStoreModel) -> None:
        tool = create_lakebase_search_tool(
            retriever=LakebaseRetrieverModel(
                vector_store=vector_store,
                search_parameters=SearchParametersModel(
                    query_type="ANN", num_results=3
                ),
            )
        )
        raw = tool.invoke({"query": "how do I reset my password?"})
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        assert len(parsed) == 3
        for item in parsed:
            assert "page_content" in item
            assert "metadata" in item
            assert "id" in item["metadata"]

    def test_tool_hybrid_with_filter_item_list(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """LLM-facing FilterItem list is converted to a suffixed dict at the
        tool boundary and reaches Postgres correctly."""
        tool = create_lakebase_search_tool(
            retriever=LakebaseRetrieverModel(
                vector_store=vector_store,
                search_parameters=SearchParametersModel(
                    query_type="HYBRID", num_results=3
                ),
            )
        )
        raw = tool.invoke(
            {
                "query": "how do I get a refund?",
                "filters": [
                    {"key": "category", "value": ["returns", "billing"]},
                ],
            }
        )
        parsed = json.loads(raw)
        assert parsed
        assert all(
            item["metadata"]["category"] in {"returns", "billing"} for item in parsed
        )

    def test_tool_dynamic_schema_rejects_unlisted_key(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """The tool's per-tool args_schema narrows filters[].key to a
        Literal — an unlisted column is rejected by Pydantic before any
        Postgres call happens."""
        tool = create_lakebase_search_tool(
            retriever=LakebaseRetrieverModel(
                vector_store=vector_store,
                search_parameters=SearchParametersModel(
                    query_type="ANN", num_results=3
                ),
            )
        )
        with pytest.raises(Exception):
            tool.invoke(
                {
                    "query": "hi",
                    "filters": [{"key": "does_not_exist_col", "value": "x"}],
                }
            )

    def test_tool_like_is_case_insensitive(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """End-to-end via the tool: LIKE with mixed-case pattern matches
        lower-case data, proving LIKE→ILIKE flows through the whole stack."""
        tool = create_lakebase_search_tool(
            retriever=LakebaseRetrieverModel(
                vector_store=vector_store,
                search_parameters=SearchParametersModel(
                    query_type="ANN", num_results=10
                ),
            )
        )
        raw = tool.invoke(
            {
                "query": "delivery",
                "filters": [{"key": "category LIKE", "value": "SHIP%"}],
            }
        )
        parsed = json.loads(raw)
        assert parsed
        assert all(item["metadata"]["category"] == "shipping" for item in parsed)

    def test_tool_from_vector_store_dict(
        self, database: DatabaseModel, seeded_table: None
    ) -> None:
        """Bare vector_store dict is coerced through LakebaseRetrieverModel."""
        tool = create_lakebase_search_tool(
            vector_store={
                "database": database.model_dump(),
                "schema_name": SCHEMA,
                "table": TABLE,
                "content_column": "passage",
                "embedding_column": "embedding",
                "tsvector_column": "passage_tsv",
                "embedding_model": EMBEDDING_ENDPOINT,
                "metadata_columns": ["category", "priority"],
            }
        )
        raw = tool.invoke({"query": "password"})
        assert json.loads(raw)


# ---------------------------------------------------------------------------
# Rerank (PR A)
# ---------------------------------------------------------------------------


class TestRerank:
    """Live coverage for FlashRank reranking on lakebase_search output.

    FlashRank is a local cross-encoder — no serving endpoint required.
    First-time run downloads the model to `~/.dao_ai/cache/flashrank/`
    (or `/tmp/dao_ai/cache/flashrank` under Model Serving).
    """

    def test_reranker_score_populated_on_every_doc(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        retriever = LakebaseRetrieverModel(
            vector_store=vector_store,
            rerank=True,  # default FlashRank model
            search_parameters=SearchParametersModel(query_type="ANN", num_results=10),
        )
        tool = create_lakebase_search_tool(retriever=retriever)
        raw = tool.invoke({"query": "how do I reset my password?"})
        parsed = json.loads(raw)
        assert parsed
        for d in parsed:
            assert "reranker_score" in d["metadata"], d["metadata"]

    def test_top_n_truncates(self, vector_store: LakebaseVectorStoreModel) -> None:
        from dao_ai.config import RerankParametersModel

        retriever = LakebaseRetrieverModel(
            vector_store=vector_store,
            rerank=RerankParametersModel(model="ms-marco-MiniLM-L-12-v2", top_n=3),
            search_parameters=SearchParametersModel(query_type="ANN", num_results=10),
        )
        tool = create_lakebase_search_tool(retriever=retriever)
        raw = tool.invoke({"query": "password"})
        parsed = json.loads(raw)
        assert len(parsed) == 3, f"top_n=3 → 3 docs, got {len(parsed)}"

    def test_rerank_orders_by_score_descending(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        retriever = LakebaseRetrieverModel(
            vector_store=vector_store,
            rerank=True,
            search_parameters=SearchParametersModel(query_type="ANN", num_results=10),
        )
        tool = create_lakebase_search_tool(retriever=retriever)
        raw = tool.invoke({"query": "password reset"})
        parsed = json.loads(raw)
        assert len(parsed) >= 2
        scores = [d["metadata"]["reranker_score"] for d in parsed]
        # Non-increasing order.
        assert scores == sorted(scores, reverse=True), scores

    def test_rerank_promotes_topically_relevant_top_1(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """Query 'password reset' → the seeded doc explicitly about
        resetting passwords should rank #1 after FlashRank, even if the
        raw ANN top-1 was a related-but-less-direct doc."""
        retriever = LakebaseRetrieverModel(
            vector_store=vector_store,
            rerank=True,
            search_parameters=SearchParametersModel(query_type="ANN", num_results=10),
        )
        tool = create_lakebase_search_tool(retriever=retriever)
        raw = tool.invoke({"query": "how do I reset my password?"})
        parsed = json.loads(raw)
        assert parsed
        top_content = parsed[0]["page_content"].lower()
        # Seeded doc d01: "To reset your password, go to Settings..."
        # or d12: "Password reset links expire..." — either is a
        # topically direct hit.
        assert "reset" in top_content and "password" in top_content

    def test_rerank_works_with_bm25(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        retriever = LakebaseRetrieverModel(
            vector_store=vector_store,
            rerank=True,
            search_parameters=SearchParametersModel(query_type="BM25", num_results=5),
        )
        tool = create_lakebase_search_tool(retriever=retriever)
        raw = tool.invoke({"query": "password reset"})
        parsed = json.loads(raw)
        assert parsed
        for d in parsed:
            assert "reranker_score" in d["metadata"]

    def test_rerank_works_with_hybrid(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        retriever = LakebaseRetrieverModel(
            vector_store=vector_store,
            rerank=True,
            search_parameters=SearchParametersModel(query_type="HYBRID", num_results=5),
        )
        tool = create_lakebase_search_tool(retriever=retriever)
        raw = tool.invoke({"query": "reset my forgotten password"})
        parsed = json.loads(raw)
        assert parsed
        for d in parsed:
            assert "reranker_score" in d["metadata"]

    def test_rerank_preserves_document_metadata(
        self, vector_store: LakebaseVectorStoreModel
    ) -> None:
        """The rerank pass adds `reranker_score` on top of the retriever's
        existing metadata — id, category, and per-mode score/distance
        should still be there."""
        retriever = LakebaseRetrieverModel(
            vector_store=vector_store,
            rerank=True,
            search_parameters=SearchParametersModel(query_type="ANN", num_results=5),
        )
        tool = create_lakebase_search_tool(retriever=retriever)
        raw = tool.invoke({"query": "password"})
        parsed = json.loads(raw)
        assert parsed
        for d in parsed:
            m = d["metadata"]
            assert "id" in m, m
            assert "category" in m, m
            assert "_distance" in m, m  # ANN metric
            assert "reranker_score" in m, m
