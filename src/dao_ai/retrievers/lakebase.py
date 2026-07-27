"""Databricks Lakebase Postgres retriever.

Uses the ``lakebase_vector`` extension for ANN similarity search and the
``lakebase_text`` extension for BM25 lexical search, combining them
client-side with Reciprocal Rank Fusion for hybrid queries.

Connection pooling and Lakebase credential rotation are delegated to
``dao_ai.memory.postgres.PostgresPoolManager`` /
``AsyncPostgresPoolManager`` so this module never touches
``w.postgres.generate_database_credential(...)`` directly.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import mlflow
from databricks_langchain import DatabricksEmbeddings
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from loguru import logger
from mlflow.entities import SpanType
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import ConfigDict, PrivateAttr

from dao_ai.config import LakebaseVectorStoreModel, SearchParametersModel

_DISTANCE_OP = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}
_OPS_CLASS = {
    "cosine": "vector_cosine_ops",
    "l2": "vector_l2_ops",
    "ip": "vector_ip_ops",
}
_RRF_K = 60

# Filter-key suffixes we recognize on the dict keys reaching `_build_where`.
# These match ai_search's `_FILTER_OPERATOR_SUFFIXES` (with the leading
# space). Order matters — longest first so we don't misparse
# `"col NOT LIKE"` as `"col NOT" + " LIKE"` or similar.
_FILTER_SUFFIXES: tuple[str, ...] = (
    " NOT LIKE",
    " LIKE",
    " <=",
    " >=",
    " NOT",
    " <",
    " >",
)


class LakebaseSearchMode(str, Enum):
    ANN = "ANN"
    BM25 = "BM25"
    HYBRID = "HYBRID"


class LakebaseRetriever(BaseRetriever):
    """LangChain retriever over a Lakebase Postgres table."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: LakebaseVectorStoreModel
    search_parameters: SearchParametersModel = SearchParametersModel()

    _embeddings: Optional[Embeddings] = PrivateAttr(default=None)
    _allowed_filter_cols: frozenset[str] = PrivateAttr(default=frozenset())

    def model_post_init(self, __context: Any) -> None:
        metadata = self.vector_store.metadata_columns or []
        self._allowed_filter_cols = frozenset([*metadata, self.vector_store.id_column])

    # ------------------------------------------------------------------
    # LangChain BaseRetriever interface
    # ------------------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        mode = self._mode()
        filters = self._effective_filters()
        k = self._num_results()

        if mode is LakebaseSearchMode.ANN:
            rows = self._run_ann_sync(self._embed(query), filters, k)
            return [self._row_to_document(r) for r in rows]
        if mode is LakebaseSearchMode.BM25:
            rows = self._run_bm25_sync(query, filters, k)
            return [self._row_to_document(r) for r in rows]
        return self._run_hybrid_sync(query, filters, k)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        mode = self._mode()
        filters = self._effective_filters()
        k = self._num_results()

        if mode is LakebaseSearchMode.ANN:
            rows = await self._run_ann_async(self._embed(query), filters, k)
            return [self._row_to_document(r) for r in rows]
        if mode is LakebaseSearchMode.BM25:
            rows = await self._run_bm25_async(query, filters, k)
            return [self._row_to_document(r) for r in rows]
        return await self._run_hybrid_async(query, filters, k)

    # ------------------------------------------------------------------
    # Mode / config helpers
    # ------------------------------------------------------------------

    def _mode(self) -> LakebaseSearchMode:
        raw = (self.search_parameters.query_type or "ANN").upper()
        try:
            return LakebaseSearchMode(raw)
        except ValueError as e:
            raise ValueError(
                f"Unsupported query_type for Lakebase: {raw!r}. "
                "Expected one of: ANN, BM25, HYBRID."
            ) from e

    def _num_results(self) -> int:
        return int(self.search_parameters.num_results or 10)

    def _effective_filters(self) -> dict[str, Any]:
        return dict(self.search_parameters.filters or {})

    def _embed(self, query: str) -> list[float]:
        if self._embeddings is None:
            self._embeddings = DatabricksEmbeddings(
                endpoint=self.vector_store.embedding_model.name
            )
        return self._embeddings.embed_query(query)

    # ------------------------------------------------------------------
    # SQL builders (pure — testable without a live DB)
    # ------------------------------------------------------------------

    def _qualified_table(self) -> sql.Composed:
        return sql.SQL("{}.{}").format(
            sql.Identifier(self.vector_store.schema_name),
            sql.Identifier(self.vector_store.table),
        )

    def _select_columns(
        self, extra: list[sql.Composable] | None = None
    ) -> sql.Composed:
        vs = self.vector_store
        cols: list[sql.Composable] = [
            sql.Identifier(vs.id_column),
            sql.Identifier(vs.content_column),
        ]
        cols.extend(sql.Identifier(c) for c in (vs.metadata_columns or []))
        if extra:
            cols.extend(extra)
        return sql.SQL(", ").join(cols)

    def _parse_filter_key(self, key: str) -> tuple[str, str]:
        """Split ``"col SUFFIX"`` into ``(col, suffix)``.

        Suffix is one of ``_FILTER_SUFFIXES`` (with leading space) or ``""``
        for equality. Matches ai_search's operator-suffix convention on
        ``FilterItem.key`` so downstream tooling / prompts are portable.
        """
        for suffix in _FILTER_SUFFIXES:
            if key.endswith(suffix):
                return key[: -len(suffix)], suffix
        return key, ""

    def _build_where(self, filters: dict[str, Any]) -> tuple[sql.Composable, list[Any]]:
        """Translate a suffix-keyed filter dict into a WHERE fragment + params.

        Keys use ai_search's operator-suffix convention (``"col LIKE"``,
        ``"col >="``, ``"col NOT"``, plain ``"col"`` for equality). Values
        may be scalars or lists — lists trigger IN semantics on the ``""`` /
        ``" NOT"`` suffixes and are rejected on comparison / LIKE suffixes.

        ``LIKE`` / ``NOT LIKE`` translate to Postgres ``ILIKE`` /
        ``NOT ILIKE`` (case-insensitive is the agent-friendly default).

        Unknown columns and operators raise ``ValueError`` before any SQL is
        built. Column names are wrapped in ``sql.Identifier`` for safe
        quoting; values are always bound as ``%s`` parameters.
        """
        if not filters:
            return sql.SQL(""), []

        clauses: list[sql.Composable] = []
        params: list[Any] = []
        for raw_key, value in filters.items():
            col_name, suffix = self._parse_filter_key(raw_key)
            if col_name not in self._allowed_filter_cols:
                raise ValueError(
                    f"Unknown filter column {col_name!r} (from key {raw_key!r}); "
                    f"allowed: {sorted(self._allowed_filter_cols)}"
                )
            col = sql.Identifier(col_name)
            is_list = isinstance(value, list)

            if suffix == "":
                if is_list:
                    clauses.append(sql.SQL("{} = ANY(%s)").format(col))
                    params.append(list(value))
                else:
                    clauses.append(sql.SQL("{} = %s").format(col))
                    params.append(value)
            elif suffix == " NOT":
                if is_list:
                    clauses.append(sql.SQL("NOT ({} = ANY(%s))").format(col))
                    params.append(list(value))
                else:
                    clauses.append(sql.SQL("{} <> %s").format(col))
                    params.append(value)
            elif suffix in {" <", " <=", " >", " >="}:
                if is_list:
                    raise ValueError(
                        f"Comparison suffix {suffix.strip()!r} on {col_name!r} "
                        "does not support list values"
                    )
                op_sql = sql.SQL(suffix.strip())
                clauses.append(sql.SQL("{} {} %s").format(col, op_sql))
                params.append(value)
            elif suffix == " LIKE":
                if is_list:
                    raise ValueError(
                        f"LIKE suffix on {col_name!r} does not support list values"
                    )
                clauses.append(sql.SQL("{} ILIKE %s").format(col))
                params.append(value)
            elif suffix == " NOT LIKE":
                if is_list:
                    raise ValueError(
                        f"NOT LIKE suffix on {col_name!r} does not support list values"
                    )
                clauses.append(sql.SQL("NOT ({} ILIKE %s)").format(col))
                params.append(value)
            else:  # pragma: no cover — suffix set is closed
                raise ValueError(f"Unsupported filter suffix {suffix!r}")

        return sql.SQL(" AND ") + sql.SQL(" AND ").join(clauses), params

    def _build_ann_sql(
        self, filters: dict[str, Any], k: int
    ) -> tuple[sql.Composed, list[Any]]:
        vs = self.vector_store
        op = _DISTANCE_OP[vs.distance_metric]
        where_sql, where_params = self._build_where(filters)
        query = sql.SQL(
            "SELECT {cols}, ({emb} {op} %s::vector) AS _distance "
            "FROM {table} "
            "WHERE {emb} IS NOT NULL{where} "
            "ORDER BY {emb} {op} %s::vector "
            "LIMIT %s"
        ).format(
            cols=self._select_columns(),
            emb=sql.Identifier(vs.embedding_column),
            op=sql.SQL(op),
            table=self._qualified_table(),
            where=where_sql,
        )
        return query, [*where_params, k]  # embedding + limit prepended per-call

    def _build_bm25_sql(
        self, filters: dict[str, Any], k: int
    ) -> tuple[sql.Composed, list[Any]]:
        vs = self.vector_store
        if vs.tsvector_column is None or vs.bm25_index_name is None:
            raise ValueError(
                "BM25 search requires 'tsvector_column' and 'bm25_index_name' on the vector store."
            )
        where_sql, where_params = self._build_where(filters)
        score_expr = sql.SQL(
            "({tsv} <@> to_bm25query(to_tsvector(%s, %s), %s::regclass))"
        ).format(tsv=sql.Identifier(vs.tsvector_column))
        query = sql.SQL(
            "SELECT {cols}, {score} AS _score "
            "FROM {table} "
            "WHERE {tsv} IS NOT NULL{where} "
            "ORDER BY {score} ASC "
            "LIMIT %s"
        ).format(
            cols=self._select_columns(),
            score=score_expr,
            table=self._qualified_table(),
            tsv=sql.Identifier(vs.tsvector_column),
            where=where_sql,
        )
        return query, [*where_params, k]

    # ------------------------------------------------------------------
    # Execution — sync
    # ------------------------------------------------------------------

    def _sync_pool(self):
        from dao_ai.memory.postgres import PostgresPoolManager

        return PostgresPoolManager.get_pool(self.vector_store.database)

    @mlflow.trace(name="lakebase_ann_search", span_type=SpanType.RETRIEVER)
    def _run_ann_sync(
        self, query_vec: list[float], filters: dict[str, Any], k: int
    ) -> list[dict[str, Any]]:
        stmt, tail = self._build_ann_sql(filters, k)
        # ANN SQL binds the vector twice (SELECT distance + ORDER BY),
        # with WHERE params in between, and k as the final LIMIT bind.
        where_params = tail[:-1]
        params = [query_vec, *where_params, query_vec, k]
        pool = self._sync_pool()
        with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(stmt, params)
            return list(cur.fetchall())

    @mlflow.trace(name="lakebase_bm25_search", span_type=SpanType.RETRIEVER)
    def _run_bm25_sync(
        self, query_text: str, filters: dict[str, Any], k: int
    ) -> list[dict[str, Any]]:
        stmt, tail = self._build_bm25_sql(filters, k)
        where_params = tail[:-1]
        lang = self.vector_store.tsv_language
        idx = self.vector_store.bm25_index_name
        # score_expr appears twice (SELECT + ORDER BY) so bind 6 tsquery params total.
        pre = [lang, query_text, idx]
        params = [*pre, *where_params, *pre, k]
        pool = self._sync_pool()
        with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(stmt, params)
            return list(cur.fetchall())

    @mlflow.trace(name="lakebase_hybrid_search", span_type=SpanType.RETRIEVER)
    def _run_hybrid_sync(
        self, query: str, filters: dict[str, Any], k: int
    ) -> list[Document]:
        fetch_k = max(k * 2, 20)
        ann_rows = self._run_ann_sync(self._embed(query), filters, fetch_k)
        bm25_rows = self._run_bm25_sync(query, filters, fetch_k)
        return self._rrf(ann_rows, bm25_rows, k)

    # ------------------------------------------------------------------
    # Execution — async
    # ------------------------------------------------------------------

    async def _async_pool(self):
        from dao_ai.memory.postgres import AsyncPostgresPoolManager

        return await AsyncPostgresPoolManager.get_pool(self.vector_store.database)

    @mlflow.trace(name="lakebase_ann_search", span_type=SpanType.RETRIEVER)
    async def _run_ann_async(
        self, query_vec: list[float], filters: dict[str, Any], k: int
    ) -> list[dict[str, Any]]:
        stmt, tail = self._build_ann_sql(filters, k)
        where_params = tail[:-1]
        params = [query_vec, *where_params, query_vec, k]
        pool = await self._async_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(stmt, params)
                return list(await cur.fetchall())

    @mlflow.trace(name="lakebase_bm25_search", span_type=SpanType.RETRIEVER)
    async def _run_bm25_async(
        self, query_text: str, filters: dict[str, Any], k: int
    ) -> list[dict[str, Any]]:
        stmt, tail = self._build_bm25_sql(filters, k)
        where_params = tail[:-1]
        lang = self.vector_store.tsv_language
        idx = self.vector_store.bm25_index_name
        pre = [lang, query_text, idx]
        params = [*pre, *where_params, *pre, k]
        pool = await self._async_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(stmt, params)
                return list(await cur.fetchall())

    @mlflow.trace(name="lakebase_hybrid_search", span_type=SpanType.RETRIEVER)
    async def _run_hybrid_async(
        self, query: str, filters: dict[str, Any], k: int
    ) -> list[Document]:
        fetch_k = max(k * 2, 20)
        ann_rows = await self._run_ann_async(self._embed(query), filters, fetch_k)
        bm25_rows = await self._run_bm25_async(query, filters, fetch_k)
        return self._rrf(ann_rows, bm25_rows, k)

    # ------------------------------------------------------------------
    # Fusion + row mapping
    # ------------------------------------------------------------------

    def _rrf(
        self,
        ann_rows: list[dict[str, Any]],
        bm25_rows: list[dict[str, Any]],
        k: int,
    ) -> list[Document]:
        id_col = self.vector_store.id_column
        fused: dict[Any, dict[str, Any]] = {}
        scores: dict[Any, float] = {}
        for rank, row in enumerate(ann_rows, start=1):
            rid = row[id_col]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank)
            fused.setdefault(rid, row)
        for rank, row in enumerate(bm25_rows, start=1):
            rid = row[id_col]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (_RRF_K + rank)
            fused.setdefault(rid, row)
        ranked = sorted(fused.items(), key=lambda kv: scores[kv[0]], reverse=True)
        docs: list[Document] = []
        for rid, row in ranked[:k]:
            doc = self._row_to_document(row)
            doc.metadata["_rrf_score"] = scores[rid]
            docs.append(doc)
        return docs

    def _row_to_document(self, row: dict[str, Any]) -> Document:
        vs = self.vector_store
        page_content = row.get(vs.content_column, "") or ""
        raw_id = row.get(vs.id_column)
        doc_id = str(raw_id) if raw_id is not None else None
        metadata: dict[str, Any] = {vs.id_column: raw_id}
        for col in vs.metadata_columns or []:
            if col in row:
                metadata[col] = row[col]
        if "_distance" in row:
            metadata["_distance"] = row["_distance"]
        if "_score" in row:
            metadata["_score"] = row["_score"]
        logger.trace(
            "Lakebase document",
            id=doc_id,
            score=metadata.get("_score"),
            distance=metadata.get("_distance"),
        )
        return Document(id=doc_id, page_content=str(page_content), metadata=metadata)
