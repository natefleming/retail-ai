"""Factory for the ``lakebase_search`` first-class tool.

Wraps ``dao_ai.retrievers.LakebaseRetriever`` in a LangChain ``StructuredTool``.
Mirrors ``create_ai_search_tool``: mutual exclusivity of ``retriever`` vs
``vector_store``, dict-to-model coercion, a `list[FilterItem]` runtime
argument with a per-tool `Literal`-narrowed key enum, and Mode A / B / C
column discovery (hand-declared / declared + Postgres enrichment / empty +
Postgres discovery).
"""

from __future__ import annotations

import json
from typing import Any, Optional

import mlflow
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from loguru import logger
from mlflow.entities import SpanType
from psycopg import sql
from psycopg.rows import dict_row
from pydantic import BaseModel, Field, create_model

from dao_ai.config import (
    ColumnInfo,
    FilterItem,
    LakebaseRetrieverModel,
    LakebaseVectorStoreModel,
    RerankParametersModel,
)
from dao_ai.retrievers.lakebase import LakebaseRetriever
from dao_ai.tools.vector_search import (
    _build_filter_item_model,
    _is_array_type,
    _normalize_declared_columns,
    build_flashrank_ranker,
    rerank_documents,
)


# Rough map from Postgres ``information_schema.columns.data_type`` values
# to the labels ``ColumnInfo.type`` uses. Anything we can't classify falls
# through as a bare "string" — matches ai_search's behavior when UC can't
# tell us.
_PG_TYPE_TO_LABEL: dict[str, str] = {
    "text": "string",
    "character varying": "string",
    "varchar": "string",
    "character": "string",
    "char": "string",
    "uuid": "string",
    "integer": "number",
    "bigint": "number",
    "smallint": "number",
    "numeric": "number",
    "real": "number",
    "double precision": "number",
    "boolean": "boolean",
    "date": "datetime",
    "timestamp": "datetime",
    "timestamp without time zone": "datetime",
    "timestamp with time zone": "datetime",
    "time": "datetime",
    "time without time zone": "datetime",
    "time with time zone": "datetime",
    "ARRAY": "array",
}


class LakebaseSearchInput(BaseModel):
    """Args exposed to the LLM for the ``lakebase_search`` tool.

    ``filters`` uses the same shape as ``ai_search``: a JSON array of
    ``{key, value}`` objects, where ``key`` is a column name optionally
    suffixed with an operator (``"col"``, ``"col NOT"``, ``"col <="``,
    ``"col LIKE"``, ``"col NOT LIKE"``, etc.). This class is the free-form
    fallback — the factory replaces it with a per-tool ``Literal``-narrowed
    subclass whenever it can enumerate the table's columns.
    """

    query: str = Field(description="Natural-language search query.")
    filters: Optional[list[FilterItem]] = Field(
        default=None,
        description=(
            "Optional metadata filters. Pass a JSON array of objects, each "
            "with 'key' and 'value'. Do NOT pass a flat dict. Example: "
            '[{"key": "category", "value": "faq"}, '
            '{"key": "priority >=", "value": 2}, '
            '{"key": "status", "value": ["published", "reviewed"]}]. '
            "The 'key' field is a column name optionally suffixed with an "
            "operator: (none) equality, 'NOT' not-equals / not-in, "
            "'< <= > >=' comparisons, 'LIKE' case-insensitive pattern "
            "match, 'NOT LIKE' case-insensitive pattern exclude. "
            "Omit or set to null when no filter applies."
        ),
    )


_DEFAULT_DESCRIPTION = (
    "Retrieve relevant documents from a Databricks Lakebase Postgres table using "
    "vector similarity (ANN), BM25 lexical search, or hybrid (RRF) — depending on "
    "the retriever's configured query_type."
)


def _build_lakebase_search_input_model(
    columns: list[str],
    operator_overrides: dict[str, list[str]] | None = None,
) -> type[BaseModel]:
    """Return a per-tool ``LakebaseSearchInput`` whose ``filters[].key`` is
    ``Literal``-narrowed to the actual column × suffix cross-product.

    When ``columns`` is empty the free-form ``LakebaseSearchInput`` is
    returned unchanged — matches the ai_search fallback path.
    """
    if not columns:
        return LakebaseSearchInput

    filter_item_cls = _build_filter_item_model(columns, operator_overrides)
    return create_model(
        "DynamicLakebaseSearchInput",
        __base__=LakebaseSearchInput,
        __module__=__name__,
        filters=(
            Optional[list[filter_item_cls]],  # type: ignore[valid-type]
            Field(
                default=None,
                description=(
                    "Optional metadata filters. Pass a JSON array of objects, "
                    "each with 'key' and 'value'. The 'key' field is "
                    "enumerated to the actual table columns × operator "
                    "suffixes — an unlisted key is rejected. Omit or set to "
                    "null when no filter applies."
                ),
            ),
        ),
    )


def _fetch_lakebase_columns(
    vector_store: LakebaseVectorStoreModel,
) -> list[tuple[str, str | None, str | None]] | None:
    """Return ``[(name, type_label, comment)]`` from Postgres.

    Postgres analogue of ``_fetch_index_columns`` in vector_search.py. One
    round-trip via the shared pool. Columns used internally by the retriever
    (id / content / embedding / tsvector) are stripped so they don't appear
    on the LLM-facing filter enum.

    ``type_label`` is one of the ``ColumnInfo.type`` labels
    (``"string" | "number" | "boolean" | "datetime" | "array"``) when
    classifiable, or ``None`` when we can't map it. ``comment`` is the
    Postgres column comment (``col_description(...)``) or ``None``.

    Returns ``None`` on any failure — caller falls back to declared
    ``metadata_columns`` if present, otherwise free-form filters.
    """
    from dao_ai.memory.postgres import PostgresPoolManager

    reserved = {
        vector_store.id_column,
        vector_store.content_column,
        vector_store.embedding_column,
    }
    if vector_store.tsvector_column:
        reserved.add(vector_store.tsvector_column)

    try:
        pool = PostgresPoolManager.get_pool(vector_store.database)
        with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT column_name, data_type, "
                "col_description("
                "(quote_ident(table_schema)||'.'||quote_ident(table_name))::regclass, "
                "ordinal_position) AS comment "
                "FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = %s "
                "ORDER BY ordinal_position",
                (vector_store.schema_name, vector_store.table),
            )
            rows = cur.fetchall()
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "Postgres information_schema lookup failed; column auto-discovery unavailable",
            schema=vector_store.schema_name,
            table=vector_store.table,
            error=f"{type(e).__name__}: {e}",
        )
        return None

    out: list[tuple[str, str | None, str | None]] = []
    for row in rows:
        name = row["column_name"]
        if not name or name in reserved:
            continue
        pg_type = row.get("data_type")
        label = _PG_TYPE_TO_LABEL.get(pg_type) if pg_type else None
        comment = row.get("comment")
        out.append((name, label, str(comment) if comment else None))
    return out or None


def create_lakebase_search_tool(
    retriever: Optional[LakebaseRetrieverModel | dict[str, Any]] = None,
    vector_store: Optional[LakebaseVectorStoreModel | dict[str, Any]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> StructuredTool:
    """Build a Lakebase retrieval tool.

    Exactly one of ``retriever`` or ``vector_store`` must be supplied.
    Both accept dict literals (auto-coerced) for YAML friendliness.

    The returned tool exposes a `list[FilterItem]` filter argument with a
    per-tool `Literal`-narrowed `key` enum when the table's columns can be
    resolved (Mode A hand-declared / Mode B declared + Postgres enrichment /
    Mode C empty + Postgres discovery — same shape as ``create_ai_search_tool``).
    """
    if retriever is None and vector_store is None:
        raise ValueError(
            "create_lakebase_search_tool requires either 'retriever' or 'vector_store'."
        )
    if retriever is not None and vector_store is not None:
        raise ValueError(
            "create_lakebase_search_tool cannot accept both 'retriever' and 'vector_store'."
        )

    if vector_store is not None:
        if isinstance(vector_store, dict):
            vector_store = LakebaseVectorStoreModel(**vector_store)
        retriever_model = LakebaseRetrieverModel(vector_store=vector_store)
    else:
        if isinstance(retriever, dict):
            retriever_model = LakebaseRetrieverModel(**retriever)
        else:
            retriever_model = retriever

    vs = retriever_model.vector_store
    lb_retriever = LakebaseRetriever(
        vector_store=vs,
        search_parameters=retriever_model.search_parameters,
    )

    tool_name = name or "lakebase_search"
    tool_description = description or _DEFAULT_DESCRIPTION

    # --- Column source of truth — three modes, matching create_ai_search_tool
    #
    #   A. Hand-declared. Any item in retriever.columns is a ColumnInfo.
    #      Names, types, descriptions, and per-column operator overrides
    #      come straight from declaration. No Postgres call.
    #   B. Bare strings only. Names from the declared list; Postgres
    #      information_schema.columns is called best-effort for type +
    #      description enrichment (used to auto-narrow ARRAY columns and
    #      colour the tool description).
    #   C. Empty. Discover names via information_schema.columns; soft-fall-
    #      back to vector_store.metadata_columns if the query fails.
    declared_items: list[Any] = list(
        retriever_model.columns or vs.metadata_columns or []
    )
    (
        declared_names,
        declared_types,
        declared_descriptions,
        operator_overrides_raw,
        any_hand_declared,
    ) = _normalize_declared_columns(declared_items)
    operator_overrides: dict[str, list[str]] = dict(operator_overrides_raw)

    columns: list[str]
    description_types: dict[str, str] = dict(declared_types)
    description_descriptions: dict[str, str] = dict(declared_descriptions)

    if any_hand_declared:
        columns = declared_names
    elif declared_names:
        columns = declared_names
        pg_cols = _fetch_lakebase_columns(vs)
        if pg_cols:
            pg_type_map = {n: t for n, t, _ in pg_cols if t}
            pg_comment_map = {n: c for n, _, c in pg_cols if c}
            for n in declared_names:
                if n not in description_types and n in pg_type_map:
                    description_types[n] = pg_type_map[n]
                if n not in description_descriptions and n in pg_comment_map:
                    description_descriptions[n] = pg_comment_map[n]
    else:
        pg_cols = _fetch_lakebase_columns(vs)
        if pg_cols:
            columns = [n for n, _, _ in pg_cols]
            for n, t, c in pg_cols:
                if t and n not in description_types:
                    description_types[n] = t
                if c and n not in description_descriptions:
                    description_descriptions[n] = c
        else:
            columns = list(vs.metadata_columns or [])

    # Array-typed columns get equality-only, unless the user set operators
    # explicitly via ColumnInfo (which _normalize_declared_columns records
    # in operator_overrides). Matches the ai_search behaviour.
    for col_name, type_str in description_types.items():
        if col_name in operator_overrides:
            continue
        if _is_array_type(type_str):
            operator_overrides[col_name] = [""]

    schema_cls = _build_lakebase_search_input_model(
        columns, operator_overrides or None
    )

    logger.debug(
        "Lakebase search columns resolved",
        schema=vs.schema_name,
        table=vs.table,
        mode="hand-declared"
        if any_hand_declared
        else ("declared" if declared_names else "discovered"),
        columns=columns,
        overrides=operator_overrides,
    )

    # Optional FlashRank ranker — parity with ai_search. Reuses the shared
    # helper from vector_search.py so both backends share the same init +
    # ONNX-compatibility patching. Returns None when rerank is unset OR
    # when init failed (soft-fail); the downstream "if ranker" guard
    # handles both uniformly.
    rerank_config = (
        retriever_model.rerank
        if isinstance(retriever_model.rerank, RerankParametersModel)
        else None
    )
    ranker = build_flashrank_ranker(rerank_config)

    @mlflow.trace(name=tool_name, span_type=SpanType.RETRIEVER)
    def _lakebase_search(
        query: str,
        filters: Optional[list[FilterItem]] = None,
    ) -> str:
        # Convert LLM-provided FilterItem list → suffixed dict (matches
        # ai_search's boundary conversion at vector_search.py:1303-1306).
        runtime_dict: dict[str, Any] = {}
        if filters:
            for item in filters:
                runtime_dict[item.key] = item.value

        static_dict = dict(retriever_model.search_parameters.filters or {})
        merged: dict[str, Any] = {**static_dict, **runtime_dict}

        # If instructed retrieval is configured, route through the shared
        # pipeline (router → decompose → parallel → RRF → FlashRank →
        # instruction rerank → verifier). Fast path below runs the raw
        # retriever + optional FlashRank when no instructed config is set.
        if retriever_model.instructed is not None:
            from dao_ai.tools.instructed_pipeline import (
                execute_instructed_pipeline,
            )

            # Backend adapter — fresh LakebaseRetriever per subquery so
            # concurrent ThreadPoolExecutor calls inside the pipeline are
            # thread-safe (no shared search_parameters mutation).
            def _run_search(qtxt: str, flt: dict[str, Any]) -> list[Document]:
                sub_params = retriever_model.search_parameters.model_copy(
                    update={"filters": flt or {}}
                )
                sub_retriever = LakebaseRetriever(
                    vector_store=vs,
                    search_parameters=sub_params,
                )
                return list(sub_retriever.invoke(qtxt))

            inst = retriever_model.instructed
            docs = execute_instructed_pipeline(
                run_search=_run_search,
                query=query,
                base_filters=merged,
                instructed_config=inst,
                router_config=inst.router,
                verifier_config=inst.verifier,
                decomposition_config=inst.decomposition,
                instruction_rerank_config=inst.rerank,
                instructed_columns=inst.columns,
                primary_key=vs.id_column,
                ranker=ranker,
                rerank_config=rerank_config,
            )
        else:
            # Fast path — no instructed pipeline. Mutation of
            # search_parameters is safe here since this thread is the only
            # caller (unlike the ThreadPoolExecutor fan-out above).
            effective_params = retriever_model.search_parameters.model_copy(
                update={"filters": merged}
            )
            lb_retriever.search_parameters = effective_params

            docs = lb_retriever.invoke(query)

            # Optional FlashRank cross-encoder pass. Same helper
            # ai_search uses — reranker_score lands in each
            # Document.metadata.
            if ranker is not None and rerank_config is not None and docs:
                logger.debug("Applying FlashRank reranking to Lakebase results")
                docs = rerank_documents(query, docs, ranker, rerank_config)

        serialized = [
            {"page_content": d.page_content, "metadata": _jsonable(d.metadata)}
            for d in docs
        ]
        return json.dumps(serialized)

    tool = StructuredTool.from_function(
        func=_lakebase_search,
        name=tool_name,
        description=tool_description,
        args_schema=schema_cls,
    )

    logger.success(
        "Lakebase search tool created",
        name=tool_name,
        schema=vs.schema_name,
        table=vs.table,
        query_type=retriever_model.search_parameters.query_type,
        filterable_columns=columns,
    )
    return tool


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of non-JSON-serialisable metadata values."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if hasattr(value, "isoformat"):  # datetime/date
        return value.isoformat()
    return value
