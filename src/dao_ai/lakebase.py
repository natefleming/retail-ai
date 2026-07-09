"""User-facing Lakebase workflow helpers.

Standalone functions that operate on :class:`LakebaseVectorStoreModel`
configs. Kept separate from ``dao_ai.tools.lakebase_search`` (the
retriever tool factory) so data operations and tool construction don't
share a module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

if TYPE_CHECKING:
    from dao_ai.config import LakebaseVectorStoreModel


class Embedder(Protocol):
    """Structural interface for an embedder — matches LangChain's
    ``Embeddings`` base class, ``databricks_langchain.DatabricksEmbeddings``,
    ``langchain_openai.OpenAIEmbeddings``, and anything else that exposes
    ``embed_documents(list[str]) -> list[list[float]]``."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def backfill_embeddings(
    vector_store: "LakebaseVectorStoreModel",
    embedder: Embedder | None = None,
    *,
    batch_size: int = 128,
) -> int:
    """Populate the embedding column for every row where it is NULL.

    Reads unembedded rows via ``execute_query``, encodes them in chunks
    (default 128 rows per encode call), and writes vectors back via
    ``execute_many``. Idempotent — re-running only encodes rows that
    still have ``embedding IS NULL``. Safe to call after every seed /
    incremental insert.

    Args:
        vector_store: Config describing the table + column shape.
            ``vector_store.database`` provides the connection;
            ``vector_store.embedding_model.name`` is used when
            ``embedder`` is ``None`` (default).
        embedder: Object with ``embed_documents(list[str]) ->
            list[list[float]]``. Defaults to
            ``vector_store.embedding_model.as_embeddings_model()`` —
            the same endpoint the retriever uses at query time, so the
            two embedding directions stay in sync.
        batch_size: Rows per ``embed_documents`` call. Tune down for
            large-dimension models under memory pressure; tune up for
            small models to reduce round-trips.

    Returns:
        Number of rows updated.
    """
    if embedder is None:
        # Delegate to InferenceEndpointModel — same helper that other
        # dao-ai call sites use to materialize a LangChain Embeddings
        # from the endpoint config. Keeps the embedding endpoint choice
        # in one place (the vector_store.embedding_model field).
        embedder = vector_store.embedding_model.as_embeddings_model()

    schema_name: str = vector_store.schema_name
    table: str = vector_store.table
    id_column: str = vector_store.id_column
    content_column: str = vector_store.content_column
    embedding_column: str = vector_store.embedding_column
    qualified: str = f"{schema_name}.{table}"

    rows: list[dict[str, Any]] = vector_store.database.execute_query(
        f"SELECT {id_column} AS id, {content_column} AS content "
        f"FROM {qualified} "
        f"WHERE {embedding_column} IS NULL"
    )
    if not rows:
        logger.debug(
            "backfill_embeddings: no unembedded rows",
            table=qualified,
            column=embedding_column,
        )
        return 0

    update_sql: str = (
        f"UPDATE {qualified} "
        f"SET {embedding_column} = %s::vector "
        f"WHERE {id_column} = %s"
    )
    total: int = 0
    for start in range(0, len(rows), batch_size):
        chunk: list[dict[str, Any]] = rows[start : start + batch_size]
        vectors: list[list[float]] = embedder.embed_documents(
            [r["content"] for r in chunk]
        )
        vector_store.database.execute_many(
            update_sql,
            [(vec, r["id"]) for vec, r in zip(vectors, chunk)],
        )
        total += len(chunk)
    logger.info(
        "backfill_embeddings: encoded and updated {} rows",
        total,
        table=qualified,
        column=embedding_column,
    )
    return total
