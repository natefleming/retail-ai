"""Factory for the ``lakebase_search`` first-class tool.

Wraps ``dao_ai.retrievers.LakebaseRetriever`` in a LangChain ``StructuredTool``.
Mirrors the shape of ``create_ai_search_tool`` (mutual exclusivity of
``retriever`` vs ``vector_store``, dict-to-model coercion) so both tools have
a consistent surface from YAML config.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import mlflow
from langchain_core.tools import StructuredTool
from loguru import logger
from mlflow.entities import SpanType
from pydantic import BaseModel, Field

from dao_ai.config import LakebaseRetrieverModel, LakebaseVectorStoreModel
from dao_ai.retrievers.lakebase import LakebaseRetriever


class LakebaseSearchInput(BaseModel):
    """Args exposed to the LLM for the ``lakebase_search`` tool."""

    query: str = Field(description="Natural-language search query.")
    filters: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional metadata filters keyed by column name. "
            "Values may be scalars (equality) or dicts of shape "
            "``{op: <=|>=|=|!=|<|>|in|not_in|like|ilike|is_null, value|values: ...}``."
        ),
    )


_DEFAULT_DESCRIPTION = (
    "Retrieve relevant documents from a Databricks Lakebase Postgres table using "
    "vector similarity (ANN), BM25 lexical search, or hybrid (RRF) — depending on "
    "the retriever's configured query_type."
)


def create_lakebase_search_tool(
    retriever: Optional[LakebaseRetrieverModel | dict[str, Any]] = None,
    vector_store: Optional[LakebaseVectorStoreModel | dict[str, Any]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> StructuredTool:
    """Build a Lakebase retrieval tool.

    Exactly one of ``retriever`` or ``vector_store`` must be supplied.
    Both accept dict literals (auto-coerced) for YAML friendliness.
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

    lb_retriever = LakebaseRetriever(
        vector_store=retriever_model.vector_store,
        search_parameters=retriever_model.search_parameters,
    )

    vs = retriever_model.vector_store
    tool_name = name or "lakebase_search"
    tool_description = description or _DEFAULT_DESCRIPTION

    @mlflow.trace(name=tool_name, span_type=SpanType.RETRIEVER)
    def _lakebase_search(
        query: str,
        filters: Optional[dict[str, Any]] = None,
    ) -> str:
        call_filters: dict[str, Any] = {
            **(retriever_model.search_parameters.filters or {}),
            **(filters or {}),
        }
        # Merge without mutating the config's static filters.
        effective_params = retriever_model.search_parameters.model_copy(
            update={"filters": call_filters}
        )
        lb_retriever.search_parameters = effective_params

        docs = lb_retriever.invoke(query)
        serialized = [
            {"page_content": d.page_content, "metadata": _jsonable(d.metadata)}
            for d in docs
        ]
        return json.dumps(serialized)

    tool = StructuredTool.from_function(
        func=_lakebase_search,
        name=tool_name,
        description=tool_description,
        args_schema=LakebaseSearchInput,
    )

    logger.success(
        "Lakebase search tool created",
        name=tool_name,
        schema=vs.schema_name,
        table=vs.table,
        query_type=retriever_model.search_parameters.query_type,
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
