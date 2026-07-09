"""Backend-agnostic instructed-retrieval pipeline.

Extracted from ``create_ai_search_tool``'s closure so ``lakebase_search``
(and any future retriever) can share the exact same pipeline logic:

    router mode selection
      → (standard search | instructed: decompose → parallel → RRF)
        → FlashRank reranking
          → instruction-aware LLM reranking
            → verifier retry loop

The only backend-coupled piece is the search call. Callers pass a
``run_search: Callable[[str, dict[str, Any]], list[Document]]`` adapter
that encapsulates their backend (AI Search's ``vs.similarity_search`` /
Lakebase's mode-specific ``_run_*_sync``), including ``k``, ``query_type``,
embedding, and any other retriever-specific state.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING

import mlflow
from flashrank import Ranker
from langchain_core.documents import Document
from loguru import logger
from mlflow.entities import SpanType

from dao_ai.config import (
    ColumnInfo,
    DecompositionModel,
    InstructedRetrieverModel,
    InstructionAwareRerankModel,
    RerankParametersModel,
    RouterModel,
    SearchQuery,
    VerifierModel,
)
from dao_ai._tracing import in_caller_context
from dao_ai.tools.instructed_retriever import (
    _get_cached_llm,
    decompose_query,
    rrf_merge,
)
from dao_ai.tools.instruction_reranker import instruction_aware_rerank
from dao_ai.tools.router import route_query
from dao_ai.tools.tracing import (
    ATTR_ROUTER_BYPASSED,
    ATTR_ROUTER_FALLBACK,
    ATTR_ROUTER_MODE,
    ATTR_VERIFIER_OUTCOME,
    ATTR_VERIFIER_RETRIES,
    ResourceInfo,
)
from dao_ai.tools.verifier import add_verification_metadata, verify_results

if TYPE_CHECKING:
    from dao_ai.state import Context


# Type alias — the backend adapter callable. Given a query string and a
# suffixed-key filter dict, return the top-k documents from the backend.
# The retriever tool captures k, query_type, embedding, etc. in the closure.
RunSearch = Callable[[str, dict[str, Any]], list[Document]]


Mode = Literal["standard", "instructed"]


# Base column name from a suffixed filter key ("priority >=" → "priority").
# Suffixes are exactly the ones defined by _FILTER_OPERATOR_SUFFIXES in
# vector_search.py — kept as a local tuple to avoid a cross-module import.
_FILTER_KEY_SUFFIXES: tuple[str, ...] = (
    " NOT LIKE",
    " LIKE",
    " NOT",
    " <=",
    " >=",
    " <",
    " >",
)


def _base_column_name(key: str) -> str:
    """Strip a trailing operator suffix from a filter key."""
    for suffix in _FILTER_KEY_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def _coerce_scalar(value: Any, col_type: str) -> Any:
    """Coerce a single scalar to the column's declared type.

    Raises ValueError on failure so the caller can drop-and-warn. Returns
    the value untouched when it's already type-compatible or when
    ``col_type`` is ``string`` / ``array`` (no coercion needed).
    """
    from datetime import datetime

    if col_type == "string" or col_type == "array":
        return value
    if col_type == "number":
        if isinstance(value, bool):
            raise ValueError(f"bool {value!r} not accepted as number")
        if isinstance(value, (int, float)):
            return value
        s = str(value).strip()
        # Prefer int when the string has no fractional part.
        try:
            return int(s)
        except ValueError:
            pass
        return float(s)  # raises ValueError on bad input
    if col_type == "boolean":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
        raise ValueError(f"{value!r} not a boolean")
    if col_type == "datetime":
        if isinstance(value, datetime):
            return value.isoformat()
        # Accept anything datetime.fromisoformat can parse; keep the
        # canonical ISO string on the wire so backends can render it into
        # SQL / VS filter unchanged.
        datetime.fromisoformat(str(value))
        return str(value)
    return value


def coerce_filter_values(
    filters: dict[str, Any],
    columns: list["ColumnInfo"],
) -> dict[str, Any]:
    """Coerce LLM-emitted filter values to their column-declared type.

    Trims a common LLM failure mode: decomposition emits
    ``{"priority": "high"}`` on an integer column, Postgres 500s. Coerces
    strings to ``int`` / ``float`` / ``bool`` / ``datetime`` per
    ``ColumnInfo.type``; on failure drops the entry and logs a warning
    tagged ``dao_ai.filter.coercion_failed`` so the query still runs
    (degraded — one filter fewer — rather than hard-failing the tool call).

    Suffixed keys (``"priority >="``) resolve to the base column via
    :func:`_base_column_name`. Unknown columns pass through untouched
    (the backend will reject them if invalid). List values coerce
    element-wise; the whole entry is dropped if any element fails.
    """
    if not filters or not columns:
        return filters
    by_name: dict[str, ColumnInfo] = {c.name: c for c in columns}
    coerced: dict[str, Any] = {}
    for key, value in filters.items():
        col = by_name.get(_base_column_name(key))
        if col is None:
            coerced[key] = value
            continue
        try:
            if isinstance(value, list):
                coerced[key] = [_coerce_scalar(v, col.type) for v in value]
            else:
                coerced[key] = _coerce_scalar(value, col.type)
        except (ValueError, TypeError) as e:
            logger.warning(
                "dao_ai.filter.coercion_failed | col={} type={} value={!r} err={}: {} — dropping filter",
                col.name,
                col.type,
                value,
                type(e).__name__,
                e,
            )
    return coerced


def _normalize_filter_values(
    filters: dict[str, Any], case: Optional[str]
) -> dict[str, Any]:
    """Normalize string filter values to specified case (uppercase/lowercase).

    Lifted verbatim from ``create_ai_search_tool`` — used when
    ``DecompositionModel.normalize_filter_case`` is set. No case normalization
    is applied when ``case`` is ``None``.
    """
    if not case or not filters:
        return filters
    normalized: dict[str, Any] = {}
    for key, value in filters.items():
        if isinstance(value, str):
            normalized[key] = value.upper() if case == "uppercase" else value.lower()
        elif isinstance(value, list):
            normalized[key] = [
                v.upper() if case == "uppercase" else v.lower() if isinstance(v, str) else v
                for v in value
            ]
        else:
            normalized[key] = value
    return normalized


@mlflow.trace(name="execute_instructed_retrieval", span_type=SpanType.RETRIEVER)
def _execute_instructed_search(
    *,
    run_search: RunSearch,
    query: str,
    base_filters: dict[str, Any],
    instructed_config: InstructedRetrieverModel,
    decomposition_config: DecompositionModel,
    instructed_columns: list[ColumnInfo],
    primary_key: Optional[str],
    previous_feedback: Optional[str] = None,
    context: Optional["Context"] = None,
) -> list[Document]:
    """Decompose → parallel search → RRF merge.

    Backend-agnostic — takes the backend adapter as ``run_search``. Falls
    back to a single unfiltered ``run_search(query, base_filters)`` call
    when decomposition yields no subqueries, all subqueries return empty
    results, or any exception is raised inside decomposition / merge.

    Matches the shape of the closure that used to live inside
    ``create_ai_search_tool`` at ``vector_search.py:1094-1254``.
    """
    logger.trace(
        "Executing instructed retrieval", query=query, base_filters=base_filters
    )
    try:
        decomposition_llm = _get_cached_llm(decomposition_config.model, context)

        subqueries: list[SearchQuery] = decompose_query(
            llm=decomposition_llm,
            query=query,
            columns=instructed_columns,
            constraints=instructed_config.constraints,
            max_subqueries=decomposition_config.max_subqueries,
            examples=decomposition_config.examples,
            previous_feedback=previous_feedback,
            resource_info=ResourceInfo(
                "model_serving",
                decomposition_config.model.on_behalf_of_user,
                decomposition_config.model.name,
            ),
        )

        if not subqueries:
            logger.warning(
                "Query decomposition returned no subqueries, using original"
            )
            return run_search(query, base_filters)

        normalized_base_filters = _normalize_filter_values(
            base_filters, decomposition_config.normalize_filter_case
        )

        def execute_search(sq: SearchQuery) -> list[Document]:
            logger.trace("Executing search", query=sq.text, filters=sq.filters)
            sq_filters_dict: dict[str, Any] = {}
            if sq.filters:
                for item in sq.filters:
                    sq_filters_dict[item.key] = item.value
            sq_filters_dict = coerce_filter_values(
                sq_filters_dict, instructed_columns
            )
            sq_filters = _normalize_filter_values(
                sq_filters_dict, decomposition_config.normalize_filter_case
            )
            # Decomposed filters take precedence over base filters
            combined_filters: dict[str, Any] = {
                **normalized_base_filters,
                **sq_filters,
            }
            logger.trace(
                "Executing search",
                query=sq.text,
                filters=combined_filters,
            )
            return run_search(sq.text, combined_filters)

        logger.debug(
            "Executing parallel searches",
            num_subqueries=len(subqueries),
            queries=[sq.text[:50] for sq in subqueries],
        )

        with ThreadPoolExecutor(
            max_workers=decomposition_config.max_subqueries
        ) as executor:
            # Wrap once so per-subquery threads inherit the parent
            # MLflow active-span ContextVar (span nesting).
            all_results = list(
                executor.map(in_caller_context(execute_search), subqueries)
            )

        merged = rrf_merge(
            all_results,
            k=decomposition_config.rrf_k,
            primary_key=primary_key,
        )

        logger.debug(
            "Instructed retrieval complete",
            num_subqueries=len(subqueries),
            total_results=sum(len(r) for r in all_results),
            merged_results=len(merged),
        )

        if not merged:
            logger.warning(
                "All instructed subqueries returned empty results, "
                "falling back to standard unfiltered search",
                num_subqueries=len(subqueries),
            )
            return run_search(query, base_filters)

        return merged

    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Instructed retrieval failed, falling back to standard search",
            error=str(e),
        )
        return run_search(query, base_filters)


@mlflow.trace(name="apply_post_processing", span_type=SpanType.RETRIEVER)
def _apply_post_processing(
    *,
    documents: list[Document],
    query: str,
    mode: Mode,
    auto_bypass: bool,
    instructed_config: Optional[InstructedRetrieverModel],
    instruction_rerank_config: Optional[InstructionAwareRerankModel],
    instructed_columns: list[ColumnInfo],
    context: Optional["Context"] = None,
) -> list[Document]:
    """Instruction-aware LLM reranking (post-FlashRank).

    Skipped when ``mode == "standard"`` and ``auto_bypass`` is true — that's
    the router's opt-out for simple queries where LLM re-scoring adds no
    value. Lifted from ``vector_search.py:1271-1312``.
    """
    if mode == "standard" and auto_bypass:
        span = mlflow.get_current_active_span()
        if span:
            span.set_attribute(ATTR_ROUTER_BYPASSED, True)
        return documents

    if instruction_rerank_config and instructed_config:
        instruction_llm = (
            _get_cached_llm(instruction_rerank_config.model, context)
            if instruction_rerank_config.model
            else None
        )
        if instruction_llm:
            documents = instruction_aware_rerank(
                llm=instruction_llm,
                query=query,
                documents=documents,
                instructions=instruction_rerank_config.instructions,
                columns=instructed_columns,
                top_n=instruction_rerank_config.top_n,
                resource_info=ResourceInfo(
                    "model_serving",
                    instruction_rerank_config.model.on_behalf_of_user,
                    instruction_rerank_config.model.name,
                )
                if instruction_rerank_config.model
                else None,
            )
    return documents


def _decide_mode(
    *,
    query: str,
    router_config: Optional[RouterModel],
    instructed_config: Optional[InstructedRetrieverModel],
    instructed_columns: list[ColumnInfo],
    context: Optional["Context"] = None,
) -> tuple[Mode, bool]:
    """Route the query to ``"standard"`` or ``"instructed"``.

    Returns ``(mode, auto_bypass)``. When there's no router but instructed
    config exists, mode is ``"instructed"`` with ``auto_bypass=False``.
    When neither is set, mode is ``"standard"`` with ``auto_bypass=True``.
    """
    mode: Mode = "standard"
    auto_bypass = True

    if router_config:
        router_llm = (
            _get_cached_llm(router_config.model, context)
            if router_config.model
            else None
        )
        auto_bypass = router_config.auto_bypass

        if router_llm and instructed_config:
            try:
                mode = route_query(
                    llm=router_llm,
                    query=query,
                    columns=instructed_columns,
                    resource_info=ResourceInfo(
                        "model_serving",
                        router_config.model.on_behalf_of_user,
                        router_config.model.name,
                    ),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Router failed, defaulting to configured default", error=str(e)
                )
                span = mlflow.get_current_active_span()
                if span:
                    span.set_attribute(ATTR_ROUTER_FALLBACK, True)
                mode = router_config.default_mode
        else:
            mode = router_config.default_mode
    elif instructed_config:
        # No router but instructed config is set — always run instructed.
        mode = "instructed"
        auto_bypass = False

    return mode, auto_bypass


def execute_instructed_pipeline(
    *,
    run_search: RunSearch,
    query: str,
    base_filters: dict[str, Any],
    instructed_config: Optional[InstructedRetrieverModel],
    router_config: Optional[RouterModel],
    verifier_config: Optional[VerifierModel],
    decomposition_config: Optional[DecompositionModel],
    instruction_rerank_config: Optional[InstructionAwareRerankModel],
    instructed_columns: list[ColumnInfo],
    primary_key: Optional[str] = None,
    ranker: Optional[Ranker] = None,
    rerank_config: Optional[RerankParametersModel] = None,
    context: Optional["Context"] = None,
) -> list[Document]:
    """Backend-agnostic full instructed-retrieval pipeline.

    Orchestrates: router mode selection → (standard | instructed search) →
    FlashRank cross-encoder rerank → instruction-aware LLM rerank →
    verifier retry loop.

    ``run_search`` is the only backend-coupled input. It receives
    ``(query, filters)`` and returns the top-k documents from the
    caller's retriever. The pipeline never touches ``vs.similarity_search``
    directly.

    All pipeline stages are optional:

    - ``router_config``: if ``None`` and ``instructed_config`` is set, always
      runs instructed mode; otherwise standard.
    - ``instructed_config`` / ``decomposition_config``: if either is ``None``,
      instructed mode degrades to standard (single ``run_search`` call).
    - ``ranker`` / ``rerank_config``: skipped if either is ``None``.
    - ``instruction_rerank_config``: skipped when standard + auto_bypass.
    - ``verifier_config``: skipped if ``None``; retry loop is a no-op.

    Returns the final ``list[Document]``. Serialization is the caller's
    responsibility.
    """
    from dao_ai.tools.vector_search import rerank_documents

    # Router / mode decision.
    mode, auto_bypass = _decide_mode(
        query=query,
        router_config=router_config,
        instructed_config=instructed_config,
        instructed_columns=instructed_columns,
        context=context,
    )
    logger.trace("Routing mode", mode=mode, auto_bypass=auto_bypass)
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute(ATTR_ROUTER_MODE, mode)

    retry_count = 0
    max_retries = verifier_config.max_retries if verifier_config else 0
    previous_feedback: Optional[str] = None
    documents: list[Document] = []

    while True:
        # 1. Search (standard or instructed).
        if mode == "instructed" and instructed_config and decomposition_config:
            documents = _execute_instructed_search(
                run_search=run_search,
                query=query,
                base_filters=base_filters,
                instructed_config=instructed_config,
                decomposition_config=decomposition_config,
                instructed_columns=instructed_columns,
                primary_key=primary_key,
                previous_feedback=previous_feedback,
                context=context,
            )
        else:
            documents = run_search(query, base_filters)

        # 2. FlashRank cross-encoder pass.
        if ranker and rerank_config and documents:
            logger.debug("Applying FlashRank reranking")
            documents = rerank_documents(query, documents, ranker, rerank_config)

        # 3. Instruction-aware LLM rerank.
        documents = _apply_post_processing(
            documents=documents,
            query=query,
            mode=mode,
            auto_bypass=auto_bypass,
            instructed_config=instructed_config,
            instruction_rerank_config=instruction_rerank_config,
            instructed_columns=instructed_columns,
            context=context,
        )

        # 4. Verifier (optional).
        if not verifier_config:
            break
        if mode == "standard" and auto_bypass:
            break

        verifier_llm = (
            _get_cached_llm(verifier_config.model, context)
            if verifier_config.model
            else None
        )
        if not verifier_llm:
            break

        constraints = instructed_config.constraints if instructed_config else None

        verification_result = verify_results(
            llm=verifier_llm,
            query=query,
            documents=documents,
            columns=instructed_columns,
            constraints=constraints,
            previous_feedback=previous_feedback,
            resource_info=ResourceInfo(
                "model_serving",
                verifier_config.model.on_behalf_of_user,
                verifier_config.model.name,
            )
            if verifier_config.model
            else None,
        )

        _span = mlflow.get_current_active_span()
        if verification_result.passed:
            if _span:
                _span.set_attribute(ATTR_VERIFIER_OUTCOME, "passed")
                _span.set_attribute(ATTR_VERIFIER_RETRIES, retry_count)
            break

        # Warn-only: annotate and stop.
        if verifier_config.on_failure == "warn":
            if _span:
                _span.set_attribute(ATTR_VERIFIER_OUTCOME, "warned")
            documents = add_verification_metadata(documents, verification_result)
            break

        # Retries exhausted.
        if retry_count >= max_retries:
            if _span:
                _span.set_attribute(ATTR_VERIFIER_OUTCOME, "exhausted")
                _span.set_attribute(ATTR_VERIFIER_RETRIES, retry_count)
            documents = add_verification_metadata(
                documents, verification_result, exhausted=True
            )
            break

        # Standard mode can't meaningfully retry (no decomposition to adjust).
        if mode != "instructed":
            if _span:
                _span.set_attribute(ATTR_VERIFIER_OUTCOME, "warned")
            documents = add_verification_metadata(documents, verification_result)
            break

        # Retry: re-execute with the verifier feedback.
        retry_count += 1
        previous_feedback = verification_result.feedback
        logger.debug(
            "Retrying search with verification feedback",
            retry=retry_count,
        )

    return documents
