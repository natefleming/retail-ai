"""
Vector search tool for retrieving documents from Databricks Vector Search.

This module provides a tool factory for creating semantic search tools
with dynamic filter schemas based on table columns, FlashRank reranking support,
instructed retrieval with query decomposition and RRF merging, and optional
query routing, result verification, and instruction-aware reranking.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker
from databricks_langchain import DatabricksVectorSearch
from flashrank import Ranker, RerankRequest
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from loguru import logger
from mlflow.entities import SpanType
from pydantic import BaseModel, ConfigDict, Field, create_model

from dao_ai._tracing import in_caller_context
from dao_ai.config import (
    ColumnInfo,
    DecompositionModel,
    FilterItem,
    InstructedRetrieverModel,
    InstructionAwareRerankModel,
    RerankParametersModel,
    AiSearchRetrieverModel,
    RouterModel,
    SearchParametersModel,
    SearchQuery,
    VectorStoreModel,
    VerifierModel,
    value_of,
)
from dao_ai.state import Context
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
    set_resource_attributes,
)
from dao_ai.tools.verifier import add_verification_metadata, verify_results
from dao_ai.utils import is_in_model_serving, normalize_host


# auth_type values that the databricks-langchain library (as of 0.20.0) extracts
# natively from workspace_client.config inside DatabricksVectorSearch.__init__.
# When wc.config.auth_type is one of these, we let the library do its own job.
# Sources of truth in databricks-sdk credentials_provider.py:
#   - "pat":                              L146 @credentials_strategy("pat", ...)
#   - "oauth-m2m":                        L237 @oauth_credentials_strategy("oauth-m2m", ...)
#   - "model_serving_user_credentials":   L1499-1501, composed from
#                                         "model_serving_" + ModelServingAuthProvider.USER_CREDENTIALS ("user_credentials")
_LIBRARY_NATIVE_AUTH_TYPES: tuple[str, ...] = (
    "pat",
    "oauth-m2m",
    "model_serving_user_credentials",
)


def _client_args_from_ambient_wc(wc: WorkspaceClient) -> dict[str, Any] | None:
    """Populate ``VectorSearchClient`` auth kwargs from an ambient WorkspaceClient
    whose ``config.auth_type`` is *not* one the ``databricks-langchain`` library
    extracts natively.

    The library inspects ``workspace_client.config.auth_type`` inside
    ``DatabricksVectorSearch.__init__`` and only builds ``client_args`` for the
    three values in ``_LIBRARY_NATIVE_AUTH_TYPES``. On Databricks Serverless v5
    notebook / job runs, ``auth_type`` is usually ``databricks-cli`` /
    ``oauth-u2m`` / ``default`` — none of which the library maps. Without help
    ``client_args`` stays empty and the underlying ``VectorSearchClient`` raises
    ``InvalidInputException`` demanding a PAT or SP.

    Returns ``None`` when:
      * the library will already handle the WC natively (auth_type in the
        native list), or
      * no bearer can be resolved from ``wc.config.authenticate()``, or
      * ``wc.config.host`` is missing (VectorSearchClient rejects a bearer
        without a workspace_url in its ``validate()`` — passing a partial dict
        would convert one error into a more confusing one).

    In every ``None`` case the caller passes ``client_args=None`` to the
    library, which then gets a clean shot at its own extraction / fallback.

    Otherwise returns a dict shaped as
    ``{workspace_url, personal_access_token}``.
    """
    try:
        auth_type = getattr(wc.config, "auth_type", None)
        if auth_type in _LIBRARY_NATIVE_AUTH_TYPES:
            return None
        headers = wc.config.authenticate() or {}
        bearer = str(headers.get("Authorization", ""))
        if not bearer.startswith("Bearer "):
            # Recognized the WC but its ambient auth isn't a Bearer we can
            # forward (e.g. Basic). Better to punt to the library than lie
            # about credentials we can't actually use.
            logger.warning(
                "dao_ai.vector_search.ambient_auth.unusable",
                auth_type=auth_type,
                note="WorkspaceClient.config.authenticate() produced no Bearer header",
            )
            return None
        host = getattr(wc.config, "host", None)
        if not host:
            # VectorSearchClient.validate() requires workspace_url when a PAT
            # is provided (databricks/ai_search/client.py:186-189). Punt.
            logger.warning(
                "dao_ai.vector_search.ambient_auth.no_host",
                auth_type=auth_type,
                note="WorkspaceClient.config.host is missing; cannot mint client_args",
            )
            return None
        return {
            "workspace_url": normalize_host(host),
            "personal_access_token": bearer[len("Bearer "):],
        }
    except Exception as e:
        logger.warning("dao_ai.vector_search.ambient_auth.exception", error=str(e))
        return None


class AiSearchInput(BaseModel):
    """Arguments for the dao-ai AI Search tool factory.

    (Formerly ``VectorSearchInput``. Databricks rebranded Vector Search to
    AI Search; the old class name remains as an alias defined at the end
    of this module for backwards compatibility.)

    Co-located with the @tool decorator so the JSON schema rendered to the LLM
    matches the factory's runtime expectations. Without an explicit args_schema,
    LangChain's @tool decorator infers schema from Annotated[...] hints and
    silently drops the structural type information for Optional[list[BaseModel]]
    fields, leaving the LLM with only a description string to guide filter
    shape (which it then commonly emits as a flat dict instead of a list).
    """

    # extra="ignore" rather than "forbid": LangChain passes the injected
    # ToolRuntime as a kwarg when invoking a tool whose @tool decorator has
    # args_schema=, and Pydantic must silently drop it. extra="forbid" would
    # reject every call with `validation error … runtime: extra_forbidden`.
    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        description="The natural-language search query to find relevant documents.",
    )
    filters: Optional[list[FilterItem]] = Field(
        default=None,
        description=(
            "Optional metadata filters. Pass a JSON array of objects, each with "
            "'key' and 'value'. Do NOT pass a flat dict. "
            'Example: [{"key": "category", "value": "B2B"}, '
            '{"key": "price <=", "value": 150}]. '
            "The 'key' is a column name optionally suffixed with an operator: "
            "(none) for equality, 'NOT' for exclusion, '< <= > >=' for numeric "
            "comparison, 'LIKE' for token match, 'NOT LIKE' to exclude tokens. "
            "Omit or set to null when no filter applies."
        ),
    )


_FILTER_OPERATOR_SUFFIXES: tuple[str, ...] = (
    "",
    " NOT",
    " <",
    " <=",
    " >",
    " >=",
    " LIKE",
    " NOT LIKE",
)


def _legal_filter_keys(
    columns: list[str],
    operator_overrides: dict[str, list[str]] | None = None,
) -> list[str]:
    """Cross product of columns × the 8 operator suffixes.

    Every column gets every suffix — no type-aware narrowing. Databricks
    Vector Search rejects unsupported filter combinations at query time
    (e.g. ``LIKE`` on a numeric); we let the API be the source of truth
    for operator applicability, matching upstream databricks-langchain.

    When ``operator_overrides`` names a column, that column's suffixes come
    from the override list verbatim (bare strings like ``["", "LIKE"]``)
    instead of the full ``_FILTER_OPERATOR_SUFFIXES`` set. This is the
    hand-declared ``ColumnInfo.operators`` knob — users who want to lock
    a column to a narrower operator set can do so.

    Override entries are the operator strings as declared on ColumnInfo
    (e.g. ``""``, ``"NOT"``, ``"LIKE"``) — no leading space; we add the
    leading space here so the emitted enum shape matches
    ``_FILTER_OPERATOR_SUFFIXES``.
    """
    result: list[str] = []
    for c in columns:
        override = (operator_overrides or {}).get(c)
        if override is not None:
            # ColumnInfo.operators is stored without the leading space
            # ("LIKE" not " LIKE"). Map to the suffix shape.
            suffixes = tuple("" if op == "" else f" {op}" for op in override)
        else:
            suffixes = _FILTER_OPERATOR_SUFFIXES
        for op in suffixes:
            result.append(f"{c}{op}")
    return result


def _vsc_for_refresh(vector_store: "VectorStoreModel") -> VectorSearchClient | None:
    """Mint a ``VectorSearchClient`` for ``VectorStoreModel.refresh()``.

    Follows the same ambient-auth extraction the factory uses at query time
    for ``DatabricksVectorSearch`` (see :func:`_client_args_from_ambient_wc`)
    so build-time ``index.describe()`` calls succeed under all four
    documented auth modes — including the ambient Serverless-v5 / CLI-profile
    mode where a bare ``VectorSearchClient()`` raises
    ``InvalidInputException: Please specify either personal access token or
    service principal client ID and secret``.
    """
    try:
        wc = vector_store.workspace_client_from(None)
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "Could not build WorkspaceClient for VS refresh; refresh may fail",
            error=f"{type(e).__name__}: {e}",
        )
        return None
    try:
        client_args = _client_args_from_ambient_wc(wc)
        if client_args:
            return VectorSearchClient(**client_args, disable_notice=True)
        # Library-native auth path or explicit env-var PAT — a default
        # VectorSearchClient() is what the provider would build anyway.
        return VectorSearchClient(disable_notice=True)
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "VectorSearchClient construction for refresh failed; "
            "letting refresh() fall through to default",
            error=f"{type(e).__name__}: {e}",
        )
        return None


def _vector_column_names_from_describe(details: dict | None) -> set[str]:
    """Return the exact vector column names to strip from an index probe.

    Handles both embedding modes and both index-spec containers:

      * **Managed embeddings** — Databricks synthesises one vector column
        per ``embedding_source_columns[].name`` entry, named
        ``<source>_vector``. We read the source names and mint the
        synthesised names.
      * **Self-managed embeddings** — the user pre-computes vectors; the
        vector column is named by ``embedding_vector_columns[].name``
        (any user-chosen name, e.g. ``vector`` or ``my_embedding``).
        Strip that exact name.
      * **Delta-Sync indexes** carry the source list under
        ``delta_sync_index_spec``; **Direct-Access indexes** under
        ``direct_access_index_spec``. We check both.

    Reading names from ``describe()`` means we strip only actual
    synthesised or user-declared vector columns — never a business column
    that happens to end in ``_vector``.
    """
    if not isinstance(details, dict):
        return set()
    names: set[str] = set()
    for spec_key in ("delta_sync_index_spec", "direct_access_index_spec"):
        spec = details.get(spec_key) or {}
        # Managed embeddings: strip ``<source>_vector``.
        for entry in spec.get("embedding_source_columns") or []:
            if isinstance(entry, dict) and (src := entry.get("name")):
                names.add(f"{src}_vector")
        # Self-managed embeddings: strip the user-declared name directly.
        for entry in spec.get("embedding_vector_columns") or []:
            if isinstance(entry, dict) and (vec := entry.get("name")):
                names.add(vec)
    return names


def _fetch_index_columns(
    vector_store: "VectorStoreModel",
) -> list[tuple[str, str | None, str | None]] | None:
    """Return ``[(name, type_str, comment)]`` from the INDEX itself.

    A Vector Search index is a UC entity and its columns are queryable via
    the Unity Catalog Tables API. This is the same pattern
    ``databricks_langchain.vector_search_retriever_tool`` uses; a single
    ``wc.tables.get(index.full_name)`` call returns the authoritative
    column list, their Databricks types, and any UC column comments the
    catalog author set.

    Vector-embedding columns are stripped automatically:
      * ``<source>_vector`` — managed-embedding synthesised (via describe)
      * anything with a leading underscore — CDF / system reserved
      * ``__db_*_vector`` — db-managed embedding vector

    Returns ``None`` on any failure (soft-fail; caller falls back to
    declared columns if present, otherwise free-form filter). No source-
    table fallback — matches upstream databricks-langchain's single-call
    behavior. Users hitting permission asymmetry (SP has source-table
    grants but not index-UC-entity grants) should hand-declare via
    ``ColumnInfo`` on ``retriever.columns``.
    """
    if vector_store.index is None:
        return None
    try:
        wc = vector_store.workspace_client_from(None)
        table = wc.tables.get(vector_store.index.full_name)
        cols = getattr(table, "columns", None) or []
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "UC Tables lookup on VS index failed; "
            "column auto-discovery unavailable",
            index=vector_store.index.full_name,
            error=f"{type(e).__name__}: {e}",
        )
        return None

    known_vector_cols = _vector_column_names_from_describe(
        getattr(vector_store, "_index_details", None)
    )
    out: list[tuple[str, str | None, str | None]] = []
    for col in cols:
        name = getattr(col, "name", None)
        if not name:
            continue
        if name.startswith("_"):
            continue
        if known_vector_cols and name in known_vector_cols:
            continue
        if not known_vector_cols and name.endswith("_vector"):
            continue
        t = getattr(col, "type_text", None) or getattr(col, "type_name", None)
        comment = getattr(col, "comment", None)
        out.append((name, str(t) if t else None, str(comment) if comment else None))
    return out or None


# ColumnInfo.type values map to human-readable Databricks type labels used
# in the tool description. Scalar types are just labels (all 8 operator
# suffixes still apply). Array is narrowed at build time to equality-only
# (see `_is_array_type` and the factory's operator_overrides logic).
_COLUMN_INFO_TYPE_LABELS: dict[str, str] = {
    "string": "STRING",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "datetime": "DATETIME",
    "array": "ARRAY",
}


def _is_array_type(type_str: str | None) -> bool:
    """True when a Databricks type string represents an ARRAY column.

    Matches ``array<string>``, ``ARRAY<INT>``, ``array<double>`` etc. from
    UC Tables ``type_text`` / ``type_name``, plus the bare ``array`` value
    users write in ``ColumnInfo.type``. Array-typed columns get a
    single-operator enum (equality/contains only) in the LLM-facing
    args_schema — other operators (LIKE, ordering, NOT LIKE) are rejected
    by VS at query time on array columns (see the Do It Best
    `hardware-iq` filtering guide, 2026-07-06) and would only mislead
    the LLM.
    """
    if not type_str:
        return False
    return type_str.strip().lower().startswith("array")


def _normalize_declared_columns(
    items: "list[str | ColumnInfo]",
) -> tuple[
    list[str],
    dict[str, str],
    dict[str, str],
    dict[str, list[str]],
    bool,
]:
    """Split a mixed ``list[str | ColumnInfo]`` into flat lookup tables.

    Returns a tuple ``(names, types, descriptions, operator_overrides,
    any_hand_declared)`` where:

    * ``names`` — column names in declaration order
    * ``types`` — ``{name: type_label}`` for ColumnInfo entries only;
      bare strings do not populate this map
    * ``descriptions`` — ``{name: description}`` for ColumnInfo entries
      whose ``description`` field is set
    * ``operator_overrides`` — ``{name: [ops]}`` for ColumnInfo entries
      whose ``operators`` field was explicitly set by the user (detected
      via Pydantic's ``model_fields_set``); other columns fall through to
      the default 8-suffix set at ``_legal_filter_keys`` time
    * ``any_hand_declared`` — ``True`` when at least one item is a
      ``ColumnInfo``. Signals to the caller that hand-declaration is in
      effect and build-time UC calls should be skipped.
    """
    # Late import — ColumnInfo is defined in config.py, which imports
    # from this module transitively via the tool factory. Import at call
    # time to avoid the circular.
    from dao_ai.config import ColumnInfo

    names: list[str] = []
    types: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    operator_overrides: dict[str, list[str]] = {}
    any_hand_declared = False

    for item in items:
        if isinstance(item, ColumnInfo):
            any_hand_declared = True
            name = item.name
            names.append(name)
            type_label = _COLUMN_INFO_TYPE_LABELS.get(item.type, item.type.upper())
            types[name] = type_label
            if item.description:
                descriptions[name] = item.description
            # Only treat operators as an override when the user explicitly
            # set them (Pydantic tracks this via model_fields_set). If the
            # field defaulted to the full 8-op list, we let the standard
            # `_legal_filter_keys` cross-product apply — same as bare
            # strings — for consistency. Exception: type=="array" narrows
            # to equality-only ("") by default, since VS rejects every
            # other operator on array columns.
            if "operators" in item.model_fields_set:
                operator_overrides[name] = list(item.operators)
            elif item.type == "array":
                operator_overrides[name] = [""]
        elif isinstance(item, str):
            names.append(item)
        else:
            logger.debug(
                "Ignoring unrecognized columns[] entry (not str or ColumnInfo)",
                item_type=type(item).__name__,
            )
    return names, types, descriptions, operator_overrides, any_hand_declared


def _build_columns_description(
    names: list[str],
    types: dict[str, str],
    descriptions: dict[str, str],
) -> str:
    """Render the "Available columns for filtering" block.

    Format matches upstream ``databricks_langchain.vector_search_retriever_tool``
    style so the LLM sees the same shape it would from the industry-standard
    library, plus dao-ai's per-column descriptions when available. Empty
    ``names`` → empty string (caller can skip the block).

    ``types`` and ``descriptions`` are best-effort. A column with neither is
    listed as ``- name``.
    """
    if not names:
        return ""

    lines: list[str] = ["Available columns for filtering:"]
    any_array = False
    for name in names:
        type_label = types.get(name)
        desc = descriptions.get(name)
        parts = [f"- {name}"]
        if type_label:
            parts.append(f"({type_label})")
        if desc:
            parts.append(f": {desc}")
        elif type_label:
            # We already emitted "(TYPE)"; nothing more needed.
            pass
        lines.append(" ".join(parts))
        if type_label and _is_array_type(type_label):
            any_array = True
    lines.append("")
    lines.append(
        "Supports operators: LIKE, NOT LIKE, NOT, <, <=, >, >=, and empty "
        "(equality/IN). Combine into the filter key like "
        '``{"key": "brand LIKE", "value": "DEWALT"}``.'
    )
    if any_array:
        lines.append("")
        lines.append(
            "Array columns match via element containment (equality only): "
            '``{"key": "tags", "value": "cordless"}`` finds records where '
            "the tags array contains 'cordless'. A list value performs "
            'OR-of-contains: ``{"key": "tags", "value": ["cordless", '
            '"brushless"]}``. Other operators (LIKE, ordering, NOT) are '
            "not supported on array columns and will be rejected."
        )
    return "\n".join(lines)


def _build_filter_item_model(
    columns: list[str],
    operator_overrides: dict[str, list[str]] | None = None,
) -> type[BaseModel]:
    """Build a per-tool FilterItem whose ``key`` is Literal-narrowed to columns.

    When ``columns`` is empty we return the free-form module-level
    :class:`FilterItem` so callers see no change. When columns are known,
    the returned model has ``key: Literal[<col>, "<col> NOT", "<col> <=",
    "<col> LIKE", …]`` — a bad key is rejected by pydantic at tool-call
    time, before the retriever is ever invoked.

    The narrowing surfaces on the LLM as a JSON-schema ``enum`` on the
    ``key`` property. That closes the "guessed a column name that doesn't
    exist" hallucination hole (regression: MLflow trace
    ``fc785d795b77675ac0e42fe5296b523a`` — LLM emitted ``"name NOT LIKE"``
    against a products index whose column is ``product_name``).

    ``operator_overrides`` accepts per-column suffix lists from
    hand-declared ``ColumnInfo.operators`` — users who want to lock a
    column to a narrower operator set (e.g. ``brand`` limited to
    ``["", "LIKE"]``) can do so.
    """
    if not columns:
        return FilterItem

    legal_keys = _legal_filter_keys(columns, operator_overrides)
    key_type = Literal[*legal_keys]  # PEP 646 unpacking; Python 3.11+.

    return create_model(
        "DynamicFilterItem",
        __base__=FilterItem,
        __module__=__name__,
        key=(
            key_type,
            Field(
                description=(
                    "Column name (optionally suffixed with an operator). "
                    "Must be one of the enumerated values. Operators: "
                    "(none) equality, 'NOT' exclusion, '< <= > >=' numeric, "
                    "'LIKE' token match, 'NOT LIKE' token exclude."
                )
            ),
        ),
    )


def _build_vector_search_input_model(
    columns: list[str],
    operator_overrides: dict[str, list[str]] | None = None,
) -> type[BaseModel]:
    """Build a per-tool AiSearchInput whose ``filters[]`` is narrowed.

    When ``columns`` is empty we return the module-level
    :class:`AiSearchInput` (behavior identical to pre-change). When
    columns are known, we build a subclass whose ``filters`` type is
    ``list[<DynamicFilterItem for these columns>]``, so the JSON schema
    the LLM sees carries the enum of legal keys.

    ``operator_overrides`` is forwarded to :func:`_build_filter_item_model`
    for hand-declared ``ColumnInfo.operators`` support.
    """
    if not columns:
        return AiSearchInput

    filter_item_cls = _build_filter_item_model(columns, operator_overrides)
    return create_model(
        "DynamicVectorSearchInput",
        __base__=AiSearchInput,
        __module__=__name__,
        filters=(
            Optional[list[filter_item_cls]],  # type: ignore[valid-type]
            Field(
                default=None,
                description=(
                    "Optional metadata filters. Pass a JSON array of objects, "
                    "each with 'key' and 'value'. Do NOT pass a flat dict. "
                    'Example: [{"key": "category", "value": "B2B"}, '
                    '{"key": "price <=", "value": 150}]. '
                    "The 'key' field is enumerated to the actual index "
                    "columns × operator suffixes — an unlisted key is "
                    "rejected. Omit or set to null when no filter applies."
                ),
            ),
        ),
    )


def build_flashrank_ranker(
    rerank_config: Optional[RerankParametersModel],
) -> Optional[Ranker]:
    """Initialize a FlashRank ``Ranker`` from a rerank config, or ``None``.

    Returns ``None`` when:
    - ``rerank_config`` is ``None`` or has no ``model`` set (nothing to init).
    - Ranker construction throws (soft-fail — logs a warning; caller should
      skip reranking when the returned value is ``None``).

    Under Databricks Model Serving the home directory isn't writable, so
    the FlashRank cache is forced to ``/tmp/dao_ai/cache/flashrank`` (any
    user-configured ``cache_dir`` is overridden with a warning).

    Also installs an ONNX-compatibility patch on the returned ranker so
    ``token_type_ids`` is always populated on the ONNX input tensor —
    some ONNX runtimes require the key even when the model doesn't use
    it, and FlashRank normally omits it when all zeros
    (https://github.com/huggingface/optimum/issues/1500).

    Shared by :func:`create_ai_search_tool` and
    :func:`create_lakebase_search_tool` — same FlashRank init for both
    backends.
    """
    if rerank_config is None or not rerank_config.model:
        return None

    logger.debug(
        "Initializing FlashRank ranker",
        model=rerank_config.model,
        top_n=rerank_config.top_n or "auto",
    )
    try:
        if is_in_model_serving():
            cache_dir = "/tmp/dao_ai/cache/flashrank"
            if rerank_config.cache_dir != cache_dir:
                logger.warning(
                    "FlashRank cache_dir overridden in Model Serving",
                    configured=rerank_config.cache_dir,
                    actual=cache_dir,
                )
        else:
            cache_dir = os.path.expanduser(rerank_config.cache_dir)
        ranker = Ranker(model_name=rerank_config.model, cache_dir=cache_dir)

        # ONNX-compatibility patch — always emit token_type_ids.
        if ranker.session is not None:
            import numpy as np

            _original_rerank = ranker.rerank  # noqa: F841 (kept for parity)

            def _patched_rerank(request):
                query = request.query
                passages = request.passages
                query_passage_pairs = [[query, p["text"]] for p in passages]

                input_text = ranker.tokenizer.encode_batch(query_passage_pairs)
                input_ids = np.array([e.ids for e in input_text])
                token_type_ids = np.array([e.type_ids for e in input_text])
                attention_mask = np.array([e.attention_mask for e in input_text])

                onnx_input = {
                    "input_ids": input_ids.astype(np.int64),
                    "attention_mask": attention_mask.astype(np.int64),
                    "token_type_ids": token_type_ids.astype(np.int64),
                }

                outputs = ranker.session.run(None, onnx_input)
                logits = outputs[0]

                if logits.shape[1] == 1:
                    scores = 1 / (1 + np.exp(-logits.flatten()))
                else:
                    exp_logits = np.exp(logits)
                    scores = exp_logits[:, 1] / np.sum(exp_logits, axis=1)

                for score, passage in zip(scores, passages):
                    passage["score"] = score

                passages.sort(key=lambda x: x["score"], reverse=True)
                return passages

            ranker.rerank = _patched_rerank

        logger.success("FlashRank ranker initialized", model=rerank_config.model)
        return ranker
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to initialize FlashRank ranker", error=str(e))
        return None


@mlflow.trace(name="rerank_documents", span_type=SpanType.RERANKER)
def rerank_documents(
    query: str,
    documents: list[Document],
    ranker: Ranker,
    rerank_config: RerankParametersModel,
) -> list[Document]:
    """
    Rerank documents using FlashRank cross-encoder model.

    Backend-agnostic — takes a ``list[Document]`` from any retriever
    (`ai_search`, `lakebase_search`, ...) and returns a re-ordered list
    with each doc's metadata carrying a ``reranker_score`` (0-1).

    Args:
        query: The search query string
        documents: List of documents to rerank
        ranker: The FlashRank Ranker instance (see :func:`build_flashrank_ranker`)
        rerank_config: Reranking configuration

    Returns:
        Reranked list of documents with reranker_score in metadata
    """
    logger.trace(
        "Starting reranking",
        documents_count=len(documents),
        model=rerank_config.model,
    )

    # Early return if no documents to rerank
    if not documents:
        logger.debug("No documents to rerank, skipping")
        return documents

    # Prepare passages for reranking
    passages: list[dict[str, Any]] = [
        {"text": doc.page_content, "meta": doc.metadata} for doc in documents
    ]

    # Create reranking request
    rerank_request: RerankRequest = RerankRequest(query=query, passages=passages)

    # Perform reranking
    results: list[dict[str, Any]] = ranker.rerank(rerank_request)

    # Apply top_n filtering
    top_n: int = rerank_config.top_n or len(documents)
    results = results[:top_n]
    logger.debug("Reranking complete", top_n=top_n, candidates_count=len(documents))

    # Convert back to Document objects with reranking scores
    reranked_docs: list[Document] = []
    for result in results:
        orig_doc: Optional[Document] = next(
            (doc for doc in documents if doc.page_content == result["text"]), None
        )
        if orig_doc:
            reranked_doc: Document = Document(
                page_content=orig_doc.page_content,
                metadata={
                    **orig_doc.metadata,
                    "reranker_score": result["score"],
                },
            )
            reranked_docs.append(reranked_doc)

    logger.debug(
        "Documents reranked",
        input_count=len(documents),
        output_count=len(reranked_docs),
        model=rerank_config.model,
    )

    return reranked_docs


def create_ai_search_tool(
    retriever: Optional[AiSearchRetrieverModel | dict[str, Any]] = None,
    vector_store: Optional[VectorStoreModel | dict[str, Any]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> StructuredTool:
    """
    Create an AI Search tool with dynamic schema and optional reranking.

    (Formerly ``create_vector_search_tool``. Databricks rebranded Vector
    Search to AI Search; the old function name remains as an alias
    defined at the end of this module for backwards compatibility.)

    Args:
        retriever: Full retriever configuration with search parameters and reranking
        vector_store: Direct vector store reference (uses default search parameters)
        name: Optional custom name for the tool
        description: Optional custom description for the tool

    Returns:
        A LangChain StructuredTool with proper schema (additionalProperties: false)
    """

    # Validate mutually exclusive parameters
    if retriever is None and vector_store is None:
        raise ValueError("Must provide either 'retriever' or 'vector_store' parameter")
    if retriever is not None and vector_store is not None:
        raise ValueError(
            "Cannot provide both 'retriever' and 'vector_store' parameters"
        )

    # Handle vector_store parameter
    if vector_store is not None:
        if isinstance(vector_store, dict):
            vector_store = VectorStoreModel(**vector_store)
        retriever = AiSearchRetrieverModel(vector_store=vector_store)
    else:
        if isinstance(retriever, dict):
            retriever = AiSearchRetrieverModel(**retriever)

    vector_store: VectorStoreModel = retriever.vector_store

    # Index is required
    if vector_store.index is None:
        raise ValueError("vector_store.index is required for vector search")

    index_name: str = vector_store.index.full_name
    # Column source of truth — three modes based on retriever.columns shape:
    #
    #   A. Hand-declared. Any item in the list is a ``ColumnInfo``. Names,
    #      types, descriptions, and (optionally) per-column operator
    #      restrictions come straight from the declaration. **No UC calls.**
    #      Matches LangChain ``AttributeInfo`` / LlamaIndex ``MetadataInfo``
    #      shape — the user owns the schema.
    #
    #   B. Bare strings only. Names come from the declared list; UC Tables
    #      on the index is called best-effort to enrich the tool
    #      description with types + column comments (not used for
    #      enforcement). Enum is built from the declared names verbatim.
    #
    #   C. Empty. Discover names via a single ``wc.tables.get(index)``
    #      call (same primary path as ``databricks_langchain``). If that
    #      fails, degrade to free-form ``FilterItem`` (pre-branch behavior).
    #
    # No source-table fallback — matches upstream. Users with permission
    # asymmetries (SP has source-table grants but not index-UC-entity
    # grants) should hand-declare via ``ColumnInfo``.
    declared_items: list[Any] = list(
        retriever.columns or vector_store.columns or []
    )
    (
        declared_names,
        declared_types,
        declared_descriptions,
        operator_overrides_raw,
        any_hand_declared,
    ) = _normalize_declared_columns(declared_items)
    # Keep as a dict throughout so downstream can add entries (e.g. the
    # auto-discovered array-narrowing loop). We coerce to `None` when
    # empty at the point we pass it to `_build_vector_search_input_model`.
    operator_overrides: dict[str, list[str]] = dict(operator_overrides_raw)

    # ``refresh()`` still runs — it populates ``_index_details`` (needed
    # for vector-column stripping via ``_vector_column_names_from_describe``)
    # and, for Direct-Access indexes, ``vector_store.columns`` from
    # ``columns_to_sync``. WARNING log only; downstream discovery is
    # resilient to a refresh failure.
    vsc_for_refresh: VectorSearchClient | None = _vsc_for_refresh(vector_store)
    try:
        vector_store.refresh(vsc=vsc_for_refresh)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Vector Search index describe() failed; "
            "column auto-discovery may be incomplete",
            index=index_name,
            error=f"{type(e).__name__}: {e}",
        )

    columns: list[str]
    description_types: dict[str, str] = dict(declared_types)
    description_descriptions: dict[str, str] = dict(declared_descriptions)

    if any_hand_declared:
        # Mode A — hand-declared authoritative. No UC calls.
        columns = declared_names
    elif declared_names:
        # Mode B — bare strings. Enum from declared names; UC lookup is
        # best-effort for description enrichment only.
        columns = declared_names
        index_cols = _fetch_index_columns(vector_store)
        if index_cols:
            # NB: use ``col_name`` instead of ``name`` — the outer function
            # takes a ``name`` parameter (the LLM-facing tool name), and
            # binding ``name`` in a for-loop here shadows it. After the loop,
            # ``name`` would leak the last declared column, and the eventual
            # ``tool_name = name or ...`` (line ~1041) would pick up that
            # stale value instead of the caller's tool name.
            uc_type_map = {col_name: t for col_name, t, _ in index_cols if t}
            uc_comment_map = {col_name: c for col_name, _, c in index_cols if c}
            for col_name in declared_names:
                if col_name not in description_types and col_name in uc_type_map:
                    description_types[col_name] = uc_type_map[col_name]
                if col_name not in description_descriptions and col_name in uc_comment_map:
                    description_descriptions[col_name] = uc_comment_map[col_name]
    else:
        # Mode C — nothing declared. Discover via UC.
        index_cols = _fetch_index_columns(vector_store)
        if index_cols:
            # Same ``col_name`` shadowing avoidance as above.
            columns = [col_name for col_name, _, _ in index_cols]
            for col_name, t, c in index_cols:
                if t and col_name not in description_types:
                    description_types[col_name] = t
                if c and col_name not in description_descriptions:
                    description_descriptions[col_name] = c
        else:
            # Direct-Access indexes: refresh may have populated
            # ``vector_store.columns`` from ``columns_to_sync``. Use those
            # if the UC lookup returned nothing.
            fallback = list(vector_store.columns or [])
            columns = fallback

    # Array-typed columns get equality-only (empty suffix). VS rejects
    # every other operator on ARRAY columns at query time; narrowing the
    # enum prevents the LLM from emitting invalid filters in the first
    # place. Hand-declared overrides (via ColumnInfo.operators) still win.
    # (``col_name`` again — see the shadowing note above.)
    for col_name, uc_type in description_types.items():
        if col_name in operator_overrides:
            continue
        if _is_array_type(uc_type):
            operator_overrides[col_name] = [""]

    logger.debug(
        "Vector Search columns resolved",
        index=index_name,
        mode="hand-declared" if any_hand_declared else ("declared" if declared_names else "discovered"),
        columns=columns,
        have_types=bool(description_types),
        have_operator_overrides=bool(operator_overrides),
    )
    search_parameters: SearchParametersModel = retriever.search_parameters
    rerank_config: Optional[RerankParametersModel] = retriever.rerank
    instructed_config: Optional[InstructedRetrieverModel] = retriever.instructed

    # Extract nested configs from instructed (all depend on schema context)
    # Columns are the single source of truth for schema context.
    instructed_columns: list[ColumnInfo] = (
        instructed_config.columns if instructed_config else []
    )
    decomposition_config: Optional[DecompositionModel] = (
        instructed_config.decomposition if instructed_config else None
    )
    router_config: Optional[RouterModel] = (
        instructed_config.router if instructed_config else None
    )
    instruction_rerank_config: Optional[InstructionAwareRerankModel] = (
        instructed_config.rerank if instructed_config else None
    )
    verifier_config: Optional[VerifierModel] = (
        instructed_config.verifier if instructed_config else None
    )

    # Initialize FlashRank ranker if configured. On init failure the helper
    # returns None and we skip the reranker; the downstream "if ranker" guard
    # handles both "not configured" and "init failed" uniformly.
    ranker: Optional[Ranker] = build_flashrank_ranker(rerank_config)

    # Log instructed retrieval configuration
    if instructed_config and decomposition_config:
        logger.success(
            "Instructed retrieval configured",
            decomposition_model=decomposition_config.model.name
            if decomposition_config.model
            else None,
            max_subqueries=decomposition_config.max_subqueries,
            rrf_k=decomposition_config.rrf_k,
        )

    # Log instruction-aware reranking configuration
    if instruction_rerank_config:
        logger.success(
            "Instruction-aware reranking configured",
            model=instruction_rerank_config.model.name
            if instruction_rerank_config.model
            else None,
            top_n=instruction_rerank_config.top_n,
        )

    # Build client_args for VectorSearchClient (SP or PAT auth only).
    # OBO auth is handled separately via workspace_client_from(context).
    client_args: dict[str, Any] = {}
    has_explicit_auth = any(
        [
            os.environ.get("DATABRICKS_TOKEN"),
            os.environ.get("DATABRICKS_CLIENT_ID"),
            vector_store.pat,
            vector_store.client_id,
        ]
    )

    if has_explicit_auth:
        databricks_host = os.environ.get("DATABRICKS_HOST")
        if not databricks_host and vector_store.workspace_host:
            databricks_host = value_of(vector_store.workspace_host)
        if databricks_host:
            client_args["workspace_url"] = normalize_host(databricks_host)

        token = os.environ.get("DATABRICKS_TOKEN")
        if not token and vector_store.pat:
            token = value_of(vector_store.pat)
        if token:
            client_args["personal_access_token"] = token

        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        if not client_id and vector_store.client_id:
            client_id = value_of(vector_store.client_id)
        if client_id:
            client_args["service_principal_client_id"] = client_id

        client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
        if not client_secret and vector_store.client_secret:
            client_secret = value_of(vector_store.client_secret)
        if client_secret:
            client_args["service_principal_client_secret"] = client_secret

    logger.debug(
        "Creating vector search tool",
        name=name,
        index=index_name,
        client_args_keys=list(client_args.keys()) if client_args else [],
    )

    # Cache for DatabricksVectorSearch - created lazily for OBO support
    _cached_vector_search: DatabricksVectorSearch | None = None

    def _get_vector_search(context: Context | None) -> DatabricksVectorSearch:
        """Get or create DatabricksVectorSearch, using context for OBO auth if available."""
        nonlocal _cached_vector_search

        # Use cached instance if available and not OBO
        if _cached_vector_search is not None and not vector_store.on_behalf_of_user:
            return _cached_vector_search

        # Get workspace client with OBO support via context
        workspace_client: WorkspaceClient = vector_store.workspace_client_from(context)

        # Create DatabricksVectorSearch. text_column stays None for
        # Databricks-managed embeddings (auto-detected from the index).
        #
        # Auth selection covers four modes, in priority order:
        #   1. OBO — vector_store.on_behalf_of_user=True: pass client_args=None
        #      so DatabricksVectorSearch uses the forwarded-bearer WorkspaceClient
        #      directly.
        #   2. Explicit PAT/SP — DATABRICKS_TOKEN / DATABRICKS_CLIENT_ID env vars
        #      OR declared vector_store.pat / .client_id / .client_secret; already
        #      collected into `client_args` upstream.
        #   3. Ambient App SP on Serverless v5 — WorkspaceClient.config.auth_type
        #      is oauth-m2m; the library handles it natively when we pass
        #      client_args=None (helper returns None).
        #   4. Ambient user on Serverless v5 (notebook / job) — auth_type is
        #      one of databricks-cli / oauth-u2m / default; the library does
        #      NOT handle these, so we mint client_args ourselves from the
        #      runtime bearer via _client_args_from_ambient_wc.
        if vector_store.on_behalf_of_user:
            effective_client_args: dict[str, Any] | None = None
        elif client_args:
            effective_client_args = client_args
        else:
            effective_client_args = _client_args_from_ambient_wc(workspace_client)
        vs: DatabricksVectorSearch = DatabricksVectorSearch(
            index_name=index_name,
            text_column=None,
            columns=columns,
            workspace_client=workspace_client,
            client_args=effective_client_args,
            primary_key=vector_store.primary_key,
            doc_uri=vector_store.doc_uri,
            include_score=True,
            reranker=(
                DatabricksReranker(columns_to_rerank=rerank_config.columns)
                if rerank_config and rerank_config.columns
                else None
            ),
        )

        # Cache for non-OBO scenarios
        if not vector_store.on_behalf_of_user:
            _cached_vector_search = vs

        return vs

    # Determine tool name and description
    tool_name: str = name or f"vector_search_{vector_store.index.name}"

    # Build tool description with available columns for filtering — matches
    # the databricks-langchain "Available columns for filtering: name
    # (TYPE): description" shape, enriched with hand-declared descriptions
    # when the user provided ColumnInfo entries.
    base_description: str = description or f"Search documents in {index_name}"
    columns_block: str = _build_columns_description(
        columns, description_types, description_descriptions
    )
    if columns_block:
        tool_description = f"{base_description}\n\n{columns_block}"
    else:
        tool_description = base_description

    # Pipeline logic (execute_instructed_retrieval + apply_post_processing +
    # verifier retry loop) lives in ``dao_ai.tools.instructed_pipeline`` so
    # ``lakebase_search`` shares the same code path. See ``_run_search``
    # inside the tool closure below for the backend adapter that wraps
    # ``vs.similarity_search``.

    # Use @tool decorator for proper ToolRuntime injection. args_schema=
    # gives the LLM a structural JSON schema for `filters` (type=array,
    # items={key,value}) — without it, Annotated[Optional[list[FilterItem]]]
    # is serialised to just a description string and the LLM commonly emits
    # filters as a flat dict.
    VectorSearchSchema: type[BaseModel] = _build_vector_search_input_model(
        columns, operator_overrides or None
    )

    @tool(
        name_or_callable=tool_name,
        description=tool_description,
        args_schema=VectorSearchSchema,
    )
    def _vector_search_tool(
        query: str,
        filters: Optional[list[FilterItem]] = None,
        runtime: ToolRuntime[Context] = None,
    ) -> str:
        """Search for relevant documents using vector similarity."""
        context: Context | None = runtime.context if runtime else None
        vs: DatabricksVectorSearch = _get_vector_search(context)

        set_resource_attributes(
            ResourceInfo("vector_search", vector_store.on_behalf_of_user, index_name)
        )

        filters_dict: dict[str, Any] = {}
        if filters:
            for item in filters:
                filters_dict[item.key] = item.value

        base_filters: dict[str, Any] = {
            **filters_dict,
            **(search_parameters.filters or {}),
        }

        # Backend adapter for the shared instructed pipeline. Captures this
        # invocation's ``vs`` (which may be OBO-scoped) plus the retriever's
        # ``k`` + ``query_type``. The pipeline calls this once per subquery
        # in instructed mode and once with base_filters in standard mode.
        def _run_search(qtxt: str, flt: dict[str, Any]) -> list[Document]:
            return vs.similarity_search(
                query=qtxt,
                k=search_parameters.num_results or 5,
                filter=flt if flt else None,
                query_type=search_parameters.query_type or "ANN",
            )

        # All routing / decomposition / rerank / verify logic lives in
        # ``dao_ai.tools.instructed_pipeline`` — same code path
        # lakebase_search uses.
        from dao_ai.tools.instructed_pipeline import execute_instructed_pipeline

        documents = execute_instructed_pipeline(
            run_search=_run_search,
            query=query,
            base_filters=base_filters,
            instructed_config=instructed_config,
            router_config=router_config,
            verifier_config=verifier_config,
            decomposition_config=decomposition_config,
            instruction_rerank_config=instruction_rerank_config,
            instructed_columns=instructed_columns,
            primary_key=vector_store.primary_key,
            ranker=ranker,
            rerank_config=rerank_config,
            context=context,
        )

        # Serialize documents to JSON format for LLM consumption
        serialized_docs: list[dict[str, Any]] = []
        for doc in documents:
            metadata_serializable: dict[str, Any] = {}
            for key, value in doc.metadata.items():
                if hasattr(value, "item"):  # numpy scalar
                    metadata_serializable[key] = value.item()
                else:
                    metadata_serializable[key] = value

            serialized_docs.append(
                {
                    "page_content": doc.page_content,
                    "metadata": metadata_serializable,
                }
            )

        return json.dumps(serialized_docs)

    logger.success("AI Search tool created", name=tool_name, index=index_name)

    return _vector_search_tool


# Backwards-compatible aliases — Vector Search naming will eventually be
# deprecated. Both names refer to the same class / function.
VectorSearchInput = AiSearchInput
create_vector_search_tool = create_ai_search_tool
