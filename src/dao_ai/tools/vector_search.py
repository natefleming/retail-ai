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
from databricks.vector_search.reranker import DatabricksReranker
from databricks_langchain import DatabricksVectorSearch
from flashrank import Ranker, RerankRequest
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from loguru import logger
from mlflow.entities import SpanType
from pydantic import BaseModel, ConfigDict, Field

from dao_ai._tracing import in_caller_context
from dao_ai.config import (
    ColumnInfo,
    DecompositionModel,
    FilterItem,
    InstructedRetrieverModel,
    InstructionAwareRerankModel,
    RerankParametersModel,
    RetrieverModel,
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


class VectorSearchInput(BaseModel):
    """Arguments for the dao-ai vector_search tool factory.

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


@mlflow.trace(name="rerank_documents", span_type=SpanType.RERANKER)
def _rerank_documents(
    query: str,
    documents: list[Document],
    ranker: Ranker,
    rerank_config: RerankParametersModel,
) -> list[Document]:
    """
    Rerank documents using FlashRank cross-encoder model.

    Args:
        query: The search query string
        documents: List of documents to rerank
        ranker: The FlashRank Ranker instance
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


def create_vector_search_tool(
    retriever: Optional[RetrieverModel | dict[str, Any]] = None,
    vector_store: Optional[VectorStoreModel | dict[str, Any]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> StructuredTool:
    """
    Create a Vector Search tool with dynamic schema and optional reranking.

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
        retriever = RetrieverModel(vector_store=vector_store)
    else:
        if isinstance(retriever, dict):
            retriever = RetrieverModel(**retriever)

    vector_store: VectorStoreModel = retriever.vector_store

    # Index is required
    if vector_store.index is None:
        raise ValueError("vector_store.index is required for vector search")

    index_name: str = vector_store.index.full_name
    columns: list[str] = list(retriever.columns or vector_store.index.columns or [])
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

    # Initialize FlashRank ranker if configured
    ranker: Optional[Ranker] = None
    if rerank_config and rerank_config.model:
        logger.debug(
            "Initializing FlashRank ranker",
            model=rerank_config.model,
            top_n=rerank_config.top_n or "auto",
        )
        try:
            # Use /tmp for cache in Model Serving (home dir may not be writable)
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

            # Patch rerank to always include token_type_ids for ONNX compatibility
            # Some ONNX runtimes require token_type_ids even when the model doesn't use them
            # FlashRank conditionally excludes them when all zeros, but ONNX may still expect them
            # See: https://github.com/huggingface/optimum/issues/1500
            if ranker.session is not None:
                import numpy as np

                _original_rerank = ranker.rerank

                def _patched_rerank(request):
                    query = request.query
                    passages = request.passages
                    query_passage_pairs = [[query, p["text"]] for p in passages]

                    input_text = ranker.tokenizer.encode_batch(query_passage_pairs)
                    input_ids = np.array([e.ids for e in input_text])
                    token_type_ids = np.array([e.type_ids for e in input_text])
                    attention_mask = np.array([e.attention_mask for e in input_text])

                    # Always include token_type_ids (the fix for ONNX compatibility)
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
        except Exception as e:
            logger.warning("Failed to initialize FlashRank ranker", error=str(e))
            rerank_config = None

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

        # Create DatabricksVectorSearch
        # Note: text_column should be None for Databricks-managed embeddings
        # (it's automatically determined from the index)
        #
        # When OBO is active, skip client_args entirely so
        # DatabricksVectorSearch uses the workspace_client (which carries
        # the user's forwarded token) instead of creating a separate
        # VectorSearchClient from client_args with ambient/SP auth.
        effective_client_args: dict[str, Any] | None = (
            client_args if client_args and not vector_store.on_behalf_of_user else None
        )
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

    # Build tool description with available columns for filtering
    base_description: str = description or f"Search documents in {index_name}"
    if columns:
        columns_list = ", ".join(columns)
        tool_description = (
            f"{base_description}. "
            f"Available filter columns: {columns_list}. "
            f"Filter operators: 'column' for equality, 'column NOT' for exclusion, "
            f"'column <', 'column <=', 'column >', 'column >=' for comparison, "
            f"'column LIKE' for token matching, 'column NOT LIKE' to exclude tokens."
        )
    else:
        tool_description = base_description

    @mlflow.trace(name="execute_instructed_retrieval", span_type=SpanType.RETRIEVER)
    def _execute_instructed_retrieval(
        vs: DatabricksVectorSearch,
        query: str,
        base_filters: dict[str, Any],
        previous_feedback: str | None = None,
        context: Context | None = None,
    ) -> list[Document]:
        """Execute instructed retrieval with query decomposition and RRF merging."""
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
                return vs.similarity_search(
                    query=query,
                    k=search_parameters.num_results or 5,
                    filter=base_filters if base_filters else None,
                    query_type=search_parameters.query_type or "ANN",
                )

            def normalize_filter_values(
                filters: dict[str, Any], case: str | None
            ) -> dict[str, Any]:
                """Normalize string filter values to specified case."""
                logger.trace("Normalizing filter values", filters=filters, case=case)
                if not case or not filters:
                    return filters
                normalized = {}
                for key, value in filters.items():
                    if isinstance(value, str):
                        normalized[key] = (
                            value.upper() if case == "uppercase" else value.lower()
                        )
                    elif isinstance(value, list):
                        normalized[key] = [
                            v.upper()
                            if case == "uppercase"
                            else v.lower()
                            if isinstance(v, str)
                            else v
                            for v in value
                        ]
                    else:
                        normalized[key] = value
                return normalized

            # Normalize base_filters once for consistent case matching
            normalized_base_filters = normalize_filter_values(
                base_filters, decomposition_config.normalize_filter_case
            )

            def execute_search(sq: SearchQuery) -> list[Document]:
                logger.trace("Executing search", query=sq.text, filters=sq.filters)
                # Convert FilterItem list to dict
                sq_filters_dict: dict[str, Any] = {}
                if sq.filters:
                    for item in sq.filters:
                        sq_filters_dict[item.key] = item.value
                sq_filters = normalize_filter_values(
                    sq_filters_dict, decomposition_config.normalize_filter_case
                )
                k: int = search_parameters.num_results or 5
                query_type: str = search_parameters.query_type or "ANN"
                # Decomposed filters take precedence over base filters
                combined_filters: dict[str, Any] = {
                    **normalized_base_filters,
                    **sq_filters,
                }
                logger.trace(
                    "Executing search",
                    query=sq.text,
                    k=k,
                    query_type=query_type,
                    filters=combined_filters,
                )
                return vs.similarity_search(
                    query=sq.text,
                    k=k,
                    filter=combined_filters if combined_filters else None,
                    query_type=query_type,
                )

            logger.debug(
                "Executing parallel searches",
                num_subqueries=len(subqueries),
                queries=[sq.text[:50] for sq in subqueries],
            )

            with ThreadPoolExecutor(
                max_workers=decomposition_config.max_subqueries
            ) as executor:
                # Wrap once: every subquery thread runs inside the caller's
                # captured contextvars so the MLflow active-span ContextVar
                # propagates and the per-subquery autolog spans nest under
                # the parent decomposed-retrieval span.
                all_results = list(
                    executor.map(in_caller_context(execute_search), subqueries)
                )

            merged = rrf_merge(
                all_results,
                k=decomposition_config.rrf_k,
                primary_key=vector_store.primary_key,
            )

            logger.debug(
                "Instructed retrieval complete",
                num_subqueries=len(subqueries),
                total_results=sum(len(r) for r in all_results),
                merged_results=len(merged),
            )

            # Fallback when decomposed filters are too restrictive
            if not merged:
                logger.warning(
                    "All instructed subqueries returned empty results, "
                    "falling back to standard unfiltered search",
                    num_subqueries=len(subqueries),
                    queries=[sq.text[:50] for sq in subqueries],
                )
                return vs.similarity_search(
                    query=query,
                    k=search_parameters.num_results or 5,
                    filter=base_filters if base_filters else None,
                    query_type=search_parameters.query_type or "ANN",
                )

            return merged

        except Exception as e:
            logger.warning(
                "Instructed retrieval failed, falling back to standard search",
                error=str(e),
            )
            return vs.similarity_search(
                query=query,
                k=search_parameters.num_results or 5,
                filter=base_filters if base_filters else None,
                query_type=search_parameters.query_type or "ANN",
            )

    @mlflow.trace(name="execute_standard_search", span_type=SpanType.RETRIEVER)
    def _execute_standard_search(
        vs: DatabricksVectorSearch,
        query: str,
        base_filters: dict[str, Any],
    ) -> list[Document]:
        """Execute standard single-query search."""
        logger.trace("Performing standard vector search", query_preview=query[:50])
        return vs.similarity_search(
            query=query,
            k=search_parameters.num_results or 5,
            filter=base_filters if base_filters else None,
            query_type=search_parameters.query_type or "ANN",
        )

    @mlflow.trace(name="apply_post_processing", span_type=SpanType.RETRIEVER)
    def _apply_post_processing(
        documents: list[Document],
        query: str,
        mode: Literal["standard", "instructed"],
        auto_bypass: bool,
        context: Context | None = None,
    ) -> list[Document]:
        """Apply instruction-aware reranking based on mode and bypass settings."""
        # Skip post-processing for standard mode when auto_bypass is enabled
        if mode == "standard" and auto_bypass:
            span = mlflow.get_current_active_span()
            if span:
                span.set_attribute(ATTR_ROUTER_BYPASSED, True)
            return documents

        # Apply instruction-aware reranking if configured
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

    # Use @tool decorator for proper ToolRuntime injection. args_schema=
    # gives the LLM a structural JSON schema for `filters` (type=array,
    # items={key,value}) — without it, Annotated[Optional[list[FilterItem]]]
    # is serialised to just a description string and the LLM commonly emits
    # filters as a flat dict.
    @tool(
        name_or_callable=tool_name,
        description=tool_description,
        args_schema=VectorSearchInput,
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

        # Determine execution mode via router or config
        mode: Literal["standard", "instructed"] = "standard"
        auto_bypass = True

        logger.trace("Router configuration", router_config=router_config)
        logger.trace("Instructed configuration", instructed_config=instructed_config)
        logger.trace(
            "Instruction-aware rerank configuration",
            instruction_aware=instruction_rerank_config,
        )

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
                except Exception as e:
                    # Router fail-safe: default to standard mode
                    logger.warning(
                        "Router failed, defaulting to standard mode", error=str(e)
                    )
                    span = mlflow.get_current_active_span()
                    if span:
                        span.set_attribute(ATTR_ROUTER_FALLBACK, True)
                    mode = router_config.default_mode
            else:
                mode = router_config.default_mode
        elif instructed_config:
            # No router but instructed is configured - use instructed mode
            mode = "instructed"
            auto_bypass = False

        logger.trace("Routing mode", mode=mode, auto_bypass=auto_bypass)
        span = mlflow.get_current_active_span()
        if span:
            span.set_attribute(ATTR_ROUTER_MODE, mode)

        # Search + verify loop: re-executes search with feedback on verification failure
        retry_count = 0
        max_retries = verifier_config.max_retries if verifier_config else 0
        previous_feedback: str | None = None

        while True:
            # Execute search based on mode
            if mode == "instructed" and instructed_config and decomposition_config:
                documents = _execute_instructed_retrieval(
                    vs, query, base_filters, previous_feedback, context=context
                )
            else:
                documents = _execute_standard_search(vs, query, base_filters)

            # Apply FlashRank reranking if configured
            if ranker and rerank_config and documents:
                logger.debug("Applying FlashRank reranking")
                documents = _rerank_documents(query, documents, ranker, rerank_config)

            # Apply instruction-aware reranking
            documents = _apply_post_processing(
                documents, query, mode, auto_bypass, context=context
            )

            # Verification (if configured)
            if not verifier_config:
                break

            # Skip verification for standard mode when auto_bypass is enabled
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

            # Warn-only: annotate and stop
            if verifier_config.on_failure == "warn":
                if _span:
                    _span.set_attribute(ATTR_VERIFIER_OUTCOME, "warned")
                documents = add_verification_metadata(documents, verification_result)
                break

            # Retries exhausted
            if retry_count >= max_retries:
                if _span:
                    _span.set_attribute(ATTR_VERIFIER_OUTCOME, "exhausted")
                    _span.set_attribute(ATTR_VERIFIER_RETRIES, retry_count)
                documents = add_verification_metadata(
                    documents, verification_result, exhausted=True
                )
                break

            # Standard mode can't meaningfully retry (no decomposition to adjust)
            if mode != "instructed":
                if _span:
                    _span.set_attribute(ATTR_VERIFIER_OUTCOME, "warned")
                documents = add_verification_metadata(documents, verification_result)
                break

            # Retry: re-execute search with verifier feedback
            retry_count += 1
            previous_feedback = verification_result.feedback
            logger.debug(
                "Retrying search with verification feedback",
                retry=retry_count,
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

    logger.success("Vector search tool created", name=tool_name, index=index_name)

    return _vector_search_tool
