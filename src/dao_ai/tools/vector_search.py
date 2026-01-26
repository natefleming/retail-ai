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

from dao_ai.config import (
    ColumnInfo,
    FilterItem,
    InstructedRetrieverModel,
    RerankParametersModel,
    RetrieverModel,
    RouterModel,
    SearchParametersModel,
    SearchQuery,
    VectorStoreModel,
    VerificationResult,
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
from dao_ai.tools.verifier import add_verification_metadata, verify_results
from dao_ai.utils import is_in_model_serving, normalize_host


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
    router_config: Optional[RouterModel] = retriever.router
    rerank_config: Optional[RerankParametersModel] = retriever.rerank
    instructed_config: Optional[InstructedRetrieverModel] = retriever.instructed
    verifier_config: Optional[VerifierModel] = retriever.verifier

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
    if instructed_config:
        logger.success(
            "Instructed retrieval configured",
            decomposition_model=instructed_config.decomposition_model.name
            if instructed_config.decomposition_model
            else None,
            max_subqueries=instructed_config.max_subqueries,
            rrf_k=instructed_config.rrf_k,
        )

    # Log instruction-aware reranking configuration
    if rerank_config and rerank_config.instruction_aware:
        logger.success(
            "Instruction-aware reranking configured",
            model=rerank_config.instruction_aware.model.name
            if rerank_config.instruction_aware.model
            else None,
            top_n=rerank_config.instruction_aware.top_n,
        )

    # Build client_args for VectorSearchClient
    client_args: dict[str, Any] = {}
    has_explicit_auth = any(
        [
            os.environ.get("DATABRICKS_TOKEN"),
            os.environ.get("DATABRICKS_CLIENT_ID"),
            vector_store.pat,
            vector_store.client_id,
            vector_store.on_behalf_of_user,
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
        vs: DatabricksVectorSearch = DatabricksVectorSearch(
            index_name=index_name,
            text_column=None,
            columns=columns,
            workspace_client=workspace_client,
            client_args=client_args if client_args else None,
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
    ) -> list[Document]:
        """Execute instructed retrieval with query decomposition and RRF merging."""
        logger.trace(
            "Executing instructed retrieval", query=query, base_filters=base_filters
        )
        try:
            decomposition_llm = _get_cached_llm(instructed_config.decomposition_model)

            # Fall back to retriever columns if instructed columns not provided
            instructed_columns: list[ColumnInfo] | None = instructed_config.columns
            if instructed_columns is None and columns:
                instructed_columns = [ColumnInfo(name=col) for col in columns]

            subqueries: list[SearchQuery] = decompose_query(
                llm=decomposition_llm,
                query=query,
                schema_description=instructed_config.schema_description,
                constraints=instructed_config.constraints,
                max_subqueries=instructed_config.max_subqueries,
                examples=instructed_config.examples,
                previous_feedback=previous_feedback,
                columns=instructed_columns,
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

            def execute_search(sq: SearchQuery) -> list[Document]:
                logger.trace("Executing search", query=sq.text, filters=sq.filters)
                # Convert FilterItem list to dict
                sq_filters_dict: dict[str, Any] = {}
                if sq.filters:
                    for item in sq.filters:
                        sq_filters_dict[item.key] = item.value
                sq_filters = normalize_filter_values(
                    sq_filters_dict, instructed_config.normalize_filter_case
                )
                k: int = search_parameters.num_results or 5
                query_type: str = search_parameters.query_type or "ANN"
                combined_filters: dict[str, Any] = {**sq_filters, **base_filters}
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
                max_workers=instructed_config.max_subqueries
            ) as executor:
                all_results = list(executor.map(execute_search, subqueries))

            merged = rrf_merge(
                all_results,
                k=instructed_config.rrf_k,
                primary_key=vector_store.primary_key,
            )

            logger.debug(
                "Instructed retrieval complete",
                num_subqueries=len(subqueries),
                total_results=sum(len(r) for r in all_results),
                merged_results=len(merged),
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
    ) -> list[Document]:
        """Apply instruction-aware reranking and verification based on mode and bypass settings."""
        # Skip post-processing for standard mode when auto_bypass is enabled
        if mode == "standard" and auto_bypass:
            mlflow.set_tag("router.bypassed_stages", "true")
            return documents

        # Apply instruction-aware reranking if configured
        if rerank_config and rerank_config.instruction_aware:
            instruction_config = rerank_config.instruction_aware
            instruction_llm = (
                _get_cached_llm(instruction_config.model)
                if instruction_config.model
                else None
            )

            if instruction_llm:
                schema_desc = (
                    instructed_config.schema_description if instructed_config else None
                )
                # Get columns for dynamic instruction generation
                rerank_columns: list[ColumnInfo] | None = None
                if instructed_config and instructed_config.columns:
                    rerank_columns = instructed_config.columns
                elif columns:
                    rerank_columns = [ColumnInfo(name=col) for col in columns]

                documents = instruction_aware_rerank(
                    llm=instruction_llm,
                    query=query,
                    documents=documents,
                    instructions=instruction_config.instructions,
                    schema_description=schema_desc,
                    columns=rerank_columns,
                    top_n=instruction_config.top_n,
                )

        # Apply verification if configured
        if verifier_config:
            verifier_llm = (
                _get_cached_llm(verifier_config.model)
                if verifier_config.model
                else None
            )

            if verifier_llm:
                schema_desc = (
                    instructed_config.schema_description if instructed_config else ""
                )
                constraints = (
                    instructed_config.constraints if instructed_config else None
                )
                retry_count = 0
                verification_result: VerificationResult | None = None
                previous_feedback: str | None = None

                while retry_count <= verifier_config.max_retries:
                    verification_result = verify_results(
                        llm=verifier_llm,
                        query=query,
                        documents=documents,
                        schema_description=schema_desc,
                        constraints=constraints,
                        previous_feedback=previous_feedback,
                    )

                    if verification_result.passed:
                        mlflow.set_tag("verifier.outcome", "passed")
                        mlflow.set_tag("verifier.retries", str(retry_count))
                        break

                    # Handle failure based on configuration
                    if verifier_config.on_failure == "warn":
                        mlflow.set_tag("verifier.outcome", "warned")
                        documents = add_verification_metadata(
                            documents, verification_result
                        )
                        break

                    if retry_count >= verifier_config.max_retries:
                        mlflow.set_tag("verifier.outcome", "exhausted")
                        mlflow.set_tag("verifier.retries", str(retry_count))
                        documents = add_verification_metadata(
                            documents, verification_result, exhausted=True
                        )
                        break

                    # Retry with feedback
                    mlflow.set_tag("verifier.outcome", "retried")
                    previous_feedback = verification_result.feedback
                    retry_count += 1
                    logger.debug(
                        "Retrying search with verification feedback", retry=retry_count
                    )

        return documents

    # Use @tool decorator for proper ToolRuntime injection
    @tool(name_or_callable=tool_name, description=tool_description)
    def _vector_search_tool(
        query: Annotated[str, "The search query to find relevant documents"],
        filters: Annotated[
            Optional[list[FilterItem]],
            "Optional filters as key-value pairs. "
            "Key operators: 'column' (equality), 'column NOT' (exclusion), "
            "'column <', '<=', '>', '>=' (comparison), "
            "'column LIKE' (token match), 'column NOT LIKE' (exclude token). "
            f"Valid columns: {', '.join(columns) if columns else 'none'}.",
        ] = None,
        runtime: ToolRuntime[Context] = None,
    ) -> str:
        """Search for relevant documents using vector similarity."""
        context: Context | None = runtime.context if runtime else None
        vs: DatabricksVectorSearch = _get_vector_search(context)

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
            instruction_aware=rerank_config.instruction_aware
            if rerank_config
            else None,
        )

        if router_config:
            router_llm = (
                _get_cached_llm(router_config.model) if router_config.model else None
            )
            auto_bypass = router_config.auto_bypass

            if router_llm and instructed_config:
                try:
                    mode = route_query(
                        llm=router_llm,
                        query=query,
                        schema_description=instructed_config.schema_description,
                    )
                except Exception as e:
                    # Router fail-safe: default to standard mode
                    logger.warning(
                        "Router failed, defaulting to standard mode", error=str(e)
                    )
                    mlflow.set_tag("router.fallback", "true")
                    mode = router_config.default_mode
            else:
                mode = router_config.default_mode
        elif instructed_config:
            # No router but instructed is configured - use instructed mode
            mode = "instructed"
            auto_bypass = False
        elif rerank_config and rerank_config.instruction_aware:
            # No router/instructed but instruction_aware reranking is configured
            # Disable auto_bypass to ensure instruction_aware reranking runs
            auto_bypass = False

        logger.trace("Routing mode", mode=mode, auto_bypass=auto_bypass)
        mlflow.set_tag("router.mode", mode)

        # Execute search based on mode
        if mode == "instructed" and instructed_config:
            documents = _execute_instructed_retrieval(vs, query, base_filters)
        else:
            documents = _execute_standard_search(vs, query, base_filters)

        # Apply FlashRank reranking if configured
        if ranker and rerank_config:
            logger.debug("Applying FlashRank reranking")
            documents = _rerank_documents(query, documents, ranker, rerank_config)

        # Apply post-processing (instruction reranking + verification)
        documents = _apply_post_processing(documents, query, mode, auto_bypass)

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
