"""
Vector search tool for retrieving documents from Databricks Vector Search.

This module provides a tool factory for creating semantic search tools
using ToolRuntime[Context, AgentState] for type-safe runtime access.
"""

import os
from typing import Any, Callable, List, Optional, Sequence

import mlflow
from databricks.vector_search.reranker import DatabricksReranker
from databricks_langchain.vectorstores import DatabricksVectorSearch
from flashrank import Ranker, RerankRequest
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from loguru import logger
from mlflow.entities import SpanType

from databricks_ai_bridge.vector_search_retriever_tool import (
    FilterItem,
    VectorSearchRetrieverToolInput,
)

from dao_ai.config import (
    RerankParametersModel,
    RetrieverModel,
    VectorStoreModel,
)
from dao_ai.state import AgentState, Context
from dao_ai.utils import normalize_host


def create_vector_search_tool(
    retriever: RetrieverModel | dict[str, Any] | None = None,
    vector_store: VectorStoreModel | dict[str, Any] | None = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[..., list[dict[str, Any]]]:
    """
    Create a Vector Search tool for retrieving documents from a Databricks Vector
    Search index.

    This function creates a tool that enables semantic search over product
    information, documentation, or other content using the @tool decorator pattern.
    It supports optional reranking of results using FlashRank for improved relevance.

    Args:
        retriever: Full retriever configuration with search parameters and
            optional reranking. Can be a RetrieverModel instance or dict.
            Mutually exclusive with vector_store.
        vector_store: Direct vector store reference (uses default search
            parameters). Can be a VectorStoreModel instance or dict.
            Mutually exclusive with retriever.
        name: Optional custom name for the tool
        description: Optional custom description for the tool

    Returns:
        A LangChain tool that performs vector search with optional reranking

    Raises:
        ValueError: If both retriever and vector_store are provided, or if
            neither is provided
    """

    # Validate mutually exclusive parameters
    if retriever is None and vector_store is None:
        raise ValueError(
            "Must provide either 'retriever' or 'vector_store' parameter"
        )
    if retriever is not None and vector_store is not None:
        raise ValueError(
            "Cannot provide both 'retriever' and 'vector_store' parameters. "
            "Use 'retriever' for full control or 'vector_store' for default "
            "search parameters."
        )

    # Handle vector_store parameter
    if vector_store is not None:
        # Convert dict to VectorStoreModel if needed
        if isinstance(vector_store, dict):
            vector_store = VectorStoreModel(**vector_store)

        logger.debug(
            "VectorStoreModel provided, using default search parameters "
            "and no reranking"
        )
        # Wrap in RetrieverModel with defaults
        retriever = RetrieverModel(vector_store=vector_store)
    else:
        # Handle retriever parameter
        if isinstance(retriever, dict):
            retriever = RetrieverModel(**retriever)

    vector_store_config: VectorStoreModel = retriever.vector_store

    # Index is required for vector search
    if vector_store_config.index is None:
        raise ValueError("vector_store.index is required for vector search")

    index_name: str = vector_store_config.index.full_name
    columns: Sequence[str] = retriever.columns or []
    search_parameters: dict[str, Any] = retriever.search_parameters.model_dump()
    primary_key: str = vector_store_config.primary_key or ""
    doc_uri: str = vector_store_config.doc_uri or ""
    text_column: str = vector_store_config.embedding_source_column

    # Extract reranker configuration
    reranker_config: Optional[RerankParametersModel] = retriever.rerank

    # Initialize FlashRank ranker once if reranking is enabled
    # This is expensive (loads model weights), so we do it once and reuse
    # across invocations
    ranker: Optional[Ranker] = None
    if reranker_config:
        logger.debug(
            "Creating vector search tool with reranking",
            name=name,
            reranker_model=reranker_config.model,
            top_n=reranker_config.top_n or "auto",
        )
        try:
            # Expand tilde in cache_dir path
            cache_dir = os.path.expanduser(reranker_config.cache_dir)
            ranker = Ranker(
                model_name=reranker_config.model, cache_dir=cache_dir
            )
            logger.success("FlashRank ranker initialized", model=reranker_config.model)
        except Exception as e:
            logger.warning(
                "Failed to initialize FlashRank ranker, reranking disabled",
                error=str(e),
            )
            # Set reranker_config to None so we don't attempt reranking
            reranker_config = None
    else:
        logger.debug("Creating vector search tool without reranking", name=name)

    # Initialize the vector store
    # Note: text_column is only required for self-managed embeddings
    # For Databricks-managed embeddings, it's automatically determined from the index

    # Build client_args for VectorSearchClient from environment variables
    # This is needed because during MLflow model validation, credentials must be
    # explicitly passed to VectorSearchClient via client_args.
    # The workspace_client parameter in DatabricksVectorSearch is only used to detect
    # model serving mode - it doesn't pass credentials to VectorSearchClient.
    client_args: dict[str, Any] = {}
    databricks_host = normalize_host(os.environ.get("DATABRICKS_HOST"))
    if databricks_host:
        client_args["workspace_url"] = databricks_host
    if os.environ.get("DATABRICKS_TOKEN"):
        client_args["personal_access_token"] = os.environ.get("DATABRICKS_TOKEN")
    if os.environ.get("DATABRICKS_CLIENT_ID"):
        client_args["service_principal_client_id"] = os.environ.get(
            "DATABRICKS_CLIENT_ID"
        )
    if os.environ.get("DATABRICKS_CLIENT_SECRET"):
        client_args["service_principal_client_secret"] = os.environ.get(
            "DATABRICKS_CLIENT_SECRET"
        )

    logger.trace(
        "Creating DatabricksVectorSearch", client_args_keys=list(client_args.keys())
    )

    # Pass both workspace_client (for model serving detection) and client_args
    # (for credentials)
    vector_store: DatabricksVectorSearch = DatabricksVectorSearch(
        index_name=index_name,
        text_column=None,  # Let DatabricksVectorSearch determine this from the index
        columns=columns,
        include_score=True,
        workspace_client=vector_store_config.workspace_client,
        client_args=client_args if client_args else None,
    )

    # Register the retriever schema with MLflow for model serving integration
    mlflow.models.set_retriever_schema(
        name=name or "retriever",
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        other_columns=list(columns),
    )

    # Helper function to perform vector similarity search
    @mlflow.trace(name="find_documents", span_type=SpanType.RETRIEVER)
    def _find_documents(
        query: str, filters: Optional[List[dict[str, Any]]] = None
    ) -> List[Document]:
        """Perform vector similarity search."""
        # Convert filters to dict format
        filters_dict: dict[str, Any] = {}
        if filters:
            for item in filters:
                # item is already a dict from Pydantic validation
                filters_dict[item["key"]] = item["value"]

        # Merge with any configured filters
        combined_filters: dict[str, Any] = {
            **filters_dict,
            **search_parameters.get("filters", {}),
        }

        # Perform similarity search
        num_results: int = search_parameters.get("num_results", 10)
        query_type: str = search_parameters.get("query_type", "ANN")

        logger.trace(
            "Performing vector search",
            query_preview=query[:50],
            k=num_results,
            filters=combined_filters,
        )

        # Build similarity search kwargs
        search_kwargs = {
            "query": query,
            "k": num_results,
            "filter": combined_filters if combined_filters else None,
            "query_type": query_type,
        }

        # Add DatabricksReranker if configured with columns
        if reranker_config and reranker_config.columns:
            search_kwargs["reranker"] = DatabricksReranker(
                columns_to_rerank=reranker_config.columns
            )

        documents: List[Document] = vector_store.similarity_search(**search_kwargs)

        logger.debug(
            "Retrieved documents from vector search", documents_count=len(documents)
        )
        return documents

    # Helper function to rerank documents
    @mlflow.trace(name="rerank_documents", span_type=SpanType.RETRIEVER)
    def _rerank_documents(query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using FlashRank.

        Uses the ranker instance initialized at tool creation time (captured
        in closure). This avoids expensive model loading on every invocation.
        """
        if not reranker_config or ranker is None:
            return documents

        logger.trace(
            "Starting reranking",
            documents_count=len(documents),
            model=reranker_config.model,
        )

        # Prepare passages for reranking
        passages: List[dict[str, Any]] = [
            {"text": doc.page_content, "meta": doc.metadata} for doc in documents
        ]

        # Create reranking request
        rerank_request: RerankRequest = RerankRequest(query=query, passages=passages)

        # Perform reranking
        logger.trace(
            "Reranking passages", passages_count=len(passages), query_preview=query[:50]
        )
        results: List[dict[str, Any]] = ranker.rerank(rerank_request)

        # Apply top_n filtering
        top_n: int = reranker_config.top_n or len(documents)
        results = results[:top_n]
        logger.debug("Reranking complete", top_n=top_n, candidates_count=len(documents))

        # Convert back to Document objects with reranking scores
        reranked_docs: List[Document] = []
        for result in results:
            # Find original document by matching text
            orig_doc: Optional[Document] = next(
                (doc for doc in documents if doc.page_content == result["text"]), None
            )
            if orig_doc:
                # Add reranking score to metadata
                reranked_doc: Document = Document(
                    page_content=orig_doc.page_content,
                    metadata={
                        **orig_doc.metadata,
                        "reranker_score": result["score"],
                    },
                )
                reranked_docs.append(reranked_doc)

        top_score = (
            reranked_docs[0].metadata.get("reranker_score", 0)
            if reranked_docs
            else None
        )
        logger.debug(
            "Documents reranked",
            input_count=len(documents),
            output_count=len(reranked_docs),
            model=reranker_config.model,
            top_score=top_score,
        )

        return reranked_docs

    # Create the main vector search tool using @tool decorator
    # Uses ToolRuntime[Context, AgentState] for type-safe runtime access
    @tool(
        name_or_callable=name or index_name,
        description=description or "Search for documents using vector similarity",
    )
    def vector_search_tool(
        query: str,
        filters: Optional[List[dict[str, Any]]] = None,  # Will be validated by Pydantic
        runtime: ToolRuntime[Context, AgentState] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for documents using vector similarity with optional reranking.

        This tool performs a two-stage retrieval process:
        1. Vector similarity search to find candidate documents
        2. Optional reranking using cross-encoder model for improved relevance

        Both stages are traced in MLflow for observability.

        Uses ToolRuntime[Context, AgentState] for type-safe runtime access.

        Returns:
            List of serialized documents with page_content and metadata
        """
        logger.trace(
            "Vector search tool called",
            query_preview=query[:50],
            filters=filters,
            reranking_enabled=reranker_config is not None,
        )

        # Step 1: Perform vector similarity search
        documents: List[Document] = _find_documents(query, filters)

        # Step 2: If reranking is enabled, rerank the documents
        if reranker_config:
            logger.trace(
                "Reranking enabled",
                model=reranker_config.model,
                top_n=reranker_config.top_n or "all",
            )
            documents = _rerank_documents(query, documents)
            logger.debug("Returning reranked documents", documents_count=len(documents))
        else:
            logger.trace("Reranking disabled, returning original results")

        # Return Command with ToolMessage containing the documents
        # Serialize documents to dicts for proper ToolMessage handling
        serialized_docs: list[dict[str, Any]] = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

        return serialized_docs

    return vector_search_tool
