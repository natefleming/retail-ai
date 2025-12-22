"""
Vector search tool for retrieving documents from Databricks Vector Search.

This module provides a tool factory for creating semantic search tools
using ToolRuntime[Context, AgentState] for type-safe runtime access.
"""

import os
from typing import Any, Callable, List, Optional, Sequence

import mlflow
from databricks.vector_search.reranker import DatabricksReranker
from databricks_ai_bridge.vector_search_retriever_tool import (
    FilterItem,
    VectorSearchRetrieverToolInput,
)
from databricks_langchain.vectorstores import DatabricksVectorSearch
from flashrank import Ranker, RerankRequest
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from loguru import logger
from mlflow.entities import SpanType

from dao_ai.config import (
    RerankParametersModel,
    RetrieverModel,
    VectorStoreModel,
)
from dao_ai.state import AgentState, Context
from dao_ai.utils import normalize_host


def create_vector_search_tool(
    retriever: RetrieverModel | dict[str, Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[..., list[dict[str, Any]]]:
    """
    Create a Vector Search tool for retrieving documents from a Databricks Vector Search index.

    This function creates a tool that enables semantic search over product information,
    documentation, or other content using the @tool decorator pattern. It supports optional
    reranking of results using FlashRank for improved relevance.

    Args:
        retriever: Configuration details for the vector search retriever, including:
            - name: Name of the tool
            - description: Description of the tool's purpose
            - primary_key: Primary key column for the vector store
            - text_column: Text column used for vector search
            - doc_uri: URI for documentation or additional context
            - vector_store: Dictionary with 'endpoint_name' and 'index' for vector search
            - columns: List of columns to retrieve from the vector store
            - search_parameters: Additional parameters for customizing the search behavior
            - rerank: Optional rerank configuration for result reranking
        name: Optional custom name for the tool
        description: Optional custom description for the tool

    Returns:
        A LangChain tool that performs vector search with optional reranking
    """

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
    # This is expensive (loads model weights), so we do it once and reuse across invocations
    ranker: Optional[Ranker] = None
    if reranker_config:
        logger.debug(
            f"Creating vector search tool with reranking: '{name}' "
            f"(model: {reranker_config.model}, top_n: {reranker_config.top_n or 'auto'})"
        )
        try:
            ranker = Ranker(
                model_name=reranker_config.model, cache_dir=reranker_config.cache_dir
            )
            logger.info(
                f"FlashRank ranker initialized successfully (model: {reranker_config.model})"
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize FlashRank ranker during tool creation: {e}. "
                "Reranking will be disabled for this tool."
            )
            # Set reranker_config to None so we don't attempt reranking
            reranker_config = None
    else:
        logger.debug(
            f"Creating vector search tool without reranking: '{name}' (standard similarity search only)"
        )

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

    logger.debug(
        f"Creating DatabricksVectorSearch with client_args keys: {list(client_args.keys())}"
    )

    # Pass both workspace_client (for model serving detection) and client_args (for credentials)
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
        query: str, filters: Optional[List[FilterItem]] = None
    ) -> List[Document]:
        """Perform vector similarity search."""
        # Convert filters to dict format
        filters_dict: dict[str, Any] = {}
        if filters:
            for item in filters:
                item_dict = dict(item)
                filters_dict[item_dict["key"]] = item_dict["value"]

        # Merge with any configured filters
        combined_filters: dict[str, Any] = {
            **filters_dict,
            **search_parameters.get("filters", {}),
        }

        # Perform similarity search
        num_results: int = search_parameters.get("num_results", 10)
        query_type: str = search_parameters.get("query_type", "ANN")

        logger.debug(
            f"Performing vector search: query='{query[:50]}...', k={num_results}, filters={combined_filters}"
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

        logger.debug(f"Retrieved {len(documents)} documents from vector search")
        return documents

    # Helper function to rerank documents
    @mlflow.trace(name="rerank_documents", span_type=SpanType.RETRIEVER)
    def _rerank_documents(query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents using FlashRank.

        Uses the ranker instance initialized at tool creation time (captured in closure).
        This avoids expensive model loading on every invocation.
        """
        if not reranker_config or ranker is None:
            return documents

        logger.debug(
            f"Starting reranking for {len(documents)} documents using model '{reranker_config.model}'"
        )

        # Prepare passages for reranking
        passages: List[dict[str, Any]] = [
            {"text": doc.page_content, "meta": doc.metadata} for doc in documents
        ]

        # Create reranking request
        rerank_request: RerankRequest = RerankRequest(query=query, passages=passages)

        # Perform reranking
        logger.debug(f"Reranking {len(passages)} passages for query: '{query[:50]}...'")
        results: List[dict[str, Any]] = ranker.rerank(rerank_request)

        # Apply top_n filtering
        top_n: int = reranker_config.top_n or len(documents)
        results = results[:top_n]
        logger.debug(
            f"Reranking complete. Filtered to top {top_n} results from {len(documents)} candidates"
        )

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

        logger.debug(
            f"Reranked {len(documents)} documents → {len(reranked_docs)} results "
            f"(model: {reranker_config.model}, top score: {reranked_docs[0].metadata.get('reranker_score', 0):.4f})"
            if reranked_docs
            else f"Reranking completed with {len(reranked_docs)} results"
        )

        return reranked_docs

    # Create the main vector search tool using @tool decorator
    # Uses ToolRuntime[Context, AgentState] for type-safe runtime access
    @tool(
        name_or_callable=name or index_name,
        description=description or "Search for documents using vector similarity",
        args_schema=VectorSearchRetrieverToolInput,
    )
    def vector_search_tool(
        query: str,
        filters: Optional[List[FilterItem]] = None,
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
        logger.debug(
            f"Vector search tool called: query='{query[:50]}...', reranking={reranker_config is not None}"
        )

        # Step 1: Perform vector similarity search
        documents: List[Document] = _find_documents(query, filters)

        # Step 2: If reranking is enabled, rerank the documents
        if reranker_config:
            logger.debug(
                f"Reranking enabled (model: '{reranker_config.model}', top_n: {reranker_config.top_n or 'all'})"
            )
            documents = _rerank_documents(query, documents)
            logger.debug(f"Returning {len(documents)} reranked documents")
        else:
            logger.debug("Reranking disabled, returning original vector search results")

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
