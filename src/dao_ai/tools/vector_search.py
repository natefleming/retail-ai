from typing import Any, List, Optional, Sequence

import mlflow
from databricks_ai_bridge.vector_search_retriever_tool import (
    FilterItem,
    vector_search_retriever_tool_trace,
)
from databricks_langchain.vector_search_retriever_tool import VectorSearchRetrieverTool
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from loguru import logger
from mlflow.entities import SpanType
from pydantic import Field

from dao_ai.config import (
    RerankerParametersModel,
    RetrieverModel,
    VectorStoreModel,
)


class RerankingVectorSearchRetrieverTool(VectorSearchRetrieverTool):
    """
    Vector search retrieval tool with optional reranking support.

    This is the default tool for all vector search operations. It performs similarity
    search using the parent class, then optionally reranks results if reranking is configured.

    Workflow:
    1. Perform vector similarity search (always)
    2. If reranking enabled: rerank candidates using cross-encoder model
    3. Return final documents

    Both retrieval steps are traced in MLflow for observability.
    """

    reranker_model: Optional[str] = Field(
        default=None,
        description="FlashRank model name for reranking. If None, reranking is disabled.",
    )
    reranker_top_n: Optional[int] = Field(
        default=None, description="Number of documents to return after reranking."
    )
    reranker_cache_dir: Optional[str] = Field(
        default="/tmp/flashrank_cache",
        description="Directory to cache model weights.",
    )

    @mlflow.trace(name="rerank_documents", span_type=SpanType.RETRIEVER)
    def _rerank_documents(
        self, query: str, documents: List[Document]
    ) -> List[Document]:
        """
        Rerank documents using FlashRank.

        This method is traced separately in MLflow for observability.

        Args:
            query: The search query
            documents: List of documents to rerank

        Returns:
            Reranked and filtered list of documents
        """
        from flashrank import Ranker, RerankRequest

        logger.debug(
            f"Starting reranking for {len(documents)} documents using model '{self.reranker_model}'"
        )

        # Log input to MLflow trace
        mlflow.log_text(
            f"Reranking {len(documents)} candidates with model '{self.reranker_model}'",
            "reranking_info.txt",
        )

        # Initialize FlashRank ranker
        try:
            ranker: Ranker = Ranker(
                model_name=self.reranker_model, cache_dir=self.reranker_cache_dir
            )
            logger.debug(
                f"FlashRank ranker initialized (model: {self.reranker_model}, cache: {self.reranker_cache_dir})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize FlashRank ranker: {e}")
            logger.warning("Returning original documents without reranking")
            return documents

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
        top_n: int = self.reranker_top_n or len(documents)
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

        logger.info(
            f"Reranked {len(documents)} documents → {len(reranked_docs)} results "
            f"(model: {self.reranker_model}, top score: {reranked_docs[0].metadata.get('reranker_score', 0):.4f})"
            if reranked_docs
            else f"Reranking completed with {len(reranked_docs)} results"
        )

        return reranked_docs

    @mlflow.trace(name="find_documents", span_type=SpanType.RETRIEVER)
    def _find_documents(
        self, query: str, filters: Optional[List[FilterItem]] = None, **kwargs
    ) -> str:
        kwargs = {**kwargs, **(self.model_extra or {})}
        # Since LLM can generate either a dict or FilterItem, convert to dict always
        filters_dict = {
            dict(item)["key"]: dict(item)["value"] for item in (filters or [])
        }
        combined_filters = {**filters_dict, **(self.filters or {})}

        # Allow kwargs to override the default values upon invocation
        num_results = kwargs.pop("k", self.num_results)
        query_type = kwargs.pop("query_type", self.query_type)

        # Ensure that we don't have duplicate keys
        kwargs.update(
            {
                "query": query,
                "k": num_results,
                "filter": combined_filters,
                "query_type": query_type,
            }
        )
        return self._vector_store.similarity_search(**kwargs)

    @vector_search_retriever_tool_trace
    def _run(
        self, query: str, filters: Optional[List[FilterItem]] = None, **kwargs
    ) -> List[Document]:
        """
        Execute the retrieval with optional reranking.

        This method performs two-stage retrieval:
        1. Vector similarity search (always, traced in MLflow)
        2. Reranking (if enabled, traced separately in MLflow)

        Args:
            query: Search query string
            filters: Optional filters to apply
            **kwargs: Additional search parameters

        Returns:
            List of Document objects, reranked if reranking is enabled
        """
        # Step 1: Always perform vector similarity search using parent implementation
        # This is automatically traced by @vector_search_retriever_tool_trace decorator
        logger.debug(
            f"Executing vector similarity search for tool '{self.name}' (reranking: {self.reranker_model is not None})"
        )

        # Call parent's _run to get similarity search results
        # This will be traced as the primary retrieval span
        documents: List[Document] = self._find_documents(query, filters, **kwargs)

        logger.info(
            f"Retrieved {len(documents)} documents from vector search (tool: '{self.name}')"
        )

        # Step 2: If reranking is enabled, rerank the documents
        if self.reranker_model:
            logger.info(
                f"Reranking enabled (model: '{self.reranker_model}', top_n: {self.reranker_top_n or 'all'})"
            )
            # This will be traced separately in its own MLflow span
            reranked_docs: List[Document] = self._rerank_documents(query, documents)
            logger.info(
                f"Returning {len(reranked_docs)} reranked documents (from {len(documents)} candidates)"
            )
            return reranked_docs

        # No reranking - return original documents
        logger.debug("Reranking disabled, returning original vector search results")
        return documents


def create_vector_search_tool(
    retriever: RetrieverModel | dict[str, Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> BaseTool:
    """
    Create a Vector Search tool for retrieving documents from a Databricks Vector Search index.

    This function creates a tool that enables semantic search over product information,
    documentation, or other content. It also registers the retriever schema with MLflow
    for proper integration with the model serving infrastructure.

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

    Returns:
        A BaseTool instance that can perform vector search operations
    """

    if isinstance(retriever, dict):
        retriever = RetrieverModel(**retriever)

    vector_store: VectorStoreModel = retriever.vector_store

    # Index is required for vector search
    if vector_store.index is None:
        raise ValueError("vector_store.index is required for vector search")

    index_name: str = vector_store.index.full_name
    columns: Sequence[str] = retriever.columns or []
    search_parameters: dict[str, Any] = retriever.search_parameters.model_dump()
    primary_key: str = vector_store.primary_key or ""
    doc_uri: str = vector_store.doc_uri or ""
    text_column: str = vector_store.embedding_source_column

    # Always use RerankingVectorSearchRetrieverTool (it handles both cases)
    reranker_config: Optional[RerankerParametersModel] = (
        retriever.reranker
        if isinstance(retriever.reranker, RerankerParametersModel)
        else None
    )

    if reranker_config:
        logger.info(
            f"Creating vector search tool with reranking: '{name}' "
            f"(model: {reranker_config.model}, top_n: {reranker_config.top_n or 'auto'})"
        )
    else:
        logger.debug(
            f"Creating vector search tool without reranking: '{name}' (standard similarity search only)"
        )

    # Build tool kwargs
    tool_kwargs: dict[str, Any] = {
        "name": name,
        "tool_name": name,
        "description": description,
        "tool_description": description,
        "index_name": index_name,
        "columns": columns,
        **search_parameters,
        "workspace_client": vector_store.workspace_client,
    }

    # Add reranking parameters if configured
    if reranker_config:
        tool_kwargs.update(
            {
                "reranker_model": reranker_config.model,
                "reranker_top_n": reranker_config.top_n,
                "reranker_cache_dir": reranker_config.cache_dir,
            }
        )

    # Always use RerankingVectorSearchRetrieverTool (handles both with/without reranking)
    vector_search_tool: BaseTool = RerankingVectorSearchRetrieverTool(**tool_kwargs)

    # Register the retriever schema with MLflow for model serving integration
    mlflow.models.set_retriever_schema(
        name=name or "retriever",
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        other_columns=list(columns),
    )

    return vector_search_tool
