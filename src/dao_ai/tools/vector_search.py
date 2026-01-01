"""
Vector search tool for retrieving documents from Databricks Vector Search.

This module provides a tool factory for creating semantic search tools
with dynamic filter schemas based on table columns and FlashRank reranking support.
"""

import json
import os
from typing import Any, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.vector_search.reranker import DatabricksReranker
from databricks_langchain import DatabricksVectorSearch
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from loguru import logger
from mlflow.entities import SpanType
from pydantic import BaseModel, ConfigDict, Field, create_model

from dao_ai.config import (
    RerankParametersModel,
    RetrieverModel,
    SearchParametersModel,
    VectorStoreModel,
    value_of,
)
from dao_ai.utils import normalize_host

# Create FilterItem model at module level so it can be used in type hints
FilterItem = create_model(
    "FilterItem",
    key=(
        str,
        Field(
            description="The filter key, which includes the column name and can include operators like 'NOT', '<', '>=', 'LIKE', 'OR'"
        ),
    ),
    value=(
        Any,
        Field(
            description="The filter value, which can be a single value or an array of values"
        ),
    ),
    __config__=ConfigDict(extra="forbid"),
)


def _create_dynamic_input_schema(
    index_name: str, workspace_client: WorkspaceClient
) -> type[BaseModel]:
    """
    Create dynamic input schema with column information from the table.

    Args:
        index_name: Full name of the vector search index
        workspace_client: Workspace client to query table metadata

    Returns:
        Pydantic model class for tool input
    """

    # Try to get column information
    column_descriptions = []
    try:
        table_info = workspace_client.tables.get(full_name=index_name)
        for column_info in table_info.columns:
            name = column_info.name
            col_type = column_info.type_name.name
            if not name.startswith("__"):
                column_descriptions.append(f"{name} ({col_type})")
    except Exception:
        logger.debug(
            "Could not retrieve column information for dynamic schema",
            index=index_name,
        )

    # Build filter description matching VectorSearchRetrieverTool format
    filter_description = (
        "Optional filters to refine vector search results as an array of key-value pairs. "
        "IMPORTANT: If unsure about filter values, try searching WITHOUT filters first to get broad results, "
        "then optionally add filters to narrow down if needed. This ensures you don't miss relevant results due to incorrect filter values. "
    )

    if column_descriptions:
        filter_description += (
            f"Available columns for filtering: {', '.join(column_descriptions)}. "
        )

    filter_description += (
        "Supports the following operators:\n\n"
        '- Inclusion: [{"key": "column", "value": value}] or [{"key": "column", "value": [value1, value2]}] (matches if the column equals any of the provided values)\n'
        '- Exclusion: [{"key": "column NOT", "value": value}]\n'
        '- Comparisons: [{"key": "column <", "value": value}], [{"key": "column >=", "value": value}], etc.\n'
        '- Pattern match: [{"key": "column LIKE", "value": "word"}] (matches full tokens separated by whitespace)\n'
        '- OR logic: [{"key": "column1 OR column2", "value": [value1, value2]}] '
        "(matches if column1 equals value1 or column2 equals value2; matches are position-specific)\n\n"
        "Examples:\n"
        '- Filter by category: [{"key": "category", "value": "electronics"}]\n'
        '- Filter by price range: [{"key": "price >=", "value": 100}, {"key": "price <", "value": 500}]\n'
        '- Exclude specific status: [{"key": "status NOT", "value": "archived"}]\n'
        '- Pattern matching: [{"key": "description LIKE", "value": "wireless"}]'
    )

    # Create the input model
    VectorSearchInput = create_model(
        "VectorSearchInput",
        query=(
            str,
            Field(description="The search query string to find relevant documents"),
        ),
        filters=(
            Optional[list[FilterItem]],
            Field(default=None, description=filter_description),
        ),
        __config__=ConfigDict(extra="forbid"),
    )

    return VectorSearchInput


@mlflow.trace(name="rerank_documents", span_type=SpanType.RETRIEVER)
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
    columns: list[str] = list(retriever.columns or [])
    search_parameters: SearchParametersModel = retriever.search_parameters
    rerank_config: Optional[RerankParametersModel] = retriever.rerank

    # Initialize FlashRank ranker if configured
    ranker: Optional[Ranker] = None
    if rerank_config and rerank_config.model:
        logger.debug(
            "Initializing FlashRank ranker",
            model=rerank_config.model,
            top_n=rerank_config.top_n or "auto",
        )
        try:
            cache_dir = os.path.expanduser(rerank_config.cache_dir)
            ranker = Ranker(model_name=rerank_config.model, cache_dir=cache_dir)
            logger.success("FlashRank ranker initialized", model=rerank_config.model)
        except Exception as e:
            logger.warning("Failed to initialize FlashRank ranker", error=str(e))
            rerank_config = None

    # Build client_args for VectorSearchClient
    # Use getattr to safely access attributes that may not exist (e.g., in mocks)
    client_args: dict[str, Any] = {}
    has_explicit_auth = any(
        [
            os.environ.get("DATABRICKS_TOKEN"),
            os.environ.get("DATABRICKS_CLIENT_ID"),
            getattr(vector_store, "pat", None),
            getattr(vector_store, "client_id", None),
            getattr(vector_store, "on_behalf_of_user", None),
        ]
    )

    if has_explicit_auth:
        databricks_host = os.environ.get("DATABRICKS_HOST")
        if (
            not databricks_host
            and getattr(vector_store, "_workspace_client", None) is not None
        ):
            databricks_host = vector_store.workspace_client.config.host
        if databricks_host:
            client_args["workspace_url"] = normalize_host(databricks_host)

        token = os.environ.get("DATABRICKS_TOKEN")
        if not token and getattr(vector_store, "pat", None):
            token = value_of(vector_store.pat)
        if token:
            client_args["personal_access_token"] = token

        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        if not client_id and getattr(vector_store, "client_id", None):
            client_id = value_of(vector_store.client_id)
        if client_id:
            client_args["service_principal_client_id"] = client_id

        client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
        if not client_secret and getattr(vector_store, "client_secret", None):
            client_secret = value_of(vector_store.client_secret)
        if client_secret:
            client_args["service_principal_client_secret"] = client_secret

    logger.debug(
        "Creating vector search tool",
        name=name,
        index=index_name,
        client_args_keys=list(client_args.keys()) if client_args else [],
    )

    # Create DatabricksVectorSearch
    # Note: text_column should be None for Databricks-managed embeddings
    # (it's automatically determined from the index)
    vector_search: DatabricksVectorSearch = DatabricksVectorSearch(
        index_name=index_name,
        text_column=None,
        columns=columns,
        workspace_client=vector_store.workspace_client,
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

    # Create dynamic input schema
    input_schema: type[BaseModel] = _create_dynamic_input_schema(
        index_name, vector_store.workspace_client
    )

    # Define the tool function
    def vector_search_func(
        query: str, filters: Optional[list[FilterItem]] = None
    ) -> str:
        """Search for relevant documents using vector similarity."""
        # Convert FilterItem Pydantic models to dict format for DatabricksVectorSearch
        filters_dict: dict[str, Any] = {}
        if filters:
            for item in filters:
                filters_dict[item.key] = item.value

        # Merge with configured filters
        combined_filters: dict[str, Any] = {
            **filters_dict,
            **(search_parameters.filters or {}),
        }

        # Perform vector search
        logger.trace("Performing vector search", query_preview=query[:50])
        documents: list[Document] = vector_search.similarity_search(
            query=query,
            k=search_parameters.num_results or 5,
            filter=combined_filters if combined_filters else None,
            query_type=search_parameters.query_type or "ANN",
        )

        # Apply FlashRank reranking if configured
        if ranker and rerank_config:
            logger.debug("Applying FlashRank reranking")
            documents = _rerank_documents(query, documents, ranker, rerank_config)

        # Serialize documents to JSON format for LLM consumption
        # Convert Document objects to dicts with page_content and metadata
        # Need to handle numpy types in metadata (e.g., float32, int64)
        serialized_docs: list[dict[str, Any]] = []
        for doc in documents:
            doc: Document
            # Convert metadata values to JSON-serializable types
            metadata_serializable: dict[str, Any] = {}
            for key, value in doc.metadata.items():
                # Handle numpy types
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

        # Return as JSON string
        return json.dumps(serialized_docs)

    # Create the StructuredTool
    tool: StructuredTool = StructuredTool.from_function(
        func=vector_search_func,
        name=name or f"vector_search_{vector_store.index.name}",
        description=description or f"Search documents in {index_name}",
        args_schema=input_schema,
    )

    logger.success("Vector search tool created", name=tool.name, index=index_name)

    return tool
