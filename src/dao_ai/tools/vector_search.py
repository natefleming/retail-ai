from typing import Any, Optional, Sequence

import mlflow
from databricks_langchain.vector_search_retriever_tool import VectorSearchRetrieverTool
from langchain_core.tools import BaseTool

from dao_ai.config import (
    RetrieverModel,
    VectorStoreModel,
)


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

    index_name: str = vector_store.index.full_name
    columns: Sequence[str] = retriever.columns
    search_parameters: dict[str, Any] = retriever.search_parameters.model_dump()
    primary_key: str = vector_store.primary_key
    doc_uri: str = vector_store.doc_uri
    text_column: str = vector_store.embedding_source_column

    vector_search_tool: BaseTool = VectorSearchRetrieverTool(
        name=name,
        description=description,
        index_name=index_name,
        columns=columns,
        **search_parameters,
        workspace_client=vector_store.workspace_client,
    )

    # Register the retriever schema with MLflow for model serving integration
    mlflow.models.set_retriever_schema(
        name=name or "retriever",
        primary_key=primary_key,
        text_column=text_column,
        doc_uri=doc_uri,
        other_columns=columns,
    )

    return vector_search_tool
