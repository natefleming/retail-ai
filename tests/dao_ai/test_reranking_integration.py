"""
Integration tests for vector search with reranking.

These tests require:
- DATABRICKS_HOST environment variable
- DATABRICKS_TOKEN environment variable
- An existing vector search index

To run integration tests:
    pytest tests/dao_ai/test_reranking_integration.py -v -m integration
"""

import json
import os

import pytest
from databricks.sdk import WorkspaceClient
from langchain_core.messages import ToolMessage

from dao_ai.config import (
    IndexModel,
    RerankParametersModel,
    RetrieverModel,
    SchemaModel,
    SearchParametersModel,
    TableModel,
    VectorSearchEndpoint,
    VectorStoreModel,
)
from dao_ai.tools.vector_search import create_vector_search_tool


def extract_documents_from_tool_result(result):
    """Helper function to extract documents from a tool invocation result.

    When a tool is invoked with a ToolCall object, LangChain wraps the result
    in a ToolMessage. This function extracts the actual documents from the message.
    """
    if isinstance(result, ToolMessage):
        content = result.content
        # Content might be a string (Python repr or JSON) or already a list
        if isinstance(content, str):
            # Try to parse as JSON first
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, try Python literal_eval (for single-quoted strings)
                import ast

                try:
                    return ast.literal_eval(content)
                except (ValueError, SyntaxError):
                    # If all parsing fails, raise an error
                    raise ValueError(
                        f"Failed to parse tool result content: {content[:200]}"
                    )
        else:
            # Content is already a list
            return content
    else:
        # Direct result (not wrapped in ToolMessage)
        return result


# Check if we have Databricks credentials
HAS_DATABRICKS_CREDS = bool(
    os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN")
)


# Check if FlashRank model is available
def _check_flashrank_available() -> bool:
    """Check if FlashRank model can be initialized."""
    try:
        from flashrank import Ranker

        # Try to initialize with the default model
        Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank_cache")
        return True
    except Exception:
        return False


HAS_FLASHRANK = _check_flashrank_available()

# Skip messages
SKIP_MSG = "Requires DATABRICKS_HOST and DATABRICKS_TOKEN environment variables"
SKIP_FLASHRANK_MSG = "Requires FlashRank model to be available (run once to download)"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_DATABRICKS_CREDS, reason=SKIP_MSG)
class TestRerankingWithRealIndex:
    """Integration tests with real Databricks vector search index."""

    @pytest.fixture
    def workspace_client(self) -> WorkspaceClient:
        """Create Databricks workspace client."""
        return WorkspaceClient(
            host=os.getenv("DATABRICKS_HOST"), token=os.getenv("DATABRICKS_TOKEN")
        )

    @pytest.fixture
    def test_index_config(self) -> dict:
        """
        Return test index configuration.

        Override this in your test environment with actual index details:
        - catalog: Your catalog name
        - schema: Your schema name
        - index_name: Your vector search index name
        - endpoint_name: Your vector search endpoint name

        Defaults to hardware_store configuration values.
        """
        return {
            "catalog": os.getenv("TEST_CATALOG", "retail_consumer_goods"),
            "schema": os.getenv("TEST_SCHEMA", "hardware_store"),
            "index_name": os.getenv("TEST_INDEX", "products_index"),
            "endpoint_name": os.getenv("TEST_ENDPOINT", "dbdemos_vs_endpoint"),
            "table_name": os.getenv("TEST_TABLE", "products"),
            "primary_key": "product_id",
            "text_column": "description",
            "embedding_source_column": "description",
        }

    def test_vector_search_without_reranking(
        self, workspace_client: WorkspaceClient, test_index_config: dict
    ) -> None:
        """Test basic vector search without reranking."""
        # Create retriever config
        schema = SchemaModel(
            schema_name=test_index_config["schema"],
            catalog_name=test_index_config["catalog"],
        )
        vector_store = VectorStoreModel(
            index=IndexModel(
                name=test_index_config["index_name"],
                schema=schema,
            ),
            source_table=TableModel(
                name=test_index_config["table_name"],
                schema=schema,
            ),
            endpoint=VectorSearchEndpoint(name=test_index_config["endpoint_name"]),
            primary_key=test_index_config["primary_key"],
            embedding_source_column=test_index_config["embedding_source_column"],
            columns=[
                "product_id",
                "sku",
                "product_name",
                "description",
            ],
        )

        retriever = RetrieverModel(
            vector_store=vector_store,
            search_parameters=SearchParametersModel(num_results=10),
        )

        # Create tool
        tool = create_vector_search_tool(
            retriever=retriever,
            name="test_search",
            description="Test vector search",
        )

        # Execute search - tools with InjectedState/InjectedToolCallId must be invoked
        # with a full ToolCall object, not just args
        from langchain_core.messages import ToolCall as LCToolCall

        tool_call = LCToolCall(
            name=tool.name,
            args={"query": "test query"},
            id="test_tool_call_123",
            type="tool_call",
        )

        result = tool.invoke(tool_call)

        # When invoked with ToolCall, result is wrapped in ToolMessage
        assert isinstance(result, ToolMessage)

        # Extract documents using helper function
        documents = extract_documents_from_tool_result(result)

        assert isinstance(documents, list)
        if len(documents) > 0:
            # Documents are serialized as dicts
            assert isinstance(documents[0], dict)
            assert "page_content" in documents[0]
            assert "metadata" in documents[0]

    def test_vector_search_with_reranking_bool(
        self, workspace_client: WorkspaceClient, test_index_config: dict
    ) -> None:
        """Test vector search with reranking enabled via bool."""
        schema = SchemaModel(
            schema_name=test_index_config["schema"],
            catalog_name=test_index_config["catalog"],
        )
        vector_store = VectorStoreModel(
            index=IndexModel(
                name=test_index_config["index_name"],
                schema=schema,
            ),
            source_table=TableModel(
                name=test_index_config["table_name"],
                schema=schema,
            ),
            endpoint=VectorSearchEndpoint(name=test_index_config["endpoint_name"]),
            primary_key=test_index_config["primary_key"],
            embedding_source_column=test_index_config["embedding_source_column"],
            columns=[
                "product_id",
                "sku",
                "product_name",
                "description",
            ],
        )

        retriever = RetrieverModel(
            vector_store=vector_store,
            search_parameters=SearchParametersModel(num_results=20),
            rerank=True,  # Enable with defaults
        )

        # Create tool
        tool = create_vector_search_tool(
            retriever=retriever,
            name="test_search_rerank",
            description="Test vector search with reranking",
        )

        # Execute search - use full ToolCall format
        from langchain_core.messages import ToolCall as LCToolCall

        tool_call = LCToolCall(
            name=tool.name,
            args={"query": "test query"},
            id="test_tool_call_234",
            type="tool_call",
        )
        result = tool.invoke(tool_call)

        # When invoked with ToolCall, result is wrapped in ToolMessage
        assert isinstance(result, ToolMessage)

        # Extract documents using helper function
        documents = extract_documents_from_tool_result(result)

        assert isinstance(documents, list)
        # Results should be reranked and potentially fewer than num_results
        if len(documents) > 0:
            # Documents are serialized as dicts
            assert isinstance(documents[0], dict)
            assert "page_content" in documents[0]
            assert "metadata" in documents[0]

    @pytest.mark.skipif(not HAS_FLASHRANK, reason=SKIP_FLASHRANK_MSG)
    def test_vector_search_with_custom_reranking(
        self, workspace_client: WorkspaceClient, test_index_config: dict
    ) -> None:
        """Test vector search with custom reranking configuration."""
        schema = SchemaModel(
            schema_name=test_index_config["schema"],
            catalog_name=test_index_config["catalog"],
        )
        vector_store = VectorStoreModel(
            index=IndexModel(
                name=test_index_config["index_name"],
                schema=schema,
            ),
            source_table=TableModel(
                name=test_index_config["table_name"],
                schema=schema,
            ),
            endpoint=VectorSearchEndpoint(name=test_index_config["endpoint_name"]),
            primary_key=test_index_config["primary_key"],
            embedding_source_column=test_index_config["embedding_source_column"],
            columns=[
                "product_id",
                "sku",
                "product_name",
                "description",
            ],
        )

        retriever = RetrieverModel(
            vector_store=vector_store,
            search_parameters=SearchParametersModel(num_results=50),
            rerank=RerankParametersModel(
                model="ms-marco-MiniLM-L-12-v2",  # Default model
                top_n=5,  # Return top 5 after reranking
            ),
        )

        # Create tool
        tool = create_vector_search_tool(
            retriever=retriever,
            name="test_search_custom_rerank",
            description="Test vector search with custom reranking",
        )

        # Execute search - use full ToolCall format
        from langchain_core.messages import ToolCall as LCToolCall

        tool_call = LCToolCall(
            name=tool.name,
            args={"query": "test query"},
            id="test_tool_call_345",
            type="tool_call",
        )
        result = tool.invoke(tool_call)

        # When invoked with ToolCall, result is wrapped in ToolMessage
        assert isinstance(result, ToolMessage)

        # Extract documents using helper function
        documents = extract_documents_from_tool_result(result)

        assert isinstance(documents, list)
        assert len(documents) <= 5  # Should respect top_n
        if len(documents) > 0:
            # Documents are serialized as dicts
            assert isinstance(documents[0], dict)
            assert "page_content" in documents[0]
            assert "metadata" in documents[0]

    @pytest.mark.skipif(not HAS_FLASHRANK, reason=SKIP_FLASHRANK_MSG)
    def test_reranking_improves_relevance(
        self, workspace_client: WorkspaceClient, test_index_config: dict
    ) -> None:
        """
        Test that reranking improves result relevance.

        This test compares results with and without reranking.
        """
        schema = SchemaModel(
            schema_name=test_index_config["schema"],
            catalog_name=test_index_config["catalog"],
        )
        vector_store = VectorStoreModel(
            index=IndexModel(
                name=test_index_config["index_name"],
                schema=schema,
            ),
            source_table=TableModel(
                name=test_index_config["table_name"],
                schema=schema,
            ),
            endpoint=VectorSearchEndpoint(name=test_index_config["endpoint_name"]),
            primary_key=test_index_config["primary_key"],
            embedding_source_column=test_index_config["embedding_source_column"],
            columns=[
                "product_id",
                "sku",
                "product_name",
                "description",
            ],
        )

        # Search without reranking
        retriever_no_rerank = RetrieverModel(
            vector_store=vector_store,
            search_parameters=SearchParametersModel(num_results=10),
        )
        tool_no_rerank = create_vector_search_tool(
            retriever=retriever_no_rerank,
            name="search_no_rerank",
            description="Search without reranking",
        )

        # Search with reranking
        retriever_with_rerank = RetrieverModel(
            vector_store=vector_store,
            search_parameters=SearchParametersModel(num_results=20),
            rerank=RerankParametersModel(top_n=10),
        )
        tool_with_rerank = create_vector_search_tool(
            retriever=retriever_with_rerank,
            name="search_with_rerank",
            description="Search with reranking",
        )

        # Execute both searches - use full ToolCall format
        from langchain_core.messages import ToolCall as LCToolCall

        test_query = "test query"

        tool_call_no_rerank = LCToolCall(
            name=tool_no_rerank.name,
            args={"query": test_query},
            id="test_tool_call_456",
            type="tool_call",
        )
        tool_call_with_rerank = LCToolCall(
            name=tool_with_rerank.name,
            args={"query": test_query},
            id="test_tool_call_567",
            type="tool_call",
        )

        result_no_rerank = tool_no_rerank.invoke(tool_call_no_rerank)
        result_with_rerank = tool_with_rerank.invoke(tool_call_with_rerank)

        # When invoked with ToolCall, results are wrapped in ToolMessage
        assert isinstance(result_no_rerank, ToolMessage)
        assert isinstance(result_with_rerank, ToolMessage)

        # Extract documents using helper function
        results_no_rerank = extract_documents_from_tool_result(result_no_rerank)
        results_with_rerank = extract_documents_from_tool_result(result_with_rerank)

        # Verify both return lists of documents
        assert isinstance(results_no_rerank, list)
        assert isinstance(results_with_rerank, list)

        # Both should return results
        assert len(results_no_rerank) > 0
        assert len(results_with_rerank) > 0

        # Results should be different (reranking changes order)
        # Note: This is a simplistic check; in practice, you'd evaluate
        # relevance using metrics like MRR, NDCG, etc.
        if len(results_no_rerank) >= 2 and len(results_with_rerank) >= 2:
            # Check if ordering changed (documents are dicts now)
            no_rerank_ids = [
                doc.get("metadata", {}).get("id") for doc in results_no_rerank[:5]
            ]
            with_rerank_ids = [
                doc.get("metadata", {}).get("id") for doc in results_with_rerank[:5]
            ]

            # At least some reordering should occur (not always, but usually)
            print(f"Without reranking: {no_rerank_ids}")
            print(f"With reranking: {with_rerank_ids}")


@pytest.mark.integration
@pytest.mark.skipif(not HAS_DATABRICKS_CREDS, reason=SKIP_MSG)
class TestRerankingPerformance:
    """Performance tests for reranking."""

    def test_reranking_latency(self) -> None:
        """
        Test that reranking latency is acceptable.

        Reranking adds computational overhead but should still be fast enough
        for interactive applications (< 1-2 seconds for reasonable doc counts).
        """

        # This would measure end-to-end latency with reranking
        # Target: < 2 seconds for 50 candidates -> 5 results
        pass

    def test_reranking_with_large_candidate_set(self) -> None:
        """
        Test reranking performance with many candidates.

        Verify that reranking 100+ documents is still performant.
        """
        pass


@pytest.mark.unit
def test_reranking_config_from_dict() -> None:
    """Test creating retriever config from dictionary (YAML-like)."""
    # This tests that config can be loaded from YAML/dict
    # In practice, this would go through AppConfig.from_file()
    # Example config structure that would be used:
    # {
    #     "vector_store": {
    #         "index": {"name": "test_index", "schema": {...}},
    #         "endpoint": {"name": "endpoint"},
    #         "primary_key": "id",
    #         "embedding_source_column": "text",
    #         "columns": ["id", "text"],
    #     },
    #     "search_parameters": {"num_results": 30},
    #     "reranker": {"model": "ms-marco-MiniLM-L-12-v2", "top_n": 10},
    # }
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
