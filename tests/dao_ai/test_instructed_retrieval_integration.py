"""
Integration tests for instructed retrieval with query decomposition, RRF merging,
and instruction-aware reranking against live Databricks.

Tests load config/examples/03_reranking/instruction_aware_reranking.yaml and
validate the full instructed retrieval pipeline including the empty-result
fallback fix.

Requires:
- DATABRICKS_HOST environment variable
- DATABRICKS_TOKEN environment variable
- The vector search index referenced in the config must exist

To run:
    pytest tests/dao_ai/test_instructed_retrieval_integration.py -v -m integration -s
"""

import json
import os

import pytest
from langchain_core.messages import ToolCall as LCToolCall
from langchain_core.messages import ToolMessage

from dao_ai.config import AppConfig
from dao_ai.tools.vector_search import create_vector_search_tool

CONFIG_PATH = "config/examples/03_reranking/instruction_aware_reranking.yaml"

HAS_DATABRICKS_CREDS = bool(
    os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN")
)
SKIP_MSG = "Requires DATABRICKS_HOST and DATABRICKS_TOKEN environment variables"


def extract_documents_from_tool_result(result: ToolMessage) -> list[dict]:
    """Extract parsed documents from a ToolMessage result.

    When a tool is invoked with a ToolCall object, LangChain wraps the result
    in a ToolMessage. This function extracts the actual documents from the message.
    """
    if isinstance(result, ToolMessage):
        content = result.content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                import ast

                try:
                    return ast.literal_eval(content)
                except (ValueError, SyntaxError):
                    raise ValueError(
                        f"Failed to parse tool result content: {content[:200]}"
                    )
        return content
    return result


@pytest.mark.integration
@pytest.mark.skipif(not HAS_DATABRICKS_CREDS, reason=SKIP_MSG)
class TestInstructedRetrievalIntegration:
    """Integration tests for the full instructed retrieval pipeline."""

    @pytest.fixture(scope="class")
    def config(self) -> AppConfig:
        """Load the instruction_aware_reranking.yaml config."""
        return AppConfig.from_file(CONFIG_PATH)

    @pytest.fixture(scope="class")
    def tool(self, config: AppConfig):
        """Create the instructed retrieval vector search tool from config."""
        retriever = config.retrievers["instruction_aware_retriever"]
        return create_vector_search_tool(
            retriever=retriever,
            name="test_instructed_search",
            description="Integration test for instructed retrieval",
        )

    @pytest.fixture(scope="class")
    def milwaukee_results(self, tool) -> list[dict]:
        """Shared fixture: invoke tool with 'milwaukee drills' once for multiple tests."""
        tool_call = LCToolCall(
            name=tool.name,
            args={"query": "milwaukee drills"},
            id="test_milwaukee_001",
            type="tool_call",
        )
        result = tool.invoke(tool_call)
        return extract_documents_from_tool_result(result)

    def test_instructed_retrieval_returns_results(
        self, milwaukee_results: list[dict]
    ) -> None:
        """Instructed retrieval returns non-empty results with RRF and rerank metadata."""
        assert len(milwaukee_results) > 0, "Instructed retrieval returned no results"

        for doc in milwaukee_results:
            assert "page_content" in doc, "Document missing page_content"
            assert "metadata" in doc, "Document missing metadata"

        # rrf_score proves decomposition + parallel search + RRF merge ran
        first_meta = milwaukee_results[0]["metadata"]
        assert "rrf_score" in first_meta, (
            "First result missing rrf_score — decomposition/RRF path may not have run"
        )

        # instruction_rerank_score proves instruction-aware reranking ran
        assert "instruction_rerank_score" in first_meta, (
            "First result missing instruction_rerank_score — "
            "instruction-aware reranking may not have run"
        )

    def test_instructed_retrieval_result_count(
        self, milwaukee_results: list[dict]
    ) -> None:
        """Result count respects the instruction reranker top_n (10)."""
        assert len(milwaukee_results) <= 10, (
            f"Expected at most 10 results (instructed.rerank.top_n), got {len(milwaukee_results)}"
        )

    def test_instructed_retrieval_brand_relevance(
        self, milwaukee_results: list[dict]
    ) -> None:
        """Majority of results have brand_name MILWAUKEE (validates filter extraction)."""
        milwaukee_count = sum(
            1
            for doc in milwaukee_results
            if doc.get("metadata", {}).get("brand_name", "").upper() == "MILWAUKEE"
        )
        total = len(milwaukee_results)
        ratio = milwaukee_count / total if total > 0 else 0

        print(f"\nBrand relevance: {milwaukee_count}/{total} MILWAUKEE ({ratio:.0%})")
        assert ratio >= 0.5, (
            f"Expected majority MILWAUKEE results, got {milwaukee_count}/{total} ({ratio:.0%})"
        )

    def test_empty_result_fallback(self, config: AppConfig) -> None:
        """When decomposed filters match nothing, fallback to standard search returns results."""
        retriever = config.retrievers["instruction_aware_retriever"]

        # Override decomposition examples with impossible filter values
        # to force all filtered subqueries to return empty results
        modified_retriever = retriever.model_copy(deep=True)
        modified_retriever.instructed.decomposition.examples = [
            {
                "query": "nonexistent brand",
                "filters": {"brand_name": "ZZZZNONEXISTENT"},
            },
            {
                "query": "impossible category",
                "filters": {"merchandise_class": "ZZZZFAKE"},
            },
        ]
        modified_retriever.instructed.constraints = [
            "Always filter by brand_name",
            "Only use columns listed",
        ]
        # Disable instruction-aware reranking so we can cleanly check fallback
        modified_retriever.instructed.rerank = None
        # Disable FlashRank reranking too
        modified_retriever.rerank = None

        tool = create_vector_search_tool(
            retriever=modified_retriever,
            name="test_fallback_search",
            description="Test empty-result fallback",
        )

        tool_call = LCToolCall(
            name=tool.name,
            args={"query": "ZZZZNONEXISTENT brand products"},
            id="test_fallback_001",
            type="tool_call",
        )
        result = tool.invoke(tool_call)
        documents = extract_documents_from_tool_result(result)

        assert len(documents) > 0, (
            "Fallback failed: instructed retrieval returned empty results "
            "even after the empty-result fallback should have triggered"
        )

        # Fallback results come from standard similarity_search,
        # so they should NOT have rrf_score
        first_meta = documents[0]["metadata"]
        assert "rrf_score" not in first_meta, (
            "Fallback results should not have rrf_score — "
            "standard search path should have been used"
        )
        print(f"\nFallback returned {len(documents)} results via standard search")

    def test_tool_returns_valid_json(self, tool) -> None:
        """Tool result is a ToolMessage with valid JSON content."""
        tool_call = LCToolCall(
            name=tool.name,
            args={"query": "cordless power tools"},
            id="test_json_001",
            type="tool_call",
        )
        result = tool.invoke(tool_call)

        assert isinstance(result, ToolMessage), (
            f"Expected ToolMessage, got {type(result).__name__}"
        )

        # Content must be valid JSON
        content = result.content
        assert isinstance(content, str), "ToolMessage content should be a string"
        parsed = json.loads(content)
        assert isinstance(parsed, list), "Parsed content should be a list"
        assert len(parsed) > 0, "Parsed content should not be empty"

        for doc in parsed:
            assert isinstance(doc, dict), (
                f"Each entry should be a dict, got {type(doc).__name__}"
            )
            assert "page_content" in doc, "Entry missing page_content"
            assert isinstance(doc["page_content"], str), (
                "page_content should be a string"
            )
            assert "metadata" in doc, "Entry missing metadata"
            assert isinstance(doc["metadata"], dict), "metadata should be a dict"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration", "-s"])
