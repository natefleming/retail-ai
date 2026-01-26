"""Unit tests for instruction-aware reranker."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from dao_ai.config import InstructionAwareRerankModel, RankedDocument, RankingResult
from dao_ai.tools.instruction_reranker import (
    _format_documents,
    instruction_aware_rerank,
)


@pytest.mark.unit
class TestInstructionAwareRerankModel:
    """Unit tests for InstructionAwareRerankModel configuration."""

    def test_default_values(self) -> None:
        """Test that InstructionAwareRerankModel has sensible defaults."""
        model = InstructionAwareRerankModel()
        assert model.model is None
        assert model.instructions is None
        assert model.top_n is None

    def test_custom_values(self) -> None:
        """Test InstructionAwareRerankModel with custom configuration."""
        model = InstructionAwareRerankModel(
            instructions="Prioritize price constraints",
            top_n=10,
        )
        assert model.instructions == "Prioritize price constraints"
        assert model.top_n == 10


@pytest.mark.unit
class TestRankedDocument:
    """Unit tests for RankedDocument structured output."""

    def test_basic_creation(self) -> None:
        """Test creating a RankedDocument with required fields."""
        rd = RankedDocument(index=0, score=0.9, reason="Good match")
        assert rd.index == 0
        assert rd.score == 0.9
        assert rd.reason == "Good match"

    def test_reason_is_optional(self) -> None:
        """Test that reason is optional with default empty string."""
        rd = RankedDocument(index=0, score=0.9)
        assert rd.reason == ""

    def test_score_accepts_any_float(self) -> None:
        """Test that score accepts any float (no validation)."""
        rd1 = RankedDocument(index=0, score=1.5, reason="test")
        assert rd1.score == 1.5
        rd2 = RankedDocument(index=0, score=-0.1, reason="test")
        assert rd2.score == -0.1


@pytest.mark.unit
class TestRankingResult:
    """Unit tests for RankingResult container model."""

    def test_basic_creation(self) -> None:
        """Test creating a RankingResult with rankings."""
        rankings = [
            RankedDocument(index=1, score=0.9, reason="Best match"),
            RankedDocument(index=0, score=0.7, reason="Good match"),
        ]
        result = RankingResult(rankings=rankings)
        assert len(result.rankings) == 2
        assert result.rankings[0].index == 1
        assert result.rankings[1].index == 0

    def test_empty_rankings(self) -> None:
        """Test creating RankingResult with empty list."""
        result = RankingResult(rankings=[])
        assert len(result.rankings) == 0


@pytest.mark.unit
class TestFormatDocuments:
    """Unit tests for _format_documents helper."""

    def test_empty_documents(self) -> None:
        """Test formatting empty document list."""
        result = _format_documents([])
        assert result == "No documents to rerank."

    def test_single_document(self) -> None:
        """Test formatting single document."""
        docs = [
            Document(
                page_content="Test content",
                metadata={"price": 100, "brand": "Milwaukee"},
            )
        ]
        result = _format_documents(docs)
        assert "[0]" in result
        assert "Test content" in result
        assert "price: 100" in result

    def test_excludes_internal_metadata(self) -> None:
        """Test that internal metadata keys are excluded."""
        docs = [
            Document(
                page_content="Test",
                metadata={
                    "brand": "Test",
                    "_verification_status": "passed",
                    "rrf_score": 0.5,
                    "reranker_score": 0.8,
                },
            )
        ]
        result = _format_documents(docs)
        assert "brand: Test" in result
        assert "_verification_status" not in result
        assert "rrf_score" not in result
        assert "reranker_score" not in result

    def test_truncates_long_content(self) -> None:
        """Test that long content is truncated."""
        long_content = "x" * 500
        docs = [Document(page_content=long_content, metadata={})]
        result = _format_documents(docs)
        assert "..." in result
        assert len(result) < len(long_content) + 100


@pytest.mark.unit
class TestInstructionAwareRerank:
    """Unit tests for instruction_aware_rerank function."""

    def _create_mock_ranking_result(self, rankings: list[dict]) -> RankingResult:
        """Create mock RankingResult."""
        return RankingResult(rankings=[RankedDocument(**r) for r in rankings])

    def _setup_mock_llm(self, rankings: list[dict]) -> MagicMock:
        """Set up a mock LLM with with_structured_output behavior."""
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = self._create_mock_ranking_result(
            rankings
        )
        mock_llm.with_structured_output.return_value = mock_structured_llm
        return mock_llm

    def test_empty_documents_returns_empty(self) -> None:
        """Test that empty input returns empty output."""
        mock_llm = MagicMock()
        result = instruction_aware_rerank(
            llm=mock_llm,
            query="test",
            documents=[],
        )
        assert result == []

    @patch("dao_ai.tools.instruction_reranker._load_prompt_template")
    @patch("dao_ai.tools.instruction_reranker.mlflow")
    def test_reranks_documents(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that documents are reranked based on LLM response."""
        mock_load_prompt.return_value = {
            "template": "{query} {instructions} {documents}"
        }
        mock_llm = self._setup_mock_llm(
            [
                {"index": 1, "score": 0.9, "reason": "Best match"},
                {"index": 0, "score": 0.7, "reason": "Second best"},
            ]
        )

        docs = [
            Document(page_content="Doc 0", metadata={"price": 200}),
            Document(page_content="Doc 1", metadata={"price": 100}),
        ]

        result = instruction_aware_rerank(
            llm=mock_llm,
            query="cheap products",
            documents=docs,
        )

        assert len(result) == 2
        # Doc 1 should be first (higher score) - sorted by Python
        assert result[0].page_content == "Doc 1"
        assert result[0].metadata["instruction_rerank_score"] == 0.9
        assert result[0].metadata["instruction_rerank_reason"] == "Best match"
        # Doc 0 should be second
        assert result[1].page_content == "Doc 0"
        assert result[1].metadata["instruction_rerank_score"] == 0.7

    @patch("dao_ai.tools.instruction_reranker._load_prompt_template")
    @patch("dao_ai.tools.instruction_reranker.mlflow")
    def test_applies_top_n_limit(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that top_n limit is applied after sorting."""
        mock_load_prompt.return_value = {
            "template": "{query} {instructions} {documents}"
        }
        mock_llm = self._setup_mock_llm(
            [
                {"index": 0, "score": 0.9, "reason": "Best match"},
                {"index": 1, "score": 0.8, "reason": "Good match"},
                {"index": 2, "score": 0.7, "reason": "Okay match"},
            ]
        )

        docs = [Document(page_content=f"Doc {i}", metadata={}) for i in range(3)]

        result = instruction_aware_rerank(
            llm=mock_llm,
            query="test",
            documents=docs,
            top_n=2,
        )

        assert len(result) == 2

    @patch("dao_ai.tools.instruction_reranker._load_prompt_template")
    @patch("dao_ai.tools.instruction_reranker.mlflow")
    def test_uses_with_structured_output(
        self,
        mock_mlflow: MagicMock,
        mock_load_prompt: MagicMock,
    ) -> None:
        """Test that instruction_aware_rerank uses with_structured_output."""
        mock_load_prompt.return_value = {
            "template": "{query} {instructions} {documents}"
        }
        mock_llm = self._setup_mock_llm([])

        docs = [Document(page_content="Test", metadata={})]
        instruction_aware_rerank(
            llm=mock_llm,
            query="test",
            documents=docs,
        )

        # Verify with_structured_output is called with RankingResult
        mock_llm.with_structured_output.assert_called_once_with(RankingResult)

    @patch("dao_ai.tools.instruction_reranker._load_prompt_template")
    @patch("dao_ai.tools.instruction_reranker.mlflow")
    def test_handles_invalid_indices(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that invalid document indices are handled gracefully."""
        mock_load_prompt.return_value = {
            "template": "{query} {instructions} {documents}"
        }
        mock_llm = self._setup_mock_llm(
            [
                {"index": 0, "score": 0.9, "reason": "Valid match"},
                {"index": 99, "score": 0.8, "reason": "Invalid index"},
                {"index": -1, "score": 0.7, "reason": "Negative index"},
            ]
        )

        docs = [Document(page_content="Doc 0", metadata={})]

        result = instruction_aware_rerank(
            llm=mock_llm,
            query="test",
            documents=docs,
        )

        # Only valid index should be included
        assert len(result) == 1
        assert result[0].page_content == "Doc 0"

    @patch("dao_ai.tools.instruction_reranker._load_prompt_template")
    @patch("dao_ai.tools.instruction_reranker.mlflow")
    def test_includes_custom_instructions_in_prompt(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that custom instructions are included in the prompt."""
        mock_load_prompt.return_value = {
            "template": "Instructions: {instructions} Query: {query} Docs: {documents}"
        }
        mock_llm = self._setup_mock_llm([])

        docs = [Document(page_content="Test", metadata={})]
        instruction_aware_rerank(
            llm=mock_llm,
            query="test",
            documents=docs,
            instructions="Prioritize price constraints",
        )

        # Verify the prompt contains the custom instructions
        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.invoke.call_args[0][0]
        assert "Prioritize price constraints" in call_args

    @patch("dao_ai.tools.instruction_reranker._load_prompt_template")
    @patch("dao_ai.tools.instruction_reranker.mlflow")
    def test_logs_average_score_metric(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that average instruction rerank score is logged."""
        mock_load_prompt.return_value = {
            "template": "{query} {instructions} {documents}"
        }
        mock_llm = self._setup_mock_llm(
            [
                {"index": 0, "score": 0.8, "reason": "Good match"},
                {"index": 1, "score": 0.6, "reason": "Okay match"},
            ]
        )

        docs = [
            Document(page_content="Doc 0", metadata={}),
            Document(page_content="Doc 1", metadata={}),
        ]

        result = instruction_aware_rerank(
            llm=mock_llm,
            query="test",
            documents=docs,
        )

        # Verify we got results back (rankings were valid)
        assert len(result) == 2
        # Average score should be (0.8 + 0.6) / 2 = 0.7
        mock_mlflow.set_tag.assert_called_with(
            "reranker.instruction_avg_score", "0.700"
        )

    @patch("dao_ai.tools.instruction_reranker._load_prompt_template")
    @patch("dao_ai.tools.instruction_reranker.mlflow")
    def test_sorts_results_by_score(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that results are sorted by score (highest first) in Python."""
        mock_load_prompt.return_value = {
            "template": "{query} {instructions} {documents}"
        }
        # Return unsorted rankings - Python should sort them
        mock_llm = self._setup_mock_llm(
            [
                {"index": 0, "score": 0.5, "reason": "Low score"},
                {"index": 1, "score": 0.9, "reason": "High score"},
                {"index": 2, "score": 0.7, "reason": "Medium score"},
            ]
        )

        docs = [Document(page_content=f"Doc {i}", metadata={}) for i in range(3)]

        result = instruction_aware_rerank(
            llm=mock_llm,
            query="test",
            documents=docs,
        )

        # Results should be sorted by score (highest first)
        assert len(result) == 3
        assert result[0].metadata["instruction_rerank_score"] == 0.9
        assert result[1].metadata["instruction_rerank_score"] == 0.7
        assert result[2].metadata["instruction_rerank_score"] == 0.5
