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
        """Test creating a RankedDocument."""
        rd = RankedDocument(index=0, score=0.9)
        assert rd.index == 0
        assert rd.score == 0.9
        assert rd.reason is None

    def test_with_reason(self) -> None:
        """Test creating a RankedDocument with reason."""
        rd = RankedDocument(
            index=1,
            score=0.8,
            reason="Matches price constraint",
        )
        assert rd.index == 1
        assert rd.score == 0.8
        assert rd.reason == "Matches price constraint"

    def test_score_bounds(self) -> None:
        """Test that score must be between 0 and 1."""
        with pytest.raises(ValueError):
            RankedDocument(index=0, score=1.5)
        with pytest.raises(ValueError):
            RankedDocument(index=0, score=-0.1)


@pytest.mark.unit
class TestRankingResult:
    """Unit tests for RankingResult container model."""

    def test_basic_creation(self) -> None:
        """Test creating a RankingResult with rankings."""
        rankings = [
            RankedDocument(index=1, score=0.9),
            RankedDocument(index=0, score=0.7),
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

    def _create_mock_llm(self, rankings: list[dict]) -> MagicMock:
        """Helper to create mock LLM with bind() behavior for invoke_with_structured_output."""
        mock_llm = MagicMock()
        mock_bound_llm = MagicMock()
        mock_llm.bind.return_value = mock_bound_llm
        # invoke_with_structured_output expects response.content to be JSON string
        mock_result = RankingResult(rankings=[RankedDocument(**r) for r in rankings])
        mock_response = MagicMock()
        mock_response.content = mock_result.model_dump_json()
        mock_bound_llm.invoke.return_value = mock_response
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
        mock_llm.with_structured_output.assert_not_called()

    @patch("dao_ai.tools.instruction_reranker._load_prompt_template")
    @patch("dao_ai.tools.instruction_reranker.mlflow")
    def test_reranks_documents(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that documents are reranked based on LLM response."""
        mock_load_prompt.return_value = {
            "template": "{query} {schema_description} {instructions} {documents}"
        }
        mock_llm = self._create_mock_llm(
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
        # Doc 1 should be first (higher score)
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
        """Test that top_n limit is applied."""
        mock_load_prompt.return_value = {
            "template": "{query} {schema_description} {instructions} {documents}"
        }
        mock_llm = self._create_mock_llm(
            [
                {"index": 0, "score": 0.9},
                {"index": 1, "score": 0.8},
                {"index": 2, "score": 0.7},
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
    def test_uses_bind_with_response_format(
        self,
        mock_mlflow: MagicMock,
        mock_load_prompt: MagicMock,
    ) -> None:
        """Test that instruction_aware_rerank uses bind() with response_format for parsing."""
        mock_load_prompt.return_value = {
            "template": "{query} {schema_description} {instructions} {documents}"
        }
        mock_llm = self._create_mock_llm([])

        docs = [Document(page_content="Test", metadata={})]
        instruction_aware_rerank(
            llm=mock_llm,
            query="test",
            documents=docs,
        )

        # Verify bind() is called with response_format containing RankingResult schema
        mock_llm.bind.assert_called_once()
        call_kwargs = mock_llm.bind.call_args[1]
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["json_schema"]["name"] == "RankingResult"

    @patch("dao_ai.tools.instruction_reranker._load_prompt_template")
    @patch("dao_ai.tools.instruction_reranker.mlflow")
    def test_handles_invalid_indices(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that invalid document indices are handled gracefully."""
        mock_load_prompt.return_value = {
            "template": "{query} {schema_description} {instructions} {documents}"
        }
        mock_llm = self._create_mock_llm(
            [
                {"index": 0, "score": 0.9},
                {"index": 99, "score": 0.8},
                {"index": -1, "score": 0.7},
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
            "template": "Instructions: {instructions} Query: {query} Schema: {schema_description} Docs: {documents}"
        }
        mock_llm = self._create_mock_llm([])

        docs = [Document(page_content="Test", metadata={})]
        instruction_aware_rerank(
            llm=mock_llm,
            query="test",
            documents=docs,
            instructions="Prioritize price constraints",
        )

        bound_llm = mock_llm.bind.return_value
        call_args = bound_llm.invoke.call_args[0][0]
        assert "Prioritize price constraints" in call_args

    @patch("dao_ai.tools.instruction_reranker._load_prompt_template")
    @patch("dao_ai.tools.instruction_reranker.mlflow")
    def test_logs_average_score_metric(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that average instruction rerank score is logged."""
        mock_load_prompt.return_value = {
            "template": "{query} {schema_description} {instructions} {documents}"
        }
        mock_llm = self._create_mock_llm(
            [
                {"index": 0, "score": 0.8},
                {"index": 1, "score": 0.6},
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
