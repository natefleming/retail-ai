"""Unit tests for result verifier."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from dao_ai.config import VerificationResult, VerifierModel
from dao_ai.tools.verifier import (
    _format_constraints,
    _format_results_summary,
    add_verification_metadata,
    verify_results,
)


@pytest.mark.unit
class TestVerifierModel:
    """Unit tests for VerifierModel configuration."""

    def test_default_values(self) -> None:
        """Test that VerifierModel has sensible defaults."""
        model = VerifierModel()
        assert model.model is None
        assert model.on_failure == "warn"
        assert model.max_retries == 1

    def test_custom_values(self) -> None:
        """Test VerifierModel with custom configuration."""
        model = VerifierModel(
            on_failure="warn_and_retry",
            max_retries=2,
        )
        assert model.on_failure == "warn_and_retry"
        assert model.max_retries == 2

    def test_on_failure_literal_validation(self) -> None:
        """Test that only valid on_failure values are accepted."""
        with pytest.raises(ValueError):
            VerifierModel(on_failure="invalid")


@pytest.mark.unit
class TestVerificationResult:
    """Unit tests for VerificationResult structured output."""

    def test_passed_result(self) -> None:
        """Test creating a passed verification result."""
        result = VerificationResult(passed=True, confidence=0.95)
        assert result.passed is True
        assert result.confidence == 0.95
        assert result.feedback is None
        assert result.suggested_filter_relaxation is None
        assert result.unmet_constraints is None

    def test_failed_result_with_feedback(self) -> None:
        """Test creating a failed result with structured feedback."""
        result = VerificationResult(
            passed=False,
            confidence=0.6,
            feedback="Results are blue, user wanted red",
            suggested_filter_relaxation={"color": "REMOVE"},
            unmet_constraints=["color preference"],
        )
        assert result.passed is False
        assert result.confidence == 0.6
        assert result.feedback == "Results are blue, user wanted red"
        assert result.suggested_filter_relaxation == {"color": "REMOVE"}
        assert result.unmet_constraints == ["color preference"]

    def test_confidence_bounds(self) -> None:
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            VerificationResult(passed=True, confidence=1.5)
        with pytest.raises(ValueError):
            VerificationResult(passed=True, confidence=-0.1)


@pytest.mark.unit
class TestFormatConstraints:
    """Unit tests for _format_constraints helper."""

    def test_none_constraints(self) -> None:
        """Test formatting None constraints."""
        result = _format_constraints(None)
        assert result == "No explicit constraints specified."

    def test_empty_constraints(self) -> None:
        """Test formatting empty constraints list."""
        result = _format_constraints([])
        assert result == "No explicit constraints specified."

    def test_multiple_constraints(self) -> None:
        """Test formatting multiple constraints."""
        result = _format_constraints(["Prefer recent items", "Price under $100"])
        assert "- Prefer recent items" in result
        assert "- Price under $100" in result


@pytest.mark.unit
class TestFormatResultsSummary:
    """Unit tests for _format_results_summary helper."""

    def test_empty_documents(self) -> None:
        """Test formatting empty document list."""
        result = _format_results_summary([])
        assert result == "No results retrieved."

    def test_single_document(self) -> None:
        """Test formatting single document."""
        docs = [
            Document(
                page_content="Test content",
                metadata={"price": 100, "brand": "Milwaukee"},
            )
        ]
        result = _format_results_summary(docs)
        assert "Test content" in result
        assert "price: 100" in result
        assert "brand: Milwaukee" in result

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
        result = _format_results_summary(docs)
        assert "brand: Test" in result
        assert "_verification_status" not in result
        assert "rrf_score" not in result
        assert "reranker_score" not in result

    def test_truncates_to_max_docs(self) -> None:
        """Test that results are truncated to max_docs."""
        docs = [Document(page_content=f"Doc {i}", metadata={}) for i in range(10)]
        result = _format_results_summary(docs, max_docs=3)
        assert "Doc 0" in result
        assert "Doc 2" in result
        assert "Doc 3" not in result


@pytest.mark.unit
class TestAddVerificationMetadata:
    """Unit tests for add_verification_metadata helper."""

    def test_adds_passed_status(self) -> None:
        """Test adding passed verification status."""
        docs = [Document(page_content="Test", metadata={"price": 100})]
        result = VerificationResult(passed=True, confidence=0.9)

        annotated = add_verification_metadata(docs, result)

        assert annotated[0].metadata["_verification_status"] == "passed"
        assert annotated[0].metadata["_verification_confidence"] == 0.9
        assert annotated[0].metadata["price"] == 100

    def test_adds_failed_status(self) -> None:
        """Test adding failed verification status."""
        docs = [Document(page_content="Test", metadata={})]
        result = VerificationResult(
            passed=False,
            confidence=0.5,
            feedback="Not matching",
        )

        annotated = add_verification_metadata(docs, result)

        assert annotated[0].metadata["_verification_status"] == "failed"
        assert annotated[0].metadata["_verification_feedback"] == "Not matching"

    def test_adds_exhausted_status(self) -> None:
        """Test adding exhausted status when max retries exceeded."""
        docs = [Document(page_content="Test", metadata={})]
        result = VerificationResult(passed=False, confidence=0.4)

        annotated = add_verification_metadata(docs, result, exhausted=True)

        assert annotated[0].metadata["_verification_status"] == "exhausted"


@pytest.mark.unit
class TestVerifyResults:
    """Unit tests for verify_results function."""

    def _create_mock_llm(self, passed: bool, confidence: float) -> MagicMock:
        """Helper to create mock LLM with with_structured_output behavior."""
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        # with_structured_output returns Pydantic instance directly
        mock_structured_llm.invoke.return_value = VerificationResult(
            passed=passed, confidence=confidence
        )
        return mock_llm

    @patch("dao_ai.tools.verifier._load_prompt_template")
    @patch("dao_ai.tools.verifier.mlflow")
    def test_returns_verification_result(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that verify_results returns a VerificationResult."""
        mock_load_prompt.return_value = {
            "template": "{query} {schema_description} {constraints} {num_results} {results_summary} {previous_feedback}"
        }
        mock_llm = self._create_mock_llm(passed=True, confidence=0.9)

        docs = [Document(page_content="Test", metadata={})]
        result = verify_results(
            llm=mock_llm,
            query="test query",
            documents=docs,
            schema_description="test schema",
        )

        assert isinstance(result, VerificationResult)
        assert result.passed is True
        assert result.confidence == 0.9

    @patch("dao_ai.tools.verifier._load_prompt_template")
    @patch("dao_ai.tools.verifier.mlflow")
    def test_uses_with_structured_output(
        self,
        mock_mlflow: MagicMock,
        mock_load_prompt: MagicMock,
    ) -> None:
        """Test that verify_results uses with_structured_output for parsing."""
        mock_load_prompt.return_value = {
            "template": "{query} {schema_description} {constraints} {num_results} {results_summary} {previous_feedback}"
        }
        mock_llm = self._create_mock_llm(passed=True, confidence=0.9)

        docs = [Document(page_content="Test", metadata={})]
        verify_results(
            llm=mock_llm,
            query="test",
            documents=docs,
            schema_description="schema",
        )

        # Verify with_structured_output is called with VerificationResult schema
        mock_llm.with_structured_output.assert_called_once_with(VerificationResult)

    @patch("dao_ai.tools.verifier._load_prompt_template")
    @patch("dao_ai.tools.verifier.mlflow")
    def test_includes_previous_feedback_in_prompt(
        self, mock_mlflow: MagicMock, mock_load_prompt: MagicMock
    ) -> None:
        """Test that previous feedback is included in the prompt."""
        mock_load_prompt.return_value = {
            "template": "Feedback: {previous_feedback} Query: {query} Schema: {schema_description} Constraints: {constraints} Results: {results_summary} Count: {num_results}"
        }
        mock_llm = self._create_mock_llm(passed=True, confidence=0.9)

        docs = [Document(page_content="Test", metadata={})]
        verify_results(
            llm=mock_llm,
            query="test",
            documents=docs,
            schema_description="schema",
            previous_feedback="Previous attempt failed",
        )

        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.invoke.call_args[0][0]
        assert "Previous attempt failed" in call_args
