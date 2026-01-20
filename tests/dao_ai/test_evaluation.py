"""
Tests for DAO AI Evaluation Module

Unit and integration tests for MLflow GenAI evaluation scorers and helpers.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from mlflow.entities import Feedback, SpanStatus, SpanStatusCode

# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------


class TestResponseCompleteness:
    """Unit tests for response_completeness scorer."""

    @pytest.mark.unit
    def test_complete_response_flat_format(self) -> None:
        """Test that a complete response passes with flat output format."""
        from dao_ai.evaluation import response_completeness

        outputs = {
            "response": "This is a complete and helpful response to your question."
        }
        result = response_completeness(outputs)

        assert isinstance(result, Feedback)
        assert result.value is True
        assert "complete" in result.rationale.lower()

    @pytest.mark.unit
    def test_complete_response_nested_format(self) -> None:
        """Test that a complete response passes with nested output format."""
        from dao_ai.evaluation import response_completeness

        outputs = {"outputs": {"response": "This is a complete and helpful response."}}
        result = response_completeness(outputs)

        assert isinstance(result, Feedback)
        assert result.value is True

    @pytest.mark.unit
    def test_short_response_fails(self) -> None:
        """Test that a too-short response fails."""
        from dao_ai.evaluation import response_completeness

        outputs = {"response": "Hi"}
        result = response_completeness(outputs)

        assert isinstance(result, Feedback)
        assert result.value is False
        assert "short" in result.rationale.lower()

    @pytest.mark.unit
    def test_incomplete_response_ellipsis(self) -> None:
        """Test that response ending with ... fails."""
        from dao_ai.evaluation import response_completeness

        outputs = {"response": "The answer to your question is..."}
        result = response_completeness(outputs)

        assert isinstance(result, Feedback)
        assert result.value is False
        assert "incomplete" in result.rationale.lower()

    @pytest.mark.unit
    def test_incomplete_response_etc(self) -> None:
        """Test that response ending with 'etc' fails."""
        from dao_ai.evaluation import response_completeness

        outputs = {"response": "You can use tools like hammers, screwdrivers, etc"}
        result = response_completeness(outputs)

        assert isinstance(result, Feedback)
        assert result.value is False

    @pytest.mark.unit
    def test_empty_response_fails(self) -> None:
        """Test that empty response fails."""
        from dao_ai.evaluation import response_completeness

        outputs = {"response": ""}
        result = response_completeness(outputs)

        assert isinstance(result, Feedback)
        assert result.value is False

    @pytest.mark.unit
    def test_missing_response_key_fails(self) -> None:
        """Test that missing response key fails."""
        from dao_ai.evaluation import response_completeness

        outputs = {"other_field": "value"}
        result = response_completeness(outputs)

        assert isinstance(result, Feedback)
        assert result.value is False

    @pytest.mark.unit
    def test_string_output_passes(self) -> None:
        """Test that a direct string output passes."""
        from dao_ai.evaluation import response_completeness

        outputs = "This is a complete and helpful response to your question."
        result = response_completeness(outputs)

        assert isinstance(result, Feedback)
        assert result.value is True
        assert "complete" in result.rationale.lower()

    @pytest.mark.unit
    def test_short_string_output_fails(self) -> None:
        """Test that a too-short string output fails."""
        from dao_ai.evaluation import response_completeness

        outputs = "Hi"
        result = response_completeness(outputs)

        assert isinstance(result, Feedback)
        assert result.value is False
        assert "short" in result.rationale.lower()


class TestToolCallEfficiency:
    """Unit tests for tool_call_efficiency scorer."""

    @pytest.mark.unit
    def test_none_trace_returns_error(self) -> None:
        """Test that None trace returns error feedback (MLflow 3.8+ behavior)."""
        from dao_ai.evaluation import tool_call_efficiency

        result = tool_call_efficiency(None)

        assert isinstance(result, Feedback)
        # MLflow 3.8+ uses error instead of value=None
        assert result.error is not None
        assert "no trace" in str(result.error).lower()

    @pytest.mark.unit
    def test_no_tool_calls_returns_true(self) -> None:
        """Test that trace with no tool calls returns True (valid response without tools)."""
        from dao_ai.evaluation import tool_call_efficiency

        mock_trace = MagicMock()
        mock_trace.search_spans.return_value = []

        result = tool_call_efficiency(mock_trace)

        assert isinstance(result, Feedback)
        assert result.value is True
        assert "no tool usage" in result.rationale.lower()

    @pytest.mark.unit
    def test_successful_unique_tool_calls_passes(self) -> None:
        """Test that unique successful tool calls pass."""
        from dao_ai.evaluation import tool_call_efficiency

        # Create mock spans with different names using typed SpanStatus
        mock_span1 = MagicMock()
        mock_span1.name = "search_products"
        mock_span1.status = SpanStatus(status_code=SpanStatusCode.OK, description="")

        mock_span2 = MagicMock()
        mock_span2.name = "get_inventory"
        mock_span2.status = SpanStatus(status_code=SpanStatusCode.OK, description="")

        mock_trace = MagicMock()
        mock_trace.search_spans.return_value = [mock_span1, mock_span2]

        result = tool_call_efficiency(mock_trace)

        assert isinstance(result, Feedback)
        assert result.value is True
        assert "2 successful" in result.rationale.lower()

    @pytest.mark.unit
    def test_redundant_tool_calls_fails(self) -> None:
        """Test that redundant tool calls fail."""
        from dao_ai.evaluation import tool_call_efficiency

        # Create mock spans with same name (redundant) using typed SpanStatus
        mock_span1 = MagicMock()
        mock_span1.name = "search_products"
        mock_span1.status = SpanStatus(status_code=SpanStatusCode.OK, description="")

        mock_span2 = MagicMock()
        mock_span2.name = "search_products"  # Same name = redundant
        mock_span2.status = SpanStatus(status_code=SpanStatusCode.OK, description="")

        mock_trace = MagicMock()
        mock_trace.search_spans.return_value = [mock_span1, mock_span2]

        result = tool_call_efficiency(mock_trace)

        assert isinstance(result, Feedback)
        assert result.value is False
        assert "redundant" in result.rationale.lower()

    @pytest.mark.unit
    def test_failed_tool_calls_fails(self) -> None:
        """Test that failed tool calls result in failure."""
        from dao_ai.evaluation import tool_call_efficiency

        # Create mock span with ERROR status using typed SpanStatus
        mock_span = MagicMock()
        mock_span.name = "search_products"
        mock_span.status = SpanStatus(
            status_code=SpanStatusCode.ERROR, description="Tool failed"
        )

        mock_trace = MagicMock()
        mock_trace.search_spans.return_value = [mock_span]

        result = tool_call_efficiency(mock_trace)

        assert isinstance(result, Feedback)
        assert result.value is False
        assert "failed" in result.rationale.lower()

    @pytest.mark.unit
    def test_trace_search_exception_handled(self) -> None:
        """Test that exceptions during trace search are handled (MLflow 3.8+ behavior)."""
        from dao_ai.evaluation import tool_call_efficiency

        mock_trace = MagicMock()
        mock_trace.search_spans.side_effect = Exception("Trace error")

        result = tool_call_efficiency(mock_trace)

        assert isinstance(result, Feedback)
        # MLflow 3.8+ uses error instead of value=None
        assert result.error is not None
        assert "error" in str(result.error).lower()


class TestResponseClarity:
    """Unit tests for response_clarity scorer."""

    @pytest.mark.unit
    def test_clear_response_passes(self) -> None:
        """Test that a clear response passes."""
        from dao_ai.evaluation import response_clarity

        outputs = {"response": "This is a clear response. It has proper sentences."}
        result = response_clarity(outputs)

        assert isinstance(result, Feedback)
        assert result.value is True
        assert "clear" in result.rationale.lower()

    @pytest.mark.unit
    def test_no_sentences_fails(self) -> None:
        """Test that response without sentence structure fails."""
        from dao_ai.evaluation import response_clarity

        outputs = {"response": "no punctuation here at all just words"}
        result = response_clarity(outputs)

        assert isinstance(result, Feedback)
        assert result.value is False
        assert "sentence" in result.rationale.lower()

    @pytest.mark.unit
    def test_question_mark_counts_as_sentence(self) -> None:
        """Test that question marks are valid sentence endings."""
        from dao_ai.evaluation import response_clarity

        outputs = {"response": "Would you like me to help you?"}
        result = response_clarity(outputs)

        assert isinstance(result, Feedback)
        assert result.value is True

    @pytest.mark.unit
    def test_exclamation_counts_as_sentence(self) -> None:
        """Test that exclamation marks are valid sentence endings."""
        from dao_ai.evaluation import response_clarity

        outputs = {"response": "Great question! I can help with that!"}
        result = response_clarity(outputs)

        assert isinstance(result, Feedback)
        assert result.value is True

    @pytest.mark.unit
    def test_string_output_passes(self) -> None:
        """Test that a direct string output passes."""
        from dao_ai.evaluation import response_clarity

        outputs = "This is a clear response. It has proper sentences."
        result = response_clarity(outputs)

        assert isinstance(result, Feedback)
        assert result.value is True
        assert "clear" in result.rationale.lower()

    @pytest.mark.unit
    def test_string_no_sentences_fails(self) -> None:
        """Test that string without sentence structure fails."""
        from dao_ai.evaluation import response_clarity

        outputs = "no punctuation here at all just words"
        result = response_clarity(outputs)

        assert isinstance(result, Feedback)
        assert result.value is False
        assert "sentence" in result.rationale.lower()


class TestHelperFunctions:
    """Unit tests for helper functions."""

    @pytest.mark.unit
    def test_create_traced_predict_fn_normalizes_output(self) -> None:
        """Test that traced predict function normalizes nested output."""
        from dao_ai.evaluation import create_traced_predict_fn

        def nested_predict(inputs: dict) -> dict:
            return {"outputs": {"response": "test response"}}

        with patch("mlflow.trace") as mock_trace:
            # Make the decorator pass through the function
            mock_trace.return_value = lambda f: f

            traced_fn = create_traced_predict_fn(nested_predict)
            result = traced_fn({"messages": [{"role": "user", "content": "hi"}]})
            assert result == {"response": "test response"}

    @pytest.mark.unit
    def test_get_default_scorers_includes_expected(self) -> None:
        """Test that default scorers include expected scorers."""

        from dao_ai.evaluation import (
            get_default_scorers,
        )

        scorers = get_default_scorers(include_trace_scorers=True)

        # Should include Safety, response_completeness, response_clarity, tool_call_efficiency
        assert len(scorers) == 4

    @pytest.mark.unit
    def test_get_default_scorers_excludes_trace_scorers(self) -> None:
        """Test that trace scorers can be excluded."""
        from dao_ai.evaluation import get_default_scorers

        scorers = get_default_scorers(include_trace_scorers=False)

        # Should exclude tool_call_efficiency
        assert len(scorers) == 3

    @pytest.mark.unit
    def test_create_guidelines_scorers(self) -> None:
        """Test creating Guidelines scorers from config."""
        from dao_ai.evaluation import create_guidelines_scorers

        # Create mock guideline config
        class MockGuideline:
            name = "test_guideline"
            guidelines = ["Be helpful", "Be accurate"]

        guidelines = [MockGuideline()]
        judge_model = "endpoints:/test-model"

        scorers = create_guidelines_scorers(guidelines, judge_model)

        assert len(scorers) == 1
        assert scorers[0].name == "test_guideline"


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestEvaluationIntegration:
    """Integration tests for MLflow evaluation."""

    @pytest.mark.integration
    def test_scorers_return_feedback_objects(self) -> None:
        """Test that all scorers return valid Feedback objects."""
        from dao_ai.evaluation import (
            response_clarity,
            response_completeness,
            tool_call_efficiency,
        )

        # Test response_completeness
        result1 = response_completeness({"response": "Test response here."})
        assert isinstance(result1, Feedback)
        assert result1.value is not None or result1.rationale is not None

        # Test response_clarity
        result2 = response_clarity({"response": "Test response here."})
        assert isinstance(result2, Feedback)

        # Test tool_call_efficiency with None trace
        result3 = tool_call_efficiency(None)
        assert isinstance(result3, Feedback)

    @pytest.mark.integration
    def test_setup_evaluation_tracking(self) -> None:
        """Test that evaluation tracking setup doesn't error."""
        from dao_ai.evaluation import setup_evaluation_tracking

        with patch("mlflow.set_registry_uri") as mock_registry:
            with patch("mlflow.set_experiment") as mock_experiment:
                with patch("mlflow.langchain.autolog") as mock_autolog:
                    setup_evaluation_tracking(experiment_name="test_experiment")

                    mock_registry.assert_called_once_with("databricks-uc")
                    mock_experiment.assert_called_once_with(
                        experiment_name="test_experiment"
                    )
                    mock_autolog.assert_called_once_with(log_traces=True)

    @pytest.mark.integration
    def test_run_evaluation_with_mocked_mlflow(self) -> None:
        """Test run_evaluation with mocked MLflow."""
        from dao_ai.evaluation import run_evaluation

        def simple_predict(inputs: dict) -> dict:
            return {"response": "Test response"}

        mock_result = MagicMock()
        mock_result.metrics = {"response_completeness/v1/mean": 1.0}

        with patch("mlflow.genai.evaluate", return_value=mock_result) as mock_eval:
            with patch("mlflow.trace") as mock_trace:
                mock_trace.return_value = lambda f: f

                data = [{"inputs": {"messages": [{"role": "user", "content": "test"}]}}]

                # Note: This would need actual MLflow setup in real integration test
                # For now, we verify the function calls correctly
                eval_result = run_evaluation(
                    data=data,
                    predict_fn=simple_predict,
                )

                mock_eval.assert_called_once()
                assert eval_result is mock_result


# -----------------------------------------------------------------------------
# Test Configuration
# -----------------------------------------------------------------------------


def pytest_configure(config: Any) -> None:
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may require external services)"
    )
