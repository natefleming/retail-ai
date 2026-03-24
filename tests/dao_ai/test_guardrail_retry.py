"""
Test guardrail retry behavior.

This test verifies that guardrails properly retry when evaluations fail,
extract tool context, handle errors gracefully, and maintain thread-safe
retry state.  Also tests the MLflow Scorer integration layer including
``JudgeScorer``, ``_interpret_feedback``, and scorer-based
``GuardrailMiddleware`` initialization.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from mlflow.entities import Feedback
from mlflow.genai.scorers.base import Scorer

from dao_ai.middleware.guardrails import (
    GuardrailMiddleware,
    JudgeScorer,
    _extract_text_content,
    _extract_tool_context,
    _interpret_feedback,
)
from dao_ai.state import AgentState, Context


@pytest.fixture
def mock_feedback_pass():
    """Create a passing MLflow Feedback object."""
    return Feedback(value=True, rationale="Response meets criteria.")


@pytest.fixture
def mock_feedback_fail():
    """Create a failing MLflow Feedback object."""
    return Feedback(
        value=False,
        rationale="The response contains fabricated information. Please acknowledge you don't have this information.",
    )


@pytest.fixture
def guardrail_middleware():
    """Create guardrail middleware with mocked make_judge."""
    with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
        mock_evaluator = Mock()
        mock_make_judge.return_value = mock_evaluator

        middleware = GuardrailMiddleware(
            name="test_guardrail",
            model="databricks:/test-judge",
            prompt="Evaluate {{ inputs }} and {{ outputs }}.",
            num_retries=3,
        )
        # Store reference to mock evaluator for test manipulation
        middleware._mock_evaluator = mock_evaluator
        yield middleware


@pytest.fixture
def runtime():
    """Mock runtime with context."""
    mock_runtime = Mock(spec=Runtime)
    mock_runtime.context = Context(user_id="test_user", thread_id="test_thread")
    return mock_runtime


def test_guardrail_retry_on_failure(guardrail_middleware, runtime, mock_feedback_fail):
    """
    Test that guardrail triggers retry when evaluation fails.
    """
    state: AgentState = {
        "messages": [
            HumanMessage(content="What is your refund policy?"),
            AIMessage(content="We have a 30-day refund policy with full refund."),
        ]
    }

    guardrail_middleware._mock_evaluator.return_value = mock_feedback_fail

    # First evaluation - should fail and request retry
    result = guardrail_middleware.after_model(state, runtime)

    # Verify retry was requested
    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)

    # Verify feedback is included
    feedback_message = result["messages"][0]
    assert "fabricated information" in feedback_message.content.lower()

    # Verify retry count incremented (thread-keyed)
    assert guardrail_middleware._get_retry_count("test_thread") == 1


def test_guardrail_max_retries_exhausted(guardrail_middleware, runtime):
    """
    Test that guardrail stops retrying after max retries and informs user.
    """
    state: AgentState = {
        "messages": [
            HumanMessage(content="Test question"),
            AIMessage(content="Bad response"),
        ]
    }

    guardrail_middleware._mock_evaluator.return_value = Feedback(
        value=False,
        rationale="Response fails criteria.",
    )

    # Exhaust all retries
    for i in range(guardrail_middleware.num_retries):
        result = guardrail_middleware.after_model(state, runtime)
        if i < guardrail_middleware.num_retries - 1:
            # Should request retry with feedback
            assert result is not None
            assert "messages" in result
            assert isinstance(result["messages"][0], HumanMessage)
        else:
            # Max retries reached - should return failure message to user
            assert result is not None
            assert "messages" in result
            assert isinstance(result["messages"][0], AIMessage)
            # Verify user is informed
            failure_msg = result["messages"][0].content
            assert "Quality Check Failed" in failure_msg
            assert "test_guardrail" in failure_msg
            assert "Response fails criteria" in failure_msg

    # Verify retry count was reset
    assert guardrail_middleware._get_retry_count("test_thread") == 0


def test_guardrail_success_on_retry(guardrail_middleware, runtime):
    """
    Test that guardrail passes after a retry with improved response.
    """
    state: AgentState = {
        "messages": [
            HumanMessage(content="What is Python?"),
            AIMessage(content="Python is a programming language."),
        ]
    }

    # First call fails
    guardrail_middleware._mock_evaluator.return_value = Feedback(
        value=False,
        rationale="Response is too brief. Provide more detail.",
    )

    # First evaluation - fails
    result = guardrail_middleware.after_model(state, runtime)
    assert result is not None
    assert guardrail_middleware._get_retry_count("test_thread") == 1

    # Agent retries with better response
    state["messages"].append(result["messages"][0])  # Add feedback
    state["messages"].append(
        AIMessage(
            content="Python is a high-level, interpreted programming language known for its readability and extensive libraries."
        )
    )

    # Second call succeeds
    guardrail_middleware._mock_evaluator.return_value = Feedback(
        value=True,
        rationale="Response is now comprehensive and accurate.",
    )

    # Second evaluation - passes
    result = guardrail_middleware.after_model(state, runtime)
    assert result is None  # No retry needed
    assert guardrail_middleware._get_retry_count("test_thread") == 0  # Reset


def test_guardrail_skips_tool_calls(guardrail_middleware, runtime):
    """
    Test that guardrail skips evaluation when AI message has tool calls.
    """
    state: AgentState = {
        "messages": [
            HumanMessage(content="Search for Python"),
            AIMessage(
                content="",
                tool_calls=[{"id": "1", "name": "search", "args": {"query": "Python"}}],
            ),
        ]
    }

    # Should not be called since we skip evaluation
    result = guardrail_middleware.after_model(state, runtime)

    # Should skip evaluation and not call judge
    assert result is None
    guardrail_middleware._mock_evaluator.assert_not_called()


def test_guardrail_handles_structured_content(guardrail_middleware, runtime):
    """
    Test that guardrail properly extracts text from structured content.
    """
    state: AgentState = {
        "messages": [
            HumanMessage(content="Test"),
            AIMessage(
                content=[
                    {"type": "text", "text": "This is structured content"},
                    {"type": "text", "text": " with multiple blocks."},
                ]
            ),
        ]
    }

    guardrail_middleware._mock_evaluator.return_value = Feedback(
        value=True, rationale="Good"
    )

    guardrail_middleware.after_model(state, runtime)

    # Verify evaluator was called with extracted text
    guardrail_middleware._mock_evaluator.assert_called_once()
    call_args = guardrail_middleware._mock_evaluator.call_args[1]
    assert "inputs" in call_args
    assert "outputs" in call_args
    # Should have extracted and joined the text blocks (with space separator)
    assert "This is structured content" in call_args["outputs"]["response"]
    assert "with multiple blocks" in call_args["outputs"]["response"]


def test_guardrail_extracts_tool_context(guardrail_middleware, runtime):
    """
    Test that guardrail extracts ToolMessage content and passes it as context.
    """
    state: AgentState = {
        "messages": [
            HumanMessage(content="What is our refund policy?"),
            AIMessage(
                content="",
                tool_calls=[{"id": "1", "name": "search", "args": {"query": "refund"}}],
            ),
            ToolMessage(
                content="Our refund policy allows returns within 30 days for a full refund.",
                tool_call_id="1",
                name="search",
            ),
            AIMessage(
                content="Based on our policy, you can return items within 30 days."
            ),
        ]
    }

    guardrail_middleware._mock_evaluator.return_value = Feedback(
        value=True, rationale="Response is grounded in context"
    )

    guardrail_middleware.after_model(state, runtime)

    # Verify evaluator was called with tool context in inputs
    guardrail_middleware._mock_evaluator.assert_called_once()
    call_args = guardrail_middleware._mock_evaluator.call_args[1]
    inputs_dict: dict = call_args["inputs"]
    assert "context" in inputs_dict
    assert "refund policy" in inputs_dict["context"].lower()
    assert "[search]" in inputs_dict["context"]


def test_guardrail_default_lets_response_through_on_error(runtime):
    """
    Test that guardrail lets response through when judge call errors
    (fail_on_error=False, the default).
    """
    with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
        mock_evaluator = Mock()
        mock_evaluator.side_effect = Exception("Network timeout")
        mock_make_judge.return_value = mock_evaluator

        middleware = GuardrailMiddleware(
            name="test_guardrail",
            model="databricks:/test-judge",
            prompt="Evaluate {{ inputs }} and {{ outputs }}.",
            num_retries=3,
            fail_on_error=False,
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="Test question"),
                AIMessage(content="Test response"),
            ]
        }

        result = middleware.after_model(state, runtime)

        # Should let response through (default: fail_on_error=False)
        assert result is None


def test_guardrail_blocks_response_on_error(runtime):
    """
    Test that guardrail blocks response when judge call errors
    and fail_on_error=True.
    """
    with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
        mock_evaluator = Mock()
        mock_evaluator.side_effect = Exception("Network timeout")
        mock_make_judge.return_value = mock_evaluator

        middleware = GuardrailMiddleware(
            name="test_guardrail",
            model="databricks:/test-judge",
            prompt="Evaluate {{ inputs }} and {{ outputs }}.",
            num_retries=3,
            fail_on_error=True,
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="Test question"),
                AIMessage(content="Test response"),
            ]
        }

        result = middleware.after_model(state, runtime)

        # Should return error message (fail closed)
        assert result is not None
        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        assert "Quality Check Error" in result["messages"][0].content


def test_guardrail_concurrent_threads(runtime):
    """
    Test that retry counts are isolated per thread_id.
    """
    with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
        mock_evaluator = Mock()
        mock_evaluator.return_value = Feedback(value=False, rationale="Fails criteria.")
        mock_make_judge.return_value = mock_evaluator

        middleware = GuardrailMiddleware(
            name="test_guardrail",
            model="databricks:/test-judge",
            prompt="Evaluate {{ inputs }} and {{ outputs }}.",
            num_retries=3,
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="Test"),
                AIMessage(content="Response"),
            ]
        }

        # Thread A
        runtime_a = Mock(spec=Runtime)
        runtime_a.context = Context(user_id="user_a", thread_id="thread_a")

        # Thread B
        runtime_b = Mock(spec=Runtime)
        runtime_b.context = Context(user_id="user_b", thread_id="thread_b")

        # Trigger failure on thread A
        middleware.after_model(state, runtime_a)
        assert middleware._get_retry_count("thread_a") == 1
        assert middleware._get_retry_count("thread_b") == 0

        # Trigger failure on thread B
        middleware.after_model(state, runtime_b)
        assert middleware._get_retry_count("thread_a") == 1
        assert middleware._get_retry_count("thread_b") == 1

        # Reset thread A by passing
        mock_evaluator.return_value = Feedback(value=True, rationale="OK")
        middleware.after_model(state, runtime_a)
        assert middleware._get_retry_count("thread_a") == 0
        assert middleware._get_retry_count("thread_b") == 1


# =============================================================================
# Unit tests for helper functions
# =============================================================================


def test_extract_text_content_string():
    """Test _extract_text_content with string content."""
    msg = HumanMessage(content="Hello world")
    assert _extract_text_content(msg) == "Hello world"


def test_extract_text_content_structured():
    """Test _extract_text_content with structured content blocks."""
    msg = AIMessage(
        content=[
            {"type": "text", "text": "Part one"},
            {"type": "text", "text": "Part two"},
        ]
    )
    assert "Part one" in _extract_text_content(msg)
    assert "Part two" in _extract_text_content(msg)


def test_extract_tool_context_basic():
    """Test _extract_tool_context extracts ToolMessage content."""
    messages = [
        HumanMessage(content="Question"),
        ToolMessage(content="Tool result data", tool_call_id="1", name="search"),
        AIMessage(content="Answer"),
    ]
    context = _extract_tool_context(messages)
    assert "[search]" in context
    assert "Tool result data" in context


def test_extract_tool_context_multiple_tools():
    """Test _extract_tool_context with multiple ToolMessages."""
    messages = [
        ToolMessage(content="Result from search", tool_call_id="1", name="search"),
        ToolMessage(content="Result from SQL", tool_call_id="2", name="sql_query"),
    ]
    context = _extract_tool_context(messages)
    assert "[search]" in context
    assert "[sql_query]" in context
    assert "Result from search" in context
    assert "Result from SQL" in context


def test_extract_tool_context_truncation():
    """Test _extract_tool_context truncates long content."""
    messages = [
        ToolMessage(content="x" * 5000, tool_call_id="1", name="big_tool"),
    ]
    context = _extract_tool_context(messages, max_length=200)
    assert len(context) <= 250  # Allow for label and truncation marker
    assert "[truncated]" in context


def test_extract_tool_context_empty():
    """Test _extract_tool_context returns empty string when no tools."""
    messages = [
        HumanMessage(content="Question"),
        AIMessage(content="Answer"),
    ]
    context = _extract_tool_context(messages)
    assert context == ""


# =============================================================================
# Specialized Guardrail Tests
# =============================================================================


class TestVeracityGuardrailMiddleware:
    """Tests for VeracityGuardrailMiddleware."""

    def test_skips_when_no_tool_context(self, runtime):
        """Veracity guardrail should skip when no ToolMessages are present."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import VeracityGuardrailMiddleware

            middleware = VeracityGuardrailMiddleware(
                model="databricks:/test-judge",
            )

            state: AgentState = {
                "messages": [
                    HumanMessage(content="What is Python?"),
                    AIMessage(content="Python is a programming language."),
                ]
            }

            result = middleware.after_model(state, runtime)

            # Should skip -- no tool context to ground against
            assert result is None
            mock_evaluator.assert_not_called()

    def test_evaluates_when_tool_context_present(self, runtime):
        """Veracity guardrail should evaluate when ToolMessages are present."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_evaluator.return_value = Feedback(
                value=True, rationale="Response is grounded."
            )
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import VeracityGuardrailMiddleware

            middleware = VeracityGuardrailMiddleware(
                model="databricks:/test-judge",
            )

            state: AgentState = {
                "messages": [
                    HumanMessage(content="What is our refund policy?"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"id": "1", "name": "search", "args": {"q": "refund"}}
                        ],
                    ),
                    ToolMessage(
                        content="30-day refund policy for all items.",
                        tool_call_id="1",
                        name="search",
                    ),
                    AIMessage(
                        content="Our refund policy allows returns within 30 days."
                    ),
                ]
            }

            result = middleware.after_model(state, runtime)

            # Should evaluate and pass
            assert result is None
            mock_evaluator.assert_called_once()

    def test_rejects_ungrounded_response(self, runtime):
        """Veracity guardrail should reject fabricated responses."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_evaluator.return_value = Feedback(
                value=False,
                rationale="The response claims a 90-day policy but the tool context says 30 days.",
            )
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import VeracityGuardrailMiddleware

            middleware = VeracityGuardrailMiddleware(
                model="databricks:/test-judge",
            )

            state: AgentState = {
                "messages": [
                    HumanMessage(content="What is our refund policy?"),
                    ToolMessage(
                        content="30-day refund policy.",
                        tool_call_id="1",
                        name="search",
                    ),
                    AIMessage(
                        content="Our refund policy allows returns within 90 days."
                    ),
                ]
            }

            result = middleware.after_model(state, runtime)

            # Should reject with retry feedback
            assert result is not None
            assert isinstance(result["messages"][0], HumanMessage)
            assert "90-day" in result["messages"][0].content


class TestRelevanceGuardrailMiddleware:
    """Tests for RelevanceGuardrailMiddleware."""

    def test_passes_relevant_response(self, runtime):
        """Relevance guardrail should pass a directly relevant response."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_evaluator.return_value = Feedback(
                value=True, rationale="Response directly addresses the query."
            )
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import RelevanceGuardrailMiddleware

            middleware = RelevanceGuardrailMiddleware(
                model="databricks:/test-judge",
            )

            state: AgentState = {
                "messages": [
                    HumanMessage(content="What is Python?"),
                    AIMessage(content="Python is a high-level programming language."),
                ]
            }

            result = middleware.after_model(state, runtime)
            assert result is None

    def test_rejects_off_topic_response(self, runtime):
        """Relevance guardrail should reject an off-topic response."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_evaluator.return_value = Feedback(
                value=False,
                rationale="The response discusses Java instead of Python.",
            )
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import RelevanceGuardrailMiddleware

            middleware = RelevanceGuardrailMiddleware(
                model="databricks:/test-judge",
            )

            state: AgentState = {
                "messages": [
                    HumanMessage(content="What is Python?"),
                    AIMessage(content="Java is a compiled programming language."),
                ]
            }

            result = middleware.after_model(state, runtime)

            assert result is not None
            assert isinstance(result["messages"][0], HumanMessage)


class TestToneGuardrailMiddleware:
    """Tests for ToneGuardrailMiddleware."""

    def test_professional_tone_profile(self, runtime):
        """ToneGuardrailMiddleware should use the professional profile prompt."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_evaluator.return_value = Feedback(
                value=True, rationale="Professional tone."
            )
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import ToneGuardrailMiddleware

            middleware = ToneGuardrailMiddleware(
                model="databricks:/test-judge",
                tone="professional",
            )

            assert middleware.guardrail_name == "tone_professional"
            assert middleware.tone == "professional"

            state: AgentState = {
                "messages": [
                    HumanMessage(content="Help me"),
                    AIMessage(content="I would be happy to assist you."),
                ]
            }

            result = middleware.after_model(state, runtime)
            assert result is None

    def test_custom_guidelines(self, runtime):
        """ToneGuardrailMiddleware should accept custom guidelines."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_evaluator.return_value = Feedback(value=True, rationale="OK")
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import ToneGuardrailMiddleware

            custom = "Always respond like a pirate. {{ inputs }} {{ outputs }}"
            middleware = ToneGuardrailMiddleware(
                model="databricks:/test-judge",
                tone="professional",  # overridden by custom_guidelines
                custom_guidelines=custom,
            )

            assert middleware.prompt == custom

    def test_invalid_tone_raises_error(self):
        """ToneGuardrailMiddleware should raise ValueError for unknown tone."""
        with patch("dao_ai.middleware.guardrails.make_judge"):
            from dao_ai.middleware.guardrails import ToneGuardrailMiddleware

            with pytest.raises(ValueError, match="Unknown tone profile"):
                ToneGuardrailMiddleware(
                    model="databricks:/test-judge",
                    tone="nonexistent_profile",
                )

    def test_all_preset_profiles_exist(self):
        """All documented tone profiles should be available."""
        from dao_ai.middleware.guardrails import TONE_PROFILES

        expected = {"professional", "casual", "technical", "empathetic", "concise"}
        assert set(TONE_PROFILES.keys()) == expected


class TestConcisenessGuardrailMiddleware:
    """Tests for ConcisenessGuardrailMiddleware."""

    def test_rejects_too_long_response(self, runtime):
        """Conciseness guardrail should reject responses exceeding max_length."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import ConcisenessGuardrailMiddleware

            middleware = ConcisenessGuardrailMiddleware(
                model="databricks:/test-judge",
                max_length=100,
                min_length=10,
                check_verbosity=False,
            )

            state: AgentState = {
                "messages": [
                    HumanMessage(content="Test"),
                    AIMessage(content="x" * 200),  # Exceeds 100
                ]
            }

            result = middleware.after_model(state, runtime)

            assert result is not None
            assert isinstance(result["messages"][0], HumanMessage)
            assert "concise" in result["messages"][0].content.lower()
            # Should NOT call the LLM judge for deterministic check
            mock_evaluator.assert_not_called()

    def test_rejects_too_short_response(self, runtime):
        """Conciseness guardrail should reject responses below min_length."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import ConcisenessGuardrailMiddleware

            middleware = ConcisenessGuardrailMiddleware(
                model="databricks:/test-judge",
                max_length=1000,
                min_length=50,
                check_verbosity=False,
            )

            state: AgentState = {
                "messages": [
                    HumanMessage(content="Tell me about Python"),
                    AIMessage(content="OK."),  # Below 50
                ]
            }

            result = middleware.after_model(state, runtime)

            assert result is not None
            assert isinstance(result["messages"][0], HumanMessage)
            assert "complete" in result["messages"][0].content.lower()

    def test_passes_length_check_no_verbosity(self, runtime):
        """Conciseness guardrail should pass when length is OK and verbosity disabled."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import ConcisenessGuardrailMiddleware

            middleware = ConcisenessGuardrailMiddleware(
                model="databricks:/test-judge",
                max_length=1000,
                min_length=10,
                check_verbosity=False,
            )

            state: AgentState = {
                "messages": [
                    HumanMessage(content="Tell me about Python"),
                    AIMessage(content="Python is a high-level programming language."),
                ]
            }

            result = middleware.after_model(state, runtime)

            # Should pass -- length OK and verbosity disabled
            assert result is None
            mock_evaluator.assert_not_called()

    def test_delegates_to_llm_when_verbosity_enabled(self, runtime):
        """Conciseness guardrail should call LLM judge when verbosity check is enabled."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_evaluator.return_value = Feedback(
                value=True, rationale="Appropriately concise."
            )
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import ConcisenessGuardrailMiddleware

            middleware = ConcisenessGuardrailMiddleware(
                model="databricks:/test-judge",
                max_length=1000,
                min_length=10,
                check_verbosity=True,
            )

            state: AgentState = {
                "messages": [
                    HumanMessage(content="Tell me about Python"),
                    AIMessage(content="Python is a high-level programming language."),
                ]
            }

            result = middleware.after_model(state, runtime)

            # Should pass after LLM check
            assert result is None
            mock_evaluator.assert_called_once()

    def test_max_retries_too_long(self, runtime):
        """Conciseness guardrail should give up after max retries for too-long."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_make_judge.return_value = mock_evaluator

            from dao_ai.middleware.guardrails import ConcisenessGuardrailMiddleware

            middleware = ConcisenessGuardrailMiddleware(
                model="databricks:/test-judge",
                max_length=50,
                min_length=10,
                check_verbosity=False,
                num_retries=2,
            )

            state: AgentState = {
                "messages": [
                    HumanMessage(content="Test"),
                    AIMessage(content="x" * 200),
                ]
            }

            # First retry
            result = middleware.after_model(state, runtime)
            assert isinstance(result["messages"][0], HumanMessage)

            # Second retry -- max reached, should return failure AIMessage
            result = middleware.after_model(state, runtime)
            assert isinstance(result["messages"][0], AIMessage)
            assert "Quality Check Failed" in result["messages"][0].content


# =============================================================================
# JudgeScorer Tests
# =============================================================================


class TestJudgeScorer:
    """Tests for JudgeScorer -- MLflow Scorer wrapping make_judge."""

    def test_is_scorer_subclass(self):
        """JudgeScorer should be a Scorer subclass."""
        assert issubclass(JudgeScorer, Scorer)

    def test_creates_evaluator_from_make_judge(self):
        """JudgeScorer should call make_judge during init."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            mock_evaluator = Mock()
            mock_make_judge.return_value = mock_evaluator

            scorer = JudgeScorer(
                name="test_scorer",
                instructions="Evaluate {{ inputs }} and {{ outputs }}.",
                model="databricks:/test-judge",
            )

            mock_make_judge.assert_called_once_with(
                name="test_scorer",
                instructions="Evaluate {{ inputs }} and {{ outputs }}.",
                feedback_value_type=bool,
                model="databricks:/test-judge",
            )
            assert scorer.name == "test_scorer"

    def test_call_delegates_to_evaluator(self):
        """JudgeScorer.__call__ should delegate to the underlying evaluator."""
        with patch("dao_ai.middleware.guardrails.make_judge") as mock_make_judge:
            expected_feedback = Feedback(value=True, rationale="Good response.")
            mock_evaluator = Mock(return_value=expected_feedback)
            mock_make_judge.return_value = mock_evaluator

            scorer = JudgeScorer(
                name="test",
                instructions="test prompt",
                model="databricks:/test",
            )

            result = scorer(
                inputs={"query": "hello"},
                outputs={"response": "hi there"},
            )

            assert result is expected_feedback
            mock_evaluator.assert_called_once_with(
                inputs={"query": "hello"},
                outputs={"response": "hi there"},
            )


# =============================================================================
# _interpret_feedback Tests
# =============================================================================


class TestInterpretFeedback:
    """Tests for _interpret_feedback helper."""

    def test_bool_true(self):
        """True boolean value should map to passed."""
        passed, comment = _interpret_feedback(True)
        assert passed is True
        assert comment == ""

    def test_bool_false(self):
        """False boolean value should map to failed."""
        passed, comment = _interpret_feedback(False)
        assert passed is False
        assert comment == ""

    def test_feedback_with_bool_value(self):
        """Feedback with boolean value should extract value and rationale."""
        feedback = Feedback(value=True, rationale="Looks good.")
        passed, comment = _interpret_feedback(feedback)
        assert passed is True
        assert comment == "Looks good."

    def test_feedback_with_false_bool(self):
        """Feedback with False value should map to failed."""
        feedback = Feedback(value=False, rationale="Contains errors.")
        passed, comment = _interpret_feedback(feedback)
        assert passed is False
        assert comment == "Contains errors."

    def test_feedback_with_yes_string(self):
        """Feedback with 'yes' string should map to passed."""
        feedback = Feedback(value="yes", rationale="Passed check.")
        passed, comment = _interpret_feedback(feedback)
        assert passed is True
        assert comment == "Passed check."

    def test_feedback_with_no_string(self):
        """Feedback with 'no' string should map to failed."""
        feedback = Feedback(value="no", rationale="Failed check.")
        passed, comment = _interpret_feedback(feedback)
        assert passed is False
        assert comment == "Failed check."

    def test_feedback_with_no_rationale(self):
        """Feedback with no rationale should return empty comment."""
        feedback = Feedback(value=True)
        passed, comment = _interpret_feedback(feedback)
        assert passed is True
        assert comment == ""

    def test_string_pass(self):
        """String 'pass' should map to passed."""
        passed, comment = _interpret_feedback("pass")
        assert passed is True

    def test_string_safe(self):
        """String 'safe' should map to passed."""
        passed, comment = _interpret_feedback("safe")
        assert passed is True

    def test_string_true(self):
        """String 'true' should map to passed."""
        passed, comment = _interpret_feedback("true")
        assert passed is True

    def test_string_false(self):
        """String 'false' should map to failed."""
        passed, comment = _interpret_feedback("false")
        assert passed is False

    def test_string_unsafe(self):
        """String 'unsafe' should map to failed."""
        passed, comment = _interpret_feedback("unsafe")
        assert passed is False

    def test_integer_one(self):
        """Integer 1 should map to passed (truthy)."""
        passed, comment = _interpret_feedback(1)
        assert passed is True

    def test_integer_zero(self):
        """Integer 0 should map to failed (falsy)."""
        passed, comment = _interpret_feedback(0)
        assert passed is False

    def test_float_one(self):
        """Float 1.0 should map to passed."""
        passed, comment = _interpret_feedback(1.0)
        assert passed is True


# =============================================================================
# Scorer-based GuardrailMiddleware Tests
# =============================================================================


class TestScorerGuardrailMiddleware:
    """Tests for GuardrailMiddleware initialized with a Scorer instance."""

    def test_accepts_scorer_instance(self, runtime):
        """GuardrailMiddleware should accept a Scorer directly."""
        mock_scorer = Mock(spec=Scorer)
        mock_scorer.return_value = Feedback(value=True, rationale="OK")

        middleware = GuardrailMiddleware(
            name="test_scorer_guard",
            scorer=mock_scorer,
            num_retries=2,
        )

        assert middleware._scorer is mock_scorer
        assert middleware.guardrail_name == "test_scorer_guard"
        assert middleware.model_endpoint is None
        assert middleware.prompt is None

    def test_scorer_evaluation_pass(self, runtime):
        """Scorer returning YES/True should let the response through."""
        mock_scorer = Mock(spec=Scorer)
        mock_scorer.return_value = Feedback(value=True, rationale="Clean.")

        middleware = GuardrailMiddleware(
            name="test_guard",
            scorer=mock_scorer,
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ]
        }

        result = middleware.after_model(state, runtime)
        assert result is None
        mock_scorer.assert_called_once()

    def test_scorer_evaluation_fail_triggers_retry(self, runtime):
        """Scorer returning NO/False should trigger retry."""
        mock_scorer = Mock(spec=Scorer)
        mock_scorer.return_value = Feedback(
            value="no",
            rationale="Contains PII.",
        )

        middleware = GuardrailMiddleware(
            name="pii_guard",
            scorer=mock_scorer,
            num_retries=2,
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="Show my data"),
                AIMessage(content="Your SSN is 123-45-6789."),
            ]
        }

        result = middleware.after_model(state, runtime)
        assert result is not None
        assert isinstance(result["messages"][0], HumanMessage)
        assert "Contains PII" in result["messages"][0].content

    def test_scorer_error_lets_through_by_default(self, runtime):
        """Scorer exception should let response through when fail_on_error=False (default)."""
        mock_scorer = Mock(spec=Scorer)
        mock_scorer.side_effect = RuntimeError("Scorer crashed")

        middleware = GuardrailMiddleware(
            name="crash_guard",
            scorer=mock_scorer,
            fail_on_error=False,
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi!"),
            ]
        }

        result = middleware.after_model(state, runtime)
        assert result is None

    def test_rejects_scorer_with_model(self):
        """Should raise ValueError when both scorer and model are provided."""
        mock_scorer = Mock(spec=Scorer)
        with pytest.raises(ValueError, match="Cannot specify both"):
            GuardrailMiddleware(
                name="bad",
                scorer=mock_scorer,
                model="databricks:/test",
                prompt="test",
            )

    def test_requires_scorer_or_model_prompt(self):
        """Should raise ValueError when neither scorer nor model/prompt provided."""
        with pytest.raises(ValueError, match="Either 'scorer' or both"):
            GuardrailMiddleware(name="bad")

    def test_requires_both_model_and_prompt(self):
        """Should raise ValueError when model is provided without prompt."""
        with pytest.raises(ValueError, match="Either 'scorer' or both"):
            GuardrailMiddleware(name="bad", model="databricks:/test")


# =============================================================================
# GuardrailModel Config Tests
# =============================================================================


class TestGuardrailModelConfig:
    """Tests for GuardrailModel config validation."""

    def test_custom_judge_config(self):
        """GuardrailModel should accept model + prompt for custom judges."""
        from dao_ai.config import GuardrailModel

        model = GuardrailModel(
            name="tone_check",
            model="databricks-claude-3-7-sonnet",
            prompt="Evaluate {{ inputs }} and {{ outputs }}.",
        )
        assert model.name == "tone_check"
        assert model.scorer is None
        assert model.model is not None
        assert model.prompt is not None

    def test_scorer_config(self):
        """GuardrailModel should accept scorer for scorer-based guardrails."""
        from dao_ai.config import GuardrailModel

        model = GuardrailModel(
            name="pii_check",
            scorer="mlflow.genai.scorers.guardrails.DetectPII",
            scorer_args={"pii_entities": ["CREDIT_CARD", "SSN"]},
        )
        assert model.name == "pii_check"
        assert model.scorer == "mlflow.genai.scorers.guardrails.DetectPII"
        assert model.scorer_args == {"pii_entities": ["CREDIT_CARD", "SSN"]}
        assert model.model is None
        assert model.prompt is None

    def test_rejects_scorer_and_model(self):
        """GuardrailModel should reject both scorer and model/prompt."""
        from dao_ai.config import GuardrailModel

        with pytest.raises(ValueError, match="Cannot specify both"):
            GuardrailModel(
                name="bad",
                scorer="mlflow.genai.scorers.guardrails.ToxicLanguage",
                model="databricks-claude-3-7-sonnet",
                prompt="test",
            )

    def test_rejects_no_scorer_no_model(self):
        """GuardrailModel should reject when nothing is provided."""
        from dao_ai.config import GuardrailModel

        with pytest.raises(ValueError, match="Either 'scorer' or both"):
            GuardrailModel(name="bad")

    def test_rejects_model_without_prompt(self):
        """GuardrailModel should reject model without prompt."""
        from dao_ai.config import GuardrailModel

        with pytest.raises(ValueError, match="Both 'model' and 'prompt'"):
            GuardrailModel(
                name="bad",
                model="databricks-claude-3-7-sonnet",
            )

    def test_scorer_with_defaults(self):
        """GuardrailModel scorer config should have correct defaults."""
        from dao_ai.config import GuardrailModel

        model = GuardrailModel(
            name="toxic_check",
            scorer="mlflow.genai.scorers.guardrails.ToxicLanguage",
        )
        assert model.scorer_args == {}
        assert model.hub is None
        assert model.num_retries == 3
        assert model.fail_on_error is False
        assert model.max_context_length == 8000

    def test_scorer_with_hub(self):
        """GuardrailModel should accept hub with scorer."""
        from dao_ai.config import GuardrailModel

        model = GuardrailModel(
            name="toxic_check",
            scorer="mlflow.genai.scorers.guardrails.ToxicLanguage",
            hub="hub://guardrails/toxic_language",
        )
        assert model.hub == "hub://guardrails/toxic_language"

    def test_rejects_hub_without_scorer(self):
        """GuardrailModel should reject hub when scorer is not set."""
        from dao_ai.config import GuardrailModel

        with pytest.raises(ValueError, match="'hub' requires 'scorer'"):
            GuardrailModel(
                name="bad",
                model="databricks-claude-3-7-sonnet",
                prompt="Evaluate {{ inputs }} and {{ outputs }}.",
                hub="hub://guardrails/toxic_language",
            )


class TestEnsureGuardrailsHub:
    """Tests for the ensure_guardrails_hub auto-install logic."""

    def test_noop_when_no_hub_uris(self):
        """Should return immediately when no guardrails have hub fields."""
        from dao_ai.guardrails_hub import _collect_hub_uris, ensure_guardrails_hub

        config = Mock()
        config.guardrails = {}
        config.agents = {}

        ensure_guardrails_hub(config)

        uris = _collect_hub_uris(config)
        assert uris == set()

    def test_collect_hub_uris_from_guardrails(self):
        """Should collect hub URIs from top-level guardrails."""
        from dao_ai.guardrails_hub import _collect_hub_uris

        g1 = Mock()
        g1.hub = "hub://guardrails/toxic_language"
        g2 = Mock()
        g2.hub = None
        g3 = Mock()
        g3.hub = "hub://guardrails/detect_pii"

        config = Mock()
        config.guardrails = {"g1": g1, "g2": g2, "g3": g3}
        config.agents = {}

        uris = _collect_hub_uris(config)
        assert uris == {
            "hub://guardrails/toxic_language",
            "hub://guardrails/detect_pii",
        }

    def test_collect_hub_uris_from_agent_guardrails(self):
        """Should collect hub URIs from agent-level guardrails."""
        from dao_ai.guardrails_hub import _collect_hub_uris

        g1 = Mock()
        g1.hub = "hub://guardrails/gibberish_text"

        agent = Mock()
        agent.guardrails = [g1]

        config = Mock()
        config.guardrails = {}
        config.agents = {"agent1": agent}

        uris = _collect_hub_uris(config)
        assert uris == {"hub://guardrails/gibberish_text"}

    def test_hub_uri_to_registry_key(self):
        """Should extract registry keys from hub URIs."""
        from dao_ai.guardrails_hub import _hub_uri_to_registry_key

        assert (
            _hub_uri_to_registry_key("hub://guardrails/toxic_language")
            == "guardrails/toxic_language"
        )
        assert (
            _hub_uri_to_registry_key("hub://guardrails/detect_pii")
            == "guardrails/detect_pii"
        )
        assert (
            _hub_uri_to_registry_key("hub://guardrails/gibberish_text")
            == "guardrails/gibberish_text"
        )
        assert (
            _hub_uri_to_registry_key("hub://guardrails/secrets_present")
            == "guardrails/secrets_present"
        )
        assert (
            _hub_uri_to_registry_key("hub://guardrails/nsfw_text/")
            == "guardrails/nsfw_text"
        )

    def test_warns_when_no_api_key_and_no_token(self):
        """Should warn when hub URIs exist but no API key is available."""
        from dao_ai.guardrails_hub import ensure_guardrails_hub

        g1 = Mock()
        g1.hub = "hub://guardrails/toxic_language"

        config = Mock()
        config.guardrails = {"g1": g1}
        config.agents = {}

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("dao_ai.guardrails_hub._has_existing_token", return_value=False),
            patch("dao_ai.guardrails_hub.logger") as mock_logger,
        ):
            ensure_guardrails_hub(config)
            mock_logger.warning.assert_called_once()
            assert "GUARDRAILSAI_API_KEY" in str(mock_logger.warning.call_args)

    def test_configures_token_from_env(self):
        """Should configure guardrails token from env var."""
        from dao_ai.guardrails_hub import ensure_guardrails_hub

        g1 = Mock()
        g1.hub = "hub://guardrails/toxic_language"

        config = Mock()
        config.guardrails = {"g1": g1}
        config.agents = {}

        with (
            patch.dict("os.environ", {"GUARDRAILSAI_API_KEY": "test-token"}),
            patch("dao_ai.guardrails_hub._configure_token") as mock_configure,
            patch(
                "dao_ai.guardrails_hub._is_validator_installed",
                return_value=True,
            ),
            patch("guardrails.install"),
        ):
            ensure_guardrails_hub(config)
            mock_configure.assert_called_once_with("test-token")

    def test_skips_already_installed_validators(self):
        """Should skip validators that are already installed."""
        from dao_ai.guardrails_hub import ensure_guardrails_hub

        g1 = Mock()
        g1.hub = "hub://guardrails/toxic_language"

        config = Mock()
        config.guardrails = {"g1": g1}
        config.agents = {}

        with (
            patch.dict("os.environ", {"GUARDRAILSAI_API_KEY": "test-token"}),
            patch("dao_ai.guardrails_hub._configure_token"),
            patch(
                "dao_ai.guardrails_hub._is_validator_installed",
                return_value=True,
            ),
            patch("guardrails.install") as mock_install,
        ):
            ensure_guardrails_hub(config)
            mock_install.assert_not_called()

    def test_installs_missing_validators(self):
        """Should install validators that are not yet installed."""
        from dao_ai.guardrails_hub import ensure_guardrails_hub

        g1 = Mock()
        g1.hub = "hub://guardrails/toxic_language"

        config = Mock()
        config.guardrails = {"g1": g1}
        config.agents = {}

        with (
            patch.dict("os.environ", {"GUARDRAILSAI_API_KEY": "test-token"}),
            patch("dao_ai.guardrails_hub._configure_token"),
            patch(
                "dao_ai.guardrails_hub._is_validator_installed",
                return_value=False,
            ),
            patch("guardrails.install") as mock_install,
        ):
            ensure_guardrails_hub(config)
            mock_install.assert_called_once_with(
                "hub://guardrails/toxic_language", quiet=True
            )

    def test_uses_existing_token_when_no_env(self):
        """Should use existing ~/.guardrailsrc token when env var is not set."""
        from dao_ai.guardrails_hub import ensure_guardrails_hub

        g1 = Mock()
        g1.hub = "hub://guardrails/toxic_language"

        config = Mock()
        config.guardrails = {"g1": g1}
        config.agents = {}

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("dao_ai.guardrails_hub._has_existing_token", return_value=True),
            patch("dao_ai.guardrails_hub._configure_token") as mock_configure,
            patch(
                "dao_ai.guardrails_hub._is_validator_installed",
                return_value=True,
            ),
            patch("guardrails.install"),
        ):
            ensure_guardrails_hub(config)
            mock_configure.assert_not_called()

    def test_falls_back_to_config_environment_vars(self):
        """Should resolve API key from config.app.environment_vars when
        os.environ does not have it (e.g. Databricks secret already resolved)."""
        from dao_ai.guardrails_hub import ensure_guardrails_hub

        g1 = Mock()
        g1.hub = "hub://guardrails/toxic_language"

        app = Mock()
        app.environment_vars = {"GUARDRAILSAI_API_KEY": "secret-from-config"}

        config = Mock()
        config.guardrails = {"g1": g1}
        config.agents = {}
        config.app = app

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("dao_ai.guardrails_hub._configure_token") as mock_configure,
            patch(
                "dao_ai.guardrails_hub._is_validator_installed",
                return_value=True,
            ),
            patch("guardrails.install"),
        ):
            ensure_guardrails_hub(config)
            mock_configure.assert_called_once_with("secret-from-config")

    def test_ignores_unresolved_secret_template_in_config(self):
        """Should not use {{secrets/...}} template strings as API keys."""
        from dao_ai.guardrails_hub import ensure_guardrails_hub

        g1 = Mock()
        g1.hub = "hub://guardrails/toxic_language"

        app = Mock()
        app.environment_vars = {
            "GUARDRAILSAI_API_KEY": "{{secrets/scope/GUARDRAILSAI_API_KEY}}"
        }

        config = Mock()
        config.guardrails = {"g1": g1}
        config.agents = {}
        config.app = app

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("dao_ai.guardrails_hub._has_existing_token", return_value=False),
            patch("dao_ai.guardrails_hub.logger") as mock_logger,
        ):
            ensure_guardrails_hub(config)
            mock_logger.warning.assert_called_once()


class TestResolveEnvironmentVarsSecretSource:
    """Tests for AppModel._find_secret_source and resolve_environment_vars
    handling of CompositeVariableModel with SecretVariableModel first option."""

    def test_find_secret_source_bare_secret(self):
        """Should return the SecretVariableModel itself."""
        from dao_ai.config import AppModel, SecretVariableModel

        secret = SecretVariableModel(scope="my_scope", secret="MY_KEY")
        result = AppModel._find_secret_source(secret)
        assert result is secret

    def test_find_secret_source_composite_with_secret_first(self):
        """Should return the first option when it's a SecretVariableModel."""
        from dao_ai.config import (
            AppModel,
            CompositeVariableModel,
            EnvironmentVariableModel,
            SecretVariableModel,
        )

        secret = SecretVariableModel(scope="my_scope", secret="MY_KEY")
        env = EnvironmentVariableModel(env="MY_KEY")
        composite = CompositeVariableModel(options=[secret, env])

        result = AppModel._find_secret_source(composite)
        assert result is secret

    def test_find_secret_source_composite_with_env_first(self):
        """Should return None when first option is not a SecretVariableModel."""
        from dao_ai.config import (
            AppModel,
            CompositeVariableModel,
            EnvironmentVariableModel,
        )

        env = EnvironmentVariableModel(env="MY_KEY")
        composite = CompositeVariableModel(options=[env])

        result = AppModel._find_secret_source(composite)
        assert result is None

    def test_find_secret_source_plain_string(self):
        """Should return None for plain string values."""
        from dao_ai.config import AppModel

        result = AppModel._find_secret_source("plain-value")
        assert result is None

    def test_find_secret_source_empty_composite(self):
        """Should return None for a CompositeVariableModel with no options."""
        from dao_ai.config import AppModel, CompositeVariableModel

        composite = CompositeVariableModel(options=[])
        result = AppModel._find_secret_source(composite)
        assert result is None

    def test_composite_secret_produces_template_string(self):
        """A CompositeVariableModel with a SecretVariableModel first option
        should produce the {{secrets/scope/key}} template via _find_secret_source."""
        from dao_ai.config import (
            AppModel,
            CompositeVariableModel,
            EnvironmentVariableModel,
            SecretVariableModel,
        )

        secret = SecretVariableModel(scope="retail_consumer_goods", secret="API_KEY")
        env = EnvironmentVariableModel(env="API_KEY")
        composite = CompositeVariableModel(options=[secret, env])

        source = AppModel._find_secret_source(composite)
        assert source is secret
        assert str(source) == "{{secrets/retail_consumer_goods/API_KEY}}"


# =============================================================================
# apply_to Config + before_model / after_model Gating Tests
# =============================================================================


class TestGuardrailModelApplyTo:
    """Tests for the apply_to field on GuardrailModel."""

    def test_defaults_to_both(self):
        from dao_ai.config import GuardrailModel

        model = GuardrailModel(
            name="test",
            scorer="mlflow.genai.scorers.guardrails.ToxicLanguage",
        )
        assert model.apply_to == "both"

    def test_accepts_input(self):
        from dao_ai.config import GuardrailModel

        model = GuardrailModel(
            name="test",
            scorer="mlflow.genai.scorers.guardrails.ToxicLanguage",
            apply_to="input",
        )
        assert model.apply_to == "input"

    def test_accepts_output(self):
        from dao_ai.config import GuardrailModel

        model = GuardrailModel(
            name="test",
            scorer="mlflow.genai.scorers.guardrails.ToxicLanguage",
            apply_to="output",
        )
        assert model.apply_to == "output"

    def test_accepts_both(self):
        from dao_ai.config import GuardrailModel

        model = GuardrailModel(
            name="test",
            scorer="mlflow.genai.scorers.guardrails.ToxicLanguage",
            apply_to="both",
        )
        assert model.apply_to == "both"

    def test_rejects_invalid_value(self):
        from dao_ai.config import GuardrailModel

        with pytest.raises(Exception):
            GuardrailModel(
                name="test",
                scorer="mlflow.genai.scorers.guardrails.ToxicLanguage",
                apply_to="never",
            )


class TestGuardrailApplyToHooks:
    """Tests for before_model / after_model gating via apply_to."""

    def _make_middleware(
        self,
        apply_to: str = "both",
        fail_on_error: bool = False,
    ) -> tuple[GuardrailMiddleware, Mock]:
        mock_scorer = Mock(spec=Scorer)
        mock_scorer.return_value = Feedback(value=True, rationale="OK")
        middleware = GuardrailMiddleware(
            name="test_guard",
            scorer=mock_scorer,
            apply_to=apply_to,
            fail_on_error=fail_on_error,
        )
        return middleware, mock_scorer

    def _input_state(self) -> AgentState:
        return {"messages": [HumanMessage(content="Hello world")]}

    def _output_state(self) -> AgentState:
        return {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
            ]
        }

    # -- before_model tests --

    def test_before_model_runs_when_apply_to_input(self, runtime):
        mw, scorer = self._make_middleware(apply_to="input")
        result = mw.before_model(self._input_state(), runtime)
        assert result is None  # pass => None
        scorer.assert_called_once()

    def test_before_model_runs_when_apply_to_both(self, runtime):
        mw, scorer = self._make_middleware(apply_to="both")
        result = mw.before_model(self._input_state(), runtime)
        assert result is None
        scorer.assert_called_once()

    def test_before_model_skipped_when_apply_to_output(self, runtime):
        mw, scorer = self._make_middleware(apply_to="output")
        result = mw.before_model(self._input_state(), runtime)
        assert result is None
        scorer.assert_not_called()

    def test_before_model_blocks_on_failure(self, runtime):
        mw, scorer = self._make_middleware(apply_to="input")
        scorer.return_value = Feedback(value=False, rationale="Toxic content.")

        result = mw.before_model(self._input_state(), runtime)
        assert result is not None
        assert result["jump_to"] == "end"
        assert isinstance(result["messages"][0], AIMessage)
        assert "Input Blocked" in result["messages"][0].content
        assert "Toxic content" in result["messages"][0].content

    def test_before_model_error_lets_through_by_default(self, runtime):
        mw, scorer = self._make_middleware(apply_to="input", fail_on_error=False)
        scorer.side_effect = RuntimeError("boom")

        result = mw.before_model(self._input_state(), runtime)
        assert result is None

    def test_before_model_error_blocks_when_fail_on_error(self, runtime):
        mw, scorer = self._make_middleware(apply_to="input", fail_on_error=True)
        scorer.side_effect = RuntimeError("boom")

        result = mw.before_model(self._input_state(), runtime)
        assert result is not None
        assert result["jump_to"] == "end"
        assert "Input Check Error" in result["messages"][0].content

    def test_before_model_no_messages(self, runtime):
        mw, scorer = self._make_middleware(apply_to="input")
        result = mw.before_model({"messages": []}, runtime)
        assert result is None
        scorer.assert_not_called()

    # -- after_model tests --

    def test_after_model_runs_when_apply_to_output(self, runtime):
        mw, scorer = self._make_middleware(apply_to="output")
        result = mw.after_model(self._output_state(), runtime)
        assert result is None
        scorer.assert_called_once()

    def test_after_model_runs_when_apply_to_both(self, runtime):
        mw, scorer = self._make_middleware(apply_to="both")
        result = mw.after_model(self._output_state(), runtime)
        assert result is None
        scorer.assert_called_once()

    def test_after_model_skipped_when_apply_to_input(self, runtime):
        mw, scorer = self._make_middleware(apply_to="input")
        result = mw.after_model(self._output_state(), runtime)
        assert result is None
        scorer.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
