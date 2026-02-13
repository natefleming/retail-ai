"""
Test guardrail retry behavior.

This test verifies that guardrails properly retry when evaluations fail,
extract tool context, handle errors gracefully, and maintain thread-safe
retry state.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from mlflow.entities import Feedback

from dao_ai.middleware.guardrails import (
    GuardrailMiddleware,
    _extract_text_content,
    _extract_tool_context,
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


def test_guardrail_fail_open_on_error(runtime):
    """
    Test that guardrail lets response through when judge call fails (fail_open=True).
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
            fail_open=True,
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="Test question"),
                AIMessage(content="Test response"),
            ]
        }

        result = middleware.after_model(state, runtime)

        # Should let response through (fail open)
        assert result is None


def test_guardrail_fail_closed_on_error(runtime):
    """
    Test that guardrail blocks response when judge call fails (fail_open=False).
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
            fail_open=False,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
