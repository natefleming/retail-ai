"""
Test guardrail retry behavior.

This test verifies that guardrails properly retry when evaluations fail.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from dao_ai.config import LLMModel, PromptModel
from dao_ai.middleware.guardrails import GuardrailMiddleware
from dao_ai.state import AgentState, Context


@pytest.fixture
def mock_judge_model():
    """Mock judge model."""
    return LLMModel(name="test-judge", temperature=0.3)


@pytest.fixture
def mock_prompt():
    """Mock prompt model."""
    return PromptModel(
        name="test_guardrail",
        default_template="Evaluate: {inputs} and {outputs}",
    )


@pytest.fixture
def guardrail_middleware(mock_judge_model, mock_prompt):
    """Create guardrail middleware with mocked dependencies."""
    return GuardrailMiddleware(
        name="test_guardrail",
        model=mock_judge_model,
        prompt=mock_prompt,
        num_retries=3,
    )


@pytest.fixture
def runtime():
    """Mock runtime."""
    runtime = Mock(spec=Runtime)
    runtime.context = Context(user_id="test_user", thread_id="test_thread")
    return runtime


def test_guardrail_retry_on_failure(guardrail_middleware, runtime):
    """
    Test that guardrail triggers retry when evaluation fails.
    """
    # Setup state with human and AI messages
    state: AgentState = {
        "messages": [
            HumanMessage(content="What is your refund policy?"),
            AIMessage(
                content="We have a 30-day refund policy with full refund."
            ),  # Made-up info
        ]
    }

    # Mock the evaluator to fail (score=0)
    with patch("dao_ai.middleware.guardrails.create_llm_as_judge") as mock_judge:
        mock_evaluator = Mock()
        mock_evaluator.return_value = {
            "score": False,  # Failed evaluation
            "comment": "The response contains fabricated information. Please acknowledge you don't have this information.",
        }
        mock_judge.return_value = mock_evaluator

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

        # Verify retry count incremented
        assert guardrail_middleware._retry_count == 1


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

    with patch("dao_ai.middleware.guardrails.create_llm_as_judge") as mock_judge:
        mock_evaluator = Mock()
        mock_evaluator.return_value = {
            "score": False,
            "comment": "Response fails criteria.",
        }
        mock_judge.return_value = mock_evaluator

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
        assert guardrail_middleware._retry_count == 0


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

    with patch("dao_ai.middleware.guardrails.create_llm_as_judge") as mock_judge:
        mock_evaluator = Mock()

        # First call fails
        mock_evaluator.return_value = {
            "score": False,
            "comment": "Response is too brief. Provide more detail.",
        }
        mock_judge.return_value = mock_evaluator

        # First evaluation - fails
        result = guardrail_middleware.after_model(state, runtime)
        assert result is not None
        assert guardrail_middleware._retry_count == 1

        # Agent retries with better response
        state["messages"].append(result["messages"][0])  # Add feedback
        state["messages"].append(
            AIMessage(
                content="Python is a high-level, interpreted programming language known for its readability and extensive libraries."
            )
        )

        # Second call succeeds
        mock_evaluator.return_value = {
            "score": True,
            "comment": "Response is now comprehensive and accurate.",
        }

        # Second evaluation - passes
        result = guardrail_middleware.after_model(state, runtime)
        assert result is None  # No retry needed
        assert guardrail_middleware._retry_count == 0  # Reset


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

    with patch("dao_ai.middleware.guardrails.create_llm_as_judge") as mock_judge:
        # Should not be called since we skip evaluation
        result = guardrail_middleware.after_model(state, runtime)

        # Should skip evaluation and not call judge
        assert result is None
        mock_judge.assert_not_called()


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

    with patch("dao_ai.middleware.guardrails.create_llm_as_judge") as mock_judge:
        mock_evaluator = Mock()
        mock_evaluator.return_value = {"score": True, "comment": "Good"}
        mock_judge.return_value = mock_evaluator

        guardrail_middleware.after_model(state, runtime)

        # Verify evaluator was called with extracted text
        mock_evaluator.assert_called_once()
        call_args = mock_evaluator.call_args[1]
        assert "inputs" in call_args
        assert "outputs" in call_args
        # Should have extracted and joined the text blocks (with space separator)
        assert "This is structured content" in call_args["outputs"]
        assert "with multiple blocks" in call_args["outputs"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
