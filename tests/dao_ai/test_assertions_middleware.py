"""
Tests for DSPy-style assertion middleware.

This module tests the assertion middleware implementations:
- AssertMiddleware: Hard constraints with retry (like dspy.Assert)
- SuggestMiddleware: Soft constraints with feedback (like dspy.Suggest)
- RefineMiddleware: Iterative improvement (like dspy.Refine)
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from dao_ai.middleware.assertions import (
    AssertMiddleware,
    ConstraintResult,
    FunctionConstraint,
    KeywordConstraint,
    LengthConstraint,
    RefineMiddleware,
    SuggestMiddleware,
    create_assert_middleware,
    create_refine_middleware,
    create_suggest_middleware,
)
from dao_ai.state import AgentState, Context


def create_mock_runtime(context: Context | None = None) -> MagicMock:
    """Create a mock runtime with the given context."""
    runtime = MagicMock()
    runtime.context = context or Context()
    return runtime


# =============================================================================
# Constraint Tests
# =============================================================================


@pytest.mark.unit
class TestConstraints:
    """Tests for constraint implementations."""

    def test_function_constraint_bool_return(self) -> None:
        """Test FunctionConstraint with bool-returning function."""

        def has_greeting(response: str, ctx: dict[str, Any]) -> bool:
            return "hello" in response.lower() or "hi" in response.lower()

        constraint = FunctionConstraint(has_greeting, name="greeting_check")

        result = constraint.evaluate("Hello there!", {})
        assert result.passed is True

        result = constraint.evaluate("Goodbye!", {})
        assert result.passed is False

    def test_function_constraint_result_return(self) -> None:
        """Test FunctionConstraint with ConstraintResult-returning function."""

        def check_format(response: str, ctx: dict[str, Any]) -> ConstraintResult:
            if response.startswith("Answer:"):
                return ConstraintResult(passed=True, feedback="Correct format")
            return ConstraintResult(
                passed=False, feedback="Response should start with 'Answer:'"
            )

        constraint = FunctionConstraint(check_format)

        result = constraint.evaluate("Answer: 42", {})
        assert result.passed is True
        assert result.feedback == "Correct format"

        result = constraint.evaluate("The answer is 42", {})
        assert result.passed is False
        assert "Answer:" in result.feedback

    def test_keyword_constraint_required(self) -> None:
        """Test KeywordConstraint with required keywords."""
        constraint = KeywordConstraint(
            required_keywords=["source", "reference"], name="sources_required"
        )

        result = constraint.evaluate(
            "According to the source, this is referenced here.", {}
        )
        assert result.passed is True

        result = constraint.evaluate("I think the answer is 42.", {})
        assert result.passed is False
        assert "source" in result.feedback.lower()

    def test_keyword_constraint_banned(self) -> None:
        """Test KeywordConstraint with banned keywords."""
        constraint = KeywordConstraint(
            banned_keywords=["password", "secret"], name="no_secrets"
        )

        result = constraint.evaluate("Here is the information you requested.", {})
        assert result.passed is True

        result = constraint.evaluate("Your password is abc123.", {})
        assert result.passed is False
        assert "password" in result.feedback.lower()

    def test_length_constraint_chars(self) -> None:
        """Test LengthConstraint with character count."""
        constraint = LengthConstraint(min_length=10, max_length=100, unit="chars")

        result = constraint.evaluate("Short", {})
        assert result.passed is False
        assert "too short" in result.feedback.lower()

        result = constraint.evaluate("This is a valid response.", {})
        assert result.passed is True

        result = constraint.evaluate("x" * 150, {})
        assert result.passed is False
        assert "too long" in result.feedback.lower()

    def test_length_constraint_words(self) -> None:
        """Test LengthConstraint with word count."""
        constraint = LengthConstraint(min_length=5, max_length=20, unit="words")

        result = constraint.evaluate("Too few", {})
        assert result.passed is False

        result = constraint.evaluate("This response has exactly five words.", {})
        # Should have 6 words, which is within range
        assert result.passed is True


# =============================================================================
# AssertMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestAssertMiddleware:
    """Tests for AssertMiddleware (hard constraints with retry)."""

    def test_passes_when_constraint_satisfied(self) -> None:
        """Test that middleware passes when constraint is satisfied."""

        def always_pass(response: str, ctx: dict[str, Any]) -> bool:
            return True

        middleware = AssertMiddleware(
            constraint=FunctionConstraint(always_pass), max_retries=3
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        }

        result = middleware.after_model(state, create_mock_runtime())
        assert result is None  # No state update needed

    def test_retries_when_constraint_fails(self) -> None:
        """Test that middleware adds retry message when constraint fails."""

        def always_fail(response: str, ctx: dict[str, Any]) -> ConstraintResult:
            return ConstraintResult(passed=False, feedback="Not good enough")

        middleware = AssertMiddleware(
            constraint=FunctionConstraint(always_fail), max_retries=3
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        }

        result = middleware.after_model(state, create_mock_runtime())

        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)
        assert "Not good enough" in result["messages"][0].content

    def test_raises_error_after_max_retries(self) -> None:
        """Test that middleware raises error after max retries exhausted."""

        def always_fail(response: str, ctx: dict[str, Any]) -> bool:
            return False

        middleware = AssertMiddleware(
            constraint=FunctionConstraint(always_fail),
            max_retries=2,
            on_failure="error",
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        }

        # First retry
        result = middleware.after_model(state, create_mock_runtime())
        assert result is not None

        # Second retry - should raise
        with pytest.raises(ValueError) as exc_info:
            middleware.after_model(state, create_mock_runtime())

        assert "failed after 2 retries" in str(exc_info.value)

    def test_fallback_message_on_failure(self) -> None:
        """Test that middleware returns fallback message when configured."""

        def always_fail(response: str, ctx: dict[str, Any]) -> bool:
            return False

        middleware = AssertMiddleware(
            constraint=FunctionConstraint(always_fail),
            max_retries=1,
            on_failure="fallback",
            fallback_message="Sorry, I couldn't help.",
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        }

        # Exhaust retries
        middleware.after_model(
            state, create_mock_runtime()
        )  # First call, triggers retry

        # Second call should use fallback
        middleware.after_model(state, create_mock_runtime())

        # The AI message should be modified
        ai_msg = state["messages"][-1]
        assert ai_msg.content == "Sorry, I couldn't help."

    def test_pass_through_on_failure(self) -> None:
        """Test that middleware passes through when on_failure='pass'."""

        def always_fail(response: str, ctx: dict[str, Any]) -> bool:
            return False

        middleware = AssertMiddleware(
            constraint=FunctionConstraint(always_fail),
            max_retries=1,
            on_failure="pass",
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="Original answer"),
            ]
        }

        # Exhaust retries
        middleware.after_model(state, create_mock_runtime())
        result = middleware.after_model(state, create_mock_runtime())

        # Should pass through without modification
        assert result is None
        assert state["messages"][-1].content == "Original answer"


# =============================================================================
# SuggestMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestSuggestMiddleware:
    """Tests for SuggestMiddleware (soft constraints with feedback)."""

    def test_passes_when_constraint_satisfied(self) -> None:
        """Test that middleware passes when constraint is satisfied."""

        def always_pass(response: str, ctx: dict[str, Any]) -> bool:
            return True

        middleware = SuggestMiddleware(constraint=FunctionConstraint(always_pass))

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        }

        result = middleware.after_model(state, create_mock_runtime())
        assert result is None

    def test_logs_feedback_but_passes_through(self) -> None:
        """Test that middleware logs feedback but doesn't block."""

        def always_fail(response: str, ctx: dict[str, Any]) -> ConstraintResult:
            return ConstraintResult(passed=False, feedback="Could be better")

        middleware = SuggestMiddleware(
            constraint=FunctionConstraint(always_fail), allow_one_retry=False
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        }

        result = middleware.after_model(state, create_mock_runtime())

        # Should pass through without modification
        assert result is None

    def test_one_retry_on_failure(self) -> None:
        """Test that middleware requests one retry when allowed."""

        def always_fail(response: str, ctx: dict[str, Any]) -> ConstraintResult:
            return ConstraintResult(passed=False, feedback="Add more detail")

        middleware = SuggestMiddleware(
            constraint=FunctionConstraint(always_fail), allow_one_retry=True
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        }

        # First call should trigger retry
        result = middleware.after_model(state, create_mock_runtime())
        assert result is not None
        assert "messages" in result
        assert "Add more detail" in result["messages"][0].content

        # Second call should pass through (only one retry allowed)
        result = middleware.after_model(state, create_mock_runtime())
        assert result is None


# =============================================================================
# RefineMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestRefineMiddleware:
    """Tests for RefineMiddleware (iterative improvement)."""

    def test_passes_when_threshold_reached(self) -> None:
        """Test that middleware passes when score meets threshold."""

        def high_score(response: str, ctx: dict[str, Any]) -> float:
            return 0.95

        middleware = RefineMiddleware(
            reward_fn=high_score, threshold=0.8, max_iterations=3
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        }

        result = middleware.after_model(state, create_mock_runtime())
        assert result is None

    def test_requests_improvement_when_below_threshold(self) -> None:
        """Test that middleware requests improvement when score is low."""

        def low_score(response: str, ctx: dict[str, Any]) -> float:
            return 0.3

        middleware = RefineMiddleware(
            reward_fn=low_score, threshold=0.8, max_iterations=3
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        }

        result = middleware.after_model(state, create_mock_runtime())

        assert result is not None
        assert "messages" in result
        assert "improve" in result["messages"][0].content.lower()

    def test_stops_after_max_iterations(self) -> None:
        """Test that middleware stops after max iterations."""

        def low_score(response: str, ctx: dict[str, Any]) -> float:
            return 0.3

        middleware = RefineMiddleware(
            reward_fn=low_score, threshold=0.8, max_iterations=2
        )

        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        }

        # First iteration - should request improvement
        result = middleware.after_model(state, create_mock_runtime())
        assert result is not None

        # Second iteration - should stop (max reached)
        result = middleware.after_model(state, create_mock_runtime())
        assert result is None

    def test_tracks_best_response(self) -> None:
        """Test that middleware tracks and uses best response."""
        scores = [0.3, 0.7, 0.4]  # Second response is best
        call_count = 0

        def varying_score(response: str, ctx: dict[str, Any]) -> float:
            nonlocal call_count
            score = scores[min(call_count, len(scores) - 1)]
            call_count += 1
            return score

        middleware = RefineMiddleware(
            reward_fn=varying_score,
            threshold=0.9,  # Won't be reached
            max_iterations=3,
            select_best=True,
        )

        # Create state with different responses for each iteration
        state: AgentState = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="Response 1"),
            ]
        }

        # First iteration (score 0.3)
        middleware.after_model(state, create_mock_runtime())

        # Second iteration (score 0.7) - this becomes the best
        state["messages"][-1] = AIMessage(content="Response 2 - Better")
        middleware.after_model(state, create_mock_runtime())

        # Third iteration (score 0.4) - worse than second
        state["messages"][-1] = AIMessage(content="Response 3 - Worse")
        middleware.after_model(state, create_mock_runtime())

        # The best response (Response 2) should be used
        assert state["messages"][-1].content == "Response 2 - Better"


# =============================================================================
# Factory Function Tests
# =============================================================================


@pytest.mark.unit
class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_assert_middleware_with_function(self) -> None:
        """Test creating AssertMiddleware with a function."""

        def check_length(response: str, ctx: dict[str, Any]) -> bool:
            return len(response) >= 10

        middleware = create_assert_middleware(
            constraint=check_length, max_retries=2, name="length_check"
        )

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, AssertMiddleware)
        assert middleware.max_retries == 2

    def test_create_assert_middleware_with_constraint(self) -> None:
        """Test creating AssertMiddleware with a Constraint object."""
        constraint = KeywordConstraint(required_keywords=["source"])

        middleware = create_assert_middleware(constraint=constraint, max_retries=3)

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, AssertMiddleware)
        assert middleware.constraint is constraint

    def test_create_suggest_middleware_with_function(self) -> None:
        """Test creating SuggestMiddleware with a function."""

        def is_polite(response: str, ctx: dict[str, Any]) -> bool:
            return "please" in response.lower() or "thank" in response.lower()

        middleware = create_suggest_middleware(
            constraint=is_polite, allow_one_retry=True
        )

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, SuggestMiddleware)
        assert middleware.allow_one_retry is True

    def test_create_refine_middleware(self) -> None:
        """Test creating RefineMiddleware."""

        def score_fn(response: str, ctx: dict[str, Any]) -> float:
            return 0.5

        middleware = create_refine_middleware(
            reward_fn=score_fn, threshold=0.9, max_iterations=5
        )

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, RefineMiddleware)
        assert middleware.threshold == 0.9
        assert middleware.max_iterations == 5
