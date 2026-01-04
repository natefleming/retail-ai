"""
DSPy-style assertion middleware for DAO AI agents.

This module provides middleware implementations inspired by DSPy's assertion
mechanisms (dspy.Assert, dspy.Suggest, dspy.Refine) but implemented natively
in the LangChain middleware pattern for optimal latency and streaming support.

Key concepts:
- Assert: Hard constraint - retry until satisfied or fail after max attempts
- Suggest: Soft constraint - provide feedback but don't block execution
- Refine: Iterative improvement - run multiple times, select best result

These work with LangChain's middleware hooks (after_model) to validate and
improve agent outputs without requiring the DSPy library.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.runtime import Runtime
from loguru import logger

from dao_ai.messages import last_ai_message, last_human_message
from dao_ai.middleware.base import AgentMiddleware
from dao_ai.state import AgentState, Context

__all__ = [
    # Types
    "Constraint",
    "ConstraintResult",
    # Middleware classes
    "AssertMiddleware",
    "SuggestMiddleware",
    "RefineMiddleware",
    # Factory functions
    "create_assert_middleware",
    "create_suggest_middleware",
    "create_refine_middleware",
]

T = TypeVar("T")


@dataclass
class ConstraintResult:
    """Result of evaluating a constraint against model output.

    Attributes:
        passed: Whether the constraint was satisfied
        feedback: Feedback message explaining the result
        score: Optional numeric score (0.0 to 1.0)
        metadata: Additional metadata from the evaluation
    """

    passed: bool
    feedback: str = ""
    score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Constraint(ABC):
    """Base class for constraints that can be evaluated against model outputs.

    Constraints can be:
    - Callable functions: (response: str, context: dict) -> ConstraintResult | bool
    - LLM-based evaluators: Use a judge model to evaluate responses
    - Rule-based: Deterministic checks like regex, keywords, length

    Example:
        class LengthConstraint(Constraint):
            def __init__(self, min_length: int, max_length: int):
                self.min_length = min_length
                self.max_length = max_length

            def evaluate(self, response: str, context: dict) -> ConstraintResult:
                length = len(response)
                if self.min_length <= length <= self.max_length:
                    return ConstraintResult(passed=True, feedback="Length OK")
                return ConstraintResult(
                    passed=False,
                    feedback=f"Response length {length} not in range [{self.min_length}, {self.max_length}]"
                )
    """

    @abstractmethod
    def evaluate(self, response: str, context: dict[str, Any]) -> ConstraintResult:
        """Evaluate the constraint against a response.

        Args:
            response: The model's response text
            context: Additional context (user input, state, etc.)

        Returns:
            ConstraintResult indicating whether constraint was satisfied
        """
        ...

    @property
    def name(self) -> str:
        """Name of this constraint for logging."""
        return self.__class__.__name__


class FunctionConstraint(Constraint):
    """Constraint that wraps a callable function.

    The function can return either:
    - bool: True = passed, False = failed with default feedback
    - ConstraintResult: Full result with feedback and score
    """

    def __init__(
        self,
        func: Callable[[str, dict[str, Any]], ConstraintResult | bool],
        name: Optional[str] = None,
        default_feedback: str = "Constraint not satisfied",
    ):
        self._func = func
        self._name = name or func.__name__
        self._default_feedback = default_feedback

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, response: str, context: dict[str, Any]) -> ConstraintResult:
        result = self._func(response, context)
        if isinstance(result, bool):
            return ConstraintResult(
                passed=result,
                feedback="" if result else self._default_feedback,
            )
        return result


class LLMConstraint(Constraint):
    """Constraint that uses an LLM judge to evaluate responses.

    Similar to LLM-as-judge evaluation but returns a ConstraintResult.
    """

    def __init__(
        self,
        model: LanguageModelLike,
        prompt: str,
        name: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """Initialize LLM-based constraint.

        Args:
            model: LLM to use for evaluation
            prompt: Evaluation prompt. Should include {response} and optionally {input} placeholders.
            name: Name for logging
            threshold: Score threshold for passing (0.0-1.0)
        """
        self._model = model
        self._prompt = prompt
        self._name = name or "LLMConstraint"
        self._threshold = threshold

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, response: str, context: dict[str, Any]) -> ConstraintResult:
        user_input = context.get("input", "")

        eval_prompt = self._prompt.format(response=response, input=user_input)

        result = self._model.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation assistant. Evaluate the response and reply with:\n"
                        "PASS: <feedback> if the constraint is satisfied\n"
                        "FAIL: <feedback> if the constraint is not satisfied\n"
                        "Be concise."
                    ),
                },
                {"role": "user", "content": eval_prompt},
            ]
        )

        content = str(result.content).strip()

        if content.upper().startswith("PASS"):
            feedback = content[5:].strip(": ").strip()
            return ConstraintResult(passed=True, feedback=feedback, score=1.0)
        elif content.upper().startswith("FAIL"):
            feedback = content[5:].strip(": ").strip()
            return ConstraintResult(passed=False, feedback=feedback, score=0.0)
        else:
            # Try to interpret as pass/fail
            is_pass = any(
                word in content.lower()
                for word in ["yes", "pass", "correct", "good", "valid"]
            )
            return ConstraintResult(passed=is_pass, feedback=content)


class KeywordConstraint(Constraint):
    """Simple constraint that checks for required/banned keywords."""

    def __init__(
        self,
        required_keywords: Optional[list[str]] = None,
        banned_keywords: Optional[list[str]] = None,
        case_sensitive: bool = False,
        name: Optional[str] = None,
    ):
        self._required = required_keywords or []
        self._banned = banned_keywords or []
        self._case_sensitive = case_sensitive
        self._name = name or "KeywordConstraint"

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, response: str, context: dict[str, Any]) -> ConstraintResult:
        check_response = response if self._case_sensitive else response.lower()

        # Check banned keywords
        for keyword in self._banned:
            check_keyword = keyword if self._case_sensitive else keyword.lower()
            if check_keyword in check_response:
                return ConstraintResult(
                    passed=False,
                    feedback=f"Response contains banned keyword: '{keyword}'",
                )

        # Check required keywords
        for keyword in self._required:
            check_keyword = keyword if self._case_sensitive else keyword.lower()
            if check_keyword not in check_response:
                return ConstraintResult(
                    passed=False,
                    feedback=f"Response missing required keyword: '{keyword}'",
                )

        return ConstraintResult(
            passed=True, feedback="All keyword constraints satisfied"
        )


class LengthConstraint(Constraint):
    """Constraint that checks response length."""

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        unit: str = "chars",  # "chars", "words", "sentences"
        name: Optional[str] = None,
    ):
        self._min_length = min_length
        self._max_length = max_length
        self._unit = unit
        self._name = name or "LengthConstraint"

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, response: str, context: dict[str, Any]) -> ConstraintResult:
        if self._unit == "chars":
            length = len(response)
        elif self._unit == "words":
            length = len(response.split())
        elif self._unit == "sentences":
            length = response.count(".") + response.count("!") + response.count("?")
        else:
            length = len(response)

        if self._min_length is not None and length < self._min_length:
            return ConstraintResult(
                passed=False,
                feedback=f"Response too short: {length} {self._unit} (min: {self._min_length})",
                score=length / self._min_length if self._min_length > 0 else 0.0,
            )

        if self._max_length is not None and length > self._max_length:
            return ConstraintResult(
                passed=False,
                feedback=f"Response too long: {length} {self._unit} (max: {self._max_length})",
                score=self._max_length / length if length > 0 else 0.0,
            )

        return ConstraintResult(
            passed=True,
            feedback=f"Length OK: {length} {self._unit}",
            score=1.0,
        )


# =============================================================================
# AssertMiddleware - Hard constraint with retry (like dspy.Assert)
# =============================================================================


class AssertMiddleware(AgentMiddleware[AgentState, Context]):
    """
    Hard constraint middleware that retries until satisfied.

    Inspired by dspy.Assert - if the constraint fails, the middleware
    adds feedback to the conversation and requests a retry. If max
    retries are exhausted, it raises an error or returns a fallback.

    Args:
        constraint: The constraint to enforce
        max_retries: Maximum retry attempts before giving up
        on_failure: What to do when max retries exhausted:
            - "error": Raise ValueError (default)
            - "fallback": Return fallback_message
            - "pass": Let the response through anyway
        fallback_message: Message to return if on_failure="fallback"

    Example:
        middleware = AssertMiddleware(
            constraint=LengthConstraint(min_length=100),
            max_retries=3,
            on_failure="fallback",
            fallback_message="Unable to generate a complete response."
        )
    """

    def __init__(
        self,
        constraint: Constraint,
        max_retries: int = 3,
        on_failure: str = "error",  # "error", "fallback", "pass"
        fallback_message: str = "Unable to generate a valid response.",
    ):
        super().__init__()
        self.constraint = constraint
        self.max_retries = max_retries
        self.on_failure = on_failure
        self.fallback_message = fallback_message
        self._retry_count = 0

    def after_model(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Evaluate constraint and retry if not satisfied."""
        messages: list[BaseMessage] = state.get("messages", [])

        if not messages:
            return None

        ai_message: AIMessage | None = last_ai_message(messages)
        human_message: HumanMessage | None = last_human_message(messages)

        if not ai_message:
            return None

        response = str(ai_message.content)
        user_input = str(human_message.content) if human_message else ""

        context = {
            "input": user_input,
            "messages": messages,
            "runtime": runtime,
        }

        logger.trace(
            "Evaluating Assert constraint", constraint_name=self.constraint.name
        )

        result = self.constraint.evaluate(response, context)

        if result.passed:
            logger.trace(
                "Assert constraint passed", constraint_name=self.constraint.name
            )
            self._retry_count = 0
            return None

        # Constraint failed
        self._retry_count += 1
        logger.warning(
            "Assert constraint failed",
            constraint_name=self.constraint.name,
            attempt=self._retry_count,
            max_retries=self.max_retries,
            feedback=result.feedback,
        )

        if self._retry_count >= self.max_retries:
            self._retry_count = 0

            if self.on_failure == "error":
                raise ValueError(
                    f"Assert constraint '{self.constraint.name}' failed after "
                    f"{self.max_retries} retries: {result.feedback}"
                )
            elif self.on_failure == "fallback":
                ai_message.content = self.fallback_message
                return None
            else:  # "pass"
                logger.warning(
                    "Assert constraint failed but passing through",
                    constraint_name=self.constraint.name,
                )
                return None

        # Add feedback and retry
        retry_prompt = (
            f"Your previous response did not meet the requirements:\n"
            f"{result.feedback}\n\n"
            f"Please try again with the original request:\n{user_input}"
        )
        return {"messages": [HumanMessage(content=retry_prompt)]}


# =============================================================================
# SuggestMiddleware - Soft constraint with feedback (like dspy.Suggest)
# =============================================================================


class SuggestMiddleware(AgentMiddleware[AgentState, Context]):
    """
    Soft constraint middleware that provides feedback without blocking.

    Inspired by dspy.Suggest - evaluates the constraint and logs feedback
    but does not retry or block the response. The feedback is captured
    in metadata for observability but the response passes through.

    Optionally, can request one improvement attempt if constraint fails.

    Args:
        constraint: The constraint to evaluate
        allow_one_retry: If True, request one improvement attempt on failure
        log_level: Log level for feedback ("warning", "info", "debug")

    Example:
        middleware = SuggestMiddleware(
            constraint=LLMConstraint(
                model=ChatDatabricks(...),
                prompt="Check if response is professional: {response}"
            ),
            allow_one_retry=True,
        )
    """

    def __init__(
        self,
        constraint: Constraint,
        allow_one_retry: bool = False,
        log_level: str = "warning",
    ):
        super().__init__()
        self.constraint = constraint
        self.allow_one_retry = allow_one_retry
        self.log_level = log_level
        self._has_retried = False

    def after_model(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Evaluate constraint and log feedback."""
        messages: list[BaseMessage] = state.get("messages", [])

        if not messages:
            return None

        ai_message: AIMessage | None = last_ai_message(messages)
        human_message: HumanMessage | None = last_human_message(messages)

        if not ai_message:
            return None

        response = str(ai_message.content)
        user_input = str(human_message.content) if human_message else ""

        context = {
            "input": user_input,
            "messages": messages,
            "runtime": runtime,
        }

        logger.trace(
            "Evaluating Suggest constraint", constraint_name=self.constraint.name
        )

        result = self.constraint.evaluate(response, context)

        if result.passed:
            logger.trace(
                "Suggest constraint passed", constraint_name=self.constraint.name
            )
            self._has_retried = False
            return None

        # Log feedback based on configured level
        if self.log_level == "warning":
            logger.warning(
                "Suggest constraint feedback",
                constraint_name=self.constraint.name,
                feedback=result.feedback,
            )
        elif self.log_level == "info":
            logger.info(
                "Suggest constraint feedback",
                constraint_name=self.constraint.name,
                feedback=result.feedback,
            )
        else:
            logger.debug(
                "Suggest constraint feedback",
                constraint_name=self.constraint.name,
                feedback=result.feedback,
            )

        # Optionally request one improvement
        if self.allow_one_retry and not self._has_retried:
            self._has_retried = True
            retry_prompt = (
                f"Consider this feedback for your response:\n"
                f"{result.feedback}\n\n"
                f"Original request: {user_input}\n"
                f"Please provide an improved response."
            )
            return {"messages": [HumanMessage(content=retry_prompt)]}

        # Pass through without modification
        self._has_retried = False
        return None


# =============================================================================
# RefineMiddleware - Iterative improvement (like dspy.Refine)
# =============================================================================


class RefineMiddleware(AgentMiddleware[AgentState, Context]):
    """
    Iterative refinement middleware that improves responses.

    Inspired by dspy.Refine - runs the response through multiple iterations,
    using a reward function to score each attempt. Selects the best response
    or stops early if a threshold is reached.

    Since middleware runs in the agent loop, this works by:
    1. Scoring the current response
    2. If below threshold and iterations remain, requesting improvement
    3. Tracking the best response across iterations
    4. Returning the best response when done

    Args:
        reward_fn: Function that scores a response (returns 0.0 to 1.0)
        threshold: Score threshold to stop early (default 0.8)
        max_iterations: Maximum improvement iterations (default 3)
        select_best: If True, track and return best response; else use last

    Example:
        def score_response(response: str, context: dict) -> float:
            # Score based on helpfulness, completeness, etc.
            return 0.85

        middleware = RefineMiddleware(
            reward_fn=score_response,
            threshold=0.9,
            max_iterations=3,
        )
    """

    def __init__(
        self,
        reward_fn: Callable[[str, dict[str, Any]], float],
        threshold: float = 0.8,
        max_iterations: int = 3,
        select_best: bool = True,
    ):
        super().__init__()
        self.reward_fn = reward_fn
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.select_best = select_best
        self._iteration = 0
        self._best_score = 0.0
        self._best_response: Optional[str] = None

    def after_model(
        self, state: AgentState, runtime: Runtime[Context]
    ) -> dict[str, Any] | None:
        """Score response and request improvement if needed."""
        messages: list[BaseMessage] = state.get("messages", [])

        if not messages:
            return None

        ai_message: AIMessage | None = last_ai_message(messages)
        human_message: HumanMessage | None = last_human_message(messages)

        if not ai_message:
            return None

        response = str(ai_message.content)
        user_input = str(human_message.content) if human_message else ""

        context = {
            "input": user_input,
            "messages": messages,
            "runtime": runtime,
            "iteration": self._iteration,
        }

        score: float = self.reward_fn(response, context)
        self._iteration += 1

        logger.debug(
            "Refine iteration",
            iteration=self._iteration,
            max_iterations=self.max_iterations,
            score=f"{score:.3f}",
            threshold=self.threshold,
        )

        # Track best response
        if self.select_best and score > self._best_score:
            self._best_score = score
            self._best_response = response

        # Check if we should stop
        if score >= self.threshold:
            logger.debug(
                "Refine threshold reached",
                score=f"{score:.3f}",
                threshold=self.threshold,
            )
            self._reset()
            return None

        if self._iteration >= self.max_iterations:
            logger.debug(
                "Refine max iterations reached", best_score=f"{self._best_score:.3f}"
            )
            # Use best response if tracking
            if self.select_best and self._best_response:
                ai_message.content = self._best_response
            self._reset()
            return None

        # Request improvement
        feedback = f"Current response scored {score:.2f}/{self.threshold:.2f}."
        if score < 0.5:
            feedback += " The response needs significant improvement."
        elif score < self.threshold:
            feedback += " The response is good but could be better."

        retry_prompt = f"{feedback}\n\nPlease improve your response to:\n{user_input}"
        return {"messages": [HumanMessage(content=retry_prompt)]}

    def _reset(self) -> None:
        """Reset iteration state for next invocation."""
        self._iteration = 0
        self._best_score = 0.0
        self._best_response = None


# =============================================================================
# Factory Functions
# =============================================================================


def create_assert_middleware(
    constraint: Constraint | Callable[[str, dict[str, Any]], ConstraintResult | bool],
    max_retries: int = 3,
    on_failure: str = "error",
    fallback_message: str = "Unable to generate a valid response.",
    name: Optional[str] = None,
) -> AssertMiddleware:
    """
    Create an AssertMiddleware (hard constraint with retry).

    Like dspy.Assert - enforces a constraint and retries if not satisfied.

    Args:
        constraint: Constraint object or callable function
        max_retries: Maximum retry attempts
        on_failure: "error", "fallback", or "pass"
        fallback_message: Message if on_failure="fallback"
        name: Name for function constraints

    Returns:
        List containing AssertMiddleware configured with the constraint

    Example:
        # Using a Constraint class
        middleware = create_assert_middleware(
            constraint=LengthConstraint(min_length=100),
            max_retries=3,
        )

        # Using a function
        def has_sources(response: str, ctx: dict) -> bool:
            return "[source]" in response.lower() or "reference" in response.lower()

        middleware = create_assert_middleware(
            constraint=has_sources,
            max_retries=2,
            on_failure="fallback",
            fallback_message="I couldn't find relevant sources.",
        )
    """
    if callable(constraint) and not isinstance(constraint, Constraint):
        constraint = FunctionConstraint(constraint, name=name)

    return AssertMiddleware(
        constraint=constraint,
        max_retries=max_retries,
        on_failure=on_failure,
        fallback_message=fallback_message,
    )


def create_suggest_middleware(
    constraint: Constraint | Callable[[str, dict[str, Any]], ConstraintResult | bool],
    allow_one_retry: bool = False,
    log_level: str = "warning",
    name: Optional[str] = None,
) -> SuggestMiddleware:
    """
    Create a SuggestMiddleware (soft constraint with feedback).

    Like dspy.Suggest - evaluates constraint and logs feedback without blocking.

    Args:
        constraint: Constraint object or callable function
        allow_one_retry: Request one improvement attempt on failure
        log_level: "warning", "info", or "debug"
        name: Name for function constraints

    Returns:
        List containing SuggestMiddleware configured with the constraint

    Example:
        def is_professional(response: str, ctx: dict) -> ConstraintResult:
            informal = ["lol", "omg", "btw", "gonna"]
            found = [w for w in informal if w in response.lower()]
            if found:
                return ConstraintResult(
                    passed=False,
                    feedback=f"Response contains informal language: {found}"
                )
            return ConstraintResult(passed=True)

        middleware = create_suggest_middleware(
            constraint=is_professional,
            allow_one_retry=True,
        )
    """
    if callable(constraint) and not isinstance(constraint, Constraint):
        constraint = FunctionConstraint(constraint, name=name)

    return SuggestMiddleware(
        constraint=constraint,
        allow_one_retry=allow_one_retry,
        log_level=log_level,
    )


def create_refine_middleware(
    reward_fn: Callable[[str, dict[str, Any]], float],
    threshold: float = 0.8,
    max_iterations: int = 3,
    select_best: bool = True,
) -> RefineMiddleware:
    """
    Create a RefineMiddleware (iterative improvement).

    Like dspy.Refine - iteratively improves responses using a reward function.

    Args:
        reward_fn: Function that scores a response (0.0 to 1.0)
        threshold: Score threshold to stop early
        max_iterations: Maximum improvement iterations
        select_best: Track and return best response across iterations

    Returns:
        List containing RefineMiddleware configured with the reward function

    Example:
        def evaluate_completeness(response: str, ctx: dict) -> float:
            # Check for expected sections
            sections = ["introduction", "details", "conclusion"]
            found = sum(1 for s in sections if s in response.lower())
            return found / len(sections)

        middleware = create_refine_middleware(
            reward_fn=evaluate_completeness,
            threshold=1.0,
            max_iterations=3,
        )
    """
    return RefineMiddleware(
        reward_fn=reward_fn,
        threshold=threshold,
        max_iterations=max_iterations,
        select_best=select_best,
    )
