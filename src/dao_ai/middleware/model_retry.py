"""
Model retry middleware for DAO AI agents.

Automatically retries failed model (LLM) calls with configurable exponential backoff.

Example:
    from dao_ai.middleware import create_model_retry_middleware

    # Retry failed model calls with exponential backoff
    middleware = create_model_retry_middleware(
        max_retries=3,
        backoff_factor=2.0,
        initial_delay=1.0,
    )
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from langchain.agents.middleware import ModelRetryMiddleware
from loguru import logger

__all__ = [
    "ModelRetryMiddleware",
    "create_model_retry_middleware",
]


def create_model_retry_middleware(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float | None = None,
    jitter: bool = False,
    retry_on: tuple[type[Exception], ...] | Callable[[Exception], bool] | None = None,
    on_failure: Literal["continue", "error"] | Callable[[Exception], str] = "continue",
) -> ModelRetryMiddleware:
    """
    Create a ModelRetryMiddleware for automatic model call retries.

    Handles transient failures in model API calls with exponential backoff.
    Useful for handling rate limits, network issues, and temporary outages.

    Args:
        max_retries: Max retry attempts after initial call. Default 3.
        backoff_factor: Multiplier for exponential backoff. Default 2.0.
            Delay = initial_delay * (backoff_factor ** retry_number)
            Set to 0.0 for constant delay.
        initial_delay: Initial delay in seconds before first retry. Default 1.0.
        max_delay: Max delay in seconds (caps exponential growth). None = no cap.
        jitter: Add ±25% random jitter to avoid thundering herd. Default False.
        retry_on: When to retry:
            - None: Retry on all errors (default)
            - tuple of Exception types: Retry only on these
            - callable: Function(exception) -> bool for custom logic
        on_failure: Behavior when all retries exhausted:
            - "continue": Return AIMessage with error, let agent continue (default)
            - "error": Re-raise exception, stop execution
            - callable: Function(exception) -> str for custom error message

    Returns:
        List containing ModelRetryMiddleware instance

    Example:
        # Basic retry with defaults
        retry = create_model_retry_middleware()

        # Custom backoff for rate limits
        retry = create_model_retry_middleware(
            max_retries=5,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=True,
        )

        # Retry only on specific exceptions, fail hard
        retry = create_model_retry_middleware(
            max_retries=3,
            retry_on=(RateLimitError, TimeoutError),
            on_failure="error",
        )

        # Custom retry logic
        def should_retry(error: Exception) -> bool:
            return "rate_limit" in str(error).lower()

        retry = create_model_retry_middleware(
            max_retries=5,
            retry_on=should_retry,
        )
    """
    logger.debug(
        "Creating model retry middleware",
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        initial_delay=initial_delay,
        max_delay=max_delay,
        jitter=jitter,
        on_failure=on_failure if isinstance(on_failure, str) else "custom",
    )

    # Build kwargs
    kwargs: dict[str, Any] = {
        "max_retries": max_retries,
        "backoff_factor": backoff_factor,
        "initial_delay": initial_delay,
        "on_failure": on_failure,
    }

    if max_delay is not None:
        kwargs["max_delay"] = max_delay

    if jitter:
        kwargs["jitter"] = jitter

    if retry_on is not None:
        kwargs["retry_on"] = retry_on

    return ModelRetryMiddleware(**kwargs)
