"""
Model call limit middleware for DAO AI agents.

Limits the number of model (LLM) calls to prevent infinite loops or excessive costs.

Example:
    from dao_ai.middleware import create_model_call_limit_middleware

    # Limit model calls per run and thread
    middleware = create_model_call_limit_middleware(
        thread_limit=10,
        run_limit=5,
    )
"""

from __future__ import annotations

from typing import Literal

from langchain.agents.middleware import ModelCallLimitMiddleware
from loguru import logger

__all__ = [
    "ModelCallLimitMiddleware",
    "create_model_call_limit_middleware",
]


def create_model_call_limit_middleware(
    thread_limit: int | None = None,
    run_limit: int | None = None,
    exit_behavior: Literal["error", "end"] = "end",
) -> ModelCallLimitMiddleware:
    """
    Create a ModelCallLimitMiddleware to limit LLM API calls.

    Prevents runaway agents from making too many API calls and helps
    enforce cost controls on production deployments.

    Args:
        thread_limit: Max model calls per thread (conversation).
            Requires checkpointer. None = no limit.
        run_limit: Max model calls per run (single invocation).
            None = no limit.
        exit_behavior: What to do when limit hit:
            - "end": Stop execution gracefully (default)
            - "error": Raise ModelCallLimitExceededError immediately

    Returns:
        List containing ModelCallLimitMiddleware instance

    Raises:
        ValueError: If no limits specified

    Example:
        # Limit to 5 model calls per run, 10 per thread
        limiter = create_model_call_limit_middleware(
            run_limit=5,
            thread_limit=10,
            exit_behavior="end",
        )
    """
    if thread_limit is None and run_limit is None:
        raise ValueError("At least one of thread_limit or run_limit must be specified.")

    logger.debug(
        "Creating model call limit middleware",
        thread_limit=thread_limit,
        run_limit=run_limit,
        exit_behavior=exit_behavior,
    )

    return ModelCallLimitMiddleware(
        thread_limit=thread_limit,
        run_limit=run_limit,
        exit_behavior=exit_behavior,
    )
