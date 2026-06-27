"""Internal tracing utilities for dao-ai.

These helpers exist to keep MLflow trace context intact across the worker
boundaries dao-ai uses (thread pools, background event loops,
``asyncio.to_thread``), and to mark request-entry boundaries where the
inherited OTel context must be cleared so the dao-ai-owned span is
exported as a true root span.
"""

from dao_ai._tracing.context import (
    detached_otel_context,
    in_caller_context,
    root_trace,
    submit_in_context,
    to_thread_in_context,
)

__all__ = [
    "detached_otel_context",
    "in_caller_context",
    "root_trace",
    "submit_in_context",
    "to_thread_in_context",
]
