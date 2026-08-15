"""Internal tracing utilities for dao-ai.

These helpers exist to keep MLflow trace context intact across the worker
boundaries dao-ai uses (thread pools, background event loops, ``asyncio.to_thread``),
and to keep credentials out of the spans that result.
"""

from dao_ai._tracing.context import (
    in_caller_context,
    submit_in_context,
    to_thread_in_context,
)
from dao_ai._tracing.redaction import (
    REDACTED,
    install_trace_redaction,
    redact_credentials,
)

__all__ = [
    "in_caller_context",
    "submit_in_context",
    "to_thread_in_context",
    "REDACTED",
    "install_trace_redaction",
    "redact_credentials",
]
