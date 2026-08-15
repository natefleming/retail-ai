"""Keep credentials out of MLflow spans.

``dao_ai.apps.handlers`` injects the whole inbound header map onto the request
so tools can act as the calling user, and that map carries a live bearer in
``x-forwarded-access-token``. The request is then passed to
``LanggraphResponsesAgent.apredict``, which is ``@mlflow.trace``-decorated, so
the bearer is serialized as that span's ``inputs`` attribute — readable by
anyone with access to the experiment, and durable long after the token itself
expires.

Filtering the outbound ``custom_outputs`` and error text (see
``dao_ai.state.context_configurable_fields``) does nothing for that sink,
because nothing there is on the outbound path. MLflow's own answer is a span
processor, which runs on every span just before export, so one registration
covers the agent span, tool spans, and anything autolog produces.

The processor mutates the span payload rather than dropping the span: a trace
whose inputs are ``{"request": {...}}`` with one value replaced by
``<redacted>`` is still the trace you need for debugging.
"""

from __future__ import annotations

from typing import Any

import mlflow
from loguru import logger
from mlflow.entities import LiveSpan

from dao_ai.diagnostics import is_secret_field_name

__all__ = [
    "REDACTED",
    "install_trace_redaction",
    "redact_credentials",
]

REDACTED: str = "<redacted>"

# Deep enough for ``inputs -> request -> custom_inputs -> configurable ->
# headers -> value`` with room to spare, shallow enough that a self-referential
# payload cannot spin here.
_MAX_DEPTH: int = 12


def redact_credentials(payload: Any, _depth: int = 0) -> Any:
    """Return ``payload`` with credential-shaped entries replaced by ``REDACTED``.

    Walks dicts and sequences, deciding on *key names* via
    :func:`~dao_ai.diagnostics.is_secret_field_name`. Values are never inspected:
    a JWT-shaped-string heuristic flags far more than it catches, and a span's
    inputs are full of user prose.

    Unlike :func:`~dao_ai.diagnostics.redact_value`, no prefix or length hint is
    preserved. That hint exists so an operator can eyeball whether an env var
    round-tripped; a bearer sitting in a trace has no such use, and a trace is
    read by a wider audience than a boot-time log.
    """
    if _depth >= _MAX_DEPTH:
        return payload

    if isinstance(payload, dict):
        return {
            key: REDACTED
            if isinstance(key, str) and is_secret_field_name(key)
            else redact_credentials(value, _depth + 1)
            for key, value in payload.items()
        }

    if isinstance(payload, (list, tuple)):
        redacted = [redact_credentials(item, _depth + 1) for item in payload]
        return tuple(redacted) if isinstance(payload, tuple) else redacted

    return payload


def _redact_span(span: LiveSpan) -> None:
    """Span processor: strip credential-shaped entries from a span's payloads."""
    inputs: Any = span.inputs
    if inputs is not None:
        span.set_inputs(redact_credentials(inputs))

    outputs: Any = span.outputs
    if outputs is not None:
        span.set_outputs(redact_credentials(outputs))


def install_trace_redaction() -> None:
    """Register :func:`_redact_span` as an MLflow span processor.

    Call once per process from an entry point, after
    ``mlflow.langchain.autolog()``. Idempotent in effect but not in cost —
    ``mlflow.tracing.configure`` replaces the processor list rather than
    appending, so a second call re-registers the same single processor.
    """
    mlflow.tracing.configure(span_processors=[_redact_span])
    logger.debug("Installed MLflow span credential redaction")
