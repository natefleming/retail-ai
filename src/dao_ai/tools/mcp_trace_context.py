"""W3C trace-context propagation over MCP ``_meta`` (SEP-414).

The MCP specification standardizes carrying OpenTelemetry trace context in the
JSON-RPC ``_meta`` block of a request via the ``traceparent`` / ``tracestate``
/ ``baggage`` keys, so a distributed trace flows unbroken from an MCP client
through to the server it calls. This module is the two-sided implementation:

* :func:`build_trace_context_meta` — client side. Reads dao-ai's active
  MLflow span and formats a W3C ``traceparent`` plus a ``baggage`` entry
  carrying the MLflow trace id. Injected into ``session.call_tool(..., meta=)``.
* :func:`extract_trace_context_meta` — server side. Reads those keys off the
  inbound request ``_meta`` and stamps them onto the active MLflow span so the
  server-side trace row correlates with the caller.

Why not the OpenTelemetry propagator? MLflow 3 runs on OpenTelemetry but does
not populate OTel's own context var — ``opentelemetry.trace.get_current_span()``
returns a non-recording span while an MLflow span is active. MLflow's
``LiveSpan`` however exposes ``trace_id`` (``tr-<32-hex>`` or a
``trace:/…/<32-hex>`` UC URI) and ``span_id`` (``<16-hex>``) whose hex payloads
are the OTel-native 128-bit trace id and 64-bit span id at the exact widths a
W3C ``traceparent`` requires — and are the same hex values that land in the
Databricks UC ``_otel_spans.trace_id`` column. We format the header from those
directly. When the hex is absent or the wrong width, we degrade to a
``baggage``-only envelope (or ``{}`` when no span is active) rather than emit a
malformed header.
"""

from __future__ import annotations

import re
from typing import Any

import mlflow
from loguru import logger

# W3C ``traceparent``: version "00", 32-hex trace id, 16-hex span id, 2-hex
# flags. Flags "01" = sampled (dao-ai always exports the traces it opens).
_TRACE_ID_HEX_LEN = 32
_SPAN_ID_HEX_LEN = 16
_TRACEPARENT_FLAGS = "01"
_TRACEPARENT_VERSION = "00"

_HEX_RE = re.compile(r"[0-9a-f]+")


def _hex_suffix(raw: str | None, width: int) -> str | None:
    """Return the trailing ``width``-char lowercase-hex token of ``raw``.

    MLflow surfaces ids as ``tr-<hex>``, ``trace:/<cat>.<schema>.<prefix>/<hex>``
    (UC), or a bare hex string; the OTel-native hex is always the final hex run.
    Returns None when no hex run of exactly ``width`` chars is present.
    """
    if not raw:
        return None
    matches = _HEX_RE.findall(raw.lower())
    if not matches:
        return None
    candidate = matches[-1]
    return candidate if len(candidate) == width else None


def build_trace_context_meta() -> dict[str, Any]:
    """Build the ``_meta`` trace-context keys for the active MLflow span.

    Returns a dict with ``traceparent`` (and ``baggage`` carrying the MLflow
    trace id) when a well-formed span context is available, a ``baggage``-only
    dict when the MLflow trace id is known but not W3C-formattable, or ``{}``
    when no span is active. Never raises — trace propagation must not break a
    tool call.
    """
    try:
        span = mlflow.get_current_active_span()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"mcp trace context: get_current_active_span failed: {exc}")
        return {}
    if span is None:
        return {}

    trace_id: str | None = getattr(span, "trace_id", None)
    span_id: str | None = getattr(span, "span_id", None)

    meta: dict[str, Any] = {}
    if trace_id:
        # Carry the full MLflow trace id for dao-ai-native correlation
        # regardless of whether a W3C traceparent can be formed.
        meta["baggage"] = f"mlflow.trace_id={trace_id}"

    trace_hex = _hex_suffix(trace_id, _TRACE_ID_HEX_LEN)
    span_hex = _hex_suffix(span_id, _SPAN_ID_HEX_LEN)
    if trace_hex and span_hex:
        meta["traceparent"] = (
            f"{_TRACEPARENT_VERSION}-{trace_hex}-{span_hex}-{_TRACEPARENT_FLAGS}"
        )
    return meta


def merge_trace_context_meta(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    """Non-destructively merge trace-context keys into a caller ``_meta`` dict.

    Caller-supplied keys (e.g. ``conversation_id``, ``progressToken``) always
    win — trace-context keys are only added when absent. Returns the original
    object (possibly ``None``) when there is no trace context to add, so the
    classic ``meta=None`` call shape is preserved.
    """
    trace_meta = build_trace_context_meta()
    if not trace_meta:
        return meta
    merged: dict[str, Any] = dict(meta or {})
    for key, value in trace_meta.items():
        merged.setdefault(key, value)
    return merged


def extract_trace_context_meta(meta: Any) -> None:
    """Stamp inbound W3C trace-context onto the active MLflow span.

    ``meta`` is the request ``_meta`` (an ``mcp.types.RequestParams.Meta`` or a
    plain mapping) surfaced by FastMCP on the server. Reads ``traceparent`` /
    ``tracestate`` / ``baggage`` and records them as span attributes so the
    server-side trace row correlates with the caller's trace. Additive and
    never raises — degrades to a no-op when meta is absent or malformed.
    """
    if meta is None:
        return
    try:
        span = mlflow.get_current_active_span()
    except Exception:  # pragma: no cover - defensive
        span = None
    if span is None:
        return

    def _get(key: str) -> Any:
        if isinstance(meta, dict):
            return meta.get(key)
        return getattr(meta, key, None)

    try:
        for key in ("traceparent", "tracestate", "baggage"):
            value = _get(key)
            if value:
                span.set_attribute(f"mcp.trace_context.{key}", str(value))
    except Exception as exc:
        logger.debug(f"mcp trace context: failed to stamp inbound context: {exc}")
