"""MLflow trace → waterfall JSON for the dao-ai Console Timeline view.

The backend owns trace retrieval (it already sets ``mlflow.set_tracking_uri``
and redacts bearer tokens from span payloads at write time), so the frontend
fetches a ready-to-render span tree from ``GET /v1/traces/{trace_id}`` rather
than talking to the Databricks trace APIs directly.

``build_trace_tree`` is a pure function over the flat span list MLflow returns
(duck-typed for testability); ``get_trace_tree`` waits for propagation, fetches,
and builds.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from loguru import logger

# Span-payload keys never surfaced to the UI (defence in depth on top of the
# write-time bearer redaction). Matched case-insensitively.
_REDACT_KEYS: frozenset[str] = frozenset(
    {"authorization", "headers", "x-forwarded-access-token", "x_forwarded_access_token"}
)


def _redact(value: Any) -> Any:
    """Drop sensitive keys from a span input/output/attribute mapping."""
    if isinstance(value, dict):
        return {
            k: _redact(v)
            for k, v in value.items()
            if str(k).lower() not in _REDACT_KEYS
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _status_str(status: Any) -> str:
    """Coerce an MLflow span status (enum, object, or string) to a label."""
    if status is None:
        return "UNSET"
    code = getattr(status, "status_code", status)
    return getattr(code, "name", str(code))


def _ns_to_ms(delta_ns: float) -> float:
    return round(delta_ns / 1_000_000, 1)


def _event_dict(event: Any, root_start_ns: int) -> dict[str, Any]:
    timestamp_ns = getattr(event, "timestamp", None)
    return {
        "name": getattr(event, "name", ""),
        "timestamp_ms": _ns_to_ms(timestamp_ns - root_start_ns)
        if isinstance(timestamp_ns, (int, float))
        else None,
        "attributes": _redact(getattr(event, "attributes", {}) or {}),
    }


def _node(span: Any, root_start_ns: int) -> dict[str, Any]:
    start_ns: int = getattr(span, "start_time_ns", root_start_ns) or root_start_ns
    end_ns: int = getattr(span, "end_time_ns", start_ns) or start_ns
    return {
        "span_id": getattr(span, "span_id", None),
        "parent_id": getattr(span, "parent_id", None),
        "name": getattr(span, "name", ""),
        "span_type": getattr(span, "span_type", None),
        "status": _status_str(getattr(span, "status", None)),
        "start_offset_ms": _ns_to_ms(start_ns - root_start_ns),
        "duration_ms": _ns_to_ms(end_ns - start_ns),
        "inputs": _redact(getattr(span, "inputs", None) or {}),
        "outputs": _redact(getattr(span, "outputs", None) or {}),
        "attributes": _redact(getattr(span, "attributes", None) or {}),
        "events": [
            _event_dict(e, root_start_ns) for e in (getattr(span, "events", None) or [])
        ],
        "children": [],
    }


def build_trace_tree(spans: list[Any], *, trace_id: str) -> dict[str, Any]:
    """Build the nested waterfall payload from MLflow's flat span list.

    ``start_offset_ms`` on every node is relative to the earliest span's start
    so the UI can draw bars without re-deriving the baseline. Spans whose
    ``parent_id`` is absent (or points outside the trace) are treated as
    top-level. Sensitive keys are stripped from all payloads.
    """
    if not spans:
        return {
            "trace_id": trace_id,
            "root_span_id": None,
            "duration_ms": 0.0,
            "spans": [],
        }

    # Baseline off real (non-zero) starts only — a span missing start_time_ns
    # must not drag the baseline to epoch and push every bar off-scale.
    starts: list[int] = [
        s for s in (getattr(sp, "start_time_ns", 0) or 0 for sp in spans) if s
    ]
    root_start_ns: int = min(starts) if starts else 0
    max_end_ns: int = max((getattr(s, "end_time_ns", 0) or 0) for s in spans)

    nodes: dict[str, dict[str, Any]] = {}
    for span in spans:
        node = _node(span, root_start_ns)
        if node["span_id"] is not None:
            nodes[node["span_id"]] = node

    roots: list[dict[str, Any]] = []
    for node in nodes.values():
        parent = nodes.get(node["parent_id"]) if node["parent_id"] else None
        # A span that parents itself (or is missing a valid parent) is a root —
        # never its own child, which would make the tree cyclic and recurse.
        if parent is None or parent is node:
            roots.append(node)
        else:
            parent["children"].append(node)

    # Deterministic order: earliest-starting first, at every level.
    def _sort(children: list[dict[str, Any]]) -> None:
        children.sort(key=lambda n: n["start_offset_ms"])
        for child in children:
            _sort(child["children"])

    _sort(roots)

    root_span_id: str | None = roots[0]["span_id"] if roots else None
    return {
        "trace_id": trace_id,
        "root_span_id": root_span_id,
        "duration_ms": _ns_to_ms(max_end_ns - root_start_ns),
        "spans": roots,
    }


def _workspace_host() -> Optional[str]:
    """The Databricks workspace host for building UI deep links.

    The Console runs on ``*.databricksapps.com``, but trace links must point at
    the workspace UI, so resolve the workspace host from the environment / SDK
    config (the browser can't derive it).
    """
    host = os.environ.get("DATABRICKS_HOST")
    if not host:
        try:
            from databricks.sdk import WorkspaceClient

            host = WorkspaceClient().config.host
        except Exception:  # noqa: BLE001 — no workspace host available
            return None
    if not host:
        return None
    host = host.rstrip("/")
    return host if host.startswith("http") else f"https://{host}"


def _active_experiment_id() -> Optional[str]:
    """Best-effort active MLflow experiment id (for control-plane ``tr-`` ids)."""
    exp = os.environ.get("MLFLOW_EXPERIMENT_ID")
    if exp:
        return exp
    try:
        from mlflow.tracking.fluent import _get_experiment_id

        return _get_experiment_id()
    except Exception:  # noqa: BLE001
        return None


def build_trace_ui_url(trace_id: str) -> Optional[str]:
    """Build a Databricks workspace deep link to view ``trace_id`` in the UI.

    Handles both id forms:

    - ``trace:/<catalog>.<schema>.<experiment_id>/<hex>`` (UC trace_location) —
      the experiment id is embedded and the UI trace id is ``tr-<hex>``.
    - ``tr-<hex>`` (control plane) — the experiment id is the active experiment.

    Returns ``None`` when the host or experiment can't be resolved, so the
    Console simply omits the link. Format:
    ``https://<host>/ml/experiments/<experiment_id>/traces/<trace_id>``.
    """
    if not trace_id:
        return None
    host = _workspace_host()
    if not host:
        return None

    if trace_id.startswith("trace:/"):
        rest = trace_id[len("trace:/") :]
        location, _, hex_id = rest.partition("/")
        experiment_id = location.split(".")[-1] if "." in location else None
        if not hex_id:
            return None
        ui_trace_id = hex_id if hex_id.startswith("tr-") else f"tr-{hex_id}"
    else:
        experiment_id = _active_experiment_id()
        ui_trace_id = trace_id

    if not experiment_id or not ui_trace_id:
        return None
    return f"{host}/ml/experiments/{experiment_id}/traces/{ui_trace_id}"


def wait_for_trace(trace_id: str, *, timeout_seconds: float = 5.0) -> bool:
    """Poll until ``trace_id`` is queryable, or the timeout elapses.

    Returns ``True`` when the trace became queryable, ``False`` otherwise. The
    request path never raises — a not-yet-propagated, permission-denied,
    corrupted, or otherwise unreadable trace all degrade to ``False`` so the
    Console shows an empty-Timeline note instead of a 500. Only a transient
    "not found" is retried (the trace may still be propagating from the UC
    trace_location); every other error is terminal and returns immediately.
    """
    import time

    import mlflow

    deadline: float = time.monotonic() + timeout_seconds
    delay: float = 0.25
    while True:
        try:
            trace = mlflow.get_trace(trace_id)
        except Exception as exc:  # noqa: BLE001 — degrade on any read failure
            # ``mlflow.get_trace`` swallows ``MlflowException`` (incl. NOT_FOUND)
            # and returns None; anything that *does* propagate here is terminal —
            # e.g. ``databricks.sdk`` ``PermissionDenied`` (the app identity can't
            # read the trace warehouse). Retrying won't help, so degrade.
            logger.warning(
                "Trace not readable; Timeline will show an empty note",
                trace_id=trace_id,
                error=str(exc),
            )
            return False
        if trace is not None:
            return True
        # None => not found yet. A trace_location (UC) trace can lag, so poll
        # until the deadline before giving up.
        if time.monotonic() >= deadline:
            return False
        time.sleep(delay)
        delay = min(delay * 2, 1.0)


def get_trace_tree(
    trace_id: str, *, timeout_seconds: float = 5.0
) -> dict[str, Any] | None:
    """Fetch a trace by id and return its waterfall tree, or ``None`` if the
    trace is not queryable within ``timeout_seconds``. Never raises — any
    retrieval error degrades to ``None`` (the route then 404s gracefully)."""
    import mlflow

    if not wait_for_trace(trace_id, timeout_seconds=timeout_seconds):
        logger.debug("Trace not queryable", trace_id=trace_id)
        return None
    try:
        trace = mlflow.get_trace(trace_id)
        spans = list(getattr(getattr(trace, "data", None), "spans", None) or [])
        return build_trace_tree(spans, trace_id=trace_id)
    except Exception as exc:  # noqa: BLE001 — a race after wait, or a read error
        logger.warning("Failed to build trace tree", trace_id=trace_id, error=str(exc))
        return None
