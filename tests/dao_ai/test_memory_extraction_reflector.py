"""Unit tests for ``_ContextAwareReflector.invoke``.

The reflector wraps langmem's ``MemoryStoreManager`` so background
memory-extraction calls run under the caller's captured
``contextvars.Context``. In addition it opts the wrapped chain out of
MLflow autolog callbacks — memory extraction runs on nested worker
threads where MLflow's per-instance tracer state (``_run_span_mapping``)
cannot follow, producing noisy ``Span for run_id X not found`` warnings
on every request. Passing ``callbacks=[]`` in the invoke config disables
those callbacks for this chain only. This test suite pins that
behavior.
"""

from __future__ import annotations

import contextvars
from typing import Any

import pytest

from dao_ai.memory.extraction import _ContextAwareReflector


class _RecordingInner:
    """Fake ``MemoryStoreManager`` that records how ``invoke`` was called."""

    def __init__(self) -> None:
        self.namespace: tuple[str, ...] = ("memory", "{user_id}")
        self.calls: list[dict[str, Any]] = []

    def invoke(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
        # Normalize the positional/keyword config into a single view.
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        self.calls.append(
            {
                "payload": payload,
                "config": config,
                "args": args,
                "kwargs": {k: v for k, v in kwargs.items() if k != "config"},
            }
        )
        return {"ok": True}


@pytest.mark.unit
def test_invoke_injects_empty_callbacks_when_no_config() -> None:
    inner = _RecordingInner()
    r = _ContextAwareReflector(inner)
    payload = {"messages": []}
    r.invoke(payload)
    assert len(inner.calls) == 1
    seen = inner.calls[0]["config"]
    assert seen is not None
    assert seen.get("callbacks") == []


@pytest.mark.unit
def test_invoke_overrides_caller_callbacks() -> None:
    """Autolog opt-out is idempotent — the reflector always wins."""
    inner = _RecordingInner()
    r = _ContextAwareReflector(inner)

    class _OtherHandler:
        pass

    caller_config = {
        "callbacks": [_OtherHandler()],
        "configurable": {"user_id": "alice"},
    }
    r.invoke({"messages": []}, config=caller_config)
    seen = inner.calls[0]["config"]
    assert seen["callbacks"] == []
    # Non-callback fields survive.
    assert seen["configurable"] == {"user_id": "alice"}


@pytest.mark.unit
def test_invoke_preserves_positional_config_form() -> None:
    """Some callers pass ``config`` positionally rather than by keyword."""
    inner = _RecordingInner()
    r = _ContextAwareReflector(inner)
    r.invoke({"messages": []}, {"configurable": {"user_id": "bob"}})
    seen = inner.calls[0]["config"]
    assert seen["callbacks"] == []
    assert seen["configurable"] == {"user_id": "bob"}


@pytest.mark.unit
def test_invoke_replays_captured_context() -> None:
    """When ``attach`` supplied a captured context, ``ctx.run`` is used."""
    inner = _RecordingInner()
    r = _ContextAwareReflector(inner)
    payload = {"messages": []}

    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="none")

    def _prep_ctx() -> contextvars.Context:
        marker.set("captured")
        return contextvars.copy_context()

    ctx = _prep_ctx()

    def _record_marker(*_a: Any, **_kw: Any) -> Any:
        # Observed inside inner.invoke — it will run inside ctx.run.
        inner.calls.append({"marker": marker.get()})
        return {"ok": True}

    inner.invoke = _record_marker  # type: ignore[method-assign]
    r.attach(payload, ctx)
    r.invoke(payload)

    # The recorded marker inside inner should reflect the captured context.
    assert inner.calls[-1]["marker"] == "captured"


@pytest.mark.unit
def test_invoke_without_context_still_opts_out() -> None:
    """No attached context → no ``ctx.run`` — but callbacks still opted out."""
    inner = _RecordingInner()
    r = _ContextAwareReflector(inner)
    # No attach() called.
    r.invoke({"messages": []})
    seen = inner.calls[0]["config"]
    assert seen["callbacks"] == []
