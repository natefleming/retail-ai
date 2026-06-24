"""Unit tests for the ``dao_ai._tracing`` context-propagation helpers and the
``_ContextAwareReflector`` proxy used by ``LazyReflectionExecutor``.

The helpers exist because Python's ``contextvars.Context`` does not propagate
across thread boundaries by default. MLflow's active-span association is
stored in a ``ContextVar`` — without these helpers, work dispatched to a
``ThreadPoolExecutor`` or ``asyncio.to_thread`` runs in a fresh, empty
context, so any LangChain Runnable / MLflow-traced call inside the worker
opens a new root trace instead of nesting under the caller's span.

The tests below pin two contracts:

1. The helpers preserve a sentinel ``ContextVar`` across the worker boundary.
2. ``_ContextAwareReflector`` looks up the captured context by payload
   identity and replays it inside ``invoke()``.
"""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from dao_ai._tracing import (
    in_caller_context,
    submit_in_context,
    to_thread_in_context,
)
from dao_ai.memory.extraction import _ContextAwareReflector

# Module-level ContextVar so the helpers see a real var, not a closure.
_PROBE: contextvars.ContextVar[str] = contextvars.ContextVar("_PROBE", default="unset")


def _read_probe() -> str:
    return _PROBE.get()


# ---------------------------------------------------------------------------
# Helpers — sanity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSubmitInContext:
    def test_propagates_contextvar_into_worker_thread(self) -> None:
        _PROBE.set("from-main-thread")
        with ThreadPoolExecutor(max_workers=1) as ex:
            assert submit_in_context(ex, _read_probe).result() == "from-main-thread"

    def test_raw_submit_drops_contextvar(self) -> None:
        """Lock-in test for the underlying bug — confirms ContextVar does
        NOT propagate without our helper. If MLflow or Python ever changes
        this default, the test will fail loudly and we can retire the helper.
        """
        _PROBE.set("from-main-thread")
        with ThreadPoolExecutor(max_workers=1) as ex:
            assert ex.submit(_read_probe).result() == "unset"


@pytest.mark.unit
class TestToThreadInContext:
    def test_propagates_contextvar_into_thread(self) -> None:
        async def run() -> str:
            _PROBE.set("from-coroutine")
            return await to_thread_in_context(_read_probe)

        assert asyncio.run(run()) == "from-coroutine"

    def test_raw_to_thread_also_propagates_in_python_311_plus(self) -> None:
        """Python 3.11+ ``asyncio.to_thread`` already wraps with
        ``contextvars.copy_context().run`` internally, so contextvars DO
        propagate without our helper. ``to_thread_in_context`` is kept for
        call-site explicitness and as a defense against future asyncio
        behavior changes. If this assertion ever flips to expecting
        ``"unset"``, asyncio dropped the default context propagation —
        re-evaluate whether the helper has become load-bearing.
        """

        async def run() -> str:
            _PROBE.set("from-coroutine")
            return await asyncio.to_thread(_read_probe)

        assert asyncio.run(run()) == "from-coroutine"


@pytest.mark.unit
class TestInCallerContext:
    def test_captures_context_at_wrap_time(self) -> None:
        _PROBE.set("captured")
        wrapper = in_caller_context(_read_probe)
        _PROBE.set("mutated-after-capture")
        # Even though we mutated _PROBE after wrap, the wrapper restores the
        # captured value when called from a fresh thread.
        with ThreadPoolExecutor(max_workers=1) as ex:
            assert ex.submit(wrapper).result() == "captured"

    def test_passes_args_through(self) -> None:
        def echo(a: int, b: int) -> int:
            return a + b

        wrapped = in_caller_context(echo)
        assert wrapped(2, 3) == 5


# ---------------------------------------------------------------------------
# _ContextAwareReflector
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContextAwareReflector:
    def test_invoke_runs_inside_attached_context(self) -> None:
        inner = MagicMock()
        inner.namespace = ("memory", "{user_id}")
        inner.invoke.side_effect = lambda payload: _read_probe()

        proxy = _ContextAwareReflector(inner)

        captured_ctx = contextvars.copy_context()
        captured_ctx.run(_PROBE.set, "inside-captured-ctx")

        payload = {"messages": ["hi"]}
        proxy.attach(payload, captured_ctx)

        # Call from a thread where the ContextVar is unset.
        with ThreadPoolExecutor(max_workers=1) as ex:
            result = ex.submit(proxy.invoke, payload).result()
        assert result == "inside-captured-ctx"
        inner.invoke.assert_called_once_with(payload)

    def test_invoke_without_attached_context_passes_through(self) -> None:
        inner = MagicMock()
        inner.namespace = ("memory", "{user_id}")
        inner.invoke.return_value = "direct"
        proxy = _ContextAwareReflector(inner)

        # No attach() call — falls back to direct invocation.
        assert proxy.invoke({"messages": []}) == "direct"
        inner.invoke.assert_called_once()

    def test_attach_is_popped_so_no_leak(self) -> None:
        inner = MagicMock()
        inner.namespace = ("memory", "{user_id}")
        inner.invoke.return_value = "ok"
        proxy = _ContextAwareReflector(inner)

        payload = {"messages": ["a"]}
        ctx = contextvars.copy_context()
        proxy.attach(payload, ctx)

        # First call consumes the context.
        proxy.invoke(payload)
        # Internal map keyed by id(payload) — pop() removed it.
        assert id(payload) not in proxy._ctx_by_payload_id  # noqa: SLF001
        assert proxy._ctx_by_payload_id == {}  # noqa: SLF001

        # Second call with the same payload uses the no-context fast path.
        proxy.invoke(payload)
        assert inner.invoke.call_count == 2

    def test_passes_through_search_methods(self) -> None:
        inner = MagicMock()
        inner.namespace = ("memory", "{user_id}")
        inner.search.return_value = ["hit-1"]
        proxy = _ContextAwareReflector(inner)

        # __getattr__ delegates non-overridden attrs to the inner manager.
        assert proxy.search(query="anything") == ["hit-1"]
        inner.search.assert_called_once_with(query="anything")

    def test_namespace_exposed_for_langmem_init(self) -> None:
        inner = MagicMock()
        inner.namespace = ("memory", "{user_id}", "preferences")
        proxy = _ContextAwareReflector(inner)
        # langmem.LocalReflectionExecutor.__init__ reads `.namespace`.
        assert proxy.namespace == ("memory", "{user_id}", "preferences")
