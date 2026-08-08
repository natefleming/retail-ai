"""Tests for the sync-over-async runner (``dao_ai._async``).

Covers GitHub issue #272: the sync ``predict`` wrapper must drive the async
graph to completion whether or not the calling thread already has a running
event loop (notebooks/IPython, Model Serving workers, plain sync code) —
without patching the caller's loop (so it stays uvloop-safe for Databricks
Apps) and without requiring ``nest_asyncio``.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlflow
import pytest

from dao_ai._async import _Runner, iter_sync, run_sync, shutdown_runner


@pytest.fixture(autouse=True)
def _fresh_runner():
    """Ensure each test starts and ends with no live runner singleton."""
    shutdown_runner()
    yield
    shutdown_runner()


# --------------------------------------------------------------------------- #
# run_sync
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_run_sync_no_running_loop():
    """run_sync works from a plain sync caller (Model Serving worker analog)."""

    async def add(a: int, b: int) -> int:
        await asyncio.sleep(0.01)
        return a + b

    assert run_sync(add(2, 3)) == 5


@pytest.mark.unit
def test_run_sync_from_within_running_loop():
    """#272 regression: run_sync must not raise when a loop is already running.

    Bare ``asyncio.run()`` raises ``RuntimeError: asyncio.run() cannot be
    called from a running event loop`` here; the runner marshals onto its own
    loop thread instead.
    """

    async def inner() -> int:
        await asyncio.sleep(0.01)
        return 42

    async def outer() -> int:
        # Emulates calling the sync predict() from inside a notebook's loop.
        return run_sync(inner())

    assert asyncio.run(outer()) == 42


@pytest.mark.unit
def test_run_sync_propagates_exception():
    """Exceptions raised by the coroutine surface to the sync caller."""

    async def boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        run_sync(boom())


# --------------------------------------------------------------------------- #
# iter_sync
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_iter_sync_yields_all_items():
    async def gen(n: int):
        for i in range(n):
            await asyncio.sleep(0.001)
            yield i

    assert list(iter_sync(lambda: gen(4))) == [0, 1, 2, 3]


@pytest.mark.unit
def test_iter_sync_propagates_exception_mid_stream():
    async def gen():
        yield 0
        yield 1
        raise RuntimeError("mid-stream")

    it = iter_sync(gen)
    assert next(it) == 0
    assert next(it) == 1
    with pytest.raises(RuntimeError, match="mid-stream"):
        next(it)


@pytest.mark.unit
def test_iter_sync_closes_generator_on_early_break():
    closed = threading.Event()

    async def gen():
        try:
            i = 0
            while True:
                yield i
                i += 1
        finally:
            closed.set()

    for item in iter_sync(gen):
        if item == 1:
            break  # early break must still run the generator's finally

    assert closed.wait(timeout=5.0), "async generator was not closed on break"


# --------------------------------------------------------------------------- #
# context propagation
# --------------------------------------------------------------------------- #


_CV: contextvars.ContextVar[str] = contextvars.ContextVar("_CV", default="unset")


@pytest.mark.unit
def test_contextvar_propagates_into_runner():
    """Caller contextvars are visible inside the coroutine on the loop thread."""

    async def read_cv() -> str:
        return _CV.get()

    _CV.set("hello")
    assert run_sync(read_cv()) == "hello"


# --------------------------------------------------------------------------- #
# MLflow trace nesting (the load-bearing propagation guarantee)
# --------------------------------------------------------------------------- #


@pytest.fixture
def tracing_enabled(monkeypatch, tmp_path):
    """Re-enable MLflow tracing for this test only (mirrors trace-nesting tests)."""
    monkeypatch.setenv("MLFLOW_TRACE_SAMPLING_RATIO", "1")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment("test-async-runner-trace-nesting")
    mlflow.tracing.enable()
    mlflow.langchain.autolog(run_tracer_inline=True)
    try:
        yield
    finally:
        mlflow.langchain.autolog(disable=True)
        mlflow.tracing.disable()


@pytest.mark.unit
def test_run_sync_nests_traced_work_under_caller_span(tracing_enabled):
    """A RunnableLambda driven via run_sync nests under the caller's span."""
    pytest.importorskip("langchain_core")
    from langchain_core.runnables import RunnableLambda
    from mlflow.entities import SpanType

    @mlflow.trace(span_type=SpanType.AGENT, name="outer_agent")
    def outer() -> str:
        chain = RunnableLambda(lambda x: x + "!")

        async def _invoke():
            return await chain.ainvoke("hi")

        run_sync(_invoke())
        return mlflow.get_active_trace_id()

    trace_id = outer()
    trace = mlflow.get_trace(trace_id)
    spans = list(trace.data.spans)

    outer_span = next(s for s in spans if s.name == "outer_agent")
    runnable = next(s for s in spans if "RunnableLambda" in s.name)
    assert runnable.parent_id == outer_span.span_id, (
        "run_sync'd RunnableLambda span should descend from outer_agent; "
        f"parent_id={runnable.parent_id} expected={outer_span.span_id}"
    )


@pytest.mark.unit
def test_concurrent_calls_get_isolated_context():
    """Each concurrent caller's coroutine sees its OWN contextvar snapshot.

    Two callers set different values and await different durations so their
    coroutine bodies interleave on the shared loop; the value each coroutine
    reads must match the value its own caller set (no cross-talk).
    """

    async def read_after(delay: float) -> str:
        await asyncio.sleep(delay)
        return _CV.get()

    def worker(val: str, delay: float) -> str:
        _CV.set(val)
        return run_sync(read_after(delay))

    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(worker, "A", 0.15)
        fb = ex.submit(worker, "B", 0.02)
        assert fa.result() == "A"
        assert fb.result() == "B"


# --------------------------------------------------------------------------- #
# concurrency, laziness, nested runs
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_concurrent_run_sync_is_io_concurrent():
    """20 concurrent callers each awaiting 0.1s finish in ~0.1s, not ~2s.

    Confirms the single shared loop provides true IO concurrency rather than
    serializing callers.
    """

    async def slow(x: int) -> int:
        await asyncio.sleep(0.1)
        return x

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=20) as ex:
        results = sorted(ex.map(lambda i: run_sync(slow(i)), range(20)))
    elapsed = time.monotonic() - start

    assert results == list(range(20))
    assert elapsed < 1.0, f"calls serialized on the loop: {elapsed:.2f}s"


@pytest.mark.unit
def test_runner_is_lazy_and_import_safe():
    """No runner thread/loop exists until the first run_sync/iter_sync call."""
    assert _Runner._instance is None
    assert not any(t.name == "dao_ai-sync-runner" for t in threading.enumerate()), (
        "runner thread created before first use"
    )

    async def noop() -> int:
        return 1

    run_sync(noop())
    assert _Runner._instance is not None


@pytest.mark.unit
def test_nested_asyncio_run_inside_coroutine():
    """A coroutine that itself calls asyncio.run works on the runner loop.

    Mirrors the self-guarded nested-run sites (e.g. Postgres store setup) that
    apply nest_asyncio when they detect a running loop, then call asyncio.run.
    """

    async def setup() -> str:
        return "ready"

    async def outer() -> str:
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.run(setup())

    assert run_sync(outer()) == "ready"
