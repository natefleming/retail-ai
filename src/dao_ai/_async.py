"""Sync-over-async runner backed by a process-singleton event loop.

dao-ai's agent graph is async (``apredict`` / ``apredict_stream``). MLflow
``pyfunc`` Model Serving, notebooks, and any plain sync caller invoke the
synchronous ``predict`` / ``predict_stream`` wrappers, which must drive that
async work to completion. The naive bridge ``asyncio.run(coro)`` throws
``RuntimeError: asyncio.run() cannot be called from a running event loop`` when
a loop is already running on the calling thread — which is exactly the case in
IPython/Databricks notebooks (see GitHub issue #272).

This module bridges sync→async **without** patching the caller's event loop
(so it never touches ``nest_asyncio`` and is safe under uvloop, which Databricks
Apps' uvicorn uses). It runs every coroutine on one dedicated daemon-thread
event loop that persists for the process lifetime, and blocks the calling
thread on the result. This mirrors the proven pattern in
``dao_ai.background.agent._BackgroundLoop`` and the eval-notebook
``run_coroutine_threadsafe`` dispatch.

Why a single shared loop rather than a fresh loop per call:

* Agent requests are overwhelmingly await-bound IO (LLM HTTP, tool IO,
  Postgres/Lakebase). One cooperative loop services many concurrent callers
  with true IO concurrency; each caller blocks only its own thread.
* Async DB pools are keyed by ``id(asyncio.get_running_loop())`` (see
  ``dao_ai.memory.postgres``/``databricks``). A stable loop reuses pools for
  the process lifetime; a per-call loop would re-open pools (and re-mint
  Lakebase OAuth) on every request and risks ``RuntimeError: got Future
  attached to a different loop``.

The loop/thread is created lazily on first use, so importing this module is
free and never spins up a thread or touches asyncio — important because Apps
imports the package but never calls the sync bridge.
"""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import threading
from concurrent.futures import Future
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Coroutine,
    Generator,
    Optional,
    TypeVar,
)

R = TypeVar("R")
T = TypeVar("T")


class _Runner:
    """Process-singleton daemon-thread event loop for sync-over-async bridging."""

    _instance: Optional["_Runner"] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def instance(cls) -> "_Runner":
        # Double-checked locking: the fast path avoids the lock once the
        # singleton exists, keeping per-call overhead to a plain attribute read.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        # A default (selector) loop — NOT uvloop — so it stays patchable by the
        # self-guarded nested ``asyncio.run`` sites (e.g. Postgres store setup)
        # that still apply nest_asyncio when they detect a running loop.
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="dao_ai-sync-runner",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _assert_not_on_loop_thread(self) -> None:
        # Blocking on a future whose coroutine runs on this same loop thread
        # would deadlock. The sync bridge is only ever entered from a caller
        # thread (Model Serving worker, notebook, plain sync code); async
        # callers must await ``apredict``/``apredict_stream`` directly.
        if threading.current_thread() is self._thread:
            raise RuntimeError(
                "dao_ai._async.run_sync/iter_sync cannot be called from the "
                "runner loop thread; await the async API (apredict/"
                "apredict_stream) directly instead."
            )

    def submit(self, coro: Coroutine[Any, Any, R]) -> "Future[R]":
        """Schedule *coro* on the runner loop, propagating caller contextvars.

        ``asyncio.Task`` copies the *current* context at ``create_task()`` time.
        We snapshot the caller's context and create the task inside it so
        MLflow's active-span ContextVar (and any request-scoped state)
        propagates across the thread hop, and each concurrent caller's task
        gets its OWN context snapshot (verified in the runner tests). This
        mirrors ``dao_ai.background.agent._BackgroundLoop.submit``. (Note
        ``call_soon_threadsafe`` also copies the caller context by default, so
        the explicit ``copy_context`` here is belt-and-suspenders — it keeps
        the create-task-in-caller-context invariant explicit and robust to the
        scheduling shape.)
        """
        caller_ctx = contextvars.copy_context()
        fut: "Future[R]" = Future()
        task_ready = threading.Event()

        def _schedule() -> None:
            def _create() -> None:
                task = self._loop.create_task(coro)
                task.add_done_callback(
                    lambda t: (
                        fut.set_exception(t.exception())
                        if t.exception() is not None
                        else fut.set_result(t.result())
                    )
                )
                task_ready.set()

            caller_ctx.run(_create)

        self._loop.call_soon_threadsafe(_schedule)
        task_ready.wait(timeout=5.0)
        return fut

    def shutdown(self) -> None:
        if self._loop.is_closed():
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)


def run_sync(coro: Coroutine[Any, Any, R]) -> R:
    """Run *coro* to completion from synchronous code and return its result.

    Works whether or not the calling thread already has a running event loop
    (notebooks, Model Serving workers, plain sync code). Exceptions raised by
    the coroutine propagate to the caller.
    """
    runner = _Runner.instance()
    runner._assert_not_on_loop_thread()
    return runner.submit(coro).result()


def iter_sync(
    agen_factory: Callable[[], AsyncGenerator[T, None]],
) -> Generator[T, None, None]:
    """Drive an async generator from sync code, yielding items on the caller thread.

    Takes a **factory** (zero-arg callable returning the async generator) rather
    than a pre-built generator so the generator object is constructed on the
    runner loop thread inside the caller's context — constructing it on the
    caller thread would bind it to the wrong loop.
    """
    runner = _Runner.instance()
    runner._assert_not_on_loop_thread()

    # Build the async generator on the loop thread (inside caller context).
    agen: AsyncGenerator[T, None] = runner.submit(_make_agen(agen_factory)).result()

    try:
        while True:
            try:
                yield runner.submit(_anext(agen)).result()
            except StopAsyncIteration:
                break
    finally:
        # Best-effort teardown so the underlying generator's ``finally`` blocks
        # (pool release, span close) run even on early break / caller exception.
        try:
            runner.submit(_aclose(agen)).result(timeout=5.0)
        except Exception:
            pass


async def _make_agen(
    agen_factory: Callable[[], AsyncGenerator[T, None]],
) -> AsyncGenerator[T, None]:
    return agen_factory()


async def _anext(agen: AsyncGenerator[T, None]) -> T:
    return await agen.__anext__()


async def _aclose(agen: AsyncGenerator[T, None]) -> None:
    await agen.aclose()


def shutdown_runner() -> None:
    """Stop the runner loop and join its thread (idempotent).

    Registered via ``atexit`` for clean process exit; also useful in tests.
    """
    with _Runner._lock:
        runner = _Runner._instance
        _Runner._instance = None
    if runner is not None:
        runner.shutdown()


atexit.register(shutdown_runner)
