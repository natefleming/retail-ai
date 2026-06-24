"""Context-propagating wrappers for background workers.

MLflow's active-span association is stored in Python ``ContextVars``.
``ContextVars`` are inherited across ``asyncio.create_task(...)`` boundaries
automatically, but **not** across thread boundaries (per MLflow's own docs
and Python's ``contextvars`` spec).

When dao-ai dispatches LangChain Runnable or MLflow-traced work to a thread
pool or to ``asyncio.to_thread`` without restoring the caller's
``Context``, MLflow's autolog hook sees no active parent span and opens a
**new root trace** — the "orphan" trace pattern observed in the experiment
when langmem's ``LocalReflectionExecutor`` runs background memory
extraction on Databricks Apps.

The helpers here bridge that gap with a single canonical pattern:
capture the caller's ``Context`` at dispatch time, then run the work via
``ctx.run(...)`` inside the worker.
"""

from __future__ import annotations

import asyncio
import contextvars
from concurrent.futures import Executor, Future
from typing import Any, Callable, TypeVar

R = TypeVar("R")


def in_caller_context(fn: Callable[..., R]) -> Callable[..., R]:
    """Wrap ``fn`` so it executes in a snapshot of the caller's contextvars.

    The snapshot is taken at the moment ``in_caller_context(fn)`` returns,
    so call this **outside** the worker and pass the returned callable into
    the worker boundary (e.g. ``threading.Thread(target=...)``).

    Useful when neither ``submit_in_context`` nor ``to_thread_in_context``
    fits the dispatch shape (for example, when constructing a coroutine
    inside a background event loop running in a separate thread).
    """
    ctx = contextvars.copy_context()

    def wrapper(*args: Any, **kwargs: Any) -> R:
        return ctx.run(fn, *args, **kwargs)

    return wrapper


def submit_in_context(
    executor: Executor, fn: Callable[..., R], /, *args: Any, **kwargs: Any
) -> "Future[R]":
    """``executor.submit(fn, *args, **kwargs)`` with caller-context propagation.

    Drop-in replacement for ``executor.submit(...)`` when the executor is a
    plain ``concurrent.futures.Executor`` (e.g. ``ThreadPoolExecutor``).
    """
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)  # type: ignore[arg-type]


async def to_thread_in_context(fn: Callable[..., R], /, *args: Any, **kwargs: Any) -> R:
    """``asyncio.to_thread(fn, *args, **kwargs)`` with caller-context propagation.

    Note: Python 3.11+ ``asyncio.to_thread`` already wraps its target with
    ``contextvars.copy_context().run`` internally, so this helper is
    primarily for call-site explicitness — using ``to_thread_in_context``
    documents *at the call site* that the worker must inherit the caller's
    trace context. It also guards against future asyncio changes; see
    ``tests/dao_ai/test_context_propagation.py::test_raw_to_thread_also_propagates_in_python_311_plus``.
    """
    ctx = contextvars.copy_context()
    return await asyncio.to_thread(ctx.run, fn, *args, **kwargs)  # type: ignore[arg-type]
