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
import contextlib
import contextvars
import functools
import inspect
from concurrent.futures import Executor, Future
from typing import Any, AsyncIterator, Callable, Iterator, TypeVar

import mlflow
from opentelemetry import context as otel_ctx

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


@contextlib.contextmanager
def detached_otel_context() -> Iterator[None]:
    """Clear the inherited OpenTelemetry context for the duration of the block.

    Databricks runtimes (Apps Agent Server, Model Serving, notebook
    execution) maintain a runtime-scoped OTel span that is *not* exported
    to dao-ai's configured trace destination. When ``@mlflow.trace`` or
    ``mlflow.start_span`` opens a new span, OTel's SDK auto-links it to
    whatever span is currently active. The result is a dao-ai "root"
    span whose ``parent_span_id`` points to a phantom span that never
    appears in the exported OTEL table — orphan-root spans.

    The ``trace_unified`` view filters roots via
    ``WHERE COALESCE(parent_span_id, '') = ''``, and the
    ``InferenceTableSpanExporter`` only exports spans whose
    ``parent is None``. Both mechanisms reject orphan-root spans, so
    traces never reach the experiment UI in either trace-location mode.

    Use at dao-ai request entry boundaries (typically via
    :func:`root_trace`) so the dao-ai-owned span starts in a fresh OTel
    context and is exported as a true root.
    """
    token = otel_ctx.attach(otel_ctx.Context())
    try:
        yield
    finally:
        otel_ctx.detach(token)


def root_trace(*trace_args: Any, **trace_kwargs: Any) -> Callable[..., Any]:
    """Decorator combining :func:`detached_otel_context` with ``@mlflow.trace``.

    Apply at dao-ai request entry points (``apredict``, ``apredict_stream``,
    notebook ``predict_fn``, etc.) where the dao-ai span must be the true
    root of an exported trace.

    Accepts the same arguments as :func:`mlflow.trace`. Supports sync,
    coroutine, sync-generator, and async-generator functions.
    """

    def decorate(fn: Callable[..., R]) -> Callable[..., R]:
        traced = mlflow.trace(*trace_args, **trace_kwargs)(fn)

        if inspect.isasyncgenfunction(fn):

            @functools.wraps(fn)
            async def agen_wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
                with detached_otel_context():
                    async for item in traced(*args, **kwargs):
                        yield item

            return agen_wrapper  # type: ignore[return-value]

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with detached_otel_context():
                    return await traced(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        if inspect.isgeneratorfunction(fn):

            @functools.wraps(fn)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
                with detached_otel_context():
                    yield from traced(*args, **kwargs)

            return gen_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with detached_otel_context():
                return traced(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorate
