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
    """Clear the inherited tracing context for the duration of the block.

    When ``@mlflow.trace`` or ``mlflow.start_span`` opens a new span,
    the OTel + MLflow SDKs auto-link it to whatever span is currently
    active. In environments where the surrounding context contains a
    runtime-scoped span that is *not* exported to dao-ai's configured
    trace destination, the resulting "root" span has a
    ``parent_span_id`` that points to a phantom span — orphan-root
    spans which are rejected by ``trace_unified``'s root-detection
    filter and by ``InferenceTableSpanExporter``.

    MLflow's tracing module (``mlflow.tracing.provider``) maintains its
    own ``ContextVarsRuntimeContext`` separate from the global OTel
    runtime context. ``start_span_in_context`` reads from MLflow's
    context first, so detaching only the global OTel context is not
    enough — both must be cleared.

    Note: MLflow's pyfunc ``ResponsesAgent.__init_subclass__``
    auto-applies ``@mlflow.trace(span_type=AGENT)`` to the subclass's
    ``predict``/``predict_stream``, so even with this helper active
    the framework will still emit a ``predict`` root in the
    ``mlflow.genai.evaluate`` path. This helper is most useful at
    boundaries you fully own (background workers, custom tools).
    """
    from mlflow.tracing.provider import mlflow_runtime_context

    otel_token = otel_ctx.attach(otel_ctx.Context())
    mlflow_token = mlflow_runtime_context.attach(otel_ctx.Context())
    try:
        yield
    finally:
        mlflow_runtime_context.detach(mlflow_token)
        otel_ctx.detach(otel_token)


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
