"""Behavioral test for MLflow trace nesting across worker boundaries.

Two cases against a real ``mlflow.langchain.autolog(run_tracer_inline=True)``
+ local ``file://`` tracking backend + a real ``langchain_core.RunnableLambda``:

* **Case A** — ``submit_in_context`` nests bg-thread autolog spans under
  the caller's manual ``@mlflow.trace`` span. The Runnable span's
  ``parent_id`` must match the outer span's ``span_id``.
* **Case B** — raw ``ex.submit(...)`` produces an orphan: the Runnable
  span's ``parent_id is None``. This is a lock-in regression test for our
  understanding of the underlying contextvars semantics. If MLflow ever
  changes its default to propagate across thread pools, this test will
  fail and we can retire the helper.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import mlflow
import pytest

pytest.importorskip("langchain_core")

from dao_ai._tracing import submit_in_context  # noqa: E402


@pytest.fixture
def tracing_enabled(monkeypatch, tmp_path):
    """Re-enable MLflow tracing for this test only.

    Same shape as ``tests/dao_ai/test_autolog_config.py::tracing_enabled``.
    """
    monkeypatch.setenv("MLFLOW_TRACE_SAMPLING_RATIO", "1")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment("test-background-trace-nesting")
    mlflow.tracing.enable()
    mlflow.langchain.autolog(run_tracer_inline=True)
    try:
        yield
    finally:
        mlflow.langchain.autolog(disable=True)
        mlflow.tracing.disable()


def _runnable_call_in_thread(submitter) -> str:
    """Helper: run a RunnableLambda.invoke in a thread via ``submitter``."""
    from langchain_core.runnables import RunnableLambda
    from mlflow.entities import SpanType

    @mlflow.trace(span_type=SpanType.AGENT, name="outer_agent")
    def outer() -> str:
        chain = RunnableLambda(lambda x: x + "!")
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = submitter(ex, chain.invoke, "hi")
            fut.result()
        return mlflow.get_active_trace_id()

    return outer()


@pytest.mark.unit
class TestBackgroundTraceNesting:
    def test_submit_in_context_nests_runnable_under_outer(
        self, tracing_enabled
    ) -> None:
        trace_id = _runnable_call_in_thread(submit_in_context)
        trace = mlflow.get_trace(trace_id)
        spans = list(trace.data.spans)

        outer = next(s for s in spans if s.name == "outer_agent")
        runnable = next(s for s in spans if "RunnableLambda" in s.name)
        assert runnable.parent_id == outer.span_id, (
            "Background-thread RunnableLambda span should descend from "
            f"outer_agent. parent_id={runnable.parent_id} expected={outer.span_id}"
        )

    def test_raw_submit_produces_orphan(self, tracing_enabled) -> None:
        """Reproduction test — locks in the underlying behavior so we know
        when this fix is no longer needed. Should fail if MLflow ever
        starts propagating contextvars across thread pools by default.
        """

        def raw_submitter(ex, fn, *args, **kwargs):
            return ex.submit(fn, *args, **kwargs)

        trace_id = _runnable_call_in_thread(raw_submitter)
        trace = mlflow.get_trace(trace_id)
        spans = list(trace.data.spans)

        # Outer span exists.
        outer = next(s for s in spans if s.name == "outer_agent")
        # The RunnableLambda span, if it landed in this trace at all,
        # would NOT have outer as parent under the raw submit. Allow two
        # acceptable shapes:
        #   (a) RunnableLambda not present at all in this trace (it became
        #       a sibling root in a different trace)
        #   (b) RunnableLambda present but with parent_id == None
        runnable_spans = [s for s in spans if "RunnableLambda" in s.name]
        if runnable_spans:
            assert runnable_spans[0].parent_id != outer.span_id, (
                "raw ex.submit() should NOT propagate the outer span — if "
                "this assertion fails the helper may no longer be necessary."
            )
