"""Tests for the ``mlflow.langchain.autolog(run_tracer_inline=True)`` contract.

dao-ai wraps the async ``apredict``/``apredict_stream`` entry points with manual
``@mlflow.trace`` decorators *and* relies on ``mlflow.langchain.autolog`` to
capture LangChain spans. When autolog runs callbacks on a thread-pool executor
(the MLflow default), the manual span's ``ContextVar`` can fail to propagate
into the worker thread, producing orphaned autolog spans and the noisy
"ContextVar was created in a different Context" warning. Setting
``run_tracer_inline=True`` keeps callback dispatch on the main async task so
the autolog spans nest under the manual AGENT span the dao-ai client sees.

Two layers:

* ``TestAutologCallSite`` — pin every dao-ai entry-point module to
  ``run_tracer_inline=True`` so future refactors can't silently drop it.
* ``TestRunTracerInlineNesting`` — exercise the flag against a real
  ``langchain_core`` runnable and assert the autolog span actually descends
  from the manual outer span.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from unittest.mock import patch

import mlflow
import pytest

# ---------------------------------------------------------------------------
# Layer 1 — call-site assertion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAutologCallSite:
    """Regression guard: every dao-ai entry point must opt into inline tracing."""

    @pytest.mark.parametrize(
        "module_name",
        ["dao_ai.apps.handlers", "dao_ai.apps.model_serving"],
    )
    def test_autolog_called_with_run_tracer_inline_true(self, module_name: str) -> None:
        sys.modules.pop(module_name, None)
        with (
            patch("mlflow.langchain.autolog") as mock_autolog,
            patch("mlflow.set_registry_uri"),
            patch("mlflow.set_tracking_uri"),
            patch("mlflow.tracing.set_destination"),
        ):
            try:
                importlib.import_module(module_name)
            except Exception:
                # Entry-point modules execute heavy setup at import time
                # (AppConfig.from_file, etc.). We only need to observe the
                # autolog kwarg, which fires before that work runs.
                pass
        assert mock_autolog.called, (
            f"{module_name} did not call mlflow.langchain.autolog"
        )
        _, kwargs = mock_autolog.call_args
        assert kwargs.get("run_tracer_inline") is True, (
            f"{module_name} must call autolog(run_tracer_inline=True); "
            f"got kwargs={kwargs}"
        )


# ---------------------------------------------------------------------------
# Layer 2 — behavioral nesting
# ---------------------------------------------------------------------------


pytest.importorskip("langchain_core")


def _run_nested_async_trace() -> str:
    """Mimic dao-ai's pattern: an outer ``@mlflow.trace`` span wrapping an
    async ``langchain_core`` ``ainvoke``. Returns the trace_id captured via
    ``mlflow.get_active_trace_id()`` from inside the outer span, exactly the
    way ``dao_ai_apredict`` exposes it in ``custom_outputs``.
    """
    from langchain_core.runnables import RunnableLambda

    @mlflow.trace(span_type="AGENT", name="outer_agent")
    async def outer() -> str:
        chain = RunnableLambda(lambda x: x + "!")
        await chain.ainvoke("hi")
        return mlflow.get_active_trace_id()

    return asyncio.run(outer())


@pytest.fixture
def tracing_enabled(monkeypatch, tmp_path):
    """Re-enable MLflow tracing for this test only.

    ``tests/conftest.py`` globally sets ``MLFLOW_TRACE_SAMPLING_RATIO=0`` and
    calls ``mlflow.tracing.disable()`` to keep the test suite quiet. This
    fixture undoes both, points tracing at a per-test file URI, and tears
    down on exit.
    """
    monkeypatch.setenv("MLFLOW_TRACE_SAMPLING_RATIO", "1")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    # The test env points MLFLOW_EXPERIMENT_ID at a Databricks experiment;
    # delete it so the file:// backend creates a local default experiment.
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment("test-run-tracer-inline")
    mlflow.tracing.enable()
    mlflow.langchain.autolog(run_tracer_inline=True)
    try:
        yield
    finally:
        mlflow.langchain.autolog(disable=True)
        mlflow.tracing.disable()


@pytest.mark.unit
class TestRunTracerInlineNesting:
    """Prove that ``run_tracer_inline=True`` keeps autolog spans nested under
    a manual outer span across an async boundary.
    """

    def test_langchain_spans_nest_under_outer_agent_span(self, tracing_enabled) -> None:
        trace_id = _run_nested_async_trace()

        trace = mlflow.get_trace(trace_id)
        assert trace is not None, f"trace {trace_id} not found"

        spans = trace.data.spans
        outer = next((s for s in spans if s.name == "outer_agent"), None)
        assert outer is not None, (
            f"expected outer_agent span; got {[s.name for s in spans]}"
        )

        # The LangChain runnable produces a span that should be a descendant
        # of the outer_agent span (not a sibling at the root).
        runnable = next((s for s in spans if "RunnableLambda" in s.name), None)
        assert runnable is not None, (
            f"expected RunnableLambda span; got {[s.name for s in spans]}"
        )

        by_id = {s.span_id: s for s in spans}

        def root_of(span):
            while span.parent_id is not None:
                span = by_id[span.parent_id]
            return span

        assert root_of(runnable).span_id == outer.span_id, (
            f"RunnableLambda span did not nest under outer_agent; "
            f"its root was {root_of(runnable).name}"
        )
