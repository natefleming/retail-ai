"""Tests for the explicit MLflow span around ``MemoryContextMiddleware.asearch``.

dao-ai's ``MemoryContextMiddleware.abefore_model`` calls langmem's
``MemoryStoreManager.asearch``, which internally invokes a top-level LangChain
Runnable (``query_gen.ainvoke``). Without an active MLflow span on the calling
async task, ``mlflow.langchain.autolog`` opens a fresh root trace for that
inner Runnable — the "Use parallel tool calling to search for distinct
memories…" orphans observed in the experiment.

The fix wraps the ``asearch`` call in ``mlflow.start_span(span_type=
SpanType.MEMORY)``. With an active span on the contextvar, autolog correctly
nests langmem's internal spans under it.

Two layers:

* ``TestMemoryContextSpan`` — import-time mock of ``mlflow.start_span`` to
  pin the call site (name, type, set_inputs/outputs ordering).
* ``TestMemoryContextNesting`` — exercise the same wrapping pattern against
  a real ``langchain_core`` runnable and assert ``parent_id`` chain:
  Runnable → memory span → outer agent span.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import mlflow
import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain")


# ---------------------------------------------------------------------------
# Layer 1 — call-site assertion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMemoryContextSpan:
    """Regression guard: the asearch call must be wrapped in a MEMORY span."""

    def test_memory_search_opens_named_memory_span(self) -> None:
        from langchain_core.messages import HumanMessage

        from dao_ai.middleware.memory_context import MemoryContextMiddleware

        manager = MagicMock()
        manager.asearch = AsyncMock(return_value=[])
        mw = MemoryContextMiddleware(manager=manager, limit=5)

        state = {"messages": [HumanMessage(content="hi there")]}

        runtime = MagicMock()
        runtime.context = MagicMock()
        runtime.context.user_id = "nate.fleming@databricks.com"
        runtime.context.model_dump = MagicMock(return_value={"user_id": runtime.context.user_id})

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)

        with patch(
            "dao_ai.middleware.memory_context.mlflow.start_span",
            return_value=ctx,
        ) as mock_start:
            asyncio.run(mw.abefore_model(state, runtime))

        mock_start.assert_called_once()
        _, kwargs = mock_start.call_args
        assert kwargs["name"] == "memory_context_search"
        # span_type may be an enum or a string depending on MLflow version
        assert str(kwargs["span_type"]).endswith("MEMORY")

        ctx.set_inputs.assert_called_once()
        inputs_arg = ctx.set_inputs.call_args.args[0]
        assert inputs_arg["query"] == "hi there"
        assert inputs_arg["limit"] == 5

        ctx.set_outputs.assert_called_once()
        outputs_arg = ctx.set_outputs.call_args.args[0]
        assert outputs_arg["memories_count"] == 0

        manager.asearch.assert_awaited_once()

    def test_no_span_when_user_id_missing(self) -> None:
        """Early-return guard fires before we open a span (cheap path)."""
        from langchain_core.messages import HumanMessage

        from dao_ai.middleware.memory_context import MemoryContextMiddleware

        manager = MagicMock()
        manager.asearch = AsyncMock(return_value=[])
        mw = MemoryContextMiddleware(manager=manager, limit=5)

        state = {"messages": [HumanMessage(content="hi")]}
        runtime = MagicMock()
        runtime.context = MagicMock()
        runtime.context.user_id = None
        runtime.context.model_dump = MagicMock(return_value={"user_id": None})

        with patch(
            "dao_ai.middleware.memory_context.mlflow.start_span"
        ) as mock_start:
            result = asyncio.run(mw.abefore_model(state, runtime))

        assert result is None
        mock_start.assert_not_called()
        manager.asearch.assert_not_awaited()


# ---------------------------------------------------------------------------
# Layer 2 — behavioral nesting
# ---------------------------------------------------------------------------


@pytest.fixture
def tracing_enabled(monkeypatch, tmp_path):
    """Re-enable MLflow tracing for this test only.

    Same shape as ``tests/dao_ai/test_autolog_config.py::tracing_enabled``:
    overrides the suite-wide ``MLFLOW_TRACE_SAMPLING_RATIO=0`` and the
    Databricks-bound ``MLFLOW_EXPERIMENT_ID`` so we can drive a fresh local
    file:// backend.
    """
    monkeypatch.setenv("MLFLOW_TRACE_SAMPLING_RATIO", "1")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    mlflow.set_tracking_uri(f"file://{tmp_path}")
    mlflow.set_experiment("test-memory-context-span")
    mlflow.tracing.enable()
    mlflow.langchain.autolog(run_tracer_inline=True)
    try:
        yield
    finally:
        mlflow.langchain.autolog(disable=True)
        mlflow.tracing.disable()


@pytest.mark.unit
class TestMemoryContextNesting:
    """Prove that a Runnable invoked inside ``mlflow.start_span(MEMORY)``
    produces autolog spans that nest under the memory span, not as siblings.

    This mirrors what happens in production: langmem's ``query_gen.ainvoke``
    is a top-level Runnable invocation, and we want it to attach to the
    parent memory_context_search span.
    """

    def test_inner_runnable_nests_under_memory_span(
        self, tracing_enabled
    ) -> None:
        from langchain_core.runnables import RunnableLambda
        from mlflow.entities import SpanType

        async def run() -> str:
            @mlflow.trace(span_type=SpanType.AGENT, name="outer_agent")
            async def outer() -> str:
                with mlflow.start_span(
                    name="memory_context_search",
                    span_type=SpanType.MEMORY,
                ) as span:
                    span.set_inputs({"query": "hi", "limit": 5})
                    chain = RunnableLambda(lambda x: x + "!")
                    await chain.ainvoke("hi")
                    span.set_outputs({"memories_count": 0})
                return mlflow.get_active_trace_id()

            return await outer()

        trace_id = asyncio.run(run())
        trace = mlflow.get_trace(trace_id)
        assert trace is not None, f"trace {trace_id} not found"

        spans = list(trace.data.spans)
        names = {s.name for s in spans}
        assert "outer_agent" in names, f"outer_agent missing; got {names}"
        assert "memory_context_search" in names, (
            f"memory_context_search missing; got {names}"
        )
        assert any("RunnableLambda" in n for n in names), (
            f"RunnableLambda span missing; got {names}"
        )

        outer = next(s for s in spans if s.name == "outer_agent")
        memory_span = next(s for s in spans if s.name == "memory_context_search")
        runnable = next(s for s in spans if "RunnableLambda" in s.name)

        # The load-bearing assertion: autolog'd Runnable is a child of the
        # manual memory span, which is itself a child of the outer agent.
        assert memory_span.parent_id == outer.span_id, (
            f"memory_context_search not parented to outer_agent; "
            f"parent_id={memory_span.parent_id} expected={outer.span_id}"
        )
        assert runnable.parent_id == memory_span.span_id, (
            f"RunnableLambda not parented to memory_context_search; "
            f"parent_id={runnable.parent_id} expected={memory_span.span_id}"
        )
